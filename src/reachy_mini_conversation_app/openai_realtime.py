import json
import asyncio
import logging
import time
import threading
import os
import tempfile
from typing import Optional, Final, Tuple

import numpy as np
import ollama

import edge_tts 

import speech_recognition as sr
from pydub import AudioSegment
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item
from numpy.typing import NDArray

from reachy_mini_conversation_app.prompts import get_session_instructions
from reachy_mini_conversation_app.tools.core_tools import (
    ToolDependencies,
    get_tool_specs,
    dispatch_tool_call,
)

ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe" 
AudioSegment.converter = ffmpeg_path
AudioSegment.ffprobe   = ffmpeg_path.replace("ffmpeg.exe", "ffprobe.exe")

logger = logging.getLogger(__name__)

INPUT_SAMPLE_RATE: Final[int] = 24000
OUTPUT_SAMPLE_RATE: Final[int] = 24000
OLLAMA_MODEL_NAME = "goekdenizguelmez/JOSIEFIED-Qwen3:8b"

EDGE_TTS_VOICE = "en-US-AriaNeural"

HARDWARE_MIC_ID = 2

class OllamaHandler(AsyncStreamHandler):
    def __init__(self, deps: ToolDependencies, gradio_mode: bool = False, instance_path: Optional[str] = None):
        super().__init__(
            expected_layout="mono",
            output_sample_rate=OUTPUT_SAMPLE_RATE,
            input_sample_rate=INPUT_SAMPLE_RATE,
        )
        self.deps = deps
        self.output_queue = asyncio.Queue()
        
        # Connection
        self.client = ollama.AsyncClient(host='http://127.0.0.1:11434')
        
        self.messages = []
        self.system_instruction = get_session_instructions()
        self._initialize_history()
        self._shutdown_requested = False

    def _initialize_history(self):
        self.messages = [{"role": "system", "content": self.system_instruction}]

    async def start_up(self) -> None:
        logger.info(f"Connecting to Ollama model: {OLLAMA_MODEL_NAME}...")
        self.main_loop = asyncio.get_running_loop()
        self.listener_thread = threading.Thread(target=self._run_local_listener, daemon=True)
        self.listener_thread.start()
        logger.info("✅ Ollama connected. Local Listener Thread started.")

    def _run_local_listener(self):
        r = sr.Recognizer()
        r.energy_threshold = 300 
        r.dynamic_energy_threshold = True
        
        mic = sr.Microphone(device_index=HARDWARE_MIC_ID)
        logger.info(f"🎤 LISTENING LOCALLY on Device Index: {HARDWARE_MIC_ID}")
        
        with mic as source:
            r.adjust_for_ambient_noise(source, duration=1.0)
            
            while not self._shutdown_requested:
                try:
                    audio = r.listen(source, timeout=None, phrase_time_limit=10)
                    try:
                        text = r.recognize_google(audio)
                        if text:
                            logger.info(f"User said: {text}")
                            asyncio.run_coroutine_threadsafe(
                                self.process_text_input(text), 
                                self.main_loop
                            )
                    except sr.UnknownValueError:
                        pass
                    except sr.RequestError:
                        logger.error("STT Unavailable")
                except Exception as e:
                    logger.error(f"Listener Error: {e}")
                    time.sleep(1)

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        pass 

    async def process_text_input(self, user_text: str):
        try:
            await self.output_queue.put(AdditionalOutputs({"role": "user", "content": user_text}))
            self.messages.append({"role": "user", "content": user_text})
            await self.run_llm_turn()
        except Exception as e:
            logger.error(f"Processing Error: {e}")

    async def run_llm_turn(self):
        tools = get_tool_specs()
        response_text = ""
        current_tool_calls = []

        try:
            stream = await self.client.chat(
                model=OLLAMA_MODEL_NAME,
                messages=self.messages,
                tools=tools, # type: ignore
                stream=True
            )

            async for chunk in stream:
                msg = chunk.get('message', {})
                content = msg.get('content', '')
                if content: response_text += content
                if msg.get('tool_calls'): current_tool_calls.extend(msg['tool_calls'])

            if current_tool_calls:
                self.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "tool_calls": current_tool_calls
                })
                for tool in current_tool_calls:
                    fn_name = tool['function']['name']
                    fn_args = tool['function']['arguments']
                    
                    await self.output_queue.put(AdditionalOutputs({
                        "role": "assistant", "content": f"🛠️ {fn_name}..."
                    }))
                    
                    fn_args_str = json.dumps(fn_args) if isinstance(fn_args, dict) else str(fn_args)
                    result = await dispatch_tool_call(fn_name, fn_args_str, self.deps)
                    self.messages.append({"role": "tool", "content": json.dumps(result)})

                await self.run_llm_turn()
                return

            if response_text:
                self.messages.append({"role": "assistant", "content": response_text})
                await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": response_text}))
                
                audio_chunk = await self.local_text_to_speech(response_text)
                if audio_chunk is not None:
                     await self.output_queue.put((OUTPUT_SAMPLE_RATE, audio_chunk))

        except Exception as e:
            logger.error(f"LLM Error: {e}")

    async def local_text_to_speech(self, text: str) -> Optional[NDArray[np.int16]]:
        """
        Uses Microsoft Edge TTS.
        No models to download. High quality.
        """
        try:

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                temp_filename = f.name
            
            communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE)
            
            await communicate.save(temp_filename)

            sound = AudioSegment.from_mp3(temp_filename)
            
            sound = sound.set_frame_rate(OUTPUT_SAMPLE_RATE).set_channels(1)
            
            channel_sounds = sound.split_to_mono()
            samples = [s.get_array_of_samples() for s in channel_sounds]
            audio_arr = np.array(samples[0], dtype=np.int16)

            try:
                os.remove(temp_filename)
            except:
                pass

            return audio_arr

        except Exception as e:
            logger.error(f"EdgeTTS Error: {e}")
            return None

    async def emit(self): return await wait_for_item(self.output_queue)
    def copy(self): return OllamaHandler(self.deps, self.gradio_mode, self.instance_path)
    async def shutdown(self): self._shutdown_requested = True
    async def apply_personality(self, profile): self.system_instruction = get_session_instructions(); self._initialize_history()
    async def get_available_voices(self): return ["default"]