import torch
import transformers
import faster_whisper
from TTS.api import TTS
import pyaudio
import numpy as np
import webrtcvad
import collections
import time
import os

class RealTimeS2SAgent:
    """
    A real-time Speech-to-Speech agent that listens, transcribes,
    generates a response, and speaks it back.
    """
    def __init__(self):
        """
        Initializes all the models and necessary components for the S2S pipeline.
        """
        print("--- Initializing S2S Agent ---")
        
        # Check for GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        print(f"Using device: {self.device.upper()}")

        # --- 1. Initialize STT (Speech-to-Text) Model ---
        # Using faster-whisper with a distilled model for speed and accuracy.
        # This model is loaded onto the GPU using CTranslate2 with float16 precision.
        stt_model_name = "distil-large-v3"
        print(f"Loading STT model: {stt_model_name}...")
        self.stt_model = faster_whisper.WhisperModel(
            stt_model_name, 
            device=self.device, 
            compute_type="float16" if self.device == "cuda" else "int8"
        )
        print("STT model loaded.")

        # --- 2. Initialize LLM (Language Model) ---
        # Using Llama-3 8B Instruct, loaded in 4-bit for efficiency.
        # Requires a Hugging Face token with access to the model.
        llm_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        print(f"Loading LLM: {llm_model_name}...")
        self.llm_pipeline = transformers.pipeline(
            "text-generation",
            model=llm_model_name,
            model_kwargs={"torch_dtype": self.torch_dtype},
            device_map=self.device,
        )
        print("LLM loaded.")

        # --- 3. Initialize TTS (Text-to-Speech) Model ---
        # Using XTTS-v2 for high-quality, streaming voice synthesis.
        tts_model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        print(f"Loading TTS model: {tts_model_name}...")
        # Check if the XTTS model path exists, otherwise it will be downloaded
        xtts_path = os.path.join(os.path.expanduser("~"), ".local/share/tts/", tts_model_name.replace("/", "--"))
        if not os.path.exists(xtts_path):
            print("XTTS model not found locally, it will be downloaded.")
        self.tts_model = TTS(tts_model_name).to(self.device)
        print("TTS model loaded.")

        # --- 4. Initialize Audio Input Stream ---
        # Using PyAudio to capture microphone input.
        # We also use WebRTC VAD for Voice Activity Detection.
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = 16000
        self.chunk_duration_ms = 30  # VAD supports 10, 20, or 30 ms chunks
        self.chunk_size = int(self.sample_rate * self.chunk_duration_ms / 1000)
        self.vad = webrtcvad.Vad(3)  # Aggressiveness level 3 (most aggressive)
        self.padding_duration_ms = 500 # 0.5 seconds of silence to detect end of speech
        self.num_padding_chunks = int(self.padding_duration_ms / self.chunk_duration_ms)
        self.audio_interface = pyaudio.PyAudio()

    def listen_and_transcribe(self) -> str:
        """
        Listens for audio from the microphone, detects speech, and transcribes it.
        
        Returns:
            The transcribed text as a string.
        """
        print("\n--- Listening... ---")
        stream = self.audio_interface.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )

        triggered = False
        ring_buffer = collections.deque(maxlen=self.num_padding_chunks)
        voiced_frames = []

        while True:
            chunk = stream.read(self.chunk_size)
            is_speech = self.vad.is_speech(chunk, self.sample_rate)

            if not triggered:
                ring_buffer.append((chunk, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > 0.8 * ring_buffer.maxlen:
                    triggered = True
                    print("Voice detected, recording...")
                    voiced_frames.extend([f for f, s in ring_buffer])
                    ring_buffer.clear()
            else:
                voiced_frames.append(chunk)
                ring_buffer.append((chunk, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > 0.9 * ring_buffer.maxlen:
                    print("End of speech detected.")
                    break
        
        stream.stop_stream()
        stream.close()

        audio_data = b''.join(voiced_frames)
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        if len(audio_np) == 0:
            return ""

        print("Transcribing...")
        segments, _ = self.stt_model.transcribe(audio_np, beam_size=5)
        transcription = " ".join([segment.text for segment in segments])
        print(f"User: {transcription}")
        return transcription

    def generate_response_stream(self, text: str):
        """
        Generates a response from the LLM as a stream of text.

        Args:
            text: The input text from the user.

        Yields:
            The generated text token by token.
        """
        messages = [
            {"role": "system", "content": "You are a friendly and helpful conversational AI. Keep your responses concise and to the point."},
            {"role": "user", "content": text},
        ]
        
        # The pipeline's __call__ method for text-generation returns a generator when streaming
        terminators = [
            self.llm_pipeline.tokenizer.eos_token_id,
            self.llm_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.llm_pipeline(
            messages,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.llm_pipeline.tokenizer.eos_token_id, # Suppress warning
        )
        
        # The output is a list containing a dictionary.
        # The generated text is in outputs[0]['generated_text'], which is itself a list of messages.
        # We only want the assistant's response.
        assistant_response = outputs[0]["generated_text"][-1]['content']
        print(f"Agent: {assistant_response}")
        return assistant_response


    def stream_speech(self, text_stream: str):
        """
        Converts a stream of text into a stream of audio and plays it.

        Args:
            text_stream: A string containing the full response.
        """
        print("Speaking...")
        # XTTS-v2's streaming mode is highly effective.
        # It synthesizes audio chunk by chunk and plays it immediately.
        chunks = self.tts_model.tts(text=text_stream, speaker=self.tts_model.speakers[0], language=self.tts_model.languages[0], stream=True)
        
        # The stream method needs to be implemented to play audio chunks
        # PyAudio can be used for this, but for simplicity in this example,
        # we'll use the built-in play function of the TTS library which handles it.
        self.tts_model.tts_to_file(text=text_stream, speaker=self.tts_model.speakers[0], language=self.tts_model.languages[0], file_path="output.wav")
        
        # For a truly streaming playback, you would do something like this:
        # p = pyaudio.PyAudio()
        # stream = p.open(format=pyaudio.paFloat32, channels=1, rate=24000, output=True)
        # for chunk in chunks:
        #     stream.write(chunk.cpu().numpy().tobytes())
        # stream.stop_stream()
        # stream.close()
        # p.terminate()

        # For simplicity, we use a helper to play the generated file
        os.system("aplay output.wav") # 'aplay' for Linux, 'afplay' for macOS, or use a python library
        

    def run(self):
        """
        The main loop for the S2S agent.
        """
        print("\n--- Agent is Ready ---")
        try:
            while True:
                user_input = self.listen_and_transcribe()
                if user_input:
                    response_text = self.generate_response_stream(user_input)
                    if response_text:
                        self.stream_speech(response_text)
                else:
                    print("No speech detected or transcription failed.")
        except KeyboardInterrupt:
            print("\n--- Agent Shutting Down ---")
        finally:
            self.audio_interface.terminate()
            if os.path.exists("output.wav"):
                os.remove("output.wav")

if __name__ == "__main__":
    agent = RealTimeS2SAgent()
    agent.run()
