import torch
import transformers
import faster_whisper
from TTS.api import TTS
import gradio as gr
import numpy as np
import time
import os
import librosa

# --- Configuration ---
# Models
STT_MODEL = "large-v3"  # Upgraded from distil-large-v3 for better accuracy
LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
# Output audio file
OUTPUT_WAV_FILE = "output.wav"

class RealTimeS2SAgent:
    """
    A real-time Speech-to-Speech agent that uses Gradio for its UI.
    It transcribes user audio, generates a response with an LLM, and speaks it back.
    """
    def __init__(self):
        """
        Initializes all the models and necessary components.
        """
        print("--- Initializing S2S Agent ---")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        print(f"Using device: {self.device.upper()}")

        # 1. STT (Speech-to-Text) Model - Enhanced initialization
        print(f"Loading STT model: {STT_MODEL}...")
        self.stt_model = faster_whisper.WhisperModel(
            STT_MODEL, 
            device=self.device, 
            compute_type="float16" if self.device == "cuda" else "int8",
            cpu_threads=4,  # Optimize for RTX A4500
            num_workers=1   # Prevent threading issues
        )
        print("STT model loaded.")

        # 2. LLM (Language Model)
        print(f"Loading LLM: {LLM_MODEL}...")
        self.llm_pipeline = transformers.pipeline(
            "text-generation",
            model=LLM_MODEL,
            model_kwargs={"torch_dtype": self.torch_dtype},
            device_map=self.device,
        )
        print("LLM loaded.")

        # 3. TTS (Text-to-Speech) Model
        print(f"Loading TTS model: {TTS_MODEL}...")
        self.tts_model = TTS(TTS_MODEL).to(self.device)
        print("TTS model loaded.")
        
        if os.path.exists(OUTPUT_WAV_FILE):
            os.remove(OUTPUT_WAV_FILE)
            
        print("\n--- Agent is Ready ---")

    def _is_hallucination(self, text: str) -> bool:
        """Check if the transcribed text is likely a hallucination."""
        text_lower = text.lower().strip()
        
        # Common hallucinations
        hallucinations = [
            "thank you", "thanks", "thank you.", "thanks.",
            "thank you very much", "thanks for watching",
            "subscribe", "like and subscribe", ".",
            "um", "uh", "ah", "oh", "mm", "hmm",
            "you", "the", "a", "an", "and", "or"
        ]
        
        # Check for exact matches or very short repetitive content
        if text_lower in hallucinations:
            return True
        
        # Check for repetitive patterns
        words = text_lower.split()
        if len(words) > 1 and len(set(words)) == 1:  # All words are the same
            return True
        
        # Check for very short transcriptions that are likely noise
        if len(text.strip()) < 3:
            return True
            
        # Check for single character or punctuation only
        if len(text.strip()) == 1:
            return True
        
        return False

    def transcribe_audio(self, audio_filepath: str) -> str:
        """Transcribes audio to text with improved error handling and preprocessing."""
        if not audio_filepath or not os.path.exists(audio_filepath):
            return ""
        
        print("Transcribing audio...")
        
        try:
            # Check audio file size first
            file_size = os.path.getsize(audio_filepath)
            if file_size < 1000:  # Less than 1KB
                print("Audio file too small, skipping transcription")
                return ""
            
            # Load and analyze audio
            audio_data, sample_rate = librosa.load(audio_filepath, sr=16000)
            
            # Check if audio is too short or mostly silence
            if len(audio_data) < 1600:  # Less than 0.1 seconds at 16kHz
                print("Audio too short, skipping transcription")
                return ""
            
            # Check for silence (RMS energy threshold)
            rms_energy = np.sqrt(np.mean(audio_data**2))
            if rms_energy < 0.001:  # Adjust threshold as needed
                print("Audio appears to be silence, skipping transcription")
                return ""
            
            # Transcribe with improved parameters
            segments, info = self.stt_model.transcribe(
                audio_filepath, 
                beam_size=5,
                temperature=0,  # Deterministic results
                language="en",  # Force English
                condition_on_previous_text=False,  # Don't condition on previous text
                vad_filter=True,  # Enable voice activity detection
                vad_parameters=dict(min_silence_duration_ms=1000),
                word_timestamps=True,
                initial_prompt="",  # Clear any initial prompt
            )
            
            # Collect segments and validate
            transcription_segments = []
            for segment in segments:
                # Filter out common hallucinations
                text = segment.text.strip()
                if text and not self._is_hallucination(text):
                    transcription_segments.append(text)
            
            transcription = " ".join(transcription_segments)
            
            # Final validation
            if not transcription.strip() or self._is_hallucination(transcription):
                print("Transcription appears to be hallucinated or empty, returning empty")
                return ""
            
            print(f"User: {transcription}")
            return transcription
            
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""

    def generate_response(self, chat_history: list) -> str:
        """Generates a response from the LLM based on chat history."""
        messages = [
            {"role": "system", "content": "You are a friendly and helpful conversational AI. Your name is Deva. Keep your responses concise and to the point."}
        ]
        for msg in chat_history:
            if msg['role'] in ['user', 'assistant']:
                messages.append(msg)
        
        terminators = [
            self.llm_pipeline.tokenizer.eos_token_id,
            self.llm_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        try:
            outputs = self.llm_pipeline(
                messages,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.llm_pipeline.tokenizer.eos_token_id,
            )
            
            assistant_response = outputs[0]["generated_text"][-1]['content']
            print(f"Agent: {assistant_response}")
            return assistant_response
            
        except Exception as e:
            print(f"LLM generation error: {e}")
            return "I apologize, but I encountered an error generating a response. Please try again."

    def convert_text_to_speech(self, text: str) -> str:
        """Converts text to speech and saves it to a file."""
        if not text.strip():
            return None
            
        print("Speaking...")
        try:
            self.tts_model.tts_to_file(
                text=text,
                speaker="Claribel Dervla",
                language="en", 
                file_path=OUTPUT_WAV_FILE
            )
            return OUTPUT_WAV_FILE
        except Exception as e:
            print(f"TTS error: {e}")
            return None

    def process_conversation_turn(self, audio_filepath: str, chat_history: list):
        """Processes a single conversational turn."""
        if audio_filepath is None:
            return chat_history, None

        # Additional audio validation
        if not os.path.exists(audio_filepath):
            print("Audio file doesn't exist")
            return chat_history, None
            
        file_size = os.path.getsize(audio_filepath)
        if file_size < 1000:  # Less than 1KB
            print("Audio file too small, skipping")
            return chat_history, None

        user_text = self.transcribe_audio(audio_filepath)
        if not user_text.strip():
            print("No valid transcription, skipping turn")
            return chat_history, None
            
        chat_history.append({"role": "user", "content": user_text})

        llm_response = self.generate_response(chat_history)
        if llm_response:
            chat_history.append({"role": "assistant", "content": llm_response})
            
            agent_audio_path = self.convert_text_to_speech(llm_response)
            return chat_history, agent_audio_path
        else:
            return chat_history, None

def build_ui(agent: RealTimeS2SAgent):
    """Builds the Gradio web interface for the agent."""
    with gr.Blocks(theme=gr.themes.Soft(), title="Real-Time S2S Agent") as demo:
        gr.Markdown("# Real-Time Speech-to-Speech AI Agent")
        gr.Markdown("Tap the microphone, speak clearly, and the agent will respond. The agent's audio will play automatically.")

        chatbot = gr.Chatbot(label="Conversation", elem_id="chatbot", height=500, type="messages")
        
        with gr.Row():
            mic_input = gr.Audio(sources=["microphone"], type="filepath", label="Tap to Talk")
            # The audio player must be visible for browser autoplay policies to work reliably.
            audio_output = gr.Audio(label="Agent Response", autoplay=True, visible=True)

        def handle_interaction(audio_filepath, history):
            history = history or []
            
            # Add audio validation
            if not audio_filepath or not os.path.exists(audio_filepath):
                return history, None
            
            # Check file size
            file_size = os.path.getsize(audio_filepath)
            if file_size < 1000:  # Less than 1KB
                print("Audio file too small, skipping")
                return history, None
            
            updated_history, agent_audio_path = agent.process_conversation_turn(audio_filepath, history)
            return updated_history, agent_audio_path

        mic_input.stop_recording(
            fn=handle_interaction,
            inputs=[mic_input, chatbot],
            outputs=[chatbot, audio_output]
        )
        
        clear_button = gr.Button("Clear Conversation")
        clear_button.click(lambda: ([], None), None, [chatbot, audio_output])

    return demo

if __name__ == "__main__":
    try:
        s2s_agent = RealTimeS2SAgent()
        ui = build_ui(s2s_agent)
        
        # Remember to change the port if your Runpod exposes a different one (e.g., 8888)
        ui.launch(server_name="0.0.0.0", server_port=7860)
    except Exception as e:
        print(f"Failed to start application: {e}")
