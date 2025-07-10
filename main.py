import torch
import transformers
import faster_whisper
import gradio as gr
import numpy as np
import time
import os
import soundfile as sf
import librosa

# --- Configuration ---
STT_MODEL = "distil-large-v3"
LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
# NEW: Using StyleTTS 2 for emotionally expressive speech
TTS_MODEL = "mca-ark/StyleTTS2-LibriTTS"
SPEAKER_REFERENCE_WAV = "speaker.wav"  # Reference audio to define the agent's voice
OUTPUT_WAV_FILE = "output.wav"

class EmotionalS2SAgent:
    """
    An emotionally intelligent Speech-to-Speech agent using StyleTTS 2.
    """
    def __init__(self):
        """
        Initializes all models, including the new StyleTTS 2.
        """
        print("--- Initializing Emotionally Intelligent S2S Agent ---")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        print(f"Using device: {self.device.upper()}")

        # 1. STT (Speech-to-Text) Model
        print(f"Loading STT model: {STT_MODEL}...")
        self.stt_model = faster_whisper.WhisperModel(
            STT_MODEL, device=self.device, compute_type="float16"
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

        # 3. NEW TTS (StyleTTS 2)
        print(f"Loading TTS model: {TTS_MODEL}...")
        self.tts_model = transformers.AutoModel.from_pretrained(TTS_MODEL).to(self.device)
        self.tts_processor = transformers.AutoProcessor.from_pretrained(TTS_MODEL)
        
        # Pre-compute the speaker embedding from the reference audio
        if not os.path.exists(SPEAKER_REFERENCE_WAV):
            raise FileNotFoundError(f"Speaker reference file not found at {SPEAKER_REFERENCE_WAV}. "
                                    "Please download it as per the README instructions.")
        
        print(f"Computing speaker embedding from {SPEAKER_REFERENCE_WAV}...")
        waveform, sample_rate = librosa.load(SPEAKER_REFERENCE_WAV, sr=self.tts_processor.sampling_rate)
        self.speaker_embedding = self.tts_processor(
            audio=waveform, return_tensors="pt", sampling_rate=sample_rate
        )["speaker_embedding"].to(self.device)
        print("TTS model and speaker embedding loaded.")

        if os.path.exists(OUTPUT_WAV_FILE):
            os.remove(OUTPUT_WAV_FILE)
            
        print("\n--- Agent is Ready ---")

    def transcribe_audio(self, audio_filepath: str) -> str:
        if not audio_filepath: return ""
        print("Transcribing audio...")
        segments, _ = self.stt_model.transcribe(audio_filepath, beam_size=5)
        transcription = " ".join([segment.text for segment in segments])
        print(f"User: {transcription}")
        return transcription

    def generate_response(self, chat_history: list) -> str:
        # NEW: Updated system prompt to encourage emotional expression
        messages = [
            {"role": "system", "content": "You are Deva, a conversational AI with a friendly, empathetic, and slightly expressive personality. Use natural language, and don't be afraid to use cues like *chuckles* or *sighs* to convey feeling. Keep your responses concise."}
        ]
        for msg in chat_history:
            if msg['role'] in ['user', 'assistant']:
                messages.append(msg)
        
        terminators = [
            self.llm_pipeline.tokenizer.eos_token_id,
            self.llm_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        outputs = self.llm_pipeline(
            messages, max_new_tokens=256, eos_token_id=terminators,
            do_sample=True, temperature=0.8, top_p=0.9, pad_token_id=self.llm_pipeline.tokenizer.eos_token_id,
        )
        assistant_response = outputs[0]["generated_text"][-1]['content']
        print(f"Agent: {assistant_response}")
        return assistant_response

    def convert_text_to_speech(self, text: str) -> str:
        """
        Converts text to speech using StyleTTS 2.
        """
        print("Speaking...")
        inputs = self.tts_processor(text, return_tensors="pt", speaker_embedding=self.speaker_embedding).to(self.device)
        
        with torch.no_grad():
            output = self.tts_model.generate(**inputs, max_new_tokens=1024)
        
        waveform = output.cpu().numpy().squeeze()
        sf.write(OUTPUT_WAV_FILE, waveform, self.tts_processor.sampling_rate)
        return OUTPUT_WAV_FILE

    def process_conversation_turn(self, audio_filepath: str, chat_history: list):
        if audio_filepath is None: return chat_history, None
        user_text = self.transcribe_audio(audio_filepath)
        if not user_text.strip(): return chat_history, None
        chat_history.append({"role": "user", "content": user_text})
        llm_response = self.generate_response(chat_history)
        chat_history.append({"role": "assistant", "content": llm_response})
        agent_audio_path = self.convert_text_to_speech(llm_response)
        return chat_history, agent_audio_path

def build_ui(agent: EmotionalS2SAgent):
    """Builds the Gradio web interface for the agent."""
    with gr.Blocks(theme=gr.themes.Soft(), title="Emotionally Intelligent S2S Agent") as demo:
        gr.Markdown("# Emotionally Intelligent Speech-to-Speech AI Agent")
        gr.Markdown("Tap the microphone, speak, and the agent will respond with an expressive voice.")
        chatbot = gr.Chatbot(label="Conversation", elem_id="chatbot", height=500, type="messages")
        with gr.Row():
            mic_input = gr.Audio(sources=["microphone"], type="filepath", label="Tap to Talk")
            audio_output = gr.Audio(label="Agent Response", autoplay=True, visible=True)
        def handle_interaction(audio_filepath, history):
            history = history or []
            return agent.process_conversation_turn(audio_filepath, history)
        mic_input.stop_recording(
            fn=handle_interaction, inputs=[mic_input, chatbot], outputs=[chatbot, audio_output]
        )
        clear_button = gr.Button("Clear Conversation")
        clear_button.click(lambda: ([], None), None, [chatbot, audio_output])
    return demo

if __name__ == "__main__":
    agent = EmotionalS2SAgent()
    ui = build_ui(agent)
    ui.launch(server_name="0.0.0.0", server_port=7860)
