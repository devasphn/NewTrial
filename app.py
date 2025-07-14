import torch
import transformers
import faster_whisper
from TTS.api import TTS
import gradio as gr
import numpy as np
import time
import os

# --- Configuration ---
# Models
STT_MODEL = "distil-large-v3"
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

        # 1. STT (Speech-to-Text) Model
        print(f"Loading STT model: {STT_MODEL}...")
        self.stt_model = faster_whisper.WhisperModel(
            STT_MODEL, 
            device=self.device, 
            compute_type="float16" if self.device == "cuda" else "int8"
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

    def transcribe_audio(self, audio_filepath: str) -> str:
        """Transcribes audio to text."""
        if not audio_filepath:
            return ""
        print("Transcribing audio...")
        segments, _ = self.stt_model.transcribe(audio_filepath, beam_size=5)
        transcription = " ".join([segment.text for segment in segments])
        print(f"User: {transcription}")
        return transcription

    def generate_response(self, chat_history: list) -> str:
        """Generates a response from the LLM based on chat history."""
        # This part handles the old tuple format from the chatbot
        formatted_messages = [
            {"role": "system", "content": "You are a friendly and helpful conversational AI. Your name is Deva. Keep your responses concise and to the point."}
        ]
        for user_msg, ai_msg in chat_history:
            if user_msg:
                formatted_messages.append({"role": "user", "content": user_msg})
            if ai_msg:
                formatted_messages.append({"role": "assistant", "content": ai_msg})
        
        terminators = [
            self.llm_pipeline.tokenizer.eos_token_id,
            self.llm_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.llm_pipeline(
            formatted_messages,
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

    def convert_text_to_speech(self, text: str) -> str:
        """Converts text to speech and saves it to a file."""
        print("Speaking...")
        self.tts_model.tts_to_file(
            text=text,
            speaker="Claribel Dervla",
            language="en", 
            file_path=OUTPUT_WAV_FILE
        )
        return OUTPUT_WAV_FILE

    def process_conversation_turn(self, audio_filepath: str, chat_history: list):
        """Processes a single conversational turn."""
        chat_history = chat_history or []
        if audio_filepath is None:
            return chat_history, None

        user_text = self.transcribe_audio(audio_filepath)
        if not user_text.strip():
            return chat_history, None
        
        # We add the user message to the history
        chat_history.append((user_text, None))
        
        # We need to generate the response based on the history up to this point
        llm_response = self.generate_response(chat_history)

        # Now we update the last entry in the history with the assistant's response
        chat_history[-1] = (user_text, llm_response)
        
        agent_audio_path = self.convert_text_to_speech(llm_response)

        return chat_history, agent_audio_path

def build_ui(agent: RealTimeS2SAgent):
    """Builds the Gradio web interface for the agent."""
    with gr.Blocks(theme=gr.themes.Soft(), title="Real-Time S2S Agent") as demo:
        gr.Markdown("# Real-Time Speech-to-Speech AI Agent")
        gr.Markdown("Tap the microphone, speak, and the agent will respond. The agent's audio will play automatically.")

        # FINAL FIX: Removed the 'type="messages"' argument, which is not supported in gradio==4.31.0.
        # This will revert to the default tuple format and fix the crash.
        chatbot = gr.Chatbot(label="Conversation", elem_id="chatbot", height=500)
        
        with gr.Row():
            mic_input = gr.Audio(sources=["microphone"], type="filepath", label="Tap to Talk")
            audio_output = gr.Audio(label="Agent Response", autoplay=True, visible=True)

        # The handle_interaction function needs to be a bit different for the tuple format
        mic_input.stop_recording(
            fn=agent.process_conversation_turn,
            inputs=[mic_input, chatbot],
            outputs=[chatbot, audio_output]
        )
        
        clear_button = gr.Button("Clear Conversation")
        clear_button.click(lambda: ([], None), None, [chatbot, audio_output])

    return demo

if __name__ == "__main__":
    s2s_agent = RealTimeS2SAgent()
    ui = build_ui(s2s_agent)
    
    # Setting share=True is recommended for cloud environments like Runpod
    ui.launch(server_name="0.0.0.0", server_port=7860, share=True)
