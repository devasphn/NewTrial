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
        
        # Clean up any previous output file
        if os.path.exists(OUTPUT_WAV_FILE):
            os.remove(OUTPUT_WAV_FILE)
            
        print("\n--- Agent is Ready ---")

    def transcribe_audio(self, audio_filepath: str) -> str:
        """
        Transcribes the given audio file to text.
        
        Args:
            audio_filepath: Path to the audio file.
            
        Returns:
            The transcribed text.
        """
        if not audio_filepath:
            return ""
            
        print("Transcribing audio...")
        segments, _ = self.stt_model.transcribe(audio_filepath, beam_size=5)
        transcription = " ".join([segment.text for segment in segments])
        print(f"User: {transcription}")
        return transcription

    def generate_response(self, text: str) -> str:
        """
        Generates a response from the LLM.

        Args:
            text: The input text from the user.

        Returns:
            The generated text response.
        """
        messages = [
            {"role": "system", "content": "You are a friendly and helpful conversational AI. Your name is Deva. Keep your responses concise and to the point."},
            {"role": "user", "content": text},
        ]
        
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
            pad_token_id=self.llm_pipeline.tokenizer.eos_token_id,
        )
        
        assistant_response = outputs[0]["generated_text"][-1]['content']
        print(f"Agent: {assistant_response}")
        return assistant_response

    def convert_text_to_speech(self, text: str) -> str:
        """
        Converts text to speech and saves it to a file.
        
        Args:
            text: The text to be converted to speech.
            
        Returns:
            The path to the saved audio file.
        """
        print("Speaking...")
        self.tts_model.tts_to_file(
            text=text, 
            speaker=self.tts_model.speakers[0], 
            language=self.tts_model.languages[0], 
            file_path=OUTPUT_WAV_FILE
        )
        return OUTPUT_WAV_FILE

    def process_conversation(self, audio_filepath: str, chat_history: list):
        """
        The main processing pipeline for a single conversational turn.
        
        Args:
            audio_filepath: The path to the user's recorded audio.
            chat_history: The current state of the conversation.
            
        Returns:
            A tuple containing the updated chat history and the path to the agent's audio response.
        """
        if audio_filepath is None:
            return chat_history, None

        # 1. Transcribe User Audio
        user_text = self.transcribe_audio(audio_filepath)
        if not user_text.strip():
            # If transcription is empty, do nothing
            return chat_history, None
            
        chat_history.append(("You", user_text))

        # 2. Generate LLM Response
        llm_response = self.generate_response(user_text)
        chat_history.append(("Agent", llm_response))
        
        # 3. Convert Response to Speech
        agent_audio_path = self.convert_text_to_speech(llm_response)

        return chat_history, agent_audio_path


# --- Gradio Web UI ---
def build_ui(agent: RealTimeS2SAgent):
    """
    Builds the Gradio web interface for the agent.
    """
    with gr.Blocks(theme=gr.themes.Soft(), title="Real-Time S2S Agent") as demo:
        gr.Markdown("# Real-Time Speech-to-Speech AI Agent")
        gr.Markdown("Tap the microphone, speak, and the agent will respond. The agent's audio will play automatically.")

        with gr.Row():
            # Chatbot component to display the conversation
            chatbot = gr.Chatbot(label="Conversation", elem_id="chatbot", height=500)
        
        with gr.Row():
            # Audio input component (microphone)
            mic_input = gr.Audio(sources=["microphone"], type="filepath", label="Tap to Talk")
            
            # Audio output component for the agent's response
            audio_output = gr.Audio(label="Agent Response", autoplay=True, visible=False)

        # Function to handle the interaction when the user provides audio
        def handle_interaction(audio_filepath, history):
            # This is a generator function to allow for streaming-like updates
            # First, update the chat with the user's (yet to be transcribed) message placeholder
            history.append(("You", "*(...speaking...)*"))
            yield history, None # Update chatbot, no audio yet
            
            # Process the conversation
            updated_history, agent_audio_path = agent.process_conversation(audio_filepath, history[:-1]) # Pass history without the placeholder
            
            # Final update with all results
            yield updated_history, agent_audio_path

        # Connect the microphone input to the handler function
        mic_input.stop_recording(
            fn=handle_interaction,
            inputs=[mic_input, chatbot],
            outputs=[chatbot, audio_output]
        )
        
        # Add a button to clear the conversation
        clear_button = gr.Button("Clear Conversation")
        clear_button.click(lambda: ([], None), None, [chatbot, audio_output])

    return demo

if __name__ == "__main__":
    # Instantiate the agent
    s2s_agent = RealTimeS2SAgent()
    
    # Build and launch the Gradio UI
    ui = build_ui(s2s_agent)
    
    # Launch on 0.0.0.0 to make it accessible on the network (e.g., Runpod)
    ui.launch(server_name="0.0.0.0", server_port=8888)
