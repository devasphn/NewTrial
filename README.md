# Real-Time Speech-to-Speech AI Agent (Web UI Version)

This project provides a complete, self-contained AI agent that interacts in real-time through voice via a web interface. It listens for spoken input from your browser's microphone, processes it, and generates a spoken response that plays back automatically.

The architecture is designed for high performance on a GPU-enabled machine (like Runpod with an NVIDIA A4500) and uses a state-of-the-art model pipeline for a fluid conversational experience.

## Core Technology Pipeline
*   **Web UI:** Gradio for a fast, interactive web interface.
*   **Speech-to-Text (STT):** `faster-whisper` (distil-large-v3 model) for fast and accurate transcription.
*   **Language Model (LLM):** Meta's `Llama-3-8B-Instruct` (in 4-bit) for intelligent and quick response generation.
*   **Text-to-Speech (TTS):** `XTTS-v2` for high-quality, streaming voice synthesis.

## Prerequisites
*   **Hardware:** A machine with a modern NVIDIA GPU with at least 16GB of VRAM (e.g., A4500, A6000, RTX 3090, RTX 4090).
*   **Operating System:** Linux (recommended for Runpod).
*   **Software:**
    *   Python 3.10+
    *   `git` and `git-lfs` (for downloading the LLM).

## Setup and Installation

1.  **Clone the Repository**
    ```bash
    # Replace with your repository URL
    git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
    cd YOUR_REPO_NAME
    ```

2.  **Create a Virtual Environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    Install all required Python libraries. This step is critical.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Log in to Hugging Face (Required for Llama 3)**
    The Llama 3 model is gated. You need a Hugging Face account and must accept the license on the model card page. Then, generate an access token with 'read' permissions and log in via the terminal:
    ```bash
    huggingface-cli login
    # Paste your token when prompted
    ```

## Running the Agent

1.  **Start the Web Server**
    Once setup is complete, run the application with:
    ```bash
    python app.py
    ```
    The first time you run it, it will download all necessary models (approx. 20-25 GB). This can take a while. Subsequent runs will be much faster.

2.  **Access the UI**
    The script will start a web server. On Runpod, a button will appear to "Connect to Port 7860". Click it, or copy the public URL from your pod's logs into your browser. It will look something like `https://<your-pod-id>.proxy.runpod.net`.

3.  **Interact with the Agent**
    *   The UI will display a chat window and a microphone button.
    *   Click the "Tap to Talk" microphone button and speak. It will automatically detect when you finish speaking.
    *   Your transcribed text will appear in the chat, followed by the agent's response.
    *   The agent's spoken audio will play automatically in your browser.
    *   Click "Clear Conversation" to start over.

4.  **Stopping the Agent**
    To stop the server, press `Ctrl+C` in the terminal.
