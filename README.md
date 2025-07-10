Real-Time Speech-to-Speech AI Agent
This project provides a complete, self-contained AI agent that interacts in real-time through voice. It listens for spoken input, processes it, and generates a spoken response.

The architecture is designed for high performance on a GPU-enabled machine (like Runpod with an NVIDIA A4500) and uses a state-of-the-art model pipeline for a fluid conversational experience.

Core Technology Pipeline
Speech-to-Text (STT): faster-whisper (distil-large-v3 model) for fast and accurate transcription.

Language Model (LLM): Meta's Llama-3-8B-Instruct (in 4-bit) for intelligent and quick response generation.

Text-to-Speech (TTS): XTTS-v2 for high-quality, streaming voice synthesis.

Prerequisites
Hardware: A machine with a modern NVIDIA GPU with at least 16GB of VRAM (e.g., A4500, A6000, RTX 3090, RTX 4090).

Operating System: Linux (recommended for Runpod).

Software:

Python 3.10+

git and git-lfs (for downloading the LLM).

portaudio library for microphone access. On Debian/Ubuntu-based systems (like the standard Runpod environment), install it with:

sudo apt-get update
sudo apt-get install portaudio19-dev

Setup and Installation
Follow these steps precisely to ensure a smooth setup.

1. Clone the Repository

First, create a new repository in your GitHub account. Then, clone it to your local machine or Runpod instance.

# Replace with your repository URL
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

2. Create a Virtual Environment

It is crucial to use a virtual environment to isolate dependencies and prevent conflicts.

python3 -m venv venv
source venv/bin/activate

Your terminal prompt should now be prefixed with (venv).

3. Install Dependencies

Install all the required Python libraries using the provided requirements.txt file. This file has pinned versions for maximum stability.

pip install -r requirements.txt

4. Log in to Hugging Face (Required for Llama 3)

The Llama 3 model is gated. You need to have a Hugging Face account and accept the license terms on the model card page.

Then, generate an access token with 'read' permissions from your Hugging Face account settings and log in via the terminal:

huggingface-cli login
# Paste your token when prompted

Running the Agent
Once the setup is complete, you can run the agent with a single command:

python main.py

The first time you run the script, it will download all the necessary models. This may take a significant amount of time and disk space (approx. 20-25 GB). Subsequent runs will be much faster as the models will be cached.

The agent will first initialize all models, and you will see progress bars in your terminal.

Once you see the "--- Agent is Ready ---" message, you can start speaking.

The agent detects the end of your speech (a moment of silence) and then begins processing.

To stop the agent, press Ctrl+C in the terminal.
