# Local-Ambient-AI
Model for making your own ambient AI


## Basic Steps:
1. Set up local LLM (Ollama is easiest)
2. Serve the LLM
3. Manage a small server to accept audio, Speech-to-text conversion (via Whisper), then send text with prompt to Ollama
4. Return Text

# Setting up Ollama
- [ollama download](https://ollama.com/)
- when downloaded, run the application, then use terminal/command line: "ollama run <model>" to download the model - I use Qwen 2.5, you can nominate an LLM size by colon (eg Qwen2.5:3b)
- after testing that it works - type in a prompt after downloading and check for response, quit using "/bye"
- then use "ollama serve" to make available on a port - default is 11434

# Setting up Flask (basic python server to manage whisper and audio)
