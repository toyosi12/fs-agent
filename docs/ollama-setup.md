# Setting up Ollama for fs-agent

This guide explains how to configure fs-agent to use local Ollama models instead of cloud-based LLMs.

## Prerequisites

1. **Install Ollama** on your Mac:
   ```bash
   # Using Homebrew (recommended)
   brew install ollama
   
   # Or download from https://ollama.com/download/mac
   ```

2. **Start Ollama service** in one terminal:
   ```bash
   ollama serve
   ```
   This starts the Ollama server on `http://localhost:11434`

3. **Pull a model** (recommended for fs-agent):
   ```bash
   # Llama 3.1 8B - good balance of capability and speed
   ollama pull llama3.1:8b
   
   # Or other options:
   ollama pull qwen2.5:7b       # Fast, good for code
   ollama pull codellama:7b     # Specialized for code
   ollama pull mistral:7b       # Lightweight and fast
   ```

## Configuration

### Option 1: Environment Variables (Recommended)

Create a `.env` file in the fs-agent project root:

```env
FS_AGENT_LLM_PROVIDER=ollama
FS_AGENT_LLM_MODEL=llama3.1:8b
# FS_AGENT_LLM_BASE_URL=http://localhost:11434/v1  # Optional, this is the default
# FS_AGENT_OPENAI_API_KEY=ollama  # Optional, not needed for Ollama
```

### Option 2: Command Line Flags

```bash
fs-agent run "Build a task manager" \
  --llm-provider ollama \
  --llm-model llama3.1:8b
```

### Option 3: Per-Agent Configuration

You can use different models for different agents:

```env
# Use larger model for architect (planning)
FS_AGENT_LLM_PROVIDER_ARCHITECT=ollama
FS_AGENT_LLM_MODEL_ARCHITECT=llama3.1:8b

# Use faster model for code generation
FS_AGENT_LLM_PROVIDER_BACKEND=ollama
FS_AGENT_LLM_MODEL_BACKEND=qwen2.5:7b

FS_AGENT_LLM_PROVIDER_FRONTEND=ollama
FS_AGENT_LLM_MODEL_FRONTEND=qwen2.5:7b
```

## Verification

Test your Ollama setup:

```bash
# Test the LLM connection
python -c "
from fs_agent.llm import build_llm_client
client = build_llm_client('ollama', model='llama3.1:8b', api_key='ollama')
result = client.generate('Say hello in one sentence')
print(result)
"
```

## Performance Tips

1. **Model Selection**: 
   - `llama3.1:8b` - Good balance of capability and speed
   - `qwen2.5:7b` - Faster, excellent for code tasks
   - `mistral:7b` - Lightweight, good for simpler tasks

2. **System Resources**: 
   - 8B models need ~8GB RAM
   - Ensure you have enough memory available
   - Close other memory-intensive applications

3. **First Run**: Models are loaded on first use, so initial requests may be slower

## Troubleshooting

### Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama if needed
ollama stop && ollama serve
```

### Model Not Found
```bash
# List available models
ollama list

# Pull the model you want to use
ollama pull llama3.1:8b
```

### Performance Issues
- Monitor memory usage: `top -o mem`
- Try smaller models if running out of RAM
- Ensure Ollama has sufficient system resources

## Example Usage

```bash
# Generate a project with Ollama
fs-agent run "Build a blog engine" \
  --llm-provider ollama \
  --llm-model llama3.1:8b \
  --pattern sequential

# Run benchmark with Ollama
fs-agent benchmark dataset/tasks.json \
  --max-tasks 3 \
  --patterns sequential,parallel \
  --llm-provider ollama \
  --llm-model qwen2.5:7b
```
