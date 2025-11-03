# vLLM Setup Guide

This guide explains how to set up and use vLLM for serving models with OpenAI-compatible API endpoints.

## Installation

```bash
pip install vllm
```

## Starting a vLLM Server

### Basic Usage

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 --dtype auto --api-key token-abc123
```

### Common Options

```bash
vllm serve <model_name> \
  --dtype auto \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --api-key <your_api_key>
```

### Key Parameters

- `--dtype auto`: Automatically detect the optimal data type
- `--host 0.0.0.0`: Bind to all network interfaces (use `127.0.0.1` for localhost only)
- `--port 8000`: Port to serve on
- `--tensor-parallel-size N`: Number of GPUs to use for tensor parallelism
- `--gpu-memory-utilization 0.9`: Fraction of GPU memory to use (0.0 to 1.0)
- `--api-key <key>`: API key for authentication (optional but recommended)

## Configuration

### Using vLLM with This Project

The project is configured to use vLLM via the `config/llm/vllm.yaml` file:

```yaml
write:
  _target_: langchain_openai.ChatOpenAI
  model: your-model-name
  base_url: ${oc.env:VLLM_BASE_URL,http://localhost:8000/v1}

code:
  _target_: langchain_openai.ChatOpenAI
  model: your-model-name
  base_url: ${oc.env:VLLM_BASE_URL,http://localhost:8000/v1}
```

### Environment Variables

You can set the vLLM server URL via environment variable:

```bash
export VLLM_BASE_URL=http://your-server:8000/v1
```

**Important:** The URL must include `/v1` at the end for OpenAI API compatibility.

Or hardcode it directly in the YAML file.

## Example: Running on a Cluster

### On the vLLM Server Machine

```bash
# Start the vLLM server
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --gpu-memory-utilization 0.9 \
  --api-key your-secret-key
```

### On the Client Machine

```bash
# Set the vLLM server URL
export VLLM_BASE_URL=http://cluster-node-01:8000/v1

# Run your application with vLLM configuration
python your_script.py llm=vllm
```

## Testing the Server

Once the server is running, you can test it with curl.

First, check available models:
```bash
curl http://localhost:8000/v1/models
```

Then test a chat completion:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key" \
  -d '{
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

If your server doesn't require authentication (no `--api-key` was set), omit the Authorization header.

## Multiple Models

To serve multiple models, start multiple vLLM instances on different ports:

```bash
# Server 1
vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8000 --api-key key1

# Server 2
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8001 --api-key key2
```

Then configure different base URLs in your YAML files or use environment variables.

## Troubleshooting

### Out of Memory

Reduce `--gpu-memory-utilization`:
```bash
vllm serve <model> --gpu-memory-utilization 0.7
```

### Multi-GPU Setup

Use tensor parallelism for large models:
```bash
vllm serve <model> --tensor-parallel-size 2
```

### Performance Tuning

- Use `--dtype auto` for optimal performance
- Adjust `--max-model-len` to limit context length if needed
- Use `--enable-prefix-caching` for repeated prompts

## Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [OpenAI API Compatibility](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
