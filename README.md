# Smart Fridge GenAI POC -- Phase 1

A minimal, framework-free proof of concept that tests how accurately
**Qwen3-VL 8B** (a local multimodal vision-language model, run through
**Ollama**) can identify food ingredients from a photo of a refrigerator.

This is intentionally *not* an app. There is no web server, no database, no
agents, and no RAG. It's raw Python so the core GenAI vision call is easy to
see and reason about before any framework is introduced.

```
fridge.jpg -> Python -> Ollama -> Qwen3-VL 8B -> structured JSON -> console
```

## 1. What this POC does

1. Loads `uploads/fridge.jpg`.
2. Validates it's a real, non-corrupted JPEG/PNG/WEBP of a reasonable size.
3. Base64-encodes it and sends it, with a detailed instruction prompt, to
   Qwen3-VL 8B via the local Ollama HTTP API.
4. Parses the model's text response as JSON.
5. Validates that JSON against a Pydantic schema (rejecting malformed
   output, enforcing `0.0 <= confidence <= 1.0`, and flagging
   low-confidence ingredients rather than silently dropping them).
6. Prints the final, validated ingredient list to the console.

Qwen3-VL is a **vision-language model**: it can take both an image and a
text prompt as input and reason about what's in the image in natural
language. Here we're testing its raw visual food-recognition ability --
nothing more.

## 2. Prerequisites

- macOS
- Python 3.9+
- [Ollama](https://ollama.com) installed and runnable locally
- The `qwen3-vl:8b` model pulled into Ollama

## 3. Install Python dependencies

```bash
cd smartfridge
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 4. Verify Ollama is running

```bash
ollama list
```

You should see `qwen3-vl:8b` in the output. If Ollama itself isn't running,
you'll get a connection error instead of a model list -- start it with:

```bash
ollama serve
```

(or just open the Ollama desktop app, which runs the server in the
background). Leave that running in its own terminal/process, then re-run
`ollama list` in another terminal to confirm.

## 5. Download / run Qwen3-VL 8B

If `qwen3-vl:8b` isn't listed yet, pull it:

```bash
ollama pull qwen3-vl:8b
```

This downloads the ~6 GB model. `main.py` does **not** pull or manage the
model for you -- it only calls the model and gives you a clear error with
this exact command if the model is missing.

## 6. Where to place the fridge photo

Put your image at:

```
uploads/fridge.jpg
```

A sample image is already included there so you can run the POC
immediately. Supported formats are JPEG, PNG, and WEBP -- detected by
actual file content, not the file extension. Swap in your own photo by
overwriting `uploads/fridge.jpg`.

## 7. Configuration (optional)

Copy `.env.example` to `.env` to override defaults:

```bash
cp .env.example .env
```

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_VISION_MODEL` | `qwen3-vl:8b` | Model tag to call |
| `VISION_CONFIDENCE_THRESHOLD` | `0.60` | Ingredients below this confidence get `needs_confirmation: true` |
| `MAX_IMAGE_SIZE_MB` | `10` | Reject images larger than this |

## 8. Run it

```bash
python main.py
```

## 9. Example output

```json
{
  "ingredients": [
    {
      "name": "potato",
      "estimated_quantity": "3",
      "unit": "pieces",
      "confidence": 0.94,
      "needs_confirmation": false
    },
    {
      "name": "tomato",
      "estimated_quantity": "4",
      "unit": "pieces",
      "confidence": 0.91,
      "needs_confirmation": false
    },
    {
      "name": "onion",
      "estimated_quantity": "2",
      "unit": "pieces",
      "confidence": 0.88,
      "needs_confirmation": false
    }
  ]
}
```

The actual ingredients, quantities, and confidence scores depend entirely
on what Qwen3-VL sees in your specific `fridge.jpg` -- nothing here is
hardcoded.

## 10. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `Image validation failed: Image file not found` | No file at `uploads/fridge.jpg`. Add one. |
| `Image validation failed: Unsupported image format` | File isn't a real JPEG/PNG/WEBP (check with `file uploads/fridge.jpg`). |
| `Ollama check failed: Could not reach Ollama` | Ollama isn't running. Run `ollama serve` or open the Ollama app. |
| `Model not found` / `ollama pull qwen3-vl:8b` message | Model hasn't been pulled yet. Run the suggested `ollama pull` command. |
| `Ollama timed out` | The model is slow on your hardware (CPU-only Macs can take minutes for an 8B vision model). Try again, or use a smaller/quantized model. |
| `Model returned an invalid response` | The model didn't return valid JSON matching the schema. Usually transient -- re-run. Consistent failures may mean the model needs a lower `temperature` or a reworded prompt. |
| Response is slow | Vision models are compute-heavy. Performance depends heavily on your Mac's RAM/GPU (Apple Silicon with more unified memory will be noticeably faster). |

## What's explicitly out of scope for this POC

By design, this first experiment does **not** include: LangChain,
LangGraph, CrewAI, MCP, PostgreSQL, pgvector, RAG, agents, YOLO, FastAPI,
or any external search/API integration. The goal is to understand the bare
GenAI vision pipeline -- one image, one prompt, one model call, one
validated JSON result -- before layering anything else on top.
