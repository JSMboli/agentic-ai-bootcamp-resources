# Provider Account and API-Key Setup

**Bootcamp edition:** 17 August 2026  
**Core recommendation:** use Groq for the hosted native-tool-calling demonstrations and Ollama for a no-cost local route. The remaining providers are optional extension paths.

## Security rules before creating any key

1. Treat every API key like a password.
2. Store keys in `resources/.env`, never in notebook source cells.
3. Never commit `resources/.env` to Git; this pack includes a `.gitignore` that excludes it.
4. If a key appears in chat, a screenshot, notebook output, Git history or a distributed ZIP, revoke it and create a new one.
5. After changing `resources/.env`, restart the Jupyter kernel and rerun Day 0.

The notebook loads the **variable name**, not the secret itself. Correct Groq example:

```python
import os
from groq import Groq
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
```

## Option A — Groq (recommended hosted path)

Official pages:
- Account/console: https://console.groq.com/
- API keys: https://console.groq.com/keys
- Quickstart: https://console.groq.com/docs/quickstart
- Active models: https://console.groq.com/docs/models
- Qwen 3.6 27B: https://console.groq.com/docs/model/qwen/qwen3.6-27b

Steps:
1. Create or sign in to a GroqCloud account.
2. Open **API Keys** and create a project key.
3. Copy `resources/.env.template` to `resources/.env`.
4. Add:

```text
GROQ_API_KEY=your_key_here
GROQ_MODEL=qwen/qwen3.6-27b
GROQ_REASONING_FORMAT=hidden
GROQ_REASONING_EFFORT=none
```

5. Restart the notebook kernel.
6. Run Day 0. The diagnostic now uses Groq's official SDK (`client.models.list()`), and if model-list discovery itself is blocked it performs a tiny official-SDK chat smoke test before deciding that Groq is unavailable.

**Why `hidden`?** It keeps the model's private reasoning out of the main teaching answer. Students see the final answer plus operational metadata (provider, model, latency, token usage and request ID where available).

## Option B — Ollama (local/no API cost)

Official pages:
- Download/install: https://ollama.com/download
- API introduction: https://docs.ollama.com/api/introduction
- Chat API: https://docs.ollama.com/api/chat
- Tool calling: https://docs.ollama.com/capabilities/tool-calling
- Python library: https://github.com/ollama/ollama-python

After installing Ollama, verify in Terminal:

```bash
ollama list
curl http://127.0.0.1:11434/api/tags
```

Example configuration for the lecturer machine used to validate this bootcamp:

```text
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3.5:0.8b
OLLAMA_THINK=false
OLLAMA_KEEP_ALIVE=15m
```

The router discovers the server's actual model list. It does not assume a model exists, and matching is case-insensitive (for example `deepseek-r1:1.5B` versus `deepseek-r1:1.5b`).

For classroom responsiveness the notebooks use bounded output, keep the model loaded between calls, and disable extended thinking by default. A 0.8B/1B model is useful for demonstrating local inference, but native multi-tool selection may be less reliable than with a stronger hosted model; that is why the controlled LangGraph workflow remains available.

## Option C — OpenAI (optional extension)

Official pages:
- Developer quickstart: https://developers.openai.com/api/docs/quickstart
- Model catalogue: https://developers.openai.com/api/docs/models
- API key/dashboard entry point: https://platform.openai.com/api-keys

Configuration:

```text
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5.6-luna
```

The provider helper uses the official OpenAI Python SDK and the Responses API for this optional route. `gpt-5.6-luna` is chosen as the cost-sensitive member of the current GPT-5.6 family for classroom experimentation; students should always re-check the model catalogue before a later delivery of the course.

## Option D — Gemini (optional extension)

Official pages:
- API-key guide: https://ai.google.dev/gemini-api/docs/api-key
- Google AI Studio: https://aistudio.google.com/
- Model catalogue: https://ai.google.dev/gemini-api/docs/models

Steps:
1. Sign in to Google AI Studio.
2. Open the API Keys page and choose **Create API key**.
3. Add one of the supported key variable names:

```text
GEMINI_API_KEY=your_key_here
# or GOOGLE_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.7-flash
```

`gemini-3.7-flash` is the current Flash model listed in Google's August 2026 documentation. Provider catalogues change, so verify the model page when reusing these resources later.

## Option E — Anthropic (optional extension)

Official pages:
- Get started: https://docs.anthropic.com/en/docs/get-started
- Model overview: https://docs.anthropic.com/en/docs/about-claude/models/overview
- Claude Console: https://console.anthropic.com/

Steps:
1. Create/sign in to a Claude Console account.
2. Open **Settings → API keys** and create a key.
3. Configure:

```text
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_MODEL=claude-sonnet-5
```

`claude-sonnet-5` is used as the current speed/intelligence balance in this bootcamp edition.

## Option F — DeepSeek (optional extension)

Official pages:
- API documentation: https://api-docs.deepseek.com/
- API-key page: https://platform.deepseek.com/api_keys
- Current models/pricing: https://api-docs.deepseek.com/quick_start/pricing

Configuration:

```text
DEEPSEEK_API_KEY=your_key_here
DEEPSEEK_MODEL=deepseek-v4-flash
```

The older `deepseek-chat` and `deepseek-reasoner` identifiers were retired in July 2026. This pack therefore uses the current `deepseek-v4-flash` API model for the optional DeepSeek route.

## What if a student has no cloud API key?

Nothing essential is lost. They can use:

1. **Ollama**, if the machine can run a local model; or
2. the deterministic **mimic** provider, which preserves the workflow, state, tool and HITL exercises without pretending to be a real LLM.

The notebooks always display the selected provider and model so a mimic run cannot be mistaken for a live model result.


### Groq Qwen classroom responsiveness

The default configuration sets `GROQ_REASONING_EFFORT=none` and `GROQ_REASONING_FORMAT=hidden` for `qwen/qwen3.6-27b`. This keeps ordinary classroom calls fast and prevents long reasoning traces from dominating the output. Set `GROQ_REASONING_EFFORT=default` only for an exercise where extended reasoning is intentionally being demonstrated.
