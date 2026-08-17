# Provider Troubleshooting Matrix

| Symptom | What it means | Correct action |
|---|---|---|
| Groq diagnostic says `HTTP 403 ... 1010`, but a direct Groq SDK chat works | False-negative diagnostic/client-signature path rather than failed Groq chat authentication | Use this v3 pack: Groq diagnostics use the official SDK and verify chat before declaring the provider unavailable |
| `GROQ_API_KEY is not set` | `.env` not loaded or variable named incorrectly | Put `GROQ_API_KEY=...` in `resources/.env`, restart kernel, rerun Day 0 |
| Groq 401 | Invalid/revoked credential | Create a fresh key and replace the environment value |
| Groq model error / 404 | Model ID not available or no permission | Inspect Day 0 model list; choose an ID returned by the account |
| Groq 429 | Rate/usage limit | Reduce loops/concurrency and inspect Groq account limits |
| Groq outputs a long `<think>...</think>` block | Raw reasoning was returned | Set `GROQ_REASONING_FORMAT=hidden` for normal teaching output |
| Old Groq Llama model suddenly fails | Model retired | Use `qwen/qwen3.6-27b`, `openai/gpt-oss-20b` or `openai/gpt-oss-120b`; do not use the retired 8B/70B Llama defaults from the older pack |
| `ollama list` works but notebook says unreachable | Jupyter is on another host/container, or host URL differs | Compare `sys.executable`, notebook host and `OLLAMA_BASE_URL`; test `/api/tags` from the same environment |
| Ollama model not found | Model tag/case differs | Use exactly the name shown by `ollama list`; router also does case-insensitive matching |
| Ollama takes several minutes | Reasoning/model loading/long generation | `OLLAMA_THINK=false`, bounded `max_tokens`, `OLLAMA_KEEP_ALIVE=15m`; inspect `ollama ps` |
| Ollama returns empty final text | Reasoning-capable small model produced no final content or incompatible client behaviour | v3 raises a clear error; update Ollama packages or try `llama3.2:latest` / a stronger local model |
| Native Ollama tool calling is weak | Capability issue, especially for 0.8B/1B models | Use the controlled LangGraph workflow locally; compare with Groq native tool calling |
| `MessagesState is not defined` | Old Day 2 function annotation resolved an inner import incorrectly | Fixed in v3 by removing the fragile inner annotation |
| `pd is not defined` immediately after Restart/Run All | Old Day 0 relied on previous kernel state | Fixed in v3; imports happen before `pd` is used |
| RAG cell displays a blank answer | Old local inference returned empty final content | Fixed by bounded local generation + thinking disabled + explicit empty-output error |
| Notebook imports fail although Terminal install worked | Wrong Jupyter kernel | Run `sys.executable`, then install with `%pip` in that kernel or select the `Agentic AI Bootcamp` kernel |
| Paths point to a  directory | Notebook was previously executed on another machine | v3 dynamically locates `resources/`; no user-specific path is hard-coded |
| Provider output is actually `mimic` | No live provider succeeded on automatic routing | Inspect the metadata table; explicit provider calls never silently become mimic |

## Groq 1010 specifically

In the executed Day 0 you supplied, the direct Groq SDK completion succeeded with `qwen/qwen3.6-27b`, while the old `/models` check failed with `403 / 1010`. That is decisive evidence that the API key/chat path worked. The v3 diagnostic therefore no longer uses the old `urllib` implementation for Groq.

## Recommended diagnostic command sequence

```bash
# 1. Verify the intended Python/Jupyter environment
python --version
python -m pip show groq ollama langgraph langchain-groq langchain-ollama

# 2. Verify Ollama locally
ollama list
curl http://127.0.0.1:11434/api/tags

# 3. Start Jupyter from the activated venv
jupyter lab
```

Then run Day 0 from a **freshly restarted kernel** and use its provider/model/latency/token metadata as the baseline before proceeding to the agent notebooks.


### Groq Qwen classroom responsiveness

The default configuration sets `GROQ_REASONING_EFFORT=none` and `GROQ_REASONING_FORMAT=hidden` for `qwen/qwen3.6-27b`. This keeps ordinary classroom calls fast and prevents long reasoning traces from dominating the output. Set `GROQ_REASONING_EFFORT=default` only for an exercise where extended reasoning is intentionally being demonstrated.
