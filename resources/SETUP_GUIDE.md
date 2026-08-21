# Agentic AI Bootcamp - Reliable Setup Guide

**Edition date:** 17 August 2026

This guide is designed so a student can start from a newly extracted ZIP and reproduce the practical environment without relying on hidden notebook state or a lecturer-specific virtual environment.

## 1. Security first

Never paste a live API key into source code, a notebook, slides, screenshots, GitHub, Teams, email or chat. If a key is exposed, revoke it and create a replacement.

For Groq, the environment variable is named `GROQ_API_KEY`:

```python
import os
from groq import Groq
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
```

The string passed to `os.environ.get(...)` is the **variable name**, not the secret key itself.

See `PROVIDER_ACCOUNT_SETUP.md` for step-by-step Groq, Ollama, OpenAI, Gemini, Anthropic and DeepSeek setup.

## 2. Extract the complete ZIP and create a clean environment

Open Terminal/PowerShell **at the package root**, the directory containing `resources/`, `slides/`, and the programme files.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r resources/requirements.txt
python -m ipykernel install --user --name agentic-ai-bootcamp --display-name "Agentic AI Bootcamp"
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r resources/requirements.txt
python -m ipykernel install --user --name agentic-ai-bootcamp --display-name "Agentic AI Bootcamp"
```

Then start Jupyter and select the **Agentic AI Bootcamp** kernel. Day 0 prints the exact `sys.executable`, working directory and resolved resource/data directories so environment mismatches are visible.

**Do not copy another person's `.venv`.** Virtual environments contain machine- and interpreter-specific paths and binaries. The distributed pack intentionally excludes `.venv`.

## 3. Configure environment variables

From the package root:

```bash
cp resources/.env.template resources/.env
```

On Windows, copy the file in Explorer or use:

```powershell
Copy-Item resources\.env.template resources\.env
```

Edit only `resources/.env`. The provider module searches:

1. `resources/.env`;
2. package-root `.env`;
3. current-working-directory `.env`.

The first existing files are loaded without overwriting variables already defined in the operating-system environment.

## 4. Groq - corrected diagnostic path

The executed notebook you supplied demonstrated an important distinction:

- a normal Groq SDK chat request to `qwen/qwen3.6-27b` **succeeded**;
- the older low-level `/models` check returned `HTTP 403: error code 1010`.

That meant the provider was usable and the old diagnostic was wrong. The revised code therefore **does not use `urllib` for Groq diagnostics**. It uses Groq's official Python SDK:

```python
client.models.list()
```

If model listing fails while credentials are present, the diagnostic performs a tiny official-SDK chat completion (`Reply with OK.`). A successful chat marks Groq as **available** and records the model-list error as diagnostic information rather than blocking every Groq practical.

Recommended configuration:

```text
GROQ_API_KEY=your_new_key_here
GROQ_MODEL=qwen/qwen3.6-27b
GROQ_REASONING_FORMAT=hidden
```

The current pack no longer includes the retired Groq fallbacks `llama-3.1-8b-instant` or `llama-3.3-70b-versatile`. Current fallbacks are `qwen/qwen3.6-27b`, `openai/gpt-oss-20b`, and `openai/gpt-oss-120b`.

## 5. Ollama - local route


Verify from the same machine that will run Jupyter:

```bash
ollama list

```

My machine used for this pack reported:

```text
qwen3.5:0.8b
deepseek-r1:1.5B
llama3.2:1b
llama3.2:latest
```

Recommended starting configuration:

```text
OLLAMA_BASE_URL=
OLLAMA_MODEL=qwen3.5:0.8b
OLLAMA_THINK=false
OLLAMA_KEEP_ALIVE=15m
```

Why these defaults?

- `think=false` prevents long hidden/reasoning phases from consuming most of a class exercise;
- bounded `max_tokens`/`num_predict` limits output time;
- `keep_alive=15m` reduces repeated model-load overhead across adjacent cells;
- model discovery uses the live Ollama list instead of a hard-coded assumption.

If `ollama list` works in Terminal but Jupyter cannot connect, Day 0 now shows the kernel executable, host URL and resolved directories. A notebook running in a container, remote server or VM has a different `127.0.0.1` from the Mac host.

## 6. Notebook run order

1. `resources/notebooks/Day_0_Setup_and_Provider_Check.ipynb`
2. `resources/notebooks/Day_1_LLM_Routing_and_Ollama_Groq_Warmup.ipynb`
3. `resources/notebooks/Day_2_LangGraph_Tool_Calling_State.ipynb`
4. `resources/notebooks/Day_3_Amazon_Visit_Observation_Agent.ipynb`
5. `resources/notebooks/Day_4_RAG_and_Human_in_the_Loop.ipynb`
6. `resources/notebooks/Day_5_Capstone_Fulfilment_Centre_Support_Agent.ipynb`

For validation, use **Kernel → Restart Kernel and Run All Cells**. A notebook that only works after cells have been executed out of order is not considered valid.

## 7. Progress and observability

Long-running LLM cells now use two complementary indicators:

- a browser-side animated spinner around blocking provider calls; and
- `tqdm`/node-level progress for loops and LangGraph workflows.

Successful LLM calls return a metadata panel containing, where the provider supplies it:

- provider;
- model;
- latency;
- input/output/total token counts;
- finish reason;
- request ID;
- endpoint;
- reasoning mode.

This prevents students from confusing an Ollama response, a Groq response and the offline mimic fallback.

## 8. Explicit versus automatic routing

Debugging call:

```python
router.chat(messages, provider="groq", fallback_on_error=False)
```

If Groq fails, the error is shown. It does **not** silently become a mimic answer.

Inclusive classroom call:

```python
router.chat(messages, provider=None)
```

Automatic routing can try configured live providers and eventually use `mimic` so a student without paid/cloud access can still execute the workflow structure.

## 9. LangGraph architecture

The pack deliberately teaches both:

### Controlled graph

`risk → retrieval → deterministic calculation → LLM synthesis → HITL gate → finalise`

This is preferred when reliability and auditability dominate, particularly with very small local models.

### Native tool-calling graph

`model → tool(s) → model → ... → final`

The model chooses tools. Groq is recommended for the main live demonstration; local Ollama tool calling remains an experiment so students can observe model-capability differences.

The HITL practical uses LangGraph `interrupt()` and a checkpointer, then resumes the same thread with `Command(resume=...)` rather than merely printing a disclaimer.

## 10. Industrial visit boundary

Thursday, **20 August 2026**, is the Amazon fulfilment-centre industrial visit. The practical treats visit material as permitted process-level observations, not as access to Amazon's confidential internal systems. Students should not record personal/customer data, credentials, prohibited photographs, security controls or proprietary thresholds.

## 11. Troubleshooting order

When a cell fails, do not start by changing model code. Check in this order:

1. kernel executable;
2. resource/data paths;
3. package import/version table;
4. `.env` discovery;
5. provider-specific diagnostic;
6. selected model exists and is permitted;
7. a minimal one-prompt provider smoke test;
8. only then the LangGraph/tool/RAG layer.

See `PROVIDER_TROUBLESHOOTING.md` and `AUDIT_REPORT_v3.md` for the concrete failures observed in the executed notebooks you supplied.


### Groq Qwen classroom responsiveness

The default configuration sets `GROQ_REASONING_EFFORT=none` and `GROQ_REASONING_FORMAT=hidden` for `qwen/qwen3.6-27b`. This keeps ordinary classroom calls fast and prevents long reasoning traces from dominating the output. Set `GROQ_REASONING_EFFORT=default` only for an exercise where extended reasoning is intentionally being demonstrated.
