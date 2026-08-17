# Agentic AI Bootcamp Resources - Reliable Providers v3

This edition makes Groq and Ollama first-class, observable providers and removes the silent-failure behaviour found in the earlier pack.

## Main engineering changes

- Groq diagnostics use the **official Groq Python SDK** for `client.models.list()`. If model listing fails but a tiny SDK chat smoke test succeeds, Groq is correctly reported as usable.
- Correct Groq environment-variable usage: `GROQ_API_KEY`.
- `qwen/qwen3.6-27b` is the primary Groq model; obsolete fallback model IDs were removed.
- Qwen 3.6 classroom calls default to `GROQ_REASONING_EFFORT=none` and `GROQ_REASONING_FORMAT=hidden` for responsive general-purpose exercises.
- Ollama discovery uses the official Python client first and `/api/tags` only as a fallback.
- Ollama model matching is case-insensitive and selects from the models actually installed on the machine.
- Ollama calls default to `think=False`, bounded output and `keep_alive` to improve classroom responsiveness.
- `.env` discovery is independent of the notebook's working directory.
- Explicit provider calls fail visibly; only automatic classroom routing may eventually use `mimic`.
- Every successful provider call returns provider/model/latency/token/endpoint/request metadata when available.
- Long calls have an animated progress indicator; loops use `tqdm`; LangGraph nodes print their progress.
- LangGraph practicals include deterministic tools, state, checkpointing, model-driven tool calling and a real Human-in-the-Loop `interrupt`/resume flow.
- Bundled `.venv`, live `.env`, caches and notebook checkpoints are removed.
- All six notebooks pass code-cell compilation and fresh-kernel offline execution tests.

Start with `PROVIDER_ACCOUNT_SETUP.md`, then `SETUP_GUIDE.md`, then `notebooks/Day_0_Setup_and_Provider_Check.ipynb`.
