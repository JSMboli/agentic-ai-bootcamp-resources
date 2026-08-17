# Executed-Notebook and Directory Audit — v3

**Audit date:** 17 August 2026  
**Source inspected:** the executed `Agentic_AI_Bootcamp_Reworked_LangGraph_Ollama_Groq_HITL_2` package supplied after running all notebooks.

This report distinguishes **provider failure**, **diagnostic failure**, **model-capability/performance behaviour**, **notebook state leakage**, and **ordinary code defects**. Those are different failure classes and should not be reported to students as one generic “LLM unavailable” problem.

## Executive diagnosis

The saved outputs show that both core routes were substantially healthier than the old diagnostics suggested:

- **Groq chat worked.** A direct official-SDK completion succeeded with `qwen/qwen3.6-27b` and returned a full response.
- The subsequent old Groq API diagnostic alone returned **`HTTP 403: error code: 1010`**. Because the real SDK chat had already succeeded, this was a **false negative in the diagnostic path**, not evidence that the Groq key/chat route was unavailable.
- **Ollama worked and discovered four local models:** `qwen3.5:0.8b`, `deepseek-r1:1.5B`, `llama3.2:1b`, and `llama3.2:latest`.
- One Ollama call took **226.487 seconds**, so “reachable” did not mean “classroom-responsive”.
- Day 2's optional local native-tool agent failed because of a Python/LangGraph code defect: **`MessagesState is not defined`**.
- Day 4's grounded-generation cell completed without a Python exception but produced **empty final text**.
- The old package included a machine-specific `.venv`, a live `.env`, caches and notebook checkpoints that should not be distributed.

## Directory and packaging audit

| Item | Executed package | v3 correction |
|---|---:|---|
| Extracted package size | ~769 MB | ~28 MB before final ZIP |
| `resources/.venv` | ~742 MB | Removed; students create their own environment |
| `resources/.env` | Present and contained a live credential | Removed from distribution; only `.env.template` retained |
| `.ipynb_checkpoints` | Present | Removed |
| `__pycache__` directories | 1,316 in the supplied tree | Removed before final packaging |
| Saved working directory | Lecturer-specific macOS/OneDrive absolute path | Runtime resource locator searches relative package structure |
| Saved Jupyter executable | `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11` | Day 0 always prints `sys.executable`; setup installs a named bootcamp kernel |
| Notebook resource/data path | Resolved correctly on lecturer machine | Explicitly displayed in every setup cell |

### Important path observation

The executed notebooks were launched from a directory ending in:

```text
.../Agentic_AI_Bootcamp_Reworked_LangGraph_Ollama_Groq_HITL_2/resources/notebooks
```

That means commands such as `pip install -r resources/requirements.txt` are **wrong when executed literally from that notebook directory**; they would look for `resources/notebooks/resources/requirements.txt`. The v3 notebook explains that from `resources/notebooks/` the relative requirements path is `../requirements.txt`, while the setup guide uses `resources/requirements.txt` only after explicitly instructing the student to run it from the package root.

---

# Day 0 — Setup and Provider Check

| Code cell | Saved result | Audit finding | v3 action |
|---|---|---|---|
| 1 | Executed as count 11 | Contains `pd.set_option(...)` before pandas is imported later. It only succeeded because `pd` remained in kernel state from an earlier run. | Removed state leakage; imports are performed before use. Fresh-run compilation/execution is now part of validation. |
| 2 | Printed Python 3.11.1, Jupyter executable, working directory and resources path | Useful diagnostic; confirmed notebook was running from the lecturer's macOS/OneDrive path. | Expanded into an environment table showing kernel, working directory, resource path, data path and `.env` candidates. |
| 3 | All required imports reported available | Environment itself was largely healthy. | Added package versions, not only booleans. |
| 4 | Ollama `True`; Groq `False`; optional providers mostly `False` | Groq status was wrong because later direct SDK inference succeeded. | Groq diagnostic rewritten around official SDK; optional providers shown separately. |
| 5 | `GROQ_API_KEY present: True`; Groq model and Ollama settings printed | Correctly proved environment-variable loading was working. | Preserved, but secrets are never printed. |
| 6 | Direct Groq SDK call succeeded with `qwen/qwen3.6-27b`; output contained a long `<think>` trace | This is decisive evidence that Groq authentication/chat was functional. Raw reasoning made teaching output noisy. | `GROQ_REASONING_FORMAT=hidden` by default; response metadata displayed separately. |
| 7 | `Groq API check failed: HTTP 403: error code: 1010` | False-negative diagnostic. It conflicted with the immediately preceding successful Groq completion. | Removed low-level `urllib` Groq check. Uses `client.models.list()` and, if that fails, a tiny SDK chat smoke test before declaring Groq unavailable. |
| 8 | Ollama reachable; four models found; local generation succeeded | Ollama installation/server/model discovery were working. | Preserved and expanded: CLI path, host, model count, selected model and discovery method. |
| 9 | Ollama explicit test took **226.487s**; Groq skipped because diagnostic said unavailable | Two separate issues: severe local latency and bad Groq gating. | Ollama `think=False`, bounded generation and keep-alive; Groq no longer skipped merely because model-list discovery fails. Progress indicator prevents an apparently frozen cell. |
| 10 | Automatic route selected Ollama and generated a response | Routing logic worked according to its incorrect status table, but therefore preferred a much slower provider over a working Groq route. | Automatic routing uses corrected diagnostics; explicit calls still fail loudly. |
| 11 | Empty cell | No issue. | Removed unnecessary empty tail cell in rebuilt notebook. |

**Day 0 conclusion:** the original headline error was not “Groq is broken”. It was “the Groq diagnostic was not representative of the successful SDK chat path”.

---

# Day 1 — LLM Routing and Warm-up

| Code cell | Saved result | Audit finding | v3 action |
|---|---|---|---|
| 1 | Environment/path output successful | Resource locator worked on lecturer machine. | Environment table standardised. |
| 2 | Primary provider set to `ollama` | Choice inherited false Groq-unavailable status. | Primary provider selection now reflects corrected Groq SDK diagnostic. |
| 3 | Provider comparison ran Ollama (`18.299s`) and mimic; Groq absent | Comparison was incomplete because the diagnostic incorrectly excluded Groq. | Comparison iterates available providers with `tqdm`, per-call spinner and provider/model/latency/token metadata. |
| 4 | Multi-turn conversation worked | Good teaching concept; application-managed history was visible. | Retained and explained as explicit state before LangGraph persistence. |
| 5 | Structured agent-card JSON generated | Worked; useful transition from prompting to system design. | Retained with clearer schema: goal, tools, state, authority, human oversight and stop/failure conditions. |
| 6 | Empty cell | No issue. | Removed in rebuild. |

---

# Day 2 — LangGraph Tools, State and HITL

| Code cell | Saved result | Audit finding | v3 action |
|---|---|---|---|
| 1 | Environment/path output successful | No runtime defect. | Standardised environment/resource setup. |
| 2 | `LIVE_PROVIDER = ollama` | Again inherited the false Groq status. | Corrected provider diagnostic and clear primary-provider banner. |
| 3 | Calculator, pick-rate and risk tools returned expected values | Deterministic tool layer was healthy. | Retained; notebook now explicitly tests tools before giving them to an agent. |
| 4 | Controlled workflow produced a grounded answer | LangGraph controlled workflow was functional. | Rebuilt with six visible node stages and LLM metadata persisted in graph state. |
| 5 | High-risk request returned a genuine LangGraph `Interrupt` payload | HITL concept was correctly implemented. | Retained and expanded review packet with provider/model provenance. |
| 6 | Resumed graph returned safe human-reviewed text | `Command(resume=...)` path worked. | Retained; state/pause/resume explained in Markdown. |
| 7 | Groq native-tool demo skipped | Caused by false diagnostic, not by proven Groq chat failure. | Corrected status allows Groq native tool-calling demo to run when SDK chat/model is available. |
| 8 | `Ollama native-tool experiment failed: name 'MessagesState' is not defined` | Genuine code defect. `MessagesState` was imported inside the factory while postponed annotation resolution looked in module globals. | Fragile function annotation removed; `MessagesState` still defines the graph state normally. |
| 9 | Empty cell | No issue. | Removed in rebuild. |

**Day 2 conclusion:** controlled LangGraph + HITL worked; the local native-tool failure was a Python/type-hint scoping bug, not an Ollama connectivity failure.

---

# Day 3 — Amazon Fulfilment-Centre Visit Bridge

| Code cell | Saved result | Audit finding | v3 action |
|---|---|---|---|
| 1 | Environment/path output successful | No path failure. | Standardised setup. |
| 2 | Provider set to Ollama; synthetic observation data loaded | Data path was healthy. | Visit date and evidence boundaries made explicit. |
| 3 | Five area analyses completed with Ollama/model reported | Functional, but provider metadata was limited and students could wait through serial calls without clear overall progress. | `tqdm` across observations plus per-call spinner and metadata columns. |
| 4 | Student-note examples printed | No exception. | Expanded into responsible note triage, explicitly excluding personal/customer/security/proprietary data. |
| 5 | Empty cell | No issue. | Removed. |

The notebook now states explicitly that the industrial visit is **Thursday, 20 August 2026**, and that the supplied pre-visit records are synthetic/generalised teaching observations rather than claims about Amazon's confidential internal systems.

---

# Day 4 — RAG and Human-in-the-Loop

| Code cell | Saved result | Audit finding | v3 action |
|---|---|---|---|
| 1 | Environment/path output successful | No path defect. | Standardised setup. |
| 2 | Provider Ollama; knowledge base loaded (10 rows) | Data source available. | Retained with provenance/freshness/access-control discussion. |
| 3 | TF-IDF retrieval returned relevant rows with similarity values | Retrieval layer itself worked. | Retained as a no-cost, deterministic and explainable baseline. |
| 4 | **No visible model answer output** despite cell executing | The grounded generation path yielded empty final content from the selected local reasoning model. This was not a retrieval failure. | Local thinking disabled by default; bounded output; provider raises an explicit error if final content is empty; spinner and metadata added. |
| 5 | High-risk request produced LangGraph interrupt | HITL gate worked. | Retained. |
| 6 | Resume with rejection returned `The proposed output was rejected...` | Correct review behaviour. | Retained and explained as authority control distinct from retrieval quality. |
| 7 | Empty cell | No issue. | Removed. |

---

# Day 5 — Capstone

| Code cell | Saved result | Audit finding | v3 action |
|---|---|---|---|
| 1 | Environment/path output successful | No path defect. | Standardised setup. |
| 2 | Primary provider `ollama` | Inherited false Groq diagnostic. | Corrected status selection. |
| 3 | Capstone system contract displayed | Good architecture-first teaching element. | Retained and expanded. |
| 4 | Four evaluation cases displayed | Good breadth: ordinary, calculation, high-risk and unsupported/confidential request. | Retained with explicit expected behaviour. |
| 5 | Graph/evaluation cell ran | Saved output was limited, making long evaluation less transparent. | Overall `tqdm` progress + node progress + per-case metadata. |
| 6 | Evaluation dataframe displayed, including interruption outcomes | Behavioural evaluation was useful. | Expanded observability columns and rubric discussion. |
| 7 | Native Groq extension skipped | Again caused by false diagnostic. | Runs when corrected Groq SDK diagnostic passes. |
| 8 | Empty cell | No issue. | Removed. |

---

# Cross-notebook engineering changes in v3

## 1. One provider abstraction, explicit metadata

Every successful `LLMResponse` now carries:

```text
provider
model
latency_seconds
input_tokens
output_tokens
total_tokens
finish_reason
request_id
endpoint
reasoning_mode
```

Provider SDKs do not all expose every field, so unavailable fields remain `None` rather than being fabricated.

## 2. Progress is visible at three levels

- **Single blocking call:** animated browser spinner with final elapsed time.
- **Batch/loop:** `tqdm` progress bar.
- **LangGraph:** `[1/6]`, `[2/6]` ... node messages plus model/tool messages.

## 3. Explicit calls cannot masquerade as success

```python
router.chat(messages, provider="groq", fallback_on_error=False)
```

raises a real provider error. Only automatic classroom routing can eventually use `mimic`.

## 4. Current model identifiers refreshed

The v3 configuration uses:

```text
Groq:      qwen/qwen3.6-27b
OpenAI:    gpt-5.6-luna          (optional)
Gemini:    gemini-3.7-flash      (optional)
Anthropic: claude-sonnet-5       (optional)
DeepSeek:  deepseek-v4-flash     (optional)
Ollama:    discovered locally; default preference qwen3.5:0.8b
```

Students should still treat provider model catalogues as changing dependencies and re-check them when the course is delivered again.

## 5. Fresh-kernel acceptance criterion

A notebook is considered valid only if:

1. every Python code cell compiles independently;
2. a clean kernel can execute cells in notebook order;
3. no secret or lecturer-specific virtual environment is required;
4. no hidden variable from an earlier run is required;
5. provider failure is distinguishable from workflow/tool/RAG failure.

The final package is subjected to these checks before release.


# Final validation performed for v3

The rebuilt package was subjected to the following release checks on 17 August 2026:

1. **Python source compilation:** every module under `resources/src/` compiles successfully.
2. **Notebook code compilation:** every ordinary Python code cell in all six notebooks compiles successfully.
3. **Fresh-kernel execution:** all six notebooks execute from start to finish in separate fresh kernels using the deliberately offline `mimic` route, proving that they do not depend on hidden state, a paid API key or an instructor-specific path.
4. **Groq provider mock test:** normal model listing/chat succeeds; the exact `models.list() -> 403/1010` plus `chat -> success` scenario is treated as **Groq available**, not as a false negative.
5. **Ollama provider mock test:** four instructor-style model IDs are discovered; a requested `deepseek-r1:1.5b` correctly resolves to installed `deepseek-r1:1.5B`; inference passes `think=False`, bounded generation and `keep_alive`.
6. **Explicit error semantics:** a failed explicit Groq call raises `ProviderError` instead of silently changing provider.
7. **Distribution security scan:** release package contains no live `.env`, `.venv`, notebook checkpoint, `__pycache__`, `.pyc`, lecturer-specific absolute macOS path or Groq key prefix.
8. **Archive integrity:** the final ZIP is tested after creation.

Live Groq and live Ollama calls must still be validated on the student's/instructor's machine because credentials, local server state, hardware and network controls are machine-specific. Day 0 is the authoritative live pre-flight check.
