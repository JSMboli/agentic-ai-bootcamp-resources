"""Reliable multi-provider LLM access for the Agentic AI Bootcamp.

Design goals
------------
* Groq and Ollama are first-class providers.
* Groq diagnostics use Groq's official Python SDK, not a hand-written urllib
  request, because security layers can treat low-level clients differently.
* Ollama diagnostics distinguish server reachability, local model discovery,
  model selection and actual inference.
* Explicit provider requests fail loudly. Automatic routing may fall back.
* Provider/model/latency/token metadata is returned for every successful call.
* Local thinking is disabled by default for classroom responsiveness; it can be
  enabled explicitly with OLLAMA_THINK=true.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - setup diagnostic handles missing package
    load_dotenv = None

RESOURCE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_DIR = RESOURCE_DIR.parent
_ENV_FILES = [RESOURCE_DIR / ".env", PACKAGE_DIR / ".env", Path.cwd() / ".env"]

if load_dotenv:
    for _path in _ENV_FILES:
        if _path.exists():
            load_dotenv(_path, override=False)


class ProviderError(RuntimeError):
    """A provider call failed with a user-actionable message."""


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name, default)
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _safe_exception(exc: Exception) -> str:
    """Return a concise error description without credentials or large bodies."""
    status = getattr(exc, "status_code", None)
    message = str(exc).strip()
    if status:
        return f"HTTP {status}: {message}"
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                payload = json.loads(body)
                err = payload.get("error", payload)
                if isinstance(err, dict):
                    body = err.get("message") or err.get("error") or str(err)
            except Exception:
                pass
        except Exception:
            body = ""
        return f"HTTP {exc.code}: {body or exc.reason}"
    return f"{type(exc).__name__}: {message}"


def _json_request(
    url: str,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 20,
) -> Any:
    """Small REST helper used for non-Groq fallback integrations."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    merged_headers = {
        "Content-Type": "application/json",
        "User-Agent": "Agentic-AI-Bootcamp/3.0",
    }
    if headers:
        merged_headers.update(headers)
    req = urllib.request.Request(url, data=data, headers=merged_headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}


@dataclass
class ProviderDiagnostic:
    provider: str
    available: bool
    detail: str
    model: Optional[str] = None
    models: List[str] = field(default_factory=list)
    endpoint: Optional[str] = None
    latency_seconds: Optional[float] = None
    checks: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "available": self.available,
            "detail": self.detail,
            "model": self.model,
            "models": self.models,
            "endpoint": self.endpoint,
            "latency_seconds": self.latency_seconds,
            "checks": self.checks,
        }


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_seconds: float
    raw: Optional[Any] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    request_id: Optional[str] = None
    endpoint: Optional[str] = None
    reasoning_mode: Optional[str] = None

    def summary(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "latency_s": self.latency_seconds,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "finish_reason": self.finish_reason,
            "request_id": self.request_id,
            "endpoint": self.endpoint,
            "reasoning_mode": self.reasoning_mode,
        }


class ProviderRouter:
    """Provider discovery, diagnostics, routing and metadata collection."""

    def __init__(
        self,
        preferred_order: Optional[List[str]] = None,
        verbose: bool = True,
        diagnostic_ttl_seconds: int = 60,
    ):
        configured_order = _env("AUTO_PROVIDER_ORDER")
        if preferred_order is None and configured_order:
            preferred_order = [p.strip().lower() for p in configured_order.split(",") if p.strip()]
        self.preferred_order = preferred_order or [
            "groq", "ollama", "openai", "gemini", "anthropic", "deepseek", "mimic"
        ]
        self.verbose = verbose
        self.diagnostic_ttl_seconds = diagnostic_ttl_seconds
        self._cache: Dict[str, tuple[float, ProviderDiagnostic]] = {}
        self.ollama_base_url = (_env("OLLAMA_BASE_URL", "http://127.0.0.1:11434") or "").rstrip("/")

    @property
    def loaded_env_candidates(self) -> List[str]:
        return [str(p) for p in _ENV_FILES if p.exists()]

    def clear_diagnostic_cache(self) -> None:
        self._cache.clear()

    def _cache_get(self, provider: str, refresh: bool) -> Optional[ProviderDiagnostic]:
        if refresh:
            return None
        entry = self._cache.get(provider)
        if not entry:
            return None
        created, value = entry
        return value if time.monotonic() - created <= self.diagnostic_ttl_seconds else None

    def _cache_put(self, provider: str, value: ProviderDiagnostic) -> ProviderDiagnostic:
        self._cache[provider] = (time.monotonic(), value)
        return value

    # ------------------------------------------------------------------
    # Ollama diagnostics and selection
    # ------------------------------------------------------------------
    @staticmethod
    def _ollama_model_names(response: Any) -> List[str]:
        models = getattr(response, "models", None)
        if models is None and isinstance(response, dict):
            models = response.get("models", [])
        names: List[str] = []
        for item in models or []:
            name = getattr(item, "model", None) or getattr(item, "name", None)
            if name is None and isinstance(item, dict):
                name = item.get("model") or item.get("name")
            if name:
                names.append(str(name))
        return names

    def diagnose_ollama(self, refresh: bool = False) -> ProviderDiagnostic:
        cached = self._cache_get("ollama", refresh)
        if cached:
            return cached

        cli = shutil.which("ollama")
        start = time.perf_counter()
        checks: Dict[str, Any] = {
            "cli_found": bool(cli),
            "cli_path": cli,
            "base_url": self.ollama_base_url,
        }
        try:
            from ollama import Client
            checks["python_sdk_importable"] = True
            client = Client(host=self.ollama_base_url)
            listing = client.list()
            models = self._ollama_model_names(listing)
            checks["discovery_method"] = "official ollama Python SDK"
        except ImportError:
            checks["python_sdk_importable"] = False
            try:
                listing = _json_request(f"{self.ollama_base_url}/api/tags", timeout=5)
                models = [m.get("name") for m in listing.get("models", []) if m.get("name")]
                checks["discovery_method"] = "Ollama REST /api/tags fallback"
            except Exception as exc:
                detail = (
                    f"Cannot reach Ollama at {self.ollama_base_url}: {_safe_exception(exc)}. "
                    + (f"CLI exists at {cli}. " if cli else "Ollama CLI is not on this kernel's PATH. ")
                    + "If `ollama list` works in Terminal but this fails in Jupyter, compare the notebook kernel, host and OLLAMA_BASE_URL."
                )
                return self._cache_put("ollama", ProviderDiagnostic(
                    "ollama", False, detail, endpoint=self.ollama_base_url,
                    latency_seconds=round(time.perf_counter() - start, 3), checks=checks,
                ))
        except Exception as exc:
            detail = (
                f"Cannot reach Ollama at {self.ollama_base_url}: {_safe_exception(exc)}. "
                + (f"CLI exists at {cli}. " if cli else "Ollama CLI is not on this kernel's PATH. ")
                + "If `ollama list` works in Terminal but this fails in Jupyter, compare the notebook kernel, host and OLLAMA_BASE_URL."
            )
            return self._cache_put("ollama", ProviderDiagnostic(
                "ollama", False, detail, endpoint=self.ollama_base_url,
                latency_seconds=round(time.perf_counter() - start, 3), checks=checks,
            ))

        checks["model_count"] = len(models)
        elapsed = round(time.perf_counter() - start, 3)
        if not models:
            return self._cache_put("ollama", ProviderDiagnostic(
                "ollama", False,
                f"Ollama is reachable at {self.ollama_base_url}, but it returned no installed models. Run `ollama list` and `ollama pull <model>`. ",
                models=[], endpoint=self.ollama_base_url, latency_seconds=elapsed, checks=checks,
            ))
        chosen = self.select_ollama_model(models=models)
        return self._cache_put("ollama", ProviderDiagnostic(
            "ollama", True,
            f"Ollama reachable; discovered {len(models)} local model(s) using {checks['discovery_method']}.",
            model=chosen, models=models, endpoint=self.ollama_base_url,
            latency_seconds=elapsed, checks=checks,
        ))

    def ollama_models(self, refresh: bool = False) -> List[str]:
        return list(self.diagnose_ollama(refresh=refresh).models)

    def select_ollama_model(
        self,
        requested: Optional[str] = None,
        models: Optional[List[str]] = None,
    ) -> Optional[str]:
        models = models if models is not None else self.ollama_models()
        if not models:
            return None
        configured = requested or _env("OLLAMA_MODEL")
        if configured:
            exact = next((m for m in models if m == configured), None)
            if exact:
                return exact
            casefold = next((m for m in models if m.lower() == configured.lower()), None)
            if casefold:
                return casefold
        # Prefer the models the instructor has already installed, then common alternatives.
        preferences = [
            "qwen3.5:0.8b", "llama3.2:latest", "llama3.2:1b", "deepseek-r1:1.5b",
            "qwen3:4b", "qwen3:8b", "gpt-oss:20b",
        ]
        for pref in preferences:
            match = next((m for m in models if m.lower() == pref.lower()), None)
            if match:
                return match
        return models[0]

    # ------------------------------------------------------------------
    # Groq diagnostics and selection
    # ------------------------------------------------------------------
    def _groq_client(self):
        key = _env("GROQ_API_KEY")
        if not key:
            raise ProviderError(
                "GROQ_API_KEY is not set. Put `GROQ_API_KEY=...` in resources/.env or the process environment, then restart the kernel."
            )
        try:
            from groq import Groq
        except ImportError as exc:
            raise ProviderError("The `groq` package is not installed. Run `%pip install -U groq` in this notebook kernel.") from exc
        return Groq(api_key=key)

    @staticmethod
    def _groq_model_ids(response: Any) -> List[str]:
        data = getattr(response, "data", None)
        if data is None and isinstance(response, dict):
            data = response.get("data", [])
        ids: List[str] = []
        for item in data or []:
            mid = getattr(item, "id", None)
            if mid is None and isinstance(item, dict):
                mid = item.get("id")
            if mid:
                ids.append(str(mid))
        return sorted(ids)

    def diagnose_groq(self, refresh: bool = False) -> ProviderDiagnostic:
        cached = self._cache_get("groq", refresh)
        if cached:
            return cached

        preferred = _env("GROQ_MODEL", "qwen/qwen3.6-27b")
        if not _env("GROQ_API_KEY"):
            return self._cache_put("groq", ProviderDiagnostic(
                "groq", False,
                "GROQ_API_KEY is not set. Create a Groq API key, store it as GROQ_API_KEY, restart the kernel, and rerun Day 0.",
                model=preferred, endpoint="https://api.groq.com/openai/v1",
                checks={"api_key_present": False},
            ))

        start = time.perf_counter()
        checks: Dict[str, Any] = {"api_key_present": True, "sdk": "groq"}
        try:
            client = self._groq_client()
            listing = client.models.list()
            models = self._groq_model_ids(listing)
            checks["model_list_ok"] = True
            checks["model_count"] = len(models)
            chosen = self.select_groq_model(models=models)
            if preferred in models:
                detail = "Groq SDK authentication succeeded and the configured model is active."
            elif chosen:
                detail = f"Groq SDK authentication succeeded; configured model {preferred!r} was not active, so {chosen!r} was selected."
            else:
                detail = "Groq SDK authentication succeeded, but no compatible chat model could be selected."
            return self._cache_put("groq", ProviderDiagnostic(
                "groq", bool(chosen), detail, model=chosen, models=models,
                endpoint="https://api.groq.com/openai/v1",
                latency_seconds=round(time.perf_counter() - start, 3), checks=checks,
            ))
        except Exception as model_exc:
            # A model-list diagnostic must never mark Groq unavailable if normal
            # chat completion works. This specifically prevents false negatives
            # from edge/security behaviour on one endpoint/client path.
            checks["model_list_ok"] = False
            checks["model_list_error"] = _safe_exception(model_exc)
            try:
                client = self._groq_client()
                kwargs: Dict[str, Any] = {
                    "model": preferred,
                    "messages": [{"role": "user", "content": "Reply with OK."}],
                    "temperature": 0,
                    "max_completion_tokens": 8,
                }
                if preferred == "qwen/qwen3.6-27b":
                    kwargs.update({"reasoning_effort": "none", "reasoning_format": "hidden"})
                completion = client.chat.completions.create(**kwargs)
                checks["chat_smoke_test_ok"] = True
                detail = (
                    "Groq chat completion works through the official SDK, but model-list discovery failed. "
                    f"The provider is therefore usable. Model-list error: {_safe_exception(model_exc)}"
                )
                return self._cache_put("groq", ProviderDiagnostic(
                    "groq", True, detail, model=preferred, models=[],
                    endpoint="https://api.groq.com/openai/v1",
                    latency_seconds=round(time.perf_counter() - start, 3), checks=checks,
                ))
            except Exception as chat_exc:
                checks["chat_smoke_test_ok"] = False
                checks["chat_smoke_test_error"] = _safe_exception(chat_exc)
                detail = (
                    "Groq diagnostics failed through the official SDK. "
                    f"Model-list error: {_safe_exception(model_exc)}; chat smoke-test error: {_safe_exception(chat_exc)}"
                )
                return self._cache_put("groq", ProviderDiagnostic(
                    "groq", False, detail, model=preferred,
                    endpoint="https://api.groq.com/openai/v1",
                    latency_seconds=round(time.perf_counter() - start, 3), checks=checks,
                ))

    def groq_models(self, refresh: bool = False) -> List[str]:
        return list(self.diagnose_groq(refresh=refresh).models)

    def select_groq_model(
        self,
        requested: Optional[str] = None,
        models: Optional[List[str]] = None,
    ) -> Optional[str]:
        configured = requested or _env("GROQ_MODEL", "qwen/qwen3.6-27b")
        if models is None:
            # Avoid a network round-trip before every chat request. The explicitly
            # configured model can be attempted directly; Day 0 validates it.
            return configured
        if not models:
            return configured
        candidates = [
            configured,
            "qwen/qwen3.6-27b",
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
        ]
        for candidate in candidates:
            if candidate and candidate in models:
                return candidate
        return models[0] if models else configured

    # ------------------------------------------------------------------
    # Unified diagnostics/routing
    # ------------------------------------------------------------------
    def diagnose(self, refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        ollama = self.diagnose_ollama(refresh=refresh)
        groq = self.diagnose_groq(refresh=refresh)
        return {
            "groq": groq.as_dict(),
            "ollama": ollama.as_dict(),
            "openai": {
                "provider": "openai", "available": bool(_env("OPENAI_API_KEY")),
                "model": _env("OPENAI_MODEL", "gpt-5.6-luna"), "detail": "Key presence check only; Day 0 reports this as optional.",
            },
            "gemini": {
                "provider": "gemini", "available": bool(_env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY")),
                "model": _env("GEMINI_MODEL", "gemini-3.7-flash"), "detail": "Key presence check only; Day 0 reports this as optional.",
            },
            "anthropic": {
                "provider": "anthropic", "available": bool(_env("ANTHROPIC_API_KEY")),
                "model": _env("ANTHROPIC_MODEL", "claude-sonnet-5"), "detail": "Key presence check only; Day 0 reports this as optional.",
            },
            "deepseek": {
                "provider": "deepseek", "available": bool(_env("DEEPSEEK_API_KEY")),
                "model": _env("DEEPSEEK_MODEL", "deepseek-v4-flash"), "detail": "Key presence check only; Day 0 reports this as optional.",
            },
            "mimic": {
                "provider": "mimic", "available": True,
                "model": "deterministic-classroom-mimic", "detail": "Offline deterministic teaching fallback.",
            },
        }

    def provider_status(self, refresh: bool = False) -> Dict[str, Any]:
        return self.diagnose(refresh=refresh)

    def choose_provider(self, refresh: bool = False) -> str:
        status = self.diagnose(refresh=refresh)
        for provider in self.preferred_order:
            if status.get(provider, {}).get("available"):
                return provider
        return "mimic"

    # ------------------------------------------------------------------
    # Chat interface
    # ------------------------------------------------------------------
    def chat(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 600,
        fallback_on_error: Optional[bool] = None,
    ) -> LLMResponse:
        if fallback_on_error is None:
            fallback_on_error = provider is None

        if provider is not None:
            try:
                return self._chat_one(provider, messages, model, temperature, max_tokens)
            except Exception as exc:
                if not fallback_on_error:
                    if isinstance(exc, ProviderError):
                        raise
                    raise ProviderError(f"{provider} failed: {_safe_exception(exc)}") from exc
                if self.verbose:
                    print(f"⚠ {provider} failed: {_safe_exception(exc)}. Falling back to mimic.", flush=True)
                return self._chat_mimic(messages, note=f"{provider} failed: {_safe_exception(exc)}")

        errors: List[str] = []
        status = self.diagnose()
        for candidate in self.preferred_order:
            if candidate == "mimic":
                break
            if not status.get(candidate, {}).get("available"):
                continue
            try:
                return self._chat_one(candidate, messages, model, temperature, max_tokens)
            except Exception as exc:
                errors.append(f"{candidate}: {_safe_exception(exc)}")
                if self.verbose:
                    print(f"⚠ Automatic route {candidate} failed: {_safe_exception(exc)}", flush=True)
        note = "; ".join(errors) if errors else "No live provider was available."
        return self._chat_mimic(messages, note=note)

    def _chat_one(
        self,
        provider: str,
        messages: List[Dict[str, str]],
        model: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        provider = provider.lower().strip()
        if provider == "groq":
            return self._chat_groq(messages, model, temperature, max_tokens)
        if provider == "ollama":
            return self._chat_ollama(messages, model, temperature, max_tokens)
        if provider == "openai":
            return self._chat_openai(
                messages, model or _env("OPENAI_MODEL", "gpt-5.6-luna"), max_tokens
            )
        if provider == "deepseek":
            return self._chat_openai_compatible(
                messages, "DEEPSEEK_API_KEY", "https://api.deepseek.com",
                model or _env("DEEPSEEK_MODEL", "deepseek-v4-flash"), "deepseek", temperature, max_tokens,
            )
        if provider == "gemini":
            return self._chat_gemini(messages, model or _env("GEMINI_MODEL", "gemini-3.7-flash"), temperature)
        if provider == "anthropic":
            return self._chat_anthropic(messages, model or _env("ANTHROPIC_MODEL", "claude-sonnet-5"), temperature, max_tokens)
        if provider == "mimic":
            return self._chat_mimic(messages)
        raise ProviderError(f"Unknown provider: {provider}")

    def _chat_groq(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        chosen = self.select_groq_model(requested=model)
        if not chosen:
            raise ProviderError("No Groq model is configured. Set GROQ_MODEL or run Day 0 model discovery.")
        client = self._groq_client()
        if self.verbose:
            print(f"▶ Groq request started | model={chosen} | max_tokens={max_tokens}", flush=True)
        start = time.perf_counter()
        kwargs: Dict[str, Any] = {
            "model": chosen,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        reasoning_mode = None
        if chosen == "qwen/qwen3.6-27b":
            # Qwen 3.6 supports explicit reasoning controls on Groq. For the
            # classroom default we disable extended reasoning so ordinary tool
            # exercises stay fast; instructors can set GROQ_REASONING_EFFORT=default
            # when they deliberately want a reasoning demonstration.
            reasoning_format = _env("GROQ_REASONING_FORMAT", "hidden")
            reasoning_effort = _env("GROQ_REASONING_EFFORT", "none")
            kwargs["reasoning_format"] = reasoning_format
            kwargs["reasoning_effort"] = reasoning_effort
            reasoning_mode = f"format={reasoning_format}; effort={reasoning_effort}"
        try:
            completion = client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise ProviderError(f"Groq SDK request failed: {_safe_exception(exc)}") from exc
        elapsed = round(time.perf_counter() - start, 3)
        message = completion.choices[0].message
        text = getattr(message, "content", None) or ""
        usage = getattr(completion, "usage", None)
        response = LLMResponse(
            text=text,
            provider="groq",
            model=chosen,
            latency_seconds=elapsed,
            raw=completion,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            finish_reason=getattr(completion.choices[0], "finish_reason", None),
            request_id=getattr(completion, "id", None),
            endpoint="https://api.groq.com/openai/v1/chat/completions",
            reasoning_mode=reasoning_mode,
        )
        if self.verbose:
            print(f"✓ Groq completed in {elapsed:.3f}s | model={chosen}", flush=True)
        return response

    def _chat_ollama(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        diag = self.diagnose_ollama()
        if not diag.available:
            raise ProviderError(diag.detail)
        chosen = self.select_ollama_model(requested=model, models=diag.models)
        if not chosen:
            raise ProviderError("Ollama is reachable but no model could be selected.")

        think = _env_bool("OLLAMA_THINK", False)
        keep_alive = _env("OLLAMA_KEEP_ALIVE", "15m")
        if self.verbose:
            print(
                f"▶ Ollama request started | model={chosen} | think={think} | max_tokens={max_tokens} | host={self.ollama_base_url}",
                flush=True,
            )
        start = time.perf_counter()
        try:
            from ollama import Client
            client = Client(host=self.ollama_base_url)
            kwargs: Dict[str, Any] = {
                "model": chosen,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
                "think": think,
                "keep_alive": keep_alive,
            }
            try:
                result = client.chat(**kwargs)
            except TypeError:
                # Older ollama-python clients may not yet expose think/keep_alive.
                kwargs.pop("think", None)
                kwargs.pop("keep_alive", None)
                result = client.chat(**kwargs)
        except ImportError as exc:
            raise ProviderError("The `ollama` Python package is not installed. Run `%pip install -U ollama`.") from exc
        except Exception as exc:
            raise ProviderError(f"Ollama request failed for {chosen!r}: {_safe_exception(exc)}") from exc

        msg = getattr(result, "message", None)
        text = getattr(msg, "content", None) if msg is not None else None
        if text is None and isinstance(result, dict):
            text = (result.get("message") or {}).get("content", "")
        text = text or ""
        elapsed = round(time.perf_counter() - start, 3)

        if not text.strip():
            raise ProviderError(
                f"Ollama model {chosen!r} returned an empty final answer. "
                "The bootcamp disables thinking by default; update Ollama/ollama-python or try OLLAMA_MODEL=llama3.2:latest if this persists."
            )

        def _field(name: str) -> Any:
            value = getattr(result, name, None)
            if value is None and isinstance(result, dict):
                value = result.get(name)
            return value

        input_tokens = _field("prompt_eval_count")
        output_tokens = _field("eval_count")
        total = (input_tokens + output_tokens) if isinstance(input_tokens, int) and isinstance(output_tokens, int) else None
        response = LLMResponse(
            text=text,
            provider="ollama",
            model=chosen,
            latency_seconds=elapsed,
            raw=result,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total,
            finish_reason=_field("done_reason"),
            endpoint=f"{self.ollama_base_url}/api/chat",
            reasoning_mode="thinking enabled" if think else "thinking disabled",
        )
        if self.verbose:
            print(f"✓ Ollama completed in {elapsed:.3f}s | model={chosen}", flush=True)
        return response

    def _chat_openai(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str],
        max_tokens: int,
    ) -> LLMResponse:
        """Call OpenAI through the official SDK and Responses API.

        This optional route is deliberately separate from Groq's OpenAI-compatible
        endpoint. It keeps the classroom provider abstraction aligned with the
        current OpenAI SDK rather than assuming Chat Completions semantics.
        """
        key = _env("OPENAI_API_KEY")
        if not key:
            raise ProviderError("Missing OPENAI_API_KEY")
        chosen = model or _env("OPENAI_MODEL", "gpt-5.6-luna")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderError("Install `openai`: `%pip install -U openai`") from exc
        start = time.perf_counter()
        try:
            client = OpenAI(api_key=key)
            response = client.responses.create(
                model=chosen,
                input=messages,
                max_output_tokens=max_tokens,
            )
        except Exception as exc:
            raise ProviderError(f"OpenAI SDK request failed: {_safe_exception(exc)}") from exc
        usage = getattr(response, "usage", None)
        return LLMResponse(
            text=getattr(response, "output_text", "") or "",
            provider="openai",
            model=chosen,
            latency_seconds=round(time.perf_counter() - start, 3),
            raw=response,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            request_id=getattr(response, "id", None),
            endpoint="https://api.openai.com/v1/responses",
        )

    def _chat_openai_compatible(
        self,
        messages: List[Dict[str, str]],
        key_env: str,
        base_url: str,
        model: str,
        provider: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        key = _env(key_env)
        if not key:
            raise ProviderError(f"Missing {key_env}")
        start = time.perf_counter()
        try:
            raw = _json_request(
                f"{base_url.rstrip('/')}/chat/completions",
                method="POST",
                headers={"Authorization": f"Bearer {key}"},
                payload={"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
                timeout=120,
            )
        except Exception as exc:
            raise ProviderError(f"{provider} request failed: {_safe_exception(exc)}") from exc
        text = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = raw.get("usage", {})
        return LLMResponse(
            text, provider, model, round(time.perf_counter() - start, 3), raw,
            input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            finish_reason=raw.get("choices", [{}])[0].get("finish_reason"),
            request_id=raw.get("id"), endpoint=f"{base_url.rstrip('/')}/chat/completions",
        )

    def _chat_gemini(self, messages: List[Dict[str, str]], model: str, temperature: float) -> LLMResponse:
        key = _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY")
        if not key:
            raise ProviderError("Missing GEMINI_API_KEY or GOOGLE_API_KEY")
        prompt = "\n".join(f"{m.get('role')}: {m.get('content', '')}" for m in messages)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        start = time.perf_counter()
        try:
            raw = _json_request(
                url, method="POST",
                payload={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": temperature},
                }, timeout=120,
            )
        except Exception as exc:
            raise ProviderError(f"Gemini request failed: {_safe_exception(exc)}") from exc
        text = raw.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        return LLMResponse(text, "gemini", model, round(time.perf_counter() - start, 3), raw, endpoint=url.split("?", 1)[0])

    def _chat_anthropic(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        key = _env("ANTHROPIC_API_KEY")
        if not key:
            raise ProviderError("Missing ANTHROPIC_API_KEY")
        if not model:
            raise ProviderError("ANTHROPIC_MODEL is not set. Use a current model ID from the Anthropic console.")
        system = "You are a careful bootcamp teaching assistant."
        anth_messages = []
        for m in messages:
            if m.get("role") == "system":
                system = m.get("content", system)
            else:
                anth_messages.append({
                    "role": "assistant" if m.get("role") == "assistant" else "user",
                    "content": m.get("content", ""),
                })
        start = time.perf_counter()
        try:
            raw = _json_request(
                "https://api.anthropic.com/v1/messages", method="POST",
                headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                payload={
                    "model": model, "max_tokens": max_tokens, "temperature": temperature,
                    "system": system, "messages": anth_messages,
                }, timeout=120,
            )
        except Exception as exc:
            raise ProviderError(f"Anthropic request failed: {_safe_exception(exc)}") from exc
        text = "".join(b.get("text", "") for b in raw.get("content", []) if b.get("type") == "text")
        return LLMResponse(text, "anthropic", model, round(time.perf_counter() - start, 3), raw, endpoint="https://api.anthropic.com/v1/messages")

    def _chat_mimic(self, messages: List[Dict[str, str]], note: str = "") -> LLMResponse:
        user_text = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
        lower = user_text.lower()
        if "pick rate" in lower or "calculate" in lower:
            answer = "I would call the calculator or pick-rate tool first, then interpret the result with safety, quality, task mix and human-review context."
        elif "amazon" in lower or "fulfilment" in lower:
            answer = "A bounded fulfilment-centre agent can retrieve approved process guidance, summarise exceptions, calculate operational metrics and pause sensitive actions for human review."
        elif "human" in lower or "approval" in lower:
            answer = "Pause at a human-in-the-loop checkpoint and present the proposed action, evidence, risk, and approve/edit/reject options."
        elif "rag" in lower or "retrieve" in lower:
            answer = "Retrieve first, cite the retrieved evidence, answer only from grounded content, and state when the knowledge base is insufficient."
        else:
            answer = "Agentic AI combines state, tools, model decisions, observations and controlled actions. Use deterministic workflow steps where reliability matters and LLM decisions where flexibility adds value."
        if note:
            answer = f"[MIMIC FALLBACK — {note}]\n\n{answer}"
        return LLMResponse(
            answer, "mimic", "deterministic-classroom-mimic", 0.0,
            endpoint="offline", reasoning_mode="none",
        )

    # ------------------------------------------------------------------
    # LangChain/LangGraph provider models
    # ------------------------------------------------------------------
    def langchain_model(
        self,
        provider: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 700,
    ):
        provider = provider.lower().strip()
        if provider == "groq":
            key = _env("GROQ_API_KEY")
            if not key:
                raise ProviderError("Missing GROQ_API_KEY")
            chosen = self.select_groq_model(requested=model)
            try:
                from langchain_groq import ChatGroq
            except ImportError as exc:
                raise ProviderError("Install `langchain-groq`: `%pip install -U langchain-groq`") from exc
            kwargs: Dict[str, Any] = {
                "model": chosen,
                "temperature": temperature,
                "api_key": key,
                "max_tokens": max_tokens,
                "timeout": 90,
                "max_retries": 2,
            }
            if chosen == "qwen/qwen3.6-27b":
                kwargs["reasoning_format"] = _env("GROQ_REASONING_FORMAT", "hidden")
            return ChatGroq(**kwargs)

        if provider == "ollama":
            diag = self.diagnose_ollama()
            if not diag.available:
                raise ProviderError(diag.detail)
            chosen = self.select_ollama_model(requested=model, models=diag.models)
            try:
                from langchain_ollama import ChatOllama
            except ImportError as exc:
                raise ProviderError("Install `langchain-ollama`: `%pip install -U langchain-ollama`") from exc
            return ChatOllama(
                model=chosen,
                base_url=self.ollama_base_url,
                temperature=temperature,
                num_predict=max_tokens,
                keep_alive=_env("OLLAMA_KEEP_ALIVE", "15m"),
            )
        raise ProviderError("This bootcamp LangGraph model factory demonstrates Groq and Ollama explicitly.")


def make_messages(
    user_prompt: str,
    system: str = "You are a precise, safety-aware teaching assistant for an Agentic AI bootcamp.",
) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]
