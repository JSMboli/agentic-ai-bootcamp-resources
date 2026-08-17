"""Notebook presentation helpers: path discovery, progress and metadata panels."""
from __future__ import annotations

import html
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import pandas as pd


def locate_resources(start: Optional[Path] = None) -> Path:
    """Find the bootcamp resources directory from common launch locations."""
    cwd = (start or Path.cwd()).resolve()
    for p in [cwd, *cwd.parents]:
        if (p / "resources" / "src").exists():
            return p / "resources"
        if (p / "src").exists() and (p / "notebooks").exists() and (p / "data").exists():
            return p
    raise RuntimeError(
        "Could not locate the bootcamp resources directory. Extract the complete ZIP and open the notebook from inside that package."
    )


def add_src_to_path(resource_dir: Path) -> Path:
    src = resource_dir / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src


def environment_table(resource_dir: Path, router: Any) -> pd.DataFrame:
    return pd.DataFrame([
        {"item": "Python", "value": sys.version.split()[0]},
        {"item": "Kernel executable", "value": sys.executable},
        {"item": "Working directory", "value": str(Path.cwd().resolve())},
        {"item": "Resources directory", "value": str(resource_dir)},
        {"item": "Data directory", "value": str(resource_dir / "data")},
        {"item": "Environment files found", "value": ", ".join(router.loaded_env_candidates) or "none"},
    ])


def provider_table(status: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for name, info in status.items():
        rows.append({
            "provider": name,
            "available": bool(info.get("available")),
            "selected_model": info.get("model"),
            "endpoint": info.get("endpoint", ""),
            "diagnostic_s": info.get("latency_seconds"),
            "detail": info.get("detail", ""),
        })
    return pd.DataFrame(rows)


def response_table(response: Any) -> pd.DataFrame:
    summary = response.summary() if hasattr(response, "summary") else {
        "provider": getattr(response, "provider", None),
        "model": getattr(response, "model", None),
        "latency_s": getattr(response, "latency_seconds", None),
    }
    return pd.DataFrame([summary])


def _display_html(markup: str, display_id: Optional[str] = None):
    try:
        from IPython.display import HTML, display
        return display(HTML(markup), display_id=display_id)
    except Exception:
        print(html.unescape(markup.replace("<br>", "\n")))
        return None


@contextmanager
def progress_indicator(label: str):
    """Show a browser-animated spinner while a blocking call runs.

    CSS animation continues in the browser even while the Python kernel is busy,
    so students have a visible indication that a local model or API call is still
    running. The final display reports elapsed time.
    """
    safe = html.escape(label)
    start = time.perf_counter()
    display_id = f"bootcamp-progress-{time.time_ns()}"
    markup = f"""
    <div style='font-family:system-ui;padding:9px 12px;border:1px solid #d0d7de;border-radius:8px;display:inline-flex;gap:10px;align-items:center'>
      <span style='width:14px;height:14px;border:3px solid #ddd;border-top-color:#555;border-radius:50%;display:inline-block;animation:bootspin .8s linear infinite'></span>
      <strong>{safe}</strong>&nbsp;— running…
    </div>
    <style>@keyframes bootspin {{ to {{ transform: rotate(360deg); }} }}</style>
    """
    handle = _display_html(markup, display_id=display_id)
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - start
        failed = f"<div style='padding:9px 12px;border:1px solid #d1242f;border-radius:8px'>✗ <strong>{safe}</strong> failed after {elapsed:.2f}s</div>"
        if handle is not None and hasattr(handle, "update"):
            try:
                from IPython.display import HTML
                handle.update(HTML(failed))
            except Exception:
                pass
        raise
    else:
        elapsed = time.perf_counter() - start
        done = f"<div style='padding:9px 12px;border:1px solid #1a7f37;border-radius:8px'>✓ <strong>{safe}</strong> completed in {elapsed:.2f}s</div>"
        if handle is not None and hasattr(handle, "update"):
            try:
                from IPython.display import HTML
                handle.update(HTML(done))
            except Exception:
                pass


def run_with_progress(label: str, func: Callable[..., Any], *args, **kwargs) -> Any:
    with progress_indicator(label):
        return func(*args, **kwargs)


def display_response(response: Any, heading: str = "Model response") -> None:
    """Render metadata first, then the model's final answer."""
    try:
        from IPython.display import Markdown, display
        display(Markdown(f"### {heading}"))
        display(response_table(response))
        display(Markdown(getattr(response, "text", "") or "*(The model returned no final text.)*"))
    except Exception:
        print(response_table(response).to_string(index=False))
        print(getattr(response, "text", ""))


def display_note(text: str, kind: str = "info") -> None:
    palette = {"info": "#0969da", "success": "#1a7f37", "warning": "#9a6700", "error": "#d1242f"}
    border = palette.get(kind, palette["info"])
    _display_html(
        f"<div style='border-left:4px solid {border};padding:8px 12px;margin:6px 0;background:#f6f8fa'>{html.escape(text)}</div>"
    )
