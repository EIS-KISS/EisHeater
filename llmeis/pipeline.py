from __future__ import annotations

import json
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic_ai import BinaryImage
from pydantic_ai.messages import (
    FinalResultEvent,
    FunctionToolResultEvent,
    ModelRequest,
    ModelResponse,
    PartEndEvent,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from .agent import SYSTEM_PROMPT
from .eisgen import eisgenerator
from .modelstr import validate_model_str
from .plotting import plot_bode, plot_nyquist
from .spectra import (
    Spectrum,
    format_data,
    format_data_polar,
    iter_spectra,
    parse_eisf,
    resample_log,
    write_csv,
)
from .tools import SpectrumDeps, parse_eisfit_output, run_eisfit

_STDOUT_LOCK = threading.Lock()


def _safe(name: str) -> str:
    return name.replace("/", "_")


def build_prompt(spec: Spectrum, res_omega, res_re, res_im, nyquist_png: bytes, bode_png: bytes):
    w0, w1 = spec.omega_range
    text = (
        "Identify the equivalent circuit model for this EIS spectrum.\n\n"
        f"Full measured data ({spec.n_points} points, omega {w0:.4e} to {w1:.4e} rad/s):\n"
        f"{format_data(spec.omega, spec.re, spec.im)}\n\n"
        f"Resampled data ({res_omega.size} log-spaced points, with |Z| and phase):\n"
        f"{format_data_polar(res_omega, res_re, res_im)}\n\n"
        "The two attached images are the Nyquist plot (Re vs -Im) and the Bode plot "
        "(|Z| and phase vs frequency) of the same spectrum."
    )
    return [
        text,
        BinaryImage(data=nyquist_png, media_type="image/png", identifier="nyquist"),
        BinaryImage(data=bode_png, media_type="image/png", identifier="bode"),
    ]


def _serialize_prompt_content(content) -> str | list:
    if isinstance(content, str):
        return content
    items = []
    for item in content:
        if isinstance(item, BinaryImage):
            items.append({"image": item.identifier or item.media_type})
        elif isinstance(item, str):
            items.append(item)
        else:
            items.append(str(item))
    return items


def _as_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n\n".join(_as_text(v) for v in value)
    return str(value)


def serialize_turns(messages) -> list[dict]:
    turns = []
    system_emitted = False
    for msg in messages:
        entry: dict = {
            "role": "request" if isinstance(msg, ModelRequest) else "response",
            "parts": [],
        }
        if isinstance(msg, ModelResponse):
            entry["model_name"] = msg.model_name
            entry["finish_reason"] = msg.finish_reason
        if isinstance(msg, ModelRequest) and not system_emitted:
            instr = getattr(msg, "instructions", None)
            if instr:
                entry["parts"].append({"kind": "system-prompt", "content": _as_text(instr)})
                system_emitted = True
        for part in msg.parts:
            if isinstance(part, UserPromptPart):
                entry["parts"].append({"kind": "user-prompt", "content": _serialize_prompt_content(part.content)})
            elif isinstance(part, SystemPromptPart):
                entry["parts"].append({"kind": "system-prompt", "content": part.content})
            elif isinstance(part, ThinkingPart):
                entry["parts"].append({"kind": "thinking", "content": part.content})
            elif isinstance(part, ToolCallPart):
                entry["parts"].append({"kind": "tool-call", "tool_name": part.tool_name, "args": part.args})
            elif isinstance(part, ToolReturnPart):
                entry["parts"].append({"kind": "tool-return", "tool_name": part.tool_name, "content": part.content})
            elif isinstance(part, TextPart):
                entry["parts"].append({"kind": "text", "content": part.content})
            else:
                entry["parts"].append({"kind": type(part).__name__, "content": str(part)})
        turns.append(entry)
    return turns


class TurnsStream:
    def __init__(self, on_turn=None, initial_request: list[dict] | None = None):
        self._on_turn = on_turn
        self._response_parts: list[dict] = []
        self.turns: list[dict] = []
        self._n = 0
        self._initial_emitted = False
        self._initial_request = initial_request

    def _emit(self, turn: dict):
        self._n += 1
        turn = {"turn": self._n, **turn}
        self.turns.append(turn)
        if self._on_turn is not None:
            self._on_turn(turn)

    def _flush_response(self):
        if self._response_parts:
            self._emit({"role": "response", "parts": self._response_parts})
            self._response_parts = []

    @staticmethod
    def _part_dict(part) -> dict | None:
        if isinstance(part, ThinkingPart):
            return {"kind": "thinking", "content": part.content}
        if isinstance(part, ToolCallPart):
            return {"kind": "tool-call", "tool_name": part.tool_name, "args": part.args}
        if isinstance(part, TextPart):
            return {"kind": "text", "content": part.content}
        return None

    async def handle(self, run_ctx, event_stream):
        if self._initial_request is not None and not self._initial_emitted:
            self._emit({"role": "request", "parts": self._initial_request})
            self._initial_emitted = True
        async for event in event_stream:
            if isinstance(event, PartEndEvent):
                d = self._part_dict(event.part)
                if d is not None:
                    self._response_parts.append(d)
            elif isinstance(event, FunctionToolResultEvent):
                self._flush_response()
                part = event.part
                self._emit(
                    {
                        "role": "request",
                        "parts": [
                            {"kind": "tool-return", "tool_name": part.tool_name, "content": part.content}
                        ],
                    }
                )
            elif isinstance(event, FinalResultEvent):
                self._flush_response()


def _print_indented(text: str, indent: str = "      "):
    for line in str(text).splitlines():
        print(indent + line, flush=True)


def _truncate(text: str, limit: int) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n{ ' ' * 6}...[truncated, {len(text) - limit} more chars]"


def _print_user_prompt(content):
    if isinstance(content, str):
        _print_indented(_truncate(content, 300))
        return
    for item in content:
        if isinstance(item, dict) and "image" in item:
            print(f"      (image: {item['image']})", flush=True)
        else:
            _print_indented(_truncate(item, 300))


def _print_turn(turn: dict):
    with _STDOUT_LOCK:
        n = turn.get("turn", "?")
        role = turn["role"]
        print(flush=True)
        print(f"── turn {n} · {role} " + "─" * 46, flush=True)
        for part in turn["parts"]:
            kind = part["kind"]
            if kind == "system-prompt":
                print("  [system prompt]", flush=True)
                _print_indented(part["content"])
            elif kind == "user-prompt":
                print("  [user prompt]", flush=True)
                _print_user_prompt(part["content"])
            elif kind == "thinking":
                print("  [thinking]", flush=True)
                _print_indented(part["content"])
            elif kind == "tool-call":
                print(f"  [tool call] {part['tool_name']}", flush=True)
                _print_indented(json.dumps(part["args"], indent=2))
            elif kind == "tool-return":
                content = part["content"]
                if isinstance(content, str) and len(content) > 1500:
                    content = content[:1500] + "\n      ...[truncated]"
                print(f"  [tool return] {part['tool_name']}", flush=True)
                _print_indented(content)
            elif kind == "text":
                print("  [text]", flush=True)
                _print_indented(part["content"])
            else:
                print(f"  [{kind}]", flush=True)
                _print_indented(json.dumps(part, default=str))


def _run_once(agent, user_prompt, deps, cfg, message_history, event_stream_handler):
    if event_stream_handler is not None:
        with agent.run_stream_sync(
            user_prompt,
            deps=deps,
            message_history=message_history,
            model_settings=cfg.model_settings,
            event_stream_handler=event_stream_handler,
        ) as stream:
            return stream.get_output(), stream.all_messages()
    result = agent.run_sync(
        user_prompt,
        deps=deps,
        message_history=message_history,
        model_settings=cfg.model_settings,
    )
    return result.output, result.all_messages()


def run_agent_with_validation(agent, prompt, deps, cfg, event_stream_handler=None):
    message_history = None
    user_prompt = prompt
    last = None
    for _ in range(cfg.max_retries + 1):
        out, messages = _run_once(agent, user_prompt, deps, cfg, message_history, event_stream_handler)
        ok, err = validate_model_str(out.model_str, eisgenerator)
        if ok:
            return out, messages
        last = (out, messages)
        message_history = messages
        user_prompt = [
            f"Your previous final answer was rejected because: {err}\n"
            "Produce a corrected final answer now."
        ]
    return last


def process_spectrum(spec: Spectrum, agent, outdir: Path, cfg) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    safe = _safe(spec.name)

    res_omega, res_re, res_im = resample_log(spec, cfg.resample_points)
    nyquist_png = plot_nyquist(spec.omega, spec.re, spec.im)
    bode_png = plot_bode(spec.omega, spec.re, spec.im)
    (outdir / f"{safe}_nyquist.png").write_bytes(nyquist_png)
    (outdir / f"{safe}_bode.png").write_bytes(bode_png)

    result: dict = {
        "name": spec.name,
        "header_model": spec.header_model,
        "model_str": None,
        "confidence": None,
        "type": None,
        "fit": None,
        "error": None,
    }

    with tempfile.TemporaryDirectory(prefix="llmeis_") as tmpdir:
        csv_path = os.path.join(tmpdir, "spectrum.csv")
        write_csv(spec, csv_path)
        deps = SpectrumDeps(
            csv_path=csv_path,
            omega=spec.omega,
            re=spec.re,
            im=spec.im,
            res_omega=res_omega,
            res_re=res_re,
            res_im=res_im,
            n_original=spec.n_points,
            omega_range=spec.omega_range,
        )
        prompt = build_prompt(spec, res_omega, res_re, res_im, nyquist_png, bode_png)
        if cfg.turns_stdout:
            initial_request = [
                {"kind": "system-prompt", "content": SYSTEM_PROMPT},
                {"kind": "user-prompt", "content": _serialize_prompt_content(prompt)},
            ]
            turns_stream = TurnsStream(on_turn=_print_turn, initial_request=initial_request)
        else:
            turns_stream = None
        try:
            out, messages = run_agent_with_validation(
                agent,
                prompt,
                deps,
                cfg,
                event_stream_handler=turns_stream.handle if turns_stream else None,
            )
        except Exception as e:
            result["error"] = f"agent run failed: {e}"
            (outdir / f"{safe}.json").write_text(json.dumps(result, indent=2))
            return result

        result["model_str"] = out.model_str
        result["confidence"] = out.confidence
        result["type"] = out.type

        if cfg.dump_turns:
            (outdir / f"{safe}.turns.json").write_text(json.dumps(serialize_turns(messages), indent=2))

        fit_report = run_eisfit(csv_path, out.model_str)
        fit = parse_eisfit_output(fit_report)
        result["fit"] = {
            "parameters": fit["parameters"],
            "fitted_model": fit["fitted_model"],
            "fit_quality": fit["fit_quality"],
            "verdict": fit["verdict"],
        }

    (outdir / f"{safe}.json").write_text(json.dumps(result, indent=2))
    return result


def process_dataset(paths: list[str], outdir: Path, agent, cfg) -> list[dict]:
    outdir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[int, Spectrum]] = []
    for i, (name, text) in enumerate(iter_spectra(paths)):
        if cfg.limit > 0 and i >= cfg.limit:
            break
        spec = parse_eisf(name, text)
        if spec is None:
            print(f"  skipping {name}: no usable data block")
            continue
        jobs.append((i, spec))

    results: list[dict] = [None] * len(jobs)

    def _run(job_pos: int, index: int, spec: Spectrum) -> None:
        with _STDOUT_LOCK:
            print(f"[{index + 1}] {spec.name}")
        res = process_spectrum(spec, agent, outdir, cfg)
        results[job_pos] = res
        status = res["error"] or (res["fit"]["verdict"] if res["fit"] else "")
        with _STDOUT_LOCK:
            print(f"    -> {res['model_str']}  (confidence {res['confidence']})")
            if status:
                print(f"       {status[:100]}")

            _run(job_pos, index, spec)
    else:
        with ThreadPoolExecutor(max_workers=cfg.concurrency) as ex:
            futures = [
                ex.submit(_run, job_pos, index, spec)
                for job_pos, (index, spec) in enumerate(jobs)
            ]
            for f in futures:
                f.result()  # propagate any unexpected exception

    results = [r for r in results if r is not None]

    summary = {
        "datasets": [os.path.basename(p) for p in paths],
        "count": len(results),
        "results": [
            {
                "name": r["name"],
                "model_str": r["model_str"],
                "confidence": r["confidence"],
                "type": r["type"],
                "error": r["error"],
            }
            for r in results
        ],
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    return results
