"""Command-line configuration for the llmeis pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class Config:
    paths: list[str]  # dataset paths: one or more .tar archives and/or .csv spectrum files
    outdir: str
    endpoint: str  # OpenAI-compatible base URL(s), comma-separated for multiple backends
    model: str  # model name to request
    api_key: str | None = None
    resample_points: int = 20
    max_retries: int = 2
    limit: int = 0  # 0 = process every spectrum
    max_tokens: int = 65536
    temperature: float = 0.0
    request_timeout: float = 3600.0  # per-request read timeout; slow reasoning models need headroom
    dump_turns: bool = False  # write a per-spectrum <name>.turns.json with the agent's turns
    turns_stdout: bool = False  # stream the agent's turns to stdout as the run progresses
    concurrency: int = 1  # spectra processed in parallel (1 = sequential)

    @property
    def model_settings(self) -> dict:
        return {"max_tokens": self.max_tokens, "temperature": self.temperature}

    @property
    def endpoints(self) -> list[str]:
        """The backend base URLs, one per comma-separated entry in `endpoint`."""
        return [e.strip() for e in self.endpoint.split(",") if e.strip()]


def parse_args(argv=None) -> Config:
    ap = argparse.ArgumentParser(
        description="Identify the equivalent circuit model of each spectrum in one or more datasets "
        "(tar archives and/or individual EISF csv files)."
    )
    ap.add_argument(
        "paths",
        nargs="+",
        help="dataset paths: one or more .tar archives and/or .csv spectrum files",
    )
    ap.add_argument("-o", "--outdir", default="results", help="output directory (default: results)")
    ap.add_argument(
        "--endpoint",
        default="http://localhost:8080/v1",
        help="OpenAI-compatible base URL(s) of the inference server(s), comma-separated for "
             "multiple backends (default: localhost:8080/v1)",
    )
    ap.add_argument("--model", required=True, help="model name to request from the server")
    ap.add_argument("--api-key", default=None, help="API key (ignored by llama-server)")
    ap.add_argument("--resample-points", type=int, default=20, help="log-spaced resample points (default: 20)")
    ap.add_argument("--max-retries", type=int, default=2, help="retries if the final circuit string is invalid")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N spectra (0 = all)")
    ap.add_argument("--max-tokens", type=int, default=65536, help="max completion tokens per turn")
    ap.add_argument("--temperature", type=float, default=0.0, help="sampling temperature")
    ap.add_argument(
        "--request-timeout",
        type=float,
        default=3600.0,
        help="per-request read timeout in seconds (default: 3600; slow reasoning models need headroom)",
    )
    ap.add_argument(
        "--turns",
        action="store_true",
        help="write a per-spectrum <name>.turns.json with the agent's full turn history (reasoning, tool calls, returns)",
    )
    ap.add_argument(
        "--turns-stdout",
        action="store_true",
        help="stream the agent's turns (reasoning, tool calls, returns) to stdout as the run progresses",
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="number of spectra to process in parallel per backend (default: 1 = sequential)",
    )
    args = ap.parse_args(argv)
    return Config(
        paths=args.paths,
        outdir=args.outdir,
        endpoint=args.endpoint,
        model=args.model,
        api_key=args.api_key,
        resample_points=args.resample_points,
        max_retries=args.max_retries,
        limit=args.limit,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        request_timeout=args.request_timeout,
        dump_turns=args.turns,
        turns_stdout=args.turns_stdout,
        concurrency=args.concurrency,
    )
