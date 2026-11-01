"""Entry point: python -m llmeis <tar> --model <name> [--endpoint <url>] ..."""

from __future__ import annotations

import sys
from pathlib import Path

from .agent import make_agent, make_model
from .config import parse_args
from .pipeline import process_dataset


def main(argv=None) -> int:
    cfg = parse_args(argv)
    missing = [p for p in cfg.paths if not Path(p).is_file()]
    if missing:
        for p in missing:
            print(f"no such file: {p}", file=sys.stderr)
        return 1

    agents = [
        make_agent(make_model(ep, cfg.model, api_key=cfg.api_key, timeout=cfg.request_timeout))
        for ep in cfg.endpoints
    ]
    n_parallel = len(agents) * max(1, cfg.concurrency)
    print(f"endpoints: {cfg.endpoint}")
    print(f"model:     {cfg.model}")
    print(f"backends:  {len(agents)} x {cfg.concurrency} concurrent = {n_parallel} parallel")
    print(f"datasets:  {', '.join(cfg.paths)}")
    print(f"output:    {cfg.outdir}")
    print()

    results = process_dataset(cfg.paths, Path(cfg.outdir), agents, cfg)
    ok = sum(1 for r in results if not r["error"])
    print(f"\ndone: {ok}/{len(results)} spectra processed, results in {cfg.outdir}")
    return 0 if ok == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
