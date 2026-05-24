# Copyright 2025 Meetween / SpeechLMM KD extension.
"""CLI entry for KD training (use instead of ``llamafactory-cli train`` for ``stage: kd``)."""

from __future__ import annotations

from .bootstrap import integrate


def main() -> None:
    integrate()
    from ...train.tuner import run_exp

    run_exp()


if __name__ == "__main__":
    main()
