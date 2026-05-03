"""Main entry point for running some experiment.

For now:

- Needs a config.toml
- Will store things in ./tmp

"""

import sys
from collections.abc import Callable

from ctm_ai_eval.qa.eval_runs import qa_compute_metrics
from ctm_ai_eval.qa.qa_experiment import qa_trace
from ctm_ai_eval.rag.haystack_experiment import (
    haystack_chunk_size,
    haystack_chunkers,
    haystack_retrievers,
)
from ctm_ai_eval.utils.rich_print import CONS

EXPERIMENTS: dict[str, Callable[[str | None], None]] = {
    # rag experiments: each can vary one or more parameters
    "retrievers": haystack_retrievers,
    "chunkers": haystack_chunkers,
    "chunksize": haystack_chunk_size,
    # qa: First run "trace" to store LLM outputs, then compute metrics.
    "qa_trace": qa_trace,
    "qa_metrics": qa_compute_metrics,
}


def _main():
    if len(sys.argv) < 2 or sys.argv[1] not in EXPERIMENTS:
        print(f"please specify an experiment: {list(EXPERIMENTS.keys())}")
        sys.exit(1)

    key = sys.argv[1]
    CONS.print(f"Running experiment: {key}", style="bold black on white", justify="center")

    if len(sys.argv) == 3:
        cfg_path = sys.argv[2]
        print(f"loading cfg from {cfg_path}")
    else:
        cfg_path = None
    EXPERIMENTS[key](cfg_path)


if __name__ == "__main__":
    _main()
