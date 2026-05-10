"""Evaluate each run."""

from pathlib import Path

from ctm_ai_eval.qa import judges
from ctm_ai_eval.qa.config import load_qa_config
from ctm_ai_eval.qa.datamodels import EvalCase, EvalTrace, FloatTraceMetric, QaQuestion
from ctm_ai_eval.utils.io_util import append_ndjson, load_list_json_generic, load_ndjson_generic

judge_sys_prompt = Path("./assets/prompts/judge_qa_sys.txt").read_text()
judge_msg_template = Path("./assets/prompts/judge_qa_msg.txt").read_text()


JUDGES: list[judges.Judge] = [
    judges.IsConcise(),
    judges.HumanRatingJudge(subsample=4),
    judges.LLMJudge("rnj-1:8b", judge_sys_prompt, judge_msg_template),
]


def qa_compute_metrics(cfg_path: str | None) -> None:
    """Compute metrics for each run."""

    cfg = load_qa_config(cfg_path)
    # load runs and dataset
    dset_name = cfg.dataset_path.stem
    traces_file = Path(f"./tmp/traces/{dset_name}.ndjson")
    traces = load_ndjson_generic(traces_file, EvalTrace)

    examples_by_id = {e.example_id: e for e in load_list_json_generic(cfg.dataset_path, QaQuestion)}
    print(f"loaded {len(traces)} runs, {len(examples_by_id)} examples")

    # where to store results
    metrics_file = Path(f"./tmp/metrics/{dset_name}.ndjson")
    metrics_file.parent.mkdir(exist_ok=True)

    # avoid recomputing done results
    done_metrics: set[tuple[str, str, str]] = (
        {m.fingerprint for m in load_ndjson_generic(metrics_file, FloatTraceMetric)}
        if metrics_file.exists()
        else set()
    )

    for judge in JUDGES:
        print(f" --- Judge: {judge.name} ---")
        for i, trace in enumerate(traces):
            if isinstance(judge, judges.HumanRatingJudge) and judge.should_skip(i):
                print(f"Skip {i}.")
            # perhaps example_id is redundant here, but nice for clarity
            fingerprint = (trace.trace_id, trace.example_id, judge.name)
            # skip already computed
            if fingerprint in done_metrics:
                print(f"ALREADY DONE {fingerprint}")
                continue

            print(f"{i + 1:3d}/{len(traces)} -> {fingerprint}")
            case = EvalCase(trace, examples_by_id[trace.example_id])

            result = judge.evaluate(case)
            # store result
            append_ndjson(metrics_file, [result])
