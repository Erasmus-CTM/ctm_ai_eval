import platform
from collections.abc import Sequence
from pathlib import Path

import tqdm

from ctm_ai_eval.qa.config import ChatTargetConfig, QaRagConfig, load_qa_config
from ctm_ai_eval.qa.datamodels import EvalTrace, QaQuestion
from ctm_ai_eval.qa.targets import ApiTarget, OllamaChatTarget, RagApiTarget
from ctm_ai_eval.rag.ai_retriever import Embedder, FaissRetriever
from ctm_ai_eval.rag.chunkers.chunker_util import resolve_chunker
from ctm_ai_eval.rag.datamodels import HaystackTarget
from ctm_ai_eval.utils import io_util
from ctm_ai_eval.utils.hashing import stable_hash
from ctm_ai_eval.utils.misc import get_git_commit
from ctm_ai_eval.utils.rich_print import CONS


def make_targets(
    chat_models: list[ChatTargetConfig], rag: QaRagConfig | None
) -> Sequence[ApiTarget]:
    """Prepare all targets to trace."""

    targets_chat = [OllamaChatTarget(c) for c in chat_models]

    if rag is None:
        return targets_chat
    else:
        targets_rag = [
            RagApiTarget(
                c,
                haystack=HaystackTarget(
                    io_util.load_all_md,
                    resolve_chunker(rag.chunker),
                    FaissRetriever(Embedder(rag.embedding_model)),
                ),
                docs_dir=rag.corpus_path,
            )
            for c in chat_models
        ]
        if rag.rag_only:
            return targets_rag
        else:
            return targets_chat + targets_rag


def trace_one_target(data_file: Path, target: ApiTarget, sys_prompts: dict[str, str]) -> None:

    # load dataset
    examples = io_util.load_list_json_generic(data_file, QaQuestion)
    dataset_name = data_file.stem
    rag_config = target.haystack.fingerprint_dict if isinstance(target, RagApiTarget) else None

    # where to store results
    # Somewhat safe for caching and deduplication.
    cfg_hash = stable_hash(
        {
            "git": get_git_commit(),
            "chat": target.chat_config.model_dump(mode="json"),
            "sys_prompts": sys_prompts,
            "rag": rag_config,
        },
        length=64,
    )
    traces_file = Path(f"./tmp/traces/{dataset_name}.ndjson")
    traces_file.parent.mkdir(exist_ok=True, parents=True)

    # previous results?
    if traces_file.exists():
        done_ids = set(t.trace_id for t in io_util.load_ndjson_generic(traces_file, EvalTrace))
    else:
        done_ids: set[str] = set()

    did_ping = False  # only ping target if we need to!
    prog = tqdm.tqdm(examples, ncols=0)
    for ex in prog:
        # hash everything
        trace_hash = stable_hash(
            {
                "cfg": cfg_hash,
                "q": ex.to_question_string(),
            },
            length=64,
        )
        if trace_hash in done_ids:
            CONS.print(f"Already done {trace_hash}, skip", style="yellow")
            continue

        # send an initial request without measuring,
        # to avoid latency spike when loading model
        if not did_ping:
            did_ping = True
            print("pinging target...")
            target.ask("hello", None)

        result = target.ask(
            ex.to_question_string(),
            sys_prompts[target.chat_config.system_prompt_id],
        )
        r = EvalTrace(
            trace_id=trace_hash,
            dataset_name=dataset_name,
            example_id=ex.example_id,
            server_url=target.server_url,
            route=target.route,
            answer=result.text,
            latency_ms=result.latency_ms,
            target_cfg=target.chat_config.model_dump(),
            rag_cfg=rag_config,
            local_host=platform.node(),
            extra_output={"thinking": result.thinking},
        )
        if r.answer:
            prog.set_description(f"{ex.question[:10]} ->  {r.answer[:60].replace('\n', ' ')}")
        io_util.append_ndjson(traces_file, [r])


def qa_trace() -> None:
    """Main function to run the QA trace collection."""

    cfg = load_qa_config()
    targets = make_targets(cfg.targets, cfg.rag)

    # load system prompts
    sys_prompts = {p.stem: p.read_text() for p in cfg.sys_prompt_dir.glob("*.txt")}

    for i, targ in enumerate(targets):
        CONS.print(f"Target {i + 1}/{len(targets)} {targ}")

        # For RAG-targets: run a ingest before.
        # NOTE (maybe one should cache the indexes?)
        if isinstance(targ, RagApiTarget):
            print("RAG-target: ingesting...")
            targ.ensure_ingested()
            print("RAG-target: ingest ok!")

        # Trace the target on the dataset.
        trace_one_target(cfg.dataset_path, targ, sys_prompts)


if __name__ == "__main__":
    qa_trace()
