"""For debugging etc."""

import json
import tempfile
from pathlib import Path

from ctm_ai_eval.qa.config import ChatTargetConfig
from ctm_ai_eval.qa.targets import OllamaChatTarget, RagApiTarget
from ctm_ai_eval.rag.chunkers.basic_chunking import TokenChunker
from ctm_ai_eval.rag.datamodels import RagPipelineTarget
from ctm_ai_eval.rag.dummy_retrievers import DummyRetriever
from ctm_ai_eval.rag.text_processing import tokenize_words
from ctm_ai_eval.utils.io_util import load_all_md
from ctm_ai_eval.utils.rich_print import CONS

# Create a dummy chat config
_CHAT_CFG = ChatTargetConfig(
    model="dummy_model",
    system_prompt_id="dummy",
    user_template_id="dummy",
    temperature=0.7,
    max_tokens=100,
    think=True,
)

_QUESTION = "Whats up?"
_SYS_PROMPT = "You are a dummy."


# Define a function to generate random sentences
def generate_random_sentence():
    import random

    words = ["Lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit", "sed"]
    return " ".join(random.choices(words, k=random.randint(5, 10)))


def preview_request_chat():
    target = OllamaChatTarget(chat_config=_CHAT_CFG)

    # Build the payload and headers
    payload = target.build_payload(_QUESTION, _SYS_PROMPT)
    headers = target.build_headers()

    # Print the payload and headers
    CONS.print("\n\n-- Chat --", style="black on white")

    CONS.print("\nPayload:", style="bold")
    CONS.print(json.dumps(payload, indent=2))
    CONS.print("\nHeaders:", style="bold")
    CONS.print(json.dumps(headers, indent=2))


def preview_request_rag():
    tmp_dir = Path(tempfile.mkdtemp())

    # Create files with random sentences
    for i in range(3):
        with open(tmp_dir / f"test{i + 1}.md", "w") as f:
            for _ in range(3):
                f.write(generate_random_sentence() + "\n")

    target = RagApiTarget(
        _CHAT_CFG,
        RagPipelineTarget(load_all_md, TokenChunker(5, 2, tokenize_words), DummyRetriever()),
        tmp_dir,
    )
    # manually ingest and retrieve
    target.ensure_ingested()
    retrieved = target.rag.retriever(_QUESTION, k=3)

    payload = target.build_payload(
        target.augment_prompt(
            retrieved,
            _QUESTION,
            rag_template=Path("assets/prompts/user_templates/rag_pass1.jinja").read_text(),
        ),
        Path("assets/prompts/chat_system_prompts/course_assistant_rag_pass1.txt").read_text(),
    )

    CONS.print("\n\n-- RAG --", style="black on white")

    CONS.print("\nPayload:", style="bold")
    CONS.print(json.dumps(payload, indent=2))


# Preview the requests
preview_request_chat()
preview_request_rag()
