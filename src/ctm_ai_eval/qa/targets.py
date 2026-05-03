import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import override

import requests

from ctm_ai_eval.qa.config import ChatTargetConfig
from ctm_ai_eval.qa.datamodels import ApiEvalResponse
from ctm_ai_eval.rag.datamodels import RagPipelineTarget, RetrievalResult


class ApiTarget(ABC):
    chat_config: ChatTargetConfig
    api_key: str = "APIKEY"
    server_url: str
    route: str

    @override
    def __str__(self) -> str:
        return f"ApiTarget({self.server_url}, {self.route})"

    def _ask(
        self,
        payload: dict[str, object],
        headers: Mapping[str, str | bytes],
    ) -> ApiEvalResponse:
        t0 = time.time()

        r = requests.post(
            f"{self.server_url}/{self.route}", json=payload, headers=headers, timeout=60
        )
        r.raise_for_status()
        latency_ms = int((time.time() - t0) * 1000)
        raw = r.json()
        return ApiEvalResponse(
            raw=raw,
            latency_ms=latency_ms,
            text=raw["message"]["content"],
            thinking=raw["message"].get("thinking"),
        )

    def _build_messages(self, prompt: str, system_prompt: str | None):
        messages: list[Mapping[str, str]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def build_headers(self) -> Mapping[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def build_payload(self, prompt: str, system_prompt: str | None) -> dict[str, object]:
        return {
            "model": self.chat_config.model,
            "messages": self._build_messages(prompt, system_prompt),
            "temperature": self.chat_config.temperature,
            "max_tokens": self.chat_config.max_tokens,
            "think": self.chat_config.think,
            "stream": False,
        }

    @abstractmethod
    def ask(
        self, question: str, system_prompt: str | None, user_template: str
    ) -> ApiEvalResponse: ...


@dataclass
class OllamaChatTarget(ApiTarget):
    chat_config: ChatTargetConfig
    # NOTE: api/chat allows some more options than standard OpenAI-V1
    route: str = "api/chat"
    server_url: str = "http://127.0.0.1:11434"

    @override
    def ask(self, question: str, system_prompt: str | None, user_template: str) -> ApiEvalResponse:
        return self._ask(self.build_payload(question, system_prompt), self.build_headers())


@dataclass
class RagApiTarget(ApiTarget):
    chat_config: ChatTargetConfig
    rag: RagPipelineTarget
    docs_dir: Path
    top_k: int = 5
    api_key: str = "APIKEY"
    route: str = "api/chat"
    server_url: str = "http://127.0.0.1:11434"

    _ingested: bool = field(default=False, init=False, repr=False)

    @override
    def __str__(self) -> str:
        return f"RAG({self.chat_config.model}, {self.rag.fingerprint_tuple}, {self.docs_dir.stem})"

    def ensure_ingested(self, *, verbose: bool = False) -> None:
        if self._ingested:
            return
        docs = self.rag.loader(self.docs_dir)
        chunks = self.rag.chunker(docs)
        self.rag.retriever.ingest(chunks)
        self._ingested = True
        if verbose:
            print("ingested!")
            print(self.rag.retriever.chunks)

    def render_source_blocks(self, hits: Iterable[RetrievalResult]):
        """
        Build:
        - blocks: text blocks inserted into the LLM prompt
        """
        blocks: list[str] = []

        for i, h in enumerate(hits, start=1):
            tag = f"[S{i}]"
            txt = (h.chunk.text or "").strip()
            blocks.append(f"{tag}\n{txt}\n")

        return blocks

    def augment_prompt(self, retrieved: list[RetrievalResult], question: str, rag_template: str):
        """Put chunks and question together.
        RAG-specific prompt engineering goes here (or in sys prompt).
        """

        sources = "\n\n---\n\n".join(self.render_source_blocks(retrieved))

        return rag_template.format(sources=sources, question=question)

    @override
    def ask(self, question: str, system_prompt: str | None, user_template: str) -> ApiEvalResponse:
        self.ensure_ingested()
        retrieved = self.rag.retriever(question, k=self.top_k)

        res = self._ask(
            self.build_payload(
                self.augment_prompt(retrieved, question, user_template), system_prompt
            ),
            self.build_headers(),
        )
        res.sources = "\n\n".join(self.render_source_blocks(retrieved))
        return res
