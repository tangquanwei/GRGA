from __future__ import annotations
import json
from typing import List
from schemas import Evidence, Reflection, Answer
from utils import chat
from prompts import REFLECTION_PROMPT_TEMPLATE


class ReflectionAgent:
    """
    Reflects on the generated answer to check for support and confidence.
    """

    def __init__(self, chat_fn):
        self.chat_fn = chat_fn

    def run(self, query: str, evidence: List[Evidence], answer: Answer) -> Reflection:
        """
        Judges if the answer is well-supported by the evidence.
        """
        evidence_str = self._format_evidence(evidence)
        answer_str = self._format_answer(answer)
        reflection_schema = json.dumps(
            Reflection.model_json_schema(), ensure_ascii=False, indent=2
        )
        prompt = REFLECTION_PROMPT_TEMPLATE.format(
            query=query,
            evidence=evidence_str,
            answer=answer_str,
            reflection_schema=reflection_schema,
        )

        response_json = self.chat_fn(prompt, schema=Reflection)

        return Reflection.model_validate(response_json)

    def _format_evidence(self, evidence: List[Evidence]) -> str:
        return "\n\n".join(
            [
                f"Citation: [{ev.utterance_id}]\n"
                f"Speaker: {ev.speaker_id}\n"
                f"Time: {ev.time_range[0]:.2f}s - {ev.time_range[1]:.2f}s\n"
                f"Transcript: {ev.transcript}"
                for ev in evidence
            ]
        )

    def _format_answer(self, answer: Answer) -> str:
        return f"Answer: {answer.answer}\nCitations: {', '.join(answer.cite_conversation_ids)}"
