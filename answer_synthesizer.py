from __future__ import annotations
import json
from typing import List, Tuple, Optional

import numpy as np
from schemas import Evidence, Answer
from utils import chat
from prompts import SYNTHESIZE_PROMPT_TEMPLATE
from rich import print

class AnswerSynthesizer:
    """
    Synthesizes a final answer from a list of evidence.
    """

    def __init__(self, chat_fn,v=False):
        self.chat_fn = chat_fn
        self.verbose = v

    def run(self, query: str, evidence: List[Evidence]) -> Answer:
        """Generate a structured answer with supporting evidence and dialogue ids."""
        evidence_str, audio, sr = self._format_evidence(evidence)
        if audio is not None and self.verbose:
            print(f"[AnswerSynthesizer] Using audio evidence with sample rate {sr} and length {len(audio)}")
        answer_schema = json.dumps(
            Answer.model_json_schema(), ensure_ascii=False, indent=2
        )
        prompt = SYNTHESIZE_PROMPT_TEMPLATE.format(
            query=query, evidence=evidence_str, answer_schema=answer_schema
        )
        answer = self.chat_fn(prompt, schema=Answer, audio=audio, sample_rate=sr)
        return answer

    def _format_evidence(self, evidence: List[Evidence]) -> Tuple[str, Optional[np.ndarray], Optional[int]]:
        audio_evidence = [ev for ev in evidence if ev.audio is not None]
        
        formatted_parts = []
        for ev in evidence:
            part = (
                f"Citation: [{ev.utterance_id}]\n"
                f"Speaker: {ev.speaker_id}\n"
                f"Time: {ev.time_range[0]:.2f}s - {ev.time_range[1]:.2f}s\n"
                f"Transcript: {ev.transcript}"
            )
            if ev.audio is not None:
                part += "\n[Audio segment available]"
            formatted_parts.append(part)
        
        evidence_str = "\n\n".join(formatted_parts)
        
        if not audio_evidence:
            return evidence_str, None, None
        
        audio_evidence.sort(key=lambda ev: ev.time_range[0] if ev.time_range[0] is not None else 0)
        
        sample_rate = audio_evidence[0].sample_rate or 16000
        
        audio_segments = []
        for ev in audio_evidence:
            if ev.audio is not None:
                audio_segments.append(ev.audio)
        
        if audio_segments:
            concatenated_audio = np.concatenate(audio_segments, axis=0)
            return evidence_str, concatenated_audio, sample_rate
        
        return evidence_str, None, None
