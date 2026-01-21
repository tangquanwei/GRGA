from __future__ import annotations
import json
from schemas import QueryIntent
from utils import chat
from prompts import DECOMPOSE_PROMPT_TEMPLATE


class QueryDecomposer:
    """
    Analyzes a user's query to extract structured intent.
    """

    def __init__(self, chat_fn):
        self.chat_fn = chat_fn

    def run(self, query: str) -> QueryIntent:
        """
        Takes a natural language query and returns a structured QueryIntent.
        """
        query_intent_schema = json.dumps(
            QueryIntent.model_json_schema(), ensure_ascii=False, indent=2
        )
        prompt = DECOMPOSE_PROMPT_TEMPLATE.format(
            query=query, query_intent_schema=query_intent_schema
        )

        response = self.chat_fn(prompt, schema=QueryIntent)

        return response
