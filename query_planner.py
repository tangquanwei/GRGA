from __future__ import annotations

import json
from typing import List, Dict, Any, Optional

from schemas import QueryIntent, ExecutionPlan
from utils import chat
from prompts import PLANNER_PROMPT_TEMPLATE, DIRECT_PLANNER_PROMPT_TEMPLATE
from tools import build_tool_prompt_block


class QueryPlanner:
    """
    Generates an execution plan based on the query intent.
    """

    def __init__(self, chat_fn):
        self.chat_fn = chat_fn

    def run(self, intent: QueryIntent | str, history: Optional[List[Dict[str, Any]]] = None) -> ExecutionPlan:
        """
        Takes a QueryIntent and generates a JSON-based execution plan.
        
        Args:
            intent: Query intent or question string
            history: Optional list of previous attempts with their failures
        
        Returns:
            ExecutionPlan for retrieving evidence
        """
        tool_descriptions = build_tool_prompt_block()
        execution_plan_schema = json.dumps(
            ExecutionPlan.model_json_schema(),
            ensure_ascii=False,
            indent=2,
        )

        if isinstance(intent, QueryIntent):
            base_prompt = PLANNER_PROMPT_TEMPLATE.format(
                intent=intent.model_dump_json(),
                tool_descriptions=tool_descriptions,
                execution_plan_schema=execution_plan_schema,
            )
        elif isinstance(intent, str):
            question = intent.strip()
            if not question:
                raise ValueError("Question string cannot be empty.")
            base_prompt = DIRECT_PLANNER_PROMPT_TEMPLATE.format(
                question=question,
                tool_descriptions=tool_descriptions,
                execution_plan_schema=execution_plan_schema,
            )
        else:
            raise ValueError("Invalid type for intent. Expected str or QueryIntent.")

        if history:
            history_text = self._format_history(history)
            prompt = f"{base_prompt}\n\n{history_text}"
        else:
            prompt = base_prompt

        response = self.chat_fn(prompt, schema=ExecutionPlan)

        return response
    
    def _format_history(self, history: List[Dict[str, Any]]) -> str:
        """Format previous attempts into a readable context."""
        lines = ["## Previous Attempts and Learnings\n"]
        
        for i, ctx in enumerate(history, 1):
            lines.append(f"### Attempt {i}")
            
            if "plan" in ctx:
                plan_summary = []
                for step in ctx["plan"].plan:
                    plan_summary.append(f"  - {step.tool_name}({step.args})")
                lines.append(f"Plan executed:\n" + "\n".join(plan_summary))
            
            if "answer" in ctx:
                lines.append(f"Answer generated: {ctx['answer'].answer[:100]}...")
            
            if "reflection" in ctx and ctx["reflection"]:
                reflection = ctx["reflection"]
                lines.append(f"Supported: {reflection.is_supported}")
                lines.append(f"Confidence: {reflection.confidence_score}/5")
                if reflection.correction_needed:
                    lines.append(f"**Problem identified**: {reflection.correction_needed}")
            
            lines.append("")  
        
        lines.append("Based on these previous failures, please create a better plan that addresses the identified problems.")
        
        return "\n".join(lines)
