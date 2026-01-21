from __future__ import annotations
from email.mime import audio
from functools import partial
from typing import Dict, List, Optional
import networkx as nx

from schemas import ExecutionPlan, Evidence
from fancy_db import (
    describe_graph,
    load_fancy_db,
    build_fancy_graph,
)
from tools import TOOL_REGISTRY, ToolContext
from rich import print


class ExecutionEngine:
    """
    Executes a plan against the database/graph.
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        node_index: Dict,
        speaker_index: Dict,
        doc_embeddings: Optional[object] = None,
        audio_path: str=None,
        v=True,
    ):
        self.graph = graph
        self.node_index = node_index
        self.speaker_index = speaker_index
        self.tool_context = ToolContext(
            graph=graph,
            node_index=node_index,
            speaker_index=speaker_index,
            doc_embeddings=doc_embeddings,
            audio_path=audio_path,
        )
        self.tools = {
            name: partial(func, self.tool_context)
            for name, func in TOOL_REGISTRY.items()
        }
        self.v = v

    def run(self, plan: ExecutionPlan) -> List[Evidence]:
        """
        Executes the plan and returns a list of evidence.
        """
        context = {} 
        final_evidence: List[Evidence] = []
        for i, step in enumerate(plan.plan):
            tool = self.tools.get(step.tool_name)
            if not tool:
                raise ValueError(f"Tool '{step.tool_name}' not found.")

            args = {}
            for k, v in step.args.items():
                if isinstance(v, str) and v.startswith("$"):
                    step_index = int(v.split("$")[-1])
                    prev_step_key = f"step_{step_index}"
                    if step.tool_name == "get_utterance" and k == "node_id":
                        args[k] = [
                            item["utterance_id"]
                            for item in context.get(prev_step_key, [])
                        ]
                    elif step.tool_name == "traverse_relations" and k == "start_nodes":
                        args[k] = [
                            item["utterance_id"]
                            for item in context.get(prev_step_key, [])
                        ]
                    else:
                        args[k] = context.get(prev_step_key)
                else:
                    args[k] = v

            result = tool(**args)
            
            if self.v:
                from pprint import pformat

                def format_arg_value(val):
                    if (
                        val
                        and isinstance(val, list)
                        and isinstance(val[0], dict)
                        and "utterance_id" in val[0]
                    ):
                        return [
                            {
                                k: v
                                for k, v in d.items()
                                if k in ["utterance_id", "speaker_id", "transcript"]
                            }
                            for d in val
                        ]
                    return val

                formatted_args = {k: format_arg_value(v) for k, v in args.items()}
                print(
                    f"Step {i} - Tool: {step.tool_name},\n"
                    f"Args: {pformat(formatted_args, compact=True, width=120)}\n"
                    f"Result Count: {len(result) if isinstance(result, list) else 'N/A'}"
                )
            step_key = f"step_{i}"
            context[step_key] = result
            if step.tool_name == "filter_by_speaker":
                final_evidence.clear()

            if step.tool_name == "load_audio_segment":
                if isinstance(result, dict):
                    final_evidence.append(
                        Evidence(
                            utterance_id=f"audio_{step_key}",
                            transcript="",  
                            speaker_id=-1,  
                            time_range=(result["start_time"], result["end_time"]),
                            source=step_key,
                            audio=result.get("audio"),
                            sample_rate=result.get("sample_rate")
                        )
                    )
            else:
                for item in result:
                    final_evidence.append(
                        Evidence(
                            utterance_id=item["utterance_id"],
                            transcript=item["transcript"],
                            speaker_id=item["speaker_id"],
                            time_range=item["time_range"],
                            source=step_key,
                            audio=None
                        )
                    )

        return self._deduplicate_evidence(final_evidence)

    def _deduplicate_evidence(self, evidence_list: List[Evidence]) -> List[Evidence]:
        unique_evidence = {}
        for ev in evidence_list:
            if ev.utterance_id not in unique_evidence:
                unique_evidence[ev.utterance_id] = ev
        return list(unique_evidence.values())


