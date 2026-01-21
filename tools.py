from __future__ import annotations

import inspect
from dataclasses import dataclass
from re import L
from typing import Any, Dict, List, Optional, Sequence, Tuple, get_type_hints, Union
import networkx as nx
import re
import soundfile as sf
import numpy as np
from pathlib import Path

from fancy_db import (
    TextEmbeddings,
    hybrid_search,
    keyword_search,
    semantic_search,
    time_range_search,
    traverse_relations,
)


@dataclass
class ToolContext:
    graph: nx.MultiDiGraph
    node_index: Dict[str, Dict]
    speaker_index: Dict[str, Dict]
    doc_embeddings: Optional[TextEmbeddings] = None
    audio_path: Optional[Union[str, Path]] = None

def load_audio_segment(
    path: Union[str, Path],
    start_sec: float,
    end_sec: float|None =None,
    mono: bool = False,
    dtype: Optional[str] = "float32",
) -> Tuple[np.ndarray, int]:
    audio_path = Path(path).expanduser()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if end_sec is None or end_sec == start_sec:
        end_sec = start_sec + 30
    if end_sec < start_sec:
        start_sec, end_sec = end_sec, start_sec
    with sf.SoundFile(audio_path, "r") as f:
        sr = f.samplerate
        total_frames = len(f)
        start_frame = max(0, int(round(start_sec * sr)))
        end_frame = min(total_frames, int(round(end_sec * sr)))
        f.seek(start_frame)
        frames = end_frame - start_frame
        chunk = f.read(frames, dtype=dtype, always_2d=True)
    if mono and chunk.shape[1] > 1:
        chunk = chunk.mean(axis=1)
    elif mono:
        chunk = chunk[:, 0]
    else:
        chunk = np.ascontiguousarray(chunk)
    return chunk, sr


def keyword_search_tool(
    ctx: ToolContext, *, term: str, min_hits: int = 1
) -> List[Dict]:
    """Find utterances scored by BM25 relevance for the given keyword or phrase.

    Args:
        term (str): Keyword or phrase to tokenize and search within transcripts.
        min_hits (int, optional): Minimum distinct query terms that must appear in
            an utterance before it is returned. Defaults to 1.

    Returns:
        List[Dict]: Utterance payloads sorted by BM25 ``score`` with
        ``matched_terms`` indicating how many query tokens hit.
    """
    return keyword_search(ctx.node_index, term, min_hits=min_hits)


def semantic_search_tool(ctx: ToolContext, *, query: str, topk: int = 10) -> List[Dict]:
    """Retrieve top utterances via embedding similarity to the query.

    Args:
        query (str): Natural-language search string to embed and compare.
        topk (int, optional): Maximum number of hits to return. Defaults to 10.

    Returns:
        List[Dict]: Utterance payloads with ``score`` representing cosine similarity.
    """
    return semantic_search(
        ctx.node_index,
        query,
        doc_embeddings=ctx.doc_embeddings,
        topk=topk,
    )


def hybrid_search_tool(
    ctx: ToolContext,
    *,
    query: str,
    topk: int = 10,
    min_hits: int = 1,
    alpha: float = 0.6,
) -> List[Dict]:
    """Blend semantic vector search with BM25 keyword retrieval to rank utterances.

    Args:
        query (str): Search string used for embeddings and BM25 keyword matching.
            Keyword mode tokenizes into terms before scoring.
        topk (int, optional): Maximum number of combined results to return. Defaults to 10.
        min_hits (int, optional): Minimum distinct keyword tokens required before scoring. Defaults to 1.
        alpha (float, optional): Weight for semantic similarity (0-1). Higher favors embeddings. Defaults to 0.6.

    Returns:
        List[Dict]: Hybrid-ranked payloads with ``score``, ``score_semantic``, and ``score_keyword`` fields.
    """
    return hybrid_search(
        ctx.node_index,
        query,
        doc_embeddings=ctx.doc_embeddings,
        topk=topk,
        min_hits=min_hits,
        alpha=alpha,
    )


def time_range_search_tool(
    ctx: ToolContext,
    *,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    require_within: bool = False,
    max_results: Optional[int] = None,
) -> List[Dict]:
    """Retrieve utterances whose timestamps intersect a specified window.

    Args:
        start_time (Optional[float], optional): Window start time in seconds. ``None``
            leaves the window open-ended on the left.
        end_time (Optional[float], optional): Window end time in seconds. ``None``
            leaves the window open-ended on the right.
        require_within (bool, optional): Set to ``True`` to enforce that utterances
            fall entirely inside the window. Defaults to ``False``.
        max_results (Optional[int], optional): Upper bound on returned utterances. Defaults to None.

    Returns:
        List[Dict]: Node payloads sorted by proximity to ``start_time`` with optional
        ``overlap_duration`` metadata.
    """
    return time_range_search(
        ctx.node_index,
        start_time=start_time,
        end_time=end_time,
        require_within=require_within,
        max_results=max_results,
    )


def traverse_relations_tool(
    ctx: ToolContext,
    *,
    start_nodes: Sequence[str] | str,
    relation_filter: Optional[Sequence[str]] = None,
    max_depth: int = 1,
    include_start: bool = True,
) -> List[Dict]:
    """Traverse graph relations and return neighboring utterances.

    Args:
        start_nodes (Sequence[str] | str): One or more utterance ids that seed the traversal.
        relation_filter (Optional[Sequence[str]], optional): Relation labels to keep when
            expanding. Use ``None`` to allow all relations. Defaults to ``None``.
        max_depth (int, optional): Hop limit for the traversal. Defaults to 1.
        include_start (bool, optional): Whether to include the starting nodes in the
            returned set. Defaults to ``True``.

    Returns:
        List[Dict]: Utterance payloads reached within the specified hop distance.
    """
    if relation_filter is None:
        relation_filter_set = None
    elif isinstance(relation_filter, dict):
        relation_filter_set = set(relation_filter.keys())
    else:
        relation_filter_set = set(relation_filter)
    related = traverse_relations(
        ctx.graph,
        start_nodes,
        relation_filter=relation_filter_set,
        max_depth=max_depth,
        include_start=include_start,
    )
    return [ctx.node_index[node_id] for node_id in related if node_id in ctx.node_index]


def _normalize_speaker_ids(speaker_id: Any) -> List[int]:
    if isinstance(speaker_id, str):
        speaker_id = speaker_id.strip()
        if speaker_id.upper().startswith("SPEAKER_"):
            speaker_id = speaker_id.split("_", 1)[1]
        else:
            match = re.search(r"\d+", speaker_id)
            if match:
                speaker_id = match.group()
        try:
            return [int(speaker_id)]
        except ValueError:
            raise ValueError(f"speaker_id string must contain an integer suffix, e.g. 'SPEAKER_1'. but get{speaker_id!r}")
    if isinstance(speaker_id, int):
        return [speaker_id]
    if isinstance(speaker_id, Sequence):
        normalized: List[int] = []
        for item in speaker_id:
            normalized.extend(_normalize_speaker_ids(item))
        return normalized
    raise ValueError("speaker_id must be an int, string, or sequence of those types.")


def speaker_filter_tool(
    ctx: ToolContext, *, nodes: List[Dict], speaker_id: int | List[int]
) -> List[Dict]:
    """Filter utterance payloads to those spoken by the target speaker.

    Args:
        nodes (List[Dict]): Utterance payloads to filter.
        speaker_id (int|List[int]): Speaker identifier(s) to keep.

    Returns:
        List[Dict]: Payloads whose ``speaker_id`` matches the requested value.
    """
    speaker_ids = set(_normalize_speaker_ids(speaker_id))
    return [node for node in nodes if node.get("speaker_id") in speaker_ids]


def get_utterance_tool(ctx: ToolContext, *, node_id: Sequence[str] | str) -> List[Dict]:
    """Fetch full payloads for the specified utterance ids.

    Args:
        node_id (Sequence[str] | str): Single utterance id or sequence of ids to fetch.

    Returns:
        List[Dict]: Payloads corresponding to the requested ids (missing ids are ignored).
    """
    if isinstance(node_id, str):
        node_ids = [node_id]
    else:
        node_ids = list(node_id)
    return [ctx.node_index[nid] for nid in node_ids if nid in ctx.node_index]


def speaker_search_tool(
    ctx: ToolContext, *, speaker_ids: int | Sequence[int] | str
) -> List[Dict]:
    """Return all utterances spoken by the specified speaker(s).

    Args:
        speaker_ids (int | Sequence[int] | str): Speaker identifier(s). Ints like 3 or [0, 3] or strings like
            ``"SPEAKER_3"`` are also supported.

    Returns:
        List[Dict]: Utterance payloads sorted by conversation id and turn order.
    """
    speaker_ids = set(_normalize_speaker_ids(speaker_ids))
    results = [
        node
        for node in ctx.node_index.values()
        if node.get("speaker_id") in speaker_ids
    ]
    results.sort(key=lambda item: (item.get("conversation_id", 0), item.get("turn_id", 0)))
    return results


def load_audio_segment_tool(
    ctx: ToolContext,
    *,
    time_range: Sequence[float],
    mono: bool = False,
    dtype: Optional[str] = "float32",
) -> Dict[str, Any]:
    """
    Load an audio segment for the provided time window. You need to listen to the audio when dealing with Acoustic or emotional-related questions. 
    

    Args:
        time_range (Sequence[float]):a two-item sequence``(start_time, end_time)``  (seconds).

    Returns:
        Dict[str, Any]: ``{"audio": np.ndarray, "sample_rate": int, "start_time": float, "end_time": float}``
    """

    def _extract_bounds(range_value: Sequence[float] | Dict[str, float]) -> Tuple[float, float]:
        if isinstance(range_value, dict):
            start = range_value.get("start")
            if start is None:
                start = range_value.get("start_time")
            end = range_value.get("end")
            if end is None:
                end = range_value.get("end_time")
            if start is None or end is None:
                raise ValueError("time_range dict must include start/start_time and end/end_time")
            return float(start), float(end)
        if isinstance(range_value, (list, tuple)):
            if len(range_value) != 2:
                raise ValueError("time_range sequence must contain exactly two values: (start, end)")
            start_value, end_value = range_value
            return float(start_value), float(end_value)
        raise ValueError("time_range must be a dict with start/end keys or a two-item sequence")

    # Extract window bounds and normalize ordering
    start_time, end_time = _extract_bounds(time_range)

    selected_path = ctx.audio_path
    if selected_path is None:
        raise ValueError("An audio path must be provided either via ctx.audio_path or the path argument.")
    audio_path = Path(selected_path).expanduser()

    audio, sample_rate = load_audio_segment(
        audio_path,
        start_time,
        end_time,
        mono=mono,
        dtype=dtype,
    )

    return {
        "audio": audio,
        "sample_rate": sample_rate,
        "start_time": start_time,
        "end_time": end_time,
    }


TOOL_REGISTRY = {
    "keyword_search": keyword_search_tool,
    "semantic_search": semantic_search_tool,
    "hybrid_search": hybrid_search_tool,
    "time_range_search": time_range_search_tool,
    "speaker_search": speaker_search_tool,
    "traverse_relations": traverse_relations_tool,
    "speaker_filter": speaker_filter_tool,
    "load_audio_segment": load_audio_segment_tool,
}


def _format_annotation(annotation: Any) -> str:
    if annotation is inspect._empty:
        return "Any"
    ann_str = repr(annotation)
    if hasattr(annotation, "__name__"):
        ann_str = annotation.__name__
    else:
        ann_str = ann_str.replace("typing.", "")
    return ann_str


def _collect_parameters(func) -> List[Dict[str, Any]]:
    signature = inspect.signature(func)
    hints = get_type_hints(func)
    collected: List[Dict[str, Any]] = []
    for param in signature.parameters.values():
        if param.name == "ctx":
            continue
        entry: Dict[str, Any] = {"param": param}
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            entry["display"] = f"**{param.name}"
            entry["doc_key"] = f"**{param.name}"
            entry["type"] = "Dict[str, Any]"
            entry["default"] = inspect._empty
        elif param.kind is inspect.Parameter.VAR_POSITIONAL:
            entry["display"] = f"*{param.name}"
            entry["doc_key"] = f"*{param.name}"
            entry["type"] = "Tuple[Any, ...]"
            entry["default"] = inspect._empty
        else:
            annotation = hints.get(param.name, param.annotation)
            entry["display"] = param.name
            entry["doc_key"] = param.name
            entry["type"] = _format_annotation(annotation)
            entry["default"] = param.default
        collected.append(entry)
    return collected


def _format_parameters(func) -> str:
    pieces: List[str] = []
    for entry in _collect_parameters(func):
        part = f"{entry['display']}: {entry['type']}"
        if entry["default"] is not inspect._empty:
            part += f" = {entry['default']!r}"
        pieces.append(part)
    return ", ".join(pieces)


def _format_return_annotation(func) -> str:
    signature = inspect.signature(func)
    hints = get_type_hints(func)
    annotation = hints.get("return", signature.return_annotation)
    return _format_annotation(annotation)


def _tool_doc_summary(func) -> str:
    doc = inspect.getdoc(func) or ""
    return doc.splitlines()[0] if doc else "No description available."


def _extract_param_docs(func) -> Dict[str, str]:
    doc = inspect.getdoc(func) or ""
    lines = doc.splitlines()
    entries: Dict[str, str] = {}
    capturing = False
    current_key: Optional[str] = None
    for raw in lines[1:]:
        stripped = raw.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if lowered in {"args:", "parameters:"}:
            capturing = True
            current_key = None
            continue
        if lowered.startswith("returns:"):
            break
        if capturing:
            if ":" in stripped:
                name, desc = stripped.split(":", 1)
                name_clean = name.split("(", 1)[0].strip()
                current_key = name_clean
                entries[current_key] = desc.strip()
            elif current_key:
                entries[current_key] += f" {stripped}"
    return entries


def _extract_returns_doc(func) -> Optional[str]:
    doc = inspect.getdoc(func) or ""
    lines = doc.splitlines()
    capturing = False
    parts: List[str] = []
    for raw in lines:
        stripped = raw.strip()
        if not stripped and capturing:
            break
        lowered = stripped.lower()
        if lowered in {"args:", "parameters:"} and capturing:
            break
        if lowered.startswith("returns:"):
            capturing = True
            suffix = stripped[len("returns:") :].strip()
            if suffix:
                parts.append(suffix)
            continue
        if capturing:
            if not raw.startswith((" ", "\t")):
                break
            parts.append(stripped)
    if not parts:
        return None
    combined = " ".join(parts)
    if ":" in combined:
        _, remainder = combined.split(":", 1)
        return remainder.strip()
    return combined


def build_tool_prompt_block() -> str:
    """Compose a human-readable description block for planner prompts."""
    lines: List[str] = []
    for name, func in TOOL_REGISTRY.items():
        param_meta = _collect_parameters(func)
        param_docs = _extract_param_docs(func)
        returns_doc = _extract_returns_doc(func)
        params = _format_parameters(func)
        return_ann = _format_return_annotation(func)
        summary = _tool_doc_summary(func)
        signature_str = f"{name}({params})"
        lines.append(f"- `{signature_str}` -> {return_ann}")
        lines.append(f"    - Purpose: {summary}")
        if param_meta:
            documented_any = False
            for entry in param_meta:
                doc = param_docs.get(entry["doc_key"])
                if doc:
                    documented_any = True
                    break
            if documented_any:
                lines.append("    - Parameters:")
                for entry in param_meta:
                    doc = param_docs.get(entry["doc_key"])
                    if not doc:
                        continue
                    qualifiers: List[str] = []
                    if entry["default"] is not inspect._empty:
                        qualifiers.append("optional")
                        qualifiers.append(f"default={entry['default']!r}")
                    qualifier_str = ""
                    if qualifiers:
                        qualifier_str = f", {'; '.join(qualifiers)}"
                    lines.append(
                        f"        - {entry['display']} ({entry['type']}{qualifier_str}): {doc}"
                    )
        if returns_doc:
            lines.append(f"    - Returns: {returns_doc}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Example usage: print the tool prompt block
    print(build_tool_prompt_block())
