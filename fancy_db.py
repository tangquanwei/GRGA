from __future__ import annotations

from collections import Counter, defaultdict
import math
import json
from dataclasses import dataclass
from itertools import chain, tee
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
import networkx as nx
import numpy as np
from utils import load_text_embedding_model, chat_sf
from functools import lru_cache
import importlib
from schemas import SpeakerProfile
from loguru import logger



CHAT_FN = chat_sf
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "so",
    "that",
    "the",
    "their",
    "them",
    "there",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "who",
    "why",
    "will",
    "with",
    "you",
    "your",
}
PRONOUNS = {
    "he",
    "she",
    "they",
    "it",
    "him",
    "her",
    "them",
    "that",
    "this",
    "those",
    "these",
}

FILLER_TOKENS = {
    "yeah",
    "yah",
    "ya",
    "ok",
    "okay",
    "okey",
    "okayy",
    "mm",
    "mmm",
    "mmhmm",
    "mm-hmm",
    "hmm",
    "huh",
    "uh",
    "uhh",
    "uh-huh",
    "uh-huh",
    "um",
    "umm",
    "erm",
    "er",
    "yep",
    "nope",
    "sure",
    "okey-doke",
    "oh"
}

FILLER_WHITELIST = {
    "no",
    "no.",
    "right?",
    "right",
    "[laughter]",
    "laughter",
    "(laughter)",
}

_SPACY_MODEL = None
_SPACY_DISABLED = False

MERGE_MAX_DURATION=30

def _lazy_load_spacy_model():
    global _SPACY_MODEL, _SPACY_DISABLED
    if _SPACY_DISABLED or _SPACY_MODEL is not None:
        return _SPACY_MODEL
    try:
        spacy = importlib.import_module("spacy")
        _SPACY_MODEL = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except Exception as exc:  
        logger.warning("spaCy unavailable for filler pruning: {}", exc)
        _SPACY_DISABLED = True
        _SPACY_MODEL = None
    return _SPACY_MODEL


def _tokenize_text(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


@dataclass(frozen=True)
class UtteranceFeatures:
    conversation_id: int
    turn_id: int
    speaker: int
    node_id: str
    transcript: str
    time_range: Tuple[float | None, float | None]
    start: float | None
    end: float | None
    tokens: Tuple[str, ...]
    keywords: frozenset[str]
    audio: str | None = None 
    asr_confidence: float | None= None
    spk_confidence: float | None= None

    def to_dict(self) -> Dict:
        return {
            "utterance_id": self.node_id,
            "conversation_id": self.conversation_id,
            "turn_id": self.turn_id,
            "speaker_id": self.speaker,
            "transcript": self.transcript,
            "time_range": self.time_range,
            "asr_confidence": self.asr_confidence,
            "spk_confidence": self.spk_confidence,
        }


@dataclass(frozen=True)
class TextEmbeddings:
    node_ids: Tuple[str, ...]
    transcripts: Tuple[str, ...]
    embeddings: np.ndarray


def pairwise(iterable: Iterable):
    first, second = tee(iterable)
    next(second, None)
    return zip(first, second)


def _keyword_set(tokens: Iterable[str]) -> set:
    return {token for token in tokens if token not in STOPWORDS}


def _resolve_utterances(records: Iterable[Dict]) -> List[Dict]:
    record_list = list(records)
    if not record_list:
        return []
    first = record_list[0]
    if "conversation_id" in first :
        return record_list
    return [
        utterance
        for record in record_list
        for utterance in record.get("convs", [])
        if "conversation_id" in utterance 
    ]


def _is_whitelisted(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return False
    if normalized in FILLER_WHITELIST:
        return True
    return "laughter" in normalized


def _looks_like_short_filler(tokens: Sequence[str]) -> bool:
    if not tokens:
        return True
    if len(tokens) > 3:
        return False
    return all(token in FILLER_TOKENS for token in tokens)


def _spaCy_marks_as_interjection(text: str) -> bool:
    nlp = _lazy_load_spacy_model()
    if nlp is None:
        return False
    doc = nlp(text)
    meaningful = [token for token in doc if not token.is_space]
    if not meaningful:
        return True
    if len(meaningful) > 3:
        return False
    return all(token.pos_ == "INTJ" for token in meaningful)


def _should_filter_utterance(utterance: Dict) -> bool:
    text = utterance.get("text","").lower().strip()
    if not text:
        return True
    if _is_whitelisted(text):
        return False
    tokens = _tokenize_text(text)
    if _looks_like_short_filler(tokens):
        return True
    if len(tokens) <= 3 and _spaCy_marks_as_interjection(text):
        return True
    asr_conf = utterance.get("asr_confidence")
    if asr_conf is not None and asr_conf < 0.35 and len(tokens) <= 2:
        return True
    return False


def _unique_turns(utterances: List[Dict]) -> List[Dict]:
    turn_map = {utterance["turn_id"]: utterance for utterance in utterances}
    return [turn_map[idx] for idx in sorted(turn_map)]


def _normalize_timestamp(raw_range: Optional[Sequence]) -> Tuple[Optional[float], Optional[float]]:
    if not raw_range:
        return None, None
    if isinstance(raw_range, (list, tuple)):
        start = raw_range[0] if len(raw_range) >= 1 else None
        end = raw_range[1] if len(raw_range) >= 2 else None
        return start, end
    return None, None


def _merge_same_speaker_turns(utterances: List[Dict]) -> List[Dict]:
    if not utterances:
        return utterances

    merged: List[Dict] = []
    for utterance in utterances:
        current = dict(utterance)
        current.setdefault("merged_turn_ids", [current.get("turn_id")])
        if not merged:
            merged.append(current)
            continue

        last = merged[-1]
        if last.get("speaker") != current.get("speaker"):
            merged.append(current)
            continue

        last_start, last_end = _normalize_timestamp(last.get("timestamp"))
        curr_start, curr_end = _normalize_timestamp(current.get("timestamp"))
        merged_start = min(
            [value for value in (last_start, curr_start) if value is not None],
            default=last_start or curr_start,
        )
        merged_end = max(
            [value for value in (last_end, curr_end) if value is not None],
            default=last_end or curr_end,
        )
        duration = (
            (merged_end - merged_start)
            if merged_start is not None and merged_end is not None
            else None
        )
        if duration is not None and duration > MERGE_MAX_DURATION:
            merged.append(current)
            continue

        last_text = (last.get("text") or "").strip()
        curr_text = (current.get("text") or "").strip()
        if last_text and curr_text:
            combined_text = f"{last_text} {curr_text}"
        else:
            combined_text = last_text or curr_text
        last["text"] = combined_text

        if merged_start is not None or merged_end is not None:
            last["timestamp"] = (merged_start, merged_end)

        last_conf = last.get("asr_confidence")
        curr_conf = current.get("asr_confidence")
        if last_conf is None:
            last["asr_confidence"] = curr_conf
        elif curr_conf is not None:
            last["asr_confidence"] = max(last_conf, curr_conf)

        merged_ids = last.setdefault("merged_turn_ids", [])
        curr_ids = current.get("merged_turn_ids") or [current.get("turn_id")]
        merged_ids.extend(curr_ids)

    return merged


def _group_conversations(utterances: Iterable[Dict]) -> Dict[int, List[Dict]]:
    buckets: Dict[int, List[Dict]] = defaultdict(list)
    for utterance in utterances:
        buckets[utterance["conversation_id"]].append(utterance)
    return {conv_id: _unique_turns(items) for conv_id, items in buckets.items()}


def _make_feature(conv_id: int, utterance: Dict) -> UtteranceFeatures:
    tokens = tuple(_tokenize_text(utterance["text"]))
    keywords = frozenset(_keyword_set(tokens))
    raw_range = tuple(utterance.get("timestamp", (None, None)))
    start = raw_range[0] if raw_range else None
    end = raw_range[1] if len(raw_range) > 1 else None
    time_range = (start, end)
    return UtteranceFeatures(
        conversation_id=conv_id,
        turn_id=utterance["turn_id"],
        speaker=utterance["speaker"],
        node_id=f"conv{conv_id}_utt{utterance['turn_id']}",
        transcript=utterance["text"],
        time_range=time_range,
        start=start,
        end=end,
        tokens=tokens,
        keywords=keywords,
        asr_confidence=utterance.get("asr_confidence"),
    )


def _build_features(
    conversations: Dict[int, List[Dict]],
    filter_utterances: bool = False,
) -> List[List[UtteranceFeatures]]:
    feature_groups: List[List[UtteranceFeatures]] = []
    for conv_id, turns in conversations.items():
        merged_turns = _merge_same_speaker_turns(turns)
        filtered_turns = [
            utterance
            for utterance in merged_turns
            if not filter_utterances or not _should_filter_utterance(utterance)
        ]
        if not filtered_turns:
            filtered_turns = merged_turns
        feature_groups.append([
            _make_feature(conv_id, utterance) for utterance in filtered_turns
        ])
    return feature_groups


def _collect_edges(
    features_per_conversation: Iterable[List[UtteranceFeatures]], reply_gap: float
):
    for features in features_per_conversation:
        yield from temporal_edges(features)
        yield from same_speaker_edges(features)
        yield from reference_edges(features)
        yield from reply_to_edges(features, reply_gap)


def _group_by_speaker(
    features: Iterable[UtteranceFeatures],
) -> Dict[int, List[UtteranceFeatures]]:
    buckets: Dict[int, List[UtteranceFeatures]] = defaultdict(list)
    for feature in features:
        buckets[feature.speaker].append(feature)
    return {
        speaker_id: sorted(items, key=lambda item: (item.conversation_id, item.turn_id))
        for speaker_id, items in buckets.items()
    }


def _default_profile_builder(
    speaker_id: int, features: Sequence[UtteranceFeatures]
) -> str:
    transcripts = [feature.transcript for feature in features]
    keyword_counts = Counter(
        token for feature in features for token in feature.keywords if token
    )
    top_keywords = ", ".join(token for token, _ in keyword_counts.most_common(5)) or ""
    snippet = " ".join(transcripts[:3])
    if len(transcripts) > 3:
        snippet = f"{snippet} ..."
    summary_parts = [f"Speaker {speaker_id}"]
    if top_keywords:
        summary_parts.append(f"keywords: {top_keywords}")
    if snippet:
        summary_parts.append(f"sample: {snippet}")
    return SpeakerProfile(
        speaker_id=str(speaker_id),
        summary=" | ".join(summary_parts),
        gender="unknown",
        gender_evidence="",
        role="unknown",
        role_evidence="",
    )


def _llm_profile_builder(
    speaker_id: int, features: Sequence[UtteranceFeatures], chat_fn=CHAT_FN
) -> str:
    logger.info(f"_llm_profile_builder [{speaker_id}]")
    from prompts import SPEAKER_PROFILE_PROMPT_TEMPLATE

    convs = []
    for utt in features:
        start, end = utt.time_range
        speaker = utt.speaker
        text = utt.transcript
        formatted_line = f"[{start:>6.2f} - {end:>6.2f}] SPEAKER_{speaker}: {text}"
        convs.append(formatted_line)
    speaker_diary = "\n".join(convs)
    speaker_profile_schema = json.dumps(
        SpeakerProfile.model_json_schema(), ensure_ascii=False, indent=2
    )
    prompt = SPEAKER_PROFILE_PROMPT_TEMPLATE.format(
        speaker_id=speaker_id,
        speaker_diary=speaker_diary,
        speaker_profile_schema=speaker_profile_schema,
    )
    response = chat_fn(prompt, schema=SpeakerProfile)

    return response


def _speaker_payloads(
    speaker_groups: Dict[int, List[UtteranceFeatures]],
    profile_builder: Callable[[int, Sequence[UtteranceFeatures]], str],
) -> List[Dict]:
    def _ensure_profile(speaker_id: int, profile: SpeakerProfile | Dict | str) -> SpeakerProfile:
        if isinstance(profile, SpeakerProfile):
            return profile
        if isinstance(profile, dict):
            return SpeakerProfile.model_validate(profile)
        if isinstance(profile, str):
            try:
                parsed = json.loads(profile)
                return SpeakerProfile.model_validate(parsed)
            except Exception:
                return SpeakerProfile(
                    speaker_id=str(speaker_id),
                    summary=profile,
                    gender="unknown",
                    gender_evidence="not provided",
                    role="unknown",
                    role_evidence="",
                )
        return SpeakerProfile(
            speaker_id=str(speaker_id),
            summary=str(profile),
            gender="unknown",
            gender_evidence="not provided",
            role="unknown",
            role_evidence="",
        )

    return [
        {
            "node_id": f"speaker_{speaker_id}",
            "node_type": "speaker",
            "speaker_id": str(speaker_id),
            "utterance_ids": tuple(feature.node_id for feature in features),
            "utterance_count": len(features),
            "profile": _ensure_profile(speaker_id, profile_builder(speaker_id, features)),
        }
        for speaker_id, features in sorted(
            speaker_groups.items(), key=lambda item: item[0]
        )
    ]


def _speaker_edges(
    payloads: Sequence[Dict], relation: str = "Speaks"
) -> List[Tuple[str, str, str]]:
    return [
        (payload["node_id"], utterance_id, relation)
        for payload in payloads
        for utterance_id in payload["utterance_ids"]
    ]


def temporal_edges(features: List[UtteranceFeatures]) -> List[Tuple[str, str, str]]:
    return [
        (left.node_id, right.node_id, "Temporal") for left, right in pairwise(features)
    ]


def same_speaker_edges(features: List[UtteranceFeatures]) -> List[Tuple[str, str, str]]:
    tracks: Dict[int, List[UtteranceFeatures]] = defaultdict(list)
    for feature in features:
        tracks[feature.speaker].append(feature)
    return [
        (left.node_id, right.node_id, "SameSpeaker")
        for track in tracks.values()
        for left, right in pairwise(track)
    ]


def reply_to_edges(
    features: List[UtteranceFeatures], reply_gap: float
) -> List[Tuple[str, str, str]]:
    edges: List[Tuple[str, str, str]] = []
    for idx, left in enumerate(features):
        if left.end is None:
            continue
        for right in features[idx + 1 :]:
            if right.speaker == left.speaker:
                continue
            if right.start is None:
                continue
            gap = right.start - left.end
            if gap < 0:
                continue
            if gap <= reply_gap:
                edges.append((left.node_id, right.node_id, "ReplyTo"))
                break
            if gap > reply_gap:
                break
    return edges


def reference_edges(features: List[UtteranceFeatures]) -> List[Tuple[str, str, str]]:
    edges: List[Tuple[str, str, str]] = []
    for idx, current in enumerate(features):
        if not any(token in PRONOUNS for token in current.tokens):
            continue
        best_score = 0
        best_candidate: UtteranceFeatures | None = None
        for previous in reversed(features[:idx]):
            if previous.speaker == current.speaker:
                continue
            overlap = len(current.keywords & previous.keywords)
            if overlap == 0 and not current.keywords:
                overlap = 1
            if overlap == 0:
                continue
            if overlap > best_score:
                best_score = overlap
                best_candidate = previous
        if best_candidate is not None:
            edges.append((best_candidate.node_id, current.node_id, "Reference"))
    return edges


def semantic_reference_edges(
    node_index: Dict[str, Dict],
    doc_embeddings: Optional[TextEmbeddings],
    *,
    similarity_threshold: float = 0.82,
    topk_per_node: int = 3,
    ) -> List[Tuple[str, str, float]]:
    if not doc_embeddings or doc_embeddings.embeddings.size == 0:
        return []

    embeddings = doc_embeddings.embeddings
    node_ids = doc_embeddings.node_ids
    if embeddings.shape[0] != len(node_ids):
        return []

    edges: List[Tuple[str, str, float]] = []
    seen_pairs: Set[Tuple[str, str]] = set()

    for idx, node_id in enumerate(node_ids):
        payload = node_index.get(node_id)
        if not payload:
            continue
        sims = embeddings @ embeddings[idx]
        sims[idx] = -1.0
        if topk_per_node < len(sims):
            candidate_idx = np.argpartition(sims, -topk_per_node)[-topk_per_node:]
        else:
            candidate_idx = np.arange(len(sims))
        for candidate in candidate_idx:
            score = float(sims[candidate])
            if score < similarity_threshold:
                continue
            candidate_id = node_ids[candidate]
            if candidate_id == node_id:
                continue
            candidate_payload = node_index.get(candidate_id)
            if not candidate_payload:
                continue
            if (
                payload.get("conversation_id")
                != candidate_payload.get("conversation_id")
            ):
                continue
            source_turn = payload.get("turn_id")
            target_turn = candidate_payload.get("turn_id")
            if (
                source_turn is not None
                and target_turn is not None
                and target_turn <= source_turn
            ):
                continue
            key = (node_id, candidate_id)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            edges.append((node_id, candidate_id, score))
    return edges


def _attach_semantic_edges(
    graph: nx.MultiDiGraph, edges: Sequence[Tuple[str, str, float]]
) -> None:
    if not edges:
        return
    for src, dst, score in edges:
        graph.add_edge(
            src,
            dst,
            relation="SemanticReference",
            weight=float(score),
            similarity=float(score),
        )


def build_fancy_graph(
    records: List[Dict],
    reply_gap: float = 5.0,
    filter_utterances: bool = False ,
    profile_builder: Optional[Callable[[int, Sequence[UtteranceFeatures]], str]] = _default_profile_builder,
) -> Tuple[nx.MultiDiGraph, Dict[str, Dict], Dict[str, Dict], TextEmbeddings]:
    utterances = _resolve_utterances(records)
    conversations = _group_conversations(utterances)
    features_per_conversation = _build_features(conversations)

    graph = nx.MultiDiGraph()

    all_features = [
        feature for features in features_per_conversation for feature in features
    ]
    node_payloads = [feature.to_dict() for feature in all_features]
    graph.add_nodes_from(
        (payload["utterance_id"], {**payload, "node_type": "utterance"})
        for payload in node_payloads
    )
    node_index = {payload["utterance_id"]: payload for payload in node_payloads}

    speaker_groups = _group_by_speaker(all_features)
    profile_fn = profile_builder
    speaker_payloads = _speaker_payloads(speaker_groups, profile_fn)
    graph.add_nodes_from((payload["node_id"], payload) for payload in speaker_payloads)
    speaker_index = {payload["node_id"]: payload for payload in speaker_payloads}

    doc_embeddings = make_document_embedding(node_index)
    semantic_edges = semantic_reference_edges(
        node_index,
        doc_embeddings,
    )

    edges = dict.fromkeys(
        chain(
            _collect_edges(features_per_conversation, reply_gap),
            _speaker_edges(speaker_payloads),
        )
    )
    graph.add_edges_from(
        (src, dst, {"relation": relation}) for src, dst, relation in edges
    )

    _attach_semantic_edges(graph, semantic_edges)
    return graph, node_index, speaker_index, doc_embeddings


def describe_graph(graph: nx.MultiDiGraph) -> Dict[str, Dict]:
    relations = Counter(
        edge_data["relation"] for _, _, edge_data in graph.edges(data=True)
    )
    return {"nodes": graph.number_of_nodes(), "edges": dict(relations)}

# MARK: retrieve 

def _embedding_signature(node_index: Dict[str, Dict]) -> Tuple[Tuple[str, str], ...]:
    return tuple(
        sorted(
            (
                node_id,
                payload["transcript"],
            )
            for node_id, payload in node_index.items()
        )
    )


@lru_cache(maxsize=1)
def _cached_document_embeddings(
    signature: Tuple[Tuple[str, str], ...],
) -> TextEmbeddings:
    logger.info("_cached_document_embeddings")
    if not signature:
        return TextEmbeddings(
            node_ids=tuple(),
            transcripts=tuple(),
            embeddings=np.empty((0, 0), dtype=np.float32),
        )
    node_ids, transcripts = zip(*signature)
    model = load_text_embedding_model()
    embeddings = model.encode(
        list(transcripts), convert_to_numpy=True, normalize_embeddings=True
    )
    return TextEmbeddings(
        node_ids=tuple(node_ids),
        transcripts=tuple(transcripts),
        embeddings=embeddings,
    )


@lru_cache(maxsize=4)
def _cached_bm25_statistics(
    signature: Tuple[Tuple[str, str], ...],
) -> Tuple[
    Tuple[Tuple[str, int], ...],
    Tuple[Tuple[str, Tuple[Tuple[str, int], ...]], ...],
    Tuple[Tuple[str, int], ...],
    float,
]:
    if not signature:
        return tuple(), tuple(), tuple(), 0.0

    doc_length_items: List[Tuple[str, int]] = []
    term_freq_items: List[Tuple[str, Tuple[Tuple[str, int], ...]]] = []
    doc_freq_counter: Counter = Counter()

    for node_id, transcript in signature:
        tokens = _tokenize_text(transcript or "")
        doc_length_items.append((node_id, len(tokens)))
        freq = Counter(tokens)
        freq_items = tuple(sorted(freq.items()))
        term_freq_items.append((node_id, freq_items))
        for term in freq:
            doc_freq_counter[term] += 1

    total_docs = len(doc_length_items)
    avgdl = (
        (sum(length for _, length in doc_length_items) / total_docs)
        if total_docs
        else 0.0
    )
    doc_freq_items = tuple(sorted(doc_freq_counter.items()))
    return (
        tuple(doc_length_items),
        tuple(term_freq_items),
        doc_freq_items,
        avgdl,
    )


def make_document_embedding(node_index: Dict[str, Dict]) -> TextEmbeddings:
    signature = _embedding_signature(node_index)
    return _cached_document_embeddings(signature)


def semantic_search(
    node_index: Dict[str, Dict], query: str, *, topk: int = 10, doc_embeddings=None
) -> List[Dict]:
    query_normalized = query.strip()
    if not query_normalized:
        return []
    doc_embeddings = doc_embeddings or make_document_embedding(node_index)
    if doc_embeddings.embeddings.size == 0:
        return []
    model = load_text_embedding_model()
    query_embedding = model.encode(
        [query_normalized], convert_to_numpy=True, normalize_embeddings=True
    )[0]
    scores = doc_embeddings.embeddings @ query_embedding
    topk_indices = np.argsort(scores)[-topk:][::-1]
    results = []
    for idx in topk_indices:
        node_id = doc_embeddings.node_ids[idx]
        score = scores[idx]
        results.append(
            {
                **node_index[node_id],
                "score": float(score),
            }
        )
    return results


def _bm25_rank(
    node_index: Dict[str, Dict],
    query_terms: Sequence[str],
    *,
    min_hits: int = 1,
    k1: float = 1.5,
    b: float = 0.75,
) -> Dict[str, Tuple[float, int]]:
    if not node_index or not query_terms:
        return {}

    query_counts = Counter(query_terms)
    unique_terms = list(query_counts.keys())
    if not unique_terms:
        return {}

    signature = _embedding_signature(node_index)
    (
        doc_length_items,
        term_freq_items,
        doc_freq_items,
        avgdl,
    ) = _cached_bm25_statistics(signature)

    total_docs = len(doc_length_items)
    if total_docs == 0:
        return {}

    doc_lengths = dict(doc_length_items)
    term_freqs = {node_id: dict(freq_items) for node_id, freq_items in term_freq_items}
    doc_freqs = Counter(dict(doc_freq_items))

    scores: Dict[str, Tuple[float, int]] = {}
    for node_id, freq in term_freqs.items():
        doc_len = doc_lengths.get(node_id, 0)
        norm = 1 - b + b * (doc_len / avgdl) if avgdl > 0 else 1.0
        matched = 0
        score = 0.0
        for term, qf in query_counts.items():
            tf = freq.get(term, 0)
            if tf == 0:
                continue
            matched += 1
            df_term = doc_freqs.get(term, 0)
            if df_term == 0:
                continue
            idf = math.log((total_docs - df_term + 0.5) / (df_term + 0.5) + 1.0)
            denominator = tf + k1 * norm
            score += idf * ((tf * (k1 + 1)) / denominator) * qf
        if matched < min_hits or score <= 0.0:
            continue
        scores[node_id] = (score, matched)
    return scores


def hybrid_search(
    node_index: Dict[str, Dict],
    query: str,
    *,
    doc_embeddings: Optional[TextEmbeddings] = None,
    topk: int = 10,
    min_hits: int = 1,
    alpha: float = 0.6,
) -> List[Dict]:
    """Combine semantic and keyword search signals for robust retrieval.

    Args:
        node_index (Dict[str, Dict]): Utterance payloads keyed by utterance id.
        query (str): Search text used for both embedding and keyword passes.
            Keyword scoring tokenizes this string on alphanumeric boundaries.
        doc_embeddings (Optional[TextEmbeddings], optional): Cached embeddings to reuse.
        topk (int, optional): Maximum number of blended results to return. Defaults to 10.
        min_hits (int, optional): Minimum keyword matches required to keep a node. Defaults to 1.
        alpha (float, optional): Weight for the semantic similarity score (0-1). Defaults to 0.6.

    Returns:
        List[Dict]: Combined results sorted by hybrid ``score`` with raw components
        attached, plus ``matched_terms`` when keyword hits are present.
    """
    query_normalized = query.strip()
    if not query_normalized:
        return []

    alpha_clamped = min(max(alpha, 0.0), 1.0)

    semantic_results = semantic_search(
        node_index,
        query_normalized,
        topk=topk,
        doc_embeddings=doc_embeddings,
    )
    keyword_results = keyword_search(
        node_index,
        query_normalized,
        min_hits=min_hits,
    )

    candidate_limit = max(topk, len(semantic_results))
    if candidate_limit:
        keyword_results = keyword_results[:candidate_limit]
    semantic_scores = {
        item["utterance_id"]: float(item.get("score", 0.0))
        for item in semantic_results
        if "utterance_id" in item
    }
    keyword_meta = {
        item["utterance_id"]: item for item in keyword_results if "utterance_id" in item
    }
    keyword_scores = {
        node_id: float(meta.get("score", 0.0)) for node_id, meta in keyword_meta.items()
    }

    if not semantic_scores and not keyword_scores:
        return []

    semantic_norm = {
        node_id: max((score + 1.0) / 2.0, 0.0)
        for node_id, score in semantic_scores.items()
    }

    if keyword_scores:
        max_keyword = max(keyword_scores.values())
    else:
        max_keyword = 0.0
    keyword_norm = {
        node_id: (score / max_keyword) if max_keyword > 0 else 0.0
        for node_id, score in keyword_scores.items()
    }

    candidate_ids = set(semantic_scores) | set(keyword_scores)
    results: List[Dict] = []
    for node_id in candidate_ids:
        payload = node_index.get(node_id)
        if not payload:
            continue
        semantic_component = semantic_norm.get(node_id, 0.0)
        keyword_component = keyword_norm.get(node_id, 0.0)
        combined = (
            alpha_clamped * semantic_component
            + (1.0 - alpha_clamped) * keyword_component
        )
        enriched = dict(payload)
        enriched["score"] = combined
        enriched["score_semantic"] = semantic_scores.get(node_id, 0.0)
        enriched["score_keyword"] = keyword_scores.get(node_id, 0.0)
        keyword_detail = keyword_meta.get(node_id)
        if keyword_detail and "matched_terms" in keyword_detail:
            enriched["matched_terms"] = keyword_detail["matched_terms"]
        results.append(enriched)

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:topk]


def keyword_search(
    node_index: Dict[str, Dict], term: str, *, min_hits: int = 1
) -> List[Dict]:
    """Rank utterances with BM25 using the provided term or phrase.

    Args:
        node_index (Dict[str, Dict]): Utterance payloads keyed by utterance id.
        term (str): Keyword phrase to tokenize for scoring.
        min_hits (int, optional): Minimum distinct query tokens required in a document.

    Returns:
        List[Dict]: Utterances sorted by BM25 ``score`` with ``matched_terms`` metadata.
    """
    term_normalized = term.strip()
    if not term_normalized:
        return []
    query_terms = _tokenize_text(term_normalized)
    if not query_terms:
        query_terms = [term_normalized.lower()]
    ranking = _bm25_rank(node_index, query_terms, min_hits=min_hits)
    if not ranking:
        return []
    sorted_nodes = sorted(ranking.items(), key=lambda item: item[1][0], reverse=True)
    results: List[Dict] = []
    for node_id, (score, matched) in sorted_nodes:
        payload = node_index.get(node_id)
        if not payload:
            continue
        enriched = dict(payload)
        enriched["score"] = float(score)
        enriched["matched_terms"] = matched
        results.append(enriched)
    return results


def time_range_search(
    node_index: Dict[str, Dict],
    *,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    require_within: bool = False,
    max_results: Optional[int] = None,
) -> List[Dict]:
    """Filter utterances whose timestamps overlap the provided time window.

    Args:
        node_index (Dict[str, Dict]): Utterance payloads keyed by utterance id.
        start_time (Optional[float], optional): Window start (inclusive). ``None``
            means no lower bound.
        end_time (Optional[float], optional): Window end (inclusive). ``None``
            means no upper bound.
        require_within (bool, optional): When ``True`` only keep utterances fully
            contained within the window. Otherwise any overlap matches. Defaults to
            ``False``.
        max_results (Optional[int], optional): Limit the number of returned items.
            ``None`` keeps all matches. Defaults to None.

    Returns:
        List[Dict]: Matching utterances sorted by proximity to ``start_time`` and
        then by start timestamp, enriched with ``overlap_duration`` metadata when
        computable.
    """
    if start_time is None and end_time is None:
        return []

    if start_time is not None and end_time is not None and start_time > end_time:
        start_time, end_time = end_time, start_time
    if start_time == end_time:
        end_time = start_time + 30
    query_start = float(start_time) if start_time is not None else float(min(0, end_time - 30))
    query_end = float(end_time) if end_time is not None else float(start_time + 30)

    matches: List[Dict] = []
    for payload in node_index.values():
        time_range = payload.get("time_range")
        if not time_range:
            continue
        segment_start: Optional[float] = None
        segment_end: Optional[float] = None
        if isinstance(time_range, (list, tuple)):
            if len(time_range) >= 1:
                segment_start = time_range[0]
            if len(time_range) >= 2:
                segment_end = time_range[1]
        if segment_start is None and segment_end is None:
            continue

        seg_start_val = (
            float(segment_start) if segment_start is not None else float("-inf")
        )
        seg_end_val = float(segment_end) if segment_end is not None else float("inf")

        if require_within:
            is_match = seg_start_val >= query_start and seg_end_val <= query_end
        else:
            is_match = seg_end_val >= query_start and seg_start_val <= query_end
        if not is_match:
            continue

        overlap_duration: Optional[float] = None
        overlap_start = max(seg_start_val, query_start)
        overlap_end = min(seg_end_val, query_end)
        if math.isfinite(overlap_start) and math.isfinite(overlap_end):
            overlap_duration = max(overlap_end - overlap_start, 0.0)

        distance_to_start: Optional[float] = None
        if start_time is not None and segment_start is not None:
            distance_to_start = abs(float(segment_start) - float(start_time))

        enriched = dict(payload)
        if overlap_duration is not None:
            enriched["overlap_duration"] = overlap_duration
        if distance_to_start is not None:
            enriched["distance_to_start"] = distance_to_start
        matches.append(enriched)

    def _start_key(item: Dict) -> float:
        time_range = item.get("time_range")
        if isinstance(time_range, (list, tuple)) and time_range:
            start_candidate = time_range[0]
            if start_candidate is not None:
                return float(start_candidate)
        return float("inf")

    matches.sort(
        key=lambda item: (
            item.get("distance_to_start", 0.0),
            _start_key(item),
        )
    )

    if max_results is not None:
        return matches[:max_results]
    return matches


def traverse_relations(
    graph: nx.MultiDiGraph,
    start_nodes: Sequence[str] | str,
    *,
    max_depth: int = 1,
    relation_filter: Optional[Set[str]] = None,
    include_start: bool = True,
) -> Set[str]:
    if isinstance(start_nodes, str):
        frontier: Set[str] = {start_nodes}
    else:
        frontier = set(start_nodes)
    visited: Set[str] = set(frontier) if include_start else set()
    for _ in range(max_depth):
        if not frontier:
            break
        next_frontier: Set[str] = set()
        for node in frontier:
            for neighbor in _neighbors(graph, node, relation_filter):
                if neighbor not in visited:
                    next_frontier.add(neighbor)
        visited.update(next_frontier)
        frontier = next_frontier
    return visited



def _neighbors(
    graph: nx.MultiDiGraph, node_id: str, relation_filter: Optional[Set[str]]
) -> Set[str]:
    candidates: Set[str] = set()
    for _, dst, edge_data in graph.out_edges(node_id, data=True):
        if relation_filter and edge_data.get("relation") not in relation_filter:
            continue
        candidates.add(dst)
    for src, _, edge_data in graph.in_edges(node_id, data=True):
        if relation_filter and edge_data.get("relation") not in relation_filter:
            continue
        candidates.add(src)
    return candidates


def subgraph_view(
    graph: nx.MultiDiGraph,
    center: str,
    *,
    radius: int = 1,
    relation_filter: Optional[Set[str]] = None,
) -> nx.MultiDiGraph:
    nodes = traverse_relations(
        graph,
        center,
        max_depth=radius,
        relation_filter=relation_filter,
        include_start=True,
    )
    return graph.subgraph(nodes).copy()

# MARK: load & save

def _write_json(target: Path, payload: Dict) -> None:
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _serialize_speaker_payloads(index: Dict[str, Dict]) -> Dict[str, Dict]:
    serialized: Dict[str, Dict] = {}
    for key, payload in index.items():
        payload_copy = dict(payload)
        profile = payload_copy.get("profile")
        if isinstance(profile, SpeakerProfile):
            payload_copy["profile"] = profile.model_dump()
        elif hasattr(profile, "model_dump"):
            payload_copy["profile"] = profile.model_dump()
        serialized[key] = payload_copy
    return serialized


def _deserialize_speaker_payloads(index: Dict[str, Dict]) -> Dict[str, Dict]:
    restored: Dict[str, Dict] = {}
    for key, payload in index.items():
        payload_copy = dict(payload)
        profile = payload_copy.get("profile")
        if isinstance(profile, dict):
            try:
                payload_copy["profile"] = SpeakerProfile.model_validate(profile)
            except Exception:
                payload_copy["profile"] = profile
        restored[key] = payload_copy
    return restored


def _save_embeddings(embeddings: TextEmbeddings, path: Path) -> None:
    np.savez_compressed(
        path,
        embeddings=embeddings.embeddings,
        node_ids=np.array(embeddings.node_ids, dtype=object),
        transcripts=np.array(embeddings.transcripts, dtype=object),
    )


def _load_embeddings(path: Path) -> TextEmbeddings:
    data = np.load(path, allow_pickle=True)
    return TextEmbeddings(
        node_ids=tuple(data["node_ids"].tolist()),
        transcripts=tuple(data["transcripts"].tolist()),
        embeddings=data["embeddings"],
    )


def _prepare_graph_for_serialization(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    graph_copy = graph.copy()
    for _, data in graph_copy.nodes(data=True):
        profile = data.get("profile")
        if isinstance(profile, SpeakerProfile):
            data["profile"] = profile.model_dump()
    return graph_copy


def save_fancy_db(
    graph: nx.MultiDiGraph,
    node_index: Dict[str, Dict],
    speaker_index: Dict[str, Dict],
    embeddings: Optional[TextEmbeddings],
    output_dir: str | Path,
) -> None:
    base_path = Path(output_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    graph_payload = nx.node_link_data(
        _prepare_graph_for_serialization(graph), edges="links"
    )
    _write_json(base_path / "graph.json", graph_payload)

    _write_json(base_path / "nodes.json", node_index)
    speaker_payload = _serialize_speaker_payloads(speaker_index)
    _write_json(base_path / "speakers.json", speaker_payload)

    if embeddings is not None:
        _save_embeddings(embeddings, base_path / "embeddings.npz")


def load_fancy_db(
    input_dir: str | Path,
) -> Tuple[nx.MultiDiGraph, Dict[str, Dict], Dict[str, Dict], Optional[TextEmbeddings]]:
    base_path = Path(input_dir)
    graph_data = json.loads((base_path / "graph.json").read_text(encoding="utf-8"))
    graph = nx.node_link_graph(graph_data, multigraph=True, edges="links")

    for _, data in graph.nodes(data=True):
        profile = data.get("profile")
        if isinstance(profile, dict):
            try:
                data["profile"] = SpeakerProfile.model_validate(profile)
            except Exception:
                data["profile"] = profile

    node_index = json.loads((base_path / "nodes.json").read_text(encoding="utf-8"))
    speaker_raw = json.loads((base_path / "speakers.json").read_text(encoding="utf-8"))
    speaker_index = _deserialize_speaker_payloads(speaker_raw)

    embeddings_path = base_path / "embeddings.npz"
    embeddings = _load_embeddings(embeddings_path) if embeddings_path.exists() else None
    return graph, node_index, speaker_index, embeddings
