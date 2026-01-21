# GRGA

> **Don't Just Listen, Try Planning: Graph-based Retrieval-Generation Agent for Long-form Audio Meeting Understanding**

---

## 📢 Notice

**This repository contains a partial release of the GRGA codebase.**

🚧 The complete code and datasets will be made publicly available upon paper acceptance.

---

## 📖 Overview

GRGA (Graph-based Retrieval-Generation Agent) is a novel framework designed for long-form audio meeting understanding. Unlike traditional approaches that rely on sequential processing, GRGA leverages graph-based retrieval and multi-step planning to effectively handle complex queries over lengthy meeting recordings.

### Key Features

- **Query Decomposition**: Intelligent parsing of user queries into structured intents with entity extraction, concept identification, and temporal constraints
- **Graph-based Planning**: Dynamic execution plan generation using a rich set of retrieval tools
- **Hybrid Retrieval**: Combines keyword search (BM25), semantic search, and graph traversal for comprehensive evidence gathering
- **Reflection Mechanism**: Self-evaluation loop to assess answer quality and trigger re-planning when necessary
- **Multi-modal Support**: Native support for both transcript text and audio segment processing

---

## 🏗️ Architecture

GRGA employs an iterative agentic pipeline with self-reflection capabilities. The system can automatically retry with improved retrieval strategies when the initial answer is deemed insufficient.

```
                                    ┌─────────────────────────────────────────────────────┐
                                    │                   Agentic Pipeline                  │
                                    │                  (max_iterations=N)                 │
                                    └─────────────────────────────────────────────────────┘
                                                              │
                                                              ▼
┌──────────────┐    ┌───────────────────┐    ┌────────────────────────────────────────────┐
│              │    │  Query Decomposer │    │              Graph Database                │
│    User      │    │  ┌─────────────┐  │    │  ┌────────────┐  ┌───────────────────┐     │
│   Question   │───▶│  │  Entities   │  │    │  │   Graph    │  │   Text Embeddings │     │
│              │    │  │  Concepts   │  │    │  │  (nx.MDG)  │  │    (Semantic)     │     │
└──────────────┘    │  │  Time Cons. │  │    │  └────────────┘  └───────────────────┘     │
                    │  │  Metadata   │  │    │  ┌────────────┐  ┌───────────────────┐     │
                    │  └─────────────┘  │    │  │ Node Index │  │  Speaker Index    │     │
                    └─────────┬─────────┘    │  │  (BM25)    │  │   (Profiles)      │     │
                              │              │  └────────────┘  └───────────────────┘     │
                              ▼              └──────────────────────────┬─────────────────┘
                    ┌───────────────────┐                               │
                    │   Query Planner   │                               │
                    │  ┌─────────────┐  │                               │
                    │  │ Plan Steps  │  │                               │
                    │  │ ┌─────────┐ │  │                               │
                    │  │ │Tool Name│ │  │                               │
                    │  │ │  Args   │ │  │                               │
                    │  │ └─────────┘ │  │                               │
                    │  └─────────────┘  │                               │
                    └─────────┬─────────┘                               │
                              │                                         │
                              ▼                                         │
                    ┌───────────────────┐                               │
                    │ Execution Engine  │◀──────────────────────────────┘
                    │  ┌─────────────┐  │
                    │  │ Tools:      │  │
                    │  │ •keyword    │  │
                    │  │ •semantic   │  │
                    │  │ •hybrid     │  │
                    │  │ •time_range │  │
                    │  │ •traverse   │  │
                    │  └─────────────┘  │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │Answer Synthesizer │───────────┐
                    │  ┌─────────────┐  │           │
                    │  │  Evidence   │  │           │
                    │  │  + Answer   │  │           │
                    │  │  + Citations│  │           │
                    │  └─────────────┘  │           │
                    └───────────────────┘           │
                                                    ▼
                    ┌───────────────────┐    ┌─────────────┐
                    │ Reflection Agent  │◀───│   Answer    │
                    │  ┌─────────────┐  │    └─────────────┘
                    │  │is_supported │  │
                    │  │ confidence  │  │
                    │  │ correction  │  │
                    │  └─────────────┘  │
                    └─────────┬─────────┘
                              │
                   ┌──────────┴──────────┐
                   │                     │
            is_supported?          NOT supported
                   │                     │
                   ▼                     ▼
            ┌─────────────┐    ┌──────────────────┐
            │   Output    │    │  Add to History  │
            │   Answer    │    │  & Retry Plan    │──────┐
            └─────────────┘    └──────────────────┘      │
                                                         │
                                        ┌────────────────┘
                                        │ (Loop back to Planner
                                        │  with failure context)
                                        ▼
                              ┌───────────────────┐
                              │   Query Planner   │
                              │   (with history)  │
                              └───────────────────┘
```

### Pipeline Flow

1. **Query Decomposition**: Extracts structured intent from natural language query
2. **Query Planning**: Generates execution plan with tool calls based on intent/question and historical failures
3. **Plan Execution**: Executes retrieval tools against the graph database to gather evidence
4. **Answer Synthesis**: Generates answer with citations from collected evidence
5. **Reflection**: Evaluates if answer is well-supported; if not, triggers re-planning with failure context
6. **Iteration**: Repeats steps 2-5 until answer is validated or max iterations reached

---

## 📁 Released Components

| File | Description |
|------|-------------|
| `query_decomposer.py` | Query intent extraction and structuring |
| `query_planner.py` | Execution plan generation |
| `execution_engine.py` | Plan execution and tool orchestration |
| `answer_synthesizer.py` | Evidence-based answer generation |
| `reflection_agent.py` | Answer quality assessment and re-planning |
| `tools.py` | Retrieval tools (keyword, semantic, hybrid search, etc.) |
| `schemas.py` | Pydantic data models |
| `prompts.py` | LLM prompt templates |
| `fancy_db.py` | Graph database and indexing utilities |
| `utils.py` | Utility functions |

---

## 🔜 Coming Soon

Upon paper acceptance, we will release:

- [ ] 📊 **Datasets**: Annotated long-form meeting audio datasets
- [ ] 📓 **Notebooks**: Tutorial notebooks with examples
- [ ] 📝 **Documentation**: Comprehensive API documentation

---
