DECOMPOSE_PROMPT_TEMPLATE = """
# Role
You are an expert in natural language understanding. 
Your task is to decompose a user's query into a structured intent format for next planning steps, this structured intent will guide the execution of tools to answer the query.

# Query
`{query}`

# Instructions
Analyze the query and extract the key entities, core concepts, time constraints, and any metadata filters.
- **entities**: Specific named person or thing, e.g., "miss Wang".
- **concepts**: The abstract topics or utterances mentioned, e.g., "expression of happiness".
- **time_constraint**: Any reference to time, e.g., "last 5 minutes".
- **metadata_filter**: Key-value pairs for filtering, e.g., `speaker_id`.

# Output
Provide the output as a JSON object conforming to the `QueryIntent` schema.

# QueryIntent Schema
```json
{query_intent_schema}
```
"""

PLANNER_PROMPT_TEMPLATE = """
# Role
You are an expert query planner. Your task is to create a step-by-step execution plan to answer a query based on its structured intent.

# Available Tools
{tool_descriptions}

# Intent
```json
{intent}
```

# Instructions
Generate a JSON `ExecutionPlan` to fulfill the intent.
- Each step in the plan must use one of the available tools.
- Use the `thought` field to explain your reasoning for each step.
- You can reference the output of a previous step using the syntax `"$N"`, where N is the 0-based index of the step.

# ExecutionPlan Schema
```json
{execution_plan_schema}
```

# Example Plan
```json
{{
  "plan": [
    {{
      "tool_name": "hybrid_search",
      "args": {{
        "query": "budget"
      }},
      "thought": "First, I need to find all utterances that mention 'budget'."
    }},
    {{
      "tool_name": "filter_by_speaker",
      "args": {{
        "nodes": "$0",
        "speaker_id": 1
      }},
      "thought": "Next, I will filter these results to only include those from Speaker 1."
    }}
  ]
}}
```

# Output
Provide the output as a JSON object conforming to the `ExecutionPlan` schema.
"""

DIRECT_PLANNER_PROMPT_TEMPLATE = """
# Role
You are an expert query planner. Your task is to generate a step-by-step plan directly from a natural language question.

# Available Tools
{tool_descriptions}

# User Question
`{question}`

# Instructions
Generate a JSON `ExecutionPlan` that answers the user's question.
- Each step must call exactly one available tool.
- Use the `thought` field to explain why the step is required.
- You may reference earlier outputs using the syntax `"$N"`, where N is the 0-based index of a previous step.
- Keep the plan as short as possible while still satisfying the question.

# ExecutionPlan Schema
```json
{execution_plan_schema}
```

# Output
Provide the output as a JSON object conforming to the `ExecutionPlan` schema.
"""

SYNTHESIZE_PROMPT_TEMPLATE = """
# Role
You are an expert summarizer. 
Your task is to synthesize a clear and concise answer to a user's query based on the provided evidence.

# User Query
`{query}`

# Evidence
```
{evidence}
```

# Instructions
- Formulate a direct answer to the user's query.
- The answer MUST be based *only* on the provided evidence.
- For each piece of information you use, you MUST include its citation tag, e.g., `[conv0_utt3]`.
- If the evidence is contradictory or insufficient, state that clearly.

# Answer Schema
```json
{answer_schema}
```

# Output
Provide the output as a JSON object conforming to the `Answer` schema.
"""

REFLECTION_PROMPT_TEMPLATE = """
# Role
You are a meticulous fact-checker. Your task is to determine if a generated answer is fully supported by the provided evidence.

# User Query
`{query}`

# Evidence
```
{evidence}
```

# Generated Answer
`{answer}`

# Instructions
1.  Carefully compare the answer against the evidence.
2.  Determine if every claim in the answer is directly supported by a piece of evidence.
3.  Assess your confidence on a scale of 1 to 5:
  - 1: Very uncertain, the answer is likely unsupported or mostly contradicted by the evidence.
  - 2: Uncertain, only a small part of the answer is supported, or there are significant gaps.
  - 3: Somewhat confident, about half the answer is supported, but there are notable uncertainties.
  - 4: Confident, most claims are supported, with only minor doubts or missing details.
  - 5: Very certain, every claim in the answer is directly and fully supported by the evidence, with no contradictions or omissions.
4.  If the answer is not fully supported, explain what is wrong (e.g., "The answer mentions a specific date that is not in the evidence.").

# Reflection Schema
```json
{reflection_schema}
```

# Output
Provide the output as a JSON object conforming to the `Reflection` schema.
"""

SPEAKER_PROFILE_PROMPT_TEMPLATE = """
# Role
You are an expert in organizational behavior and dialogue analysis.

# Context
You are given the full transcript of a meeting. Speakers are labeled as speaker_0, speaker_1, etc. The excerpt below contains every utterance spoken by `speaker_id`.

# Task
Study the utterances carefully and produce a structured portrait of `speaker_id`. Base every conclusion strictly on the transcript. If the evidence is insufficient for a field, state that explicitly.

# Transcript for `speaker_{speaker_id}`
```
{speaker_diary}
```

# Guidelines
- Provide at least one relationship entry if evidence exists; otherwise return an empty list.
- Always ground every statement in the transcript. Do not invent facts.
- Keep the response concise yet specific.

# SpeakerProfile Schema
```json
{speaker_profile_schema}
```

# Output
Provide the output as a JSON object conforming to the `SpeakerProfile` schema.
"""
