from typing import Literal
from openai import OpenAI, cli
import sys
import os
import numpy as np
import multiprocessing as mp
from functools import lru_cache
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt
from sentence_transformers import SentenceTransformer 

from rich import print
from loguru import logger


@lru_cache(maxsize=1)
def load_text_embedding_model(
    model_name: str ,
    device=None,
) -> SentenceTransformer:
    if device is None:
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading text embedding model: {model_name} on device: {device}")
    return SentenceTransformer(model_name, device=device)



@lru_cache(maxsize=1)
def load_openai_client(
    base_url: str = "http://localhost:8000/v1",
) -> OpenAI:
    import openai

    client = openai.OpenAI(base_url=base_url, api_key="EMPTY")
    return client


@retry(stop=stop_after_attempt(5))
def chat_v1(
    content: str,
    schema: BaseModel = None,
    max_tokens: int = 1024,
    v=False,
) -> BaseModel | str:
    client = load_openai_client()
    # print(client.models.list())
    model = client.models.list().data[0].id
    pay_load = {
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
    }
    if schema:
        pay_load["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "schema",
                "schema": schema.model_json_schema() if schema else None,
            },
        }
    if v:
        print(pay_load)
    res = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        stream=False,
        **pay_load,
    )
    if schema:
        return schema.model_validate_json(res.choices[0].message.content)
    return res.choices[0].message.content


@retry(stop=stop_after_attempt(5))
def chat(
    content: str,
    audio: str | np.ndarray=None,
    schema: BaseModel = None,
    max_tokens: int = 1024,
    sample_rate: int = 16000,
    v=False,
) -> BaseModel | str:
    """Chat with audio input
    
    Args:
        content: Text prompt/question
        audio: Audio file path (str) or audio numpy array
        schema: Optional Pydantic model for structured output
        max_tokens: Maximum tokens in response
        sample_rate: Sample rate for numpy array audio (default: 16000)
        v: Verbose mode for debugging
        
    Returns:
        Structured output (BaseModel) or string response
    """
    import base64
    import io
    import soundfile as sf
    
    client = load_openai_client()
    model = client.models.list().data[0].id
    # Construct multimodal payload
    pay_load = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": content},
                ]
            }
        ],
    }
    if audio is not None:
      # Process audio to base64
      if isinstance(audio, str):
          # Read from file path
          with open(audio, 'rb') as f:
              audio_bytes = f.read()
          base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
      elif isinstance(audio, np.ndarray):
          # Convert numpy array to WAV format
          buffer = io.BytesIO()
          sf.write(buffer, audio, sample_rate, format='WAV')
          buffer.seek(0)
          audio_bytes = buffer.read()
          base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
      else:
          raise ValueError(f"Unsupported audio type: {type(audio)}")
      # Add audio part to payload
      pay_load["messages"][0]["content"].append(                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": f"data:audio/wav;base64,{base64_audio}"
                        }
                    })
    
    # Add structured output schema if provided
    if schema:
        pay_load["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "schema",
                "schema": schema.model_json_schema(),
            },
        }
    
    if v:
        print(f"Model: {model}")
        if audio is not None:
            print(f"Audio base64 length: {len(base64_audio)}")
    
    res = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        stream=False,
        **pay_load,
    )
    
    if schema:
        return schema.model_validate_json(res.choices[0].message.content)
    return res.choices[0].message.content
