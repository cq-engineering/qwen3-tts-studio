"""Pydantic request/response models for the REST API.

Sampling params are range-validated here so the UI can never push the worker
into hour-long runs (max_new_tokens 128–4096, temperature 0–2, top-p 0–1,
top-k 1–500).
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class KeyRequest(BaseModel):
    provider: str
    key: str = Field(max_length=500)


class CustomVoiceRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    language: str = "English"
    speaker: str = "Aiden"
    instruct: Optional[str] = ""
    model: str = "custom-1.7B"
    max_new_tokens: int = Field(2048, ge=128, le=4096)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=1, le=500)
    seed: Optional[int] = None


class VoiceDesignRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    language: str = "English"
    instruct: str = Field(min_length=3, max_length=2000)
    max_new_tokens: int = Field(2048, ge=128, le=4096)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=1, le=500)
    seed: Optional[int] = None


class DialogueRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    speaker_a: str = "Aiden"
    speaker_b: str = "Ryan"
    language: str = "English"
    model: str = "custom-1.7B"
    instruct: Optional[str] = ""
    max_new_tokens: int = Field(2048, ge=128, le=4096)


class StitchRequest(BaseModel):
    filenames: List[str]
    prefix: str = Field("stitch", max_length=20)


class ClientError(BaseModel):
    msg: str = Field(max_length=2000, default="")
    url: str = Field(max_length=500, default="")
    line: str = Field(max_length=50, default="")
