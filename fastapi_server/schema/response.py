from typing import List, Optional
from pydantic import BaseModel

class Transcript(BaseModel):
    raw: str
    itn: str

class Intent(BaseModel):
    recommended_tag: str
    original_tag: str
    probability: float

class Entity(BaseModel):
    tag: str
    substring: str
    start_index: int
    end_index: int
    extracted_value: str

class InferenceResult(BaseModel):
    input_id: str
    transcript: Transcript
    intent: Intent
    entities: List[Entity]

class ResponseStatus(BaseModel):
    success: bool
    message: Optional[str] = ""

class InferenceResponse(BaseModel):
    output: Optional[List[InferenceResult]] = []
    status: ResponseStatus
