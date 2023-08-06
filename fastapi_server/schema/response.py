from typing import List, Optional
from pydantic import BaseModel

# class Transcript(BaseModel):
#     raw: str
#     itn: str

# class Intent(BaseModel):
#     recommended_tag: str
#     original_tag: str
#     probability: float

class Entity(BaseModel):
    entity: str
    word: str
    start: int
    end: int
    value: str

class InferenceResult(BaseModel):
    id: str
    source: str
    entities: Optional[List[Entity]] = []
    intent: Optional[str] = ""

class ResponseStatus(BaseModel):
    success: bool
    message: Optional[str] = ""

class InferenceResponse(BaseModel):
    output: Optional[List[InferenceResult]] = []
    status: ResponseStatus
