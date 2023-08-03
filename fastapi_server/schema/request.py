from typing import List, Optional
from pydantic import BaseModel

from .common import Audio, AudioFormat, Language

class AudioConfig(BaseModel):
    audioFormat: AudioFormat
    language: Language
    encoding: Optional[str] = "base64"

class ControlConfig(BaseModel):
    dataTracking: bool = True

class InferenceRequest(BaseModel):
    audio: List[Audio]
    config: AudioConfig
    controlConfig: Optional[ControlConfig] = ControlConfig()
