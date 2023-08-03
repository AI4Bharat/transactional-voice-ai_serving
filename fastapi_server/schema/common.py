from typing import List, Optional
from pydantic import BaseModel, AnyHttpUrl

from enum import Enum

class AudioFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    FLV = "flv"
    OGG = "ogg"

class Language(BaseModel):
    sourceLanguage: str
    # sourceScriptCode: Optional[str] = ""

class Audio(BaseModel):
    audioContent: Optional[str] = ""
    audioUri: Optional[str] = ""
