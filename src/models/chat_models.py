from pydantic import BaseModel, FilePath, field_validator
from pathlib import Path

class UploadRequestModel(BaseModel):
    file: FilePath

    @field_validator('file')
    @classmethod
    def check_extension(cls, v: Path):
        if v.suffix.lower() not in ['.pdf', '.docx','.txt']:
            raise ValueError('Only PDF, DOCX, and TXT files are allowed')
        return v
    
class APIResponse(BaseModel):
    status: str
    message: str
    data: dict | None = None

class QueryRequest(BaseModel):
    query: str