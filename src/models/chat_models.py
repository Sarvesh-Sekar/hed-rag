from pydantic import BaseModel, FilePath, field_validator
from pathlib import Path


    
class APIResponse(BaseModel):
    status: str
    message: str
    data: dict | None = None

class QueryRequest(BaseModel):
    query: str