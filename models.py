from pydantic import BaseModel

class Context(BaseModel):
    emotion: str
    transcript: str