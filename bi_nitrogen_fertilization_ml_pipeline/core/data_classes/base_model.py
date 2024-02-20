from pydantic import BaseModel as PydanticBaseModel, Extra


class BaseModel(PydanticBaseModel):
    class Config:
        allow_mutation = False
        extra = Extra.forbid
