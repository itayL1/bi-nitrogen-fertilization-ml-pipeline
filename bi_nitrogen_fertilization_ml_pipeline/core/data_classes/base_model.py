from pydantic import BaseModel as PydanticBaseModel, Extra


class BaseModel(PydanticBaseModel):
    class Config:
        extra = Extra.forbid
        allow_mutation = True
        arbitrary_types_allowed = True
