from pydantic import BaseModel,EmailStr,Field
from typing import Optional 

class user(BaseModel):
    name:str = "enter your name"
    """using EmailStr if u assign a default value in EmailStr there is not error"""
    email: EmailStr  = Field(default="acb",description="pip install email-validator, used for email validation ")
    age:Optional[int] = None
    number: int = Field(gt=0,le=10,default=7,description="LLM can generate a structured output by looking at the description")

user1 = user(email="acb");
print(user1)