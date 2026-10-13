from pydantic import BaseModel, Field, ConfigDict
from typing import Literal


class InputEmailSchema(BaseModel):
        email_address: str
        subject: str
        body: str

class ClassifierSchema(BaseModel):
        model_config = ConfigDict(extra="forbid")
        category: Literal["Order Status", "Refund Request", "Complaint", "Product Question", "Others", "Spam"]
        priority: Literal["Critical","High","Medium","Low"]
        spam_probability: float = Field(ge=0, le=1)
        summary: str

class ClassifierOutputSchema(BaseModel):
        model_config = ConfigDict(extra="ignore")
        email_address: str
        subject: str
        body: str
        summary: str
        category: str
        priority: str
        spam_probability: float = Field(ge=0, le=1)

class ResponderSchema(BaseModel):
      model_config = ConfigDict(extra="forbid")
      reply: str
      escalate: bool
      close: bool        

class ResponderOutputSchema(BaseModel):
      model_config = ConfigDict(extra="ignore")
      email_address: str
      subject: str
      body: str
      summary: str
      reply: str      
      category: str
      priority: str
      spam_probability: float = Field(ge=0, le=1)
      escalate: bool
      close: bool