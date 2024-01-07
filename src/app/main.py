import numpy as np
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel

THRESHOLD = 0.5

app = FastAPI()

class ToxicCommentModal(BaseModel):
    input_text : str
    response: list


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/make_inference")
def make_inference(input_text: ToxicCommentModal):
    response = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
    
    return {"response": response, "input_text": ToxicCommentModal.input_text}
