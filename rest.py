from fastapi import FastAPI
from pydantic import BaseModel

from service import Predictor
from utils_prompt_2 import Problem


class PredictionResult(BaseModel):
    answer: str
    explanation: str


predictor = Predictor(
    explanation_model='models/MM-CoT-UnifiedQA-base-Rationale',
    answer_model='models/MM-CoT-UnifiedQA-base-Answer',
    explanation_prompt_format='QCM-LE',
    answer_prompt_format='QCMG-A',
    use_caption=False,
    options=["A", "B", "C", "D", "E"],
    input_len=512,
    img_type='detr'
)

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/get_answer_and_explanation")
def get_answer_and_explanation(p: Problem):
    answer, explanation = predictor.predict(p)
    return PredictionResult(answer=answer, explanation=explanation)
