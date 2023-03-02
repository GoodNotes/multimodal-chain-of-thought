from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from pathlib import Path
import uuid

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

from detr_model.detr import DETRModel
detr_model = DETRModel()


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/get_answer_and_explanation")
def get_answer_and_explanation(p: Problem):
    answer, explanation = predictor.predict(p)
    return PredictionResult(answer=answer, explanation=explanation)


@app.post("/get_answer_and_explanation_w_image")
async def get_answer_and_explanation_w_image(p: Problem, image_file: UploadFile = File(...)):
    new_fp = f'/tmp/{uuid.uuid4()}.png'

    image_file.filename = new_fp
    contents = await image_file.read()

    with open(new_fp, "wb") as f:
        f.write(contents)

    answer, explanation = predictor.predict(p, image_path=new_fp)
    print(answer)
    print(explanation)
    return PredictionResult(answer=answer, explanation=explanation)
