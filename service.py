from transformers import T5Tokenizer
from model import T5ForMultimodalGeneration
from typing import Tuple
import torch

from detr_model.detr import DETRModel
from utils_data_2 import img_shape, ScienceQAInputEncoder
from utils_prompt_2 import Problem


class Predictor:
    def __init__(
            self,
            explanation_model,
            answer_model,
            explanation_prompt_format,
            answer_prompt_format,
            use_caption,
            options,
            input_len,
            img_type
    ):
        patch_size = img_shape[img_type]
        detr_model = DETRModel()

        self.tokeniser_1 = T5Tokenizer.from_pretrained(explanation_model)
        padding_idx = self.tokeniser_1._convert_token_to_id(self.tokeniser_1.pad_token)

        self.input_encoder_1 = ScienceQAInputEncoder(
            self.tokeniser_1,
            explanation_prompt_format,
            use_caption,
            options,
            input_len,
            img_type,
            detr_model
        )
        self.model_1 = T5ForMultimodalGeneration.from_pretrained(
            explanation_model,
            patch_size=patch_size,
            padding_idx=padding_idx,
            save_dir=explanation_model
        )
        self.model_1.eval()

        self.tokenizer_2 = T5Tokenizer.from_pretrained(answer_model)
        padding_idx = self.tokenizer_2._convert_token_to_id(self.tokenizer_2.pad_token)

        self.input_encoder_2 = ScienceQAInputEncoder(
            self.tokenizer_2,
            answer_prompt_format,
            use_caption,
            options,
            input_len,
            img_type,
            detr_model
        )
        self.model_2 = T5ForMultimodalGeneration.from_pretrained(
            answer_model,
            patch_size=patch_size,
            padding_idx=padding_idx,
            save_dir=answer_model
        )
        self.model_2.eval()

    def _predict_explanation(self, problem: Problem, image_path: str = None):
        model_input = self.input_encoder_1.encode_input(problem, image_path=image_path, le_data=None)
        results = self.model_1.generate(
            **model_input,
            do_sample=True,
            top_k=30,
            top_p=0.95,
            max_length=512
        )
        return self.tokeniser_1.batch_decode(results, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

    def _predict_answer(self, problem: Problem, image_path: str = None):
        model_input = self.input_encoder_2.encode_input(problem, image_path=image_path, le_data=problem.solution)
        results = self.model_2.generate(
            **model_input,
            do_sample=True,
            top_k=30,
            top_p=0.95,
            max_length=512
        )
        return self.tokenizer_2.batch_decode(results, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

    def predict(self, problem, image_path: str = None) -> Tuple[str, str]:
        explanation = self._predict_explanation(problem, image_path)
        problem.solution = explanation
        answer = self._predict_answer(problem, image_path)
        return answer, explanation


if __name__ == '__main__':
    _explanation_model = 'models/MM-CoT-UnifiedQA-base-Rationale'
    _answer_model = 'models/MM-CoT-UnifiedQA-base-Answer'
    _explanation_prompt_format = 'QCM-LE'
    _answer_prompt_format = 'QCMG-A'
    _use_caption = False
    _options = ["A", "B", "C", "D", "E"]
    _input_len = 512
    _img_type = 'detr'

    seed = 42

    torch.backends.cudnn.deterministic = True

    predictor = Predictor(
        explanation_model=_explanation_model,
        answer_model=_answer_model,
        explanation_prompt_format=_explanation_prompt_format,
        answer_prompt_format=_answer_prompt_format,
        use_caption=_use_caption,
        options=_options,
        input_len=_input_len,
        img_type=_img_type
    )

    p = Problem(
        question='Which of these states is farthest north?',
        choices=[
            "West Virginia",
            "Louisiana",
            "Arizona",
            "Oklahoma"
        ]
    )

    _image_path = '/Users/hugochu/PycharmProjects/multimodal-chain-of-thought/data/raw_images/train/1/image.png'

    _answer, _explanation = predictor.predict(p, image_path=_image_path)
    print(f'{_answer}\n{_explanation}')
