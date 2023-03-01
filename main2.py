import argparse
from transformers import T5Tokenizer
from model import T5ForMultimodalGeneration
from typing import Tuple
import numpy as np
import torch
import random
import re

from utils_data_2 import img_shape, ScienceQAInputEncoder
from utils_prompt_2 import Problem


def extract_ans(ans):
    pattern = re.compile(r'The answer is \(([A-Z])\)')
    res = pattern.findall(ans)

    if len(res) == 1:
        answer = res[0]  # 'A', 'B', ...
    else:
        answer = "FAILED"
    return answer


def predict(problem, args) -> Tuple[str, str]:
    explanation = predict_explanation(problem, args)
    problem.solution = explanation
    answer = predict_answer(problem, args)
    return answer, explanation


def predict_explanation(problem, args) -> str:
    tokenizer = T5Tokenizer.from_pretrained(args.model_1)
    patch_size = img_shape[args.img_type]
    padding_idx = tokenizer._convert_token_to_id(tokenizer.pad_token)

    input_encoder = ScienceQAInputEncoder(tokenizer, args)
    model = T5ForMultimodalGeneration.from_pretrained(
        args.model_1,
        patch_size=patch_size,
        padding_idx=padding_idx,
        save_dir=args.model_1
    )
    model_input = input_encoder.encode_input(problem, None)
    results = model.generate(
        **model_input,
        do_sample=True,
        top_k=30,
        top_p=0.95,
        max_length=512
    )
    return tokenizer.batch_decode(results, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]


def predict_answer(problem, args) -> str:
    tokenizer = T5Tokenizer.from_pretrained(args.model_2)
    patch_size = img_shape[args.img_type]
    padding_idx = tokenizer._convert_token_to_id(tokenizer.pad_token)

    input_encoder = ScienceQAInputEncoder(tokenizer, args)
    model = T5ForMultimodalGeneration.from_pretrained(
        args.model_2,
        patch_size=patch_size,
        padding_idx=padding_idx,
        save_dir=args.model_2
    )

    model_input = input_encoder.encode_input(problem, None)
    results = model.generate(
        **model_input,
        do_sample=True,
        top_k=30,
        top_p=0.95,
        max_length=512
    )
    return tokenizer.batch_decode(results, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_1', type=str, default='models/MM-CoT-UnifiedQA-base-Rationale')
    parser.add_argument('--model_2', type=str, default='models/MM-CoT-UnifiedQA-base-Answer')
    parser.add_argument('--input_len', type=int, default=512)
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    parser.add_argument('--use_caption', action='store_true', help='use image captions or not')
    parser.add_argument('--prompt_format', type=str, default='QCM-LE', help='prompt format template',
                        choices=['QCM-A', 'QCM-LE', 'QCMG-A', 'QCM-LEA', 'QCM-ALE'])
    parser.add_argument('--img_type', type=str, default='detr', choices=['detr', 'clip', 'resnet'],
                        help='type of image features')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    _args = parser.parse_args()

    random.seed(_args.seed)
    torch.manual_seed(_args.seed)  # pytorch random seed
    np.random.seed(_args.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    p = Problem(
        question='Which of the following food is vegan friendly?',
        choices=[
            "Steak",
            "Ice cream",
            "Tofu",
            "Chicken breast"
        ]
    )
    _answer, _explanation = predict(p, _args)
    print(f'{_answer}\n{_explanation}')
