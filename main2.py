import argparse
from transformers import T5Tokenizer
from model import T5ForMultimodalGeneration
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


def predict(problem, args):
    tokenizer = T5Tokenizer.from_pretrained(args.model)

    patch_size = img_shape[args.img_type]
    padding_idx = tokenizer._convert_token_to_id(tokenizer.pad_token)

    input_encoder = ScienceQAInputEncoder(tokenizer, args)
    model = T5ForMultimodalGeneration.from_pretrained(
        args.model,
        patch_size=patch_size,
        padding_idx=padding_idx,
        save_dir=args.model
    )
    print(f'BLAH: {model.encoder.main_input_name}')
    model_input = input_encoder.encode_input(problem, None)

    results = model.generate(model_input)
    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/MM-CoT-UnifiedQA-base-Rationale')
    parser.add_argument('--input_len', type=int, default=512)
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    parser.add_argument('--use_caption', action='store_true', help='use image captions or not')
    parser.add_argument('--prompt_format', type=str, default='QCM-LE', help='prompt format template',
                        choices=['QCM-A', 'QCM-LE', 'QCMG-A', 'QCM-LEA', 'QCM-ALE'])
    parser.add_argument('--img_type', type=str, default='detr', choices=['detr', 'clip', 'resnet'],
                        help='type of image features')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)  # pytorch random seed
    np.random.seed(args.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    p = Problem(
        question='Which of these states is farthest north?',
        choices=[
            "West Virginia",
            "Louisiana",
            "Arizona",
            "Oklahoma"
        ]
    )
    predict(p, args)

