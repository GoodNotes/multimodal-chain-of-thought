import torch
import numpy as np

from utils_prompt_2 import Problem, build_model_input


img_shape = {
    "resnet": (512, 2048),
    "clip": (49, 2048),
    "detr": (100, 256),
}


class ScienceQAInputEncoder:
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def encode_input(self, problem: Problem, le_data=None):
        source_text = build_model_input(
            problem,
            prompt_format=self.args.prompt_format,
            use_caption=self.args.use_caption,
            options_format=self.args.options,
            le_data=le_data
        )

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.args.input_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()

        if not problem.image:
            shape = img_shape[self.args.img_type]
            image = np.zeros(shape)
        else:
            image = problem.image

        image_ids = torch.tensor(image, dtype=torch.float32).squeeze()

        return {
            "input_ids": torch.unsqueeze(source_ids, dim=0),
            "attention_mask": torch.unsqueeze(source_mask, dim=0),
            "image_ids": torch.unsqueeze(image_ids, dim=0)
        }
