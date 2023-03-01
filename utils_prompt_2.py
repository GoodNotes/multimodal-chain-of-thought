from pydantic import BaseModel
from typing import Optional, List, Any


class Problem(BaseModel):
    question: str
    hint: Optional[str] = None
    caption: Optional[str] = None
    choices: List[str]
    lecture: Optional[str] = None
    solution: Optional[str] = None
    image: Optional[Any] = None


class QCMInput(BaseModel):
    question: str
    context: str
    choice_txt: str

    @staticmethod
    def from_problem(p: Problem, use_caption, options_format):
        return QCMInput(
            question=p.question,
            context=get_context_text(p, use_caption),
            choice_txt=get_choice_text(p, options_format)
        )


def get_context_text(problem, use_caption):
    txt_context = problem.hint if problem.hint else ""
    img_context = problem.caption if use_caption and problem.caption else ""
    context = " ".join([txt_context, img_context]).strip()
    if context == "":
        context = "N/A"
    return context


def get_choice_text(problem, options_format):
    choices = problem.choices
    choice_list = []
    for i, c in enumerate(choices):
        choice_list.append("({}) {}".format(options_format[i], c))
    choice_txt = " ".join(choice_list)
    return choice_txt


def build_model_input(problem, prompt_format, use_caption, options_format, le_data=None):
    input_format, output_format = prompt_format.split("-")
    qcm_input = QCMInput.from_problem(
        problem,
        use_caption=use_caption,
        options_format=options_format
    )

    input = ''
    if input_format == 'QCM':
        input = f"Question: {qcm_input.question}\nContext: {qcm_input.context}\nOptions: {qcm_input.choice_txt}\n"

    elif input_format == "QCMG":
        input = f"Question: {qcm_input.question}\nContext: {qcm_input.context}\n" \
                f"Options: {qcm_input.choice_txt}\n{le_data}\n"

    if output_format == 'A':
        text = input + f'Answer:'

    else:
        text = input + f'Solution:'

    text = text.replace("  ", " ").strip()
    return " ".join(text.split())
