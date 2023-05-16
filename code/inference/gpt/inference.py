import ast
from typing import Any, Dict, Union
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MAX_LENGTH = 50
DO_SAMPLE = True


def model_fn(model_dir: str) -> Dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)

    return {
        "tokenizer": tokenizer,
        "model": model,
    }


def predict_fn(
    data: Dict[str, Union[int, float, str]],
    model: Dict[str, Any],
) -> Dict[str, Union[int, float, str]]:
    tokenizer, model = model["tokenizer"], model["model"]

    use_magic_prompt = data.pop("use_magic_prompt", "False")
    use_magic_prompt = ast.literal_eval(use_magic_prompt)

    if use_magic_prompt:
        prompt = data.pop("prompt")
        prompt = prompt if prompt[-1] == "," else prompt + ","
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output = model.generate(input_ids, max_length=MAX_LENGTH, do_sample=DO_SAMPLE)
        prompt = tokenizer.decode(output[0], skip_special_tokens=True)
        data["prompt"] = prompt

        return data
    return data
