#
import os
from pathlib import Path
from collections import defaultdict
from typing import List, Union, Dict, Any, Tuple
from natsort import natsorted
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import dotenv
dotenv.load_dotenv("./.env")

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from transformers import AutoModel, AutoTokenizer
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import AutoModelForCausalLM
from transformers import Qwen2VLForConditionalGeneration
from transformers.image_processing_utils import BatchFeature
from qwen_vl_utils import process_vision_info, fetch_image

from utils import load_image_internvl


IMG_START_TOKEN='<img>'
IMG_END_TOKEN='</img>'
IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'


model2name_map = {
    "OpenGVLab/InternVL2_5-8B": "InternVL2_5-8B",
    "OpenGVLab/InternVL3-8B": "InternVL3-8B",
    "Qwen/Qwen2-VL-7B-Instruct": "Qwen2-VL-7B",
    "allenai/Molmo-7B-D-0924": "Molmo-7B",
    "microsoft/Phi-3.5-vision-instruct": "Phi-3.5",
    "llava-hf/llava-1.5-7b-hf": "Llava-1.5-7B",
}
model2id_map = {v: k for k, v in model2name_map.items()}
model2id_map.update({k: k for k, v in model2name_map.items()})
model2id_map.update({
    "InternVL": "OpenGVLab/InternVL2_5-8B",
    "InternVL3": "OpenGVLab/InternVL3-8B",
    "Qwen": "Qwen/Qwen2-VL-7B-Instruct",
    "Molmo": "allenai/Molmo-7B-D-0924",
    "Phi": "microsoft/Phi-3.5-vision-instruct",
    "Llava": "llava-hf/llava-1.5-7b-hf",
})

families = [
    "InternVL",
    "Qwen",
    "Molmo-7B-D",
    "Phi-3.5",
    "llava-1.5",
]
model2family_map = {}
for model_id, model_name in model2name_map.items():
    for family in families:
        if family in model_name:
            model2family_map[model_name] = family
            break


def load_model(model_id="OpenGVLab/InternVL2_5-8B", device="cuda"):
    if device == "auto":
        kwargs = {"device_map": "auto"}
    else:
        kwargs = {}

    if "InternVL" in model2id_map[model_id]:
        processor = AutoTokenizer.from_pretrained(model2id_map[model_id], trust_remote_code=True, use_fast=True)
        processor.padding_side = 'left'

        model = AutoModel.from_pretrained(
            model2id_map[model_id],
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            **kwargs
            ).eval()

        img_context_token_id = processor.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        model.img_context_token_id = img_context_token_id
    elif "Qwen2-VL" in model_id:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model2id_map[model_id],
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            # device_map="auto",
            **kwargs
            ).eval()
        # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        min_pixels = 4*28*28
        max_pixels = 8*28*28
        processor = AutoProcessor.from_pretrained(model2id_map[model_id], min_pixels=min_pixels, max_pixels=max_pixels, use_fast=True)
        processor.tokenizer.padding_side = 'left'
    elif model2name_map[model2id_map[model_id]] in ["Molmo-7B", "Phi-3.5"]:
        if "Phi" in model2id_map[model_id]:
            kwargs["_attn_implementation"] = 'flash_attention_2'
            # kwargs["_attn_implementation"] = 'eager'

        # Note: set _attn_implementation='eager' if you don't have flash_attn installed
        model = AutoModelForCausalLM.from_pretrained(
            model2id_map[model_id],
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            **kwargs
            ).eval()

        # for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
        processor = AutoProcessor.from_pretrained(
            model2id_map[model_id],
            trust_remote_code=True,
            use_fast=True,
            num_crops=4
        )
        processor.tokenizer.padding_side = 'left'
    elif model2name_map[model2id_map[model_id]] == "Llava-1.5-7B":
        model = LlavaForConditionalGeneration.from_pretrained(
            model2id_map[model_id],
            trust_remote_code=True,
            torch_dtype=torch.float16,
            **kwargs
            ).eval()
        processor = AutoProcessor.from_pretrained(model2id_map[model_id], trust_remote_code=True, use_fast=True)
        processor.tokenizer.padding_side = 'left'

    if device != "auto":
        model = model.to(device)

    return model, processor


# Code for processing the prompts in batch
def process_batch_molmo(
    processor: AutoProcessor,
    texts: List[str],
    images_list: List[List[Image.Image]]
) -> Dict[str, torch.Tensor]:
    """
    Process in batch.
    
    Args:
        processor: The original processor.
        texts: List of text inputs
        images_list: List of lists containing PIL images.
        
    Returns:
        Dict with padded input_ids, images, image_input_idx, image_masks.
    """
    batch_size = len(texts)
    tokens_list = []
    for text in texts:
        tokens = processor.tokenizer.encode(" " + text, add_special_tokens=False)
        tokens_list.append(tokens)
    images_arrays_list = []
    image_idxs_list = []
    for images in images_list:
        if images:
            image_arrays = []
            for image in images:
                if isinstance(image, Image.Image):
                    image = image.convert("RGB")
                    image = ImageOps.exif_transpose(image)
                    image_arrays.append(np.array(image))
                else:
                    assert len(image.shape) == 3 and image.shape[-1] == 3
                    image_arrays.append(image.astype(np.uint8))
            images_arrays_list.append(image_arrays)
            image_idx = [-1] * len(image_arrays)
            image_idxs_list.append(image_idx)
        else:
            images_arrays_list.append(None)
            image_idxs_list.append(None)
    images_kwargs = {
        "max_crops": 12,
        "overlap_margins": [4, 4],
        "base_image_input_size": [336, 336],
        "image_token_length_w": 12,
        "image_token_length_h": 12,
        "image_patch_size": 14,
        "image_padding_mask": True,
    }
    outputs_list = []
    for i in range(batch_size):
        tokens = tokens_list[i]
        images = images_arrays_list[i]
        image_idx = image_idxs_list[i]
        out = processor.image_processor.multimodal_preprocess(
            images=images,
            image_idx=image_idx,
            tokens=np.asarray(tokens).astype(np.int32),
            sequence_length=1536,
            image_patch_token_id=processor.special_token_ids["<im_patch>"],
            image_col_token_id=processor.special_token_ids["<im_col>"],
            image_start_token_id=processor.special_token_ids["<im_start>"],
            image_end_token_id=processor.special_token_ids["<im_end>"],
            **images_kwargs,
        )
        outputs_list.append(out)

    batch_outputs = {}
    for key in outputs_list[0].keys():
        tensors = [torch.from_numpy(out[key]) for out in outputs_list]
        batch_outputs[key] = torch.nn.utils.rnn.pad_sequence(
            tensors, batch_first=True, padding_value=-1
        )
    bos = processor.tokenizer.bos_token_id or processor.tokenizer.eos_token_id
    batch_outputs["input_ids"] = torch.nn.functional.pad(
        batch_outputs["input_ids"], (1, 0), value=bos
    )
    if "image_input_idx" in batch_outputs:
        image_input_idx = batch_outputs["image_input_idx"]
        batch_outputs["image_input_idx"] = torch.where(
            image_input_idx < 0, image_input_idx, image_input_idx + 1
        )
    return batch_outputs


def stack_inputs_for_phi(inputs: list[BatchFeature], pad_token_id: int) -> BatchFeature:
    def pad_left(seqs: list[torch.Tensor], pad_token_id: int) -> torch.Tensor:
        """Example: pad_left([[1, 2], [3, 4, 5]], pad_token_id=0) -> [[0, 1, 2], [3, 4, 5]]"""
        max_len = max(len(seq) for seq in seqs)
        padded = torch.full((len(seqs), max_len), pad_token_id)
        for i, seq in enumerate(seqs):
            padded[i, -len(seq) :] = seq
        return padded

    listof_input_ids = [i.input_ids[0] for i in inputs]
    new_input_ids = pad_left(listof_input_ids, pad_token_id=pad_token_id)
    data = dict(
        pixel_values=torch.cat([i.pixel_values for i in inputs], dim=0),
        image_sizes=torch.cat([i.image_sizes for i in inputs], dim=0),
        input_ids=new_input_ids,
        attention_mask=(new_input_ids != pad_token_id).long(),
    )
    new_inputs = BatchFeature(data).to("cuda")
    return new_inputs


def process_inputs(model_id, processor, tokenizer, queries, images=None, num_image_token=256):
    if "InternVL" in model2id_map[model_id] and images is not None:
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token + IMG_END_TOKEN
        queries = [q.replace('<image>', image_tokens, 1) for q in queries]
        inputs = tokenizer(queries, padding=True, return_tensors='pt')

        images = [load_image_internvl(image) for image in images]
        inputs["pixel_values"] = torch.cat(images, dim=0).to(torch.bfloat16)
    elif "Qwen" in model2id_map[model_id] and images is not None:
        image_inputs = [fetch_image({"image": image}) for image in images]

        inputs = processor(
            text=queries,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
    elif model2id_map[model_id] == "allenai/Molmo-7B-D-0924" and images is not None:
        images = [[image] for image in images]
        inputs = process_batch_molmo(processor, queries, images)
        inputs["images"] = inputs["images"].to(torch.bfloat16)
    elif model2id_map[model_id] == "llaava-hf/llava-1.5-7b-hf" and images is not None:
        inputs = processor(
            text=queries,
            images=images,
            padding=True,
            return_tensors="pt").to(torch.float16)
    elif model2id_map[model_id] == "microsoft/Phi-3.5-vision-instruct" and images is not None:
        listof_inputs = [processor(q, [im], return_tensors="pt") for q, im in zip(queries, images)]
        inputs = stack_inputs_for_phi(listof_inputs, pad_token_id=tokenizer.pad_token_id)
    else:
        inputs = tokenizer(queries, padding=True, return_tensors="pt")

    return inputs
