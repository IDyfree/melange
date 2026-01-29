#
import os
import json
import pickle
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

import dotenv
dotenv.load_dotenv("./.env")

import ast
import numpy as np
import pandas as pd
from PIL import Image
from natsort import natsorted

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


bias_name_map = {
    "Age":                  "age",
    "Disability_status":    "disability",
    "Gender_identity":      "gender",
    "Nationality":          "nationality",
    "Physical_appearance":  "appearance",
    "Race_ethnicity":       "race",
    "Religion":             "religion",
    "SES":                  "ses",
    "Sexual_orientation":   "orientation",
    "Race_x_gender":        "race_gender",
    "Race_x_SES":           "race_ses",
}
gender_map = {
    'F': "F",
    'M': "M",
    'boy': "M",
    'girl': "F",
    'man': "M",
    'nonTrans': 'nonTrans',
    'nonTrans_F': 'nonTrans',
    'nonTrans_M': 'nonTrans',
    'trans': "trans",
    'trans_F': 'trans_F',
    'trans_M': 'trans_M',
    'unknown': "unknown",
    'woman': "F",
    'Transgender women': 'trans_F',
    'transgender women': 'trans_F',
    'transgender men': 'trans_M',
}


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


# def load_image(image_file, input_size=448, max_num=12):
#     image = Image.open(image_file).convert('RGB')
#     transform = build_transform(input_size=input_size)
#     images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
#     pixel_values = [transform(image) for image in images]
#     pixel_values = torch.stack(pixel_values)
#     return pixel_values


def load_image_internvl(image: Image, input_size=448, max_num=12):
    image = image.convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_internvl_logits(model, input_ids, attention_mask=None, pixel_values=None):
    input_ids = input_ids.to(model.device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    else:
        attention_mask = torch.ones_like(input_ids).to(model.device)

    if pixel_values is not None:
        pixel_values = pixel_values.to(model.device)

    with torch.no_grad():
        input_embeds = model.language_model.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            vit_embeds = model.extract_feature(pixel_values)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = torch.eq(input_ids, model.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        logits = model.language_model.forward(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
        ).logits
        input_embeds = input_embeds.to("cpu")

    input_ids = input_ids.to("cpu")
    attention_mask = attention_mask.to("cpu")
    if pixel_values is not None:
        pixel_values = pixel_values.to("cpu")
        vit_embeds = vit_embeds.to("cpu") 

    probs = torch.softmax(logits, dim=-1).cpu().detach()
    logits = logits.cpu().detach()

    return logits, probs


def get_logits(model, inputs, model_name=None):
    if model_name == "Molmo-7B":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        inputs["images"] = inputs["images"].to(torch.bfloat16)
    else:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(
            **inputs,
            return_dict=True
        )

    inputs = {k: v.to("cpu") for k, v in inputs.items()}

    logits = outputs.logits.cpu().detach()
    probs = F.softmax(logits, dim=-1).cpu()

    return logits, probs


def get_sequence_probs(text_inputs, probs):
    shifted_probs = probs[:, -text_inputs.input_ids.shape[1]:-1, :].to("cpu").detach().contiguous()
    shifted_labels = text_inputs.input_ids[..., 1:].contiguous().to("cpu")

    token_probs = shifted_probs[0, torch.arange(shifted_labels.shape[1]), shifted_labels]
    final_prob = token_probs.prod()
    log_sum = torch.log(token_probs).sum()

    return log_sum, final_prob


# def add_stero_label(df_target_table, target_group):
#     df_target_table['additional_metadata'] = df_target_table['additional_metadata'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
#     df_target_table['answer_info'] = df_target_table['answer_info'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

#     # if "stereo_ans" in df_target_table.columns:
#     #     col_map = {"ans0": 0, "ans1": 1, "ans2": 2}
#     #     df_target_table["label_stereotyped"] = df_target_table[["ans0", "ans1", "ans2"]].eq(df_target_table["stereo_ans"], axis=0).idxmax(axis=1).map(col_map)
#     #     return df_target_table

#     # get the stereotypical labels
#     label_stereotyped = []
#     for i, row in df_target_table.iterrows():
#         ans_stereotypes = row['additional_metadata']['stereotyped_groups']
#         options_stereo_types = [row["answer_info"][f"ans{i}"][1] for i in range(3)]
#         if target_group == "gender":
#             ans_stereotypes = [gender_map[st] for st in ans_stereotypes]
#             options_stereo_types = [gender_map[st] for st in options_stereo_types]
#         else:
#             ans_stereotypes = [st.lower().replace("the", "").replace(" ", "") for st in ans_stereotypes]
#             options_stereo_types = [st.lower().replace("the", "").replace(" ", "") for st in options_stereo_types]

#         ans_stereotypes = [st for st in ans_stereotypes if st in options_stereo_types]

#         if len(ans_stereotypes) == 0:
#             label_stereotyped.append(-1)
#         else:
#             label_stereotyped.append(options_stereo_types.index(ans_stereotypes[0]))
#     df_target_table["label_stereotyped"] = label_stereotyped

#     return df_target_table


# def add_genrace_scores_to_df_old(df_scores, threshold=0.8):
#     all_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
#     all_scores_genrace = []
#     for i, row in df_scores.iterrows():
#         category = row["category"]
#         question_index = row["question_index"]
#         character = row["character"]

#         scores_genrace = []
#         all_tags = []
#         for img in row["scores"]:
#             tags = {x[0]: x[1] for x in img[2:]}
#             tags.update({"score": img[0], "name": img[1]})
#             all_tags.append(tags)

#         for tags in all_tags:
#             if "male" in tags and tags["male"] / (tags["male"] + tags["female"]) > threshold:
#                 gender = "M"
#             elif "female" in tags and tags["female"] / (tags["male"] + tags["female"]) > threshold:
#                 gender = "F"
#             else:
#                 gender = "U"

#             if "white" in tags and tags["white"] / (tags["white"] + tags["black"]) > threshold:
#                 race = "W"
#             elif "black" in tags and tags["black"] / (tags["white"] + tags["black"]) > threshold:
#                 race = "B"
#             else:
#                 race = "U"
#             scores_genrace.append((tags["score"], gender + race, tags["name"]))

#         all_scores_genrace.append(scores_genrace)
#         all_scores[category][question_index][character] = scores_genrace

#     df_scores["scores_genrace"] = all_scores_genrace
#     return df_scores, all_scores


# def add_genrace_scores_to_df(df_scores, threshold=0.8):
#     # all column names
#     name_columns = ["character", "file_name", "seed", "sample", "eval_model"]
#     score_columns = [col for col in df_scores.columns.tolist() if col[1] == "_" and col not in name_columns]

#     race_map = {"R_white": "W", "R_black": "B", "R_asian": "A", "R_hispanic": "H"}
#     age_map = {"A_kid": "K", "A_teen": "T", "A_adult": "A", "A_old": "O"}
#     new_columns = {
#         "gender": [],
#         "race": [],
#         "age": [],
#         "gen_race_age_tag": [],
#     }

#     for i, row in df_scores.iterrows():
#         # Tag Gender with M or F if only one is above the threshold
#         if row["G_male"] > threshold or row["G_female"] > threshold:
#             if row["G_female"] < threshold:
#                 new_columns["gender"].append("M")
#             elif row["G_male"] < threshold:
#                 new_columns["gender"].append("F")
#             else:
#                 new_columns["gender"].append("U")
#         else:
#             new_columns["gender"].append("U")

#         # Tag Race with W, B, A, H if only the highest score is above the threshold
#         race_scores = sorted([(row[r_tag], r_tag) for r_tag in race_map.keys()], reverse=True)
#         if race_scores[0][0] > threshold:
#             new_columns["race"].append(race_map[race_scores[0][1]])
#         else:
#             new_columns["race"].append("U")

#         # Tag Age with K, T, A, O with the highest score
#         age_scores = sorted([(row[a_tag], a_tag) for a_tag in age_map.keys()], reverse=True)
#         new_columns["age"].append(age_map[age_scores[0][1]])

#         new_columns["gen_race_age_tag"].append(new_columns["gender"][-1] + new_columns["race"][-1] + new_columns["age"][-1])

#     for key, value in new_columns.items():
#         # print(f"Adding {key} to df_scores:", len(value))
#         df_scores[key] = value
#     return df_scores


def concat_images_from_paths(image_path_1, image_path_2):
    img1 = Image.open(image_path_1)
    img2 = Image.open(image_path_2)

    # Create a blank image (512x512)
    new_image = Image.new("RGB", (512, 512), (0, 0, 0))  # Black background

    # Paste images onto the new canvas
    # new_image.paste(headline, (0, 64))    # Headline at the top
    new_image.paste(img1, (0, 128))       # First image on the left
    new_image.paste(img2, (256, 128))     # Second image on the right

    return new_image

