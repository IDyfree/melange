import sys
sys.path.append("../")

import io
import json
import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image
import natsort
import os

import dotenv
dotenv.load_dotenv("../.env")

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

import transformers

from utils import *
from model_utils import *
from query_templates import *


model_name = "OpenGVLab/InternVL2_5-8B"
num_images_per_query = 10

model_id = model2id_map[model_name]
model_name = model2name_map[model_id]

target_groups = ["age", "appearance", "gender", "race", "religion", "ses"]


for target_group in target_groups:
    print("processing", target_group)

    df_path = F"./group_tables/df_{target_group}.pkl"
    df_target_table = pd.read_pickle(df_path)
    df_target_table = add_stero_label(df_target_table, target_group)

    # add core_context
    df_scenario_path = F"./group_tables/scenarios/{target_group}_with_scenarios.csv"
    df_scenario = pd.read_csv(df_scenario_path)
    df_target_table["core_context"] = df_scenario["core_context"]

    generated_images_path = Path(F"~/projects/vlm_bias/bias_images/bias_{target_group}/generated_images_{target_group}")

    df_img_scores = pd.read_pickle(F"./all_image_scores_pickles/df_scores_{target_group}.pkl")
    df_img_scores, all_scores = add_genrace_scores_to_df(df_img_scores)

    # Create queries
    print("Creating queries...")
    count = 0
    all_rows_image_pairs = []
    for i, row in df_target_table.iterrows():
        context = row["context"].strip()
        question = row["question"].strip()
        character_path = generated_images_path / F"{target_group}_{row['question_index']}"

        characters = [row[F"ans{i}"] for i in range(3) if i != row["label"]]
        if len(characters) != 2:
            print("ERROR: Not 2 charcters in example.")

        character_map = {
            characters[0]: "Image on the left",
            characters[1]: "Image on the right",
            row[F"ans{row['label']}"].strip(): row[F"ans{row['label']}"].strip(),
            }
        options = [character_map[row[F"ans{i}"].strip()] for i in range(3)]

        char_1_images = all_scores[target_group][row["question_index"]][characters[0].lower()]
        char_2_images = all_scores[target_group][row["question_index"]][characters[1].lower()]
        all_image_pairs = []
        for char1_img in char_1_images[:25]:
            for char2_img in char_2_images[:25]:
                if (target_group in ["appearance", "ses"] and char1_img[1] == char2_img[1]) or \
                    (target_group in ["age", "gender"] and char1_img[1][1] == char2_img[1][1]) or \
                    (target_group in ["race", "religion"] and char1_img[1][0] == char2_img[1][0]):
                    all_image_pairs.append((char1_img[0] * char2_img[0], character_path / char1_img[2], character_path / char2_img[2]))

        all_image_pairs = sorted(all_image_pairs, key=lambda x: x[0], reverse=True)[:num_images_per_query]
        all_rows_image_pairs.append(all_image_pairs)
        count += len(all_image_pairs)
    df_target_table["image_path_pairs"] = all_rows_image_pairs

    df_enriched_path = F"./group_tables/df_{target_group}_enriched.pkl"
    df_target_table.to_pickle(df_enriched_path)

    print("Number of queries:", count)
