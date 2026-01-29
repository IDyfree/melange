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


def add_stero_label(df_target_table, target_group):
    df_target_table['additional_metadata'] = df_target_table['additional_metadata'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df_target_table['answer_info'] = df_target_table['answer_info'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # if "stereo_ans" in df_target_table.columns:
    #     col_map = {"ans0": 0, "ans1": 1, "ans2": 2}
    #     df_target_table["label_stereotyped"] = df_target_table[["ans0", "ans1", "ans2"]].eq(df_target_table["stereo_ans"], axis=0).idxmax(axis=1).map(col_map)
    #     return df_target_table

    # get the stereotypical labels
    label_stereotyped = []
    for i, row in df_target_table.iterrows():
        ans_stereotypes = row['additional_metadata']['stereotyped_groups']
        options_stereo_types = [row["answer_info"][f"ans{i}"][1] for i in range(3)]
        if target_group == "gender":
            ans_stereotypes = [gender_map[st] for st in ans_stereotypes]
            options_stereo_types = [gender_map[st] for st in options_stereo_types]
        else:
            ans_stereotypes = [st.lower().replace("the", "").replace(" ", "") for st in ans_stereotypes]
            options_stereo_types = [st.lower().replace("the", "").replace(" ", "") for st in options_stereo_types]

        ans_stereotypes = [st for st in ans_stereotypes if st in options_stereo_types]

        if len(ans_stereotypes) == 0:
            label_stereotyped.append(-1)
        else:
            label_stereotyped.append(options_stereo_types.index(ans_stereotypes[0]))
    df_target_table["label_stereotyped"] = label_stereotyped

    return df_target_table


def add_genrace_scores_to_df_old(df_scores, threshold=0.8):
    all_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    all_scores_genrace = []
    for i, row in df_scores.iterrows():
        category = row["category"]
        question_index = row["question_index"]
        character = row["character"]

        scores_genrace = []
        all_tags = []
        for img in row["scores"]:
            tags = {x[0]: x[1] for x in img[2:]}
            tags.update({"score": img[0], "name": img[1]})
            all_tags.append(tags)

        for tags in all_tags:
            if "male" in tags and tags["male"] / (tags["male"] + tags["female"]) > threshold:
                gender = "M"
            elif "female" in tags and tags["female"] / (tags["male"] + tags["female"]) > threshold:
                gender = "F"
            else:
                gender = "U"

            if "white" in tags and tags["white"] / (tags["white"] + tags["black"]) > threshold:
                race = "W"
            elif "black" in tags and tags["black"] / (tags["white"] + tags["black"]) > threshold:
                race = "B"
            else:
                race = "U"
            scores_genrace.append((tags["score"], gender + race, tags["name"]))

        all_scores_genrace.append(scores_genrace)
        all_scores[category][question_index][character] = scores_genrace

    df_scores["scores_genrace"] = all_scores_genrace
    return df_scores, all_scores


def add_genrace_scores_to_df(df_scores, threshold=0.8):
    # all column names
    name_columns = ["character", "file_name", "seed", "sample", "eval_model"]
    score_columns = [col for col in df_scores.columns.tolist() if col[1] == "_" and col not in name_columns]

    race_map = {"R_white": "W", "R_black": "B", "R_asian": "A", "R_hispanic": "H"}
    age_map = {"A_kid": "K", "A_teen": "T", "A_adult": "A", "A_old": "O"}
    new_columns = {
        "gender": [],
        "race": [],
        "age": [],
        "gen_race_age_tag": [],
    }

    for i, row in df_scores.iterrows():
        # Tag Gender with M or F if only one is above the threshold
        if row["G_male"] > threshold or row["G_female"] > threshold:
            if row["G_female"] < threshold:
                new_columns["gender"].append("M")
            elif row["G_male"] < threshold:
                new_columns["gender"].append("F")
            else:
                new_columns["gender"].append("U")
        else:
            new_columns["gender"].append("U")

        # Tag Race with W, B, A, H if only the highest score is above the threshold
        race_scores = sorted([(row[r_tag], r_tag) for r_tag in race_map.keys()], reverse=True)
        if race_scores[0][0] > threshold:
            new_columns["race"].append(race_map[race_scores[0][1]])
        else:
            new_columns["race"].append("U")

        # Tag Age with K, T, A, O with the highest score
        age_scores = sorted([(row[a_tag], a_tag) for a_tag in age_map.keys()], reverse=True)
        new_columns["age"].append(age_map[age_scores[0][1]])

        new_columns["gen_race_age_tag"].append(new_columns["gender"][-1] + new_columns["race"][-1] + new_columns["age"][-1])

    for key, value in new_columns.items():
        # print(f"Adding {key} to df_scores:", len(value))
        df_scores[key] = value
    return df_scores


