#
import os
import json
import pickle
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
from natsort import natsorted

import dotenv
dotenv.load_dotenv("./.env")

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from utils import get_internvl_logits, get_logits
from dataset_utils import add_stero_label
from model_utils import load_model, model2id_map, model2name_map, model2family_map
from query_templates import llm_template_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="InternVL")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    model_id = model2id_map[args.model_name]
    model_name = model2name_map[model_id]
    model_family = model2family_map[model_name]

    model, proc_or_tokenizer = load_model(model_id, args.device)
    if "InternVL" in model_family:
        tokenizer = proc_or_tokenizer
    else:
        tokenizer = proc_or_tokenizer.tokenizer

    tokens = ["A", "a", "B", "b", "C", "c"]
    ABC_token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # InternVL tokens
    # ABC_token_ids = [290, 264, 309, 260, 289, 271] # A, a, B, b, C, c

    target_groups = ["age", "appearance", "gender", "race", "religion", "ses", "disability"]
    # target_groups = ["age"]

    for target_group in target_groups:
        # df_path = F"group_tables/df_{target_group}.pkl"
        # df_target_table = pd.read_pickle(df_path)
        df_path = F"input_csv_files_corrected/{target_group}_with_object_types.csv"
        df_target_table = pd.read_csv(df_path)

        df_target_table = add_stero_label(df_target_table, target_group)

        # Create queries
        all_queries = []
        for i, row in df_target_table.iterrows():
            context = row["context"].strip()
            question = row["question"].strip()
            options = [row[F"ans{i}"].strip() for i in range(3)]
            query = llm_template_map[model_family].format(context=context, question=question, OPT1=options[0], OPT2=options[1], OPT3=options[2])
            all_queries.append(query)

        # Get logits
        all_choices = []
        for bi in tqdm(range(0, len(all_queries), args.batch_size)):
            query_batch = all_queries[bi:bi + args.batch_size]

            model_inputs = tokenizer(text=query_batch, padding=True, return_tensors='pt')

            if "InternVL" in model_family:
                logits, probs = get_internvl_logits(model, **model_inputs)
            # elif model_name in ["Qwen2-VL-7B", "Molmo-7B", "Phi-3.5", "Llava-1.5-7B"]:
            else:
                logits, probs = get_logits(model, model_inputs)
            probs = probs.cpu().detach()

            all_ABC = probs[:, -1, ABC_token_ids] #.sum(dim=-1).tolist()
            # make all_ABC from 6 numbers to 3 by adding numbers together
            all_ABC = all_ABC.reshape(-1, 3, 2).sum(dim=-1).float().numpy()

            # append index of the highest probability
            all_choices.extend(all_ABC.argmax(axis=-1))

        df_scores = df_target_table.copy()
        df_scores[F"choice_{model_name}"] = all_choices

        os.makedirs("results/vlm_settings1_scores", exist_ok=True)
        df_scores.to_pickle(F"results/vlm_settings1_scores/df_{target_group}_vlm_settings1_{model_name}.pkl")

        correct_count = (df_scores["label"] == df_scores[F"choice_{model_name}"]).sum()
        stereo_count = (df_scores["label_stereotyped"] == df_scores[F"choice_{model_name}"]).sum()

        with open("llm_bias_results.txt", "a+") as f:
            f.write(F"{model_name}_{target_group}: correct: {round(correct_count/len(df_scores), 3)}, stereo: {round(stereo_count/len(df_scores), 3)}, count {len(df_scores)}\n")
        print(F"{target_group}:")
        print(F"correct: {round(correct_count/len(df_scores), 3)}, stereo: {round(stereo_count/len(df_scores), 3)}, count {len(df_scores)}")
