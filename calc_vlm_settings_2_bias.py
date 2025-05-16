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

import transformers
from transformers import AutoModel, AutoTokenizer
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from utils import *
from model_utils import *
from query_templates import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="InternVL")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_images_per_query", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    model_id = model2id_map[args.model_name]
    model_name = model2name_map[model_id]

    model, processor = load_model(model_id, args.device)

    tokens = ["A", "a", "B", "b", "C", "c"]
    if model_name in ["InternVL2_5-8B"]:
        tokenizer = processor
        ABC_token_ids = tokenizer.convert_tokens_to_ids(tokens)
        num_image_token = model.num_image_token
    else:
        tokenizer = processor.tokenizer
        ABC_token_ids = tokenizer.convert_tokens_to_ids(tokens)
        num_image_token = 256

    # InternVL tokens
    # ABC_token_ids = [290, 264, 309, 260, 289, 271] # A, a, B, b, C, c

    target_groups = ["age", "appearance", "gender", "race", "religion", "ses"]
    target_groups = ["age", "race"]

    for target_group in target_groups:
        print("processing", target_group)

        df_path = F"group_tables/df_{target_group}_enriched.pkl"
        df_target_table = pd.read_pickle(df_path)
        if "label_stereotyped" not in df_target_table.columns:
            df_target_table = add_stero_label(df_target_table, target_group)
        generated_images_path = Path(F"./bias_images/bias_{target_group}/generated_images_{target_group}")

        if "image_path_pairs" not in df_target_table.columns:
            df_img_scores = pd.read_pickle(F"./all_image_scores_pickles/df_scores_{target_group}.pkl")
            df_img_scores, all_scores = add_genrace_scores_to_df(df_img_scores)

        # Create queries
        print("Creating queries...")
        all_queries = []
        all_images = []
        all_indices = []
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

            if "image_path_pairs" not in row:
                char_1_images = all_scores[target_group][row["question_index"]][characters[0].lower()]
                char_2_images = all_scores[target_group][row["question_index"]][characters[1].lower()]
                all_image_pairs = []
                for char1_img in char_1_images[:15]:
                    for char2_img in char_2_images[:15]:
                        if (target_group in ["appearance", "ses"] and char1_img[1] == char2_img[1]) or \
                            (target_group in ["age", "gender"] and char1_img[1][1] == char2_img[1][1]) or \
                            (target_group in ["race", "religion"] and char1_img[1][0] == char2_img[1][0]):
                            all_image_pairs.append((char1_img[0] * char2_img[0], character_path / char1_img[2], character_path / char2_img[2]))

                all_image_pairs = sorted(all_image_pairs, key=lambda x: x[0], reverse=True)[:args.num_images_per_query]
            else:
                all_image_pairs = row["image_path_pairs"][:args.num_images_per_query]
            new_images = [concat_images_from_paths(*img_pair[1:]) for img_pair in all_image_pairs]

            query = vlm_settings_2_templates[model_id].format(question=question, OPT1=options[0], OPT2=options[1], OPT3=options[2])

            for image in new_images:
                all_images.append(image)
                all_queries.append(query)
                all_indices.append(i)

        print("Number of queries:", len(all_queries))

        # Get logits
        print("Getting logits...")
        all_choices = [[] for _ in range(len(df_target_table))]
        for bi in tqdm(range(0, len(all_queries), args.batch_size)):
            query_batch = all_queries[bi:bi + args.batch_size]
            image_batch = all_images[bi:bi + args.batch_size]
            indices_batch = all_indices[bi:bi + args.batch_size]

            model_inputs = process_inputs(model_id, processor, tokenizer, query_batch, images=image_batch, num_image_token=num_image_token)

            if model_name == "InternVL2_5-8B":
                logits, probs = get_internvl_logits(model, **model_inputs)
            # elif model_name in ["Qwen2-VL-7B", "Molmo-7B", "Phi-3.5", "Llava-1.5-7B"]:
            else:
                logits, probs = get_logits(model, model_inputs, model_name=model_name)
            probs = probs.cpu().detach()

            all_ABC = probs[:, -1, ABC_token_ids] #.sum(dim=-1).tolist()
            # make all_ABC from 6 numbers to 3 by adding numbers together
            all_ABC = all_ABC.reshape(-1, 3, 2).sum(dim=-1).float().numpy()

            # append index of the highest probability
            # all_choices.extend(all_ABC.argmax(axis=-1))
            for idx, choice in zip(indices_batch, all_ABC.argmax(axis=-1)):
                all_choices[idx].append(choice)

            # if bi > 50:
            #     break

        print("Saving scores...")
        df_scores = df_target_table.copy()
        df_scores[F"choice_{model_name}"] = all_choices

        os.makedirs(F"results/vlm_settings2_scores", exist_ok=True)
        df_scores.to_pickle(F"results/vlm_settings2_scores/df_{target_group}_vlm_settings2_{model_name}.pkl")

        correct_count = sum([sum([1 for ch in all_choices[i] if ch == l]) for i, l in enumerate(df_scores["label"])])
        stereo_count = sum([sum([1 for ch in all_choices[i] if ch == l]) for i, l in enumerate(df_scores["label_stereotyped"])])
        total_count = float(len(all_queries))

        with open("vlm_bias_results.txt", "a+") as f:
            f.write(F"Set2: {model_name}_{target_group}:\t\tcorrect: {round(correct_count/total_count, 3)}, stereo: {round(stereo_count/total_count, 3)}, count {len(df_scores)}\n")
        print(F"{target_group}:")
        print(F"correct: {round(correct_count/total_count, 3)}, stereo: {round(stereo_count/total_count, 3)}, count {total_count}")
