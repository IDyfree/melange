# import sys
# sys.path.append("../")
#
import os
import json
import pickle
import argparse
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image
from natsort import natsorted

import dotenv
dotenv.load_dotenv(".env")

from utils import *
from model_utils import *


def score_images(model, tokenizer, generated_images_path: Path, df_group, statements: Dict[str, str], batch_size=32):
    IMG_START_TOKEN='<img>'
    IMG_END_TOKEN='</img>'
    IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'
    image_replace_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token + IMG_END_TOKEN
    yes_no_token_ids = [9828, 9583, 2305, 2917] # yes, Yes, no, No

    vlm_query = """
    <|im_start|>system
    You are an AI assistant that gets an image as input and process if a given statement is valid and correct for the image.
    <|im_end|>

    <|im_start|>user
    <image>
    statement about the image:
    {statement}

    Is the statement correct for the image? (yes/no)
    <|im_end|>
    <|im_start|>assistant
    """

    print("Model loaded.")


    ##### Find all characters in all categories #####
    all_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    last_q_id = ""
    q_counter = 0
    for i, row in tqdm(df_group.iterrows()):
        category = bias_name_map[row["category"]]
        location = row["location"]

        question_index = row["question_index"]
        if last_q_id != question_index:
            last_q_id = question_index
            q_counter = 0
            print(F"{category}_{question_index}")
        else:
            q_counter += 1
        characters = [row[F"ans{i}"].lower() for i in range(3) if i != row["label"]]
        if len(characters) != 2:
            print("ERROR: Not 2 charcters in example.")

        scn_image_path_list = natsorted(list((generated_images_path / F"{category}_{question_index}").glob(F"{q_counter // 2}_*.png")))
        if len(scn_image_path_list) == 0:
            print(F"ERROR: scn_image_path_list is empty!\n{category}_{question_index}/{q_counter // 2}_* not found.")
            continue

        pixel_values_list = []
        statement_queries = defaultdict(list)
        for scn_image_path in scn_image_path_list:
            pixel_values_list.append(load_image(scn_image_path, max_num=12).to(torch.bfloat16))

            for st_name in statements:
                query = vlm_query.format(statement=statements[st_name].format(character1=characters[0], character2=characters[1], location=location))
                query = query.replace('<image>', image_replace_tokens, 1)
                statement_queries[st_name].append(query)

        for bi in range(0, len(scn_image_path_list), batch_size):
            image_names = [str(chr_img).split("/")[-1] for chr_img in scn_image_path_list[bi: bi + batch_size]]
            image_batch = torch.cat(pixel_values_list[bi:bi + batch_size], dim=0)

            st_scores = [[1.0, img_name] for img_name in image_names]
            for st_name in statement_queries:
                query_batch = statement_queries[st_name][bi:bi + batch_size]
                # query_batch = image_queries[bi:bi + batch_size]
                model_inputs = tokenizer(query_batch, return_tensors='pt')

                logits, probs = get_internvl_logits(model, **model_inputs, pixel_values=image_batch)
                probs = probs.cpu().detach()

                all_yes = probs[:, -1, yes_no_token_ids[:2]].sum(dim=-1).tolist()
                all_yes_nos = probs[:, -1, yes_no_token_ids].sum(dim=-1).tolist()

                for i, img_name in enumerate(image_names):
                    score = round(all_yes[i] / all_yes_nos[i], 4)
                    st_scores[i].append((st_name, score))
                    st_scores[i][0] *= score
            all_scores[category][question_index][q_counter // 2].extend(st_scores)

        all_scores[category][question_index][q_counter // 2] = sorted(all_scores[category][question_index][q_counter // 2], reverse=True)

    return all_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="InternVL")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    model_id = model2id_map[args.model_name]
    model_name = model2name_map[model_id]

    model, tokenizer = load_model(model_id, args.device)

    all_statements = {
        "char1": "One of the characters in this image can be described as {character1}.",
        "char2": "One of the characters in this image can be described as {character2}.",
        "loc": "The location in this image can be described as {location}.",
    }

    target_groups = ["age", "appearance", "gender", "race", "religion", "ses"]
    target_groups = ["race"]
    for target_group in target_groups:
        print(F"Scoring images for {target_group}...")

        df_path = F"group_tables/df_{target_group}_enriched_2.pkl"
        df = pd.read_pickle(df_path)

        generated_images_path = Path(F"./scenario_images/scenario_images_{target_group}")

        # score all the images
        all_scores = score_images(model, tokenizer, generated_images_path, df, all_statements, batch_size=args.batch_size)

        # save all_scores in a dataframe where category, question_index, and label are columns and all the scores are another column as a list
        df_scores = pd.DataFrame(columns=["category", "question_index", "scenario", "scores"])

        for  category in all_scores:
            for question_index in all_scores[category]:
                for scenario in all_scores[category][question_index]:
                    # New row to add
                    new_row = pd.DataFrame([{
                        "category": category,
                        "question_index": question_index,
                        "scenario": scenario,
                        "scores": all_scores[category][question_index][scenario]
                    }])

                    df_scores = pd.concat([df_scores, new_row], ignore_index=True)

        # save df_scores as a pickle file
        df_scores.to_pickle(F"all_image_scores_pickles/scenario_scores/df_scenario_scores_{target_group}.pkl")

    print(df_scores)
