from dotenv import load_dotenv
import pandas as pd
import os
from tqdm import tqdm
from tools import check_num_tokens_fn

load_dotenv()

for fname in os.listdir("output"):
    path = os.path.join("output", fname)
    df = pd.read_csv(path)

    for i, row in tqdm(df.iterrows(), desc=fname):
        original_sentence = row["original_sentence"]
        original_triplet = row["original_triplet"]

        counterfact1 = row["counterfact1"]
        counterfact_triplet1 = row["counterfact_triplet1"]

        counterfact2 = row["counterfact2"]
        counterfact_triplet2 = row["counterfact_triplet2"]

        counterfact3 = row["counterfact3"]
        counterfact_triplet3 = row["counterfact_triplet3"]

        if isinstance(original_triplet, str):
            original_triplet = eval(original_triplet)
        if isinstance(counterfact_triplet1, str):
            counterfact_triplet1 = eval(counterfact_triplet1)
        if isinstance(counterfact_triplet2, str):
            counterfact_triplet2 = eval(counterfact_triplet2)
        if isinstance(counterfact_triplet3, str):
            counterfact_triplet3 = eval(counterfact_triplet3)

        original_triplet = original_triplet[0]

        if original_triplet[0] == "null":
            continue

        report1 = check_num_tokens_fn(original_sentence, counterfact1,
                                      original_triplet, counterfact_triplet1)
        report2 = check_num_tokens_fn(original_sentence, counterfact2,
                                      original_triplet, counterfact_triplet2)
        report3 = check_num_tokens_fn(original_sentence, counterfact3,
                                      original_triplet, counterfact_triplet3)
        
        report1 = eval(report1)
        report2 = eval(report2)
        report3 = eval(report3)
        
        if not (report1["is_equal"] and report2["is_equal"] and report3["is_equal"]):
            print(f"Not equal index {i}")
            print(report1)
            print(report2)
            print(report3)