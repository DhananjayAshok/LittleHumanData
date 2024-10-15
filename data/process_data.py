from datasets import load_dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys
import yaml
# numpy set random seet 42
np.random.seed(42)

if os.path.basename(os.getcwd()) == "data":
    raise Exception("Do not run this script from the data folder")

with open("proj_params.yml", "r") as f:
    proj_params = yaml.safe_load(f)

data_dir = proj_params["data_path"]
# if train, valid and test folders do not exist in data_dir, create them
for split in ["train", "valid", "test"]:
    if not os.path.exists(os.path.join(data_dir, split)):
        os.makedirs(os.path.join(data_dir, split))


split_map = {"train": "train", "validation": "valid", "test": "test", "valid": "valid", "dev": "valid", "val": "valid", "trn": "train", "tst": "test" ,"eval": "test"}


def drop_col_nans(df, cols):
    for col in cols:
        df = df[~df[col].isna()]
        df = df[df[col].apply(lambda x: x.strip()) != ""]
    return df.reset_index(drop=True)

def process_fever():
    ds = load_dataset("copenlu/fever_gold_evidence")
    label_map = {"SUPPORTS": 1, "REFUTES": 0, "NOT ENOUGH INFO": 0}
    def get_evidence(ev):
        evs = []
        for e in ev:
            evs.append(e[2].replace("-LRB-", "(").replace("-RRB-", ")"))
        return "\n".join(evs)
    dfs = {}
    def proc_df(df):
        df["dataset_name"] = "fever"
        df["task"] = "fv" 
        df["context"] = df["evidence"].apply(get_evidence)
        # check for context ID likely not useful here
        df["label"] = df["label"].map(label_map)
        df["meta"] = None
        df["context_id"] = df.index
        df = df[["dataset_name", "claim", "task", "context", "context_id", "label", "meta"]]
        df = drop_col_nans(df, cols=["context", "claim"])
        return df
    for split in ["train", "test", "validation"]:
        dfs[split_map[split]] = proc_df(ds[split].to_pandas())
    for split in dfs:
        df = dfs[split]
        if split == "train":
            df = df.sample(frac=0.3).reset_index(drop=True)
        df.to_csv(os.path.join(data_dir, split, "fever.csv"), index=False)
    print(f"Processed FEVER dataset")
    return 

def process_narrativeqa():
    ds = load_dataset("deepmind/narrativeqa")
    def extract_answers(answers):
        final_answers = []
        for answer in answers:
            final_answers.append(answer["text"])
        return final_answers[0]
    def proc_df(df):
        df["dataset_name"] = "narrativeqa"
        df["task"] = "qa"
        df["context"] = df["document"].apply(lambda x: x["summary"]["text"])
        df["context_id"] = df["document"].apply(lambda x: x["id"])
        df["question"] = df["question"].apply(lambda x: x["text"])
        df["answer"] = df["answers"].apply(extract_answers)
        df["meta"] = df["document"].apply(lambda x: x["kind"])
        df = df[["dataset_name", "task", "question", "answer", "context", "context_id", "meta"]]
        df = drop_col_nans(df, cols=["context", "question", "answer"])
        df = df.dropna(axis=0).reset_index(drop=True)
        return df
    
    dfs = {}
    for split in ["train", "validation", "test"]:
        df = ds[split].to_pandas()
        dfs[split_map[split]] = proc_df(df)
    for split in dfs:
        dfs[split].to_csv(os.path.join(data_dir, split, "narrativeqa.csv"), index=False)
    print(f"Processed NarrativeQA dataset")
    return

def process_coqa():
    ds = load_dataset("stanfordnlp/coqa")
    def proc_df(df):
        df["dataset_name"] = "coqa"
        df["task"] = "qa"
        df["context"] = df["story"]
        df["context_id"] = df.index
        df["question"] = df["questions"].apply(lambda x: x[0])
        df["answer"] = df["answers"].apply(lambda x: x['input_text'][0])
        df["meta"] = None
        df = df[["dataset_name", "task", "question", "answer", "context", "context_id", "meta"]]
        df = drop_col_nans(df, cols=["context", "question", "answer"])
        return df
    dfs = {}
    for split in ["train", "validation"]:
        df = ds[split].to_pandas()
        if split == "validation":
            dfs["test"] = proc_df(df)
        else:
            dfs[split_map[split]] = proc_df(df)
    for split in dfs:
        dfs[split].to_csv(os.path.join(data_dir, split, "coqa.csv"), index=False)
    print(f"Processed CoQA dataset")
    return

def process_fairytaleqa():
    ds = load_dataset("GEM/FairytaleQA")
    def proc_df(df):
        indices = df["content"].unique().tolist()
        df["dataset_name"] = "fairytaleqa"
        df["task"] = "qa"
        df["context"] = df["content"]
        df["context_id"] = df["content"].apply(lambda x: indices.index(x))
        df.rename({"attribute": "meta"}, axis=1, inplace=True)
        df = df[["dataset_name", "task", "question", "answer", "context", "context_id" ,"meta"]]
        df = drop_col_nans(df, cols=["context", "question", "answer"])
        return df
    dfs = {}
    for split in ["train", "validation", "test"]:
        df = ds[split].to_pandas()
        dfs[split_map[split]] = proc_df(df)
    for split in dfs:
        dfs[split].to_csv(os.path.join(data_dir, split, "fairytaleqa.csv"), index=False)

def process_factify(split_frac=0.2):
    path = f"{data_dir}/tmp/public_folder"
    ones = ["Support_Text"]
    zeros = ["Insufficient_Text", "Refute"]
    others = ["Insufficient_Multimodal", "Support_Multimodal"]
    label_map = {}
    for item in ones + zeros + others:
        to_map = -1
        if item in ones:
            to_map = 1
        elif item in zeros:
            to_map = 0
        label_map[item] = to_map

    def proc_df(df):
        df["dataset_name"] = "factify"
        df["task"] = "fv"
        df["context"] = df["document"]
        df["context_id"] = df["Id"]
        df["label"] = df["Category"].map(label_map)
        df["meta"] = None
        # get the non -1 labels
        df = df[df["label"] != -1].reset_index(drop=True)
        df = df[["dataset_name", "task", "context", "context_id", "claim", "label", "meta"]]
        df = drop_col_nans(df, cols=["context", "claim"])
        return df
    dfs = {}
    for split in ["train", "val"]:
        df = pd.read_csv(f"{path}/{split}.csv")
        dfs[split_map[split]] = proc_df(df)
    train_df = dfs["train"]
    dfs["test"] = dfs["valid"]
    dfs["valid"] = None
    # shuffle the train_df
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    # split the train_df into train and valid
    split_idx = int(len(train_df) * (1 - split_frac))
    dfs["train"] = train_df.iloc[:split_idx]
    dfs["valid"] = train_df.iloc[split_idx:].reset_index(drop=True)
    for split in dfs:
        dfs[split].to_csv(os.path.join(data_dir, split, "factify.csv"), index=False)
    print(f"Processed Factify dataset")
    

def process_wanli(split_frac=0.2):
    path = f"{data_dir}/tmp/wanli"
    label_map = {"neutral": 0, "contradiction": 0, "entailment": 1}
    def proc_df(df):
        df["dataset_name"] = "wanli"
        df["task"] = "fv"
        df["context"] = df["premise"]
        df["context_id"] = df["id"]
        df["claim"] = df["hypothesis"]
        df["label"] = df["gold"].map(label_map)
        df["meta"] = df["genre"]
        df = df[["dataset_name", "task", "context", "context_id", "claim", "label", "meta"]]
        df = drop_col_nans(df, cols=["context", "claim"])
        return df
    train_df = proc_df(pd.read_json(f"{path}/train.jsonl", lines=True))
    test_df = proc_df(pd.read_json(f"{path}/test.jsonl", lines=True))
    dfs = {"test": test_df}
    # shuffle the train_df
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    # split the train_df into train and valid
    split_idx = int(len(train_df) * (1 - split_frac))
    dfs["train"] = train_df.iloc[:split_idx]
    dfs["valid"] = train_df.iloc[split_idx:].reset_index(drop=True)
    for split in dfs:
        dfs[split].to_csv(os.path.join(data_dir, split, "wanli.csv"), index=False)
    print(f"Processed Wanli dataset")
    return


def process_qaconv():
    segment_df = pd.read_json(f"{data_dir}/tmp/QAConvData/article_segment.json")
    def get_context(article_id):
        if article_id not in segment_df:
            return None
        dia = segment_df[article_id].seg_dialog
        context = ""
        for item in dia:
            speaker = item["speaker"].strip()
            text = item["text"].strip()
            context += f"{speaker}: {text}|"
        return context


    def proc_df(df):
        df = df[df["answers"].apply(lambda x: len(x) != 0)].reset_index(drop=True)
        df["dataset_name"] = "qaconv"
        df["task"] = "qa"
        df["context"] = df["article_segment_id"].apply(get_context)
        df["context_id"] = df["article_segment_id"]
        df["answer"] = df["answers"].apply(lambda x: x[0])
        df["meta"] = df["QG"]
        df = df[["dataset_name", "task", "context", "context_id", "question", "answer", "meta"]]
        df = drop_col_nans(df, cols=["context", "question", "answer"])
        return df
    for split in ["trn", "tst", "val"]:
        df = pd.read_json(f"{data_dir}/tmp/QAConvData/{split}.json")
        df = proc_df(df)
        df.to_csv(os.path.join(data_dir, split_map[split], "qaconv.csv"), index=False)
    print(f"Processed QAConv dataset")
    return

    
def process_ropes(split_frac=0.2):
    ds = load_dataset("allenai/ropes")
    dfs = {}
    for split in ["train", "validation"]:
        df = ds[split].to_pandas()
        empty_answer = df["answers"].apply(lambda x: len(x["text"]) == 0)
        df = df[~empty_answer].reset_index(drop=True)
        df["dataset_name"] = "ropes"
        df["task"] = "qa"
        df["context"] = df["background"] + "\n" + df["situation"]
        df["context_id"] = df["id"]
        df["answer"] = df["answers"].apply(lambda x: x["text"][0])
        df["meta"] = None
        df = df[["dataset_name", "task", "context", "context_id", "question", "answer", "meta"]]
        df = drop_col_nans(df, cols=["context", "question", "answer"])
        dfs[split_map[split]] = df
    train_df = dfs["train"]
    dfs["test"] = dfs["valid"]
    dfs["valid"] = None
    # shuffle the train_df
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    # split the train_df into train and valid
    split_idx = int(len(train_df) * (1 - split_frac))
    dfs["train"] = train_df.iloc[:split_idx]
    dfs["valid"] = train_df.iloc[split_idx:].reset_index(drop=True)
    for split in dfs:
        dfs[split].to_csv(os.path.join(data_dir, split_map[split], "ropes.csv"), index=False)
    print(f"Processed Ropes dataset")
    return


def process_scifact(split_frac=0.2):
    dataset = load_dataset("allenai/scifact_entailment")
    label_map = {"SUPPORT": 1, "NEI": 0, "CONTRADICT": 0}
    def proc_df(df):
        df["dataset_name"] = "scifact"
        df["task"] = "fv"
        df["context"] = df["title"] + " " + df["abstract"].apply(lambda x: " ".join(x))
        df["context_id"] = df["abstract_id"]
        df["label"] = df["verdict"].map(label_map)
        df["meta"] = None
        df = df[["dataset_name", "task", "context", "context_id", "claim", "label", "meta"]]
        df = drop_col_nans(df, cols=["context", "claim"])
        return df
    dfs = {}
    train_df = proc_df(dataset["train"].to_pandas())
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    split_idx = int(len(train_df) * (1 - split_frac))
    dfs["train"] = train_df.iloc[:split_idx]
    dfs["valid"] = train_df.iloc[split_idx:].reset_index(drop=True)
    dfs["test"] = proc_df(dataset["validation"].to_pandas())
    for split in dfs:
        dfs[split].to_csv(os.path.join(data_dir, split, "scifact.csv"), index=False)
    return

if __name__ == "__main__":
    process_coqa()
    #process_narrativeqa()
    #process_fairytaleqa()
    #process_factify()
    #process_scifact()
    #process_wanli()
    #process_qaconv()
    #process_ropes()
    #print("All datasets processed")
