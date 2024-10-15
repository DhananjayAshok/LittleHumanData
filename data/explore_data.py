import pandas as pd
from sklearn.metrics import confusion_matrix
import os
import matplotlib.pyplot as plt
import yaml

with open("proj_params.yml", "r") as f:
    proj_params = yaml.safe_load(f)

data_dir = proj_params["data_path"]

def token_count(model="gpt-3.5-turbo", cot=False, split="train",):
    data_path = os.path.join(data_dir, "generated", model, f"cot_{cot}", split)
    datasets = os.listdir(data_path)
    datasets.remove("batches")
    qa_dfs = []
    fv_dfs = []
    for dataset in datasets:
        df = pd.read_csv(data_path+"/"+dataset)
        task = df["task"].iloc[0]
        if "narrative" in df.dataset_name.iloc[0]:
            df = df.dropna(axis=0).reset_index(drop=True)
        df["context_token_count"] = df["context"].apply(lambda x: len(x.split()))
        if task == "qa":
            df["question_token_count"] = df["question"].apply(lambda x: len(x.split()))
            df["answer_token_count"] = df["answer"].apply(lambda x: len(x.split()))
            df["syn_question_token_count"] = df["syn_question"].apply(lambda x: len(x.split()))
            df["syn_answer_token_count"] = df["syn_answer"].apply(lambda x: len(x.split()))
            qa_dfs.append(df)
        elif task == "fv":
            df["claim_token_count"] = df["claim"].apply(lambda x: len(x.split()))
            df["syn_claim_token_count"] = df["syn_claim"].apply(lambda x: len(x.split()))
            fv_dfs.append(df)

    qa_df = pd.concat(qa_dfs, axis=0, ignore_index=True)
    fv_df = pd.concat(fv_dfs, axis=0, ignore_index=True)
    return qa_df, fv_df

def show_dataset_stats(split="train"):
    qa_df, fv_df = token_count(split="train")
    print(f"QA Datasets:")
    print(qa_df["dataset_name"].value_counts())
    grouped = qa_df[["dataset_name","context_token_count", "question_token_count", "answer_token_count", "syn_question_token_count", "syn_answer_token_count"]].groupby("dataset_name")
    print(grouped.mean())
    print(grouped.sum())

    # do same for fv with columns: context_token_count, claim_token_count
    print(f"Fact Verification Dataset Stats for {split} split:")
    print(fv_df[["context_token_count", "claim_token_count"]].describe())
    print(f"Dataset Grouped Counts:")
    print(fv_df["dataset_name"].value_counts())
    print(fv_df["dataset_name"].value_counts()/len(fv_df))
    grouped = fv_df[["dataset_name","context_token_count", "claim_token_count", "syn_claim_token_count", "label", "syn_label"]].groupby("dataset_name")
    print(grouped.mean())
    print(grouped.sum())
    for dataset in fv_df["dataset_name"].unique():
        sub_df = fv_df[fv_df["dataset_name"]==dataset].reset_index(drop=True)
        print(f"Dataset: {dataset}")
        y_true = sub_df["label"]
        y_pred = sub_df["syn_label"]
        print(confusion_matrix(y_true, y_pred, normalize="all"))
    return

def show_individual_context_lengths(split="train"):
    qa_df, fv_df = token_count(split="train")
    for dataset in qa_df["dataset_name"].unique():
        print(f"Dataset: {dataset}")
        print(qa_df[qa_df["dataset_name"]==dataset][["context_token_count", "question_token_count"]].describe())
    for dataset in fv_df["dataset_name"].unique():
        print(f"Dataset: {dataset}")
        print(fv_df[fv_df["dataset_name"]==dataset][["context_token_count", "claim_token_count"]].describe())

def show_sample(row, fv=True):
    context = row["context"]
    print(f"Context: {context}")
    if fv:
        claim = row["claim"]
        print(f"Claim: {claim} | Label: {row['label']}")
        syn_claim = row["syn_claim"]
        print(f"Synthetic Claim: {syn_claim} | Synthetic Label: {row['syn_label']}")
    else:
        question = row["question"]
        print(f"Question: {question}")
        answer = row["answer"]
        print(f"Answer: {answer}")
        syn_question = row["syn_question"]
        print(f"Synthetic Question: {syn_question}")
        syn_answer = row["syn_answer"]
        print(f"Synthetic Answer: {syn_answer}")


def print_samples(split="train"):
    qa_df, fv_df = token_count(split="train")
    for dataset in qa_df["dataset_name"].unique():
        print(f"Dataset: {dataset}")
        sub_df = qa_df[qa_df["dataset_name"]==dataset].sample(5).reset_index(drop=True)
        for i in range(5):
            row = sub_df.iloc[i]
            show_sample(row, fv=False)
    for dataset in fv_df["dataset_name"].unique():
        print(f"Dataset: {dataset}")
        sub_df = fv_df[fv_df["dataset_name"]==dataset].sample(5).reset_index(drop=True)
        for i in range(5):
            row = sub_df.iloc[i]
            show_sample(row, fv=True)


if __name__ == "__main__":
    #show_dataset_stats(split="train")
    #show_individual_context_lengths(split="train")
    print_samples()
