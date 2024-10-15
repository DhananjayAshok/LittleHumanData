import click
import pandas as pd
import numpy as np
import os
import yaml
from tqdm import tqdm

if os.path.basename(os.getcwd()) == "data":
    raise Exception("Do not run this script from the data folder")

with open("proj_params.yml", "r") as f:
    proj_params = yaml.safe_load(f)

data_dir = proj_params["data_path"]

@click.command()
@click.option("--dataset_name", help="Name of the dataset to process")
@click.option("--model", default="gpt-3.5-turbo", help="Model used for synthetic data generation")
@click.option("--cot", default=False, help="Whether to use cot data or not")
@click.option("--random_seed", default=42, help="Random seed to use")
def create_splits(dataset_name, model, cot, random_seed):
    np.random.seed(random_seed)
    for synth_frac in [0.0, 0.5, 0.95, 0.975, 1.0]:
        for split in ["train", "valid", "test"]:
            # check that data exists, both real and synthetic
            dataset_name = dataset_name.replace(".csv", "")
            synth_data_path = data_dir + f"/generated/{model}/cot_{cot}/{split}/{dataset_name}.csv"
            needs_synth = synth_frac > 0
            if needs_synth and split == "test":
                continue # Will always skip because we never touch test so dont have synthetic data for it
            if needs_synth and not os.path.exists(synth_data_path):
                if split == "train":
                    raise Exception(f"Synthetic data path {synth_data_path} does not exist. Set synth_frac to 0 to ignore this")
                else:
                    print(f"Synthetic data path {synth_data_path} does not exist for validation. Skipping...")
                    continue
            elif not os.path.exists(synth_data_path):
                data = pd.read_csv(data_dir + f"/{split}/{dataset_name}.csv")
            else:
                data = pd.read_csv(synth_data_path)
            if data['dataset_name'].unique()[0] == "fever":
                if split == "train":
                    pass
                else:
                    data = data.sample(frac=0.25).reset_index(drop=True) # FEVER is expensive so we just don't use all of it
            data["id"] = data.index
            data["text"] = None
            general_dest_suffix = f"{data_dir}/ft/[EXP]/{model}/cot_{cot}/{dataset_name}/"
            dest_dir = general_dest_suffix.replace("[EXP]", "exp1") + f"frac_{synth_frac}/"
            #if not os.path.exists(dest_dir):
            #    os.makedirs(dest_dir)
            #create_experiment_1_splits(dest_dir+f"/{split}.csv", data.copy(deep=True), synth_frac)
            #print(f"Saved split {split} to {dest_dir}/{split}.csv ...")
            if split != "train":
                continue
            nruns = 3
            sizes = [1000, 3000, 5000]
            if dataset_name == "scifact":
                sizes = [100, 300, 500]
            for size in sizes:
                for run in range(nruns):
                    dest_dir = general_dest_suffix.replace("[EXP]", "exp2") + f"size_{size}/frac_{synth_frac}/run_{run}/"
                    if not os.path.exists(dest_dir):
                        os.makedirs(dest_dir)
                    sampled_data = data.sample(size).reset_index(drop=True)
                    #create_experiment_2_splits(dest_dir+f"/{split}.csv", sampled_data, synth_frac)
                    #print(f"Saved split {split} to {dest_dir}/{split}.csv ...")
            base_sizes = [1000, 2000, 3000]
            if dataset_name == "scifact":
                base_sizes = [100, 200, 300]
            dest_dir = f"{data_dir}/ft/exp3/{dataset_name}/"
            real_add = 200
            if dataset_name == "scifact":
                real_add = 20
            create_experiment_3_splits(dest_dir, data, base_sizes, real_add=real_add)
            print(f"Saved experiment 3 data to {dest_dir} ...")


def get_data_split(data, synth_frac):
    data["synthetic"] = None
    task = data['task'].iloc[0]
    target_text_col = "text"
    target_col = "label"
    if task == "qa":
        text_col = "question"
        second_col = "answer"
    elif task == "fv":
        data["target"] = data["label"]
        if "syn_label" in data.columns:
            data["syn_target"] = data["syn_label"]
        text_col = "claim"
        second_col = "target"
    else:
        raise ValueError(f"Task {task} not recognized")
    syn_text_col = f"syn_{text_col}"
    syn_second_col = f"syn_{second_col}"
    data[target_text_col] = data["context"] + " | "
    if synth_frac == 0:
        data[target_text_col] = data[target_text_col] + data[text_col]
        data[target_col] = data[second_col]
        data["synthetic"] = False
    elif synth_frac == 1:
        data[target_text_col] = data["context"] + " | " + data[syn_text_col]
        data[target_col] = data[syn_second_col]
        data["synthetic"] = True
    else:
        n_synth = int(len(data) * synth_frac)
        indices = np.arange(len(data))
        synth_indices = np.random.choice(indices, n_synth, replace=False)
        data.loc[synth_indices, target_text_col] = data["context"] + " | " + data[syn_text_col]
        data.loc[synth_indices, target_col] = data[syn_second_col]
        data.loc[synth_indices, "synthetic"] = True
        real_indices = ~data.index.isin(synth_indices)
        data.loc[real_indices, target_text_col] = data["context"] + " | " + data[text_col]
        data.loc[real_indices, target_col] = data[second_col]
        data.loc[real_indices, "synthetic"] = False
    data = data[["id", target_text_col, target_col, "synthetic", "meta"]]
    if task == "qa":
        data["input"] = data[target_text_col] + " ANSWER: "
        data[target_text_col] = data[target_text_col] + " ANSWER: " + data[target_col] + " |"
    return data


def create_experiment_1_splits(dest_file, data, synth_frac):
    data = get_data_split(data, synth_frac)
    data.to_csv(dest_file, index=False)


def create_experiment_2_splits(dest_file, data, synth_frac):
    small_dest_file = dest_file.replace(".csv", "_small.csv")
    data = get_data_split(data, synth_frac)
    data.to_csv(dest_file, index=False)
    if synth_frac == 1:
        return
    data = data[data['synthetic'] == False].reset_index(drop=True)
    data.to_csv(small_dest_file, index=False)

def create_experiment_3_splits(dest_dir, data, base_sizes, real_add):
    synth_data = get_data_split(data, 1)
    real_data = get_data_split(data, 0)
    base_sizes = sorted(base_sizes, reverse=True)
    synth_max_gap = base_sizes[0] - base_sizes[1]
    n_synth_adds = (synth_max_gap // real_add) - 1
    big_df = synth_data
    for base_size in base_sizes:
        base_sample = big_df.sample(base_size)
        not_base_sample = big_df[~big_df.index.isin(base_sample.index)].reset_index(drop=True)
        base_sample = base_sample.reset_index(drop=True)
        true_dest_dir = dest_dir + f"size_{base_size}/"
        if not os.path.exists(true_dest_dir):
            os.makedirs(true_dest_dir)
        base_sample.to_csv(true_dest_dir + "train.csv", index=False)
        real_addition = real_data.sample(real_add).reset_index(drop=True)
        base_w_real = pd.concat([base_sample, real_addition], axis=0).reset_index(drop=True)
        base_w_real.sample(frac=1).reset_index(drop=True).to_csv(true_dest_dir + f"real_train.csv", index=False)
        for i in range(n_synth_adds):
            n_add = real_add * (i + 1)
            synth_addition = not_base_sample.loc[:n_add-1]
            base_w_synth = pd.concat([base_sample, synth_addition], axis=0).reset_index(drop=True)
            base_w_synth.sample(frac=1).reset_index(drop=True).to_csv(true_dest_dir + f"synth_{i}_train.csv", index=False)
        big_df = base_sample

if __name__ == "__main__":
    create_splits()