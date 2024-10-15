import pandas as pd
import numpy as np
from tqdm import tqdm
import click
import evaluate
import logging

@click.command()
@click.option("--input_file", type=str, required=True, help="Path to input file")
@click.option("--output_column", type=str, default="prediction", help="Name of input column")
@click.option("--target_column", type=str, default="label", help="Path to output file")
@click.option("--log_file", type=str, default=None, help="Path to log file")
def main(input_file, output_column, target_column, log_file):
    df = pd.read_csv(input_file)
    df_len = len(df)
    isnans = df[output_column].isna() | df[target_column].isna()
    df = df[~isnans].reset_index(drop=True)
    if len(df) < df_len:
        print(f"Removed {df_len - len(df)} rows with NaN values")
    y_true = df[target_column].apply(lambda x: x.lower().strip())
    y_pred = df[output_column].apply(lambda x: x.split("|")[0].lower().strip())
    exact_match = compute_exact_match(y_true, y_pred)
    inclusion = compute_inclusion(y_true, y_pred)
    reverse_inclusion = compute_inclusion(y_pred, y_true)
    bleu = compute_bleu(y_true, y_pred)
    rouge = compute_rouge(y_true, y_pred)
    bertscore = compute_bertscore(y_true, y_pred)
    if log_file is not None:
        logging.basicConfig(filename=log_file, level=logging.INFO)
        logging.info(f"Exact Match: {exact_match}")
        logging.info(f"Inclusion: {inclusion}")
        logging.info(f"Reverse Inclusion: {reverse_inclusion}")
        logging.info(f"Bleu: {bleu}")
        logging.info(f"Rouge 1: {rouge}")
        logging.info(f"Bertscore: {bertscore}")
    else:
        print(f"Exact Match: {exact_match}")
        print(f"Inclusion: {inclusion}")
        print(f"Reverse Inclusion: {reverse_inclusion}")
        print(f"Bleu: {bleu}")
        print(f"Rouge 1: {rouge}")
        print(f"Bertscore: {bertscore}")


def compute_exact_match(y_true, y_pred):
    return (y_pred == y_true).mean()

def compute_inclusion(y_true, y_pred):
    s = 0
    for i in range(len(y_true)):
        if y_true[i] in y_pred[i]:
            s += 1
    return s / len(y_true)

def compute_bleu(y_true, y_pred):
    bleu = evaluate.load("bleu")
    s = 0
    print(f"Computing BLEU")
    for i in tqdm(range(len(y_true))):
        if len(y_pred[i].split()) == 0:
            s += float(int(y_true[i] == y_pred[i]))
            continue
        res = bleu.compute(predictions=[y_pred[i]], references=[y_true[i]])
        if res['reference_length'] > 3:
            s += res['bleu']
        else:
            precs = res['precisions'][:res['reference_length']]
            weight = 1 / res['reference_length']
            bleu_score = res['brevity_penalty'] * np.exp(np.sum([weight * np.log(p) for p in precs]))
            s += bleu_score
    return s / len(y_true)

def compute_rouge(y_true, y_pred):
    rouge = evaluate.load("rouge")
    s = 0
    print(f"Computing ROUGE")
    for i in tqdm(range(len(y_true))):
        if len(y_pred[i].split()) == 0:
            s += float(int(y_true[i] == y_pred[i]))
            continue
        s += rouge.compute(predictions=[y_pred[i]], references=[y_true[i]])['rouge1']
    return s / len(y_true)

def compute_bertscore(y_true, y_pred):
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0
    bertscore = evaluate.load("bertscore")
    s = 0
    print(f"Computing BERTSCORE")
    for i in tqdm(range(len(y_true))):
        if len(y_pred[i].split()) == 0:
            s += float(int(y_true[i] == y_pred[i]))
            continue
        s += bertscore.compute(predictions=[y_pred[i]], references=[y_true[i]], lang="en")['f1'][0]
    return s / len(y_true)

if __name__ == "__main__":
    main()