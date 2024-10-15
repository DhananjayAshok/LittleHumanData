import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import evaluate
import warnings
# suppress warnings
warnings.filterwarnings("ignore")
import click
import os

bleu = evaluate.load("bleu")

def guarded_lower(x):
    if not isinstance(x, str):
        return x
    return x.lower().strip()



@click.command()
@click.option('--input_file', '-i', help='Input file path')
@click.option('--output_file', '-o', help='Output file path')
@click.option('--rewrite', '-r', help='Can we rewrite the input file if its same as output_file', default=False)
def main(input_file, output_file, rewrite):
    if input_file == output_file and not rewrite:
        raise ValueError("Input file and output file are the same. If you want to overwrite the input file, use the -r flag")
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir) and output_dir != "":
        os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_file)
    if "task" in df.columns:
        task = df["task"].iloc[0]
    elif "answer" in df.columns:
        task = "qa"
    elif "claim" in df.columns:
        task = "fv"
    elif "label" in df.columns:
        if not df["label"].dtype == "object":
            task = "fv"
        else:
            task = "qa"
        if task == "qa":
            df["context"] = df["input"].apply(lambda x: "|".join(x.split("|")[:-1]))
            df["question"] = df["input"].apply(lambda x: x.split("|")[-1].split("ANSWER:")[0])
            df["answer"] = df["text"].apply(lambda x: x.split("|")[-1].split("ANSWER:")[-1].strip("|"))
        if task == "fv":
            df["context"] = df["text"].apply(lambda x: "|".join(x.split("|")[:-1]))
            df["claim"] = df["text"].apply(lambda x: x.split("|")[-1])
    else:
        raise ValueError(f"No label column found in and no claim or answer column found either {df.columns}")
    if task == "qa":
        text_column = "question"
    else:
        text_column = "claim"
    config = {"task": task, "text_column": text_column}
    df["task"] = task
    df["context"] = df["context"].apply(guarded_lower)
    if task == "qa":
        df["question"] = df["question"].apply(guarded_lower)
        df["answer"] = df["answer"].apply(guarded_lower)
        if "syn_question" in df.columns:
            df["syn_question"] = df["syn_question"].apply(guarded_lower)
            df["syn_answer"] = df["syn_answer"].apply(guarded_lower)
    else:
        df["claim"] = df["claim"].apply(guarded_lower)
        if "syn_claim" in df.columns:
            df["syn_claim"] = df["syn_claim"].apply(guarded_lower)
    add_bleu_w_context(df, config)
    if task == "qa":
        add_answer_location(df, config)
        add_question_word(df, config)
    if "prediction" in df.columns:
        if task == "qa":
            df["prediction"] = df["prediction"].apply(lambda x: x.split("|")[0].lower().strip())
            add_bleu_w_answer(df, config)
        else:
            df["score"] = 1-np.abs(df["label"] - df["prediction"])
        # reorder the data with decreasing score
        df = df.sort_values("score", ascending=False)
    df.to_csv(output_file, index=False)

def compute_bleu(y_true, y_pred):
    if y_true is None or y_pred is None or not isinstance(y_true, str) or not isinstance(y_pred, str):
        return 0
    if len(y_true.split()) == 0 or len(y_pred.split()) == 0:
        return float(int(y_true == y_pred))
    res = bleu.compute(predictions=[y_pred], references=[y_true])
    if res['reference_length'] > 3:
        return res['bleu']
    else:
        precs = res['precisions'][:res['reference_length']]
        weight = 1 / res['reference_length']
        bleu_score = res['brevity_penalty'] * np.exp(np.sum([weight * np.log(p) for p in precs]))
        return bleu_score
    
def get_best_bleu_details(context, text):
    sent_options = context.split(".")
    n_sents = len(sent_options)
    if n_sents == 1:
        return 0, 0, compute_bleu(sent_options[0], text)
    elif n_sents == 0:
        return None, None, None
    best_bleu = None
    best_sent = None
    for i in range(n_sents):
        sent = sent_options[i]
        bleu_score = compute_bleu(sent, text)
        if best_bleu is None or bleu_score > best_bleu:
            best_bleu = bleu_score
            best_sent = i
    return best_sent, best_sent / n_sents, best_bleu


def add_bleu_w_context(df, config):
    text_column = config["text_column"]
    task = config["task"]
    for i in tqdm(range(len(df))):
        context = df.loc[i, "context"]
        text = df.loc[i, text_column]
        if task == "qa":
            df.loc[i, f"{text_column}_best_bleu_w_context"] = get_best_bleu_details(context, text)[2]
            if f"syn_{text_column}" in df.columns:
                syn_text = df.loc[i, f"syn_{text_column}"]
                df.loc[i, f"syn_{text_column}_best_bleu_w_context"] = get_best_bleu_details(context, syn_text)[2]
        else:
            location, location_ratio, confidence = get_best_bleu_details(context, text)
            df.loc[i, f"{text_column}_best_bleu_w_context"] = confidence
            df.loc[i, f"{text_column}_location"] = location
            df.loc[i, f"{text_column}_location_ratio"] = location_ratio
            if f"syn_{text_column}" in df.columns:
                syn_text = df.loc[i, f"syn_{text_column}"]
                syn_location, syn_location_ratio, syn_confidence = get_best_bleu_details(context, syn_text)
                df.loc[i, f"syn_{text_column}_best_bleu_w_context"] = syn_confidence
                df.loc[i, f"syn_{text_column}_location"] = syn_location
                df.loc[i, f"syn_{text_column}_location_ratio"] = syn_location_ratio


def add_bleu_w_answer(df, config):
    for i in tqdm(range(len(df))):
        answer = df.loc[i, "answer"]
        prediction = df.loc[i, "prediction"]
        df.loc[i, f"score"] = compute_bleu(answer, prediction)
    

def add_answer_location(df, config):
    for i in tqdm(range(len(df))):
        context = df.loc[i, "context"]
        text = df.loc[i, "answer"]
        location, loc_ratio, confidence = get_best_bleu_details(context, text)
        df.loc[i, "answer_location"] = location
        df.loc[i, "answer_location_ratio"] = loc_ratio
        df.loc[i, "answer_location_bleu"] = confidence
        if "syn_answer" in df.columns:
            syn_text = df.loc[i, "syn_answer"]
            syn_location, syn_loc_ratio, syn_confidence = get_best_bleu_details(context, syn_text)
            df.loc[i, "syn_answer_location"] = syn_location
            df.loc[i, "syn_answer_location_ratio"] = syn_loc_ratio
            df.loc[i, "syn_answer_location_bleu"] = syn_confidence
        if "prediction" in df.columns:
            pred = df.loc[i, "prediction"]
            pred_location, pred_loc_ratio, pred_confidence = get_best_bleu_details(context, text)
            df.loc[i, "pred_location"] = pred_location
            df.loc[i, "pred_location_ratio"] = pred_loc_ratio
            df.loc[i, "pred_location_bleu"] = pred_confidence


def determine_question_word(question):
    for question_word in ["what", "who", "where", "when", "why", "how", "which"]:
        if question_word in question.lower():
            return question_word
    return None

def add_question_word(df, config):
    for i in tqdm(range(len(df))):
        question = df.loc[i, "question"]
        question_word = determine_question_word(question)
        df.loc[i, "question_word"] = question_word
        if "syn_question" in df.columns:
            syn_question = df.loc[i, "syn_question"]
            syn_question_word = determine_question_word(syn_question)
            df.loc[i, "syn_question_word"] = syn_question_word



if __name__ == "__main__":
    main()