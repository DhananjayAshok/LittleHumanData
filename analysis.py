import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import click
import numpy as np
import os

dataset_dict={"factify": "Factify", "scifact": "SciFact", "wanli": "WANLI", "fever": "FEVER", "ropes": "ROPES", "qaconv": "QAConv", "coqa": "CoQA", "narrativeqa": "NarrativeQA", "fairytaleqa": "FairytaleQA"}

def guarded_split(x):
    if not isinstance(x, str):
        return []
    return x.split()


sns.set_theme()
#options are path dataset model cot 
@click.command()
@click.option('--path', '-p', help='Path to the results csv file')
def main(path):
    df = pd.read_csv(path)
    task = df["task"].iloc[0]
    path_breakdown = path.split("analysis/")[-1]
    model = path_breakdown.split("/")[0]
    cot = "True" in path_breakdown.split("/")[1]
    dataset = path_breakdown.split("/")[2]
    for item in dataset_dict:
        if item in dataset:
            dataset = item
            break
    figpath=f"data/analysis/{model}/cot_{cot}/{dataset}/"
    if not os.path.exists(os.path.dirname(figpath)):
        os.makedirs(figpath+"generated/", exist_ok=True)
        os.makedirs(figpath+"predictions/real/", exist_ok=True)
        os.makedirs(figpath+"predictions/synthetic/", exist_ok=True)
    dataset = dataset_dict[dataset]
    title = f"{dataset}: {model} {'with CoT' if cot else ''}"
    if "syn_question" in df.columns or "syn_claim" in df.columns:
        do_genfile_analysis(df, task, figpath+"generated/", title)
    else:
        pathsuff = "real"
        if "1.0" in path:
            pathsuff = "synthetic"
        do_predfile_analysis(df, task, figpath+"/predictions/"+pathsuff+"/", title)


def synth_print_hist(df, col, displayname, figpath, title):
    syn_col = f"syn_{col}"
    print(f"Real {displayname}")
    print(f"{df[col].describe()}")
    print(f"Synthetic {displayname}")
    print(f"{df[syn_col].describe()}")
    df[col].hist(alpha=0.5, label="Real", weights=np.ones(len(df)) / len(df))
    df[syn_col].hist(alpha=0.5, label="Synthetic", weights=np.ones(len(df)) / len(df))
    plt.title(title)
    plt.xlabel(displayname)
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"{figpath}/{displayname} Histogram.png")
    plt.clf()

def good_bad_print_hist(good_df, bad_df, col, displayname, figpath, title):
    print(f"{displayname} for Top 25%")
    print(f"{good_df[col].describe()}")
    print(f"{displayname} for Bottom 25%")
    print(f"{bad_df[col].describe()}")
    good_df[col].hist(alpha=0.5, label="Top 25%", weights=np.ones(len(good_df)) / len(good_df))
    bad_df[col].hist(alpha=0.5, label="Bottom 25%", weights=np.ones(len(bad_df)) / len(bad_df))
    plt.title(title)
    plt.xlabel(displayname)
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"{figpath}/{displayname} Histogram.png")
    plt.clf()


def do_genfile_analysis(df, task, figpath, title):
    if task == "qa":
        print(f"Question Words:")
        print(100*df["question_word"].value_counts(normalize=True))
        print(f"Synthetic Question Words:")
        print(100*df["syn_question_word"].value_counts(normalize=True))
        print(f"Question lengths:")
        # make the histogram transparent and red
        df["Question Length"] = df["question"].apply(lambda x: len(guarded_split(x)))
        df["syn_Question Length"] = df["syn_question"].apply(lambda x: len(guarded_split(x)))
        synth_print_hist(df, "Question Length", "Question Length", figpath, title)
        print(f"Answer lengths:")
        df["Answer Length"] = df["answer"].apply(lambda x: len(guarded_split(x)))
        df["syn_Answer Length"] = df["syn_answer"].apply(lambda x: len(guarded_split(x)))
        synth_print_hist(df, "Answer Length", "Answer Length", figpath, title)
        synth_print_hist(df, "question_best_bleu_w_context","Question BLEU with Evidence Text (Best)", figpath, title)
        synth_print_hist(df, "answer_location_bleu" ,"Answer BLEU with Evidence Text (Best)", figpath, title)
        synth_print_hist(df,"answer_location_ratio","Relative Position of Answer in Evidence Text", figpath, title)
    else:
        print(f"Claim lengths:")
        df["Claim Length"] = df["claim"].apply(lambda x: len(guarded_split(x)))
        df["syn_Claim Length"] = df["syn_claim"].apply(lambda x: len(guarded_split(x)))
        synth_print_hist(df, "Claim Length", "Claim Length", figpath, title)
        synth_print_hist(df, "claim_best_bleu_w_context", "Claim BLEU with Evidence Text (Best)", figpath, title)
        synth_print_hist(df, "claim_location_ratio", "Relative Position of Claim in Evidence Text", figpath, title)
    show_ngram_counts(df, task)

def do_predfile_analysis(df, task, figpath, title):
    n_points = len(df)
    df = df.sort_values("score", ascending=False) # just in case but should be done already
    best_df = df[:int(n_points * 0.25)]
    worst_df = df[int(n_points * 0.75):]
    print(f"Difference of score means between best and worst quartiles")
    print(best_df["score"].mean() - worst_df["score"].mean())
    print(f"Difference of medians")
    print(best_df["score"].median() - worst_df["score"].median())
    good_bad_print_hist(best_df, worst_df, "score", "Score", figpath, title)
    if task == "qa":
        print(f"Question Words:")
        print((100*df["question_word"].value_counts(normalize=True)))
        print(f"Question Words for Best and Worst Quartiles")
        print(100*best_df["question_word"].value_counts(normalize=True))
        print(100*worst_df["question_word"].value_counts(normalize=True))

        for col in ["question", "answer", "prediction"]:
            best_df[f"{col}_length"] = best_df[col].apply(lambda x: len(guarded_split(x)))
            worst_df[f"{col}_length"] = worst_df[col].apply(lambda x: len(guarded_split(x)))
            good_bad_print_hist(best_df, worst_df, f"{col}_length", f"{col[0].upper()+col[1:]} Length", figpath, title)
        coltodo = [("answer_location_ratio", "Answer Relative Position in Evidence Text (% from start)"), ("pred_location_ratio", "Prediction Relative Position in Evidence Text (% from start)"), ("question_best_bleu_w_context", "Question BLEU with Evidence Text (Best)"), ("answer_location_bleu", "Answer BLEU with Evidence Text (Best)"), ("pred_location_bleu", "Prediction BLEU with Evidence Text (Best)")]
        for col, displayname in coltodo:
            good_bad_print_hist(best_df, worst_df, col, displayname, figpath, title)

    else:
        print(f"Label Base Rate: \n{df['label'].value_counts(normalize=True)}")
        print(f"Score grouped by label")
        print(df.groupby("label")["score"].describe())
        print(f"Score grouped by prediction")
        df["pred"] = (df["prediction"] > 0.5).astype(int)
        print(df.groupby("pred")["score"].describe())
        best_df["claim_length"] = best_df["claim"].apply(lambda x: len(guarded_split(x)))
        worst_df["claim_length"] = worst_df["claim"].apply(lambda x: len(guarded_split(x)))
        good_bad_print_hist(best_df, worst_df, "claim_best_bleu_w_context", "Claim BLEU with Evidence Text (Best)", figpath, title)
        good_bad_print_hist(best_df, worst_df, "claim_location_ratio", "Claim Relative Position in Evidence Text (% from start)", figpath, title)
        good_bad_print_hist(best_df, worst_df, "claim_length", "Claim Length", figpath, title)


def get_n_gram(text, n=4):
    ngrams = []
    words = guarded_split(text)
    for i in range(len(words) - n + 1):
        ngram = " ".join(words[i:i+n])
        ngrams.append(ngram)
    return ngrams

# dataframe wide
def compute_mode_ngrams(df, col, top_n=10):
    ngrams = []
    for i in range(len(df)):
        ngrams += get_n_gram(df.loc[i, col])
    ngram_counts = pd.Series(ngrams).value_counts()
    return (100*ngram_counts / ngram_counts.sum())[:top_n]

def show_ngram_counts(df, task):
    if task == "qa":
        text_column = "question"
    else:
        text_column = "claim"
    print(f"Top ngrams % not decimal")
    print(text_column)
    print(compute_mode_ngrams(df, text_column))
    if f"syn_{text_column}" in df.columns:
        print(f"Synthetic {text_column}")
        print(compute_mode_ngrams(df, f"syn_{text_column}"))
    if task == "qa":
        print("Answer")
        print(compute_mode_ngrams(df, "answer"))
        if "syn_answer" in df.columns:
            print("Synthetic Answer")
            print(compute_mode_ngrams(df, "syn_answer"))
        if "prediction" in df.columns:
            print("Prediction")
            print(compute_mode_ngrams(df, "prediction"))

if __name__ == "__main__":
    main()

