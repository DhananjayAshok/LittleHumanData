from prompt_models import pd, PromptGenerator, data_dir, OpenAIGPT
import os
from tqdm import tqdm
import json
import warnings
import numpy as np


np.random.seed(42)
if not os.path.exists(data_dir + "/generated"):
    os.makedirs(data_dir + "/generated")
if not os.path.exists(data_dir + "/generated/train"):
    os.makedirs(data_dir + "/generated/train")
if not os.path.exists(data_dir + "/generated/valid"):
    os.makedirs(data_dir + "/generated/valid")


def parse_qa_output(output):
    # easiest case is when there is a label or answer in the line:
    if "Label" in output and output.count("Label") == 1:
        question_line, answer_line = output.split("Label")
    elif "Verdict" in output and output.count("Verdict") == 1:
        question_line, answer_line = output.split("Verdict")
    elif "Fact" in output and output.count("Fact") == 1:
        question_line, answer_line = output.split("Fact")
    elif "Answer" in output and output.count("Answer") == 1:
        question_line, answer_line = output.split("Answer")
    elif output.find("|") != -1:
        lines = output.split("|") # keep nonempty lines
        keeplines = [line for line in lines if line.strip() != ""]
        if len(keeplines) == 1:
            if "False Claim".lower() in keeplines[0].lower() or "Untrue Claim".lower() in keeplines[0].lower():
                question_line = keeplines[0].split("Claim: ")[1]
                answer_line = "False"
            elif "True Claim".lower() in keeplines[0].lower():
                question_line = keeplines[0].split("Claim: ")[1]
                answer_line = "True"
            else:
                warnings.warn("Could not parse QA output: " + output)
                return None, None
        else:
            question_line = keeplines[0]
            # answer line is the line with most words in it after the first line
            sizes = [len(line.split()) for line in keeplines[1:]]
            answer_line = keeplines[sizes.index(max(sizes)) + 1]
    else:
        if output.count("\n") == 1:
            question_line, answer_line = output.split("\n")
        else:
            warnings.warn("Could not parse QA output: " + output)
            return None, None
    # strip empty spaces, \n and colons
    question_line = question_line.strip().strip(":")
    answer_line = answer_line.strip().strip(":")
    first_colon = question_line.find(":")
    question = question_line[first_colon + 1:].strip()
    first_colon = answer_line.find(":")
    answer = answer_line[first_colon + 1:].strip()
    question = question.strip("||")
    answer = answer.strip("||")
    return question, answer

#["true", "yes", "correct", "right"]
# "no", "incorrect", "wrong", "misleading", "inaccurate", "unsubstantiated", "unverified"]
def parse_fv_output(output, true_words=["true", "correct", "right"], false_words=["untrue", "false", "wrong", "misleading", "inaccurate", "unsubstantiated", "unverified", "unsupported", "not supported", "untruth", "unthinkable"]):
    claim, label = parse_qa_output(output)
    if claim is None or label is None:
        return None, None
    istrue = None
    isfalse = None
    for word in false_words:
        if word in label.lower():
            isfalse = True
            break
    for word in true_words:
        if word in label.lower() and "not " + word not in label.lower() and "un"+word not in label.lower():
            istrue = True
            break
    if istrue and isfalse:
        warnings.warn(f"Both true and false words found in label: {output} split into: {claim} ||| {label}")
        return None, None
    if istrue:
        label = 1
    elif isfalse:
        label = 0
    else:
        warnings.warn(f"Neither true nor false words found in label: {output} split into: {claim} ||| {label}")
        return None, None
    return claim, label

def parse_cot_qa_output(output):
    # easiest case is when there is a label or answer in the line:
    expquestion_line = None
    for label_key in ["Label", "Verdict", "Fact", "Answer"]:
        if label_key in output and output.count(label_key) == 1:
            expquestion_line, answer_line = output.split(label_key)
            break
    explanation_line = None
    if expquestion_line is not None:
        for question_key in ["Question", "Claim"]:
            if question_key in output and output.count(question_key) == 1:
                explanation_line, question_line = expquestion_line.split(question_key)
                break
        if explanation_line is None:
            if expquestion_line.strip("||").find("||") == 1:
                explanation_line, question_line = expquestion_line.split("||")
            else:
                raise ValueError("Could not find explanation line in COT QA output: " + output)
    elif output.find("||") != -1:
        lines = output.split("||") # keep nonempty lines
        keeplines = [line for line in lines if line.strip() != ""]
        assert len(keeplines) == 3
        explanation_line = keeplines[0]
        question_line = keeplines[1]
        answer_line = keeplines[2]
    else:
        if output.count("\n") == 2:
            explanation_line, question_line, answer_line = output.split("\n")
        else:
            warnings.warn("Could not parse QA output: " + output)
            return None, None, None
    # strip empty spaces, \n and outside colons
    question_line = question_line.strip().strip(":")
    answer_line = answer_line.strip().strip(":")
    first_colon = explanation_line.find(":")
    explanation = explanation_line[first_colon + 1:].strip()
    first_colon = question_line.find(":")
    question = question_line[first_colon + 1:].strip()
    first_colon = answer_line.find(":")
    answer = answer_line[first_colon + 1:].strip()
    explanation = explanation.strip("||")
    question = question.strip("||")
    answer = answer.strip("||")
    return explanation, question, answer

def parse_cot_fv_output(output, true_words=["true", "correct", "right"], false_words=["untrue", "false", "wrong", "misleading", "inaccurate", "unsubstantiated", "unverified"]):
    explanation, claim, label = parse_cot_qa_output(output)
    if explanation is None or claim is None or label is None:
        return None, None
    istrue = None
    isfalse = None
    for word in false_words:
        if word in label.lower():
            isfalse = True
            break
    for word in true_words:
        if word in label.lower() and "not " + word not in label.lower() and "un"+word not in label.lower():
            istrue = True
            break
    if istrue and isfalse:
        warnings.warn(f"Both true and false words found in label: {output} split into: {claim} ||| {label}")
        return None, None
    if istrue:
        label = 1
    elif isfalse:
        label = 0
    else:
        warnings.warn(f"Neither true nor false words found in label: {output} split into: {claim} ||| {label}")
        return None, None
    return explanation, claim, label



def run_gen(dataset_name, split, k=3, checkpoint_every=100):
    dataset_name = dataset_name.replace(".csv", "")
    pg = PromptGenerator(dataset_name, k)
    df = pg.df.sample(frac=0.01).reset_index(drop=True)
    task = pg.task
    checkpoint_path = data_dir + f"/generated/{split}/" + dataset_name + ".csv"
    if os.path.exists(checkpoint_path):
        df = pd.read_csv(checkpoint_path)
        checkpoint_idx = df["checkpoint_idx"].iloc[0]
        if checkpoint_idx == len(df):
            print(f"Dataset {dataset_name} already generated split {split}...")
            return df
    else:
        df["checkpoint_idx"] = 0        
        if task == "qa":
            df["syn_question"] = ""
            df["syn_answer"] = ""
        elif task == "fv":
            df["syn_claim"] = ""
            df["syn_label"] = ""
        checkpoint_idx = 0
    OpenAIGPT.reset_usage_track()
    print(f"Running Gen for: {dataset_name} split: {split}")
    if checkpoint_idx != 0:
        print(f"Resuming from checkpoint index: {checkpoint_idx}...")
    for i in tqdm(range(checkpoint_idx, len(df))):
        context = df.context.iloc[i]
        prompt = pg.get_prompt(context)
        output = OpenAIGPT.request_model(prompt)
        if task == "qa":
            question, answer = parse_qa_output(output)
            df.at[i, "syn_question"] = question
            df.at[i, "syn_answer"] = answer
        elif task == "fv":
            claim, label = parse_fv_output(output)
            df.at[i, "syn_claim"] = claim
            df.at[i, "syn_label"] = label
        df.loc[0, "checkpoint_idx"] = i
        if i % checkpoint_every == 0:
            df.to_csv(checkpoint_path, index=False)
    df["checkpoint_idx"] = len(df)
    df.to_csv(checkpoint_path, index=False)
    OpenAIGPT.print_usage()
    return df


def send_gen_batch(dataset_name, split, k=3, cot=False, max_points=None):
    dataset_name = dataset_name.replace(".csv", "")
    checkpoint_path = data_dir + f"/generated/{OpenAIGPT.model}/cot_{cot}/{split}/" + dataset_name + ".csv"
    if os.path.exists(checkpoint_path):
        df = pd.read_csv(checkpoint_path)
        checkpoint_idx = df["checkpoint_idx"].iloc[0]
        if checkpoint_idx == len(df):
            print(f"Dataset {dataset_name} already generated split {split}... Not sending batch unless you delete the file")
            return df
    batchdest = data_dir+f"/generated/{OpenAIGPT.model}/cot_{cot}/{split}/batches/" + dataset_name + "/"
    if not os.path.exists(batchdest):
        os.makedirs(batchdest)
    pg = PromptGenerator(dataset_name, cot=cot, k=3)
    if max_points is None or max_points > len(pg.df):
        df = pg.df
    else:
        df = pg.df.sample(n=max_points).reset_index(drop=True)
    nested_msg_list = []
    for i in range(len(df)):
        context = df.context.iloc[i]
        if pg.task == "fv":
            label = df.label.iloc[i]
        else:
            label = None
        prompt = pg.get_prompt(i, context, label=label)
        nested_msg_list.append(prompt)
    # split nested_msg_list into batches of 20% data
    per = int(len(nested_msg_list) * 0.20)
    for i in range(5):
        batch = nested_msg_list[i * per: (i + 1) * per]
        batch_file_path = f"{batchdest}/batch{i}.jsonl"
        OpenAIGPT.do_batch(batch, batch_file_path)
    print(f"Batch {dataset_name} split {split} sent to OpenAI. Batch info saved at: {batchdest}")
    return

def parse_batch_results(dataset_name, split, cot=False, max_points=None):
    dataset_name = dataset_name.replace(".csv", "")
    checkpoint_path = data_dir + f"/generated/{OpenAIGPT.model}/cot_{cot}/{split}/" + dataset_name + ".csv"
    if os.path.exists(checkpoint_path):
        df = pd.read_csv(checkpoint_path)
        checkpoint_idx = df["checkpoint_idx"].iloc[0]
        if checkpoint_idx >= len(df):
            print(f"Dataset {dataset_name} already generated split {split}... Calling parse_batch_results would overwrite this file")
            return df
    pg = PromptGenerator(dataset_name, cot=cot, k=3)
    if max_points is None or max_points > len(pg.df):
        df = pg.df
    else:
        df = pg.df.sample(n=max_points).reset_index(drop=True)
    task = pg.task
    batch_file_path = data_dir + f"/generated/{OpenAIGPT.model}/cot_{cot}/{split}/batches/" + dataset_name + "/batch[N].jsonl"
    dfs = []
    add_acc = 0
    for i in range(5):
        batch_file_instance = batch_file_path.replace("[N]", str(i))
        results = OpenAIGPT.get_batch_results(batch_file_instance)
        answer_df = pd.DataFrame(data=[json.loads(line) for line in results.text.strip().split('\n')])
        answer_df["text"] = answer_df["response"].apply(lambda x: x["body"]["choices"][0]["message"]["content"])
        answer_df["custom_id"] = answer_df["custom_id"].astype(int) + add_acc
        add_acc += len(answer_df)
        dfs.append(answer_df)
    answer_df = pd.concat(dfs, axis=0, ignore_index=True).reset_index(drop=True)
    for j in tqdm(range(len(answer_df))):
        item = answer_df.iloc[j]
        i = int(item["custom_id"])
        output = item["text"]
        if task == "qa":
            if not cot:
                question, answer = parse_qa_output(output)
            else:
                explanation, question, answer = parse_cot_qa_output(output)
            df.at[i, "syn_question"] = question
            df.at[i, "syn_answer"] = answer
        elif task == "fv":
            if not cot:
                claim, label = parse_fv_output(output)
            else:
                explanation, claim, label = parse_cot_fv_output(output)
            df.at[i, "syn_claim"] = claim
            df.at[i, "syn_label"] = label
    df["checkpoint_idx"] = len(df)
    print(f"Completed Generation. Has {len(df)}")
    if task == "qa":
        nan_cols = (df["syn_question"].isna()) | (df["syn_answer"].isna())
        print(f"Missing {nan_cols.sum()} questions or answers")
    elif task == "fv":
        nan_cols = (df["syn_claim"].isna()) | (df["syn_label"].isna())
        print(f"Missing {nan_cols.sum()} claims or labels")
    print("Dropping rows with missing values...")
    df = df[~nan_cols].reset_index(drop=True)
    if "syn_label" in df.columns:
        df["syn_label"] = df["syn_label"].astype(int)
    print(f"Saving to {checkpoint_path}...")
    df.to_csv(checkpoint_path, index=False)
    return df

def run_all_gen():
    datasets = os.listdir(data_dir + "/train")
    for dataset in datasets:
        run_gen(dataset, "train")
        breakpoint()
        run_gen(dataset, "valid")
    return



if __name__ == "__main__":
    #for dset in ["wanli", "fever", "scifact", "fairytaleqa", "ropes"]:
    #    send_gen_batch(dset, "train", cot=True, max_points=5005)

    #for dset in ["wanli", "fever", "scifact", "fairytaleqa", "ropes"]:
    #    parse_batch_results(dset, "train", cot=True, max_points=5005)

    parse_batch_results("wanli", "train", cot=True, max_points=10)
    pass