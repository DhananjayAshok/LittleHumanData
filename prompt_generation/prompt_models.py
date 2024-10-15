from openai import OpenAI
import time
import yaml
import numpy as np
import pandas as pd
import json
import re

def remove_urls(text):
    # Regular expression pattern to match URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    # Substitute URLs with an empty string
    cleaned_text = url_pattern.sub('', text)
    return cleaned_text

np.random.seed(42)
with open("../proj_params.yml", "r") as f:
    proj_params = yaml.safe_load(f)

data_dir = proj_params["data_path"]


def tok_len(text):
    return len(text.split())


class OpenAIGPT:
    #model = "gpt-4o"
    model = "gpt-3.5-turbo"
    seconds_per_query = (60 / 20) + 0.01
    max_tokens=120
    max_retries=3
    verbose = False
    client = OpenAI()

    @staticmethod
    def openai_api_calculate_cost(message):
        model = message.model
        pricing = { # price per million tokens in USD
            'gpt-3.5-turbo-0125': {
                'prompt': 0.5,
                'completion': 1.5,
            },
            "gpt-4-turbo": {
                'prompt': 10,
                'completion': 30,
            },            
        }
        usage = message.usage
        model_pricing = pricing[model]
        million = 1_000_000
        prompt_cost = usage.prompt_tokens * (model_pricing['prompt'] /  million)
        completion_cost = usage.completion_tokens * (model_pricing['completion'] / million)
        return usage.prompt_tokens, usage.completion_tokens, prompt_cost, completion_cost

    @staticmethod
    def reset_usage_track():
        OpenAIGPT.prompt_tokens = 0
        OpenAIGPT.completion_tokens = 0
        OpenAIGPT.prompt_cost = 0
        OpenAIGPT.completion_cost = 0

    @staticmethod
    def get_usage():
        return {
            "prompt_tokens": OpenAIGPT.prompt_tokens,
            "completion_tokens": OpenAIGPT.completion_tokens,
            "prompt_cost": OpenAIGPT.prompt_cost,
            "completion_cost": OpenAIGPT.completion_cost,
            "total_cost": OpenAIGPT.prompt_cost + OpenAIGPT.completion_cost
        }

    @staticmethod
    def print_usage():
        usage = OpenAIGPT.get_usage()
        print(f"Prompt tokens: {usage['prompt_tokens']}")
        print(f"Completion tokens: {usage['completion_tokens']}")
        print(f"Prompt cost: {usage['prompt_cost']}")
        print(f"Completion cost: {usage['completion_cost']}")
        print(f"Total cost: {usage['total_cost']}")

    @staticmethod
    def request_model(msgs):
        """
        msgs is a list with role and content: e.g. [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
        """
        message = OpenAIGPT.client.chat.completions.create(model=OpenAIGPT.model, messages=msgs, stream=False, temperature=0)
        output = message.choices[0].message.content
        prompt_tokens, completion_tokens, prompt_cost, completion_cost = OpenAIGPT.openai_api_calculate_cost(message)
        OpenAIGPT.prompt_tokens += prompt_tokens
        OpenAIGPT.completion_tokens += completion_tokens
        OpenAIGPT.prompt_cost += prompt_cost
        OpenAIGPT.completion_cost += completion_cost
        return output

    @staticmethod
    def execute(msgs, max_new_tokens=50):
        if OpenAIGPT.verbose:
            for i, msg in enumerate(msgs):
                # print the role in green and text in blue
                # set print color to green
                color = "\033[92m" if i < len(msgs) - 1 else "\033[94m"
                print(color)
                print(f"{msg['role']}: {msg['content']}")
                # reset print color to default
                print("\033[0m")
        retries = 0
        OpenAIGPT.max_tokens = max_new_tokens
        while retries < OpenAIGPT.max_retries:
            try:
                output = OpenAIGPT.request_model(msgs)
                if OpenAIGPT.verbose:
                    print("\033[91m")
                    print(f"{output}")
                    print("\033[0m")
                return output
            except Exception as e:
                print(e)
                print(f"Sleeping for {OpenAIGPT.seconds_per_query} seconds")
                time.sleep(OpenAIGPT.seconds_per_query)
                retries += 1
            else:
                print(f"Shouldn't be getting here")
                retries = OpenAIGPT.max_retries
        raise ValueError(f"Failed to get response after {OpenAIGPT.max_retries} retries")
    
    
    @staticmethod
    def build_batch_file(nested_msg_list, batch_file_path):
        data = []
        columns = ["custom_id", "method", "url", "body"]
        url = "/v1/chat/completions"
        method = "POST"
        for i, msgs in enumerate(nested_msg_list):
            custom_id = f"{i}"
            body = {"model": OpenAIGPT.model, "messages": msgs, "max_tokens": OpenAIGPT.max_tokens}
            data.append([custom_id, method, url, body])
        df = pd.DataFrame(data, columns=columns)
        df.to_json(batch_file_path, lines=True, orient='records')
        return
    
    @staticmethod
    def send_batch(batch_file_path):
        batch_input_file = OpenAIGPT.client.files.create(
            file=open(batch_file_path, "rb"),
            purpose="batch"
            )
        batch_input_file_id = batch_input_file.id
        batch_data = OpenAIGPT.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
            "description": "idk"
            }
        )
        batch_id = batch_data.id
        # write the batch_id and batch_input_file_id to a json file
        batch_info = {"batch_id": batch_id, "batch_input_file_id": batch_input_file_id}
        batch_info_path = batch_file_path.replace(".jsonl", "_info.json")
        with open(batch_info_path, "w") as f:
            json.dump(batch_info, f)
        return batch_info_path
    
    @staticmethod
    def do_batch(nested_msg_list, batch_file_path):
        OpenAIGPT.build_batch_file(nested_msg_list, batch_file_path)
        batch_info_path = OpenAIGPT.send_batch(batch_file_path)
        return batch_info_path
    
    @staticmethod
    def list_batches():
        batches = OpenAIGPT.client.batches.list()
        return batches

    
    @staticmethod
    def get_batch(batch_file_path):
        batch_info_path = batch_file_path.replace(".jsonl", "_info.json")
        with open(batch_info_path, "r") as f:
            batch_info = json.load(f)
        batch_id = batch_info["batch_id"]
        batch = OpenAIGPT.client.batches.retrieve(batch_id)
        return batch
    
    @staticmethod
    def get_batch_results(batch_file_path):
        batch = OpenAIGPT.get_batch(batch_file_path)
        output_file_id = batch.output_file_id
        file_response = OpenAIGPT.client.files.content(output_file_id)

        return file_response


def truncate_context(context, max_tokens=3000):
    return " ".join(context.split()[:max_tokens])

class PromptGenerator:
    def __init__(self, dataset_name, k=3, cot=False):
        self.k = k
        self.dataset_name = dataset_name
        self.df = pd.read_csv(data_dir + "/train/" + dataset_name.replace(".csv", "") + ".csv")
        if self.df.dataset_name.iloc[0] == "narrativeqa":
            self.df = self.df.dropna(subset=["context", "question", "answer"]).reset_index(drop=True)
        self.task = self.df["task"].iloc[0]
        self.cot = cot
        if self.cot:
            self.cot_df = pd.read_csv("cot.csv")
            if dataset_name not in self.cot_df.dataset_name.unique():
                raise ValueError(f"Dataset {dataset_name} not found in cot.csv")
            self.cot_df = self.cot_df[self.cot_df.dataset_name == dataset_name].reset_index(drop=True)
        self.prompt_temp = f"Context: [CONTEXT]"
        if self.task == "qa":
            if self.cot:
                self.system_prompt = "Given the context, come up with a question and answer pair. Explain why it is a good example before providing it in the format below:"  
                self.ans_temp = f"Explanation: [EXPLANATION]||\nQuestion: [QUESTION]||\nAnswer: [ANSWER]"
            else:
                self.system_prompt = "Given the context, come up with a question and answer pair. Answer in the format below:"
                self.ans_temp = f"Question: [QUESTION]||\nAnswer: [ANSWER]"
        elif self.task == "fv":
            if self.cot:
                self.system_prompt = "Given the context, come up with a [LABEL] claim. Explain why it is a good example before providing it in the format below:"
                self.ans_temp = f"Explanation: [EXPLANATION]||\nClaim: [CLAIM]||\nLabel: [LABEL]"
            else:
                self.system_prompt = "Given the context, come up with a [LABEL] claim. Answer in the format below:"
                self.ans_temp = f"Claim: [CLAIM]||\nLabel: [LABEL]"


    def get_examples(self, i_avoid, label=None):
        # get a dataframe without i_avoid index only
        sample_df = self.df.drop(i_avoid)
        system_prompt = self.system_prompt
        if label is not None:
            sample_df = sample_df[sample_df["label"] == label]
            system_prompt = system_prompt.replace("[LABEL]", "True" if label == 1 else "Untrue (either False or just not supported)")
        examples = sample_df.sample(self.k).reset_index(drop=True)
        message = [{"role": "system", "content": system_prompt}]
        for i in range(self.k):
            context = examples.context.iloc[i]
            mdict = {"role": "user", "content": remove_urls(truncate_context(self.prompt_temp.replace("[CONTEXT]", context)))}
            message.append(mdict)
            if self.task == "qa":
                question = examples.question.iloc[i]
                answer = examples.answer.iloc[i]
                mdict = {"role": "assistant", "content": remove_urls(self.ans_temp.replace("[QUESTION]", question).replace("[ANSWER]", answer))}
                message.append(mdict)
            elif self.task == "fv":
                claim = examples.claim.iloc[i]
                label = examples.label.iloc[i]
                if label == 1:
                    label = "True"
                else:
                    label = "Untrue"
                mdict = {"role": "assistant", "content": remove_urls(self.ans_temp.replace("[CLAIM]", claim).replace("[LABEL]", label))}
                message.append(mdict)
        return message
    

    def get_cot_examples(self, i_avoid, context, label=None):
        system_prompt = self.system_prompt
        if label is not None:
            sample_df = self.cot_df[self.cot_df["label"] == label]
            system_prompt = system_prompt.replace("[LABEL]", "True" if label == 1 else "Untrue (either False or just not supported)")
        examples = sample_df.sample(self.k).reset_index(drop=True)
        message = [{"role": "system", "content": system_prompt}]
        for i in range(self.k):
            context = examples.context.iloc[i]
            explanation = examples.explanation.iloc[i]
            mdict = {"role": "user", "content": remove_urls(truncate_context(self.prompt_temp.replace("[CONTEXT]", context)))}
            message.append(mdict)
            if self.task == "qa":
                question = examples.question.iloc[i]
                answer = examples.answer.iloc[i]
                mdict = {"role": "assistant", "content": remove_urls(self.ans_temp.replace("[QUESTION]", question).replace("[ANSWER]", answer).replace("[EXPLANATION]", explanation))}
                message.append(mdict)
            elif self.task == "fv":
                claim = examples.claim.iloc[i]
                label = examples.label.iloc[i]
                if label == 1:
                    label = "True"
                else:
                    label = "Untrue"
                mdict = {"role": "assistant", "content": remove_urls(self.ans_temp.replace("[CLAIM]", claim).replace("[LABEL]", label).replace("[EXPLANATION]", explanation))}
                message.append(mdict)
        return message

    
    def get_prompt(self, i_avoid, context, label=None):
        assert label in [None, 0, 1], f"Label must be either 0 or 1, not {label}"
        if not self.cot:
            message = self.get_examples(i_avoid, label=label)
        else:
            message = self.get_cot_examples(i_avoid, context, label=label)
        message.append({"role": "user", "content": remove_urls(truncate_context(self.prompt_temp.replace("[CONTEXT]", context)))})
        return message


def test_batch():
    msgs = [{"role": "user", "content": "Explain quantum physics using fluid dynamics"}]
    other_msgs = [{"role": "user", "content": "Explain solar system as an onion"}]
    batch_file_path = "batch.jsonl"
    OpenAIGPT.do_batch([msgs, other_msgs], batch_file_path)
    return

def check_test():
    batch_file_path = "batch.jsonl"
    batch_info = OpenAIGPT.get_batch(batch_file_path)
    print(batch_info)
    return


def get_test():
    batch_file_path = "batch.jsonl"
    res = OpenAIGPT.get_batch_results(batch_file_path)
    print(res)
    return res

if __name__ == "__main__":
    test_batch()