from datasets import load_dataset, DatasetDict, Dataset

# text data
wikisum_ds = load_dataset("zhengxuanzenwu/wikitext-2-split-128")
text_train = []
for example in wikisum_ds["train"]:
    text_train += [example["text"]]

ag_ds = load_dataset("fancyzhx/ag_news")
text_test = []
for example in ag_ds["train"]:
    text_test += [example["text"]]

# math data
gsm_ds = load_dataset("openai/gsm8k", "main")
math_train = []
for example in gsm_ds["train"]:
    math_train += [example["answer"]]

comp_ds = load_dataset("qwedsacf/competition_math")
math_test = []
for example in comp_ds["train"]:
    math_test += [example["problem"]]

# code data
code_ds = load_dataset("christopher/rosetta-code")
code_all = []
for example in code_ds["train"]:
    if len(example["code"]) > 500:
        code_all += [example["code"][:500]]
code_train = code_all[:len(code_all)//2]
code_test = code_all[len(code_all)//2:]

data = {
    "text_train": text_train[:1000],
    "text_test": text_test[:1000],
    "math_train": math_train[:1000],
    "math_test": math_test[:1000],
    "code_train": code_train[:1000],
    "code_test": code_test[:1000],
}
dataset = DatasetDict({
    "text_train": Dataset.from_dict({"input": data["text_train"]}),
    "text_test": Dataset.from_dict({"input": data["text_test"]}),
    "math_train": Dataset.from_dict({"input": data["math_train"]}),
    "math_test": Dataset.from_dict({"input": data["math_test"]}),
    "code_train": Dataset.from_dict({"input": data["code_train"]}),
    "code_test": Dataset.from_dict({"input": data["code_test"]}),
})

dataset.save_to_disk("seed_sentences")


# text instructions
dolly_ds = load_dataset("databricks/databricks-dolly-15k")
text_train = []
for example in dolly_ds["train"]:
    if example["category"] == "open_qa" and example["context"] == "":
        if len(example["instruction"]) < 500:
            text_train += [example["instruction"]]

# math instructions
gsm_ds = load_dataset("openai/gsm8k", "main")
math_train = []
for example in gsm_ds["train"]:
    if len(example["question"]) < 500:
        math_train += [example["question"]]

# code instructions
alpaca_ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca")
code_train = []
for example in alpaca_ds["train"]:
    if example["input"] == "":
        if len(example["instruction"]) < 500:
            code_train += [example["instruction"]]

data = {
    "text_train": text_train[:1000],
    "math_train": math_train[:1000],
    "code_train": code_train[:1000],
    "text_test": text_train[1000:2000],
    "math_test": math_test[1000:2000],
    "code_test": code_test[1000:2000],
}
dataset = DatasetDict({
    "text_train": Dataset.from_dict({"input": data["text_train"]}),
    "text_test": Dataset.from_dict({"input": data["text_test"]}),
    "math_train": Dataset.from_dict({"input": data["math_train"]}),
    "math_test": Dataset.from_dict({"input": data["math_test"]}),
    "code_train": Dataset.from_dict({"input": data["code_train"]}),
    "code_test": Dataset.from_dict({"input": data["code_test"]}),
})

dataset.save_to_disk("seed_instructions")