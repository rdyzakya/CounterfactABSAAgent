from smolagents import CodeAgent
from smolagents import OpenAIServerModel
from dotenv import load_dotenv
import pandas as pd
import os
from tqdm import tqdm
from tools import check_num_tokens, tokenize, preprocess, split, check_num_tokens_fn
from argparse import ArgumentParser

load_dotenv()

parser = ArgumentParser()
parser.add_argument("--input-csv", "-i", type=str, help="Csv file input path", required=True)
parser.add_argument("--out-folder", "-o", type=str, help="Output folder", required=True)
args = parser.parse_args()

os.makedirs(args.out_folder, exist_ok=True)
_, filename = os.path.split(args.input_csv)

model = OpenAIServerModel(
    model_id=os.getenv("OPENAI_MODEL"),
    api_base="https://api.openai.com/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
)

agent = CodeAgent(
    tools=[check_num_tokens, tokenize], 
    model=model,
    additional_authorized_imports=['json']
)

# 1. change aspect
# 2. change opinion and sentiment

EXAMPLE = preprocess("Example text.", ("aspect term", "opinion term", "sentiment value"))

def prompt_aspect(input_text, input_tuple):
    result_format = '{"text" : "output text", "tuple" : ("aspect term", "opinion term", "sentiment value")}'

    # (you can even put some nonsense aspect term to replace the former one)
    
    prompt = f"""You are an agent that creates a counterfactual sentence given an aspect based sentiment analysis data (text and tuple pair). Below are the requirements:

1. work on aspect based sentiment analysis in the given language
2. receives a text and a tuple containing: (a) aspect term (b) opinion term (c) sentiment value
3. make a counterfactual version of the input. CHANGE ONLY THE ASPECT TERM in the text and the tuple.
4. THERE IS A CONSTRAINT: you should check the number of tokens between the preprocessed input and output using the given tool. always use the original text to compare to the counterfact version (output). if the token length is different, find another alternative.
5. as an example, this is how the preprocessed text looks like (you don't need to do this, this is automated by the `check_num_tokens` tool): `{EXAMPLE}`
6. give the final answer using this format => `{result_format}`

The text is `{input_text}` and the tuple is `{input_tuple}`"""
    return prompt

def prompt_opinion(input_text, input_tuple):
    result_format = '{"text" : "output text", "tuple" : ("aspect term", "opinion term", "sentiment value")}'

    # (you can even put some nonsense aspect term to replace the former one)
    
    prompt = f"""You are an agent that creates a counterfactual sentence given an aspect based sentiment analysis data (text and tuple pair). Below are the requirements:

1. work on aspect based sentiment analysis in the given language
2. receives a text and a tuple containing: (a) aspect term (b) opinion term (c) sentiment value
3. make a counterfactual version of the input. CHANGE THE OPINION TERM AND ADJUST THE SENTENCE to make it make sense with the sentiment value. It is not mandatory to make the new opinion term to be the opposite of the former one.
4. THERE IS A CONSTRAINT: you should check the number of tokens between the preprocessed input and output using the given tool. always use the original text to compare to the counterfact version (output). if the token length is different, find another alternative.
5. as an example, this is how the preprocessed text looks like (you don't need to do this, this is automated by the `check_num_tokens` tool): `{EXAMPLE}`
6. give the final answer using this format => `{result_format}`

The text is `{input_text}` and the tuple is `{input_tuple}`"""
    return prompt

df = pd.read_csv(args.input_csv)
fail = []
neutral = []

df_result = df.copy()

def save_ckpt():
    df_result.to_csv(os.path.join(args.out_folder, filename), index=False)
    with open(os.path.join(args.out_folder, filename.replace(".csv", "_fail.txt")), 'w') as fp:
        fp.write('\n'.join(list(map(str, fail))))
    with open(os.path.join(args.out_folder, filename.replace(".csv", "_neutral.txt")), 'w') as fp:
        fp.write('\n'.join(list(map(str, neutral))))

for i, row in tqdm(df.iterrows()):
    original_pair = row["original_pair"]
    corrupted_pair = row["corrupted_pair"]

    if isinstance(corrupted_pair, str):
        corrupted_pair = corrupted_pair.strip()

    if not pd.isnull(corrupted_pair):
        continue

    input_text, input_tuple = split(original_pair)
    result_aspect = agent.run(prompt_aspect(input_text, input_tuple))
    
    try:
        result_aspect = eval(result_aspect) if isinstance(result_aspect, str) else result_aspect
        output_text = result_aspect.get("text")
        output_tuple = result_aspect.get("tuple")
    except:
        fail.append(i)
        save_ckpt()
        continue

    aspect_report = check_num_tokens_fn(input_text, output_text, input_tuple, output_tuple)

    if not(aspect_report["is_equal_full_text"] and aspect_report["is_equal_aspect"] and aspect_report["is_equal_opinion"]):
        fail.append(i)
        save_ckpt()
        continue

    if output_tuple[2] == "positive":
        output_tuple = (output_tuple[0], output_tuple[1], "negative")
    elif output_tuple[2] == "negative":
        output_tuple = (output_tuple[0], output_tuple[1], "positive")
    elif output_tuple[2] == "neutral":
        neutral.append(i)
        save_ckpt()
        continue
    else:
        fail.append(i)
        save_ckpt()
        continue

    result_opinion = agent.run(prompt_opinion(output_text, output_tuple))

    try:
        result_opinion = eval(result_opinion) if isinstance(result_opinion, str) else result_opinion
        output_text = result_opinion.get("text")
        output_tuple = result_opinion.get("tuple")
    except:
        fail.append(i)
        save_ckpt()
        continue

    opinion_report = check_num_tokens_fn(input_text, output_text, input_tuple, output_tuple)

    if not(opinion_report["is_equal_full_text"] and opinion_report["is_equal_aspect"] and opinion_report["is_equal_opinion"]):
        fail.append(i)
        save_ckpt()
        continue

    df_result.loc[i, "corrupted_pair"] = preprocess(output_text, output_tuple)
    save_ckpt()

    

# def create_prompt_b(input_text, input_tuple):
#     result_format = r'{"text" : "output text", "tuple" : "output tuple"}'
#     prompt = f"""You are an agent that creates a counterfactual sentence given an aspect based sentiment analysis data (text and tuple pair). Below are the requirements:

# 1. work on aspect based sentiment analysis
# 2. receives a text and a tuple containing: (a) aspect term (b) opinion term (c) sentiment value
# 3. make a counterfactual version of the input. change only the opinion term in the text and the tuple (without changing the sentiment value).
# 4. there is a constraint: you should check the number of tokens between the preprocessed input and output using the given tool. always use the original text to compare to the counterfact version (output). if the token length is different, find another alternative.
# 5. as an example, this is how the preprocessed text looks like (you don't need to do this, this is automated by the tool): `{EXAMPLE}`
# 6. give the final answer using this format => `{result_format}`

# The text is `{input_text}` and the tuple is `{input_tuple}`"""
#     return prompt

# def create_prompt_c(input_text, input_tuple):
#     result_format = r'{"text" : "output text", "tuple" : "output tuple"}'
#     prompt = f"""You are an agent that creates a counterfactual sentence given an aspect based sentiment analysis data (text and tuple pair). Below are the requirements:

# 1. work on aspect based sentiment analysis
# 2. receives a text and a tuple containing: (a) aspect term (b) opinion term (c) sentiment value
# 3. make a counterfactual version of the input. change the sentiment value (sometimes this makes the opinion term or any other components in the sentence needs to be adjusted to fit the sentiment value and make the sentence make sense, don't change the aspect term).
# 4. there is a constraint: you should check the number of tokens between the preprocessed input and output using the given tool. always use the original text to compare to the counterfact version (output). if the token length is different, find another alternative.
# 5. as an example, this is how the preprocessed text looks like (you don't need to do this, this is automated by the tool): `{EXAMPLE}`
# 6. give the final answer using this format => `{result_format}`

# The text is `{input_text}` and the tuple is `{input_tuple}`"""
#     return prompt

# def format_triplet(triplet_string):
#     triplet = eval(triplet_string) if isinstance(triplet_string, str) else triplet_string
#     if isinstance(triplet, tuple):
#         if len(triplet) == 3:
#             triplet = [triplet]
#     elif isinstance(triplet, list):
#         if len(triplet) == 1:
#             triplet = triplet
#         elif len(triplet) == 3:
#             triplet = [tuple(triplet)]
#     return str(triplet)

# def run(agent, input_text, input_tuple):
#     result_a = agent.run(
#         create_prompt_a(input_text, input_tuple)
#     )
#     result_b = agent.run(
#         create_prompt_b(input_text, input_tuple)
#     )
#     result_c = agent.run(
#         create_prompt_c(input_text, input_tuple)
#     )
#     if isinstance(result_a, str):
#         result_a = eval(result_a)
#     if isinstance(result_b, str):
#         result_b = eval(result_b)
#     if isinstance(result_c, str):
#         result_c = eval(result_c)
    
#     original = tokenizer.tokenize(
#         preprocess(input_text, input_tuple)
#     )
#     counterfact_a = tokenizer.tokenize(
#         preprocess(result_a["text"], result_a["tuple"])
#     )
#     counterfact_b = tokenizer.tokenize(
#         preprocess(result_b["text"], result_b["tuple"])
#     )
#     counterfact_c = tokenizer.tokenize(
#         preprocess(result_c["text"], result_c["tuple"])
#     )

#     assert len(original) == len(counterfact_a) == len(counterfact_b) == len(counterfact_c)
    
#     return result_a, result_b, result_c

# os.makedirs("output", exist_ok=True)
# for fname in os.listdir("data"):
#     input_path = os.path.join("data", fname)
#     output_path = os.path.join("output", fname)

#     if os.path.exists(output_path):
#         continue

#     df = pd.read_csv(input_path)
#     result_df = df.copy()
#     to_remove = []
#     for i, row in tqdm(df.iterrows(), desc=fname):
#         input_text = row["original_sentence"]
#         input_tuple = row["original_triplet"]
#         if isinstance(input_tuple, str):
#             input_tuple = eval(input_tuple)[0]
#             assert len(input_tuple) == 3
#         if input_tuple[0] == 'null':
#             to_remove.append(row["index"])
#             continue

#         while True:
#             try:
#                 result_a, result_b, result_c = run(agent, input_text, input_tuple)
#                 break
#             except AssertionError as e:
#                 print(f"Error in index {i}, repeat")

#         result_df.loc[i, "counterfact1"] = str(result_a["text"])
#         result_df.loc[i, "counterfact_triplet1"] = str(format_triplet(result_a["tuple"]))
#         result_df.loc[i, "counterfact2"] = str(result_b["text"])
#         result_df.loc[i, "counterfact_triplet2"] = str(format_triplet(result_b["tuple"]))
#         result_df.loc[i, "counterfact3"] = str(result_c["text"])
#         result_df.loc[i, "counterfact_triplet3"] = str(format_triplet(result_c["tuple"]))

#         if i%10 == 0:
#             result_df.to_csv(output_path, index=False) # checkpoint
#     result_df = result_df.loc[~result_df["index"].isin(to_remove)]
#     result_df.to_csv(output_path, index=False)
