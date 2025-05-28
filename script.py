from smolagents import CodeAgent
from smolagents import OpenAIServerModel
from dotenv import load_dotenv
import pandas as pd
import os
from tqdm import tqdm
from tools import check_num_tokens, tokenize, preprocess, tokenizer

load_dotenv()

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

EXAMPLE = preprocess("Example text.", ("aspect term", "opinion term", "sentiment value"))

def create_prompt_a(input_text, input_tuple):
    result_format = r'{"text" : "output text", "tuple" : "output tuple"}'
    prompt = f"""You are an agent that creates a counterfactual sentence given an aspect based sentiment analysis data (text and tuple pair). Below are the requirements:

1. work on aspect based sentiment analysis
2. receives a text and a tuple containing: (a) aspect term (b) opinion term (c) sentiment value
3. make a counterfactual version of the input. change only the aspect term in the text and the tuple (you can even put some nonsense aspect term to replace the former one).
4. there is a constraint: you should check the number of tokens between the preprocessed input and output using the given tool. always use the original text to compare to the counterfact version (output). if the token length is different, find another alternative.
5. as an example, this is how the preprocessed text looks like (you don't need to do this, this is automated by the tool): `{EXAMPLE}`
6. give the final answer using this format => `{result_format}`

The text is `{input_text}` and the tuple is `{input_tuple}`"""
    return prompt

def create_prompt_b(input_text, input_tuple):
    result_format = r'{"text" : "output text", "tuple" : "output tuple"}'
    prompt = f"""You are an agent that creates a counterfactual sentence given an aspect based sentiment analysis data (text and tuple pair). Below are the requirements:

1. work on aspect based sentiment analysis
2. receives a text and a tuple containing: (a) aspect term (b) opinion term (c) sentiment value
3. make a counterfactual version of the input. change only the opinion term in the text and the tuple (without changing the sentiment value).
4. there is a constraint: you should check the number of tokens between the preprocessed input and output using the given tool. always use the original text to compare to the counterfact version (output). if the token length is different, find another alternative.
5. as an example, this is how the preprocessed text looks like (you don't need to do this, this is automated by the tool): `{EXAMPLE}`
6. give the final answer using this format => `{result_format}`

The text is `{input_text}` and the tuple is `{input_tuple}`"""
    return prompt

def create_prompt_c(input_text, input_tuple):
    result_format = r'{"text" : "output text", "tuple" : "output tuple"}'
    prompt = f"""You are an agent that creates a counterfactual sentence given an aspect based sentiment analysis data (text and tuple pair). Below are the requirements:

1. work on aspect based sentiment analysis
2. receives a text and a tuple containing: (a) aspect term (b) opinion term (c) sentiment value
3. make a counterfactual version of the input. change the sentiment value (sometimes this makes the opinion term or any other components in the sentence needs to be adjusted to fit the sentiment value and make the sentence make sense, don't change the aspect term).
4. there is a constraint: you should check the number of tokens between the preprocessed input and output using the given tool. always use the original text to compare to the counterfact version (output). if the token length is different, find another alternative.
5. as an example, this is how the preprocessed text looks like (you don't need to do this, this is automated by the tool): `{EXAMPLE}`
6. give the final answer using this format => `{result_format}`

The text is `{input_text}` and the tuple is `{input_tuple}`"""
    return prompt

def run(agent, input_text, input_tuple):
    result_a = agent.run(
        create_prompt_a(input_text, input_tuple)
    )
    result_b = agent.run(
        create_prompt_b(input_text, input_tuple)
    )
    result_c = agent.run(
        create_prompt_c(input_text, input_tuple)
    )
    if isinstance(result_a, str):
        result_a = eval(result_a)
    if isinstance(result_b, str):
        result_b = eval(result_b)
    if isinstance(result_c, str):
        result_c = eval(result_c)
    
    original = tokenizer.tokenize(
        preprocess(input_text, input_tuple)
    )
    counterfact_a = tokenizer.tokenize(
        preprocess(result_a["text"], result_a["tuple"])
    )
    counterfact_b = tokenizer.tokenize(
        preprocess(result_b["text"], result_b["tuple"])
    )
    counterfact_c = tokenizer.tokenize(
        preprocess(result_c["text"], result_c["tuple"])
    )

    assert len(original) == len(counterfact_a) == len(counterfact_b) == len(counterfact_c)
    
    return result_a, result_b, result_c

os.makedirs("output", exist_ok=True)
for fname in os.listdir("data"):
    input_path = os.path.join("data", fname)
    output_path = os.path.join("output", fname)
    df = pd.read_csv(input_path)
    result_df = df.copy()
    for i, row in tqdm(df.iterrows(), desc=fname):
        input_text = row["original_sentence"]
        input_tuple = row["original_triplet"]
        if isinstance(input_tuple, str):
            input_tuple = eval(input_tuple)[0]
            assert len(input_tuple) == 3
        if input_tuple[0] == 'null':
            continue

        while True:
            try:
                result_a, result_b, result_c = run(agent, input_text, input_tuple)
                break
            except AssertionError as e:
                print(f"Error in index {i}, repeat")

        result_df.loc[i, "counterfact1"] = str(result_a["text"])
        result_df.loc[i, "counterfact_triplet1"] = str(result_a["tuple"])
        result_df.loc[i, "counterfact2"] = str(result_b["text"])
        result_df.loc[i, "counterfact_triplet2"] = str(result_b["tuple"])
        result_df.loc[i, "counterfact3"] = str(result_c["text"])
        result_df.loc[i, "counterfact_triplet3"] = str(result_c["tuple"])

        if i%10 == 0:
            result_df.to_csv(output_path, index=False) # checkpoint
    
    result_df.to_csv(output_path, index=False)
