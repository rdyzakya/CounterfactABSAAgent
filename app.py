import streamlit as st
from smolagents import CodeAgent
from smolagents import OpenAIServerModel
from dotenv import load_dotenv
import os
from tools import check_num_tokens, tokenize, preprocess

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

def create_prompt(input_text, input_tuple):
    result_format = r'{"counterfact_a" : {"text" : "text a", "tuple" : "tuple a"}, "counterfact_b" : {"text" : "text b", "tuple" : "tuple b"}, "counterfact_c" : {"text" : "text c", "tuple" : "tuple c"}}'
    prompt = f"""You are an agent that creates a counterfactual sentence given an aspect based sentiment analysis data (text and tuple pair). Below are the requirements:

1. work on aspect based sentiment analysis
2. receives a text and a tuple containing: (a) aspect term (b) opinion term (c) sentiment value
3. make a counterfactual version of the input. there are 3 version of counterfactual sentences:
    a. change aspect term in the text and the tuple (you can even put some nonsense aspect term to replace the former one)
    b. change opinion term in the text and the tuple (without changing the sentiment value)
    c. change the sentiment value (sometimes this makes the opinion term or any other components in the sentence needs to be adjusted to fit the sentiment value and make the sentence make sense, don't change the aspect term)
4. there is a constraint: you should check the number of tokens between the preprocessed input and output using the given tool. always use the original text to compare to the counterfact version (output). if the token length is different, find another alternative.
5. as an example, this is how the preprocessed text looks like (you don't need to do this, this is automated by the tool): `{EXAMPLE}`
6. give the final answer using this format => `{result_format}`

The text is `{input_text}` and the tuple is `{input_tuple}`"""
    return prompt

# Streamlit interface
st.set_page_config(page_title="SmolAgent Interface", layout="centered")
st.title("🧠 SmolAgent Code Assistant")

input_text = st.text_area("Text", height=68)
input_tuple = st.text_area("Tuple", height=68)

if st.button("Run Agent"):
    if input_text.strip():
        with st.spinner("Thinking..."):
            try:
                response = agent.run(
                    create_prompt(input_text, input_tuple)
                )
                st.success("Agent Response:")
                st.code(response, language='python')
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a prompt.")