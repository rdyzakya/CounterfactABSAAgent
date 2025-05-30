from transformers import AutoTokenizer
from smolagents import tool
from dotenv import load_dotenv
from pprint import pformat
import os

load_dotenv()

tokenizer = AutoTokenizer.from_pretrained(os.getenv("MODEL_NAME_OR_PATH"))

def preprocess_text(text):
    return text + " [A] [O] [S]"

def preprocess_tuple(tup):
    return f"[A] {tup[0]} [O] {tup[1]} [S] {tup[2]}"

def preprocess(text, tup):
    return preprocess_text(text) + ' ' + preprocess_tuple(tup)

@tool
def tokenize(string: str) -> str:
    """
    This tool will give the tokenized result of a string.

    Args:
        string: String to be tokenize
    """
    return str(tokenizer.tokenize(string))

def check_num_tokens_fn(input_text: str, output_text: str, input_tuple: tuple, output_tuple: tuple) -> str:
    """
    This tool will compare the number of tokens between input and output after preprocessing.
    The result will be a json string containing the tokenized_input, tokenized_output, num_tokens_input, 
    num_tokens_output, and is_equal.

    Args:
        input_text: Input text
        output_text: Output text
        input_tuple: Input tuple
        output_tuple: Output tuple
    """
    tokenized_input = tokenizer.tokenize(preprocess(input_text, input_tuple))
    tokenized_output = tokenizer.tokenize(preprocess(output_text, output_tuple))
    num_tokens_input = len(tokenized_input)
    num_tokens_output = len(tokenized_output)
    is_equal = num_tokens_input == num_tokens_output
    
    report = {
        "tokenized_input" : tokenized_input,
        "tokenized_output" : tokenized_output,
        "num_tokens_input" : num_tokens_input,
        "num_tokens_output" : num_tokens_output,
        "is_equal" : is_equal
    }

    report_str = pformat(report, indent=2)

    return report_str

@tool
def check_num_tokens(input_text: str, output_text: str, input_tuple: tuple, output_tuple: tuple) -> str:
    """
    This tool will compare the number of tokens between input and output after preprocessing.
    The result will be a json string containing the tokenized_input, tokenized_output, num_tokens_input, 
    num_tokens_output, and is_equal.

    Args:
        input_text: Input text
        output_text: Output text
        input_tuple: Input tuple
        output_tuple: Output tuple
    """
    return check_num_tokens_fn(input_text, output_text, input_tuple, output_tuple)