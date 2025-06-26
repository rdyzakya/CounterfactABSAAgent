from transformers import AutoTokenizer
from smolagents import tool
from dotenv import load_dotenv
import os

load_dotenv()

tokenizer = AutoTokenizer.from_pretrained(os.getenv("MODEL_NAME_OR_PATH"))

def preprocess_text(text):
    return text + " [A] [O] [S]"

def preprocess_tuple(tup):
    return f"[A] {tup[0]} [O] {tup[1]} [S] {tup[2]}"

def preprocess(text, tup):
    return preprocess_text(text) + ' ' + preprocess_tuple(tup)

def split(full_text):
    text, tup_text = full_text.split("[A] [O] [S]")

    text = text.strip()
    tup_text = tup_text.strip()

    splitted_tup_text = tup_text.split()

    a_index = splitted_tup_text.index("[A]")
    o_index = splitted_tup_text.index("[O]")
    s_index = splitted_tup_text.index("[S]")

    aspect = ' '.join(splitted_tup_text[a_index+1:o_index])
    opinion = ' '.join(splitted_tup_text[o_index+1:s_index])
    sentiment = ' '.join(splitted_tup_text[s_index+1:])

    return text, (aspect, opinion, sentiment)

@tool
def tokenize(string: str) -> list:
    """
    This tool will give the tokenized result of a string.

    Args:
        string: String to be tokenize
    """
    return tokenizer.tokenize(string)

def check_num_tokens_fn(input_text: str, output_text: str, input_tuple: tuple, output_tuple: tuple) -> dict:
    """
    This tool will compare the number of tokens between input and output after preprocessing.
    The result will be a json string containing the tokenized_input, tokenized_output, num_tokens_input, 
    num_tokens_output, and is_equal.

    Args:
        input_text: Input text
        output_text: Output text
        input_tuple: Input tuple
        output_tuple: Output tuple

    Return:
        Report dictionary containing `tokenized_input_full_text`, `tokenized_output_full_text`, `tokenized_input_aspect`, `tokenized_output_aspect`, `tokenized_input_opinion`, `tokenized_output_opinion`, `is_equal_full_text`, `is_equal_aspect`, `is_equal_opinion`
    """
    tokenized_input = tokenizer.tokenize(preprocess(input_text, input_tuple))
    tokenized_output = tokenizer.tokenize(preprocess(output_text, output_tuple))

    tokenized_input_aspect = tokenizer.tokenize(input_tuple[0])
    tokenized_output_aspect = tokenizer.tokenize(output_tuple[0])

    tokenized_input_opinion = tokenizer.tokenize(input_tuple[1])
    tokenized_output_opinion = tokenizer.tokenize(output_tuple[1])

    num_tokens_input = len(tokenized_input)
    num_tokens_output = len(tokenized_output)

    num_tokens_input_aspect = len(tokenized_input_aspect)
    num_tokens_output_aspect = len(tokenized_output_aspect)

    num_tokens_input_opinion = len(tokenized_input_opinion)
    num_tokens_output_opinion = len(tokenized_output_opinion)

    is_equal_full_text = num_tokens_input == num_tokens_output
    is_equal_aspect = num_tokens_input_aspect == num_tokens_output_aspect
    is_equal_opinion = num_tokens_input_opinion == num_tokens_output_opinion

    report = {
        "tokenized_input_full_text" : tokenized_input,
        "tokenized_output_full_text" : tokenized_output,
        "tokenized_input_aspect" : tokenized_input_aspect,
        "tokenized_output_aspect" : tokenized_output_aspect,
        "tokenized_input_opinion" : tokenized_input_opinion,
        "tokenized_output_opinion" : tokenized_output_opinion,
        "is_equal_full_text" : is_equal_full_text,
        "is_equal_aspect" : is_equal_aspect,
        "is_equal_opinion" : is_equal_opinion
    }

    return report

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
    
    Return:
        Report dictionary containing `tokenized_input_full_text`, `tokenized_output_full_text`, `tokenized_input_aspect`, `tokenized_output_aspect`, `tokenized_input_opinion`, `tokenized_output_opinion`, `is_equal_full_text`, `is_equal_aspect`, `is_equal_opinion`
    """
    return check_num_tokens_fn(input_text, output_text, input_tuple, output_tuple)