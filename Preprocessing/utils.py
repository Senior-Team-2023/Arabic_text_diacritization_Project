import re
import numpy as np

# from character_tokenizer import CharacterTokenizer
# from nltk.tokenize.stanford_segmenter import StanfordSegmenter


# read ./dataset/{file_name}.txt line by line and return a list of filtered strings
def read_data(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        # train_set = f.read().splitlines()
        return f.read()


# filter data takes a list of strings and removes unwanted patterns
def filter_data(data: str) -> str:
    # data = re.sub(r"\( \d+ (/ \d+)? \)", "", data)
    # remove all numbers
    data = re.sub(r"\d+", "", data)
    # regex to remove all special characters
    data = re.sub(r"[][//,;?()$:-{}_*]", "", data)
    # remove all english letters
    data = re.sub(r"[a-zA-Z]", "", data)
    # Substituting multiple spaces with single space
    data = re.sub(r"([ \r\t\f])+", " ", data, flags=re.I)
    return data


# split data into sentences
def split_data_to_sentences(data: str) -> list:
    sentences = re.split(r"[.?!\n]", data)
    return sentences


# character tokenizer returns a list of unique characters
def char_tokenizer(data: str):
    characters = list(set(re.sub(r" ", "", data)))
    # characters.sort()
    # char_to_int = dict((c, i) for i, c in enumerate(characters))
    # int_to_char = dict((i, c) for i, c in enumerate(characters))
    return characters
