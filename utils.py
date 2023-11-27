import re
import numpy as np
from character_tokenizer import CharacterTokenizer
# from nltk.tokenize.stanford_segmenter import StanfordSegmenter


# read ./dataset/{file_name}.txt line by line and return a list of filtered strings
def read_data(file_name):
    with open(f"./dataset/{file_name}.txt", "r", encoding="utf-8") as f:
        # train_set = f.read().splitlines()
        return f.read()


# filter data takes a list of strings and removes unwanted patterns
def filter_data(data):
    # data = re.sub(r"\( \d+ (/ \d+)? \)", "", data)
    # remove all numbers
    data = re.sub(r"\d+", "", data)
    # regex to remove all special characters
    data = re.sub(r"[][//,;?()$:-{}_]", "", data)
    # remove all english letters
    data = re.sub(r"[a-zA-Z]", "", data)
    # Substituting multiple spaces with single space    
    data = re.sub(r'([ \r\t\f])+', ' ', data, flags=re.I)
    return data


# character tokenizer
def char_tokenizer(data):
    tokenizer = CharacterTokenizer()
    tokenizer.train("./dataset/train.txt")
    return tokenizer.tokenize(data[0:1000])
