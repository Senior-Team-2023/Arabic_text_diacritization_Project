import re
import numpy as np
import tkseem as tk

# from nltk.tokenize.stanford_segmenter import StanfordSegmenter


# read ./dataset/{file_name}.txt line by line and return a list of filtered strings
def read_data(file_name):
    with open(f"./dataset/{file_name}.txt", "r", encoding="utf-8") as f:
        # train_set = f.read().splitlines()
        return filter_data(f.read())


# filter data takes a list of strings and removes this patern ( number / number )
def filter_data(data):
    # return [re.sub(r'\( \d+ (/ \d+)? \)', '', line) for line in data]
    # regex to remove all special characters
    data = re.sub(r"\( \d+ (/ \d+)? \)", "", data)
    data = re.sub(r"[^\w\s]", "", data)
    return data


# character tokenizer
def char_tokenizer(data):
    return [list(line) for line in data]
