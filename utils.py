import re
import numpy as np

# read ./dataset/{file_name}.txt line by line and return a list of filtered strings
def read_data(file_name):
    with open(f'./dataset/{file_name}.txt', 'r', encoding='utf-8') as f:
        train_set = f.read().splitlines()
        return filter_data(train_set)
    

# filter data takes a list of strings and removes this patern ( number / number )
def filter_data(data):
    return [re.sub(r'\( \d+ / \d+ \)', '', line) for line in data]
