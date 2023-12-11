import re
import numpy as np
from Preprocessing import character_encoding
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
    data = re.sub(r"[][//,;\?؟()$:\-{}_*؛،.:«»`–\"~!]", "", data)
    # remove all english letters
    data = re.sub(r"[a-zA-Z]", "", data)
    # Substituting multiple spaces with single space
    data = re.sub(r"([ \r\t\f])+", " ", data, flags=re.I)
    return data


# split data into sentences

def split_data_to_sentences(data: str) -> list:
    # Split data into sentences using punctuation marks and newlines as delimiters
    # sentences = re.split(r"[.?!\n]", data)
    sentences = re.split(r"[\n]", data)
    # Remove empty sentences
    sentences = [sentence for sentence in sentences if sentence.strip()]
    return sentences



# character tokenizer returns a list of unique characters
def char_tokenizer(data: str):
    characters = list(set(re.sub(r" ", "", data)))
    # characters.sort()
    # char_to_int = dict((c, i) for i, c in enumerate(characters))
    # int_to_char = dict((i, c) for i, c in enumerate(characters))
    return characters

def split_data_to_words(data: str) -> list:
    words = re.split(r" ", data)
    return words

def concatinate_word_char_embeddings(text_without_diacritics, diacritic_list, embedding_model):
    concatinated_vector = []
    diacritic_list_2 = []
    for i, word in enumerate(text_without_diacritics):
        # if word does not have corresponding embedding don't add it to the training set and remove its corresponding diacritic list
        try:
            word_vector = embedding_model.vector(word)
            diacritic_list_2.append(diacritic_list[i])
        except:
            # print(f"Word: \"{word}\" not found in the vocabulary")
            # char_vector = character_encoding.CharToOneHOt(char)
            # concatinated_vector.append(np.concatenate((word_vector, char_vector), axis=None))
            continue
        for char in word:
            char_vector = character_encoding.CharToOneHOt(char)
            concatinated_vector.append(np.concatenate((char_vector , word_vector), axis=None))

    return concatinated_vector, diacritic_list_2

