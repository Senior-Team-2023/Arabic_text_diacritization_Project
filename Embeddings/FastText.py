from gensim.models import FastText
import os
import re


class FastTextEmbedding:
    def __init__(self, data, vector_size, is_character = False):
        self.data = data
        self.vector_size = vector_size
        self.model = None
        self.word_embeddings = None
        self.is_character = is_character

    def train(self):
        # Split the documents into words
        if self.is_character == False:
            # tokenized_docs = [doc.split() for doc in self.data]
            tokenized_docs = [split_data_to_words(doc) for doc in self.data]
        else:
            tokenized_docs = [char_tokenizer(doc) for doc in self.data]
        # Train the FastText model
        self.model = FastText(tokenized_docs, min_count=1, vector_size = self.vector_size)

        # Get the word embeddings
        self.word_embeddings = self.model.wv
    
    def Word_embeddings(self):
        return self.word_embeddings

    def vector(self, word):
        if not self.word_embeddings.has_index_for(word):
            raise Exception("Word not found in the vocabulary")
        return self.word_embeddings[word]
        
    def cosine_similarity(self, word1, word2):
        return self.word_embeddings.similarity(word1, word2)

    def save_model(self, file_path):
        # Save the FastText model
        self.model.save(file_path)
        print("FastText model saved to: ", file_path)

    def load_model(self, file_path):
        # Load the existing FastText model
        self.model = FastText.load(file_path)
        self.word_embeddings = self.model.wv
        print("FastText model loaded from: ", file_path)

    def is_model_saved(self, file_path):
        # Check if the model file already exists
        return os.path.exists(file_path)
    
def char_tokenizer(data: str):
    characters = list(re.sub(r" ", "", data))
    return characters

def split_data_to_words(data: str) -> list:
    words = re.split(r"[^\S\n]+", data)
    return words

# docs = ['كان يوم سعيد',
#         'ماشاءهللا عمل جيد',
#         'ممتاز',
#         'عمل مكتمل',
#         'اعتقد بانه ضعيف',
#         'يوجد ثغرات ونقاط ضعف',
#         'ليس جيدا',
#         'كان عمل متعب']

# fastText = FastTextEmbedding(docs, vector_size = 5, is_character = False)
# fastText.train()

# for word in fastText.Word_embeddings().key_to_index:
#     vector = fastText.vector(word)
#     print(f"Word: {word}, Vector: {vector}")



