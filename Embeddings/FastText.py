from gensim.models import FastText
import os

class FastTextEmbedding:
    def __init__(self, data, vector_size):
        self.data = data
        self.vector_size = vector_size
        self.model = None
        self.word_embeddings = None

    def train(self):
        # Split the documents into words
        # split each document into list of characters
        tokenized_docs = [list(doc) for doc in self.data]
        # Train the FastText model
        self.model = FastText(tokenized_docs, min_count=1, vector_size=self.vector_size)

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


# docs = ['كان يوم سعيد',
#         'ماشاءهللا عمل جيد',
#         'ممتاز',
#         'عمل مكتمل',
#         'اعتقد بانه ضعيف',
#         'يوجد ثغرات ونقاط ضعف',
#         'ليس جيدا',
#         'كان عمل متعب']

# fastText = FastTextEmbedding(docs, vector_size = 5)
# fastText.train()

# for word in fastText.Word_embeddings().key_to_index:
#     vector = fastText.vector(word)
#     print(f"Word: {word}, Vector: {vector}")
