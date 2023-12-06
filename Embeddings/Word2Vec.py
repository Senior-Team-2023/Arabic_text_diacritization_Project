from gensim.models import Word2Vec
import os

class W2V:
    def __init__(self, data, vector_size):
        self.data = data
        self.vector_size = vector_size
        self.model = None
        self.word_embeddings = None

    def train(self):
        # Split the documents into words
        tokenized_docs = [doc.split() for doc in self.data]

        # Train the Word2Vec model
        self.model = Word2Vec(tokenized_docs, min_count=1, vector_size=self.vector_size)

        # Get the word embeddings
        self.word_embeddings = self.model.wv
    
    def Word_embeddings(self):
        return self.word_embeddings

    def vector(self, word):
        if not self.word_embeddings.has_index_for(word):
           raise "Word not found in the vocabulary"
        return self.word_embeddings[word]
        

    def Cos_similarity(self, word1, word2):
        return self.word_embeddings.similarity(word1, word2)


    def save_model(self, file_path):
        # Save the Word2Vec model
        self.model.save(file_path)
        print("Word2Vec model saved to : ", file_path)

    def load_model(self, file_path):
        # Load the existing Word2Vec model
        self.model = Word2Vec.load(file_path)
        self.word_embeddings = self.model.wv
        print("Word2Vec model loaded from : ", file_path)

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

# w2v = W2V(docs, vector_size = 5)
# w2v.train()

# for word in w2v.Word_embeddings().key_to_index:
#     vector = w2v.vector(word)
#     print(f"Word: {word}, Vector: {vector}")
