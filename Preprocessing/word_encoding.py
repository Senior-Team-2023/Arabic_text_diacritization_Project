import nltk
from nltk.corpus import stopwords

if not nltk.data.find('corpora/stopwords'):
    nltk.download('stopwords')


def RemoveStopWords(sentence):
    arabic_stop_words = set(stopwords.words('arabic'))

    words = sentence.split()
    filtered_words = [word for word in words if word not in arabic_stop_words]

    return ' '.join(filtered_words)

from nltk.stem import ISRIStemmer

def StemSentence(text):
    stemmer = ISRIStemmer()
    stemmed_words = [stemmer.stem(word) for word in text.split()]
    stemmed_text = ' '.join(stemmed_words)
    return stemmed_text

# # Example usage:
# input_corpus = "أحب اللغة العربية. أحب قراءة الكتب."
# input_corpus = "فائدة قال بعضهم يؤخذ من شرط تمام الملك عدم زكاة حلي الكعبة و المساجد من قناديل وعلائق وصفائح أبواب"
# result = stem_arabic(input_corpus)
# print(result)

# # Example Removing Stop Words usage:
# input_sentence = "و هذا هو مثال على جملة باللغة العربية"
# result = RemoveStopWords(input_sentence)
# # print(result)
