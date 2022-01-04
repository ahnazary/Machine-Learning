from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
import nltk
# nltk.download('punkt')

warnings.filterwarnings(action='ignore')

import gensim
from gensim.models import Word2Vec

#  Reads ‘alice.txt’ file
sample = open("/home/amirhossein/Documents/GitHub/Machine-Learning/text.txt", "r")
s = sample.read()

# Replaces escape character with space
f = s.replace("\n", " ")

data = []

# iterate through each sentence in the file
for i in sent_tokenize(f):
    temp = []

    # tokenize the sentence into words
    for j in word_tokenize(i):
        temp.append(j.lower())

    data.append(temp)

# Create CBOW model
print(data)
model1 = gensim.models.Word2Vec(data, min_count=3,
                                window=3)

# Print results
word1 = "king"
word2 = "copyright"
print("Cosine similarity between '" + word1 + "' " +
      "and '" + word2 + "' - CBOW : ",
      model1.wv.similarity(word1, word2))

# Create Skip Gram model
model2 = gensim.models.Word2Vec(data, min_count=3, window=3, sg=1)

# Print results
print("Cosine similarity between '" + word1 + "' " +
      "and '" + word2 + "' - Skip Gram : ",
      model2.wv.similarity(word1, word2))

