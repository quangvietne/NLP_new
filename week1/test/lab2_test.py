from preprocessing.regex_tokenizer import RegexTokenizer 
from representations.count_vectorizer import CountVectorizer


tokenizer = RegexTokenizer()  

# Instantiate CountVectorizer
vectorizer = CountVectorizer(tokenizer)

# Sample corpus
corpus = [
    "I love NLP.",
    "I love programming.",
    "NLP is a subfield of AI."
]

# Fit and transform
matrix = vectorizer.fit_transform(corpus)

# Print results
print("Learned vocabulary:")
print(vectorizer.vocabulary_)

print("\nDocument-term matrix:")
for i, vec in enumerate(matrix):
    print(f"Document {i+1}: {vec}")