from typing import List, Dict
from core.interfaces import Vectorizer, Tokenizer

class CountVectorizer(Vectorizer):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.vocabulary_: Dict[str, int] = {}
    
    def fit(self, corpus: List[str]) -> None:
        """
        Learns the vocabulary from the corpus.
        """
        unique_tokens = set()
        for document in corpus:
            tokens = self.tokenizer.tokenize(document)
            unique_tokens.update(tokens)
        
        # Sort the unique tokens and assign indices
        sorted_tokens = sorted(unique_tokens)
        self.vocabulary_ = {token: index for index, token in enumerate(sorted_tokens)}

    def transform(self, documents: List[str]) -> List[List[int]]:
        """
        Transforms documents into count vectors.
        """
        if not self.vocabulary_:
            raise ValueError("Vocabulary not learned. Call fit() first.")
        
        vectors: List[List[int]] = []
        for document in documents:
            tokens = self.tokenizer.tokenize(document)
            vector = [0] * len(self.vocabulary_)
            for token in tokens:
                if token in self.vocabulary_:
                    vector[self.vocabulary_[token]] += 1
            vectors.append(vector)
        return vectors

    def fit_transform(self, corpus: List[str]) -> List[List[int]]:
        """
        Convenience method: fit and then transform.
        """
        self.fit(corpus)
        return self.transform(corpus)