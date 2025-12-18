from core.interfaces import Tokenizer

class SimpleTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        text_lower = text.lower()
        punctuation = ".,? ,!"
        for char in punctuation:
            text_lower = text_lower.replace(char, ' ' + char + ' ')
        tokens = text_lower.split()
        return tokens
    
