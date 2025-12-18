from core.interfaces import Tokenizer
import re
class RegexTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        text = text.lower()
        pattern = r"\w+|[^\w\s]"
        tokens = re.findall(pattern, text)
        return tokens