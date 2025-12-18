from preprocessing.simple_tokenizer import SimpleTokenizer
from preprocessing.regex_tokenizer import RegexTokenizer
from core.dataset_loaders import load_raw_text_data

simple_tokenizer = SimpleTokenizer()
regex_tokenizer = RegexTokenizer()

sentences = [
    "Hello, world! This is a test.",
    "NLP is fascinating... isn't it?",
    "Let's see how it handles 123 numbers and punctuation!"
]
print("--- Testing Tokenizers ---")
for text in sentences:
    print("Input text:", text)
    tokens = simple_tokenizer.tokenize(text)
    regex_tokens = regex_tokenizer.tokenize(text)
    print("SimpleTokenizer Tokens:", tokens)
    print("RegexTokenizer Tokens:", regex_tokens)


dataset_path = "data/UD_English-EWT/UD_English-EWT/en_ewt-ud-train.txt"
raw_text = load_raw_text_data(dataset_path)
sample_text = raw_text[:500] # First 500 characters
print("\n--- Testing Tokenizers on file data ---")
print("\n--- Tokenizing Sample Text from UD_English-EWT ---")
print(f"Original Sample: {sample_text[:100]}...")
simple_tokens = simple_tokenizer.tokenize(sample_text)
print(f"SimpleTokenizer Output (first 20 tokens): {simple_tokens[:20]}")
regex_tokens = regex_tokenizer.tokenize(sample_text)
print(f"RegexTokenizer Output (first 20 tokens): {regex_tokens[:20]}")