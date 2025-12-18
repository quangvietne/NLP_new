# NLP Labs Report week 1 

---

# LAB 1

## Mô tả công việc 

Hiểu và triển khai bước tiền xử lý văn bản cơ bản: tokenization. Tạo cả tokenizer đơn giản và tokenizer nâng cao bằng regex.

## Các bước triển khai 

1. **Chuẩn bị interface Tokenizer**  
   - Định nghĩa trong `src/core/interfaces.py` một abstract base class `Tokenizer` với phương thức:
     ```python
     tokenize(self, text: str) -> list[str]
     ```
   - Đây là phần **core** để tách riêng logic xử lý token khỏi cách load dữ liệu.

2. **SimpleTokenizer**  
   - File code: `src/preprocessing/simple_tokenizer.py`  
   - Chuyển toàn bộ văn bản về chữ thường.  
   - Tách từ dựa trên khoảng trắng.  
   - Xử lý các dấu câu cơ bản (`.`, `,`, `!`, `?`) thành token riêng.  
   - Kết hợp với interface `Tokenizer` để có chuẩn phương thức chung.

3. **RegexTokenizer**  
   - File code: `src/preprocessing/regex_tokenizer.py`  
   - Sử dụng biểu thức chính quy `\w+|[^\w\s]` để tách token chi tiết hơn, ví dụ "isn't" -> `isn` + `'` + `t`.

4. **Load dữ liệu từ dataset**  
   - File loader: `src/core/dataset_loaders.py`  
   - Load dữ liệu UD_English-EWT từ `data/UD_English-EWT/en_ewt-ud-train.txt`.  
   - Lấy 500 ký tự đầu để thử tokenization và so sánh output của SimpleTokenizer và RegexTokenizer.

5. **Test/demo Lab 1**  
   - File: `test/lab1_test.py`  
   - Chạy thử tokenizer trên câu mẫu và sample từ dataset, in ra token.


## Cách chạy code Lab 1

```bash
python -m week1.test.lab1_test
```

## Kết quả chạy code

### Ví dụ câu test

--- Testing Tokenizers ---
Input text: Hello, world! This is a test.
SimpleTokenizer Tokens: ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']
RegexTokenizer Tokens:   ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']

Input text: NLP is fascinating... isn't it?
SimpleTokenizer Tokens: ['nlp', 'is', 'fascinating', '.', '.', '.', "isn't", 'it', '?']
RegexTokenizer Tokens:   ['nlp', 'is', 'fascinating', '.', '.', '.', 'isn', "'", 't', 'it', '?']

Input text: Let's see how it handles 123 numbers and punctuation!
SimpleTokenizer Tokens: ["let's", 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']
RegexTokenizer Tokens:   ['let', "'", 's', 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']

### Sample dataset UD\_English-EWT (first 20 tokens)
--- Testing Tokenizers on file data ---

--- Tokenizing Sample Text from UD_English-EWT ---
Original Sample: Al-Zaman : American forces killed Shaikh Abdullah al-Ani, the preacher at the
mosque in the town of ...

SimpleTokenizer Output (first 20 tokens): ['al-zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al-ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the', 'town', 'of', 'qaim', ',']
RegexTokenizer Output (first 20 tokens): ['al', '-', 'zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al', '-', 'ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the']


## Giải thích kết quả

- **SimpleTokenizer**: chỉ xử lý một số dấu câu cố định → giữ nguyên contraction và từ ghép, nhưng không bao phủ hết các trường hợp.
- **RegexTokenizer**: tổng quát hơn, tách được nhiều loại ký tự → nhưng dễ gây tách sai contraction và từ có dấu gạch ngang.
  _Ví dụ:_ `al-zaman → ['al', '-', 'zaman']`, `let's → ['let', "'", 's']`

---
## Khó khăn và cách giải quyết

- **Khó khăn** : Xử lý ngôn ngữ không phải tiếng Anh và ký tự đặc biệt : Biểu thức mặc định `\w+` chủ yếu khớp ký tự từ tiếng Anh (A-Z, a-z, 0-9, _) và sẽ bỏ sót các ký tự có dấu (ví dụ: `"café"`, `"tiếng Việt"`).
- **Cách giải quyết** : Sử dụng các biểu thức chính quy hỗ trợ Unicode tổng quát hơn. Ví dụ, `r"\p{L}+"` khớp với mọi ký tự chữ cái (yêu cầu thư viện `regex` của Python thay vì `re` tiêu chuẩn).


## Tài liệu tham khảo

 - **Python `re` module**: [https://docs.python.org/3/library/re.html](https://docs.python.org/3/library/re.html) - Tài liệu chính thức về biểu thức chính quy trong Python.
    