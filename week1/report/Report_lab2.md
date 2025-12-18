# NLP Labs Report week 1 

---

# LAB 2

## Mô tả công việc

- Xây dựng **Vectorizer Interface**, định nghĩa class trừu tượng `Vectorizer` với 3 phương thức:

  - `fit(self, corpus: list[str])`
  - `transform(self, documents: list[str]) -> list[list[int]]`
  - `fit_transform(self, corpus: list[str]) -> list[list[int]]`

- Cài đặt và triển khai **CountVectorizer**.

## Các bước triển khai


1. **Định nghĩa interface Vectorizer**

   * File: `src/core/interfaces.py`
   * Abstract base class `Vectorizer` với phương thức:

     ```python
     fit(self, corpus: list[str])
     transform(self, documents: list[str]) -> list[list[int]]
     fit_transform(self, corpus: list[str]) -> list[list[int]]
     ```

2. **Triển khai CountVectorizer**

   * File: `src/representations/count_vectorizer.py`
   * Nhận một tokenizer từ Lab 1.
   * Tạo `vocabulary_` từ tập hợp các token duy nhất.
   * Chuyển danh sách văn bản thành **document-term matrix**.

3. **Test/demo Lab 2**

   * File: `labs/lab2/test_lab2_vectorizer.py`
   * Chạy CountVectorizer trên corpus mẫu và in ra vocabulary, document-term matrix.

## Cách chạy code Lab 2

```bash
python -m week1.test.lab2_test
```

## Kết quả chạy code

```
Learned vocabulary:
{'.': 0, 'a': 1, 'ai': 2, 'i': 3, 'is': 4, 'love': 5, 'nlp': 6, 'of': 7, 'programming': 8, 'subfield': 9}

Document-term matrix:
Document 1: [1, 0, 0, 1, 0, 1, 1, 0, 0, 0]
Document 2: [1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
Document 3: [1, 1, 1, 0, 1, 0, 1, 1, 0, 1]
```

## Giải thích kết quả

- **Vocabulary**: gồm 10 phần tử. Dấu `"."` vẫn xuất hiện trong vocabulary vì RegexTokenizer coi nó là token hợp lệ → dẫn đến từ vựng chứa ký hiệu không có nhiều giá trị ngữ nghĩa.
- **Document-Term Matrix (DTM)**:

  - Mỗi document được biểu diễn thành vector đếm số lần xuất hiện của từng token trong vocabulary.
  - Các câu có độ dài và số lượng từ khác nhau được phản ánh trực tiếp trong số lượng token được gán giá trị > 0.


## Khó khăn và cách giải quyết

- **Khó khăn gặp phải**:

  - Dấu câu được giữ lại trong vocabulary → tạo ra token nhiễu.
  - Contraction và từ ghép bị tách sai bởi RegexTokenizer (ví dụ: `isn't`, `let's`).

- **Cách giải quyết**:

  - Thêm bước tiền xử lý loại bỏ dấu câu.
  - Chuẩn hóa contraction về dạng đầy đủ (`isn't → is not`).
  - Với corpus lớn, nên chuyển sang **TF-IDF Vectorizer** hoặc áp dụng giảm chiều để xử lý tính thưa của ma trận.


## Tài liệu tham khảo

**scikit-learn Documentation: CountVectorizer**: [https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) - Tài liệu tham khảo toàn diện về cách triển khai thực tế trong thư viện phổ biến, với đầy đủ các tham số.
   
