

# BÁO CÁO THỰC HÀNH week7

## I. Tổng quan

Bài thực hành tập trung vào việc sử dụng thư viện **SpaCy** để thực hiện phân tích cú pháp phụ thuộc (Dependency Parsing). Mục tiêu là hiểu cấu trúc ngữ pháp của câu, mối quan hệ giữa các từ (từ cha - từ con) và ứng dụng cấu trúc này để trích xuất thông tin (Information Extraction).

---

## II. Các bước triển khai

### 1. Cài đặt môi trường và Tải mô hình

* Cài đặt thư viện `spacy`.
* Tải mô hình ngôn ngữ tiếng Anh `en_core_web_md`.
* *Lý do chọn `md` (medium):* Mô hình này chứa cả vector từ (word vectors) và độ chính xác cao hơn bản `sm` (small) cho các tác vụ phân tích cú pháp.


```python
!python -m spacy download en_core_web_md

```



### 2. Trực quan hóa cây cú pháp (Visualization)

* Sử dụng module `displacy` để vẽ cây phụ thuộc cho câu ví dụ: *"The quick brown fox jumps over the lazy dog."*
* Quan sát mối quan hệ giữa các từ (Root, Subject, Object, Modifier).

### 3. Phân tích chi tiết Token và Quan hệ (Dependency Attributes)

* Duyệt qua từng token trong câu *"Apple is looking at buying U.K. startup for $1 billion"*.
* Trích xuất các thuộc tính:
* **Text:** Từ gốc.
* **Dep:** Nhãn quan hệ phụ thuộc (Dependency label).
* **Head:** Từ cha (Head word).
* **Children:** Các từ con phụ thuộc vào nó.



### 4. Trích xuất thông tin (Information Extraction)

* **Trích xuất bộ ba (Subject - Verb - Object):** Dựa vào cấu trúc cây, tìm các động từ (VERB), sau đó tìm con của nó đóng vai trò chủ ngữ (`nsubj`) và tân ngữ (`dobj`).
* **Trích xuất cụm danh từ và tính từ bổ nghĩa:** Tìm các danh từ (NOUN) và các con của nó có nhãn `amod` (adjective modifier).

### 5. Bài tập thực hành (Cấu trúc cây)

* Viết hàm tìm động từ chính (Main Verb / ROOT).
* Viết hàm trích xuất cụm danh từ thủ công dựa trên các từ con (`det`, `amod`, `compound`).
* Viết hàm tìm đường đi từ một từ bất kỳ đến gốc (Root) của cây.

---

## III. Cách chạy code và Ghi log kết quả

### 1. Cách chạy

* **Môi trường:** Google Colab hoặc Jupyter Notebook (khuyến nghị có GPU để load model nhanh hơn, dù CPU vẫn chạy tốt với SpaCy).
* **Thực thi:** Chạy tuần tự các cell.
* **Lưu ý quan trọng:** Sau khi chạy lệnh `pip install` và tải model, cần **Restart Runtime/Kernel** để hệ thống nhận diện gói mới cài đặt.

### 2. Ghi log kết quả

Kết quả được ghi nhận từ output của notebook:

**Kết quả Trực quan hóa (DisplaCy):**

> Hiển thị biểu đồ SVG với:
> * **ROOT:** jumps
> * **Subject:** fox (quan hệ `nsubj` với jumps)
> * **Prep:** over (quan hệ `prep` với jumps)
> 
> 

**Kết quả Phân tích chi tiết (Apple example):**

```text
TEXT         | DEP        | HEAD TEXT    | HEAD POS | CHILDREN
----------------------------------------------------------------------
Apple        | nsubj      | looking      | VERB     | []
looking      | ROOT       | looking      | VERB     | ['Apple', 'is', 'at']
startup      | dobj       | buying       | VERB     | ['U.K.', 'for']
...

```

**Kết quả Trích xuất thông tin:**

> * **Triplets found:** `(cat, chased, mouse)`, `(dog, watched, them)`
> * **Modifiers:** Danh từ 'cat' được bổ nghĩa bởi: `['big', 'fluffy', 'white']`
> 
> 

**Kết quả Bài tập tự luyện:**

> * **Main Verb:** `jumps`
> * **Noun Chunks:** `['The big white cat', 'a small mouse']`
> * **Path to Root:** `['is', 'chasing']` (Lưu ý: kết quả trong log gốc là `['is', 'chasing']` cho input `doc[3]` là 'cat' trong câu "The big cat is chasing...", nghĩa là 'cat' -> 'chasing' (HEAD) -> 'chasing' (ROOT)).
> 
>

---

## IV. Giải thích kết quả

1. **Cấu trúc cây phụ thuộc:**
* Mỗi câu luôn có một từ gốc (**ROOT**), thường là động từ chính (ví dụ: "jumps", "looking").
* Các từ khác kết nối với nhau qua các cung có hướng (từ Head đến Child). Ví dụ: "The" và "quick" là con của "fox", bổ nghĩa cho "fox".


2. **Tại sao trích xuất được bộ ba (S-V-O)?**
* Code dựa vào quy tắc ngữ pháp phổ quát trong tiếng Anh: Hành động (Verb) là trung tâm. Người thực hiện hành động thường gắn nhãn `nsubj` (nominal subject), và đối tượng chịu tác động thường gắn nhãn `dobj` (direct object). Việc duyệt cây con của Verb giúp lấy chính xác các thành phần này.


3. **Ý nghĩa của việc duyệt cây (Tree Traversal):**
* Trong bài tập 3, việc tìm đường đi đến Root (`while token.head != token`) giúp hiểu được cấp độ sâu của từ trong cây cú pháp. Từ càng gần Root thì vai trò ngữ pháp càng quan trọng trong việc định hình cấu trúc câu.



---

## V. Khó khăn gặp phải và Cách giải quyết

### 1. Lỗi chưa load được Model/Thư viện

* **Vấn đề:** Trong log cài đặt có cảnh báo `Restart to reload dependencies`. Nếu không restart, lệnh `spacy.load("en_core_web_md")` sẽ báo lỗi không tìm thấy model.
* **Giải quyết:** Thực hiện Restart Runtime (trên Colab: Runtime -> Restart session) sau khi chạy cell cài đặt đầu tiên.

### 2. DisplaCy Server (Local vs Colab)

* **Vấn đề:** Hàm `displacy.serve()` thường chạy một web server cục bộ (localhost:5000), điều này có thể làm treo cell trong Jupyter Notebook nếu không được ngắt đúng cách.
* **Giải quyết:** Trong Colab, `displacy` thường tự động render HTML (`output_type: display_data`) như trong log đã thể hiện. Nếu chạy máy local, nên dùng `displacy.render(style='dep', jupyter=True)` thay vì `.serve()` để hiển thị tĩnh ngay trong notebook.

---

## VI. Nguồn tham khảo và Thông tin Model

1. **Mô hình sử dụng:**
* **Tên:** `en_core_web_md` (English Core Web Medium).
* **Nguồn:** SpaCy (Explosion AI).
* **Đặc điểm:** Chứa Tagger, Parser, NER và Word Vectors (20k unique vectors).


2. **Tài liệu tham khảo:**
* SpaCy Documentation: [https://spacy.io/usage/linguistic-features#dependency-parse](https://www.google.com/search?q=https://spacy.io/usage/linguistic-features%23dependency-parse)
* Dependency Labels Scheme (ClearNLP/Universal Dependencies).