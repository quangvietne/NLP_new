

# BÁO CÁO THỰC HÀNH: GÁN NHÃN TỪ LOẠI (POS TAGGING) VỚI SIMPLE RNN

## I. Tổng quan

Bài thực hành xây dựng một mô hình **Recurrent Neural Network (RNN)** cơ bản để giải quyết bài toán Gán nhãn từ loại (Part-of-Speech Tagging). Mô hình được huấn luyện để gán nhãn ngữ pháp (như Danh từ, Động từ, Tính từ...) cho từng từ trong câu tiếng Anh.

* **Dữ liệu:** Universal Dependencies (UD_English-EWT).
* **Mô hình:** Simple RNN (Custom build).
* **Thư viện:** PyTorch.

---

## II. Các bước triển khai

### 1. Xử lý dữ liệu (Data Preprocessing)

* **Đọc dữ liệu CoNLL-U:** Viết hàm `load_conllu` để trích xuất từ (`FORM`) và nhãn từ loại (`UPOS`) từ file dữ liệu thô.
* **Xây dựng từ điển (Vocabulary):**
* Tạo `word_to_ix`: Ánh xạ từ sang chỉ số. Bao gồm token đặc biệt `<UNK>` (cho từ lạ) và `<PAD>` (cho đệm). Kích thước từ điển: **16,656** từ.
* Tạo `tag_to_ix`: Ánh xạ nhãn sang chỉ số. Kích thước: **18** nhãn (bao gồm nhãn đệm).



### 2. Chuẩn bị Pipeline dữ liệu (Dataset & DataLoader)

* **Dataset:** Lớp `POSDataset` chuyển đổi văn bản thành các tensor chỉ số. Xử lý từ lạ bằng cách gán về chỉ số của `<UNK>`.
* **Collator & Padding:**
* Sử dụng hàm `collate_fn` và `pad_sequence` để xử lý các câu có độ dài không đồng đều trong một batch.
* Các câu ngắn được thêm token `<PAD>` vào cuối để đạt độ dài bằng câu dài nhất trong batch.


* **DataLoader:** Chia dữ liệu thành các batch kích thước 32 (Batch Size = 32).

### 3. Kiến trúc Mô hình (Model Architecture)

Mô hình `SimpleRNNForTokenClas` bao gồm 3 thành phần chính:

1. **Embedding Layer:** Kích thước `(Vocab_Size, 100)`. Chuyển đổi chỉ số từ thành vector đặc trưng 100 chiều. Hỗ trợ tham số `padding_idx` để bỏ qua tính toán cho token đệm.
2. **RNN Layer:** Kích thước ẩn `hidden_dim = 128`. Nhận chuỗi embedding và trả về chuỗi trạng thái ẩn (hidden states) đại diện cho ngữ cảnh của từng từ.
3. **Linear Layer:** Kích thước `(128, 18)`. Ánh xạ trạng thái ẩn sang xác suất của 18 nhãn từ loại.

### 4. Thiết lập Huấn luyện

* **Loss Function:** `CrossEntropyLoss` với `ignore_index=PAD_TAG_INDEX`. Điều này cực kỳ quan trọng để mô hình không tính lỗi (loss) tại các vị trí là token đệm `<PAD>`.
* **Optimizer:** `Adam` với learning rate `0.001`.
* **Training Loop:** Thực hiện huấn luyện qua 10 epochs, có tích hợp đánh giá (Validation) trên tập Dev sau mỗi epoch để theo dõi hiệu năng.

---

## III. Cách chạy code và Ghi log kết quả

### 1. Cách chạy code

* **Môi trường:** Python 3.11, PyTorch. Có thể chạy trên Google Colab hoặc Local Jupyter Notebook.
* **Thực thi:** Chạy lần lượt các cell từ trên xuống dưới.
* **Lưu ý:** Cần đảm bảo đường dẫn file dữ liệu (`.conllu`) chính xác. Code hỗ trợ tự động chuyển sang GPU (`cuda`) để tăng tốc.

### 2. Kết quả thống kê từ log thực tế

Dưới đây là bảng số liệu đầy đủ được ghi nhận từ quá trình huấn luyện 10 Epochs trong notebook:

| Epoch | Thời gian (s) | Train Loss | Train Accuracy | Dev Loss | Dev Accuracy | Nhận xét |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2.20s | 0.3493 | 96.31% | 1.4303 | 87.85% | Khởi đầu tốt |
| **2** | **1.83s** | **0.3027** | **96.84%** | **1.4790** | **88.08%** | **Best Dev Accuracy** 🏆 |
| 3 | 1.81s | 0.2639 | 97.27% | 1.5426 | 87.94% | Dev Acc giảm nhẹ |
| 4 | 1.81s | 0.2273 | 97.66% | 1.5930 | 87.87% |  |
| 5 | 1.83s | 0.1955 | 97.99% | 1.6623 | 87.79% |  |
| 6 | 1.82s | 0.1699 | 98.27% | 1.7088 | 87.82% |  |
| 7 | 1.80s | 0.1454 | 98.52% | 1.7773 | 87.87% |  |
| 8 | 1.82s | 0.1251 | 98.72% | 1.8454 | 87.41% | Dev Loss tăng cao |
| 9 | 1.84s | 0.1080 | 98.90% | 1.9170 | 87.56% |  |
| 10 | 1.80s | 0.0945 | 99.01% | 1.9912 | 87.02% | Overfitting rõ rệt |

---

## IV. Giải thích các kết quả thu được

### 1. Phân tích quá trình huấn luyện

* **Điểm tối ưu:** Mô hình đạt hiệu suất tốt nhất trên tập kiểm thử (Dev) tại **Epoch 2** với độ chính xác **88.08%**.
* **Hiện tượng Overfitting:**
* Trên tập Train: Loss giảm đều đặn (0.34 -> 0.09) và Accuracy tăng tiệm cận mức tuyệt đối (96% -> 99%).
* Trên tập Dev: Từ sau Epoch 2, Accuracy bắt đầu chững lại và giảm dần (88.08% -> 87.02%), trong khi Dev Loss tăng mạnh (1.47 -> 1.99).
* *Kết luận:* Mô hình Simple RNN bắt đầu học thuộc lòng dữ liệu huấn luyện thay vì tổng quát hóa quy luật ngữ pháp từ sau Epoch 2.



### 2. Phân tích kết quả dự đoán (Inference Task)

Mô hình đã được thử nghiệm với câu: *"I love NLP and PyTorch"*

| Từ (Token) | Nhãn dự đoán | Nhãn thực tế (Kỳ vọng) | Đánh giá |
| --- | --- | --- | --- |
| **I** | `PRON` (Đại từ) | `PRON` | ✅ Chính xác |
| **love** | `VERB` (Động từ) | `VERB` | ✅ Chính xác |
| **NLP** | `PROPN` (Danh từ riêng) | `PROPN` | ✅ Chính xác |
| **and** | `CCONJ` (Liên từ) | `CCONJ` | ✅ Chính xác |
| **PyTorch** | `ADV` (Trạng từ) | `PROPN` | ❌ Sai |

* **Giải thích lỗi:** Từ "PyTorch" bị gán nhãn sai thành `ADV` (Trạng từ).
* *Nguyên nhân:* "PyTorch" có thể là từ không có trong từ điển (`<UNK>`). Mô hình RNN đơn giản có thể gặp khó khăn khi dựa vào ngữ cảnh "and ..." để suy luận ra đây là một danh từ riêng, dẫn đến dự đoán sai.



---

## V. Khó khăn gặp phải và Cách giải quyết

### 1. Vấn đề độ dài câu không đồng nhất

* **Khó khăn:** Không thể gom các câu có độ dài khác nhau vào cùng một Tensor để tính toán song song trên GPU.
* **Giải quyết:** Sử dụng kỹ thuật **Padding**. Thêm các giá trị `0` (hoặc index của `<PAD>`) vào cuối câu ngắn. Khi tính Loss, sử dụng tham số `ignore_index` để bỏ qua các vị trí này.

### 2. Từ vựng chưa biết (Out-Of-Vocabulary - OOV)

* **Khó khăn:** Khi gặp các từ mới (ví dụ tên riêng, thuật ngữ mới như "PyTorch") trong tập test, mô hình sẽ bị lỗi nếu không có cơ chế xử lý.
* **Giải quyết:** Xây dựng token đặc biệt `<UNK>` trong từ điển. Mọi từ không tìm thấy trong `word_to_ix` sẽ được ánh xạ về index của `<UNK>`.

### 3. Hạn chế của Simple RNN

* **Khó khăn:** Simple RNN gặp vấn đề **Vanishing Gradient** (biến mất đạo hàm), khiến nó khó học được các phụ thuộc xa trong câu dài (ví dụ: chủ ngữ ở đầu câu ảnh hưởng đến động từ ở cuối câu).
* **Giải quyết (Định hướng):** Trong các bài nâng cao, nên thay thế `nn.RNN` bằng `nn.LSTM` (Long Short-Term Memory) hoặc `nn.GRU` để cải thiện khả năng ghi nhớ dài hạn và tăng độ chính xác.

---

## VI. Thông tin Model và Nguồn tham khảo

### 1. Thông tin Model (Custom Build)

Mô hình được xây dựng từ đầu (Train from scratch), không sử dụng pre-trained weights.

* **Loại mô hình:** Token Classification (Sequence Labeling).
* **Kiến trúc:** Embedding (100) -> Simple RNN (128) -> Linear -> Softmax.
* **Prompt/Input:** Câu văn bản tiếng Anh được tách từ (tokenized).

### 2. Nguồn tham khảo

* **Dữ liệu:** Universal Dependencies (UD) - English EWT (English Web Treebank).
* **Tài liệu kỹ thuật:**
* [PyTorch Documentation - RNN](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
* [PyTorch Documentation - Padding Sequence](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html)