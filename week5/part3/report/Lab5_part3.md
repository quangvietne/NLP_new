

# BÁO CÁO THỰC HÀNH PART 3: PART-OF-SPEECH TAGGING VỚI RNN 


### ⊠ Các bước triển khai

Để giải quyết bài toán này, mình đã chia quy trình thành 3 công đoạn chính như sau:

**Task 1: Chuẩn bị dữ liệu**
Đầu tiên, mình viết hàm `load_conllu` để đọc dữ liệu thô. Từ file gốc chứa rất nhiều trường thông tin, mình chỉ lọc lấy cặp quan trọng nhất là `(Từ, Nhãn UPOS)`.
Sau đó, mình xây dựng bộ từ điển (Vocabulary):

* **Từ vựng:** Quét toàn bộ tập train, mình thu được **16,656** từ. Để xử lý các tình huống thực tế, mình thêm 2 token đặc biệt là `<UNK>` (cho từ lạ chưa gặp bao giờ) và `<PAD>` (để lấp đầy các câu ngắn).
* **Nhãn:** Tổng cộng có **18** nhãn (bao gồm cả nhãn padding).
* **Dataloader:** Vì các câu dài ngắn khác nhau, mình dùng hàm `collate_fn` kết hợp `pad_sequence` để "ép" chúng về cùng độ dài trong một batch thì mới đưa vào GPU tính toán được.

**Task 2: Xây dựng mô hình**
Mình tự code class `SimpleRNNForTokenClas` chứ không dùng model có sẵn. Kiến trúc khá cổ điển gồm 3 tầng:

1. **Embedding:** Chuyển index của từ thành vector 100 chiều.
2. **RNN:** Đây là trái tim của mô hình, dùng để quét qua chuỗi vector. Mình đặt kích thước ẩn (hidden dim) là 128.
3. **Linear:** Tầng cuối cùng để chiếu kết quả ra 18 nhãn xác suất.

**Task 3: Huấn luyện và Đánh giá**

* Mình dùng hàm loss `CrossEntropyLoss` (nhớ cài đặt `ignore_index` để mô hình không bị phạt oan khi dự đoán sai ở mấy chỗ padding).
* Tối ưu hóa bằng `Adam` với learning rate 0.001.
* Chạy liên tục 10 vòng (epochs), cứ học xong một vòng là mình cho kiểm tra (evaluate) ngay trên tập Dev để xem tình hình thế nào.

---

### ⊠ Cách chạy code và ghi log kết quả

**Cách chạy:**
Mở code trong notebook/NLP_pos_tagging.ipynb và chạy nó 


**Nhật ký huấn luyện (Log thực tế):**
Đây là kết quả chi tiết mình ghi lại được sau 10 epoch cày cuốc:

| Epoch | Train Loss | Train Accuracy | Dev Loss | Dev Accuracy | Nhận xét nhanh |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.3493 | 96.31% | 1.4303 | 87.85% | Khởi đầu khá ổn |
| **2** | **0.3027** | **96.84%** | **1.4790** | **88.08%** | **Đỉnh cao phong độ (Best Model)** 🏆 |
| 3 | 0.2639 | 97.27% | 1.5426 | 87.94% | Bắt đầu có dấu hiệu tụt dốc |
| 4 | 0.2273 | 97.66% | 1.5930 | 87.87% |  |
| 5 | 0.1955 | 97.99% | 1.6623 | 87.79% |  |
| 6 | 0.1699 | 98.27% | 1.7088 | 87.82% |  |
| 7 | 0.1454 | 98.52% | 1.7773 | 87.87% |  |
| 8 | 0.1251 | 98.72% | 1.8454 | 87.41% | Loss tăng cao quá |
| 9 | 0.1080 | 98.90% | 1.9170 | 87.56% |  |
| 10 | 0.0945 | 99.01% | 1.9912 | 87.02% | Học vẹt (Overfitting) nặng |

* **Kết luận:** Mô hình tốt nhất là ở **Epoch 2**.
* **Độ chính xác chốt hạ trên tập Dev:** **88.08%**.

---

### • [] Giải thích các kết quả thu được

1. **Chuyện gì đã xảy ra khi train?**
Nhìn vào bảng số liệu, có thể thấy mô hình Simple RNN này học rất nhanh, ngay epoch 2 đã đạt đỉnh. Tuy nhiên, từ epoch 3 trở đi xảy ra hiện tượng "học vẹt" (overfitting). Trong khi điểm trên tập Train cứ tăng vùn vụt (lên tới 99%) thì điểm trên tập Dev lại đi xuống và Loss thì tăng mạnh. Điều này chứng tỏ mô hình đang cố nhớ đáp án của tập train thay vì học quy luật ngữ pháp tổng quát.
2. **Thử thách với câu mới (Task Nâng cao):**
Mình đã viết hàm `predict_sentence` để test thử một câu ngoài lề: *"I love NLP and PyTorch"*.
Kết quả mô hình trả về như sau:
* `I` -> `PRON` (Đúng)
* `love` -> `VERB` (Đúng)
* `NLP` -> `PROPN` (Đúng)
* `and` -> `CCONJ` (Đúng)
* `PyTorch` -> **`ADV` (Sai)**.


*Tại sao sai?* Từ "PyTorch" có lẽ không có trong tập train, nên nó bị quy về token lạ `<UNK>`. Cộng thêm việc mô hình RNN đơn giản khả năng nhớ ngữ cảnh xa hơi kém, nên nó đoán bừa thành Trạng từ thay vì Danh từ riêng.

---

### • [] Nêu rõ các khó khăn gặp phải và cách giải quyết

Trong quá trình làm cũng gặp vài cái "hố", cụ thể là:

1. **Câu dài câu ngắn không đều:**
* *Vấn đề:* Máy tính thích ma trận vuông vức, mà câu văn thì độ dài vô chừng.
* *Giải quyết:* Mình phải dùng `pad_sequence` để chèn thêm token đệm `<PAD>` vào đuôi câu ngắn. Quan trọng nhất là lúc tính điểm số (Accuracy/Loss), mình phải bảo code lờ mấy cái token này đi (`ignore_index`), không thì kết quả sai lệch hết.


2. **Từ lạ chưa gặp bao giờ (OOV):**
* *Vấn đề:* Lúc test gặp từ mới toanh thì mô hình bị lỗi ngay.
* *Giải quyết:* Mình quy hoạch tất cả các từ lạ về một mối là token `<UNK>`. Thà đoán sai (như vụ PyTorch ở trên) còn hơn là sập chương trình.


3. **Hạn chế của kiến trúc RNN cổ điển:**
* *Vấn đề:* Train được tí xíu là bị Overfitting và khó học được mấy câu phức tạp do vấn đề biến mất đạo hàm (vanishing gradient).
* *Hướng giải quyết:* Bài này yêu cầu dùng RNN thì mình dùng, chứ nếu muốn xịn hơn thì chắc chắn phải chuyển sang LSTM hoặc GRU.



---

### • [] Nguồn tham khảo

* **Dữ liệu:** Mình sử dụng bộ Universal Dependencies v2.5 (English-EWT).
* **Code:** Tham khảo tài liệu chính chủ của PyTorch về `nn.RNN` và `CrossEntropyLoss`.

---

### • [] Thông tin Model

* **Model:** Đây là mô hình mình **tự xây dựng và huấn luyện từ đầu (train from scratch)**, hoàn toàn không sử dụng bất kỳ pre-trained model nào như BERT hay GPT.
* **Kiến trúc:** Simple RNN thuần túy.s