# Báo cáo Lab 5: Xây dựng mô hình RNN cho bài toán Nhận dạng Thực thể Tên (NER)



## 2. Các bước thực hiện (Implementation Details)

### Task 1: Tải và Tiền xử lý Dữ liệu
* **Dữ liệu:** Sử dụng thư viện `datasets` của Hugging Face để tải bộ dữ liệu `conll2003`.
* **Xây dựng từ điển (Vocabulary):**
    * Tạo `word_to_ix`: Ánh xạ từ vựng sang chỉ số (index). Bổ sung hai token đặc biệt là `<PAD>` (dùng cho padding) và `<UNK>` (dùng cho từ lạ).
    * Tạo `tag_to_ix`: Ánh xạ các nhãn thực thể (B-PER, I-ORG, O,...) sang số nguyên.
* **Kết quả thống kê:**
    * Kích thước bộ từ điển (Vocab size): ~23,625 từ.
    * Số lượng nhãn (Num labels): 9 nhãn.

### Task 2: Tạo Dataset và DataLoader
* **Lớp `NERDataset`:** Kế thừa từ `torch.utils.data.Dataset`. Chức năng chính là chuyển đổi các câu văn dạng text sang các tensor chỉ số (indices) dựa trên từ điển đã xây dựng.
* **Hàm `collate_fn`:** Đây là thành phần quan trọng để xử lý Batch.
    * Sử dụng `pad_sequence` để thêm đệm (padding) sao cho các câu trong cùng một batch có độ dài bằng nhau.
    * Giá trị padding cho từ (tokens): `0` (index của `<PAD>`).
    * Giá trị padding cho nhãn (tags): `-1` (để hàm Loss bỏ qua không tính lỗi).

### Task 3: Xây dựng Kiến trúc Mô hình
Mô hình `SimpleRNNForNER` được thiết kế với 3 lớp chính:
1.  **Embedding Layer:** Chuyển đổi index của từ thành vector đặc trưng (Dimension = 100).
2.  **RNN Layer:** Lớp mạng hồi quy để nắm bắt thông tin chuỗi (Hidden Dimension = 128).
3.  **Linear Layer:** Lớp kết nối đầy đủ (Fully Connected) để ánh xạ trạng thái ẩn sang xác suất của 9 lớp nhãn NER.

### Task 4 & 5: Huấn luyện và Đánh giá
* **Loss Function:** Sử dụng `CrossEntropyLoss` với tham số `ignore_index=-1`. Điều này cực kỳ quan trọng để đảm bảo mô hình không học từ các token đệm.
* **Optimizer:** Sử dụng `Adam` với learning rate `0.01`.
* **Metric:** Accuracy (được tính toán bằng cách loại bỏ các vị trí padding).

---

## 3. Hướng dẫn chạy code (How to Run)

### Yêu cầu thư viện
- Đảm bảo đã cài đặt các thư viện cần thiết:
- ```bash
- pip install torch datasets seqeval
- Các bước thực hiện
- Mở file lab5_rnn_for_ner.ipynb 

- Chạy lần lượt các cell từ trên xuống dưới (Task 1 -> Task 5).

- Mô hình sẽ tự động tải dữ liệu, huấn luyện trong 5 epochs và in ra kết quả dự đoán mẫu.

## 4. Phân tích kết quả thực nghiệm
### Bảng kết quả huấn luyện

| Epoch | Average Loss | Nhận xét |
| :---: | :---: | :--- |
| 1 | 0.3697 | Loss giảm nhanh, mô hình bắt đầu học được các đặc trưng cơ bản. |
| 2 | 0.1077 | Loss giảm mạnh, mô hình hội tụ tốt. |
| 3 | 0.0539 | Tốc độ giảm chậm lại, bắt đầu đi vào chi tiết. |
| 4 | 0.0377 | |
| 5 | 0.0309 | Mô hình đạt trạng thái ổn định. |

- Kết quả đánh giá trên tập Validation
- Accuracy: 95.04%
###  Phân tích : 
 + Về chỉ số Accuracy: Con số 95.04% thoạt nhìn rất ấn tượng. Tuy nhiên, trong bài toán NER, phần lớn các nhãn là O (Outside - không phải thực thể). Do đó, accuracy cao có thể do mô hình dự đoán tốt lớp O, chưa chắc đã nhận diện tốt các thực thể hiếm (B-LOC, B-MISC...). Để đánh giá chính xác hơn, trong tương lai cần sử dụng F1-score.

 + Về kết quả dự đoán thực tế:

 + Câu input: "VNU University is located in Hanoi"

 + Dự đoán: VNU (B-ORG), University (I-ORG), Hanoi (O)

 + Nhận xét: Mô hình nhận diện đúng tổ chức "VNU University" nhưng lại bỏ sót địa danh "Hanoi" (dự đoán là O thay vì B-LOC).

 + Nguyên nhân: Có thể do kiến trúc RNN cơ bản (Vanilla RNN) gặp vấn đề Vanishing Gradient khi xử lý các chuỗi dài hoặc ngữ cảnh xa, hoặc do từ "Hanoi" xuất hiện ít trong tập train.

## 5. Khó khăn và Giải pháp
Trong quá trình thực hiện Lab 5, tôi đã gặp một số thách thức và giải quyết như sau:

### 1. Vấn đề kích thước Batch không đồng nhất (Variable Length Sequences)

Khó khăn: Các câu trong dataset có độ dài khác nhau, không thể xếp chồng thành một Tensor hình chữ nhật để đưa vào GPU.

Giải pháp: Sử dụng collate_fn kết hợp với pad_sequence của PyTorch để thêm padding vào các câu ngắn hơn. Đồng thời thiết lập batch_first=True cho lớp RNN để khớp dimension.

### 2. Tính toán Loss cho phần Padding

Khó khăn: Nếu tính cả phần padding vào Loss, mô hình sẽ bị nhiễu vì nó phải học cách dự đoán token vô nghĩa <PAD>.

Giải pháp: Gán giá trị nhãn padding là -1 và thiết lập ignore_index=-1 trong hàm CrossEntropyLoss.

### 3. Shape Mismatch khi tính Loss

Khó khăn: Output của RNN có dạng (Batch, Seq_len, Num_classes) nhưng CrossEntropyLoss yêu cầu Input (N, C).

Giải pháp: Sử dụng hàm .view(-1, OUTPUT_SIZE) để "làm phẳng" (flatten) tensor dự đoán và nhãn trước khi đưa vào hàm loss.