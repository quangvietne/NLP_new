# BÁO CÁO THỰC HÀNH week6
## Lưu ý , code đặt ở notebook/nlp-lab6.ipynb


## I. Tổng quan
Bài thực hành nhằm mục đích làm quen và ứng dụng thư viện `transformers` của Hugging Face để giải quyết ba bài toán NLP cơ bản:
1.  **Masked Language Modeling (MLM):** Điền từ còn thiếu vào câu.
2.  **Text Generation (Causal LM):** Sinh văn bản tự động từ câu mồi.
3.  **Sentence Embedding:** Trích xuất vector đặc trưng của câu sử dụng BERT và kỹ thuật Mean Pooling thủ công.

---

## II. Các bước triển khai

### 1. Chuẩn bị môi trường
- Cài đặt thư viện `transformers` để có thể tải và sử dụng các mô hình pre-trained.
- pip install -q transformers
### 2. Tác vụ 1: Điền từ vào chỗ trống (Fill-Mask)
Mô hình: bert-base-uncased.

Input: Câu văn chứa token đặc biệt [MASK]: "Hanoi is the [MASK] of Vietnam."

Quy trình:

Khởi tạo pipeline fill-mask.

Mô hình dự đoán các từ có khả năng điền vào vị trí [MASK].

Lấy ra 5 kết quả có xác suất cao nhất (top_k=5).

### 3. Tác vụ 2: Sinh văn bản (Text Generation)
Mô hình: gpt2 (mặc định của pipeline text-generation).

Input: Câu mồi (prompt): "The best thing about learning NLP is"

Quy trình:

Khởi tạo pipeline text-generation.

Thiết lập tham số max_length=50 (độ dài tối đa) và num_return_sequences=1.

Mô hình tự động dự đoán các từ tiếp theo để hoàn thành đoạn văn.

### 4. Tác vụ 3: Trích xuất Vector câu (Sentence Embedding)
Mô hình: bert-base-uncased.

Input: Câu văn bản: "This is a sample sentence."

Quy trình:

Tokenize câu đầu vào và chuyển thành PyTorch tensor.

Đưa qua mô hình BERT để lấy last_hidden_state.

Mean Pooling (Thủ công):

Sử dụng attention_mask để xác định token thật và token đệm (padding).

Nhân trạng thái ẩn với mask để loại bỏ giá trị của padding.

Tính tổng và chia cho số lượng token thật để ra vector trung bình.

# III. Cách chạy code và Ghi log kết quả
### 1. Cách chạy
Chạy code trên môi trường Jupyter Notebook (Google Colab, Kaggle hoặc Local).

Thực thi tuần tự từng cell (Run All).

Lưu ý: Cần có kết nối Internet để tải model từ Hugging Face Hub trong lần chạy đầu tiên.

### 2. Ghi log kết quả
Kết quả được in trực tiếp ra màn hình console (stdout):

Kết quả Tác vụ 1 (Fill-Mask):

Câu gốc: Hanoi is the [MASK] of Vietnam.

Dự đoán: 'capital' (Score: 0.9991)

Các dự đoán khác: 'center', 'birthplace', 'headquarters' (xác suất rất thấp ~0.0001).

Kết quả Tác vụ 2 (Text Generation):

Câu mồi: 'The best thing about learning NLP is'

Văn bản sinh ra: "The best thing about learning NLP is that it's so easy to grasp... [đoạn văn tiếp diễn]..."

Kết quả Tác vụ 3 (Sentence Embedding):

Vector biểu diễn: tensor([[-6.3875e-02, -4.2837e-01, ...]])

Kích thước: torch.Size([1, 768])

# IV. Giải thích kết quả
Tại sao BERT chọn "capital"?

BERT là mô hình Encoder-only với kiến trúc Bidirectional (hai chiều). Nó quan sát được ngữ cảnh cả trước ("Hanoi is the") và sau ("of Vietnam") token [MASK]. Dựa trên tri thức đã học, mô hình xác định mối quan hệ giữa "Hanoi" và "Vietnam" là quan hệ thủ đô - đất nước.

Tại sao GPT sinh được văn bản?

GPT là mô hình Decoder-only (Autoregressive). Nó hoạt động theo cơ chế Unidirectional (một chiều), chỉ nhìn các từ phía trước để dự đoán từ tiếp theo. Điều này làm cho GPT rất giỏi trong việc sáng tạo và duy trì mạch văn xuôi.

Ý nghĩa kích thước vector 768?

Đây là kích thước hidden_size mặc định của mô hình bert-base. Vector này nén thông tin ngữ nghĩa của toàn bộ câu input. Việc sử dụng Attention Mask trong bước tính trung bình là bắt buộc để đảm bảo vector không bị sai lệch bởi các token đệm (padding) vô nghĩa.

# V. Khó khăn gặp phải và Cách giải quyết
### 1. Cảnh báo thư viện (Warnings)
Khó khăn : Khi chạy code xuất hiện nhiều thông báo lỗi đỏ như Unable to register cuFFT factory hoặc AttributeError: 'MessageFactory'....

Nguyên nhân: Xung đột phiên bản nội bộ giữa các thư viện (TensorFlow/PyTorch) hoặc driver CUDA trên môi trường thực thi (thường gặp trên Kaggle/Colab).

Giải quyết: Đây chỉ là cảnh báo (warning) và không ảnh hưởng đến logic chính. Code vẫn chạy ra kết quả đúng. Nếu kernel bị treo, cần thực hiện Restart Runtime và chạy lại.

### 2. Xử lý Padding khi tính Embedding
Khó khăn : Nếu tính trung bình cộng trực tiếp trên toàn bộ output của BERT, các giá trị của token [PAD] (padding) sẽ làm sai lệch vector đại diện câu.

Giải quyết: Áp dụng thuật toán Weighted Mean Pooling:

# VI. Nguồn tham khảo
### Dữ liệu: File notebook nlp-lab6.ipynb.

### Mô hình:

- bert-base-uncased (Google) - Hugging Face Hub.

- gpt2 (OpenAI) - Hugging Face Hub.

### Tài liệu kỹ thuật: Hugging Face Transformers Documentation (Pipeline, Model Architecture).