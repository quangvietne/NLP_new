
# BÁO CÁO THỰC HÀNH PART 1: TÌM HIỂU VỀ RNNs VÀ PHÂN LOẠI TOKEN (PYTORCH BASICS)

## I. Các bước triển khai


### 1. Nghiên cứu lý thuyết (Lý thuyết nền tảng)


* **RNN (Recurrent Neural Networks):** Mô hình mạng nơ-ron chuyên xử lý dữ liệu chuỗi (như văn bản, âm thanh) bằng cách duy trì trạng thái ẩn (hidden state) qua các bước thời gian.
* **LSTM (Long Short-Term Memory) & GRU (Gated Recurrent Unit):** Các biến thể cải tiến của RNN giúp giải quyết vấn đề biến mất đạo hàm (vanishing gradient) khi xử lý chuỗi dài. * **Token Classification:** Bài toán gán nhãn cho từng từ (token) trong câu, ví dụ như gán nhãn từ loại (POS Tagging) hoặc nhận diện tên riêng (NER).

### 2. Thực hành PyTorch (Triển khai code)

Các bước thực hành cụ thể trong file `lab5_pytorch_introduction.ipynb`:

* **Bước 1: Làm quen với Tensor (Task 1):**
* Khởi tạo Tensor từ List và NumPy array.
* Tạo các Tensor đặc biệt (`ones`, `rand`).
* Thực hiện các phép toán cơ bản: Cộng, Nhân vô hướng, Nhân ma trận.
* Thao tác Indexing, Slicing và thay đổi kích thước (`view`, `reshape`).


* **Bước 2: Cơ chế Autograd (Task 2):**
* Thiết lập `requires_grad=True` để theo dõi lịch sử tính toán.
* Thực hiện lan truyền thuận (Forward pass) qua các biểu thức toán học.
* Thực hiện lan truyền ngược (Backward pass) để tính đạo hàm tự động.


* **Bước 3: Xây dựng Module mạng nơ-ron (Task 3):**
* Sử dụng `nn.Linear` cho các phép biến đổi tuyến tính.
* Sử dụng `nn.Embedding` để tạo bảng vector biểu diễn từ.
* Kết hợp các lớp này trong một class kế thừa từ `nn.Module` (lớp `MyFirstModel`) để tạo thành một mạng nơ-ron hoàn chỉnh gồm: Embedding -> Linear -> ReLU -> Linear -> Output.



---

## II. Cách chạy code và Ghi log kết quả

### 1. Cách chạy code

* **Môi trường:** Jupyter Notebook (Google Colab hoặc Local) cài đặt Python 3, thư viện `torch` và `numpy`.
* **Thao tác:** Chạy tuần tự các cell từ trên xuống dưới.

### 2. Ghi log kết quả

Kết quả dưới đây được trích xuất trực tiếp từ output của các cell trong notebook:

**Log Task 1 (Tensor Operations):**

> * **Tensor từ list:** `tensor([[1, 2], [3, 4]])`
> * **Phép cộng:** `tensor([[2, 4], [6, 8]])`
> * **Phép nhân ma trận (@):** `tensor([[5, 11], [11, 25]])`
> * **Shape/Reshape:** Chuyển từ `torch.Size([4, 4])` sang `torch.Size([16, 1])`.
> 
> 

**Log Task 2 (Autograd):**

> * **Giá trị x:** `tensor([1.], requires_grad=True)`
> * **Giá trị z (Forward):** `tensor([27.], grad_fn=<MulBackward0>)`
> * **Đạo hàm x.grad (Backward):** `tensor([18.])`
> 
> 

**Log Task 3 (Neural Network):**

> * **Linear Output:** `tensor([[ 0.4736, -1.0041], ...])` (Shape: `[3, 2]`)
> * **Embedding Output:** `tensor([[ 0.0562,  0.6852,  1.7225], ...])` (Shape: `[4, 3]`)
> * **MyFirstModel Output:** `tensor([[[-0.2743,  0.1835], ...]])` (Shape: `[1, 4, 2]`)
> 
> 

---

## III. Giải thích các kết quả thu được

### 1. Kết quả Autograd (Tại sao x.grad = 18?)

Trong Task 2, ta có quy trình tính toán:

1. Khởi tạo x = 1.
2. y = x + 2 = 3.
3. z = y \times y \times 3 = 3y^2.
4. Giá trị forward: z = 3 \times (3)^2 = 27.

Khi gọi `z.backward()`, PyTorch tính đạo hàm \frac{\partial z}{\partial x} theo quy tắc chuỗi (chain rule):



Tại x = 1: \frac{\partial z}{\partial x} = 6(1+2) = 18.
Kết quả `x.grad` trả về `18.` là hoàn toàn chính xác về mặt toán học.

### 2. Kết quả `MyFirstModel`

Mô hình `MyFirstModel` có kiến trúc:

* **Input:** Batch size = 1, Sequence length = 4 (4 từ).
* **Embedding:** Chuyển mỗi từ thành vector 16 chiều \rightarrow Shape: `[1, 4, 16]`.
* **Linear 1 + ReLU:** Chuyển từ 16 chiều xuống 8 chiều \rightarrow Shape: `[1, 4, 8]`.
* **Linear 2 (Output):** Chuyển từ 8 chiều ra 2 chiều (output dim) \rightarrow Shape: `[1, 4, 2]`.
Kết quả log `torch.Size([1, 4, 2])` khớp với lý thuyết thiết kế mạng.

---

## IV. Khó khăn gặp phải và Cách giải quyết

### 1. Lỗi gọi `backward()` lần thứ hai

* **Khó khăn:** Nếu bỏ comment dòng lệnh `z.backward()` lần 2 trong Task 2, chương trình sẽ báo lỗi `RuntimeError`.
* **Nguyên nhân:** PyTorch sử dụng biểu đồ tính toán động (dynamic computation graph). Sau khi gọi `backward()`, biểu đồ này mặc định bị giải phóng (free) để tiết kiệm bộ nhớ.
* **Giải quyết:**
* Cách 1: Chỉ gọi `backward()` một lần sau mỗi lần forward.
* Cách 2: Nếu cần gọi nhiều lần (ví dụ tính đạo hàm bậc cao), sử dụng tham số `retain_graph=True` trong lần gọi đầu: `z.backward(retain_graph=True)`.



### 2. Quản lý kích thước (Shape Mismatch)

* **Khó khăn:** Khi kết nối các lớp `Linear` hoặc `Embedding`, nếu kích thước `output_dim` của lớp trước không khớp với `input_dim` của lớp sau, code sẽ lỗi.
* **Giải quyết:** Luôn kiểm tra `.shape` của tensor đầu ra sau từng lớp (như cách code đã in ra `Model output shape`) để debug. Sử dụng `.view()` hoặc `.reshape()` nếu cần làm phẳng dữ liệu.

---

## V. Tài liệu tham khảo

1. **Tài liệu lý thuyết (theo yêu cầu Lab):**
* `lab5_pytorch_introduction.pdf` (Kiến thức cơ bản RNNs/LSTMs).
* `lecture5_rnn_token_classification.pdf` (Kiến thức về bài toán Token Classification).


2. **Tài liệu thực hành:**
* File notebook: `lab5_pytorch_introduction.ipynb`.


3. **Tài liệu ngoài:**
* PyTorch Documentation ([https://pytorch.org/docs/](https://www.google.com/search?q=https://pytorch.org/docs/)).



---

## VI. Thông tin Model tạo sẵn

* **Mô hình sử dụng:** Class `MyFirstModel` là một mạng nơ-ron đơn giản (Simple Neural Network) được xây dựng từ đầu (scratch) sử dụng các module cơ bản (`nn.Embedding`, `nn.Linear`, `nn.ReLU`) của PyTorch.