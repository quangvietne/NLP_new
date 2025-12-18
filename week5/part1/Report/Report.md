

# BÁO CÁO THỰC HÀNH: week5_part1

## I. Tổng quan

Bài thực hành nhằm mục đích làm quen với **PyTorch**, một thư viện mã nguồn mở phổ biến cho Deep Learning. Nội dung bao gồm 3 phần chính:

1. **Tensor:** Cấu trúc dữ liệu cơ bản trong PyTorch (tương tự NumPy array nhưng chạy được trên GPU).
2. **Autograd:** Cơ chế tự động tính đạo hàm phục vụ cho việc huấn luyện mô hình.
3. **torch.nn:** Các module để xây dựng mạng nơ-ron (Neural Networks).

---

## II. Các bước triển khai

### 1. Thao tác với Tensor (Task 1)

* **Tạo Tensor:**
* Từ List Python: `torch.tensor(data)`.
* Từ NumPy array: `torch.from_numpy(np_array)`.
* Tensor đặc biệt: `torch.ones_like` (toàn số 1), `torch.rand_like` (ngẫu nhiên).
* Kiểm tra thuộc tính: `shape`, `dtype`, `device`.


* **Các phép toán:**
* Cộng, nhân vô hướng, nhân ma trận (`@` hoặc `matmul`).


* **Indexing & Slicing:** Truy cập phần tử tương tự như list hoặc numpy array.
* **Thay đổi kích thước (Reshape):** Sử dụng `.view()` hoặc `.reshape()` để thay đổi chiều của tensor mà không thay đổi dữ liệu gốc.

### 2. Thực hành với Autograd (Task 2)

* **Khởi tạo:** Tạo tensor với tham số `requires_grad=True` để thông báo cho PyTorch theo dõi các phép toán trên tensor này.
* **Lan truyền thuận (Forward):** Thực hiện các phép toán tính toán (ví dụ: y = x + 2, z = 3y^2).
* **Lan truyền ngược (Backward):** Gọi `z.backward()` để tính đạo hàm của z theo x.
* **Kết quả:** Đạo hàm được lưu tại thuộc tính `.grad` của tensor gốc.

### 3. Xây dựng Mô hình với `torch.nn` (Task 3)

* **Lớp Tuyến tính (`nn.Linear`):** Thực hiện phép biến đổi y = xA^T + b.
* **Lớp Embedding (`nn.Embedding`):** Bảng tra cứu để chuyển đổi các chỉ số (index) thành vector dày đặc (dense vectors).
* **Xây dựng lớp `MyFirstModel` kế thừa từ `nn.Module`:**
* Định nghĩa các lớp con trong `__init__`: Embedding, Linear, Activation (ReLU).
* Định nghĩa luồng dữ liệu trong `forward`: Input -> Embedding -> Linear -> ReLU -> Linear -> Output.



---

## III. Cách chạy code và Ghi log kết quả

### 1. Cách chạy

* **Môi trường:** Jupyter Notebook cài đặt sẵn thư viện `torch` và `numpy`.
* **Thực thi:** Chạy tuần tự các cell từ trên xuống dưới.
* **Lưu ý:** Code hiện tại đang chạy trên CPU (mặc định). Nếu muốn chạy trên GPU, cần bỏ chú thích dòng `torch.set_default_device('cuda')` (nếu có phần cứng hỗ trợ).

### 2. Ghi log kết quả

Kết quả được in ra màn hình console (stdout) tại mỗi bước:

**Kết quả Task 1 (Tensor):**

> * **Tensor từ list:** `tensor([[1, 2], [3, 4]])`
> * **Shape:** `torch.Size([2, 2])`
> * **Phép nhân ma trận:** Kết quả là `tensor([[5, 11], [11, 25]])`.
> 
> 

**Kết quả Task 2 (Autograd):**

> * **x:** `tensor([1.], requires_grad=True)`
> * **z:** `tensor([27.], grad_fn=<MulBackward0>)`
> * **Đạo hàm (x.grad):** `tensor([18.])`
> 
> 

**Kết quả Task 3 (Neural Network):**

> * **Input Shape:** `torch.Size([1, 4])` (Batch size = 1, Sequence length = 4).
> * **Output Shape:** `torch.Size([1, 4, 2])` (Mỗi từ trong câu 4 từ được map sang vector 2 chiều).
> * **Model Output:** `tensor([[[-0.2743, 0.1835], ...]])`
> 
> 

---

## IV. Giải thích kết quả

1. **Tại sao đạo hàm bằng 18?**
* Biểu thức: y = x + 2; z = 3y^2.
* Suy ra: z = 3(x+2)^2.
* Đạo hàm theo x: \frac{dz}{dx} = 3 \cdot 2 \cdot (x+2) = 6(x+2).
* Tại x = 1: \frac{dz}{dx} = 6(1+2) = 18.
* Kết quả `x.grad = 18` là chính xác.


2. **Cấu trúc mô hình `MyFirstModel`:**
* Đầu vào là các chỉ số (indices) của từ.
* Đi qua `nn.Embedding`: Chuyển số nguyên thành vector (ví dụ 16 chiều).
* Đi qua `nn.Linear`: Giảm chiều (từ 16 xuống 8).
* Đi qua `ReLU`: Loại bỏ các giá trị âm (Non-linearity).
* Đi qua `nn.Linear` cuối: Ra output dimension (ví dụ 2 chiều).



---

## V. Khó khăn gặp phải và Cách giải quyết

### 1. Lỗi gọi `backward()` lần 2

* **Vấn đề:** Nếu gọi `z.backward()` lần thứ hai mà không chạy lại lan truyền thuận, PyTorch sẽ báo lỗi `RuntimeError: Trying to backward through the graph a second time...`.
* **Nguyên nhân:** PyTorch sử dụng cơ chế **Dynamic Computation Graph**. Sau khi gọi `backward()`, biểu đồ tính toán sẽ được giải phóng (free) để tiết kiệm bộ nhớ.
* **Giải quyết:**
* Chỉ gọi `backward()` một lần sau mỗi lần `forward`.
* Nếu muốn gọi nhiều lần (ví dụ trong high-order derivatives), cần thêm tham số `retain_graph=True` vào lần gọi đầu: `z.backward(retain_graph=True)`.



### 2. Kích thước Tensor (Shape Mismatch)

* **Vấn đề:** Lỗi kích thước khi nhân ma trận hoặc đưa vào lớp Linear.
* **Giải quyết:** Luôn kiểm tra `.shape` của tensor trước khi thực hiện phép toán. Sử dụng `.view()` hoặc `.reshape()` để điều chỉnh kích thước cho phù hợp (ví dụ: làm phẳng ảnh trước khi đưa vào lớp Linear).

---

## VI. Nguồn tham khảo

1. **Tài liệu:** File notebook `lab5_pytorch_introduction.ipynb`.
2. **Thư viện:**
* [PyTorch Documentation](https://www.google.com/search?q=https://pytorch.org/docs/stable/index.html) (Tensors, Autograd, nn Module).
* NumPy Documentation.