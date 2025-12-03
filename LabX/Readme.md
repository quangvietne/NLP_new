# Báo cáo nghiên cứu: Text-to-Speech (TTS) – Tổng quan, hướng triển khai, và pipeline tối ưu





## Tổng quan về TTS

TTS là công nghệ chuyển đổi văn bản thành âm thanh giọng người. Hệ thống TTS thường gồm ba khối chính: phân tích văn bản, mô-đun sinh đặc trưng giọng nói (prosody, phoneme/char), và bộ biến đổi đặc trưng thành dạng sóng (vocoder).

- **Mục tiêu cốt lõi:**  
  - **Tính tự nhiên:** Gần với giọng người thật.  
  - **Hiệu suất:** Tốc độ sinh nhanh, độ trễ thấp.  
  - **Đa ngôn ngữ:** Khả năng hỗ trợ nhiều ngôn ngữ/giọng địa phương.  
  - **Cảm xúc:** Biểu đạt cảm xúc, nhấn nhá hợp ngữ cảnh.  
  - **Khả dụng:** Ít công sức người dùng, phù hợp thiết bị phổ thông.

- **Ba cấp độ tiếp cận (level):**  
  - **Level 1:** Dựa trên luật âm tiết và quy tắc phát âm.  
  - **Level 2:** Dựa trên mô hình học sâu (Deep Learning).  
  - **Level 3:** Few-shot/one-shot voice cloning theo giọng cho trước.

---

## So sánh các hướng triển khai

| Level | Cách tiếp cận | Ưu điểm | Nhược điểm | Trường hợp sử dụng |
|------|----------------|--------|------------|--------------------|
| 1 | Luật âm tiết và quy tắc phát âm | Nhanh; dễ mở rộng đa ngôn ngữ; ít tài nguyên | Giọng “máy”, thiếu tự nhiên và cảm xúc | Thiết bị nhúng; trình đọc văn bản trợ năng; hệ thống yêu cầu độ trễ cực thấp |
| 2 | Học sâu (DL TTS) | Giọng tự nhiên; cá nhân hóa; linh hoạt pipeline | Cần nhiều dữ liệu; khó đa ngôn ngữ nếu thiếu corpora; chi phí huấn luyện | Trợ lý ảo; audiobook; e-learning; ứng dụng thương mại |
| 3 | Few-shot/one-shot cloning | Giữ đặc trưng giọng cá nhân với ít dữ liệu; trải nghiệm cao | Mô hình phức tạp; tốn tài nguyên; rủi ro deepfake | Giải trí, game/phim; thương hiệu cá nhân; bản địa hóa nhanh |



---

## Phân tích chi tiết từng hướng

### Level 1: TTS dựa luật âm tiết

- **Bản chất kỹ thuật:**  
  - **Quy tắc phát âm:** Chuyển văn bản thành chuỗi âm vị/âm tiết dựa từ điển và luật ghép.  
  - **Prosody đơn giản:** Nhấn/nhịp/độ cao tuyến tính hoặc theo mẫu cố định.  
  - **Synthesis:** Ghép waveform từ đơn vị ghi âm sẵn (concatenative) hoặc tham số (formant).

- **Ưu điểm:**  
  - **Hiệu suất:** Rất nhanh, phù hợp thời gian thực.  
  - **Tài nguyên:** Ít yêu cầu tính toán và dữ liệu.  
  - **Đa ngôn ngữ:** Chỉ cần xây luật/ từ điển cho ngôn ngữ mới.

- **Nhược điểm:**  
  - **Tự nhiên:** Giọng cứng, máy móc; khó biểu đạt cảm xúc.  
  - **Linh hoạt:** Khó thích ứng ngữ cảnh/phong cách.

- **Ứng dụng phù hợp:**  
  - **Thiết bị nhúng:** IoT, thiết bị trợ năng, hệ thống thông báo.  
  - **Bài toán tốc độ cao:** Đọc văn bản đơn giản, ticker thông tin.

- **Pipeline :**  
  - **Tiền xử lý văn bản:**  
    - **Chuẩn hóa:** Số, ký hiệu, viết tắt.  
    - **Tách câu/đoạn:** Giảm lỗi ngắt nghỉ.  
  - **Phonemization lai:**  
    - **Quy tắc + thống kê:** Dùng luật làm nền, bổ sung mô hình thống kê cho từ mượn/ngoại lệ.  
  - **Prosody tăng cường:**  
    - **Mẫu nhấn nhịp động:** Theo dấu câu, loại câu hỏi/câu cảm thán.  
  - **Vocoder nhẹ:**  
    - **Concatenative tối ưu:** Kho âm phong phú hơn theo ngữ cảnh; giảm ghép thô.  
  - **Đánh giá nhanh:**  
    - **MOS nội bộ:** Đánh giá cảm nhận; tối ưu chỗ ngắt nghỉ.

### Level 2: TTS học sâu (Deep Learning)

- **Bản chất kỹ thuật:**  
  - **Encoder-decoder/Transformer:** Học ánh xạ từ text/phoneme sang spectrogram.  
  - **Vocoder DL:** Biến spectrogram thành waveform (WaveNet/WaveRNN/HiFi-GAN).  
  - **Cá nhân hóa:** Fine-tune theo giọng riêng.

- **Ưu điểm:**  
  - **Tự nhiên:** Giọng gần người thật, nhấn nhá hợp ngữ cảnh.  
  - **Linh hoạt:** Dễ thêm emotion/style; mở rộng pipeline.

- **Nhược điểm:**  
  - **Dữ liệu:** Cần corpora lớn/đa dạng; annotation tốn công.  
  - **Tài nguyên:** Huấn luyện/vocoder có thể đắt đỏ.  
  - **Đa ngôn ngữ:** Thiếu dữ liệu cho ngôn ngữ hiếm.

- **Ứng dụng phù hợp:**  
  - **Thương mại/chất lượng cao:** Trợ lý ảo, audiobook, giảng dạy số.  
  - **Cá nhân hóa:** Voice brand cho dịch vụ.

- **Pipeline :**  
  - **Chuẩn bị dữ liệu:**  
    - **Làm sạch/chuẩn hóa:** Loại nhiễu, thống nhất transcript.  
    - **Data augmentation:** Tăng đa dạng (speed/pitch, room impulse).  
  - **Huấn luyện đa nhiệm:**  
    - **Multi-speaker/Multilingual:** Chia sẻ tham số để giảm yêu cầu dữ liệu từng ngôn ngữ.  
    - **Transfer learning:** Khởi tạo từ mô hình nền, fine-tune cho domain.  
  - **Emotion/style control:**  
    - **Global Style Tokens (GST) hoặc embedding cảm xúc:** Điều khiển prosody.  
  - **Vocoder tối ưu:**  
    - **Lightweight GAN:** HiFi-GAN nhỏ; chuẩn hóa âm lượng và lọc nhiễu đầu ra.  
  - **Triển khai:**  
    - **Quantization/Pruning:** Giảm kích thước và độ trễ.  
    - **Batching/ONNX/TensorRT:** Tối ưu inference.  
  - **Đánh giá:**  
    - **MOS và objective metrics:** PESQ/STOI; latency; footprint bộ nhớ.

### Level 3: Few-shot/one-shot voice cloning

- **Bản chất kỹ thuật:**  
  - **Speaker encoder:** Trích xuất embedding giọng từ vài giây audio.  
  - **Synthesizer:** Tạo spectrogram theo text + speaker embedding.  
  - **Vocoder:** Tái tạo waveform theo spectrogram.

- **Ưu điểm:**  
  - **Cá nhân hóa nhanh:** Ít dữ liệu; giữ đặc trưng giọng.  
  - **Trải nghiệm:** Tự nhiên và sát giọng mục tiêu.

- **Nhược điểm:**  
  - **Tài nguyên:** Mô hình phức tạp, cần GPU cho chất lượng cao.  
  - **Đạo đức:** Nguy cơ giả mạo, deepfake; yêu cầu kiểm soát.

- **Ứng dụng phù hợp:**  
  - **Giải trí/sản xuất nội dung:** Dubbing, game, phim.  
  - **Thương hiệu cá nhân:** Bản địa hóa nhanh đa nền tảng.

- **Pipeline :**  
  - **Xác thực quyền giọng:**  
    - **Consent/bản quyền:** Lưu trữ bằng chứng cho phép sử dụng.  
  - **Speaker encoder robust:**  
    - **Huấn luyện đa miền:** Chống nhiễu, nhiều môi trường thu.  
    - **Voice activity detection (VAD):** Lọc đoạn phát âm rõ.  
  - **Cloning guardrails:**  
    - **Watermark:** Nhúng dấu nhận diện AI vào audio đầu ra.  
    - **Usage policy:** Giới hạn domain sử dụng, nhật ký truy cập.  
  - **Tối ưu hiệu suất:**  
    - **Distillation/Quantization:** Rút gọn mô hình.  
    - **Cache embedding:** Cho người dùng lặp lại.  
  - **Đánh giá:**  
    - **Speaker similarity score:** Cosine giữa embedding; MOS về tự nhiên & giống giọng.  
    - **Latency/throughput:** Đảm bảo dùng được thực tế.

---

## Thách thức chung và chiến lược khắc phục

- **Hiệu suất nhanh:**  
  - **Chiến lược:** Mô hình nhẹ; tối ưu inference; streaming TTS; vocoder nhanh.  
- **Ít tài nguyên tính toán:**  
  - **Chiến lược:** Quantization, pruning, distillation; dùng kiến trúc hiệu quả (Conformer/Light-Transformer).  
- **Tính tự nhiên cao:**  
  - **Chiến lược:** Huấn luyện với dữ liệu đa dạng; control prosody/emotion; cải thiện vocoder.  
- **Đa ngôn ngữ:**  
  - **Chiến lược:** Multilingual training, chuyển giao (transfer) giữa ngôn ngữ gần nhau; xây từ điển/phoneme chung (IPA).  
- **Cảm xúc giọng nói:**  
  - **Chiến lược:** Embedding cảm xúc, GST, prompt-based style; fine-tune theo domain.  
- **Ít công sức người dùng:**  
  - **Chiến lược:** Few-shot; giao diện thu âm đơn giản; tự động làm sạch audio.

---

## Đạo đức và bảo vệ người dùng

- **Watermark bắt buộc:** Nhúng dấu nhận diện vào mọi đầu ra AI để phân biệt với giọng thật, giảm rủi ro thông tin sai lệch.  
- **Xin phép & minh bạch:**  
  - **Consent rõ ràng:** Chỉ clone khi có quyền.  
  - **Thông báo sử dụng:** Minh bạch khi phát audio tổng hợp.  
- **Chống lạm dụng:**  
  - **Giới hạn tính năng:** Ngăn tạo giọng người nổi tiếng/nhạy cảm nếu không có quyền.  
  - **Giám sát & nhật ký:** Theo dõi truy cập để phát hiện hành vi bất thường.

---


## Khuyến nghị theo bối cảnh sử dụng

- **Thiết bị nhúng/độ trễ thấp:**  
  - **Chọn:** Level 1 hoặc Level 2 tối giản với vocoder nhẹ.  
- **Sản phẩm thương mại cần tự nhiên:**  
  - **Chọn:** Level 2 với control prosody và vocoder chất lượng, tối ưu inference.  
- **Cá nhân hóa nhanh theo giọng riêng:**  
  - **Chọn:** Level 3, triển khai thêm watermark và quy trình consent.

---


