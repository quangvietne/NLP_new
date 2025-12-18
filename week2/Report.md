# Báo cáo – Lab 17: Spark NLP Pipeline

## 1. Các bước triển khai

- Đọc dữ liệu từ file `data/c4-train.00000-of-01024-30K.json` (giới hạn số dòng đọc vào qua biến `limitDocuments`, mặc định 1000, có thể sửa trực tiếp trong code).
- Tiền xử lý văn bản:
  - Tách từ (tokenization) bằng `Tokenizer` hoặc `RegexTokenizer`.
  - Loại bỏ stopwords với `StopWordsRemover`.
- Vector hóa:
  - Sử dụng `HashingTF` để chuyển từ thành vector đặc trưng (TF).
  - Sử dụng `IDF` để tính trọng số nghịch đảo tần suất tài liệu (TF-IDF).
- Thực hiện các pipeline nâng cao:
  - Thay đổi loại tokenizer.
  - Thay đổi kích thước vector hóa (numFeatures).
  - Thêm bước phân loại với `LogisticRegression`.
  - Thử vector hóa bằng `Word2Vec`.
- Ghi log và kết quả ra các file riêng biệt trong thư mục `log/` và `results/`.
- **Bổ sung các tính năng nâng cao:**
  - Đo thời gian chi tiết từng bước (đọc dữ liệu, fit, transform, ghi file kết quả) cho tất cả các pipeline và ghi vào log tương ứng.
  - Chuẩn hóa vector đầu ra của Word2Vec bằng `Normalizer`.
  - Tìm kiếm tài liệu tương tự nhất bằng cosine similarity: chọn một văn bản bất kỳ, tính độ tương đồng cosine giữa vector của văn bản đó với tất cả các văn bản còn lại, in ra 5 văn bản có độ tương đồng cao nhất (không tính chính nó).

---

## 2. Cách chạy code và log kết quả (How to run the code and log the results)

1. Biên dịch và chạy chương trình bằng sbt:

   - clone code về và chạy như sau :
   - sbt "runMain com.quangviet.spark.Lab17_NLPPipeline" ( với file Lab17_NLPPipeline )
   - sbt "runMain com.quangviet.spark.Lab17_NLPPipeline_Word2Vec_lr" ( với file Lab17_NLPPipeline_Word2Vec_lr , đây là file có thêm Word2Vec và Logistic Regression như phần Exercises)

2. log và result ( trong folder log và result , copy past vào thì dài quá nên em ko để ở đây )
    - Kết quả sẽ được ghi ra các file:
    - Log: `log/lab17_metrics_word2vec_lr.log` , `log/lab17_metrics.log`
    - Output: `lab17_pipeline_output_word2vec_lr.txt` , `lab17_pipeline_output.txt`
    

## 3. Giải thích kết quả thu được

1. **Switch Tokenizers**

- Tokenizer: tách từ đơn giản theo khoảng trắng → nhanh, ít lỗi nhưng không xử lý dấu câu.
- RegexTokenizer: tách chi tiết hơn (loại bỏ ký tự đặc biệt, dấu câu).

2. **Giảm numFeatures từ 20000 → 1000**

- Vector TF-IDF ngắn hơn, ít chiều hơn.
- Khi vocab thực tế lớn hơn 1000 → xảy ra hash collision (nhiều từ khác nhau ánh xạ vào cùng một index).

3. **Logistic Regression**

- Có thể train mô hình phân loại cơ bản sau khi có vector TF-IDF.
- Vì dataset không có nhãn thật → cần tạo label giả (vd: random hoặc rule-based) để demo.

4. **Word2Vec**

- Thay vì vector TF-IDF thưa, Word2Vec sinh vector dense (embedding).
- Captures ngữ nghĩa tốt hơn TF-IDF, nhưng tốn tài nguyên tính toán hơn.

## 4. Khó khăn gặp phải và cách giải quyết (Difficulties and Solutions)

- Dataset không có nhãn → Logistic Regression không train được.
- Giải pháp: tạo cột label giả để chạy thử mô hình


## 5. Tài liệu tham khảo

Tài liệu chính thức Apache Spark MLlib:
- https://spark.apache.org/docs/latest/ml-guide.html

