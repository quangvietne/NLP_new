# Lab 5: Text Classification 

# 1. Giải thích các bước thực hiện


### Bước 1: Xây dựng TextClassifier với Scikit-learn (Task 1 , 2 , 3)

### 1 . Tạo TextClassifier: Lớp TextClassifier đã được triển khai trong tệp src/models/text_classifier.py.
- Lớp này được khởi tạo với một vectorizer (ví dụ: TfidfVectorizer).
- Phương thức fit nhận vào texts và labels, sử dụng vectorizer.fit_transform để chuyển đổi văn bản và sau đó huấn luyện một mô hình LogisticRegression.
- Phương thức predict sử dụng vectorizer.transform trên văn bản mới và trả về dự đoán từ mô hình đã huấn luyện.
- Phương thức evaluate tính toán và trả về một dictionary các chỉ số (accuracy, precision, recall, f1) bằng cách sử dụng các hàm từ sklearn.metrics.

### 2 . Tạo tệp Test Case: Tệp lab5_test.ipynb được tạo để kiểm tra TextClassifier.
- Tập dữ liệu texts và labels nhỏ từ tài liệu hướng dẫn đã được định nghĩa.
- Dữ liệu được chia thành tập huấn luyện (4 mẫu) và tập kiểm tra (2 mẫu) bằng train_test_split.
- Một TfidfVectorizer của scikit-learn và TextClassifier đã được khởi tạo.
- Mô hình được huấn luyện trên tập X_train, dự đoán trên X_test, và các chỉ số đánh giá được in ra.

### Bước 2: Chạy Pipeline PySpark cơ bản (advand Task)
- Chạy và phân tích một pipeline học máy quy mô lớn hơn bằng Apache Spark (lab5_spark_sentiment_analysis.ipynb)

1 . Khởi tạo và Tải dữ liệu: Khởi tạo một SparkSession. Dữ liệu được tải từ tệp CSV, và cột label được chuẩn hóa thành 0 (negative) và 1 (positive). Các hàng có giá trị null đã bị loại bỏ.

2 . Xây dựng Pipeline: Một Pipeline của Spark ML đã được xây dựng , bao gồm các giai đoạn sau:
- Tokenizer: Tách văn bản thành các từ (tokens).
- StopWordsRemover: Loại bỏ các từ dừng phổ biến.
- HashingTF: Chuyển đổi token thành các vectơ đặc trưng thô bằng kỹ thuật băm.
- IDF: Tính toán lại trọng số của các vectơ đặc trưng HashingTF.
- LogisticRegression: Mô hình phân loại tuyến tính.

3 . Huấn luyện và Đánh giá: Pipeline được huấn luyện trên dữ liệu training bằng pipeline.fit(). Mô hình sau đó được đánh giá trên dữ liệu test bằng MulticlassClassificationEvaluator để tính toán các chỉ số Accuracy, F1, Precision và Recall.

### Bước 3: Thử nghiệm Cải thiện Mô hình (Task 4)
- Các thử nghiệm này được thực hiện trong tệp lab5_improvement_test.ipynb.
- Ba thử nghiệm đã được tiến hành:

Thử nghiệm 1: Thay thế TF-IDF bằng Word2Vec:
- Trong pipeline, các giai đoạn HashingTF và IDF đã được thay thế bằng một giai đoạn Word2Vec.
- Word2Vec học các nhúng từ (với vectorSize=100) và tạo ra một vectơ đặc trưng duy nhất cho mỗi tài liệu.
- Mô hình LogisticRegression vẫn được giữ nguyên.

Thử nghiệm 2: Thay thế LogisticRegression bằng Naive Bayes:
- Pipeline quay trở lại sử dụng HashingTF và IDF để tạo đặc trưng.
- Mô hình LogisticRegression ở giai đoạn cuối đã được thay thế bằng mô hình NaiveBayes, một mô hình xác suất thường hoạt động tốt cho văn bản.

Thử nghiệm 3: Thay thế LogisticRegression bằng Neural Network (MLP):
- Pipeline vẫn sử dụng HashingTF (với numFeatures=10000) và IDF.
- Mô hình LogisticRegression được thay thế bằng MultilayerPerceptronClassifier (MLP).
- Kiến trúc mạng được định nghĩa bằng layers = [10000, 64, 32, 2], trong đó 10000 là kích thước đầu vào (khớp với numFeatures), 64 và 32 là các lớp ẩn, và 2 là lớp đầu ra (cho 2 nhãn 0 và 1).

# 2. Hướng dẫn chạy code 
### Với task 1, 2 , 3 . 
- Chạy file lab5_test.ipynb và xem kết quả : 
- Result

 Train: 4  Test: 2

=== EVALUATION RESULTS ===
accuracy  : 0.0000
precision : 0.0000
recall    : 0.0000
f1        : 0.0000

Predictions vs True labels:
Text: This movie is fantastic and I love it!
   True: POSITIVE | Pred: NEGATIVE

Text: Could not finish watching, so bad.
   True: NEGATIVE | Pred: POSITIVE
- Giải thích : Do data nhỏ , nên mô hình học chưa tốt từ dữ liệu có sẵn -> Dự đoan sai tất 
### Với task advand : 
- Chạy file lab5_spark_sentiment_analysis.ipynb  để xem kq
- Result : 
Loaded 5792 rows initially, dropped 1 null rows, final count: 5791
Accuracy: 0.7295
F1 Score: 0.7266
Weighted Precision: 0.7255
Weighted Recall: 0.7295
### Với task 4 (improvement)
- Chạy file lab5_improvement_test.ipynb để xem kq
- Result : 
- Thay TF-IDF thành word2vec : 
Loaded 5792 rows initially, dropped 1 null rows, final count: 5791
Accuracy: 0.6411
F1 Score: 0.5710
Weighted Precision: 0.6222
Weighted Recall: 0.6411
- Thay LogisticRegression bằng NaiveBayes :
Loaded 5792 rows initially, dropped 1 null rows, final count: 5791
Accuracy: 0.6844
F1 Score: 0.6842
Weighted Precision: 0.6841
Weighted Recall: 0.6844
- Thay LogisticRegression bằng neural network : 
Loaded 5792 rows initially, dropped 1 null rows, final count: 5791
Starting model training (MLP)... This may take a while.
Model training complete.
Accuracy: 0.7755
F1 Score: 0.7736
Weighted Precision: 0.7730
Weighted Recall: 0.7755
# 3. Phân tích kết quả 
Phần này báo cáo các chỉ số hiệu suất của mô hình PySpark cơ sở và các mô hình cải tiến, đồng thời phân tích lý do cho sự khác biệt.

### 3.1. Hiệu suất Mô hình Cơ sở (Baseline)
Mô hình cơ sở sử dụng pipeline TF-IDF + LogisticRegression như được định nghĩa trong lab5_spark_sentiment_analysis.ipynb.
Accuracy: 0.7295
F1 Score: 0.7266

### 3.2. Hiệu suất các Mô hình Cải tiến
- Ba thử nghiệm cải tiến đã được thực hiện trong lab5_improvement_test.ipynb. Kết quả được tóm tắt trong bảng dưới đây
- Thử nghiệm (Pipeline),Accuracy,F1 Score
0. Baseline (TF-IDF + LogisticRegression), Accuracy : 0.7295, F1 Score: 0.7266,-
1. Word2Vec + LogisticRegression,Accuracy : 0.6411, F1 Score: 0.5710
2. TF-IDF + NaiveBayes, Accuracy : 0.6844, F1 Score: 0.6842
3. TF-IDF + Neural Network (MLP),Accuracy : 0.6844 0.7755, F1 Score: 0.7736

- Baseline (TF-IDF + LogisticRegression): Mô hình tuyến tính cơ sở hoạt động khá tốt, đạt Accuracy 0.7295.
- Thất bại (Word2Vec, NaiveBayes):
+ Word2Vec + LR thất bại vì việc tính trung bình các vector từ để biểu diễn câu là một kỹ thuật quá đơn giản, làm mất đi các sắc thái quan trọng của văn bản.
+ TF-IDF + NaiveBayes hoạt động kém hơn vì LogisticRegression có khả năng gán trọng số quan trọng cho các đặc trưng (từ) có tính dự đoán cao, trong khi NaiveBayes thì không.

- Thành công (Neural Network):
+ TF-IDF + MLP cho kết quả tốt nhất (Accuracy 0.7755). 
+ Lý do là LogisticRegression là một mô hình tuyến tính, trong khi MLP (mạng nơ-ron) có thể học các mối quan hệ phi tuyến tính phức tạp giữa các đặc trưng TF-IDF. Điều này cho phép nó tìm ra các mẫu (patterns) tinh vi hơn trong dữ liệu mà mô hình tuyến tính bỏ lỡ.

# 4. Thách thức và Giải pháp
- Thử nghiệm Sklearn (Task 1): Accuracy là 0.0. Nguyên nhân do dữ liệu quá nhỏ (4 mẫu train), không đủ để huấn luyện.
- Mô hình MLP (Task 4): Thời gian huấn luyện MultilayerPerceptronClassifier lâu hơn đáng kể so với LogisticRegression. Đây là sự đánh đổi chấp nhận được để tăng độ chính xác.

# 5. Trích dẫn Tài liệu tham khảo

* Scikit-learn Documentation: [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
* Apache Spark MLlib Guide: [https://spark.apache.org/docs/latest/ml-guide.html](https://spark.apache.org/docs/latest/ml-guide.html)
* OpenAI. (2025). ChatGPT (GPT-4/5) [AI language model]. Truy cập từ https://chat.openai.com
