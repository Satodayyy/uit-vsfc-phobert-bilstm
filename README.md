# Phân loại cảm xúc trong phản hồi của sinh viên bằng BiLSTM và PhoBERT

**Đồ án cuối kỳ môn Học sâu | Nhóm 07 | Trường Đại học Sài Gòn**

## Giới thiệu

Đồ án sử dụng bộ dữ liệu UIT-VSFC để phân loại phản hồi bằng tiếng Việt của sinh viên vào một trong ba nhóm cảm xúc tiêu cực, trung tính hoặc tích cực. Nhóm tập trung so sánh BiLSTM và PhoBERT khi hai mô hình cùng học từ một tập phản hồi. Phương pháp TF-IDF kết hợp hồi quy logistic được dùng làm mốc đối chứng.

Qua các thí nghiệm, nhóm khảo sát ảnh hưởng của việc tách từ, lượng dữ liệu huấn luyện và số tầng PhoBERT được tinh chỉnh. Do phản hồi trung tính chiếm tỷ lệ nhỏ, nhóm sử dụng **Macro-F1** làm chỉ số chính để mỗi lớp có trọng số ngang nhau khi đánh giá. Nhóm cũng đọc các câu dự đoán sai để tìm hiểu những trường hợp mô hình dễ nhầm lẫn.

Kho mã này dùng để lưu mã nguồn, notebook Google Colab, nhật ký và kết quả của từng lần thử nghiệm, giúp người đọc theo dõi quá trình thực hiện và tái lập kết quả của nhóm.
