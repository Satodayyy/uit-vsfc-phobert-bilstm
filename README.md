# Phân loại cảm xúc phản hồi sinh viên tiếng Việt trên UIT-VSFC

**Đồ án cuối kỳ môn Học sâu · DL11 · Nhóm 08 · Trường Đại học Sài Gòn**

Từ một phản hồi bằng tiếng Việt, mô hình dự đoán một trong ba nhãn **tiêu cực**, **trung tính** hoặc **tích cực**. Nhóm so sánh bốn mô hình: TF-IDF + hồi quy logistic, BiLSTM, BiLSTM + attention + fastText và PhoBERT-base-v2.

Đồ án khảo sát ba câu hỏi: lượng dữ liệu gán nhãn ảnh hưởng thế nào đến kết quả, tách từ có giúp cải thiện phân loại hay không và cần tinh chỉnh bao nhiêu tầng của PhoBERT.

## Dữ liệu

Nhóm giữ cách chia tập gốc của UIT-VSFC. Khi lọc các câu trùng giữa các tập, nhóm ưu tiên giữ câu ở tập test.

| Tập dữ liệu sau lọc | Số câu | Tiêu cực | Trung tính | Tích cực |
|---|---:|---:|---:|---:|
| Huấn luyện | 11.346 | 5.322 | 455 | 5.569 |
| Validation | 1.576 | 705 | 73 | 798 |
| Test | 3.166 | 1.409 | 167 | 1.590 |

[Bảng thống kê dữ liệu](ket_qua/EDA/theo_tap.csv) · [Mã tiền xử lý](src/du_lieu.py)

## Thành viên

| Thành viên | MSSV | Phần phụ trách |
|---|---|---|
| Nguyễn Trí Sự | 3123580041 | Dữ liệu, tiền xử lý, PhoBERT và tích hợp bản nộp. |
| Huỳnh Nhật Thành | 3123580044 | Baseline, BiLSTM và thí nghiệm thay đổi lượng dữ liệu huấn luyện. |
| Nguyễn Quốc Thuận | 3123580048 | Đánh giá, phân tích lỗi, demo, báo cáo và slide. |

## Baseline và BiLSTM

**TF-IDF + hồi quy logistic (PyTorch).** Nhóm dùng `TfidfVectorizer` để biểu diễn câu bằng đặc trưng unigram và bigram. Bộ phân loại được triển khai bằng `torch.nn.Linear` và huấn luyện với `torch.optim.LBFGS`, điều chuẩn L2 và trọng số lớp.

Nhóm thử `C` bằng `0.1`, `1.0` và `10.0` trên ba seed 42, 43, 44. Cấu hình `C = 10` đạt **Macro-F1 validation trung bình 0,7965** trên 1.576 câu và được chọn để chấm test. [Mã baseline](src/baseline.py) · [Kết quả validation](ket_qua/baseline.json)

**BiLSTM.** Nhóm huấn luyện hai biến thể: BiLSTM cơ bản và BiLSTM kết hợp attention với vector từ tiếng Việt fastText. [Mã BiLSTM](src/bilstm.py) · [Kết quả chọn cấu hình](ket_qua/tuning/)


## PhoBERT

Nhóm tinh chỉnh toàn bộ encoder của **PhoBERT-base-v2** để phân loại phản hồi sinh viên thành ba nhãn cảm xúc. Mô hình được huấn luyện trên GPU với 11.346 câu đã tách từ, seed 42, batch size 32 và learning rate `2e-5`.

Sau 3 epoch, nhóm chọn **checkpoint ở epoch 3** vì đạt **Macro-F1 cao nhất là 0,8645 trên 1.576 câu validation**.

[Mã huấn luyện](src/pho_toan_bo.py) · [Nhật ký 3 epoch](mo_hinh/pho_42/run.json) · [Checkpoint PhoBERT](mo_hinh/pho_42/) · [Nguồn mô hình và giấy phép](NGUON.md)
