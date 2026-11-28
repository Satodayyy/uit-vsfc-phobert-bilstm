# Nguồn dữ liệu, mô hình và công cụ

| Tài nguyên | Sử dụng trong đồ án | Nguồn |
|---|---|---|
| UIT-VSFC, Nguyen và cộng sự (KSE 2018) | Ba split và nhãn cảm xúc. Bản nộp lưu ID, nhãn, dự đoán và diễn giải các ví dụ. | [Bài báo](https://doi.org/10.1109/KSE.2018.8573337) · [Bản dữ liệu cố định](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/tree/2d76906157232f87e9883fa37ebfc91ffdee1f83) |
| PhoBERT, Nguyen & Nguyen (Findings of EMNLP 2020), VinAI | PhoBERT-base-v2, BPE, cấu hình của encoder. Trọng số trong mo_hinh/pho_42 là bản đã tinh chỉnh cho ba nhãn UIT-VSFC, lưu BF16 và chia nhỏ. Giữ thông báo AGPL-3.0 của nguồn; mã huấn luyện và ghép checkpoint đi kèm. | [Bài báo](https://aclanthology.org/2020.findings-emnlp.92/) · [Bản mô hình cố định](https://huggingface.co/vinai/phobert-base-v2/tree/86cd7fd4c148980922ac11a2cf5e257f2ba639e1) · [Giấy phép](nguon_ben_ngoai/PhoBERT-AGPL-3.0.txt) |
| fastText tiếng Việt, Grave và cộng sự (LREC 2018) | Các vector 300 chiều khớp từ vựng train được dùng để khởi tạo BiLSTM + attention. Embedding được cập nhật khi huấn luyện. Nguồn phát hành theo CC BY-SA 3.0. | [Vector và giấy phép gốc](https://fasttext.cc/docs/en/crawl-vectors.html) · [Bản giấy phép](nguon_ben_ngoai/fastText-CC-BY-SA-3.0.txt) |
| Underthesea | Tách từ trên CPU, dùng cùng phiên bản cho thí nghiệm và demo. | [Dự án](https://github.com/undertheseanlp/underthesea) |
| PyTorch, Transformers, scikit-learn, Gradio | Huấn luyện, đánh giá và giao diện dự đoán. Phiên bản có trong requirements.txt và requirements-demo.txt. | [PyTorch](https://pytorch.org/) · [Transformers](https://github.com/huggingface/transformers) · [scikit-learn](https://scikit-learn.org/) · [Gradio](https://www.gradio.app/) |

Kiến trúc BiLSTM dựa trên Hochreiter & Schmidhuber (1997) và Schuster & Paliwal (1997). Báo cáo trình bày trích dẫn chi tiết. ChatGPT hỗ trợ triển khai, chạy thực nghiệm, kiểm tra tệp và soạn tài liệu. Kết quả được đối chiếu bằng dự đoán theo ID và nhật ký máy. Bảng phân công mô tả trách nhiệm của ba thành viên khi tiếp nhận, kiểm tra và bàn giao.

Trong baseline, scikit-learn tạo đặc trưng TF-IDF. PyTorch huấn luyện bộ phân loại hồi quy logistic với torch.nn.Linear và torch.optim.LBFGS.
