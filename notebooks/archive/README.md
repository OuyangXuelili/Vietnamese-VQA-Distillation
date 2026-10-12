# Notebook lịch sử

Giữ nguyên nội dung/output các kiểm thử trước M1. Không cần chạy lại chúng.

| Notebook | Vai trò |
| --- | --- |
| 00 | Chuẩn bị Colab và dữ liệu smoke |
| 01 | Suy luận Qwen gốc trên 20 câu |
| 02 | SFT 8 bước và nạp lại adapter |
| 04 | So sánh M0/M1 trên 20 câu |
| 05 | Kiểm tra dữ liệu đầy đủ |
| 06 | Đo token trên 512 mẫu |
| 07 | Một lượt forward/backward, không cập nhật optimizer |
| 08 | Kiểm tra optimizer và dừng 10 → resume 20 |

Không dùng kết quả smoke thay kết quả full M1. Notebook 05–07 có thể chọn
release mới nhất khi chạy lại; 08 ghim release rehearsal cũ. Không Run all
để xem output lịch sử. Đường dẫn cloud trong notebook được giữ nguyên.

Điểm vào hiện hành: [03](../03_portable_runner.ipynb).
Trạng thái: [TIEN_DO.md](../../TIEN_DO.md).
