# Quy ước làm việc trong repo

- Đọc `TIEN_DO.md` trước khi tiếp tục và ghi lại bằng chứng sau mỗi mốc. `docs/research_plan.md` là kế hoạch; không nhầm việc đã viết code với việc đã chạy GPU.
- Mục tiêu là VQA tiếng Việt có lời giải thích. Giữ train/validation/test tách ảnh; không dùng test để chọn checkpoint, prompt hay siêu tham số.
- Dùng `configs/environments/*.yaml` cho đường dẫn local/Colab; không hard-code Google Drive vào `src/`. Logic dùng lại ở `src/`, notebook chỉ điều phối và hiển thị.
- Trước khi đồng bộ source: chạy `python scripts/check_project.py`; nếu sửa dữ liệu hoặc pipeline đầu vào thì chạy thêm `--full-data` ở máy đã có đủ ảnh. Ghi rõ lệnh, kết quả và nơi lưu run trong `TIEN_DO.md`.
- `sft-smoke` chỉ là kiểm tra kỹ thuật. Optimizer pilot M1 và rehearsal dừng chủ động bước 10 → resume 20 đã qua GPU T4; trạng thái full M1 và đánh giá chất lượng đọc từ `TIEN_DO.md`. Đừng đổi nhãn smoke/pilot thành kết quả M1 chính thức hoặc coi train xong là chất lượng đã đạt. Không tự khởi chạy full GPU train hoặc DPO chỉ vì pilot đã qua.
- Không sửa/ghi đè run, checkpoint, source release hoặc output notebook lịch sử. Tạo run/release mới và xác minh hash trước khi dùng. Không đưa dữ liệu ảnh, checkpoint, token hay credentials vào Git/source ZIP.
- Source release tạo bởi `scripts/package_source.py`. Chỉ ghi “đã đồng bộ Drive” sau khi upload và kiểm tra được file trên Drive; nếu chưa, ghi trạng thái local-only.
