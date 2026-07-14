# Lộ trình nâng cấp

Roadmap này ưu tiên các thay đổi có thể đo lường được và không yêu cầu train lại
trừ khi chất lượng model hiện tại đã chạm giới hạn của dữ liệu fine-tune.

## Giai đoạn 1 — Đo lường và ổn định

- Xây dựng tập kiểm thử cố định, tách khỏi dữ liệu training.
- Đo Teacher và Student bằng Exact Match, token-level F1, ROUGE-L và BERTScore.
- Ghi nhận thời gian nạp model, thời gian sinh, VRAM đỉnh và số token/giây.
- Kiểm tra ảnh trùng, câu hỏi trùng, câu trả lời rỗng và phân bố độ dài trong dataset.
- Thêm test cho data loader, model registry và các lệnh CLI.

**Mục tiêu:** biết chính xác Student nhẹ hơn bao nhiêu và giảm chất lượng bao nhiêu
so với Teacher trước khi tối ưu tiếp.

## Giai đoạn 2 — Cải thiện chất lượng câu trả lời

- Đồng bộ prompt lúc inference với định dạng prompt đã dùng khi fine-tune.
- Bổ sung dữ liệu mô tả dài nếu muốn model sinh đoạn 3–5 câu ổn định.
- Làm sạch nhãn sai, câu trả lời mơ hồ và mẫu có thông tin không xuất hiện trong ảnh.
- Thử nghiệm LoRA rank, learning rate, số epoch và tỉ lệ dữ liệu chi tiết/ngắn.
- Đánh giá hallucination: model nói thêm vật thể hoặc thuộc tính không có trong ảnh.

**Lưu ý:** model đã fine-tune chủ yếu bằng câu trả lời VQA ngắn sẽ không tự nhiên trở
thành model caption dài chỉ bằng cách tăng `max_new_tokens`; khi đó cần bổ sung dữ liệu
instruction/caption và fine-tune tiếp.

## Giai đoạn 3 — Tăng tốc suy luận

- Resize ảnh về giới hạn pixel phù hợp trước khi đưa vào processor.
- Thử Student 2B ở FP16, BF16 và 4-bit trên cùng GPU để chọn cấu hình nhanh nhất.
- Bật Flash Attention khi GPU và phiên bản thư viện hỗ trợ.
- Thêm tùy chọn preload model khi khởi động để lần bấm đầu tiên không phải chờ nạp.
- Hiển thị riêng thời gian nạp model, thời gian sinh và token/giây.
- Cho phép unload model để giải phóng VRAM khi chuyển Teacher ↔ Student.

## Giai đoạn 4 — Nâng cấp sản phẩm demo

- So sánh Teacher và Student cạnh nhau trên cùng ảnh/câu hỏi.
- Lưu lịch sử câu hỏi, câu trả lời, model và thời gian suy luận trong phiên.
- Xuất kết quả ra JSON/CSV.
- Hỗ trợ xử lý nhiều ảnh và nhiều câu hỏi theo batch.
- Thêm REST API bằng FastAPI cho ứng dụng khác gọi model.
- Thêm bộ ảnh ví dụ hợp lệ về giấy phép ngay trong giao diện.

## Giai đoạn 5 — Triển khai

- Tạo Dockerfile cho CUDA và kiểm tra health endpoint.
- Đưa Student demo lên Hugging Face Spaces hoặc máy chủ GPU.
- Thêm logging có cấu trúc, giới hạn hàng đợi và xử lý timeout.
- Gắn version cho code, dataset và adapter để kết quả có thể tái lập.
- Tự động chạy smoke test và kiểm tra file lớn bằng GitHub Actions.

## Thứ tự khuyến nghị

1. Benchmark Teacher/Student và kiểm tra dataset.
2. Tối ưu tốc độ Student 2B mà không train lại.
3. Thêm chế độ so sánh hai model và lịch sử trong giao diện.
4. Chỉ fine-tune tiếp khi benchmark chứng minh prompt hiện tại chưa đủ cải thiện chất lượng.
5. Đóng gói Docker hoặc Hugging Face Space sau khi chất lượng và latency ổn định.
