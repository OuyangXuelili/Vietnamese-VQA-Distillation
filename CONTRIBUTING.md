# Contributing

Cảm ơn bạn đã quan tâm đến Vietnamese VQA Distillation.

## Thiết lập môi trường

```bash
python -m venv .venv
pip install -r requirements-app.txt
python app.py --check
```

Nếu thay đổi pipeline huấn luyện, cài bộ thư viện đầy đủ bằng
`pip install -r requirements.txt`.

## Quy ước

- Tạo branch riêng cho mỗi thay đổi.
- Không commit model weights, dataset cache, token hoặc file `.env`.
- Chạy `python -m compileall -q app.py src scripts` trước khi tạo pull request.
- Mô tả rõ thay đổi ảnh hưởng đến Teacher, Student hay cả hai.
- Dùng UTF-8 và giữ nguyên dấu tiếng Việt trong tài liệu.

## Báo lỗi

Khi tạo issue, vui lòng cung cấp hệ điều hành, phiên bản Python, GPU/VRAM,
model đã chọn và traceback đầy đủ. Không đăng Hugging Face token hoặc thông tin
đăng nhập vào issue.
