# Reproduction guide

## Chuẩn bị annotation và manifest

```bash
python scripts/prepare_vivqax.py
python scripts/download_coco_smoke.py
```

Tập smoke được cố định bằng seed `20260918`, gồm 400 ảnh train, 50 validation
và 50 test. Các ảnh trùng giữa validation và test không được chọn. Kết quả chuẩn:

- 500 ảnh JPEG hợp lệ;
- 588 cặp hỏi–đáp;
- 830 lời giải thích;
- không có ảnh trùng giữa ba split;
- SHA-256 của `coco_smoke_500.zip`:
  `4e0fce747dee9a1beaf36e2617deae6582115020d7c219f315525e329a4024f9`.

## Thứ tự thực nghiệm

1. Tạo môi trường và cài package.
2. Chạy notebook Colab bootstrap và kiểm tra đủ 500 ảnh.
3. Chạy inference M0 trên tập smoke.
4. Chạy SFT vài step trên tập smoke và xác nhận loss/checkpoint.
5. Chỉ sau khi smoke test đạt mới lấy đủ 28.180 ảnh.
6. Huấn luyện M1, tạo preference data, huấn luyện M2/M3.
7. Đóng băng checkpoint rồi đánh giá test.
