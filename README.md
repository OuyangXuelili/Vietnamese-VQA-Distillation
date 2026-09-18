# Vietnamese VQA Explanation

Mã nguồn cho đề tài **Nghiên cứu cải thiện chất lượng lời giải thích trong hỏi đáp hình ảnh tiếng Việt**.

Hệ thống nhận ảnh và câu hỏi tiếng Việt, sau đó sinh đáp án kèm lời giải thích. Thực nghiệm chính so sánh mô hình pretrained, SFT, SFT tiếp trên dữ liệu tốt và DPO trên các cặp lời giải thích được ưu tiên.

## Câu hỏi nghiên cứu

DPO có giúp lời giải thích phù hợp với nội dung ảnh hơn SFT tiếp trên cùng nguồn dữ liệu tốt hay không, và độ chính xác đáp án thay đổi như thế nào?

## Nguyên tắc thiết kế

- Logic dùng lại nằm trong `src/`; notebook chỉ gọi CLI và hiển thị kết quả.
- Mỗi thí nghiệm có cấu hình riêng, seed, Git commit và phiên bản dữ liệu.
- Test được giữ riêng; không dùng test để chọn prompt, checkpoint hoặc siêu tham số.
- Ảnh, model cache, checkpoint và log lớn nằm ngoài Git.
- Dữ liệu ưu tiên giữ nguồn gốc, loại lỗi và trạng thái kiểm tra.

## Cấu trúc

```text
configs/                 cấu hình dữ liệu, model, train và từng thí nghiệm
src/vqa_explanation/     package chính
scripts/                 các entry point chạy từ command line
tests/                   unit test và integration smoke test
notebooks/               khám phá dữ liệu và điều phối Colab
app/                     Gradio demo
docs/                    giao thức dữ liệu, thí nghiệm và tái lập
data/sample/             dữ liệu nhỏ phục vụ test
reports/                 bảng và hình dùng trong báo cáo
```

## Bốn cấu hình chính

| Mã | Cấu hình | Vai trò |
|---|---|---|
| M0 | Pretrained | Mốc trước thích nghi |
| M1 | SFT trên ViVQA-X | Baseline chính |
| M2 | M1 + SFT trên chosen | Kiểm soát lợi ích của dữ liệu tốt |
| M3 | M1 + DPO trên chosen/rejected | Kiểm tra lợi ích của học ưu tiên |

## Thiết lập ban đầu

```bash
python -m venv .venv
pip install -e ".[dev]"
pytest
```

Kiểm tra file ViVQA-X đã chuẩn hóa:

```bash
python scripts/prepare_vivqax.py
```

Lệnh trên đọc ba annotation gốc trong `data/raw/vivqax/annotations/`, tạo JSONL
chuẩn hóa, thống kê và danh sách chính xác các ảnh COCO cần tải trong
`data/processed/vivqax/`.

Tải tập smoke test cố định gồm 400 ảnh train, 50 validation và 50 test:

```bash
python scripts/download_coco_smoke.py
```

Script chỉ chọn các ảnh thuộc riêng một split, kiểm tra định dạng JPEG và có thể
chạy lại an toàn vì ảnh hợp lệ đã tồn tại sẽ được bỏ qua.

Kiểm tra dữ liệu ưu tiên:

```bash
python scripts/validate_preferences.py \
  --input data/sample/preference_sample.jsonl
```

## Trạng thái

Schema thật của ViVQA-X đã được xác nhận và pipeline chuẩn hóa hiện xử lý 32.886
mẫu hỏi–đáp, 41.817 lời giải thích và 28.180 ảnh độc nhất. Bước kế tiếp là tải
một tập ảnh nhỏ để smoke test backbone trên GPU cloud trước khi lấy toàn bộ ảnh.

## Lưu trữ cloud

- Repo: code, config, test và tài liệu.
- Google Drive: annotation, manifest, preference data, checkpoint, prediction và log.
- `/content/vqa_work`: ảnh COCO đã giải nén và cache tạm trong phiên Colab.

Không đưa token, ảnh COCO, checkpoint hoặc model cache lên GitHub.
