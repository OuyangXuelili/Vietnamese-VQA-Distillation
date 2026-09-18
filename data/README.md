# Data policy

Không commit ảnh COCO, annotation đầy đủ, model-generated outputs hoặc preference data chưa được kiểm tra vào repository.

Repo chỉ giữ:

- schema và script xử lý;
- manifest không chứa dữ liệu nhạy cảm nếu được phép công bố;
- một vài bản ghi giả lập trong `data/sample/` để kiểm thử.

Dữ liệu thật nằm trong Google Drive hoặc storage của GPU cloud. Mỗi phiên bản dữ liệu cần có checksum, ngày tạo, script tạo và nguồn gốc.

## Bố trí hiện tại

```text
data/raw/vivqax/annotations/   ba file JSON gốc từ ViVQA-X
data/processed/vivqax/        JSONL chuẩn hóa và stats.json
data/processed/vivqax/manifests/
                               danh sách 28.180 ảnh COCO cần lấy
```

Tái tạo toàn bộ dữ liệu dẫn xuất bằng:

```bash
python scripts/prepare_vivqax.py
```

Google Drive dùng cấu trúc tương ứng dưới `Vietnamese_VQA_NLE/datasets/`:
`annotations`, `images`, `manifests`, `processed`, `preference` và `audit`.
