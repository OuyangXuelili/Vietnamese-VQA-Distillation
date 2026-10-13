# Vietnamese VQA Explanation

Mã nguồn cho đề tài **Nghiên cứu cải thiện chất lượng lời giải thích trong
hỏi đáp hình ảnh tiếng Việt**. Đầu vào là ảnh và câu hỏi tiếng Việt; đầu ra
gồm đáp án và lời giải thích.

Đọc [kế hoạch](docs/research_plan.md) để thực hiện và [TIEN_DO.md](TIEN_DO.md)
để biết việc đã chạy. [Mô tả dự án](docs/project_description.md) giải thích
bài toán, dữ liệu và kết quả.

## Trạng thái — 14/10/2026

- M1 đã học một epoch, chọn checkpoint-3683 bằng loss trên 256 validation.
  Đánh giá đủ 1.407 validation: EM 70,93%, Token-F1 75,11%, ROUGE-L 44,29%,
  BERTScore 82,47%. [Bảng điểm và CI](reports/generated/m0_m1_validation_20261011/README.md).
- Đã xử lý hết 2.000 ứng viên: xem 1.966 ảnh, tách 34 câu theo phạm vi.
  Kiểm lại 201 cặp giữ, rút 15 cặp yếu; khóa **186 cặp / 186 ảnh train**.
  [Bộ dùng train](artifacts/preference_editor_m1_3683_2000_20261008/review_results_final_20261014.html).
  Có 78 rejected M1 nguyên văn và 108 negative có kiểm soát; không phải nhãn người.
- Trainer M2/M3 đã kiểm local; chưa train GPU. Demo replay đã có,
  live chưa nghiệm thu GPU. Chưa có kết quả test cuối.
- Bàn giao bản duyệt: `scripts/finalize_preference_review.py` audit/lưu snapshot;
  chỉ export/freeze khi nhãn được xác nhận và N khớp. Không tự train.

## Các nhánh

| Mã | Cấu hình |
|---|---|
| M0 | Qwen2.5-VL-3B-Instruct gốc |
| M1 | SFT trên ViVQA-X |
| M2 | Từ M1, SFT tiếp trên chosen |
| M3 | Từ cùng M1, DPO trên chosen/rejected |

M2/M3 dùng cùng câu/ảnh/chosen, seed, epoch và LR 5e-6. Sửa protocol trước
train để giảm nhiễu do khác LR; không coi là cùng chi phí tính toán.
Test giữ kín tới khi khóa lựa chọn; ROUGE/BERTScore không thay đánh giá theo ảnh.

## Tổ chức

```text
src/vqa_explanation/     dữ liệu, train, suy luận, đánh giá và demo
scripts/                CLI
configs/                thuật toán và đường dẫn từng môi trường
tests/                  kiểm thử
notebooks/              04_train_preferences.ipynb; lịch sử M1 ở archive
app/static/             giao diện demo
docs/, reports/         hướng dẫn, bảng và hình
data/, runs/, artifacts/ dữ liệu, checkpoint và bằng chứng ngoài Git
```

[Mục lục artifacts](artifacts/README.md) chỉ rõ bản duyệt, báo cáo pilot và
source release đang dùng. HTML/CSS/JavaScript của editor nằm riêng trong
`src/vqa_explanation/preference/templates/` và được nhúng khi xuất trang duyệt.
[Quy trình review](docs/code_review.md) mô tả các kiểm tra cần có trước khi đồng bộ.
Notebook `04_train_preferences.ipynb` có 7 cell, setup/kiểm đầu vào và gọi M2/M3. Bản M1 đã
chạy giữ trong `notebooks/archive/`, gồm cả output tải từ Colab.
[Mở M1 có output](https://colab.research.google.com/drive/13Abjl7kJyDKzbxyYsCetJ540OzdK2P_E).

## Local

Tại repo, dùng Python 3.10+ và môi trường riêng:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe scripts/check_project.py
```

Máy hiện tại có annotation và 28.180 ảnh ViVQA-X tham chiếu, không tải lại.
Profile `configs/environments/local_full.yaml` dùng dữ liệu đầy đủ;
`local.yaml` dùng smoke 500 ảnh. Local không mount Drive hoặc train Qwen CPU.
Checkout mới: xem [dữ liệu](docs/data_protocol.md) và
[local/cloud](docs/local_cloud.md). Kiểm trainer nhỏ cần thêm
`.[verification]`, không cài môi trường CUDA lên máy này.

## Việc tiếp theo

1. Khôi phục bộ đã khóa, kiểm đầu vào thật và token trên GPU.
2. Gate riêng rồi train M2/M3 từ M1 sạch: [hướng dẫn](docs/preference_training.md).
3. Validation/test, chấm theo ảnh: [giao thức](docs/experiment_protocol.md),
   [chấm mù](docs/human_evaluation.md).
4. Hoàn thiện [demo](docs/demo.md), phân tích và báo cáo.

[Harness](docs/harness.md) là kiểm code/provenance, không phải kết quả mô hình.
Notebook 00–08 và các run cũ giữ để tái lập, không chạy lại lần lượt.

## Lưu trữ

Git giữ source/config/test; Drive giữ dataset bundle, checkpoint, prediction
và log. Colab `/content` là ổ tạm. Source release có checksum, không chứa
ảnh, trọng số hoặc credentials.

CI chạy kiểm CPU trên Python 3.10 và 3.11, có dependency cho trainer nhỏ,
API và Node để kiểm lưu nháp. `pytest -m "not historical" --fail-on-skip`
không cho bỏ qua kiểm tra source vì thiếu dependency. Nhóm `historical` cần
artifact run ngoài Git và vẫn được chạy khi kiểm toàn bộ trên máy này.
