# Notebook đang sử dụng

[03_portable_runner.ipynb](03_portable_runner.ipynb) điều phối M1 trên Colab;
logic dùng lại nằm trong `src/` và `scripts/`. M1 đã train đủ một epoch ngày
07/10/2026. Trạng thái và bằng chứng mới nhất ở [TIEN_DO.md](../TIEN_DO.md).

- Ô 1 đọc run, heartbeat, checkpoint và log; không sửa dữ liệu đã lưu.
- Ô 2 khôi phục môi trường theo source ghim 28/09; ô 3 chuẩn bị ảnh.
  Nếu ô 2 yêu cầu restart Python, chạy lại ô 1–2 rồi tiếp tục.
- Ô 4 kiểm checkpoint-3683 và adapter được Trainer chọn trên CPU,
  lưu báo cáo nghiệm thu riêng; không train hoặc sửa checkpoint.
- Ô 5 đo riêng loss checkpoint-3683 bằng source/môi trường M1 đã ghim.
  Đưa hai script
  `evaluate_m1_epoch.py` và `resume_pinned_m1.py` từ local vào `/content`
  bằng ngăn Tệp lần đầu; ô 5 kiểm hash và giữ bản công cụ riêng trên Drive.
- Ô 6 khôi phục source suy luận 02/10 và đúng môi trường của cache M0,
  khác môi trường đo loss. Sau restart ở ô 6, chạy lại ô 1 và 6–7;
  không chạy ô 2 vì nó cài lại môi trường train.
  Cache phải khớp hash của bản M0 đã xuất đủ 1.407 câu. Nếu hai vị trí
  cũ không đọc được, tải `m0.jsonl` và `m0.jsonl.manifest.json` từ
  `artifacts/validation_review_20261003/checkpoint_600` vào ngăn Tệp.
  Ô 6 chỉ khôi phục bản nguyên vẹn vào `evaluation_runs/m0_full_validation_cache_v1`,
  không ghi đè cache cũ hoặc sinh lại M0.
- Ô 7 đọc lại báo cáo M1 cuối đã hoàn tất, rồi đánh giá checkpoint-1200
  trên đủ 1.407 validation theo cùng giao thức. Dùng chung runner/cache M0;
  báo cáo đã có chỉ đọc lại, prediction dang dở được resume sau kiểm tra.
  Mốc 1200 phải khớp hash trong run đã dừng đúng lịch; không train lại.
- Ô 9 ghim M1-3683 bằng evidence epoch/loss/full validation; chỉ kiểm metadata,
  không cần GPU. Source mới đọc từ `source/preference_stage_v2.json`; không
  thay source hay run lịch sử. Annotation validation được lưu cạnh biên bản.
- Ô 10 sinh 2.000 ứng viên train từ pool đã cố định, lưu vào Drive. Cần GPU
  và tự khôi phục thư viện suy luận nếu thiếu. Chạy 1 → 9 → 10; nếu yêu cầu
  restart hoặc ngắt phiên thì chạy lại ba ô này để resume đúng job,
  không tạo pool hoặc notebook khác. Không chạy M2/M3 khi chưa có accepted.
- Mục lịch sử bắt đầu ở ô 11. Ô 12–19 giữ kiểm token, lệnh train và các
  đánh giá cũ để truy vết, không thuộc luồng hiện tại và không chạy lại.

Không bấm Chạy tất cả. Giữ nguyên output lịch sử và source train đã ghim.
Các notebook kiểm thử cũ nằm trong [archive/](archive/README.md); không cần
chạy lại hoặc tạo notebook mới cho mỗi phiên.

Đã đo riêng checkpoint-3683 trên 256 validation cố định và chọn mốc này.
Đã đánh giá M1 cuối đủ 1.407 validation: EM 70,93%, Token-F1 75,11%,
ROUGE-L 44,29%; mốc 1200 cũng đã đánh giá xong. Rà 100 ảnh là chẩn đoán
song song, không bắt đủ số lượng này mới được ghim M1. Xem
[hướng dẫn SFT](../docs/sft_full.md). Loss `final_validation`
hiện tại thuộc adapter được Trainer chọn, không phải checkpoint cuối.

[Mở Colab](https://colab.research.google.com/drive/1emxS1s8mZHYQxVqR4e4tA8j8G_xrzhl0)
· [Kế hoạch thực nghiệm](../docs/research_plan.md)
