# Tiến độ dự án

Cập nhật: 12/10/2026.

## Trạng thái hiện hành

| Phần | Trạng thái và bằng chứng |
|---|---|
| Dữ liệu | ViVQA-X `image_disjoint_v1`, 29.459 train / 1.407 validation / 1.968 test; 28.180 ảnh đã kiểm local và bundle cloud |
| M0/M1 | M1 một epoch, 3.683 update; chọn checkpoint-3683 từ loss 256 validation; đã xuất đủ validation M0/M1 |
| Chỉ số M0 → M1 | EM 41,01 → 70,93; F1 52,41 → 75,11; ROUGE-L 27,05 → 44,29; BERTScore 75,82 → 82,47; chỉ số nhân 100 |
| Cặp ưu tiên | 2.000 ứng viên train-only, pilot 100 ảnh; 9 nháp sửa tự nhiên và 4 nháp synthetic sau kiểm lại; 0 accepted |
| M2/M3 | Trainer và resume đã kiểm local bằng mô hình nhỏ; chưa train GPU |
| Đánh giá cuối | Đã cố định mẫu 200 ảnh test; chưa có prediction test bốn nhánh hoặc nhãn người |
| Demo | Replay 12 ví dụ đã kiểm API và Chrome; live chưa nghiệm thu GPU |

Bảng validation và CI ở
[reports/generated/m0_m1_validation_20261011](reports/generated/m0_m1_validation_20261011/README.md).
Hướng nghiên cứu và các việc tiếp theo ở [docs/research_plan.md](docs/research_plan.md).

## 12/10 — sửa import trong CI trên Linux

- Actions phát hiện `ModuleNotFoundError: scripts` khi gọi executable pytest.
  CI nay dùng `python -m pytest`; cấu hình pytest thêm gốc repo vào pythonpath
  để các test import script có cùng cách resolve khi dùng executable hoặc module.
- `check_project.py` trên source hiện hành qua Ruff và toàn bộ 831 test.
  Kiểm source-only local qua với fail-on-skip; sáu kiểm lịch sử được deselect.
- Actions của commit `dec65fb` thành công trên cả Python 3.10 và 3.11:
  [CI 38243446356](https://github.com/OuyangXuelili/Vietnamese-VQA-Distillation/actions/runs/38243446356).
- Release cuối sau sửa CI: `source_20261012_113653_701242`, 231 file.
  Đã upload ZIP/checksum vào Drive `source/releases`, tải lại ZIP và hash khớp
  `9a4c37bee29caf9fc4b60905a1e74d116b0eb4819166e9e20c6692a07bc946c3`.
  Code/config trong release khớp commit CI `dec65fb`; nhật ký cập nhật riêng sau đóng ZIP.
- Giữ marker historical và fail-on-skip của CI source. Không đổi code train,
  dữ liệu, checkpoint hoặc giao thức. Log failure và regression ở
  `artifacts/code_cleanup_20261012`.

## 12/10 — chốt đối chứng và kiểm cặp nháp

- Protocol mới `vqa_nle_matched_lr_v2`: M2 chính dùng
  `sft_chosen_matched_lr.yaml`, cùng LR 5e-6 với M3. Sửa trước hai run GPU;
  giữ bản 2e-5 chưa train làm lịch sử, không thêm nhánh mặc định.
- CLI kiểm cả hai cấu hình: LR, N, seed, epoch, batch, pixels, dtype,
  dropout và token guard. Hash study/hai config nằm trong data plan và
  resume identity. Config cũ bị từ chối trước đọc dữ liệu/tải model.
- Mở lại cả 15 ảnh có nháp: bỏ 149381000 (chosen sai về lối đi, đối lập
  chưa chắc) và 314983003 (lồng/giỏ chủ yếu khác thuật ngữ). Còn 13 nháp:
  9 rejected nguyên văn M1, 4 synthetic; 0 accepted. Kiểm lại 9 raw text
  với candidates, khớp. 85 nhận xét còn lại kế thừa, không chấm độc lập mới.
- Chosen của 8/9 sửa tự nhiên dài hơn rejected: trung bình 12,67 so 8,56
  từ phân cách bằng khoảng trắng, không phải token Qwen. Ghi thiên lệch này
  trong `quality_check.json`; không lấy độ dài làm bằng chứng bám ảnh.
- `check_project.py --full-data` với `PYTEST_ADDOPTS=--fail-on-skip` qua:
  Ruff, 831 test, 26 warning fixture PEFT, audit 32.834 câu/28.180 ảnh và
  preflight đủ 29.459 train. Không ảnh thiếu/hỏng, không trùng split.
  Runs `audit_20261012_111242_833454`, `preflight_20261012_111417_286914`;
  log ở `artifacts/preference_ai_triage_m1_3683_pilot_20261012_qc`.
- Canonical, notes pending, candidates, pilot và mẫu test 200 giữ hash cũ.
  Không đổi checkpoint M1, notebook hoặc chạy test/GPU M2/M3.
- Trang mới `pilot_review_qc_20261012.html` đã kiểm trên Chrome: ảnh,
  cảnh báo và 100 pending; không còn nháp của câu bị rút. Hỗ trợ AI chỉ
  hiện khi yêu cầu. Server local 8765 được khởi động lại, chỉ bind 127.0.0.1.
- Báo cáo pilot vẫn `in_progress`, `training_gate_open=false`. Cần chốt
  quyết định pilot và N cặp hợp lệ; 1.000 hiện chỉ là dự toán, không quota.
- Source `source_20261012_111657_299458` đã ở Drive releases, 230 file;
  tải lại ZIP khớp SHA-256 `c0040034978a15f6055fedc17e8a63ac5d44e0efb17a10b48b30b8a49abdf187`.
  Năm tài liệu cập nhật phiên bản cũ: research_plan v10,
  preference_training v6, preference_annotation v5, experiment_protocol v3,
  project_description v4. Word không thay đổi.
- `preference_local/pilot_qc_20261012.zip` và checksum đã tải lên; ZIP có
  12 entry, không ảnh/weights. Trong thao tác kiểm tải lại, ZIP bị chuyển
  vào Thùng rác; đã khôi phục đúng file về preference_local. Dùng mục
  Tải xuống bên trong menu tệp thay vì nút toolbar bị menu che; tải lại
  khớp `0f9155ebd8b1a62b34610f4eabfd40d9678404a69cab73dfeb3c9446f67eba8b`.
  Không mất dữ liệu hoặc đụng tệp khác trong Thùng rác. Bản ZIP source
  giữ nhật ký trước upload; tổng kết này được cập nhật riêng vào file cũ.

## 12/10 — sửa lỗi review và đối chiếu đồ án sinh viên

- CI tách sáu kiểm thử cần hồ sơ run vào marker `historical`; job source dùng
  `-m "not historical" --fail-on-skip`. Vẫn thất bại nếu kiểm source thiếu
  dependency. Mô phỏng không có artifacts đã trả exit 0 thay vì 1.
- Demo bắt asyncio.TimeoutError để tương thích Python 3.10; regression test
  kiểm exception riêng trả HTTP 504. Không thay luồng sinh hoặc khóa GPU.
- Trang chấm mù mới tự lưu lúc nhập, khôi phục theo nguồn/rater, cảnh báo
  rời trang và lỗi lưu. Nháp dở được giữ; nhãn thiếu không xuất được.
  Kiểm JavaScript nhập/reload, nguồn sai, lỗi storage và JSONL đã qua.
- `check_project.py` qua Ruff và toàn bộ 817 test, 26 warning fixture PEFT;
  40 test liên quan qua. Log ở `artifacts/review_fixes_20261012`.
  Không chạy full-data vì không đổi input/collator/trainer. Hash canonical,
  notes pending và mẫu test 200 ảnh vẫn khớp trước lượt sửa.
- Chrome chặn URL file của fixture; chưa nghiệm thu giao diện chấm mới trên
  Chrome. Không đổi trình duyệt/security setting hoặc HTML lịch sử để thử.
- Đối chiếu hai project Stanford CS231n 2024: BLIP so SFT/DPO/ORPO và
  IDEFICS2 so base/one-stage/two-stage. Ghi nguồn và khác biệt trong
  `docs/related_repositories.md`; không coi đồ án môn là chuẩn tốt nghiệp.
  Giữ một backbone Qwen/bốn cấu hình, ưu tiên đối chứng LR; Vintern zero-shot
  chỉ là baseline ngoài tùy ngân sách, chưa viết runner hoặc chạy GPU.
- Giải nén source ZIP vào checkout sạch, chạy đúng lệnh CI trên Python 3.13.14:
  811 passed, 6 historical deselected, không skip; log `source_only_ci.log`.
  Chưa chạy GitHub Actions/Linux hoặc trực tiếp Python 3.10/3.11.
- Source `source_20261012_100818_659104` đã ở Drive releases: 230 file;
  tải lại ZIP, SHA-256 khớp `d001595839cd3a11e30b5bcc1c3daf879e05b13fdd9809d8ba9c99ae8d0f1445`.
  Documents đã nhận research_plan v9, human_evaluation v2,
  preference_training v5 và related_repositories mới. Word không đổi.
  Không đổi LR, epoch, nhãn, checkpoint hay notebook. ZIP snapshot giữ nguyên;
  nhật ký hoàn tất được cập nhật riêng vào file cũ trên Drive.

## 12/10 — rà lại thiết kế và khả năng tái lập

- Ruff và `pytest --fail-on-skip` qua trên Python 3.13.14: 807 passed,
  26 warning fixture PEFT. Không chạy GPU hoặc đổi checkpoint/dữ liệu.
- Mô phỏng checkout không có artifacts: sáu kiểm thử lịch sử skip,
  `--fail-on-skip` trả exit 1. CI cần tách kiểm source khỏi kiểm hồ sơ lịch sử;
  không khắc phục bằng cách đưa dataset/checkpoint vào Git.
- Demo bắt builtin TimeoutError, chưa tương thích asyncio.TimeoutError của
  Python 3.10. Mô phỏng exception này trả HTTP 503 thay vì 504.
- Trang chấm mù giữ nhãn trong RAM, chưa autosave/cảnh báo rời trang.
  Cần sửa trước khi giao 800 output/người chấm; trang duyệt preference đã có autosave.
- Thiết kế cần chốt đối chứng LR trước train/test. Bộ cặp vẫn 0 accepted;
  chưa có căn cứ chốt N=1.000 hoặc kết luận DPO cải thiện giải thích theo ảnh.
- Các mục trên là phát hiện review, chưa sửa code hoặc thay giao thức.
  Giữ M1, split và pool 2.000; ưu tiên bộ cặp/đánh giá, không thêm backbone,
  notebook hay lượt smoke không gắn với thay đổi cần kiểm. Nhật ký này local-only.

## Các đầu vào đã ghim

- Backbone: Qwen2.5-VL-3B-Instruct,
  revision `66285546d2b821cf421d4f5eb2576359d3770cd3`.
- M1 run cuối: `runs/m1/sft-full_20261007_023531_870900`.
- Selection: `artifacts/colab_history/20261007/m1_selection_frozen_v2/selection.json`.
  Sáu đường dẫn evidence giữ layout Drive gốc, chưa resolve từ bản archive local.
- Bundle review: `artifacts/colab_history/20261008/preference_candidates_completed/review_2000.zip`,
  SHA-256 `365f080096f964a4a1d670faa7192a62697f45fb92f0480cdd555c7319b8af45`.
- Trang pilot: `artifacts/preference_editor_m1_3683_2000_20261008/pilot_review_qc_20261012.html`.
- Nhận xét AI đã kiểm lại: `artifacts/preference_ai_triage_m1_3683_pilot_20261012_qc`.
- Báo cáo quy mô hiện hành: `artifacts/preference_ai_triage_m1_3683_pilot_20261012_qc/pilot_report`.

M1 giữ source/môi trường của run lịch sử. M2/M3 cùng bắt đầu từ M1-3683,
cùng bundle đã duyệt; chốt N trước train. Mẫu test và các nhãn pending giữ nguyên.

## 12/10 — sửa code và biên tập tài liệu

- Dùng hàm đọc thời gian chung, nhận hậu tố Z do trình duyệt xuất trên Python 3.10+.
- CI cài dependency CPU cho trainer/API và Node; thêm chế độ thất bại khi test bị skip.
- Tách fingerprint dependency trainer khỏi fingerprint toàn release. Thay đổi
  giao diện giữ identity train; thay loss/collator/validator/runtime làm identity đổi.
- Bỏ triage viết sẵn theo ID và fallback coi ID chưa đánh giá là pass. Hàm hiện hành
  yêu cầu file visual-review có nguồn/hash. Các nhận xét lịch sử giữ trong release cũ.
- Tách template/CSS/JavaScript của editor khỏi Python, giữ output và khóa nháp.
- Rút gọn comparison/local-cloud/preference-training, thêm mục lục artifacts,
  bỏ package models và các thư mục report khung chưa có nội dung.
- Nhật ký đầy đủ trước lượt này được giữ nguyên byte tại
  `artifacts/code_cleanup_20261012/TIEN_DO_before_cleanup.md` và các source release
  ngày 12/10 trước khi sửa. Nhật ký hiện hành giữ quyết định đang áp dụng và bằng chứng mới.
- `scripts/check_project.py`: Ruff và 807 test qua, 26 warning fixture PEFT.
  Log cuối: `artifacts/code_cleanup_20261012/check_project_final.log`.
- `scripts/check_project.py --full-data` qua: 32.834 bản ghi, ba split tách ảnh,
  `errors=[]`; preflight xác nhận đủ ảnh cho 29.459 câu train.
  Run: `runs/audit_20261012_045935_695217` và
  `runs/preflight_20261012_050016_998010`; log ở thư mục cleanup.
- 61 liên kết file Markdown được kiểm, không có file thiếu trong nhóm hiện hành.
  Nhóm rule bảo mật Ruff S đã chạy và được ghi theo ngữ cảnh trong
  [docs/code_review.md](docs/code_review.md).
- Tải lại source Drive ngày 12/10 trong lượt này: hash khớp bản đã ghi.
  Các file mới đã được kiểm mẫu token trước khi đóng gói. Drive đã nhận
  `source_20261012_112049_100436.zip` và checksum, 231 file; UI xác nhận tải
  xong hai mục và đúng tên/kích thước. SHA-256 local:
  `c756564d911dd182bf689f6c11160e071af7414ad6a5ec4fc67c813b5e12b174`.
  Đã tải lại bằng mục Tải xuống trong menu tệp, SHA-256 khớp local.
  Bản ghim M1 và các pointer lịch sử giữ nguyên.

## Đồng bộ đã xác minh trước lượt sửa

Drive `source/releases` có `source_20261012_040415_495004.zip` và checksum.
Nhật ký lượt trước ghi đã tải lại ZIP và SHA-256 khớp:
`341456fe12ef75e0ccf2e7509454a9fbdf7418c5b7fa5c1e74681c7923f1c91a`.
Đã kiểm trực tiếp tên bản ZIP/checksum trên Drive trong lượt này.
Source sau sửa đã được upload như ghi ở mục trên; bản ghim M1 tiếp tục dùng nguyên bản.

GitHub đã có `OuyangXuelili/Vietnamese-VQA-Distillation`, chứa dự án cũ.
Repo local của nhánh VQA Explanation chưa có remote trước lượt này.
Source hiện hành đã push lên nhánh `codex/vqa-explanation`, commit
`569b6b8103af15b93c528770cf85d0cac08c3fa6`; `git ls-remote` khớp HEAD local.
Git giữ source/config/test/tài liệu; data đầy đủ, checkpoint và log nằm ngoài Git.
CI code hiện hành đã qua trên Python 3.10/3.11; xem biên bản ở mục CI phía trên. Mỗi lần cập nhật nhật ký sau đóng ZIP
được đồng bộ riêng, còn snapshot source đã đóng giữ nguyên.

## Việc tiếp theo

1. Xác nhận quyết định pilot, chốt N và export/freeze bộ cặp chung.
2. Kiểm GPU cho từng trainer, train M2/M3 và giữ các run riêng.
3. Xuất validation/test theo giao thức; chấm mù 200 ảnh và phân tích M3–M2.
4. Nghiệm thu demo live và bổ sung kết quả vào báo cáo.
