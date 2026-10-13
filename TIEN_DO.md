# Tiến độ dự án

Cập nhật: 13/10/2026.

## Trạng thái hiện hành

| Phần | Trạng thái và bằng chứng |
|---|---|
| Dữ liệu | ViVQA-X `image_disjoint_v1`, 29.459 train / 1.407 validation / 1.968 test; 28.180 ảnh đã kiểm local và bundle cloud |
| M0/M1 | M1 một epoch, 3.683 update; chọn checkpoint-3683 từ loss 256 validation; đã xuất đủ validation M0/M1 |
| Chỉ số M0 → M1 | EM 41,01 → 70,93; F1 52,41 → 75,11; ROUGE-L 27,05 → 44,29; BERTScore 75,82 → 82,47; chỉ số nhân 100 |
| Cặp ưu tiên | Pilot 100 không đổi. Đã xem 350 ảnh, giữ 48 cặp; tách 34 câu theo phạm vi. Full notes: 48 giữ / 177 loại / 159 chưa chắc / 1.616 pending |
| M2/M3 | Trainer/resume đã kiểm local; có CLI bàn giao review → freeze; chưa train GPU |
| Đánh giá cuối | Đã cố định mẫu 200 ảnh test; chưa có prediction test bốn nhánh hoặc nhãn người |
| Demo | Replay 12 ví dụ đã kiểm API và Chrome; live chưa nghiệm thu GPU |

Bảng validation và CI ở
[reports/generated/m0_m1_validation_20261011](reports/generated/m0_m1_validation_20261011/README.md).
Hướng nghiên cứu và các việc tiếp theo ở [docs/research_plan.md](docs/research_plan.md).

## 13/10 — tiếp tục duyệt 100 ảnh

- Dùng đúng 100 ID pending đã khóa ở lượt trước. Xem cả lượt qua bảng ảnh,
  mở lại ảnh gốc cho các cặp giữ. Giữ 17 cặp: 7 negative M1 nguyên văn,
  10 có kiểm soát; 49 câu loại, 34 chưa chắc. Không sửa pilot hoặc nhãn
  ngoài lượt này. Một cặp về râu được để uncertain vì chi tiết không rõ.
- Bản hiện hành: `data/preferences/review_batch_251_350_20261013`.
  Audit đủ 2.000 dòng qua; bộ metadata có 48 cặp, cùng ID/chosen M2/M3.
  Tổng nguồn rejected: 25 model / 23 synthetic. Chosen dài hơn rejected
  ở 23/48 cặp, đếm từ theo khoảng trắng, chưa phải thống kê token Qwen.
- Trang kết quả: `review_results_350_20261013.html`, chỉ hiển thị kết quả;
  actor/nguồn/recipe vẫn giữ trong dữ liệu. 0 cặp xác nhận bởi người;
  cổng dữ liệu train chính vẫn đóng. Không train lại M1, tạo notebook
  mới hoặc ghi đè output lịch sử.
- `continuation.json` khóa 100 ID pending tiếp theo. 200 là mốc dự toán,
  không phải ngưỡng chất lượng; N cuối quyết định từ phạm vi đã duyệt.
  Kế hoạch và pointer study đã cập nhật, giữ LR/epoch/test như trước.
- `check_project.py --full-data` qua Ruff và 910 test, không skip;
  audit 32.834 bản ghi không lỗi/trùng ảnh giữa split, preflight đủ ảnh
  cho 29.459 câu train. Log: `artifacts/preference_review_350_20261013/check_project.log`.
  Run audit `audit_20261013_110108_289172`, preflight
  `preflight_20261013_110148_219422`. 26 warning từ fixture PEFT.
- Drive `preference_local` đã nhận gói 34 file, không có ảnh/weights;
  tải lại ZIP khớp SHA-256
  `05cc8956d53bea722fa246811ab57358e2f0cb79f77c27326f26a41f830dc812`.
  `reviewed_notes.jsonl` cập nhật cùng file lên phiên bản 3; hai handoff
  lên phiên bản 4. Bản notes tải lại khớp local.
- Source cuối `source_20261013_111553_904173` gồm 238 file, tải lại khớp
  SHA-256 `2910d49c59968af508ad3ba7606682c784474c6c51d8a905b4aef8f8457c7980`.
  Bản 110700 giữ làm lịch sử, chưa cập nhật số đếm trong README; không
  dùng làm source hiện hành. Nhật ký đồng bộ riêng sau snapshot ZIP.
- Tài liệu Drive cập nhật cùng file: research_plan v18,
  preference_annotation v11, preference_training v12,
  project_description v5. Kế hoạch tải lại khớp local.
- Đối chiếu lại 304 file lịch sử: còn đủ và không đổi hash. Không đổi
  checkpoint M1, notebook/output hoặc nhãn của các lượt trước.
- Record khởi chạy Drive v4 ghim source cuối, tải lại khớp local.
  Inspect M2/M3 đều dừng đúng ở điều kiện chưa khóa bộ dữ liệu/N;
  không tải model hay khởi chạy GPU. Biên nhận ở
  `artifacts/preference_review_350_20261013/sync/receipt.json`.

## 13/10 — rà trước khi commit GitHub

- Rút gọn `docs/code_review.md`: giữ quy trình và công cụ đang dùng,
  bỏ đoạn giới thiệu chung và danh sách công cụ chưa tích hợp.
- Thêm `.gitattributes`: text dùng LF, notebook giữ nguyên byte.
  Máy đang bật autocrlf; quy tắc này tránh đổi hash notebook lưu trữ hoặc
  bootstrap khi clone trên Windows. Source package kèm quy tắc Git này.
- Metadata nguồn duyệt/văn bản giữ đúng; không đổi kết quả mô hình,
  nhãn, split, checkpoint hoặc output lịch sử.
- Thông điệp commit: `Update review and training workflow`.
- `check_project.py` qua Ruff và 910 test. Kiểm Git index: không có ảnh,
  checkpoint, source ZIP hoặc credential; notebook M1 lưu trữ khớp byte
  local và checksum bootstrap trong notebook khớp script lưu ở Git.
- Drive có source `source_20261013_070227_503592` (238 file), tải lại khớp
  SHA-256 `1a2c48f7a1f5e89f14e40904307b7e45546ef7f2945a3de05393cf0054f417d7`.
  `code_review.md` lên phiên bản 2; notebook M1 và output giữ nguyên.
- Record khởi chạy trên Drive lên phiên bản 3, ghim source cuối ở trên.
  Dữ liệu vẫn chưa khóa; lượt này không chạy GPU.

## 13/10 — rà tên file local

- Đổi notebook chưa chạy `03_portable_runner.ipynb` thành
  `04_train_preferences.ipynb`, đúng vai trò M2/M3. Đầu notebook và README
  có liên kết trực tiếp tới bản M1 đã chạy trên Colab.
- Đổi CLI đọc record `run_preference_training.py` thành
  `launch_preference_training.py`; trainer vẫn ở `train_preference.py`.
  Đổi test lịch sử thành `test_m1_notebook_history.py`.
- Config M2 chính LR 5e-6 dùng tên ổn định `configs/training/sft_chosen.yaml`.
  Bản thiết kế 2e-5 chuyển nguyên byte vào
  `configs/archive/sft_chosen_lr2e5.yaml`. Cập nhật đường dẫn trong study,
  experiment, CLI helper, test và tài liệu hiện hành.
- Quy ước tên trong `docs/code_review.md`. Tên mô tả công việc, dùng
  snake_case; ngày/phiên bản chỉ dùng khi cần truy vết. Tên các bản lịch sử,
  release, dữ liệu và checkpoint giữ nguyên. `check_project.py` qua Ruff
  và 910 test local. Đối chiếu 304 file bằng chứng còn nguyên hash;
  CLI kiểm hai nhánh xác nhận LR/batch/epoch không đổi. Link Markdown
  local không hỏng; 183 file Python/YAML dùng tên thống nhất.
- Rà tên output: log/state/result/manifest dùng tên theo chức năng; run và
  checkpoint đủ dấu mốc truy vết. Các thư mục sửa nhiều lượt có tên dài như
  corrected/qa/rechecked giữ làm lịch sử, không dùng cho lượt mới. Mục lục
  `artifacts/README.md` ghi bản hiện hành và quy ước output; không đổi prediction,
  notes, log, checkpoint hoặc source release đã chạy.
- Drive: notebook đổi tên/cập nhật trên cùng ID
  `1emxS1s8mZHYQxVqR4e4tA8j8G_xrzhl0`, phiên bản 111; tải lại khớp local.
  Source `source_20261013_063823_176343`, 237 file, đã tải lại khớp
  SHA-256 `8210eb1bb8c92515f082922a3d4bb326e8e039593cdccb4210a0a50c47149d98`.
  Record khởi chạy phiên bản 2 ghim source/config mới, tải lại khớp;
  giữ `data_locked=false`, hai nhánh inspect chưa mở train.
- Tài liệu Drive cập nhật trên file cũ: research_plan v17,
  preference_training v11, local_cloud v2, experiment_protocol v4;
  code_review đã tải lên. Mục lục notebooks đã tải lên, mục lục archive v2.
  Source ZIP giữ snapshot trước biên nhận đồng bộ;
  nhật ký này cập nhật riêng trong Drive source.

## 13/10 — tổ chức notebook và rà code local

- Notebook 03 từ 27 xuống 7 cell, chỉ điều phối M2/M3. Logic train/loss
  ở `src/`; CLI và bootstrap ở `scripts/`. Một lần mount, một setup,
  hai cell gọi nhánh và một cell đọc kết quả; mặc định không train.
- Bản local trước sửa giữ nguyên byte tại
  `notebooks/archive/03_m1_and_preference_before_reorganization_20261013.ipynb`,
  SHA-256 `83972dbdae24e32b7edfda204b156b1aa988c2f52387e53f4d5adacc8124863d`.
  Bản tải từ Colab có 20 cell, 24 output, lưu tại
  `notebooks/archive/03_m1_completed_with_outputs_20261013.ipynb`,
  SHA-256 `59744c89896b1a9b5dbf541f664c774df920610f7ccccb32bf164b3e4e8dee2d`.
  Không xóa output cũ; source ZIP/Git không chứa bản raw output này.
- Đưa thử nghiệm gate riêng 31 cặp chưa chạy ra khỏi luồng chính, giữ
  nguyên file trong `artifacts/code_organization_20261013/retired_gpu_gate_31`.
  Chốt bộ cặp cuối trước một gate GPU phù hợp, không mở vòng smoke mới.
- Sửa resume: output mới theo tên phiên của từng nhánh, dùng lại đúng
  gate gốc. Bootstrap kiểm source rồi khôi phục vào thư mục mới, không
  xóa checkout cũ. Bỏ `.egg-info` sinh tự động khỏi source release.
- Runtime mới khôi phục dataset bundle bằng verifier đã có. Đường dẫn
  dữ liệu lấy từ profile, không đưa Drive vào `src/`. Record khởi chạy
  chưa khóa N nên dừng trước đọc stage/tải model; M1 không chạy lại.
- Inventory tự động kiểm 237 file source: Python/notebook/YAML, cú pháp
  JavaScript và liên kết Markdown local. Chưa thấy lỗi cú pháp, file
  nguồn trùng hệt hoặc link local hỏng. Review tập trung thêm vào loss,
  collator, selection, split, resume và API demo; không coi inventory là
  đọc thủ công từng dòng. Hồ sơ ở `artifacts/code_organization_20261013`.
- Đối chiếu 304 file bằng chứng trước sửa: tất cả còn nguyên hash,
  gồm annotation, notes preference, notebook lịch sử và run cũ.
  Full-data audit `audit_20261013_053234_893435` qua 32.834 câu, không lỗi
  hoặc trùng split; preflight `preflight_20261013_053244_822302` đủ ảnh
  cho 29.459 train. Không sửa split, nhãn, điểm hoặc checkpoint M1.
- Gate toàn repo qua Ruff và 907 test, `--fail-on-skip`, không bỏ qua;
  26 warning từ fixture PEFT. Kết quả máy đọc ở `tests.xml` trong hồ sơ.
  Mặc định inspect và điều kiện khôi phục bundle đã kiểm bằng test hành vi.
- Drive đã có hai bản 03 lưu trữ; tải lại bản có output khớp hash local.
  Sáu notebook kiểm thử cũ được chuyển vào `notebooks/archive` bằng
  thao tác Move, giữ file và lịch sử phiên bản.
- Source `source_20261013_054116_710805` (237 file) đã lên Drive releases,
  tải lại khớp SHA-256 `52e9b3bfead77a681f4ea9c17a337a1cbd08c039d0379e5b16c8de2bcd27bbcc`.
  Bootstrap khôi phục ZIP thật qua trên local; hai lệnh inspect M2/M3
  đều báo chưa khóa N, không train.
- File 03 gốc đã chuyển từ Colab Notebooks vào `Vietnamese_VQA_NLE/notebooks`,
  giữ ID `1emxS1s8mZHYQxVqR4e4tA8j8G_xrzhl0`. Cập nhật phiên bản 110
  trên cùng file, không tạo bản (1). Tải lại khớp source local:
  SHA-256 `9ff74ef1fc9724ddb2fd78b1de2d485bd8edafaa82b3b3530bfc5f159a8c6c08`.
  Source ZIP giữ snapshot trước upload. Chưa train GPU M2/M3.
- Drive `source/` có bootstrap và record khởi chạy với `data_locked=false`;
  record tải lại khớp local. Tài liệu cập nhật trên file cũ: research_plan
  phiên bản 16, preference_training phiên bản 10; local_cloud đã tải lên.
  Hồ sơ đồng bộ `artifacts/code_organization_20261013/sync_receipt.json`.
  Bản bootstrap được xác nhận upload bằng giao diện; tải riêng `.py` chưa
  hoàn tất trong Chrome. Byte của script trong source ZIP tải lại đã kiểm.
  Nhật ký này cập nhật riêng sau snapshot, không tạo thêm source release.

## 13/10 — duyệt lượt 151–250

- Khóa 100 ID theo bản tiếp tục trước, không sinh lại ứng viên. Xem từng
  ảnh gốc, giữ 14 cặp, loại 56, để 30 chưa chắc. Rà lại cặp định giữ và
  rút hai cặp yếu: màu ảnh sepia và diễn đạt hình tượng.
- Quy tắc trước duyệt: tối đa 10 negative có kiểm soát, một chi tiết
  nhìn rõ, cùng đáp án; lấy trường hợp hợp lệ đầu tiên theo thứ tự. Giữ
  9 negative tổng hợp và 5 rejected M1 nguyên văn, không bù đủ quota.
  Điểm hỗ trợ vẫn chấm M1 gốc, không gán lỗi tạo mới cho M1.
- Tách 34 câu về ý định, sở thích hoặc sức khỏe chỉ theo câu hỏi, để
  uncertain. Không xem ảnh hoặc chấm M1 cho nhóm này; trường bằng chứng
  và điểm hỗ trợ để null. Không sửa canonical, split, candidate hoặc M1.
- Bản hiện hành: `data/preferences/review_batch_151_250_20261013`.
  Full notes đủ 2.000 dòng: 31 accepted / 128 rejected / 125 uncertain /
  1.716 pending. Trong 284 câu đã xử lý có 250 xem ảnh và 34 tách phạm vi.
  Pilot vẫn 12/47/41. Audit/export/freeze thật qua; 31 chosen M2 khớp M3,
  18 nguồn rejected model và 13 synthetic; 0 human_verified.
- Trang kết quả `review_results_250_20261013.html` chỉ có kết quả/lý do;
  nguồn kỹ thuật giữ trong JSONL/recipe. Các trang và bản dữ liệu cũ giữ
  nguyên. `continuation.json` khóa 100 pending tiếp theo.
- Chưa khóa N cuối, chưa train GPU M2/M3, không tạo notebook. Cổng train
  vẫn đóng; kế hoạch giữ phần chấm mù 200 ảnh test bởi người.
- Chrome kiểm bộ lọc: 31 cặp giữ, 100 câu lượt mới, 91 chưa chắc đã
  xem ảnh; ảnh tải đúng. Ảnh bằng chứng `results_chrome.jpg` trong bản
  hiện hành. Không hiển thị nguồn duyệt trên trang kết quả.
- Gate source qua Ruff và 872 test, 26 warning fixture PEFT. Full-data
  audit 32.834 câu qua, preflight đủ 29.459 train; run
  `audit_20261012_180928_798165`, `preflight_20261012_181005_071749`.
  Đây là kiểm dữ liệu/code local, không phải train hoặc số đo GPU.
- Drive nhận `source_20261012_181127_963187` (234 file, không ảnh/weights)
  và `preference_review_250_20261013.zip` (bản đầy đủ 31 mục, 407.164 byte). Tải lại
  cả hai ZIP khớp checksum local. Bằng chứng ở
  `artifacts/preference_ai_triage_m1_3683_pilot_20261012_qc/sync_250`.
  Full notes được cập nhật trên file cũ trong `preference_local`;
  handoff.md/json lên phiên bản 3. Tài liệu cũ được cập nhật:
  research_plan v15, preference_annotation v10, preference_training v9.
  Nhật ký tiếp tục ở `source/TIEN_DO.md`, tải riêng sau source snapshot.
  Không tạo notebook, không xóa hay ghi đè các release/run trước.
- Rà gói lưu phát hiện thiếu hai file nguồn mà cặp pilot cũ tham chiếu:
  triage.jsonl và drafts.jsonl. Bổ sung cả hai, giữ bản đóng gói đầu local;
  ZIP/checksum Drive cập nhật phiên bản 2 trên cùng file. Nhãn, cặp,
  recipe, canonical và điểm không đổi. Bản hoàn chỉnh ở `sync_250/complete`.

## 12/10 — chốt pilot và mở rộng cặp

- Người thực hiện giao quyền chốt pilot trong cuộc trò chuyện. Giữ đúng
  actor trong dữ liệu, không gán tên/mã người vào quyết định AI. Sửa exporter
  từ nhãn cố định `human_review_candidate` thành theo actor; audit/freeze
  chặn AI ghi human_verified hoặc văn bản AI ghi manual.
- Xem lại 13 cặp định giữ, rút `311428001`: hai người trong ảnh không cùng
  trạng thái mặc áo nên câu hỏi “mọi người” mơ hồ. Pilot hoàn tất: 12 giữ,
  47 loại, 41 chưa chắc. Hồ sơ `data/preferences/pilot_ai_20261012`;
  8 rejected M1 nguyên văn, 4 negative có kiểm soát.
- Xem trực tiếp thêm 50 ảnh theo thứ tự pool, giữ 5 cặp lỗi tự nhiên,
  loại 25 câu, để 20 chưa chắc. Bỏ một nháp thả diều vì không thấy rõ người
  đang điều khiển hay quan sát. Không thay reference/candidates/split/M1.
- Bản tiếp tục: `data/preferences/ai_review_150_20261012/reviewed_notes.jsonl`.
  Có đủ 2.000 dòng: 17 accepted / 72 rejected / 61 uncertain / 1.850 pending.
  Audit và export/freeze thật qua; 17 chosen M2 khớp chosen của 17 cặp M3.
  Bộ này đang mở rộng, không dùng làm run chính; config vẫn chưa khóa N mới.
- Trang `review_results_150_20261012.html` chỉ hiển thị kết quả và lý do
  theo yêu cầu, không có dòng nguồn duyệt. Nguồn vẫn giữ trong JSONL và
  manifest. Bản nháp/trang và hồ sơ trước không bị ghi đè.
- Chính sách duyệt `ai_visual_review_v1` được ghi trước train. Kiểm hết
  pool hiện có, lấy N thực tế; không ép đủ 1.000 hoặc sinh lại ứng viên.
  Phần chấm mù 200 ảnh test bởi người vẫn giữ nguyên.
- Ruff và 872 test qua, fail-on-skip; 26 warning PEFT từ fixture, không có
  test bỏ qua. Log `artifacts/pilot_finalization_verification_20261012.log`.
  Full-data audit 32.834 câu không lỗi/trùng split; preflight đủ 29.459
  train qua. Run `audit_20261012_165403_505346` và
  `preflight_20261012_165440_000837`.
- Chrome đã kiểm bộ lọc giữ đúng 17 cặp, chụp ở
  `data/preferences/ai_review_150_20261012/results_chrome.png`.
- Drive đã nhận source `source_20261012_165659_077735` (234 file) và
  `preference_review_150_20261012.zip` (201.732 byte, không ảnh/weights).
  Tải lại hai ZIP khớp checksum local; bằng chứng ở
  `artifacts/preference_ai_triage_m1_3683_pilot_20261012_qc/sync_150`.
  `preference_local` có `reviewed_notes.jsonl` mới; handoff.md/json cập nhật
  phiên bản 2 trên file cũ. Không tạo bản tải trùng trong Downloads.
  Tài liệu Drive cập nhật trên file cũ: research_plan v14,
  preference_annotation v9, preference_training v8. Nhật ký tiếp tục dùng
  `source/TIEN_DO.md`; bằng chứng đồng bộ ở `sync_150/sync_receipt.json`.
  Source ZIP giữ snapshot trước khi đồng bộ, nhật ký tải lên riêng.
  Chưa chạy GPU M2/M3.

## 12/10 — triển khai lệnh bàn giao preference

- Thêm `scripts/finalize_preference_review.py`; logic dùng lại ở
  `preference/handoff.py`. Một lệnh audit bản xuất, lưu snapshot/báo cáo
  và liệt kê ID cặp nháp cần kiểm. Không tạo notebook hoặc tự sửa nhãn.
- Chế độ freeze gọi exporter/validator hiện có, chỉ khi pilot đã chốt,
  có accepted và N đúng bằng số cặp. Hai nhánh nhận cùng chosen; chặn
  nguồn/bundle/canonical đổi, dữ liệu test và output đã tồn tại. Công bố
  output sau khi hoàn tất; lỗi giữa chừng không để bộ train dở.
- Chạy trên bản kết quả pilot hiện hành: 2.000 pending, 100 draft_pending,
  13 cặp nháp đủ trường, 0 accepted. Hồ sơ dùng hiện tại ở
  `artifacts/preference_handoff_20261012_v2`; cổng train vẫn đóng.
- Lượt đầu ghi báo cáo nhưng stdout Windows cp1258 lỗi với dấu tiếng
  Việt. Đã sửa JSON console về ASCII escape, giữ báo cáo UTF-8 và thêm
  regression. Hồ sơ đầu giữ nguyên, lượt v2 hoàn tất exit 0.
- 35 kiểm thử handoff/pilot qua, gồm export/freeze thật bằng fixture,
  giữ provenance và đối chiếu chosen M2/M3; không tải model. Gate toàn repo
  qua Ruff và 856 test, fail-on-skip, 26 warning fixture PEFT.
  Log `artifacts/preference_handoff_verification_20261012.log`.
- Cập nhật kế hoạch và hướng dẫn annotation/training trên file cũ.
  Không thay canonical, checkpoint, notebook hoặc collator/train; không
  chạy lại full-data hay GPU. M2/M3 vẫn cần bản duyệt được xác nhận,
  chốt phạm vi/N rồi GPU gate trước train chính.
- Source `source_20261012_161552_662617`, 234 file, đã upload ZIP/checksum
  vào Drive releases; tải lại ZIP khớp SHA-256
  `55d1aa963a64aecda38e59daa5f6c40360a9d9715fd0640cb182af96dcce07fd`.
  ZIP không có ảnh, weights hoặc notes review. Không dùng bản đóng gói
  trước đó trong lượt này vì hướng dẫn còn cần một câu đính chính.
- Tài liệu trên Drive cập nhật trên file cũ: research_plan v13,
  preference_annotation v8, preference_training v7. `preference_local`
  có `handoff.md`/`handoff.json` mới; chưa có bundle train được duyệt.
  Bằng chứng upload/readback ở `artifacts/preference_handoff_20261012_sync`.

## 12/10 — rút gọn cách hiển thị kết quả pilot

- Theo yêu cầu, bỏ tiền tố “AI đề xuất”/“AI chưa chắc” trong bằng chứng
  và lý do của 100 câu: 187 trường được rút gọn. Hiển thị kết quả từng
  câu và tổng 13 giữ / 47 loại / 40 chưa chắc; không đổi đánh giá.
- Trang hiện hành: `pilot_review_results_20261012.html` trong thư mục
  review M1-3683. Nhận xét chỉ còn nội dung; nguồn AI giữ trong mục kỹ
  thuật thu gọn và manifest. Không đổi thành nhãn người hoặc accepted.
- Sao lưu JSONL từ tab trước khi chỉnh. Chosen/rejected, nguồn, điểm,
  người/thời gian và quyết định giữ nguyên; 1.900 câu ngoài pilot không
  đổi. Hồ sơ ở `ai_prefill/results_view`; worksheet cũ giữ nguyên.
- Audit qua; xuất lại từ Chrome và đối chiếu đủ 2.000 dòng. Không sửa
  code train, canonical, checkpoint hoặc mở train M2/M3.
- Drive `preference_local`: đã tải gói `pilot_results_20261012.zip` và
  checksum; tải lại ZIP khớp SHA-256
  `bbca77095771d27d8e9fc85e92b53d0b2c0c5cf5fb1e7394af8efa95efdedb80`.
  Gói 338.265 byte, 7 entry, không ảnh/weights. Không tạo source release mới.
- Đường dẫn trong tài liệu Drive đã cập nhật trên file cũ:
  research_plan v12, preference_annotation v7.

## 12/10 — điền nháp đủ 100 câu pilot

- Theo yêu cầu, xem trực tiếp đủ 100 ảnh gốc cùng câu hỏi, tham chiếu và
  đầu ra M1. Điền đánh giá đáp án, hỗ trợ giải thích và bằng chứng nhìn thấy.
  Đề xuất giữ 13 cặp, loại 47 câu và để 40 câu chưa chắc. Không tự ép
  câu chủ quan hoặc chi tiết khó thấy thành nhãn đúng/sai.
- 13 cặp có chosen/rejected, đoạn cần kiểm, loại lỗi và nguồn văn bản:
  9 rejected nguyên văn M1, 4 negative synthetic. Nhãn đánh giá M1 luôn
  chấm đầu ra gốc, không chấm negative synthetic thay cho M1.
- Trang hiện hành: `pilot_review_prefilled_20261012.html` trong thư mục
  review M1-3683. Đã mở trên cùng tab Chrome. Bản cũ và notes gốc giữ nguyên;
  bản xuất trước thao tác được lưu ở `ai_prefill/browser_before.jsonl`.
- `ai_prefill/notes_ai_prefilled.jsonl` có đủ 2.000 dòng: 100 nháp đã điền,
  1.900 dòng ngoài pilot không đổi. Không ghi người/time xác nhận; toàn bộ
  vẫn pending, 0 accepted. Đây là nháp AI để người thực hiện kiểm lại,
  không phải nhãn người độc lập hoặc dữ liệu đã được phép train.
- Đối chiếu hash 100 ảnh với manifest nguồn; audit notes qua, pilot có
  100 draft_pending. Audit kiểm trường và nguồn, không chứng minh nhãn đúng.
  Hồ sơ: `artifacts/preference_ai_triage_m1_3683_pilot_20261012_qc/ai_prefill`.
- Xuất JSONL từ Chrome, đọc lại và so đủ 2.000 dòng: nội dung khớp bản
  worksheet đã dựng. Bằng chứng `ai_prefill/browser_roundtrip.json`;
  screenshot `ai_prefill/prefilled_chrome.png`.
- Không sửa canonical, candidates, checkpoint M1, code train, split hoặc
  mẫu test. Chưa train M2/M3. Lượt này chỉ tạo nháp và cập nhật đường dẫn
  tài liệu, không chạy lại full-data hoặc công bố test GPU mới.
- Drive `preference_local`: đã upload `pilot_prefilled_20261012.zip` và
  checksum; tải lại ZIP khớp SHA-256
  `d168de7317ce62b4f9a1425e4687061d9939e30c05bc0f4e57563631ee120399`.
  Gói có 13 entry, 391.802 byte, không ảnh/weights. Hai tài liệu ở
  `documents` đã cập nhật trên file cũ: research_plan v11,
  preference_annotation v6. Không tạo source release mới vì code/config
  không thay đổi.

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

1. Duyệt tiếp pool đã khóa, chốt N và export/freeze bộ cặp chung; pilot đã chốt.
2. Kiểm GPU cho từng trainer, train M2/M3 và giữ các run riêng.
3. Xuất validation/test theo giao thức; chấm mù 200 ảnh và phân tích M3–M2.
4. Nghiệm thu demo live và bổ sung kết quả vào báo cáo.
