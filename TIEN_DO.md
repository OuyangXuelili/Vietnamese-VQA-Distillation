# Tiến độ dự án

Cập nhật: 14/10/2026.

## Trạng thái hiện hành

| Phần | Trạng thái và bằng chứng |
|---|---|
| Dữ liệu | ViVQA-X `image_disjoint_v1`, 29.459 train / 1.407 validation / 1.968 test; 28.180 ảnh đã kiểm local và bundle cloud |
| M0/M1 | M1 một epoch, 3.683 update; chọn checkpoint-3683 từ loss 256 validation; đã xuất đủ validation M0/M1 |
| Chỉ số M0 → M1 | EM 41,01 → 70,93; F1 52,41 → 75,11; ROUGE-L 27,05 → 44,29; BERTScore 75,82 → 82,47; chỉ số nhân 100 |
| Cặp ưu tiên | Hết 2.000 ứng viên: 1.966 ảnh đã xem, 34 câu tách theo phạm vi, 0 pending. Khóa 186 cặp sau rút 15 cặp yếu; pilot/notes cũ không đổi |
| M2/M3 | Đã train trên T4, cùng 186 cặp, mỗi nhánh một epoch / 24 update; policy cập nhật, adapter lưu được kiểm tra; reference M3 giữ nguyên |
| Đánh giá cuối | Đã cố định mẫu 200 ảnh test; chưa có prediction test bốn nhánh hoặc nhãn người |
| Demo | Replay 12 ví dụ đã kiểm API và Chrome; live chưa nghiệm thu GPU |

Bảng validation và CI ở
[reports/generated/m0_m1_validation_20261011](reports/generated/m0_m1_validation_20261011/README.md).
Hướng nghiên cứu và các việc tiếp theo ở [docs/research_plan.md](docs/research_plan.md).

## 14/10 — hoàn tất train M2/M3

- Hai gate đã qua: M2 5 update / 41,14 giây / 4,35 GiB VRAM;
  M3 5 update / 109,50 giây / 5,16 GiB, reference giữ nguyên.
  Run chính nạp lại M1-3683, không nối trọng số từ gate.
- M2 `m2_train`: đủ 24 update, epoch 1,0; thời gian train 170,44 giây,
  loss 1,5811, peak VRAM 4,37 GiB. M3 `m3_train`: đủ 24 update,
  epoch 1,0; 503,20 giây, DPO loss 0,6513, peak VRAM 5,17 GiB.
  Cả hai `status=passed`, `policy_updated=true`, `saved_policy_verified=true`;
  M3 thêm `reference_unchanged=true`. Loss hai nhánh không so trực tiếp.
- Checkpoint cuối ở Drive `runs/preferences/vivqax_m1_3683_v1/`
  `m2_train/training/checkpoint-24` và `m3_train/training/checkpoint-24`.
  Output run/gate cũ giữ nguyên. Chưa chạy validation/test cho M2/M3.
- Bản notebook đã chạy giữ tại
  `notebooks/archive/04_preferences_completed_with_outputs_20261014.ipynb`;
  file Drive vẫn cùng ID `1emxS1s8mZHYQxVqR4e4tA8j8G_xrzhl0`, đủ output.
  Code khớp template local, khác duy nhất MODE của phiên đang dùng (`train`);
  template local mặc định `inspect` để tránh khởi chạy GPU ngoài ý muốn.
- Báo cáo JSON local được trích từ stdout notebook, không thay checkpoint
  hoặc các file run gốc trên Drive. Xem
  `reports/generated/preference_training_20261014/README.md`.
  Source đã dùng là `source_20261014_021042_322687` (243 file), launch v13;
  code đã push ở `56dfb7e`. 937 test và full-data audit/preflight qua trước train.
- Việc tiếp theo: xuất 1.407 validation cho M2/M3, kiểm các manifest trước
  so với M0/M1; khóa inference rồi mới xuất test và chấm mù 200 ảnh.
  Không train thêm epoch chỉ từ loss.
- `documents/preference_training.md` và `research_plan.md` đã cập nhật
  trên Drive, tải lại khớp local. Gói `reports/training_evidence_20261014.zip`
  cùng `.sha256` đã upload; readback khớp ZIP và đủ sáu file: notebook đầy đủ
  output, hai report JSON trích stdout, hai ảnh kết quả và README.
  ZIP SHA-256 `8f16818b33f1eae47bbab22781ee34262050aedaf65cbbb8c24f656a9441531e`.
  Không có ảnh dữ liệu hoặc trọng số trong gói này; checkpoint gốc vẫn ở runs.
  Đối chiếu lại 132 hash source trong từng report đều khớp code local.
  Notebook có đủ execution count cho sáu cell code, không có output lỗi.

## 14/10 — khởi động notebook 04 trên T4

- Lượt hiện hành: cell 1 đã qua, output `Mounted at /content/drive` và
  `Project storage ready: /content/drive/MyDrive/Vietnamese_VQA_NLE`.
  Sửa notebook 04 dùng `timeout_ms=600000`, kiểm launch record tồn tại và
  in kho dự án; chỉ một lệnh mount. Output trước sửa giữ tại
  `runtime_setup/04_before_mount_timeout_fix_20261014.ipynb`.
  `check_project.py` qua Ruff và 926 test. Source
  `source_20261014_013804_404975` đã lên Drive, tải lại khớp SHA-256
  `a251701e972b4ace5f222cddf7e8d861fe37e5e7a1e57cfdc7f44140990122d1`
  và kiểm đủ 238 member. Launch v12 đã tải lại khớp local.
  Cell 2 cài được gói ghim nhưng `pip check` của kernel thất bại: RAPIDS
  cần CUDA 13, Gradio/diffusers cần huggingface-hub mới hơn. Output đầy đủ
  đã giữ tại `mount_timeout_20261014/04_dependency_conflicts.ipynb`.
  Chưa chạy GPU gate/train.
  Đây là sửa khởi động Colab, không đổi bộ cặp hoặc cấu hình học.
- Tách môi trường bằng `setup_preference_environment.py`, không nạp system
  site-packages. CLI ảnh/M2/M3 dùng Python riêng; log stdout/stderr được
  chuyển tiếp ra notebook, không bỏ qua lỗi phụ thuộc. Không thêm cell.
  `check_project.py --full-data` qua Ruff, 937 test và đủ dữ liệu; run
  `audit_20261014_020814_802091`, `preflight_20261014_020850_321433`.
  Bộ 186 cặp, M1 và các thông số học không đổi. Bản môi trường riêng chưa
  được chạy GPU. Source `source_20261014_021042_322687` đã lên Drive, tải
  lại khớp SHA-256 `7c0f3c7cfdeb405cfc76f2b2103178294864eb262f381665bf1e4184f3d44809`
  và kiểm đủ 243 member. Launch v13 tải lại khớp local; cell 2 đang cài
  vào `/content/vqa_envs/source_20261014_021042_322687`, có log trực tiếp.
  Các cell 3–5 đã chuyển cùng Python riêng; chưa thực thi trước khi setup qua.
- Mở đúng file Drive `04_train_preferences.ipynb`, ID
  `1emxS1s8mZHYQxVqR4e4tA8j8G_xrzhl0`, tài khoản lưu dự án. Không tạo
  notebook mới, không đổi dữ liệu/siêu tham số hoặc chạy lại M1.
- Colab đã kết nối Python 3/GPU T4, RAM 12,67 GB. Cell 1 hết thời gian
  chờ và báo `ValueError: mount failed`; hộp kết nối Drive vẫn mở.
  Các cell source/preflight/gate/train chưa chạy. Sau xác thực phải chạy
  lại cell 1, không chạy tất cả hoặc bỏ qua gate.
  Đã yêu cầu người dùng hoàn tất xác thực; không tự bấm quyền Drive.
- Sau khi người dùng báo xác thực xong, đã lưu notebook có output lỗi ở
  `runtime_setup/04_before_mount_retry_20261014.ipynb` (7 cell, 1 output lỗi).
  Chạy lại đúng cell 1 trên T4 hiện tại; Colab mở hộp kết nối Drive mới,
  lần này cũng báo `mount failed` khi chưa hoàn tất xác thực. Không tạo
  notebook hoặc chạy cell source/train. Cần đóng hộp hết hạn, chạy cell 1
  rồi hoàn tất quyền ngay trong lần chạy đó.
  Nhật ký lượt retry đã lưu local/Git; chưa cập nhật Drive vì tab Drive
  không phản hồi thao tác kết nối. Source và launch v11 đã đồng bộ ở lượt trước.
- Rà launcher phát hiện trường pin decision record đã được ghi nhưng chưa
  được đối chiếu. Thêm kiểm checksum quyết định và manifest frozen trước
  đọc stage/model; cùng 186 cặp nhưng khác bundle cũng bị từ chối.
  34 test launcher qua; kiểm riêng bộ 186 cặp khớp record đã khóa.
- `check_project.py --full-data` qua Ruff và 925 test, không skip. Audit
  32.834 dòng và preflight 29.459 train qua; run
  `audit_20261013_200114_972933`, `preflight_20261013_200149_880093`.
  Log ở `artifacts/preference_review_complete_20261014/runtime_setup/check_project.log`.
  26 warning từ fixture PEFT, không phải lỗi train.
- Source `source_20261013_200507_278622` đã tải lên và tải lại kiểm đủ 238
  member, SHA-256 `0664c988874b34b4468906aed3c514b57229f75350557d42f1631e32259f7236`.
  Record launch Drive v11 và `documents/preference_training.md` cũng đã tải
  lại khớp local. Không đóng lại gói dữ liệu vì bộ 186 cặp không đổi.
  Readback ở `runtime_setup/readbacks`; 304 file lịch sử khớp hash cũ.
- M2/M3 chưa train GPU. Bộ cuối, M1 và output lịch sử giữ nguyên.

## 14/10 — hoàn tất pool và khóa bộ train

- Xử lý 916 ID còn lại theo selection đã lưu trước khi xem: thêm 48 cặp,
  584 loại, 284 chưa chắc. Đã xem đủ 1.966 ảnh; 34 câu tách theo phạm vi.
  Full notes 201 giữ / 1.106 loại / 693 chưa chắc / 0 pending.
- Rà toàn bộ 201 cặp trên 26 bảng ảnh, mở lại ảnh gốc khi cần. Rút 15 cặp
  yếu khỏi export train; không sửa quyết định pilot hay notes lịch sử.
  Lý do từng ID ở `data/preferences/vivqax_m1_3683_v1/quality_review.json`.
  Một nháp bị lỗi cụm rejected đã giữ ở `attempt_01`; sửa/rút trước công bố.
- Khóa `data/preferences/vivqax_m1_3683_v1/frozen`: 186 cặp / 186 ảnh,
  78 rejected M1 nguyên văn và 108 negative có kiểm soát. M2/M3 có đúng
  cùng ID/chosen; validator kiểm canonical, nguồn, hash ảnh và train-only.
  Đây là tập preference nhỏ do cùng trợ lý kiểm, không phải nhãn người.
- Hai config và study chốt N=186 trước train/test; giữ LR 5e-6, seed 42,
  epoch 1, M1-3683. Dự kiến 24 optimizer update/nhánh. Thêm kiểm từ chối
  config khác N đã khóa. Không tạo notebook hoặc train lại M1.
- Trang hiện hành `review_results_final_20261014.html`; lịch sử 1.050 và
  1.966 ảnh giữ nguyên. Kế hoạch hiện hành rút gọn, không lặp nhật ký từng lượt.
- `check_project.py --full-data` qua Ruff và 914 test không skip; audit
  đủ 32.834 dòng, preflight 29.459 train. Run `audit_20261013_190756_222601`
  và `preflight_20261013_190832_479387`; 26 warning PEFT từ fixture.
  Kiểm riêng bộ cuối: 0 ảnh validation/test, cùng 186 ID/chosen, nguồn
  model khớp M1-3683, recipe synthetic đúng hash; 304 file lịch sử không đổi.
- Độ dài trung bình chosen/rejected 10,34/9,28 từ (không phải token).
  Chosen dài hơn 80 cặp, ngắn hơn 28, bằng nhau 78; báo rủi ro học độ dài.
- Gói nhãn 112 file, chỉ metadata, SHA-256
  `ad6ce32a0de7051ad82e9d06d7acf21a2ebfd692c547c608bec4374c0eaf5392`.
  Đã tải lại ZIP và notes từ Drive, khớp local. Notes v9; handoff md/json v10.
- Drive `preference_local/vivqax_m1_3683_v1/frozen` có đủ ba file đầu vào;
  tải lại cả ba khớp byte. Hồ sơ quyết định/quality review đặt cùng phiên bản.
  Source 238 file `source_20261013_191411_197153` tải lại và kiểm từng member,
  SHA-256 `bf29279ab7e179880e4b8b72638c4eab0be47aa961246a9697073d7837dd2ed2`.
- Record launch v10 ghim source mới, N=186 và đúng đường dẫn bộ cuối.
  Tài liệu cập nhật trên file Drive cũ; tải lại cả bốn khớp local. Notebook 04
  cập nhật cùng file, tải lại khớp; chỉ sửa mô tả, giữ 7 cell và output M1 cũ.
  Không tạo notebook mới hoặc source release thứ hai để đổi nhật ký upload.
- Trong kiểm tải ZIP, điều khiển theo index đã phát sinh một bản sao.
  Đã đổi tên `audit_download_copy_20261014.zip`, không dùng train. Hủy
  di chuyển khi Drive hỏi đổi người có quyền; không đổi chia sẻ hoặc xóa file.
  Chuyển sang thao tác bằng nhãn/phím của Drive; gói chuẩn đã tải lại khớp.
- Trang cuối được kiểm trên trình duyệt: 186 cặp, 15 lý do rút; chỉ hiện
  kết quả. Bằng chứng, checksum tải lại và Git ở
  `artifacts/preference_review_complete_20261014/sync/receipt.json`.
- Bước kế: notebook 04 kiểm input thật/token trên runtime GPU, gate 5 update
  mỗi nhánh; nạp lại M1 sạch rồi train M2/M3 một epoch. M2/M3 chưa train GPU,
  chưa có bằng chứng chất lượng tăng. Giữ phần chấm mù test bởi người.

## 14/10 — duyệt liền 300 ảnh, lượt 751–1.050

- Hoàn thành ba nhóm 100 ID theo continuation đã khóa; xem đủ ảnh qua
  bảng ảnh, mở lại ảnh gốc của 43 cặp giữ. Không sinh thêm ứng viên.
- Kết quả từng nhóm: 16/45/39, 15/41/44 và 12/44/44 (giữ/loại/chưa chắc).
  Thêm 13 rejected M1 nguyên văn và 30 đối chứng có kiểm soát. Đối chứng
  không được tính thành lỗi gốc của M1; thiếu liên quan ghi riêng với bịa chi tiết.
- Hai nháp đối chứng cuối được rút về một thuộc tính trước công bố.
  Bản đầu vẫn ở `artifacts/preference_review_1050_20261014/initial_publish`.
  Không đổi nhãn trước lượt, pilot, M1, split, LR, epoch hoặc output notebook.
- Bản hiện hành `data/preferences/review_batch_951_1050_20261014`:
  153 cặp cùng ID/chosen cho M2/M3, 61 model / 92 synthetic. Còn 916 pending;
  100 ID tiếp khóa trong continuation. M2/M3 chưa train GPU, N chưa chốt.
- Trang `review_results_1050_20261014.html`. Kiểm toàn repo, đóng gói và
  đồng bộ sau cả 300 ảnh. Không tạo thêm notebook hoặc sửa output train cũ.
- `check_project.py --full-data` qua Ruff và 910 test không skip. Audit
  32.834 dòng, không trùng ảnh giữa split; preflight đủ 29.459 train.
  Run `audit_20261013_173154_212649`, `preflight_20261013_173226_965140`
  đặt theo UTC. Log và kết quả rà ở `artifacts/preference_review_1050_20261014`.
  26 warning PEFT từ fixture; 304 file lịch sử không đổi hash, không đổi
  nhãn ngoài 300 ID. M2/M3 cùng 153 ID/chosen, 153 ảnh không trùng.
- Drive nhận gói nhãn 70 file, notes v8 và handoff md/json v9. Tải lại ZIP
  và notes khớp local; SHA-256 ZIP
  `edfbb1e40f0d39d6ee33b766793e081579686663b494a38f18b83819b39dfd9f`.
- Source 238 file `source_20261013_173643_954254` đã tải lại từ Drive khớp
  SHA-256 `43c379292db35863b44cc276275c3dd751950ea59662d1528818aa53485820a6`.
  Không có ảnh/weights trong source ZIP. Nhật ký hiện hành đồng bộ riêng
  sau snapshot source, không tạo thêm release chỉ để sửa trạng thái upload.
- Tài liệu giữ cùng file: research_plan v23, preference_annotation v16,
  preference_training v17, project_description v10. Kế hoạch tải lại khớp.
  Record launch v9 ghim source mới, tải lại khớp, vẫn `data_locked=false`.
  Inspect M2/M3 dừng trước tải model vì N/bộ cuối chưa khóa.
- Bằng chứng tải lại, bộ lọc trang và Git cuối lưu ở
  `artifacts/preference_review_1050_20261014/sync/receipt.json`.

## 13/10 — duyệt lượt 651–750

- Tiếp tục đúng 100 ID pending đã khóa; xem đủ bảng ảnh, mở lại ảnh
  gốc của toàn bộ cặp giữ. Giữ 15 cặp: 5 lỗi M1 nguyên văn và 10 có
  kiểm soát; 54 loại, 31 chưa chắc. Pilot và nhãn cũ không đổi.
- Một negative ván có cánh quạt quá phi thực tế được thay bằng ván trượt/
  xe đạp trước công bố. Bản nháp đầu giữ trong
  `artifacts/preference_review_750_20261013/initial_publish`; không xóa.
  Không tạo cặp từ màu trời hoặc cách diễn đạt cầm vợt và bóng mơ hồ.
- Bản hiện hành `data/preferences/review_batch_651_750_20261013`; audit
  2.000 dòng qua, cùng 110 ID/chosen cho M2/M3: 48 model / 62 synthetic.
  Còn 1.216 pending; 100 ID tiếp khóa trong continuation. Không đổi M1,
  split, LR hoặc epoch. M2/M3 chưa chạy GPU và chưa khóa bộ cuối/N.
- Trang kết quả `review_results_750_20261013.html`. Kiểm repo và đồng bộ
  đang thực hiện; không tạo notebook hoặc sửa output lịch sử.
- `check_project.py --full-data` qua Ruff, 910 test không skip, audit
  32.834 dòng và preflight 29.459 train. Audit
  `audit_20261013_162718_456774`, preflight `preflight_20261013_162754_649124`.
  Log ở `artifacts/preference_review_750_20261013/check_project.log`;
  26 warning PEFT từ fixture. 304 file lịch sử còn nguyên hash.
- Máy chủ xem ảnh local đã dừng sau đổi phiên; khởi động lại chỉ trên
  `127.0.0.1:8765`. Bộ lọc trang kết quả qua: 750 tổng, 100 mới, 110 giữ.
- Gói nhãn 54 file và notes v7 đã tải lại từ Drive, khớp local. ZIP SHA-256
  `5485b7d3158fb8cd09f074be01894d50f70ef95046832fd20db56227ee0055b0`.
  Handoff md/json v8; không đổi chia sẻ hoặc tạo bản notes trùng tên.
  Cấu hình inspect M2/M3 dừng trước tải model vì dữ liệu/N chưa khóa.
- Source 238 file `source_20261013_163213_805717` tải lại từ Drive khớp
  SHA-256 `bc2ab48554f68e844e9286caf93386b0b3415a26cbafb0714133f9e16d820923`.
  Nhật ký đồng bộ riêng sau snapshot ZIP; không tạo thêm source release.
- Tài liệu giữ cùng file Drive: research_plan v22, preference_annotation
  v15, preference_training v16, project_description v9. Kế hoạch tải lại
  khớp local. Record launch ghim source mới, vẫn `data_locked=false`.
  Kiểm tải lại nhật ký/record và Git lưu trong
  `artifacts/preference_review_750_20261013/sync/receipt.json`.

## 13/10 — duyệt lượt 551–650

- Tiếp tục đúng 100 ID pending đã khóa. Xem đủ ảnh qua bảng, mở lại ảnh
  gốc cho 16 cặp giữ: 6 lỗi M1 nguyên văn, 10 có kiểm soát; 56 loại,
  28 chưa chắc. Không đổi pilot, nhãn cũ, split hoặc cấu hình train.
- Bản hiện hành `data/preferences/review_batch_551_650_20261013`;
  audit 2.000 dòng qua, 95 cặp cùng ID/chosen cho M2/M3: 43 model,
  52 synthetic. Còn 1.316 pending; 100 ID tiếp đã khóa trong continuation.
- Trang `review_results_650_20261013.html` chỉ hiện kết quả, giữ nguồn
  trong metadata. Bộ cuối chưa chốt N; M2/M3 chưa chạy GPU. Không tạo
  notebook hoặc sửa output M1 cũ.
- `check_project.py --full-data` qua Ruff, 910 test không skip, audit
  32.834 dòng và preflight 29.459 train. Audit
  `audit_20261013_142046_160384`, preflight `preflight_20261013_142120_181576`.
  Log ở `artifacts/preference_review_650_20261013/check_project.log`;
  26 warning PEFT từ fixture. 304 file lịch sử còn nguyên hash.
- Bộ lọc trang kết quả qua: 650 tổng, 100 lượt mới, 95 cặp giữ.
- Gói nhãn 49 file và notes v6 tải lại từ Drive khớp local, ZIP SHA-256
  `0a856bf9a4aeba49e743b0f8925dcf4182ea3a263b6ddfd07b5dcf54467ee8db`.
  Handoff md/json v7; giữ đầy đủ recipe và nguồn, không chứa ảnh/weights.
- Source 238 file `source_20261013_142332_856372` đã tải lại từ Drive,
  khớp SHA-256
  `873e573fe1199ba07c2f1ffc6726f9d4d10e7a5425f7f154f9bf510e1b0485f8`.
  Snapshot ZIP giữ nguyên; tài liệu và nhật ký đồng bộ riêng sau kiểm.
  Cấu hình inspect M2/M3 dừng trước tải model vì bộ cuối/N chưa khóa.
- Tài liệu trên file Drive cũ: research_plan v21, preference_annotation
  v14, preference_training v15, project_description v8. Kế hoạch tải lại
  khớp local. Record khởi chạy ghim source mới, vẫn `data_locked=false`.
  Nhật ký và record được đồng bộ riêng sau snapshot source; bằng chứng
  kiểm tải lại/Git lưu tại
  `artifacts/preference_review_650_20261013/sync/receipt.json`.

## 13/10 — duyệt lượt 451–550

- Duyệt đúng 100 ID pending khóa từ lượt trước; lưu selection trước khi
  xem. Xem bảng đủ 100 ảnh, mở lại ảnh gốc cho toàn bộ cặp giữ. Chốt
  16 cặp: 6 lỗi M1 nguyên văn, 10 có kiểm soát; 47 loại, 37 chưa chắc.
- Không dùng đối lập đen/xám đậm khi màu lông còn nhạy với ánh sáng.
  Câu về quốc tịch, cảm xúc, tập tính, dinh dưỡng và thời điểm ảnh để
  uncertain nếu không thể xác nhận từ ảnh. Pilot và nhãn ngoài lượt không đổi.
- Bản hiện hành `data/preferences/review_batch_451_550_20261013`;
  audit đủ 2.000 dòng qua, 79 cặp chung ID/chosen cho M2/M3, gồm
  37 model / 42 synthetic. Chưa có nhãn người hay bộ train cuối.
- Trang `review_results_550_20261013.html` chỉ hiển thị kết quả; nguồn
  actor/text giữ trong JSONL/recipe. Còn 1.416 pending, 100 ID tiếp khóa
  trong continuation. Không train lại M1 hoặc tạo notebook mới.
- `check_project.py --full-data` qua Ruff, 910 test không skip, audit
  32.834 dòng và preflight 29.459 train. Run audit
  `audit_20261013_130751_741733`, preflight `preflight_20261013_130911_667068`.
  Log `artifacts/preference_review_550_20261013/check_project.log`;
  26 warning PEFT từ fixture. 304 file lịch sử còn nguyên hash.
- Gói nhãn 44 file giữ đầy đủ recipe/nguồn, không có ảnh hay weights.
  Gói tải lại từ Drive khớp SHA-256
  `7e6b411660c24b7065dac0d0b7f8d45be6b09d0d1f444dc1735896cd31508398`.
  Notes v5 tải lại khớp local; handoff md/json v6. Giữ cùng file Drive.
- Sau ngắt phiên, đã kết nối lại và cập nhật tài liệu trên file cũ:
  research_plan v20, preference_annotation v13, preference_training v14,
  project_description v7. Kế hoạch tải lại khớp local. Source 238 file
  `source_20261013_131527_902844` tải lại khớp SHA-256
  `ce631341efa4d6a7f6f0b2f350d2fc9938e10ceee3e62d068cd3977efe740ca5`.
  Nhật ký đồng bộ riêng sau snapshot ZIP, không tạo thêm source release.
- Record khởi chạy Drive v6 ghim source mới; hai nhánh inspect dừng
  trước tải model vì dữ liệu/N chưa khóa. Giữ `data_locked=false`, LR,
  epoch và batch như kế hoạch. Biên nhận ở
  `artifacts/preference_review_550_20261013/sync/receipt.json`.

## 13/10 — duyệt lượt 351–450

- Khóa 100 ID trước khi xem ảnh, lấy đúng `continuation.json` lượt trước.
  Xem đủ ảnh qua bảng, mở lại ảnh gốc cho các cặp giữ. Chốt 15 cặp:
  6 rejected M1 nguyên văn, 9 có kiểm soát; 58 loại, 27 chưa chắc.
- Rút trường hợp độ mềm chuối, hành động ăn cỏ và màu đĩa bay sau rà lại.
  Một cụm sân tennis đọc nhầm từ bảng ảnh bị audit chặn vì không có trong
  M1 gốc; đã loại trước export. Snapshot hai file của lần xuất lỗi giữ
  trong `artifacts/preference_review_450_20261013/failed_publish`.
- Bản hiện hành: `data/preferences/review_batch_351_450_20261013`.
  Audit 2.000 dòng qua; 63 cặp cùng ID/chosen cho M2/M3, gồm 31 model
  và 32 synthetic. Không đổi pilot hoặc nhãn ngoài lượt. 0 nhãn người.
- Trang kết quả `review_results_450_20261013.html`; ảnh, text M1 và các
  bản lịch sử giữ nguyên. Khóa 100 ID tiếp theo; còn 1.516 pending.
  M1 không chạy lại, M2/M3 chưa train hoặc khóa N. Không tạo notebook mới.
- `check_project.py --full-data` qua Ruff và 910 test, không skip;
  audit 32.834 dòng và preflight 29.459 train qua. Run audit
  `audit_20261013_115719_134769`, preflight `preflight_20261013_115841_377679`.
  Log ở `artifacts/preference_review_450_20261013/check_project.log`;
  26 warning từ fixture PEFT. 304 file lịch sử còn nguyên hash.
- Gói nhãn 39 file có đủ recipe/nguồn, không chứa ảnh hoặc weights.
  Drive `preference_local` nhận ZIP/sha, tải lại khớp
  `48aaa4570463e40b551573ef981301d3f8a1fc80ac27fa05cdc0d3e122caffd1`.
  Notes lên phiên bản 4, tải lại khớp local; handoff md/json v5.
- Tài liệu cập nhật cùng file Drive: research_plan v19,
  preference_annotation v12, preference_training v13, project_description v6.
  Kế hoạch tải lại khớp local. Source 238 file `source_20261013_120450_965965`
  tải lại khớp SHA-256
  `a26ab31c0285861582b0937e9320551430c8966ba17c16efa98772e330e264be`.
  Nhật ký được đồng bộ riêng sau snapshot source.
- Record khởi chạy Drive v5 ghim source mới, tải lại khớp local.
  Hai nhánh inspect dừng trước tải model vì bộ dữ liệu/N chưa khóa;
  `data_locked=false` và cấu hình LR/epoch/batch vẫn giữ nguyên.
  Biên nhận: `artifacts/preference_review_450_20261013/sync/receipt.json`.

## 13/10 — duyệt lượt 251–350

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
