# Huấn luyện M2/M3

M1-3683 đã được chọn và có đủ prediction validation. Pilot đã chốt; bộ cặp
đang mở rộng có 153 cặp, chưa chốt cho run chính. Cần chốt N và đóng băng
bộ cuối. Dữ liệu duyệt bằng AI giữ actor rõ trong metadata, không ghi
thành nhãn người. Trạng thái tại
[TIEN_DO.md](../TIEN_DO.md); cách duyệt/export ở
[preference_annotation.md](preference_annotation.md).

## Chuẩn bị đầu vào

- `accepted.jsonl`, `chosen_only.jsonl`, `manifest.json` từ cùng bộ đã freeze.
- Adapter M1-3683 và selection record đã ghim model/revision và hash.
- Sáu evidence của selection: kết quả epoch, Trainer state, loss selection,
  prediction validation, manifest prediction và annotation validation.
- Profile environment trỏ tới ảnh/annotation đầy đủ và nơi lưu run.

Đường dẫn evidence được tính từ selection record. Bản tải ở
`artifacts/colab_history/20261007/m1_selection_frozen_v2` giữ layout Drive gốc;
cần khôi phục đủ layout/evidence trước khi dùng làm đầu vào trainer trên máy khác.
Demo replay có YAML trỏ trực tiếp ba evidence cần dùng.

Chốt `expected_pairs=N` cho các config sẽ chạy. Hai nhánh dùng cùng ID,
ảnh, đáp án và chosen. Sau preflight CPU, mỗi trainer chạy gate GPU tối đa
10 update, rồi run chính nạp lại M1 sạch.

`scripts/finalize_preference_review.py` bàn giao bản duyệt bằng một lệnh:
audit/báo cáo trước, export/freeze chỉ khi có `--freeze` và N chính xác.
Đọc `handoff.md`, kiểm manifest và dùng thư mục `frozen` làm preferences.
Lệnh không cập nhật config, chọn N hoặc chạy GPU thay người thực hiện.

## Trainer và môi trường mới

Logic ở `training/preference.py`, input/collator ở `training/preference_data.py`.
CLI `scripts/train_preference.py` đọc config và chạy trainer.

Notebook `04_train_preferences.ipynb` có 7 cell: kho lưu, source, preflight,
M2, M3, kết quả và một phần mô tả ngắn. `scripts/launch_preference_training.py` chuyển
record khởi chạy thành tham số CLI trên; không chứa loss hoặc trainer mới.
Hai cell M2/M3 gọi cùng helper, tránh chép lại phần dựng lệnh.

Record `source/preference_training_launch_v1.json` giữ đường dẫn môi trường,
preferences, M1/selection, hai config và `run_group`. Mặc định chưa khóa
dữ liệu; 153 cặp hiện tại không mở train chính. Chỉ đặt `data_locked=true`
sau khi chốt N/bộ cuối và cập nhật hai config khớp N.
Runtime mới khôi phục bundle ảnh vào ổ `/content` bằng verifier hiện có;
record ghim đường dẫn/hash bundle. Ảnh của bộ cặp tiếp tục được đối chiếu
trong preflight. Không đọc hàng nghìn JPEG trực tiếp từ Drive.

Trong notebook: `MODE='inspect'` chỉ kiểm metadata; `'gate'` chạy 5 update;
`'train'` cần kết quả gate cùng nhánh. Run group không đổi khi chạy lại
cell cấu hình, nên bước train đọc đúng gate đã lưu. Chọn nhóm run mới khi
đổi source/dữ liệu, không ghi đè output cũ. Resume truyền checkpoint qua
`RESUME`, đặt `ATTEMPT` của nhánh đó thành tên mới (ví dụ `resume25`).
Output mới nằm cùng nhóm run; gate gốc vẫn được dùng. Không lặp setup M1.

M2 chính dùng `configs/training/sft_chosen.yaml`; M3 dùng
`configs/training/dpo.yaml`, cùng LR 5e-6. Bản M2 2e-5 giữ làm thiết kế cũ,
chưa train. Protocol sửa ngày 12/10 trước run để tránh nhiễu do khác LR.
CLI mặc định đọc `study_budget_v2.yaml`, kiểm hai nhánh và lưu hash trong
run. Bản cũ bị từ chối nếu khai là nhánh chính. Nếu thay thiết kế, cần file
study mới qua `--study-config`, không đổi protocol sau khi thấy test.

Kiểm cấu hình, không cần GPU/dữ liệu:

```powershell
python scripts/train_preference.py --config configs/training/sft_chosen.yaml --check-comparison
```

Sau khi chốt N, cập nhật `expected_pairs` ở hai config chính rồi kiểm lại.
Lệnh chạy M2 dùng config matched-LR, M3 dùng dpo; cả hai cần gate riêng.

- M2 nạp **trọng số adapter M1**, không khởi tạo LoRA ngẫu nhiên mới.
- M3 nạp hai bản adapter M1 trên cùng backbone 4-bit: policy trainable,
  reference frozen. Reference không phải model gốc tắt adapter.
- Cả hai tắt dropout, cùng chosen và thứ tự dữ liệu, seed 42, một epoch,
  batch 1 × tích lũy 8. Với 1.000 cặp: 125 optimizer update mỗi nhánh.
- DPO sigmoid dùng tổng log-prob của completion, mask prompt/ảnh; dùng ảnh ở
  chosen, rejected và reference. Không có reward model, rollout RL hoặc cache ref.
- Giới hạn token là cổng từ chối, không cắt chuỗi. Quét collator thật một lần
  trước train. DPO có `max_length=null`, guard cả hai phía ở 1536 token.
- Dtype ở config train có hiệu lực: auto = FP16 trên T4, BF16 chỉ khi native.
  Dtype legacy trong model YAML không điều khiển hai stage mới.

Trong **runtime GPU mới**, cài `requirements-cloud-preference.txt`, chạy
`pip check`, ghi phiên bản và GPU. File này ghim các thư viện chính; không phải
lock toàn bộ phụ thuộc bắc cầu hoặc bằng chứng đã chạy GPU. CUDA wheel/runtime
còn phải xác nhận trong cổng GPU. Torch/Torchvision ghim bản `+cu128`;
đã kiểm wheel Python 3.13/Linux trong [index PyTorch chính thức](https://download.pytorch.org/whl/cu128/torch/).
Không thay môi trường M1.

Ví dụ sau khi có bundle và selection thật, tại repo:

```powershell
python scripts/train_preference.py --config configs/training/dpo.yaml --environment configs/environments/local_full.yaml --preferences <FROZEN_BUNDLE> --m1-adapter <M1_ADAPTER> --m1-selection <SELECTION_JSON> --preflight-only
```

Lệnh này chỉ kiểm metadata và ảnh; không tải model, không train. Trên cloud
dùng profile environment cloud riêng. Nếu thiếu accepted/selection, dừng;
không thay bằng sample JSONL để qua cổng.

Exporter lưu `original_candidate_origin` cho mọi cặp, kể cả bản người sửa hoặc
synthetic. Trước train, model ID và fingerprint adapter của nguồn này phải khớp
M1 được chọn. Nguồn `model` ở chosen/rejected cũng phải khớp. Bundle cũ thiếu
trường này cần export lại từ đúng candidate manifest; không tự điền hash M1.

Mỗi trainer mới chạy một GPU gate tối đa 10 update (`--pilot-steps 5`, output
mới), kiểm loss, policy update, reference giữ nguyên và byte weights đã lưu.
Run chính bỏ pilot-steps, thêm `--gpu-gate <GATE_RUN/training_result.json>`.
Khởi lại từ M1 sạch, không dùng adapter gate. Hash source/data/config/M1,
runtime và dtype phải khớp. DPO dự toán vượt 8 giờ thì xem lại ngân sách chung.

Run mới ghi data_plan, execution, Trainer log/state, checkpoint mỗi 25 update
và ở đích cuối, adapter policy và training_result. Giữ các checkpoint đã lưu.
Mỗi checkpoint có `resume_manifest.json` ghi hash adapter, optimizer, scheduler,
RNG, scaler FP16 và các file khác. Marker chỉ ghi sau khi Trainer lưu xong.

Khi Colab mất VM, dùng `--resume-from-checkpoint <CHECKPOINT>` và một
`--output-dir` mới. Nguồn, dữ liệu, M1, config, phiên bản thư viện, dtype,
loại GPU và ngân sách phải khớp. Checkpoint mất file hoặc khác hash bị từ chối
trước tải model. Trainer khôi phục trạng thái và bỏ qua dữ liệu đã học; reference
được nạp lại từ M1 cố định, policy nạp từ checkpoint. Resume không tăng epoch.

`--stop-after-step <N>` dừng chủ động và lưu ở N, giữ lịch LR/đích gốc;
trạng thái là `paused`, chưa phải kết quả chính. Chỉ nối checkpoint nhỏ hơn
đích gốc. Gate GPU bị dừng chủ động chưa được coi là gate passed.
Các ô train M1 cũ không dùng CLI này. Run cuối ghi số update của phiên hiện
tại riêng với global step; đọc loss theo `log_history` khi đã resume vì
`Trainer.train_loss` không phải trung bình đầy đủ của các phiên trước.
CPU tests và toy training không xác nhận Qwen 3B/QLoRA chạy vừa T4.

## Cơ sở thiết lập

[NLX-GPT](https://github.com/fawazsammani/nlxgpt) tách dữ liệu, train và đánh giá
VQA-NLE. Repo này dùng cùng nguyên tắc, không lấy architecture hay kết quả của họ.
[TRL 0.29.0 DPO source](https://github.com/huggingface/trl/blob/v0.29.0/trl/trainer/dpo_trainer.py)
là đối chiếu cho completion masking và loss; không dùng nguyên trainer vì có
cast QLoRA trainable sang BF16. Bản 0.28 được kiểm chưa chuyển `image_grid_thw`
trong đường forward DPO. Vì vậy dùng Transformers Trainer và loss chuẩn DPO
riêng, giữ lại collator Qwen đã kiểm. Đây là lựa chọn triển khai, không phải thuật toán mới.

Chưa train M2/M3, chưa tạo nhãn người. Trạng thái đồng bộ đọc từ TIEN_DO.
Công cụ chấm test theo [hướng dẫn](human_evaluation.md); demo còn cần hoàn thiện.
