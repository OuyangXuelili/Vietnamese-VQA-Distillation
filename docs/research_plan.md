# Kế hoạch thực hiện

Cập nhật 14/10/2026. Đề tài: **Nghiên cứu cải thiện chất lượng lời giải thích
trong hỏi đáp hình ảnh tiếng Việt**.

Câu hỏi nghiên cứu: từ cùng M1 và cùng bộ chosen, DPO có giúp giải thích được
ảnh hỗ trợ hơn SFT bổ sung không? Hai nhánh chính cùng LR 5e-6, seed, dữ liệu
chosen và epoch. DPO còn đọc rejected/reference nên chi phí tính vẫn khác.

M0–M3 là bốn cấu hình của **một backbone Qwen**, không phải bốn kiến trúc.
Đồ án hiện kiểm phương pháp học trên Qwen; chưa kiểm khả năng khái quát sang
backbone khác. Có tiền lệ đồ án môn học dùng một backbone để so các phương
pháp, nhưng không suy từ đó ra điểm hoặc chuẩn chấm của trường mình.
[Đối chiếu đồ án sinh viên](related_repositories.md#đồ-án-môn-học-quốc-tế--12102026).

## Trạng thái

| Phần | Đã xong | Còn lại |
|---|---|---|
| Dữ liệu | ViVQA-X `image_disjoint_v1`: 29.459 train, 1.407 validation, 1.968 test; đã tách ảnh | Giữ nguyên split |
| M0/M1 | M1 một epoch, 3.683 update; chọn checkpoint-3683; đủ validation M0/M1 | Đánh giá test sau khi khóa thiết kế |
| Preference | Xử lý hết 2.000 ứng viên; khóa 186 cặp sau kiểm cuối | Khôi phục bundle thật trên cloud |
| M2/M3 | Trainer/resume local, bộ cặp đã đóng băng | GPU gate và train thật |
| Đánh giá cuối | Scorer và mẫu cố định 200 ảnh test | Prediction bốn nhánh và nhãn chấm |
| Demo | Replay M1 và backend live đã viết | Nghiệm thu suy luận ảnh mới trên GPU |

Bằng chứng và trạng thái đồng bộ: [TIEN_DO.md](../TIEN_DO.md).
Kết quả M0/M1: [báo cáo validation](../reports/generated/m0_m1_validation_20261011/README.md).
Tên protocol chọn M1 là `validation_loss_full_validation_v2`: loss được đo
trên 256 câu cố định, còn prediction được kiểm trên đủ 1.407 validation.

## 1. Bộ cặp đã khóa

[Bộ dùng train](../artifacts/preference_editor_m1_3683_2000_20261008/review_results_final_20261014.html)
gồm 186 cặp trên 186 ảnh train. Hồ sơ:
`data/preferences/vivqax_m1_3683_v1/decision_record.json`; đầu vào ở
`frozen/accepted.jsonl`, `chosen_only.jsonl` và `manifest.json`.

Đã xử lý hết pool 2.000 ứng viên: xem 1.966 ảnh, tách 34 câu theo phạm vi.
Notes có 201 giữ / 1.106 loại / 693 chưa chắc, không còn pending.
Rà lại toàn bộ 201 cặp và rút 15 cặp chưa có đối lập chắc chắn khỏi bản
train. `quality_review.json` ghi từng ID/lý do; không sửa pilot hoặc notes cũ.
Lịch sử từng lượt và bản nháp giữ tại [TIEN_DO.md](../TIEN_DO.md).

Bộ cuối có 78 rejected M1 nguyên văn và 108 negative có kiểm soát.
Chosen và quyết định do cùng trợ lý AI tạo/kiểm; không phải nhiều người
độc lập hoặc nhãn người. Giao diện chỉ hiện kết quả, metadata giữ nguồn.
Phần chấm mù 200 ảnh test bởi người vẫn giữ nguyên.

N=186 thay dự toán 1.000 trước khi train M2/M3 hoặc xem test. Không tạo
cặp yếu để đủ quota; không tăng số mẫu bằng lặp dữ liệu. Đây là tập
preference nhỏ và được chọn lọc, nhiều đối chứng object/attribute dễ.
Kết luận phải giới hạn trong bộ này, không suy rộng toàn ViVQA-X hay
backbone khác. Báo nguồn model/synthetic riêng; kiểm độ dài để xem
shortcut. Không gọi toàn bộ nhóm model là bịa đối tượng.

Quy tắc cặp: cùng ảnh/câu hỏi/đáp án đúng, chosen hỗ trợ đáp án và rejected
có vấn đề xác định được theo ảnh. Không dùng câu mơ hồ, hai phía đều đúng
hoặc chỉ khác văn phong. Tham chiếu nghi sai được loại, không sửa canonical.
Phản hồi AI và negative tổng hợp có tiền lệ ở
[RLAIF-V](https://arxiv.org/abs/2405.17220) và
[POVID](https://arxiv.org/abs/2402.11411); dự án không tái hiện các pipeline đó.

## 2. Kiểm GPU rồi train M2/M3

Dùng CLI `scripts/train_preference.py`; [lệnh và môi trường](preference_training.md).
Không train lại M1 hoặc sinh lại 2.000 ứng viên.

| Nhánh | Khởi đầu / mục tiêu | LR | Ngân sách |
|---|---|---:|---|
| M2 | M1-3683; SFT trên chosen | 5e-6 | Một epoch, batch hiệu dụng 8 |
| M3 | Cùng M1; DPO chosen/rejected; reference M1 frozen | 5e-6, beta 0,1 | Cùng N và số epoch |

Cả hai tắt dropout, giữ cùng seed 42, preprocessing và template. DPO có ảnh
ở policy/reference và loss chỉ trên completion; không dùng reference là
backbone đã tắt adapter. Giới hạn 1.536 token là cổng từ chối, không cắt ảnh
hoặc bỏ riêng mẫu dài cho một nhánh. N=186 tương ứng 24 update mỗi nhánh.

Protocol `vqa_nle_matched_lr_v2` chốt ngày 12/10 trước khi M2/M3 chạy GPU.
M2 chính dùng `configs/training/sft_chosen.yaml`; bản 2e-5 giữ trong
`configs/archive/sft_chosen_lr2e5.yaml`. CLI kiểm cả hai cấu hình và lưu
protocol/hash vào run; đổi một nhánh phải khai báo protocol mới trước train.

1. Hai config đã chốt `expected_pairs=186`; kiểm bundle/M1/hash trên runtime thật.
2. Mỗi trainer kiểm GPU tối đa 10 update, một lượt khi đầu vào/source chưa
   đổi: loss hữu hạn, policy đổi, reference giữ nguyên, save/reload và resume.
3. Đo giây/update, VRAM, lưu checkpoint; dự toán DPO trên 8 giờ thì xem lại
   ngân sách chung trước run chính. Không lấy tốc độ CPU/SFT cũ làm số đo DPO.
4. Nạp lại M1 sạch để chạy chính. Lưu mỗi 25 update và cuối epoch; khi mất
   phiên nối đủ optimizer/scheduler/RNG/scaler bằng output mới.

Điều kiện xong: hai run `passed`, đủ update/epoch, adapter load lại được và
metadata khớp bundle/M1. Không dùng adapter gate làm điểm khởi đầu run chính.

## 3. Đánh giá và phân tích

- Xuất đủ 1.407 validation cho M2/M3; dùng lại M0/M1 khi manifest khớp.
- Khóa adapter, prompt và generation config, sau đó xuất đủ 1.968 test cho
  bốn nhánh. Không chọn checkpoint hoặc sửa rubric theo test.
- Chấm mù mẫu 200 ảnh đã đóng băng, một câu/ảnh. Người chấm nhìn ảnh và
  output, không thấy nhãn model; một người thì không báo agreement hai người.
- Chỉ số chính: chênh lệch hỗ trợ 0–2 M3–M2, paired image-bootstrap
  2.000 lần, seed 42, CI 95%.
- Chỉ số bổ sung: đúng đáp án kèm hỗ trợ 2, chi tiết không được ảnh hỗ trợ
  và uncertain; giữ toàn mẫu trong mẫu số. EM/F1 theo dõi đáp án,
  ROUGE-L/BERTScore đo tương đồng văn bản, không chứng minh bám ảnh.
- Phân tích object, attribute, relation/action, thiếu bằng chứng và lỗi
  diễn đạt. Báo kết quả âm, cỡ mẫu từng nhóm và ví dụ chọn theo quy tắc.

Điều kiện xong: bảng đủ ID, CI, nhãn chấm thật và phân tích trả lời câu hỏi
nghiên cứu. CI phản ánh bất định mẫu, không thay nhiều seed train.
Chi tiết: [giao thức](experiment_protocol.md), [công cụ chấm](human_evaluation.md).

## 4. Demo và báo cáo

Làm song song với duyệt cặp. [Demo](demo.md) nhận ảnh/câu hỏi mới ở chế độ GPU;
replay chỉ xem đầu ra đã lưu. Local không tải Qwen hoặc mount Drive.

- Kiểm live trên ít nhất ba ảnh/câu mới; ghi identity, latency/VRAM, reload
  và lỗi runtime. Không công khai endpoint không xác thực.
- Chỉ thêm lựa chọn M2/M3 khi có adapter hoàn tất.
- Báo cáo gồm dữ liệu/split, SFT–DPO, cách tạo cặp, thiết kế đối chứng,
  bảng kết quả, phân tích lỗi, giới hạn và cách tái lập.
- Đóng góp là quy trình dữ liệu có truy vết và nghiên cứu thực nghiệm;
  Qwen/LoRA/DPO là phương pháp kế thừa. Không tuyên bố thuật toán mới hoặc
  giải thích phản ánh phép tính nội bộ của mạng.

## Thử nghiệm phụ

Ưu tiên bốn nhánh chính, không thêm backbone trước khi có kết quả M2/M3.
Đối chứng LR đã đưa vào bốn nhánh chính; không cần train thêm M2 chỉ để
sửa khác biệt LR. Nếu có thêm ngân sách, chỉ chọn một kiểm tra bổ sung có
câu hỏi rõ, sau khi hoàn tất M2/M3 và bộ đánh giá chính.

- **Backbone tham khảo:** có thể đánh giá zero-shot
  [Vintern-3B-beta](https://huggingface.co/5CD-AI/Vintern-3B-beta) trên cùng
  validation tiếng Việt. Chưa triển khai hoặc đo VRAM. Chốt revision, xem
  custom code và preprocessing trước chạy; báo bảng riêng, không dùng cổng
  adapter-only vốn yêu cầu cùng backbone. So Qwen gốc với Vintern gốc,
  không dùng Vintern chưa tinh chỉnh để chứng minh hiệu quả riêng của DPO.
  Một baseline ngoài không chứng minh DPO khái quát sang kiến trúc khác.

- **Chuyển miền:** OpenViVQA là ứng viên khảo sát 100 ảnh ngoài, không thay
  test ViVQA-X. Dev đã kiểm metadata: 1.070 ảnh, 3.545 QA, không có nhãn
  giải thích/loại câu. Test công khai không có đáp án.

Trước phép kiểm ngoài: kiểm quyền dùng, phạm vi không chuyên OCR, ảnh thiếu
và trùng/gần trùng với dữ liệu chính; lưu namespace riêng. Ghi đúng source
split dev, chốt ID/seed trước output và không dùng để tuning. Không tạo
rationale GPT làm gold hoặc báo ROUGE/BERTScore khi không có nhãn giải thích.
Đánh giá ngoài cần đường riêng, không thêm rationale giả để qua CLI ViVQA-X.
Đây là khảo sát bổ sung, chưa xác nhận không trùng pretraining.

## Quy ước thực hiện

Code dùng lại ở `src`, CLI ở `scripts`, cấu hình ở `configs`; notebook 04
chỉ điều phối M2/M3, mặc định kiểm metadata. Lịch sử/output M1 được lưu
riêng, không thêm cell lịch sử vào luồng đang dùng. Bản kiểm riêng 31 cặp
chưa chạy được đưa ra khỏi luồng chính; chốt bộ cuối trước kiểm GPU.
Kiểm `scripts/check_project.py` trước release; full-data chỉ khi đổi
pipeline đầu vào. Giữ source/checkpoint/run cũ, xác nhận upload rồi mới ghi
đã đồng bộ. Việc đã làm và hash nằm trong TIEN_DO, không tạo note tiến độ mới.

M1 và ứng viên đã xong; thời gian còn lại phụ thuộc công duyệt, N và số đo
GPU. Không cộng ETA từ smoke cũ hoặc cam kết phiên GPU miễn phí liên tục.

Thiết kế trước lượt rút gọn được giữ trong source release
`source_20261011_165719_142146` trên Drive. Giao thức M1 lịch sử ở
[sft_full.md](sft_full.md), không phải lệnh cần chạy tiếp.

## Nguồn

[ViVQA-X](https://github.com/duongtruongbinh/ViVQA-X),
[DPO](https://arxiv.org/abs/2305.18290),
[e-ViL](https://arxiv.org/abs/2105.03761),
[mDPO](https://arxiv.org/abs/2406.11839),
[VINLE-GRPO](https://github.com/T-Sunm/VINLE-GRPO) và
[OpenViVQA](https://huggingface.co/datasets/uitnlp/OpenViVQA-dataset/blob/main/README.md).
