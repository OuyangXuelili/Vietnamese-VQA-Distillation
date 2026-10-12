# Quy ước ghi nhận thí nghiệm

Mỗi run phải lưu:

- experiment ID;
- Git commit;
- model ID và revision;
- dataset manifest checksum;
- seed;
- cấu hình train và generation;
- log, checkpoint và prediction.

Không chọn checkpoint hoặc thay rubric dựa trên test.

Đây là danh sách tối thiểu. Cổng kiểm tra M1 và cách so M0/M1 được mô tả cụ thể
ở [kế hoạch](research_plan.md) và [comparison_gate.md](comparison_gate.md).

## Giao thức nghiên cứu — cập nhật 12/10/2026

Phần dưới là thiết kế đánh giá, không phải kết quả M2/M3. Evaluator có
EM, Token-F1, ROUGE-L, BERTScore, kiểm giao thức và paired image-bootstrap.
Trainer M2/M3 đã có bản local và kiểm thử resume trên fixture CPU; chưa kiểm
GPU Qwen/QLoRA. Công cụ chấm ẩn tên model đã có, mẫu 200 ảnh đã cố định;
chưa có prediction test hoặc nhãn người chấm. Xem
[thiết lập giai đoạn mới](preference_training.md) và
[hướng dẫn chấm người](human_evaluation.md).
Đối chứng hiện hành là `vqa_nle_matched_lr_v2` trong [kế hoạch](research_plan.md).
Quy tắc chọn M1 và run một epoch lịch sử giữ nguyên.

### Các nhánh và ngân sách

- M0: Qwen gốc; M1: SFT đáp án kèm giải thích.
- M2: từ M1, SFT tiếp trên chosen của bộ preference đã duyệt.
- M3: từ cùng M1, DPO trên đúng các cặp đó, reference model cố định là M1.
- M2/M3 dùng cùng danh sách câu hỏi, cùng số lượt đi qua chosen, effective
  batch, prompt, độ phân giải và cách chọn checkpoint. Báo cáo riêng số update,
  token và giờ GPU: DPO còn đọc rejected/reference, nên cùng epoch không có
  nghĩa cùng chi phí. Không tuyên bố đối chứng compute-matched nếu chưa làm.
- Ngân sách chọn learning rate/beta trên validation phải ghi trước khi chạy;
  lưu cả các cấu hình không tốt. Không chọn bằng điểm test. Nếu đủ tài nguyên,
  chạy ba seed cho M2/M3; nếu một seed, ghi rõ kết quả chưa đo độ ổn định train.
- M1 đã hoàn tất một epoch, 3.683 update; chọn checkpoint-3683 bằng loss
  trên 256 validation cố định và kiểm prediction trên đủ 1.407 câu.
  Selection đã đóng băng; giữ nguyên source, dtype, run và checkpoint cũ.
  Giao thức tái lập ở [sft_full.md](sft_full.md), không chạy lại để bắt đầu M2/M3.
  M2/M3 chốt **checkpoint cuối sau một epoch** cho cả hai nhánh, không chọn
  lại bằng test hoặc so minimum DPO loss với minimum SFT loss. Một seed 42
  cho vòng chính; M2/M3 cùng LR 5e-6, M3 beta 0,1, effective batch 8.
  Chốt ngày 12/10 trước cả hai run; chưa gọi là siêu tham số tối ưu.
  Tắt dropout ở cả hai nhánh, cùng preprocessing ảnh xác định; reference frozen.
  Không grid search mặc định. Nếu thêm thử nghiệm chất lượng, ghi sửa protocol
  và ngân sách trước khi chạy; mọi lựa chọn chỉ dựa validation.

### Điểm tự động

1. Đáp án: normalized exact match và whitespace Token-F1 theo code repo.
   Không gọi chúng là VQA consensus accuracy: annotation hiện có một đáp án
   chuẩn, không phải 10 phiếu đáp án của VQA v2.
2. Giải thích: ROUGE-L theo chính tokenizer/reference aggregation của repo;
   chỉ đo giống từ. Bảng M0–M3 cuối bổ sung BERTScore làm thước đo ngữ nghĩa
   phụ, dùng `google-bert/bert-base-multilingual-cased` tại commit
   `3f076fdb1ab68d5b2880cb87a0886f315b8146f8`, layer 9, `bert-score==0.3.13`,
   `transformers==4.57.6`, không IDF/baseline rescaling/fast tokenizer. Tính
   đủ mọi ID và mọi reference; không kết luận bám ảnh từ hai điểm văn bản này.
3. Báo cáo cả tỷ lệ sai định dạng, câu rỗng và phân bố độ dài. Không âm thầm
   loại các output lỗi khỏi mẫu số. Lỗi chạy hệ thống phải hoàn tất/retry có
   log, không thay bằng câu dự đoán giả.
4. Test cuối: 1.968 câu cùng ID cho mọi nhánh, sau khi đóng băng cấu hình.
   So với bài công bố khác chỉ mang tính tham khảo nếu split, backbone,
   chuẩn hóa đáp án hoặc evaluator khác; không tuyên bố vượt họ từ bảng lệch
   giao thức. Split validation của repo đã thay đổi so với ViVQA-X gốc.

### Đánh giá bằng người trên ảnh

- Thử rubric với 30 ảnh validation, chỉnh hướng dẫn rồi đóng băng. Các mẫu này
  không thuộc tập đánh giá cuối. Không xem output test trong lúc sửa rubric.
- Mục tiêu 200 ảnh test khác nhau, chọn ngẫu nhiên có seed trước khi xem
  prediction; mỗi ảnh chọn một câu hỏi. Dùng đúng các câu đó cho M0–M3.
  Đây là mục tiêu khả thi ban đầu, không phải tính toán power hay bảo đảm
  phát hiện mức cải thiện nhỏ. Báo cáo loại câu hỏi và số mẫu mỗi nhóm.
- Hai người chấm độc lập, ẩn tên model, xáo thứ tự output; người chấm nhìn ảnh,
  câu hỏi, đáp án và giải thích. Không cho một LLM chấm làm bằng chứng duy nhất.
- Ghi riêng: (a) đáp án đúng/sai/mơ hồ; (b) giải thích có hỗ trợ đáp án không,
  thang 0/1/2; (c) có chi tiết không được ảnh hỗ trợ không, có/không/không đủ
  bằng chứng. Rubric ví dụ: 0 sai/không liên quan/rỗng, 1 đúng một phần nhưng
  chưa giải thích đủ, 2 có bằng chứng liên quan và đủ hỗ trợ đáp án.
- Giữ nhãn mơ hồ và báo cáo tỷ lệ, không ép thành đúng/sai. Ghi các lỗi object,
  attribute, relation/action và chỉ lặp lại đáp án. Lưu nhãn độc lập trước khi
  thảo luận bất đồng, báo cáo agreement (weighted kappa cho điểm thứ bậc,
  kappa cho nhãn phân loại) và quy tắc phân xử.
- Báo cáo chất lượng giải thích trên toàn mẫu và trên nhóm đáp án đúng của
  từng model, luôn kèm mẫu số. Thêm tỷ lệ **đáp án đúng và giải thích đạt 2**
  trên toàn mẫu để tránh chọn riêng các câu dễ. Không chỉ chấm giao tập mà mọi
  model đều trả lời đúng. Đây là cách đánh giá thích nghi cho đồ án, không
  tuyên bố tái lập nguyên xi human protocol của e-ViL.
- Chênh lệch M3–M2 là so sánh chính. Ước lượng khoảng tin cậy 95% bằng paired
  bootstrap theo ảnh, 2.000 lần với seed cố định; với tự động nhiều câu/ảnh,
  lấy mẫu cả cụm ảnh và giữ mọi câu trong cụm. Khoảng tin cậy này phản ánh
  bất định tập đánh giá, không thay thế nhiều seed huấn luyện.
- Chỉ số chính là điểm hỗ trợ 0–2 trên toàn mẫu. Chỉ số bổ sung gồm tỷ lệ
  đúng đáp án kèm hỗ trợ 2 và tỷ lệ chi tiết không được ảnh hỗ trợ. Tỷ lệ
  `uncertain` giữ riêng cùng mẫu số; `yes` giảm không tự chứng minh chất lượng
  tăng nếu câu mơ hồ tăng. CI từng chỉ số không hiệu chỉnh nhiều so sánh.

Nếu không có người chấm thứ hai, vẫn lưu đánh giá một người nhưng nêu rõ
giới hạn, không báo độ đồng thuận giả. Mục tiêu là lời giải thích phù hợp với
ảnh và câu trả lời, không phải chứng minh mô hình suy luận nội bộ như văn bản
nó sinh ra. Cơ sở: [e-ViL](https://arxiv.org/abs/2105.03761).

### Sửa đối chứng trước train — 12/10

M2 chính chuyển sang `sft_chosen_matched_lr.yaml`, LR 5e-6 như M3. Bản 2e-5
ở `sft_chosen.yaml` là kế hoạch chưa thực thi, giữ để truy vết, không được
ghép với kết quả chính. Không thay M1, split, candidate hoặc test.

CLI đọc phần training của `study_budget_v2.yaml`, kiểm hai nhánh có cùng
LR, N, seed, epoch, batch, dropout, pixels, dtype và token guard. Config
không khớp bị từ chối trước đọc bundle/tải model; hash cả hai config và
study được lưu trong data plan/resume identity. Các mục baseline trong YAML
là lịch sử, không phải lệnh tái chạy M1. Khi chốt N, sửa cả hai nhánh trước gate.

Cùng LR giảm một nguồn nhiễu, không đồng nghĩa cùng FLOPs, gradient hoặc
token giám sát. Kết quả vẫn giới hạn trong một backbone, seed và bộ cặp.
