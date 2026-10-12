# Gán nhãn cặp ưu tiên

Cập nhật 12/10/2026. Bộ chính là 2.000 ứng viên từ M1-3683, train-only.
Pilot cố định 100 ảnh có 13 nháp AI, chưa có cặp accepted.

Trang dùng hiện tại:
`artifacts/preference_editor_m1_3683_2000_20261008/pilot_review_qc_20261012.html`.
Các trang checkpoint-800 và bản nháp trước giữ làm lịch sử, không chạy lại.
Bản review gốc: `artifacts/colab_history/20261008/preference_candidates_completed/review_2000.zip`,
SHA-256 `365f080096f964a4a1d670faa7192a62697f45fb92f0480cdd555c7319b8af45`.

## Rubric

| Phần | Cách kiểm |
|---|---|
| Đáp án | correct/incorrect/uncertain theo ảnh; không chỉ dựa exact match |
| Hỗ trợ giải thích M1 | 0: sai/không liên quan; 1: một phần; 2: đủ bằng chứng liên quan |
| Chosen | Giải thích đúng, có bằng chứng hỗ trợ đáp án |
| Rejected | Có vấn đề cụ thể: mâu thuẫn, chi tiết không thấy, hoặc thiếu hỗ trợ đáp án |
| Quyết định | accepted: có cặp hợp lệ; rejected: không có cặp/lý do loại; uncertain: chưa xác định |

Hai completion giữ cùng ảnh, câu hỏi và đáp án đúng; chỉ khác giải thích.
Ghi cụm cần kiểm và bằng chứng nhìn thấy. Cặp chỉ khác văn phong/độ dài hoặc
hai phía đều đúng không dùng. Câu về cảm xúc, sở thích hay thuộc tính không
thấy thì giữ uncertain; không tạo bằng chứng đoán. Không loại câu khó khỏi test.

`error_type` dùng object, attribute, relation_action hoặc natural_model_error.
Lỗi thuật ngữ/diễn đạt ghi riêng; thiếu liên quan không tự được gọi là bịa
đối tượng. Tham chiếu nghi sai: ghi lý do loại, không sửa canonical giữa run.
Từ đồng nghĩa có thể đúng theo ảnh nhưng không được đổi luật EM sau khi thấy điểm.

## Duyệt và lưu

1. Xem ảnh, câu hỏi, đầu ra M1 và tham chiếu. Điền đánh giá theo rubric.
2. Kiểm/sửa nháp; ghi chosen/rejected, loại lỗi, cụm cần kiểm và bằng chứng.
3. Ghi mã người xác nhận thực tế, thời gian và quyết định. Mã ổn định có thể
   thay tên thật; không điền mã người cho nhận xét AI.
4. Xuất JSONL để lưu bền vững. Trang tự lưu nháp ở cùng trình duyệt/origin;
   khi đổi máy/cổng hoặc từ `file://` sang localhost, xuất rồi nhập lại.

Form có ba nhóm: đánh giá ảnh, cặp giải thích và quyết định. Nguồn/hash nằm
trong mục kỹ thuật thu gọn. Nút Điền đề xuất giữ pending và điền nguồn AI;
không tự xác nhận. Khi tự sửa, kiểm lại nguồn và recipe, không tự đặt hash.

Bộ pilot chỉ hiển thị 100 câu nhưng xuất đủ 2.000 dòng, giữ nhãn ngoài pilot.
AI đã rà 100 ảnh; lượt 12/10 mở lại 15 ảnh có nháp, giữ 9 sửa tự nhiên và
4 lỗi có kiểm soát. Bỏ cặp lồng/giỏ và khu rừng vì đối lập chưa đủ rõ;
85 nhận xét còn lại kế thừa, không gọi là lượt chấm độc lập mới. Xem
`artifacts/preference_ai_triage_m1_3683_pilot_20261012_qc`.
Gợi ý AI có thể gây thiên lệch neo; ghi rõ quy trình trong báo cáo.

## Truy vết văn bản

- Rejected giữ nguyên giải thích M1: nguồn model, không ghi thành manual.
- Rejected do người sửa: manual, giữ nguồn ứng viên M1 ban đầu.
- Negative có kiểm soát hoặc chosen AI được xác nhận nguyên văn: synthetic,
  lưu source và SHA-256 của recipe. Xác nhận không đổi tác giả thành người.
- Khi người tự viết chosen, dùng manual; không để mặc định manual cho văn
  bản AI được chép nguyên. Giữ cùng template, tránh chosen luôn dài hơn.

Exporter lưu `original_candidate_origin` kể cả khi sửa hai phía.
Các nguồn model phải khớp M1 được chọn; trainer kiểm lại fingerprint.
Mọi cặp accepted có hash canonical/ảnh, người xác nhận/timezone, bằng chứng,
đoạn rejected và provenance chosen/rejected. Đoạn lỗi phải có trong rejected.

## Audit → báo cáo pilot → export → freeze

Ví dụ dưới chạy từ repo sau khi lưu file xác nhận thật vào
`data/preferences/reviewed_notes.jsonl`. Hiện chưa có file đã chốt;
không thay bằng nháp để bắt đầu train. Dùng output version mới nếu đã tồn tại.

```powershell
$reviewDir = 'artifacts/preference_editor_m1_3683_2000_20261008'
$reviewNotes = 'data/preferences/reviewed_notes.jsonl'
$reviewPilot = "$reviewDir/pilot_review_100.json"

python scripts/audit_preference_review.py --review-dir $reviewDir --notes $reviewNotes --pilot $reviewPilot
python scripts/summarize_preference_pilot.py --review-dir $reviewDir --notes $reviewNotes --pilot $reviewPilot --triage artifacts/preference_ai_triage_m1_3683_pilot_20261012_qc/release/triage.jsonl --output-dir artifacts/preference_pilot_report_reviewed_v1 --target-pairs 1000

python scripts/export_preference_review.py --bundle artifacts/colab_history/20261008/preference_candidates_completed/review_2000.zip --expected-sha256 365f080096f964a4a1d670faa7192a62697f45fb92f0480cdd555c7319b8af45 --notes $reviewNotes --annotations data/processed/vivqax/image_disjoint_v1/all.jsonl --image-root data/raw/coco --output-dir data/preferences/export_v1
python scripts/validate_preferences.py --input data/preferences/export_v1/reviewed_pairs.jsonl --annotations data/processed/vivqax/image_disjoint_v1/all.jsonl --image-root data/raw/coco --output-dir data/preferences/frozen_v1
```

Audit kiểm đầy đủ trường và nguồn, không chứng minh nhãn thị giác đúng.
Resolved cần người/timezone; rejected/uncertain cần lý do; accepted cần hai
giải thích, đoạn lỗi, bằng chứng và nguồn. Pending có sửa nháp được đếm riêng.

Báo cáo pilot chỉ ước lượng sau khi đủ 100 quyết định hợp lệ. Báo tổng
accepted và nhóm rejected nguyên văn M1 riêng; không dùng negative tạo sẵn
để ngoại suy pool lỗi tự nhiên. Bảng chéo AI–người chỉ mô tả, không tính
agreement vì hai bộ nhãn khác semantics.

Sau pilot, chốt N/phạm vi trước train. 1.000 là dự toán, không bắt hạ chuẩn
để đủ quota. Có thể export accepted trong pool còn pending; giữ file notes
đầy đủ để truy vết. Không có accepted thì exporter/validator dừng.

Freeze xuất `accepted.jsonl`, `chosen_only.jsonl`, `manifest.json`: cùng
ID/ảnh/đáp án/chosen cho M2/M3. Kiểm train-only, hash, prompt/canonical,
trùng nguồn và nguồn văn bản. Schema-only `--input` không đủ để train.
Trạng thái `validated_metadata` không thay nghiệm thu chất lượng ảnh hoặc
GPU gate. Bước sau: [train M2/M3](preference_training.md).

## Tái tạo công cụ

Chỉ dùng khi cần bộ review mới; các artifact hiện hành đã có.

- `select_preference_questions.py`: shuffle câu seed 42, khử trùng theo
  ID/path ảnh; không phải lấy đều theo ảnh hoặc khử gần trùng nội dung.
- `generate_preference_candidates.py`: sinh từ M1 đã chọn và manifest;
  giữ raw output/model/revision/seed, không ép đáp án tham chiếu.
- `prepare_preference_review.py`: kiểm ZIP/hash và tạo editor.
- `attach_preference_drafts.py`, `build_preference_pilot.py`: gắn nháp và
  hàng đợi; không tạo nhãn người. Xuất nháp trước khi refresh trang.

Đánh giá cuối: [chấm mù](human_evaluation.md). DPO có thể ưu tiên văn bản
mà ít dùng ảnh; [mDPO](https://arxiv.org/abs/2406.11839) là cơ sở nêu rủi ro,
không phải một kiến trúc được thêm vào đồ án.
