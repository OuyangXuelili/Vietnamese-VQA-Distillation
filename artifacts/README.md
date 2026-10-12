# Tệp đầu ra và bản dùng hiện tại

Thư mục này lưu bằng chứng local. Git giữ mục lục này; source release, ảnh,
checkpoint, log và kết quả chạy được lưu riêng. Trạng thái mới nhất nằm trong
[TIEN_DO.md](../TIEN_DO.md).

| Mục đích | Vị trí hiện hành |
|---|---|
| Duyệt cặp M1 | `preference_editor_m1_3683_2000_20261008/pilot_review_qc_20261012.html` |
| Nhận xét AI đã kiểm lại | `preference_ai_triage_m1_3683_pilot_20261012_qc/` |
| Báo cáo quy mô pilot | `preference_ai_triage_m1_3683_pilot_20261012_qc/pilot_report/` |
| Source đóng gói | `releases/source_<UTC>/` |
| Hồ sơ chạy trên Colab | `colab_history/<ngày>/` |
| Word hiện hành | `deliverables/current/` |
| Rà và sửa source 12/10 | `code_cleanup_20261012/` |

`project_source.zip` ở gốc là snapshot ngày 18/09/2026. Để tiếp tục dự án,
dùng release đã xác minh trong `releases/` và đọc nhật ký để chọn đúng bản.
Môi trường M1 dùng release ghim của run M1; source mới dành cho các bước tiếp theo.

Các thư mục có hậu tố ngày, `v2`, `v3`, `corrected`, `qa` hoặc `rechecked`
lưu những lượt đã thực hiện. Giữ đường dẫn và byte của chúng để các hash và
liên kết provenance còn đối chiếu được. Khi mở lại một bước, tìm bản hiện hành
trong bảng trên trước; các bản khác phục vụ truy vết.
