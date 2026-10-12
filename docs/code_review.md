# Review thay đổi trong dự án

Code do người hoặc AI hỗ trợ viết đều cần kiểm theo cùng tiêu chí: thực hiện
đúng yêu cầu, có bằng chứng chạy và đủ rõ để người khác tiếp tục. Với repo
VQA NLE, ưu tiên dữ liệu/split, completion masking, reference DPO, resume và
cách chấm kết quả; số file hoặc số test chỉ mô tả quy mô kiểm tra.

## Luồng review

1. Đọc diff theo từng mục tiêu. Kiểm thay đổi có cần cho stage hiện tại,
   có trùng chức năng đã có và có làm sai hướng dẫn hoặc lineage của run cũ.
2. Chạy Ruff và bộ test local bằng `scripts/check_project.py`.
   Nếu đổi dữ liệu hoặc pipeline đầu vào, chạy thêm `--full-data` ở máy đủ ảnh.
3. Dùng test hành vi: dữ liệu sai bị từ chối, prompt/ảnh bị mask đúng,
   gradient cập nhật policy, reference giữ nguyên, resume khớp run liên tục,
   và nháp trình duyệt giữ đúng câu/nguồn khi nhập hoặc khôi phục.
4. Đọc cảnh báo scanner theo ngữ cảnh rồi ghi cách xử lý. Kiểm dependency và
   token trước khi xuất source. CI xác nhận kiểm tra trên môi trường sạch.
5. Review kết quả mô hình trên ảnh và mẫu đã định; ghi rõ bước nào chỉ kiểm
   code, bước nào đã chạy Qwen trên GPU và bước nào đã có nhãn người.

## Công cụ và vai trò

| Công cụ | Phần cần kiểm |
|---|---|
| Git diff và review nhánh/PR | Scope, lịch sử thay đổi, API, cấu trúc và giải thích quyết định |
| Ruff | Lỗi Python, import, các mẫu code dễ sai; nhóm S hỗ trợ rà bảo mật |
| pytest và CI | Hành vi, regression và khả năng chạy trên môi trường sạch |
| CodeQL hoặc Semgrep | Những mẫu và luồng dữ liệu có thể gây lỗi bảo mật |
| pip-audit | Lỗ hổng đã công bố trong dependency Python |
| Secret scanning | Token/API key lỡ đưa vào source hoặc lịch sử Git |

Ruff và pytest/CI đã tích hợp trong repo. CI CPU dùng Python 3.10/3.11,
dependency verification/demo và Node, với `--fail-on-skip`.
Các công cụ còn lại có thể bổ sung theo nhu cầu; chúng chưa được coi là đã
chạy chỉ vì có tên trong bảng này.

Lượt 12/10 đã chạy Ruff nhóm S trên `src/` và `scripts/`; báo cáo tại
`artifacts/code_cleanup_20261012/ruff_security.json`. Cảnh báo RNG áp dụng
cho seed lấy mẫu/bootstrap, subprocess dùng danh sách argv không qua shell,
và URL COCO được tạo từ base HTTPS cố định. Kết quả scanner cần review các
luồng tương ứng; tránh sửa RNG hoặc giao thức thí nghiệm chỉ để hết cảnh báo.

AI review có thể giúp tìm chỗ đáng nghi hoặc giải thích diff. Người thực hiện
đối chiếu từng nhận xét với dòng code, test và input thực tế trước khi sửa.
Các test chỉ tìm chuỗi trong source cần được bổ sung hành vi nếu chúng đang
bảo vệ một chức năng quan trọng.

## Tài liệu công cụ

- [Ruff linter](https://docs.astral.sh/ruff/linter/)
- [CodeQL trên GitHub](https://docs.github.com/en/code-security/concepts/code-scanning/codeql/codeql-code-scanning)
- [Semgrep CLI](https://docs.semgrep.dev/running-rules)
- [pip-audit](https://pypi.org/project/pip-audit/)
- [GitHub secret scanning](https://docs.github.com/en/code-security/concepts/secret-security/secret-scanning)
