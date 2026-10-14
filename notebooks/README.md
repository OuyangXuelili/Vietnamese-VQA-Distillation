# Notebook đang dùng

[04_train_preferences.ipynb](04_train_preferences.ipynb) điều phối M2/M3 trên Colab.
[M1: code và output đã chạy](https://colab.research.google.com/drive/13Abjl7kJyDKzbxyYsCetJ540OzdK2P_E)
giữ trong `archive/03_m1_completed_with_outputs_20261013.ipynb`.

Phần train có 7 cell: mô tả, kết nối kho lưu, setup source, kiểm đầu vào,
M2, M3 và đọc kết quả. Mục validation ở cuối gọi CLI kiểm checkpoint rồi
xuất cả bốn nhánh; không sửa các cell train hoặc output cũ. Logic nằm trong
`src/`; notebook gọi CLI ở `scripts/`. Chỉ có một lần mount.

Mặc định `MODE='inspect'`. Bộ cuối đã khóa 186 cặp; M2/M3 đã train xong.
Khi chạy nhóm mới: kiểm metadata → `MODE='gate'` → `MODE='train'`.
Hai nhánh dùng cùng run group, bộ chosen và checkpoint M1; không dùng
adapter gate làm điểm khởi đầu run chính.

[Lịch sử M1 và các bước kiểm cũ](archive/README.md) giữ source/output
nguyên vẹn. Không chạy lại notebook lịch sử để xem kết quả.
Notebook dùng số thứ tự cho giai đoạn, tên sau số mô tả công việc.
Không đổi tên bản đã chạy hoặc viết thêm công việc mới vào lịch sử M1.

[Trạng thái](../TIEN_DO.md) · [Huấn luyện M2/M3](../docs/preference_training.md)
