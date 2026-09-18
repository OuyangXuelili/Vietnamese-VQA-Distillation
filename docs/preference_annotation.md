# Preference annotation protocol draft

Mỗi cặp preference phải giữ nguồn câu hỏi, ảnh, chosen, rejected, loại lỗi, phương pháp sinh và trạng thái kiểm tra.

Các loại lỗi ban đầu:

- object
- attribute
- relation_action
- natural_model_error

Không gán nhãn rejected chỉ vì câu khác tham chiếu. Chỉ chấp nhận cặp khi có thể xác định lỗi từ ảnh và câu hỏi.

