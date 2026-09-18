# Research plan

## Objective

Đánh giá DPO có cải thiện mức phù hợp với ảnh của lời giải thích so với SFT tiếp trên cùng nguồn chosen hay không, đồng thời theo dõi độ chính xác đáp án.

## Experimental matrix

- M0 pretrained
- M1 SFT on ViVQA-X
- M2 M1 plus SFT on chosen responses
- M3 M1 plus DPO on preference pairs

M2 và M3 phải xuất phát từ cùng checkpoint M1. Test được giữ riêng đến khi checkpoint và rubric được đóng băng.

