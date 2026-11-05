# Báo cáo Khảo sát Repository và Môi trường (Giai đoạn 0)

## 1. Cấu trúc Repository và Dữ liệu hiện có

Thư mục làm việc hiện tại: `/home/vinnievu/Documents/ZALO AI CHALLENGE 26/SUBTASK 2: DATAMATE`

Các artifact gốc của cuộc thi đã được giải nén đầy đủ:

- `train.zip` (128,118,095 bytes) -> giải nén tại `train/`:
  - `train/rulebook.md` (142 dòng): Bộ quy tắc nghiệp vụ, cách đọc file, và quy chuẩn chấm điểm chính thức cho tập train.
  - `train/calendar.csv` (1,462 dòng, 1,461 ngày từ `2024-01-01` đến `2027-12-31`): Lịch dương - âm và ngày lễ.
  - `train/questions.jsonl` (3,000 câu hỏi): Mỗi dòng gồm `id`, `business_id`, `question`, `as_of_date`.
  - `train/answers.jsonl` (3,000 đáp án nhãn chuẩn): Mỗi dòng gồm `id`, `type`, `answer`, `alias`.
  - `train/data/biz_101` .. `train/data/biz_130` (30 doanh nghiệp): Gồm file `README.md` mô tả đặc tả từng shop và các file dữ liệu (`.csv`, `.xlsx`, `.sqlite`).
- `public_test.zip` (64,149,714 bytes) -> giải nén tại `public_test/`:
  - `public_test/rulebook.md` (176 dòng): Mở rộng thêm các quy tắc cho một số câu hỏi xếp hạng mới (khách trung thành, khách sộp, khách hay đổi ý, mặt hàng ưa chuộng, giờ cao điểm, ngày đắt khách, trung bình ngày thứ X) và 3 ngành mới (`Mẹ và bé`, `Sửa xe`, `Cửa hàng điện thoại`), đồng thời bổ sung quy tắc `Topping ghi chung ô với món` (ngành Cà phê) và `Sổ không có cột trạng thái` (xem cột Ghi chú).
  - `public_test/calendar.csv` (1,462 dòng): Giống hệt `train/calendar.csv` (kiểm chứng bằng `diff -u`).
  - `public_test/questions.jsonl` (1,000 câu hỏi): Phân bổ trên 14 doanh nghiệp `dev_001` .. `dev_014`.
  - `public_test/data/dev_001` .. `public_test/data/dev_014` (14 doanh nghiệp): Tổng cộng 252 file trên cả hai tập `train` và `public_test`.

## 2. Môi trường Runtime và Phần cứng

### 2.1 Môi trường máy cục bộ (Local Development & Verification)
- **OS:** Linux (Arch Linux)
- **CPU:** 13th Gen Intel Core i5-13420H (12 luồng)
- **RAM:** 15 GiB (~6.6 GiB khả dụng) + 15 GiB Swap
- **GPU cục bộ:** NVIDIA GeForce RTX 3050 Laptop GPU (4096 MiB VRAM, CUDA Driver 615.78.08 / CUDA 13.4)
- **Python:** Python 3.14.8 (`/usr/bin/python3`)
- **Thư viện Python có sẵn:** `pandas==2.3.3`, `numpy==2.5.4`, `openpyxl==3.1.5`, `pydantic==2.14.1`, `orjson==3.13.0`, `PyYAML==6.0.3`, `jsonschema==4.26.0`, `sqlite3` (stdlib), `requests==2.34.2`, `httpx==0.28.1`, `aiohttp==3.14.3`, `tqdm==4.70.1`, `rich==15.0.0`, `unittest` (stdlib).
- **Thư viện không cài sẵn trên máy cục bộ:** `pip`, `pytest`, `duckdb`, `polars`, `rapidfuzz`, `torch`, `transformers`, `vllm`.
  - Hệ quả kỹ thuật: Toàn bộ test suite phải chạy được qua `python3 -m unittest` (và tương thích `pytest` khi chạy trên Kaggle). Bộ xử lý bảng và truy vấn dùng `pandas` + `sqlite3` + `openpyxl` làm nhân thực thi xác định, không phụ thuộc bắt buộc vào `duckdb` hay `polars`.

### 2.2 Môi trường mục tiêu (Target Compute: Kaggle T4 x2)
- **GPU:** 2 x NVIDIA Tesla T4 (16 GB VRAM x 2 = 32 GB VRAM tổng, kiến trúc Turing SM75).
- **Đặc điểm phần cứng cần lưu ý trên T4 x2:**
  - Kiến trúc Turing (SM75) hỗ trợ tăng tốc phần cứng cho `FP16` và `INT8`/`INT4` (AWQ/GPTQ), **không** hỗ trợ tốt `bfloat16` nguyên bản (nếu chạy `bf16` trên T4 sẽ bị fallback hoặc lỗi trong vLLM). Do đó config LLM trên Kaggle T4 x2 bắt buộc đặt `dtype="float16"`, `tensor_parallel_size=2`.
  - Với 32 GB VRAM (`tp=2`), có thể nạp mô hình 14B ở `FP16` (khoảng 28 GB) hoặc mô hình 14B/32B ở `AWQ 4-bit` (khoảng 10–18 GB VRAM) với `max_model_len=8192` và `gpu_memory_utilization=0.90`.

## 3. Những gì đang thiếu trong Repository gốc và hướng giải quyết

1. **Chưa có script evaluator tách rời:** Luật chấm điểm và nhãn chuẩn (`train/answers.jsonl` kèm `alias`, cùng mục 9 của `rulebook.md`) đã được phát hành đầy đủ trong gói dữ liệu. Ta sẽ xây dựng `src/evaluator.py` và `scripts/evaluate.py` bám sát tuyệt đối mục 9 của `rulebook.md` và cấu trúc `train/answers.jsonl`.
2. **Chưa có mã nguồn baseline, profiler, test suite:** Toàn bộ pipeline từ đọc dữ liệu thô, chuẩn hóa bảng theo quy ước `README.md` + `rulebook.md`, biên dịch rulebook cho LLM/engine, giải quyết thời gian theo `as_of_date` + `calendar.csv`, thực thi truy vấn, kiểm chứng bằng chứng, đến xuất `submission.jsonl` sẽ được triển khai mới trong `src/`, `scripts/`, `configs/`, `tests/`.
