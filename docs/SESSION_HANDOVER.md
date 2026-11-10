# Tài liệu Bàn giao Kỹ thuật (Session Handover)

Tài liệu này tổng hợp toàn bộ bối cảnh dự án, hiện trạng mã nguồn, chẩn đoán nguyên nhân khoảng cách điểm số (0.9160 vs 0.37200) và kế hoạch hành động tiếp theo cho agent hoặc kỹ sư kế thừa.

---

## 1. Bối cảnh Dự án và Ràng buộc Môi trường

- Cuộc thi: Zalo AI Challenge 2026 — Subtask 2: DataMate (Vietnamese Tabular QA).
- Mục tiêu: Trả lời câu hỏi kinh doanh tự nhiên bằng tiếng Việt trên dữ liệu bảng dị thể (CSV, XLSX nhiều sheet, SQLite) của 44 doanh nghiệp (30 train, 14 public test).
- Thang điểm: Penalized Accuracy:
  \[
  \text{PA} = \frac{\text{Correct} - \text{Wrong}}{\text{Total}}
  \]
  - Đúng: +1 điểm.
  - Sai (sai giá trị `WRONG_VALUE`, sai định dạng `FORMAT_ERROR`, sai kiểu `WRONG_TYPE`): -1 điểm.
  - Bỏ quyền trả lời (`ABSTAIN`) hoặc xác định câu hỏi không đủ dữ liệu/chủ quan (`UNANSWERABLE` chính xác): 0 điểm.
- Quy định Git & Nhánh:
  - Chỉ làm việc và push trên nhánh: `vinni_cseAI` tại remote `git@github.com:vohoangloc-32/datamate.git`.
  - Không sử dụng hoặc phụ thuộc vào nhánh `main`.
  - Tuyệt đối không commit hoặc push dữ liệu thi đấu (`train/`, `public_test/`, `data/`, `artifacts/`, `*.zip`, `*.pkl`, `submission.jsonl` đều nằm trong `.gitignore`).
- Môi trường phần cứng:
  - Máy phục vụ phát triển và chạy nộp bài (submission runtime): 1x NVIDIA A100 40GB VRAM.
  - Model mục tiêu: 14B tham số (ưu tiên kiểm chuẩn Qwen3-14B và Qwen2.5-Coder-14B-Instruct) chạy `bf16` hoặc `AWQ 4-bit` qua `vLLM`.

---

## 2. Kiến trúc Hệ thống Hiện tại

Hệ thống được tổ chức theo mô hình Canonical Data Engine + Semantic Routing + Deterministic Execution:

```
Input Record (question, business_id, as_of_date)
  │
  ▼
DataLoader (src/loader.py, src/excel_fast.py)
  │
  ▼
Canonical Normalizer (src/canonical.py) ── Chuẩn hoá bảng thô về schema thống nhất:
  │                                        khử dòng trùng lặp (§8), forward-fill ô trống,
  │                                        xác định đơn huỷ/trả hàng/số lượng âm,
  │                                        quy đổi đơn vị cơ bản, tách Topping chuỗi.
  ▼
Rulebook & Temporal Engine (src/rule_engine.py, src/temporal.py)
  │                                        Xử lý ngày/tháng âm-dương, mốc thời gian relative,
  │                                        áp dụng luật đặc thù từng shop.
  ▼
Question Parser (src/parser.py) ───────── Trích xuất intent và thực thể (LinkedEntities).
  │                                        (HIỆN LÀ ĐIỂM YẾU LỚN NHẤT CỦA HỆ THỐNG).
  ▼
Typed Intermediate Representation (src/ir.py: QueryPlanIR)
  │
  ▼
Deterministic Executor (src/executor.py) ─ Tính toán nghiệp vụ bằng Pandas/Python thuần.
  │                                        Đã được kiểm chứng độ chính xác cao khi
  │                                        nhận đúng intent và slot.
  ▼
Answerability Guard & Verifier (src/guard.py, src/verifier.py)
  │                                        Phát hiện câu hỏi có ràng buộc bị bỏ sót
  │                                        để chủ động chuyển sang ABSTAIN (0 điểm thay vì -1).
  ▼
Exact Contract Serializer (src/serializer.py) ── Định dạng JSONL chuẩn quy cách nộp bài.
```

### Danh mục các module trong `src/`:
- `canonical.py`: Chuẩn hóa 44 schema doanh nghiệp về mô hình dữ liệu đồng nhất.
- `entity_linker.py`: Ánh xạ thực thể tên sản phẩm, nhân viên/thợ, khách hàng, danh mục.
- `evaluator.py`: Triển khai chính thức hàm chấm điểm Penalized Accuracy và đối sánh nhãn.
- `excel_fast.py`: Bộ đọc Excel hiệu năng cao.
- `executor.py`: Bộ thực thi câu truy vấn tất định trên bảng canonical.
- `guard.py`: Bộ phòng thủ kiểm tra tính đầy đủ của slot trước khi trả lời.
- `ir.py`: Định nghĩa cấu trúc trung gian `QueryPlanIR` và `SubmissionRecord`.
- `llm_client.py`: Khung kết nối vLLM cho mô hình mã nguồn mở.
- `loader.py`: Nạp đa định dạng (CSV, XLSX, SQLite).
- `parser.py`: Trích xuất intent và thực thể từ câu hỏi tiếng Việt.
- `rule_engine.py`: Xử lý luật nghiệp vụ và từ điển quy đổi.
- `runner.py`: Bộ điều phối chạy theo lô và ghi nhận trace chẩn đoán.
- `serializer.py`: Chuyển đổi kết quả thực thi sang bản ghi nộp bài.
- `temporal.py`: Bộ giải quyết biểu thức thời gian tiếng Việt và lịch âm.
- `value_parser.py`: Chuẩn hóa số, phần trăm, tiền tệ.
- `verifier.py`: Kiểm chứng tính nhất quán của dữ liệu.

---

## 3. Thực trạng Điểm số và Chẩn đoán Khoảng cách (Gap Analysis)

### Kết quả đo lường:
- Tập Train (3,000 câu):
  - Normalized Score: 0.9160 (Điểm: +2748).
  - Phân bố: 2,874 CORRECT, 112 WRONG_VALUE, 14 WRONG_TYPE, 0 FORMAT_ERROR, 0 ABSTAIN.
- Tập Public Test Leaderboard (1,000 câu):
  - Normalized Score: 0.37200.

### Nguyên nhân gốc rễ đã được xác minh qua trace (`artifacts/trace_public.jsonl`):
1. **Lỗi crash toàn bộ một doanh nghiệp (`dev_012` - Tiệm sửa xe):**
   - Triệu chứng: Toàn bộ 50 câu hỏi của `dev_012` bị crash với lỗi `KeyError: 'alias_norm'` tại `src/entity_linker.py`, khiến 100% câu của shop này rơi vào fallback `ABSTAIN` (0 điểm).
   - Đã xử lý trong phiên: Điều chỉnh `src/entity_linker.py` để sử dụng đúng trường `mechanic_name` và `mechanic_norm` tương thích với `_build_auto_repair` trong `src/canonical.py`. Đã kiểm chứng chạy lại thành công cả 50 câu của `dev_012`.
2. **Hiện tượng âm thầm bỏ rơi ràng buộc (Constraint Dropping) của Regex Router:**
   - Triệu chứng: `src/parser.py` sử dụng tập regex và từ khóa khớp chuỗi con được tinh chỉnh trên tập train. Khi gặp các câu hỏi public mang ngữ cảnh diễn đạt mới (teencode, phương ngữ, cấu trúc câu đảo ngữ), router không nhận diện được intent chi tiết và tự động rơi về intent mặc định (thường là `revenue_total` hoặc `order_count`).
   - Hậu quả: Hệ thống tính toán tổng doanh thu hoặc tổng đơn hàng của toàn quán, bỏ qua hoàn toàn các ràng buộc về ca làm việc (sáng/chiều), nhân viên/thợ phụ trách, nhóm khách hàng, trạng thái huỷ đơn, ngưỡng tiền, hoặc số ngày.
   - Minh chứng từ audit 30 câu ngẫu nhiên:
     - Câu: `"nv nào bán đc dthu cao nhứt"` -> Router trả về `revenue_total` (số tiền 1.2 tỷ) thay vì tên nhân viên (`string`).
     - Câu: `"NV001 bán được bao nhiêu tiền"` -> Router tính tổng doanh thu cả quán thay vì filter theo NV001.
     - Câu: `"wan mik huỷ bnhiu đơn"` -> Router trả về `order_count` (tổng 1770 đơn) thay vì số đơn huỷ.
     - Câu: `"Khách nhóm thân thiết 08/2026 mang ve may % tong doanh thu"` -> Router trả về `revenue_total` thay vì tỷ lệ phần trăm (`percent`).
   - Tác động điểm số: Do Penalized Accuracy phạt -1 cho mỗi câu sai, việc trả lời sai khi thiếu ràng buộc làm trừ điểm nặng hơn nhiều so với việc chủ động abstain. Tỷ lệ sai trên số câu trả lời ở public ước tính ~30% (khoảng 289 câu sai / 950 câu trả lời), kéo điểm tụt từ 0.916 xuống 0.372.
3. **Lỗi phương pháp luận (Methodological Flaw):**
   - Con số 0.916 trên train không phản ánh khả năng tổng quát hóa, vì các rule và regex trước đây được điều chỉnh trực tiếp dựa trên nhãn của toàn bộ 3,000 câu train mà không có tập validation held-out tách biệt.

---

## 4. Công việc Đã Triển khai trong Phiên Này

1. **Sửa lỗi crash `dev_012`:**
   - File: `src/entity_linker.py`.
   - Khớp nối trường thợ sửa xe giữa canonical data và linker. 50 câu của `dev_012` đã chạy thông suốt.
2. **Xây dựng module phòng ngự `src/guard.py`:**
   - Mục đích: So sánh câu hỏi nguyên bản với intent/kiểu dữ liệu đã được parse. Nếu phát hiện câu hỏi chứa tín hiệu ràng buộc rõ ràng (dấu hiệu hỏi người `who_cue`, dấu hiệu ngày `date_cue`, bảng biểu `table_cue`, tỷ lệ phần trăm `percent_cue`, đơn huỷ `cancel_cue`, ngưỡng số lượng `threshold_cue`) nhưng intent được chọn là dạng tổng gộp cơ bản (`revenue_total`, `order_count`), guard sẽ trả về mã lý do để hệ thống chuyển sang `ABSTAIN`.
   - Kết quả đo trên train: Chặn chính xác 38 câu sai và chỉ chặn nhầm 1 câu đúng (net gain: +37 điểm).
   - Kết quả trên public: Nhận diện và chặn 100 câu hỏi rơi vào bẫy bỏ sót ràng buộc.

---

## 5. Kế hoạch Hành động Kế tiếp (Roadmap cho Agent Mới)

### Bước 1: Chuẩn hóa Quy trình Đánh giá (Evaluation Protocol)
- Tách tập Train thành 2 nhóm theo doanh nghiệp (Business-level split):
  - 22 doanh nghiệp dùng để phát triển / dev.
  - 8 doanh nghiệp giữ làm held-out validation (tuyệt đối không nhìn nhãn để viết rule/prompt).
- Gán nhãn thủ công khoảng 50 - 100 câu hỏi thuộc public test (lấy mẫu đều qua 14 doanh nghiệp `dev_001` đến `dev_014`) để có tín hiệu kiểm chứng độc lập gần với bảng xếp hạng.

### Bước 2: Tích hợp Mô hình 14B làm Semantic Parser (Phương án Hybrid)
- **Vai trò của LLM:** Mô hình 14B **KHÔNG** dùng để sinh code Python tự do (vì code tự do từ LLM 14B dễ vi phạm các định nghĩa phức tạp của rulebook như xử lý đơn âm, nợ nhiều sổ, khử dòng lặp). Thay vào đó, LLM đóng vai trò bộ phân tích ngữ nghĩa có cấu trúc:
  - Đầu vào: Câu hỏi, `as_of_date`, danh sách thực thể rút gọn của doanh nghiệp hiện tại.
  - Đầu ra: Bản ghi JSON tuân thủ schema của `QueryPlanIR` (intent, start_date, end_date, product_key, staff_id, customer_group, threshold, shift, v.v.).
- **Bộ máy thực thi:** Giữ nguyên `src/executor.py`. Sau khi có `QueryPlanIR` từ LLM, executor tất định sẽ chạy tính toán trên bảng canonical để bảo đảm 100% tính nhất quán về công thức doanh thu, đơn huỷ, tỷ lệ và làm tròn `ROUND_HALF_UP`.
- **Cấu hình trên A100 40GB:**
  - Khởi tạo `vLLM` với model Qwen3-14B hoặc Qwen2.5-Coder-14B-Instruct.
  - Sử dụng định dạng `bfloat16`, `gpu_memory_utilization=0.90`, `max_model_len=8192`.
  - Áp dụng guided decoding (JSON schema) để đầu ra luôn parse được thành object IR.

### Bước 3: Cơ chế Gating Kết hợp (Router + LLM Gate)
- Nếu Router regex truyền thống và LLM Parser cho kết quả đồng thuận -> Chấp nhận thực thi.
- Nếu Router chọn intent tổng quát nhưng LLM trích xuất được slot chi tiết -> Ưu tiên kết quả của LLM.
- Nếu cả hai nguồn không chắc chắn hoặc phát hiện mâu thuẫn -> Fallback sang `ABSTAIN` (0 điểm) để bảo vệ điểm Penalized Accuracy.

---

## 6. Các Lệnh Vận hành Chính

Chạy pipeline sinh file nộp bài:
```bash
python3 scripts/run_baseline.py --split public_test --output submission.jsonl --trace artifacts/trace_public.jsonl
```

Chạy đánh giá trên tập train:
```bash
python3 scripts/evaluate.py --pred submission.jsonl --gold train/answers.jsonl
```

Chạy toàn bộ unit test:
```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

Đo đạc module guard:
```bash
python3 scratch/eval_guard.py
```
