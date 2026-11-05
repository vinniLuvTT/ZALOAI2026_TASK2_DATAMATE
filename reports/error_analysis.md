# Bao Cao Phan Tich Loi Va Tien Do Baseline v1 -> v3

## 1. Tong Quan Ket Qua Qua Cac Phien Ban (Tap train - 3.000 cau hoi)

| Metric | Baseline v1 | Baseline v2 | Baseline v3 |
|---|---|---|---|
| Tong so cau hoi | 3000 | 3000 | 3000 |
| CORRECT | 1507 (50.23%) | 2562 (85.40%) | 2874 (95.80%) |
| ABSTAIN | 18 | 0 | 0 |
| WRONG_VALUE | 851 | 297 | 112 |
| WRONG_TYPE | 624 | 141 | 14 |
| FORMAT_ERROR | 0 | 0 | 0 |
| Tong diem (+1/-1/0) | +32 | +2124 | +2748 |
| Normalized Score | 0.0107 | 0.7080 | 0.9160 |
| Thoi gian chay | 30.12s | 31.96s | 31.94s |
| QPS | 99.60 | 93.86 | 93.93 |

### Phan ra do chinh xac theo answer_type (Baseline v3)
- `date`: 145/145 (100.00%)
- `string`: 87/88 (98.86%)
- `percent`: 568/578 (98.27%)
- `list`: 188/193 (97.41%)
- `unanswerable`: 68/70 (97.14%)
- `number`: 1785/1884 (94.75%)
- `boolean`: 21/23 (91.30%)
- `table`: 12/19 (63.16%)

---

## 2. Cac Loi Chinh Da Siet Chat Trong v3

1. **Va cham chuoi unaccented `thu tu` trong `wday_map` (`src/temporal.py`):**
   - Nguyen nhan: Cum tu `"doanh thu từ..."` sau khi xoa dau thanh `"doanh thu tu"`, match vao `\bthu tu\b`, ep `specific_weekday = 3` (thu Tu) cho hang tram cau hoi doanh thu thong thuong.
   - Khac phuc: Chuyen sang regex bat co dau tieng Viet co negative lookbehind `(?<!doanh\s)\bthứ\s*(?:hai|2)\b...`.

2. **Va cham tien to san pham (`src/parser.py`):**
   - Nguyen nhan: `"hang nao"` nam ben trong `"thang nao"` lam 13 cau hoi xac dinh thang ban chay nhat bi nhan dien thanh `top_revenue_product` (`string`) thay vi `multi_period_best` (`date`).
   - Khac phuc: Bo sung word boundaries `\b(?:mat hang|san pham|hang)\s+nao\b`.

3. **Routing khach hang theo nguong so don (`src/executor.py`):**
   - Nguyen nhan: Kiem tra `if cb.customers.empty` tra ve `unanswerable` cho cac cua hang khong co bang `khach_hang.csv` rieng biet (nhu `biz_106`), du bang `don_hang.csv` van co cot `customer_key` voi 241 khach dinh danh.
   - Khac phuc: Kiem tra truc tiep su ton tai cua khach dinh danh trong `cb.orders["customer_key"]`.

4. **Tach biet mat hang do uong voi banh/topping tai quan cafe (`src/executor.py`):**
   - Nguyen nhan: Cau hoi "ban duoc bao nhieu ly" tai quan cafe cong ca banh croissant va topping pudding vao san luong ly.
   - Khac phuc: Loai tru category `banh`, `topping`, `snack`, `do an` khi hoi don vi ly.

5. **Phan biet thuoc ke don tai hieu thuoc (`src/executor.py`):**
   - Ho tro filter doanh thu thuoc ke don dua tren truong `is_prescription` tu danh muc `san_pham.csv`.

---

## 3. Phan Tich Cac Loi Con Lai (126 loi tren tap train)

- **Cong no lich su theo hoa don (72 loi):**
  - Cac cua hang co bang `thanh_toan.csv` hoac `BanHang_PhieuThu.xlsx` thu nợ theo nhieu dot khac nhau. Viec xac dinh chinh xac cong no con lai tai mot ngay `as_of_date` trong qua khu can mo hinh hoa toan ven dong tien theo tung phieu thu.
- **Top bang cong no (7 loi):**
  - Mot so bang can can chinh thu tu sap xep giua ma khach hang va so no khi dong hang co gia tri bang nhau.
- **Doanh thu nhap hang va nha cung cap (14 loi):**
  - Cua hang co nhieu lan nhap hang voi don vi quy doi phuc tap can doi chieu voi bang `BanHang_DonViQuyDoi.xlsx`.

---

## 4. Trang Thai File Nop Bai `submission.jsonl`

- Tap du lieu: `public_test` (1.000 cau hoi).
- So ban ghi: Dung 1.000 dong JSONL.
- Thu tu ID: Khop 100% voi thu tu trong `public_test/questions.jsonl`.
- Schema: 100% tuan thu hop dong (`id`, `type`, `answer`).
- Gia tri: Khong chua NaN, Infinity; `unanswerable` va `abstain` co `answer: null`.
