# Báo cáo Khảo sát và Profiling Dữ liệu (Giai đoạn 2 — Dataset Findings)

## 1. Quy mô và Phân bố Dữ liệu

| Tập dữ liệu | Số doanh nghiệp | Số câu hỏi | Số file dữ liệu (không tính README) | Định dạng xuất hiện |
|---|---:|---:|---:|---|
| `train` | 30 (`biz_101` .. `biz_130`) | 3,000 | 140 | `.csv`, `.xlsx`, `.sqlite` |
| `public_test` | 14 (`dev_001` .. `dev_014`) | 1,000 | 68 | `.csv`, `.xlsx`, `.sqlite` |
| **Tổng cộng** | **44** | **4,000** | **208** (+ 44 `README.md`) | `.csv`, `.xlsx`, `.sqlite` |

Phân bố nhãn chuẩn trên `train/answers.jsonl` (3,000 câu):
- `number`: 1,884 câu (62.80%)
- `percent`: 578 câu (19.27%)
- `list`: 193 câu (6.43%)
- `date`: 145 câu (4.83%)
- `string`: 88 câu (2.93%)
- `unanswerable`: 70 câu (2.33%)
- `boolean`: 23 câu (0.77%)
- `table`: 19 câu (0.63%)

---

## 2. Phân loại Kiến trúc Bảng (Business Schema Archetypes)

Qua khảo sát toàn bộ 44 doanh nghiệp, dữ liệu được tổ chức thành các nhóm cấu trúc đặc trưng:

1. **Bảng phẳng 1 file CSV/XLSX (Flat Order-Line):**
   - `biz_101`, `biz_114`, `biz_125`, `biz_107`, `biz_113`, `biz_126`, `biz_110`, `dev_009`.
   - Riêng `biz_107`, `biz_113`, `biz_126` không ghi mã đơn và không ghi tên khách (mỗi dòng 1 mặt hàng bán).
   - Riêng `biz_110`, `dev_009` là danh sách hóa đơn cấp đơn hàng (`danh_sach_hoa_don.csv`), không có chi tiết mặt hàng.
2. **Sổ chi tiết bán hàng tách sheet theo tháng hoặc tách file (`_1`, `_2`, `_2026`, `BanHang_T*`):**
   - `biz_102`, `biz_103`, `biz_104`, `biz_112`, `biz_115`, `biz_116`, `biz_118`, `biz_121`, `biz_122`, `biz_130`, `dev_001`, `dev_002`, `dev_005`, `dev_007`, `dev_008`.
   - Có hiện tượng các dòng sau của cùng một đơn để trống thông tin đơn (`Mã đơn hàng`, `Ngày đặt hàng`, `Trạng thái đơn hàng`, `Tên khách hàng`) — bắt buộc phải forward-fill từ dòng đầu của đơn theo `rulebook.md` §8.
3. **Sổ bán hàng ghi tay chuyển sang Excel (`SoBanHang*.xlsx`, `BangGia.xlsx`, `TonKho.xlsx`, `SoNhapHang.xlsx`):**
   - `biz_105`, `biz_117`, `dev_004`.
   - Có dòng header gộp nhóm ở dòng 1 (`('Hàng', None, None, 'Tiền', None, None, 'Đơn')`), header cột thật ở dòng 2.
   - Đơn trả hàng không ghi trạng thái `"trả hàng"` mà ghi một dòng có **số lượng âm** cùng mã đơn với đơn đã bán (`rulebook.md` §1 & `README.md`: "đơn có dòng số lượng âm là đơn có trả hàng" -> loại toàn bộ đơn đó khỏi doanh thu/số đơn).
4. **Hệ thống POS đa bảng (Excel nhiều file/sheet hoặc SQLite `pos.sqlite`):**
   - Excel POS: `biz_106`, `biz_108`, `biz_109`, `biz_119`, `biz_120`, `biz_124`, `biz_128`, `dev_003`.
   - CSV quan hệ 5 bảng (`don_hang.csv`, `chi_tiet_don.csv`, `san_pham.csv`, `khach_hang.csv`, `thanh_toan.csv`): `biz_111`, `biz_127`, `biz_129`, `dev_010`.
   - SQLite POS (`pos.sqlite` 17 bảng): `biz_123`, `dev_006`.
5. **Ba ngành đặc thù mới trong `public_test`:**
   - **Cửa hàng điện thoại (`dev_011`, `dev_014`):** Có sheet `Ghi tắt` ánh xạ tên viết tắt (`IP16 128 đen` -> `iPhone 16 128GB Đen`, loại `máy mới`/`máy cũ`/`phụ kiện`), cột `IMEI` (dòng 2 máy ghi 2 số cách nhau `" / "`), `Thu máy cũ`, và file `SoNo.xlsx` gồm 3 sheet (`Sổ nợ`, `Trả góp`, `Bảo hành`).
   - **Tiệm sửa xe (`dev_012`):** File `SoSuaXe.xlsx` (14 sheet tháng) và `DanhSachTho.xlsx`. Cột `Xe` gộp tên dòng xe + biển số (`"future 59B6-219.16"`). Cột `Làm gì` gộp nhiều việc cách nhau bằng `" + "` hoặc `","`. Cột `Thợ` ghi tên gọi kèm tiền tố (`"A.Đức"`, `"a Đức"`, `"Đức"` -> khớp với họ tên `"... Đức"` trong `DanhSachTho.xlsx`).
   - **Tiệm mẹ và bé (`dev_013`):** File `BanHang_2025.xlsx`, `BanHang_2026.xlsx` và `QuanLy.xlsx` (`DanhMuc`, `KhachHang`, `NhapHang`, `KhuyenMai`). Cuối mỗi hóa đơn có thể có dòng đặc biệt `Tên hàng` là `"Voucher"`, `"Dùng <n> điểm"` (TT âm) hoặc `"Phí ship"` (TT dương).

---

## 3. Các Bẫy Dữ liệu Thực tế Đã Phát hiện và Xử lý

1. **Banner tiêu đề ở đầu sheet Excel:** Nhiều sheet có 1–3 dòng tiêu đề báo cáo (`"SỔ CHI TIẾT BÁN HÀNG"`, `"DANH SÁCH HÓA ĐƠN"`, `"Trạng thái đơn hàng | Đặt hàng (Đã duyệt)"`, `"Từ ngày 01/01/2025..."`) trước dòng tên cột thật. Đã xử lý tự động trong `src/excel_fast.py:detect_header_row`.
2. **Định dạng số tiền và số lượng hỗn hợp:**
   - Tiền tệ ghi dưới dạng `"55.000đ"`, `"38,000"`, `"92,286.0000"`, `"16 076"`, `"25k"`, `"18400k"`, `"1,01tr"`, `"1,6256tr"`, `"19tr990"`.
   - Số lượng ghi kèm đơn vị như `"3 chai"`, `"1 ly"`, `"2 đĩa"`.
   - Đã xử lý chính xác trong `src/value_parser.py`.
3. **Định dạng ngày tháng hỗn hợp:**
   - Số nguyên/thực Excel serial (`45658` -> `2025-01-01`, `45689.250289` -> `2025-02-01 06:00:25`), chuỗi `"01/02/2025"`, `"1 thg 7 2025"`, và ngày cụt năm `"1/5"` (lấy năm từ tên sheet chứa dòng đó theo `rulebook.md` §8).
4. **Đơn có trả hàng ẩn trong bảng lô hoặc dòng âm:**
   - Ở `biz_105`, `biz_117`, `dev_004`: đơn trả hàng phát hiện qua dòng có `so_luong < 0`.
   - Ở các nhà thuốc (`biz_119`, `biz_120`, `biz_121`, `biz_122`, `biz_123`, `biz_124`, `dev_005`, `dev_006`): ngoài cột trạng thái đơn, nếu bảng `DonHang_Lo` / `DanhSachHoaDonChiTietLo` / `BanHang_BanTheoLo` / `tra_hang` có `Số lượng trả > 0` cho `Mã đơn hàng` đó thì toàn bộ đơn đó là **đơn trả hàng** (`README.md`: `"lớn hơn 0: đơn đó có trả hàng"`) và phải bị loại khỏi doanh thu và số đơn!
5. **Hai dòng trùng hệt nhau trong cùng một đơn (`rulebook.md` §8):**
   - Ví dụ trong `dev_003/BanHang_TongHop_2025.xlsx` (xem log dòng 102–107: hai dòng của `ORD000001`, `ORD000002`, `ORD000003` giống hệt nhau ở mọi ô). Đây là lỗi xuất file được mô tả tại `rulebook.md` §8: `"Hai dòng của cùng một đơn giống nhau ở mọi ô: lỗi xuất file, tính một lần"`! Nếu không deduplicate theo toàn bộ các ô của dòng khi bảng có mã đơn, doanh thu sẽ bị nhân đôi!
