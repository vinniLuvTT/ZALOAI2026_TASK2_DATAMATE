# Competition Contract Chính thức — DataMate (Zalo AI Challenge 2026)

Tài liệu này khóa chặt các quy định đầu vào, đầu ra, chấm điểm và quy tắc nghiệp vụ dựa trực tiếp trên các file chính thức trong bộ dữ liệu: `train/rulebook.md`, `public_test/rulebook.md`, `train/questions.jsonl`, `train/answers.jsonl`, `public_test/questions.jsonl`, `train/calendar.csv` và 44 file `README.md` của từng doanh nghiệp.

---

## 1. Bảng xác minh trạng thái Contract

| Hạng mục | Giá trị chính thức | Nguồn (file/section/line) | Trạng thái |
|---|---|---|---|
| Input schema | JSONL (`questions.jsonl`), mỗi dòng gồm `id` (str), `business_id` (str), `question` (str), `as_of_date` (`YYYY-MM-DD`) | `train/questions.jsonl`, `public_test/questions.jsonl`, `rulebook.md` L5 | VERIFIED |
| Output schema | JSONL (`submission.jsonl`), mỗi dòng gồm `{"id": "...", "type": "...", "answer": ...}` (hoặc `unanswerable` / `abstain`) | `train/rulebook.md` §9 L116-130, `public_test/rulebook.md` §9 L150-164 | VERIFIED |
| Scoring formula | Đúng **+1**, Sai **−1**, Bỏ trống (`{"type": "abstain"}` hoặc không có dòng) **0** | `train/rulebook.md` §9 L116, `public_test/rulebook.md` §9 L150 | VERIFIED |
| Abstention & Unanswerable | `unanswerable`: câu hỏi không đủ dữ liệu để trả lời, xuất `{"id": "...", "type": "unanswerable", "reason": "..."}` (chấm +1 nếu nhãn là `unanswerable`, −1 nếu sai). `abstain`: chủ động bỏ qua, xuất `{"id": "...", "type": "abstain"}` (luôn nhận 0 điểm) | `rulebook.md` §7 L97-103 (`train`) / L130-136 (`public_test`) & §9 | VERIFIED |
| Calendar & Time | Lịch dương - âm tại `calendar.csv` (`date,weekday,lunar_day,lunar_month,lunar_year,lunar_leap,holiday`). Tuần bắt đầu thứ Hai (`weekday=1`). `as_of_date` là "hôm nay" của từng câu | `train/calendar.csv`, `rulebook.md` L5 & §2 | VERIFIED |
| Runtime target | Kaggle GPU T4 x2 (2x 16GB VRAM = 32GB VRAM) theo thông tin người dùng cung cấp; giới hạn container chi tiết của BTC chưa đi kèm trong zip | Thông tin người dùng / Portal | PARTIAL / UNKNOWN (Docker spec) |
| Model/API policy | Chạy độc lập trên dữ liệu phát hành, không dùng hiểu biết bên ngoài (`rulebook.md` §7). Trọng số mô hình mở chạy trên Kaggle T4 x2 | `rulebook.md` §7 L98 | VERIFIED |

---

## 2. Input Schema

- **File câu hỏi:** `<split>/questions.jsonl` (UTF-8, mỗi dòng 1 JSON object).
- **Các trường bắt buộc:**
  - `id` (`string`): Mã định danh duy nhất của câu hỏi (ví dụ: `"q_001883280e83"`).
  - `business_id` (`string`): Mã thư mục dữ liệu doanh nghiệp trong `<split>/data/<business_id>/` (tập `train`: `biz_101`..`biz_130`; tập `public_test`: `dev_001`..`dev_014`).
  - `question` (`string`): Câu hỏi tiếng Việt tự nhiên, có thể chứa viết tắt (`dthu`, `bnhiu`, `sp`, `đc`, `hnay`, `trc`, `k`, `z`...), lỗi gõ phím Telex/VNI (`ddieenj`, `tieenf`, `naof`, `cuar`, `nguwowif`, `nhieuđơn`...) hoặc không dấu.
  - `as_of_date` (`string`, định dạng `YYYY-MM-DD`): Ngày tham chiếu ("hôm nay") của câu hỏi.
- **Quy tắc cắt thời gian theo `as_of_date` (`rulebook.md` L5):**
  - File dữ liệu có thể chứa bản ghi có ngày sau `as_of_date`. Với câu hỏi tính theo "hôm nay" (như "tháng này", "tuần này", "tính đến hôm nay", "hôm nay", "N ngày gần nhất", "từ đầu năm tới nay"...), những bản ghi sau `as_of_date` coi như chưa xảy ra.
  - **Ngoại lệ quan trọng (`rulebook.md` L5 & §1):** Trạng thái huỷ, trả hàng của đơn lấy theo trạng thái ghi trên đơn (hoặc bảng trả hàng/lô), **kể cả khi phiếu trả ghi ngày sau hôm nay** ("Đơn trả hàng sau kỳ, kể cả trả sau hôm nay, vẫn bị loại khỏi doanh thu kỳ đặt").

---

## 3. Output Schema (`submission.jsonl`)

Mỗi dòng là một JSON object: `{"id": "...", "type": "...", "answer": ...}`.

| `type` | Định dạng `answer` | Quy tắc làm tròn & so khớp (`rulebook.md` §9) |
|---|---|---|
| `number` | JSON number (`int` hoặc `float`), không kèm đơn vị, không dấu phân cách nghìn | Tiền là số nguyên VND (làm tròn về đồng, nửa đồng làm tròn lên: `floor(x + 0.5)`). Trung bình của một khoản tiền cũng làm tròn về đồng. Số trung bình khác làm tròn 2 chữ số thập phân |
| `percent` | JSON number biểu diễn số phần trăm (ví dụ `12.4` cho 12,4%, **không** viết `0.124`) | Làm tròn 2 chữ số thập phân (`round(x, 2)`). Giảm thì ra số âm |
| `date` | Chuỗi `"YYYY-MM-DD"`; nếu câu hỏi hỏi theo tháng thì `"YYYY-MM"` | Khớp chuỗi chuẩn ISO |
| `boolean` | JSON boolean `true` hoặc `false` | Khớp tuyệt đối |
| `string` | Chuỗi tên đối tượng lấy nguyên từ dữ liệu (hoặc mã đối tượng) | Không phân biệt hoa/thường, khoảng trắng, dấu câu. Chấp nhận mọi cách viết tên có trong file của shop, hoặc mã (`alias`), hoặc tên khách kèm SĐT như trong file |
| `list` | Mảng JSON `[...]`, giữ nguyên thứ tự; 1 phần tử vẫn là mảng | Đúng cả tập và thứ tự. Câu hỏi nhiều số thì ghi list theo đúng thứ tự câu hỏi nêu. Top-k xếp giảm dần theo tiêu chí; bằng nhau xếp theo **mã tăng dần** (không có mã thì theo tên tăng dần). Liệt kê theo hạn dùng: hạn sớm nhất trước, cùng hạn theo mã tăng dần. Liệt kê chỉ lọc điều kiện (không có số đo): theo mã tăng dần |
| `table` | `{"columns": ["...", ...], "rows": [[...], ...]}` | Cột đầu là tên/mã đối tượng, các cột sau là số đo (JSON number) theo đúng thứ tự câu hỏi nêu. Thứ tự các dòng trong `rows` không ảnh hưởng kết quả chấm |
| `unanswerable` | `{"id": "...", "type": "unanswerable", "reason": "..."}` (trường `reason` tùy chọn, nếu có phải là `str`) | Dùng khi kỳ hỏi nằm hoàn toàn ngoài khoảng ngày có dữ liệu của shop, hoặc shop không ghi loại thông tin câu hỏi cần, hoặc câu hỏi đòi hỏi tri thức ngoài dữ liệu |
| `abstain` | `{"id": "...", "type": "abstain"}` (hoặc bỏ qua dòng đó) | Nhận 0 điểm, không bị phạt −1 |

---

## 4. Tóm tắt các Quy tắc Nghiệp vụ Cốt lõi (`rulebook.md` §1 – §8)

1. **Đơn hàng & Doanh thu (§1):**
   - Đơn tính vào doanh thu và số đơn: **không huỷ** và **không trả hàng** (trạng thái `hoàn thành` hoặc `đang giao`). Đơn trả một phần cũng là đơn trả hàng -> bỏ cả đơn.
   - Doanh thu một đơn = tổng **thành tiền các dòng hàng** (đã trừ giảm giá dòng). **Không** trừ giảm giá tổng đơn/voucher shop, **không** cộng VAT, **không** gồm phí giao hàng.
   - Tỉ lệ đơn huỷ trong kỳ = `(số đơn huỷ đặt trong kỳ / tổng mọi đơn đặt trong kỳ ở mọi trạng thái) * 100`, làm tròn 2 chữ số thập phân.
   - Hình thức thanh toán: lấy trên đơn; `COD` là hình thức riêng (không gộp vào tiền mặt); đơn ghi nợ không tính vào tiền mặt/chuyển khoản.
   - Nhóm khách: khách lẻ / không ghi khách không thuộc nhóm nào (kể cả nhóm tên `lẻ`), nhưng mẫu số tỉ trọng doanh thu nhóm khách vẫn chia cho tổng doanh thu cả shop (gồm cả khách lẻ).
2. **Thời gian (§2):**
   - Tuần bắt đầu **thứ Hai** (`weekday=1`). `Tuần này`: thứ Hai tuần chứa `as_of_date` đến `as_of_date`. `Tuần trước / tuần rồi`: thứ Hai đến Chủ nhật tuần liền trước.
   - `Tháng này`: từ mùng 1 tháng chứa `as_of_date` đến `as_of_date`. So với cùng kỳ tháng trước: từ mùng 1 tháng trước đến ngày `min(as_of_date.day, last_day_of_prev_month)`.
   - `N ngày gần nhất`: `[as_of_date - (N-1) days, as_of_date]`.
   - `Tết năm Y / dịp Tết`: từ ngày dương của `23/12` âm lịch năm `Y-1` đến hết ngày dương của `10/01` âm lịch năm `Y` (tra `calendar.csv`, tạo thành khoảng 17 hoặc 18 ngày tùy năm âm). So với trước Tết = cùng số ngày ngay trước 23 tháng Chạp.
   - `Ca sáng`: `06:00:00` – `13:59:59`; `Ca chiều`: `14:00:00` – `21:59:59`.
   - `Trung bình mỗi ngày thứ X của kỳ` (`public_test`): chia cho tổng số ngày thứ X có trong kỳ lịch đó (kể cả ngày không phát sinh đơn).
3. **Xếp hạng & Thực thể (§3):**
   - Dòng `Phí phục vụ` không phải mặt hàng (không đếm mặt hàng, không xếp hạng mặt hàng), nhưng **vẫn tính vào doanh thu** (kể cả mẫu số tỉ trọng nhóm hàng).
   - `Bán chạy / bán nhiều nhất / hot nhất`: xếp theo **tổng số lượng quy về đơn vị cơ bản** (bỏ đơn huỷ, trả).
   - `Bán được bao nhiêu <đơn vị>`: chỉ cộng số lượng quy về đơn vị cơ bản của các mặt hàng **có đơn vị cơ bản đúng bằng `<đơn vị>`**.
   - `Hàng lời nhất`: theo tổng lãi gộp (`doanh thu dòng − số lượng bán * giá vốn 1 đơn vị bán ghi trên dòng bán`).
   - `Khách trung thành nhất`: nhiều ngày có mua nhất trong kỳ. `Khách sộp nhất`: giá trị trung bình 1 đơn cao nhất trong kỳ, chỉ xét khách có `>= 3` đơn hợp lệ trong kỳ. `Khách hay đổi ý nhất`: nhiều đơn huỷ nhất trong kỳ. `Mặt hàng ưa chuộng nhất`: có mặt trong nhiều đơn hợp lệ nhất trong kỳ. `Ngày đắt khách nhất`: nhiều đơn hợp lệ nhất trong kỳ. `Giờ cao điểm`: khung giờ `h` (`0..23`) có nhiều đơn hợp lệ nhất.
4. **Công nợ (§4):**
   - Đơn tính vào nợ: đơn **không huỷ** (**đơn trả hàng vẫn tính** nếu còn nợ). Bỏ đơn `COD` và đơn bán qua `Sàn TMĐT`.
   - Nợ 1 đơn tại ngày X = `Tổng phải trả cả đơn (đã trừ giảm giá đơn, gồm phí ship) − Tổng tiền đã trả đến hết ngày X`.
   - `Nợ thêm trong kỳ`: phần chưa trả ngay trong ngày mua của các đơn ghi nợ (không huỷ) đặt trong kỳ.
   - `Trả nợ / thu nợ trong kỳ`: các khoản trả **sau ngày mua** có ngày trả nằm trong kỳ, của mọi đơn ghi nợ (kể cả đơn đặt kỳ trước hoặc đơn trả hàng).
5. **Kho, Nhập hàng, Giá vốn (§5):**
   - Tiền nhập hàng: tổng tiền phiếu nhập có ngày phiếu trong kỳ, bỏ phiếu huỷ (tính cả phiếu nhập đầu kỳ nếu ngày phiếu trong kỳ). `Nhà cung cấp nhập nhiều nhất` = tổng tiền nhập cao nhất; `Nhà cung cấp chính` = nhiều phiếu nhập nhất.
   - Giá trị tồn kho = tồn hết hôm nay (đơn vị cơ bản) * giá vốn 1 đơn vị cơ bản trong danh mục hàng (chỉ tính hàng tồn > 0).
   - Lãi ròng tháng = Lãi gộp tháng − tổng mọi khoản chi phí trong sổ chi phí tháng đó.
6. **Quy tắc riêng theo ngành (§6):**
   - **Cà phê:** Size `(M)` / `(L)` ở cuối tên món (không phân biệt hoa thường). Topping ghi chung ô với món (`dev_002`) phải tách thành dòng mặt hàng topping riêng, tiền topping tách khỏi tiền món.
   - **Nhà thuốc:** Thuốc kê đơn lấy theo cột kê đơn trong danh mục thuốc; tỉ lệ kê đơn tính theo doanh thu. Hàng hết hạn trong N ngày tới: lô còn tồn có hạn dùng trong `[D, D + N]`, tiền vốn tính theo giá nhập của chính lô đó quy về đơn vị cơ bản. Hoạt chất khớp chính xác chuỗi hoạt chất (thuốc phối hợp nhiều hoạt chất là loại riêng).
   - **Vật liệu xây dựng:** Số đợt giao trong tháng X = số cặp `(mã đơn, lần giao)` có ngày giao trong tháng X (tính đến hết hôm nay), kể cả đơn sau đó trả hàng.
   - **Mẹ và bé (`dev_013`):** Tiền giảm voucher, tiền giảm dùng điểm, phí ship trả số dương, tính theo ngày hoá đơn của đơn không huỷ, không trả. Số sữa / size bỉm lấy từ cột `Số / Size` của danh mục; phần trăm của 1 size/hãng bỉm chia cho tổng số bịch mọi hàng nhóm `bỉm tã` cùng kỳ.
   - **Sửa xe (`dev_012`):** Mỗi dòng là 1 lượt xe (1 phiếu), kể cả bảo hành (`Tiền = 0` hoặc ghi chú bảo hành). Trung bình tiền/phiếu chỉ chia cho phiếu có thu tiền. `Nhớt` trơn = nhớt máy (không phải nhớt láp); `ruột` trơn = thay ruột mới (không phải vá); `Sửa điện` không gồm thay bóng đèn, thay ắc quy; phiếu bảo hành không tính vào số lượt làm việc đó. `Xe khách quen` = biển số có `>= 2` phiếu có thu tiền trong kỳ.
   - **Cửa hàng điện thoại (`dev_011`, `dev_014`):** Doanh thu không trừ tiền thu máy cũ và không trừ giảm giá cả đơn (`Bớt`). Trả góp tính vào công nợ và thu nợ. Dòng máy = phần tên trước `...GB`. Hãng = chữ đầu tên hàng (`iPhone` -> `Apple`).
7. **Quy tắc đọc file (§8):**
   - Ô để trống ở các dòng sau của cùng một đơn/nhóm dòng: kế thừa giá trị dòng đầu nhóm (forward-fill theo đơn).
   - Hai dòng của **cùng một đơn** giống nhau ở **mọi ô**: lỗi xuất file -> khử trùng lặp (chỉ tính 1 lần). Bảng không ghi mã đơn: không khử trùng lặp hai dòng giống nhau.
   - Ngày không ghi năm (ví dụ `"1/5"`): lấy năm của tên sheet chứa dòng đó.
