"""Rulebook compiler, rounding, tie-breaking, and domain-specific prompt builder.

Designed to enforce `rulebook.md` invariants both in deterministic execution
and when constructing compact, high-signal prompts for LLM inference on Kaggle T4 x2.
"""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def round_money_vnd(val: float | int | Decimal) -> int:
    """Rulebook §3 & §9:
    Tiền ra số lẻ làm tròn về đồng, nửa đồng làm tròn lên.
    Trung bình của một khoản tiền cũng là tiền, làm tròn về đồng.
    """
    d = Decimal(str(float(val)))
    return int(d.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def round_2dec(val: float | int | Decimal) -> float:
    """Rulebook §1 & §9:
    Phần trăm và số trung bình khác làm tròn 2 chữ số thập phân (ROUND_HALF_UP).
    """
    d = Decimal(str(float(val)))
    res = float(d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    # Avoid -0.0
    if abs(res) < 1e-12:
        return 0.0
    return res


def normalize_order_status(raw_status: Any, raw_note: Any = None) -> str:
    """Map raw order status (or note when status column is absent) into one of 4 canonical statuses:
    - 'completed'
    - 'delivering'
    - 'cancelled'
    - 'returned'
    Per Rulebook §8:
    - 'Sổ không có cột trạng thái: trạng thái đơn ghi ở cột Ghi chú. Ghi chú không nhắc trạng thái là đơn hoàn thành.'
    - 'Mỗi đơn có đúng một trong bốn trạng thái: hoàn thành, huỷ, trả hàng, đang giao.'
    """
    if raw_status is not None and str(raw_status).strip().lower() not in ("", "nan", "none"):
        text = str(raw_status).strip().lower()
        if any(
            k in text
            for k in (
                "trả hàng", "đổi trả", "khách trả", "hoàn trả", "hoàn hàng",
                "hàng trả về", "trả lại", "returned", "tra hang", "doi tra", "tra lai",
            )
        ):
            return "returned"
        if any(k in text for k in ("huỷ", "hủy", "cancel", "không giao được", "khong giao duoc")):
            return "cancelled"
        if any(
            k in text
            for k in (
                "đang giao", "chờ giao", "đang vận chuyển", "đang xử lý",
                "đang gửi", "đang ship", "shipping", "delivering", "dang giao",
            )
        ):
            return "delivering"
        return "completed"

    if raw_note is not None and str(raw_note).strip().lower() not in ("", "nan", "none"):
        text = str(raw_note).strip().lower()
        if any(
            k in text
            for k in (
                "trả hàng", "trả lại", "khách trả lại", "khách mang trả",
                "đổi trả", "hoàn trả", "hoàn hàng", "hàng trả về",
            )
        ):
            return "returned"
        if any(
            k in text
            for k in (
                "huỷ đơn", "hủy đơn", "đã huỷ", "đã hủy",
                "khách huỷ", "khách hủy", "đơn huỷ", "đơn hủy", "cancel",
            )
        ):
            return "cancelled"
        if any(k in text for k in ("đang giao", "chờ giao", "đang vận chuyển", "đang gửi", "đang ship")):
            return "delivering"

    return "completed"


def detect_business_domain(table_names: List[str], all_columns: List[str], readme_text: str = "") -> str:
    """Return primary domain identifier for a business package."""
    t_set = {t.lower() for t in table_names}
    c_set = {c.lower() for c in all_columns}
    txt = (readme_text + " " + " ".join(table_names) + " " + " ".join(all_columns)).lower()
    if "sosuaxe" in t_set or "danhsachtho" in t_set:
        return "auto_repair"
    if "imei" in c_set or "thumaycu" in t_set or "trả góp" in t_set or "ghi tắt" in t_set:
        return "phone_shop"
    if "tên bé" in c_set or "số / size" in c_set or "khuyenmai" in t_set:
        return "mother_baby"
    if any(t in t_set for t in ("danhsachthuoc", "banhang_thuoc", "thong_tin_thuoc")):
        return "pharmacy"
    if any(t in t_set for t in ("sogiaohang", "sohanno", "bảng kê phiếu giao hàng", "chi tiết công nợ theo hạn")):
        return "construction"
    if any(k in txt for k in ("cà phê", "trà đào", "bạc xỉu", "topping", "trà sữa", "soda", "croissant", "tiramisu")):
        return "cafe"
    if any(k in c_set for k in ("bàn", "phòng/bàn", "khu vực")):
        return "restaurant"
    return "grocery"



class RulebookEngine:
    """Loads `configs/compiled_rulebook.yaml` and raw `rulebook.md` to provide:
    1. Domain detection for any business.
    2. Prompt-optimized rule slices for LLM calls on Kaggle T4 x2.
    """

    def __init__(
        self,
        compiled_yaml_path: Path | str = "configs/compiled_rulebook.yaml",
        raw_rulebook_path: Optional[Path | str] = None,
    ) -> None:
        self.compiled_path = Path(compiled_yaml_path)
        if self.compiled_path.exists():
            with open(self.compiled_path, "r", encoding="utf-8") as f:
                self.config: Dict[str, Any] = yaml.safe_load(f) or {}
        else:
            self.config = {}

        self.raw_rulebook = ""
        if raw_rulebook_path and Path(raw_rulebook_path).exists():
            self.raw_rulebook = Path(raw_rulebook_path).read_text(encoding="utf-8")

    def detect_domains(self, readme_text: str, file_names: List[str]) -> List[str]:
        """Detect applicable industry domains from business README and file names."""
        txt = (readme_text + " " + " ".join(file_names)).lower()
        domains: List[str] = []
        if any(k in txt for k in ("sửa xe", "sosuaxe", "danhsachtho", "nhớt", "biển số")):
            domains.append("auto_repair")
        if any(k in txt for k in ("imei", "thumaycu", "thu máy cũ", "trả góp", "cửa hàng điện thoại")):
            domains.append("phone_shop")
        if any(k in txt for k in ("mẹ và bé", "tên bé", "bỉm", "số / size")):
            domains.append("mother_baby")
        if any(k in txt for k in ("thuốc kê đơn", "hoạt chất", "danhsachthuoc", "banhang_thuoc", "thong_tin_thuoc")):
            domains.append("pharmacy")
        if any(k in txt for k in ("xi măng", "sắt thép", "cát đá", "lần giao", "lượt giao", "sogiaohang", "phiếu giao hàng")):
            domains.append("construction")
        if any(k in txt for k in ("cà phê", "trà đào", "bạc xỉu", "topping", "trà sữa", "soda")):
            domains.append("cafe")
        if any(k in txt for k in ("phí phục vụ", "phòng lạnh", "phòng/bàn")):
            domains.append("restaurant")
        if not domains:
            domains.append("grocery")
        return domains

    def build_llm_rule_prompt(
        self, domains: List[str], question: str, as_of_date: str
    ) -> str:
        """Construct a compact, zero-noise rulebook prompt tailored to the business domain
        and question keywords, maximizing accuracy within an 8K context budget on Kaggle T4 x2.
        """
        q_low = question.lower()
        lines = [
            f"NGÀY HÔM NAY (as_of_date): {as_of_date}. Mọi bản ghi có ngày > {as_of_date} coi như chưa xảy ra (trừ trạng thái huỷ/trả của đơn vẫn giữ nguyên).",
            "QUY TẮC CỐT LÕI:",
            "1. Đơn tính vào DOANH THU và SỐ ĐƠN: chỉ lấy đơn KHÔNG HUỶ và KHÔNG TRẢ HÀNG (tức 'completed' hoặc 'delivering'). Đơn trả 1 phần vẫn bỏ cả đơn.",
            "2. DOANH THU 1 ĐƠN = tổng Thành tiền các dòng hàng (đã trừ giảm giá dòng). KHÔNG trừ giảm giá tổng đơn/voucher, KHÔNG cộng VAT, KHÔNG gồm phí giao hàng.",
            "3. Dòng 'Phí phục vụ' KHÔNG phải mặt hàng (không đếm/xếp hạng mặt hàng), nhưng VẪN cộng vào doanh thu.",
            "4. BÁN CHẠY / BÁN NHIỀU NHẤT = tổng số lượng quy về ĐƠN VỊ CƠ BẢN (bỏ đơn huỷ, trả).",
            "5. LÀM TRÒN: Tiền và trung bình tiền làm tròn về đồng (nửa đồng làm tròn lên). Phần trăm và số trung bình khác làm tròn 2 chữ số thập phân.",
            "6. XẾP HẠNG BẰNG NHAU (TIE-BREAK): xếp theo MÃ tăng dần (nếu không có mã thì theo TÊN tăng dần).",
        ]

        if any(k in q_low for k in ("nợ", "thiếu", "trả góp", "mua chịu", "thu hồi")):
            lines.append(
                "7. CÔNG NỢ: tính đơn không huỷ (ĐƠN TRẢ HÀNG VẪN TÍNH nếu còn nợ > 0); bỏ đơn COD và Sàn TMĐT. Nợ 1 đơn tại ngày X = Tổng khách phải trả cả đơn (đã trừ giảm giá đơn, gồm phí giao) - Tổng tiền đã trả đến hết ngày X. Nợ thêm trong kỳ = phần chưa trả ngay trong ngày mua của đơn ghi nợ đặt trong kỳ. Thu/trả nợ trong kỳ = khoản trả SAU ngày mua có ngày trả trong kỳ."
            )
        if any(k in q_low for k in ("nhập", "tồn", "kho", "lãi", "giá vốn", "ế", "hết hàng", "tăng giá")):
            lines.append(
                "8. KHO & GIÁ VỐN: Tiền nhập = tổng tiền phiếu nhập có ngày trong kỳ (bỏ phiếu huỷ, tính cả nhập đầu kỳ). NCC nhập nhiều nhất = tổng tiền nhập cao nhất; NCC chính = nhiều phiếu nhập nhất. Lãi gộp = Doanh thu - tổng (số lượng bán * giá vốn 1 ĐV bán ghi trên dòng bán). Lãi ròng tháng = Lãi gộp tháng - tổng chi phí trong sổ chi phí tháng đó."
            )

        domain_cfg = self.config.get("domain_rules", {})
        for dom in domains:
            d_info = domain_cfg.get(dom)
            if d_info and d_info.get("rules"):
                lines.append(f"QUY TẮC NGÀNH ({dom.upper()}):")
                for r in d_info["rules"]:
                    lines.append(f"- {r}")

        lines.append(
            "KHÔNG TRẢ LỜI ĐƯỢC (unanswerable): Khi kỳ hỏi nằm HOÀN TOÀN ngoài khoảng ngày có dữ liệu của shop, hoặc shop không ghi cột/bảng thông tin câu hỏi cần, hoặc hỏi đánh giá chủ quan/kiến thức bên ngoài -> trả về type='unanswerable' (KHÔNG trả về 0)."
        )
        return "\n".join(lines)
