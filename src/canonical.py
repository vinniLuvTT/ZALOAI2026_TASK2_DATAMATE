"""Canonical Business Data Model normalizer for DataMate across all 44 businesses."""

from __future__ import annotations

import os
import pickle
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from src.loader import BusinessPackage, load_business_package
from src.rule_engine import detect_business_domain, normalize_order_status
from src.temporal import parse_flexible_date, parse_flexible_hour
from src.value_parser import parse_qty_and_unit, parse_vn_number


def strip_accents(text: str) -> str:
    if not text:
        return ""
    s = str(text).replace("đ", "d").replace("Đ", "D")
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def norm_key(text: Any) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = str(text).strip()
    if not s or s.lower() in ("none", "nan", "null", "#value!"):
        return ""
    s = re.sub(r"\s+", " ", s)
    return strip_accents(s).lower().strip()


def clean_str(text: Any) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = str(text).strip()
    if s.lower() in ("none", "nan", "null", "#value!"):
        return ""
    return re.sub(r"\s+", " ", s)


QUESTION_TOKEN_MAP: Dict[str, str] = {
    # Un-composed Telex tokens
    "bans": "bán", "bawngf": "bằng", "bieets": "biết", "bij": "bị", "binhf": "bình",
    "bowts": "bớt", "chayj": "chạy", "chieems": "chiếm", "chiuj": "chịu", "conf": "còn",
    "coongj": "cộng", "cos": "có", "cuar": "của", "cungf": "cùng", "cuoois": "cuối",
    "cuwar": "cửa", "ddax": "đã", "ddeems": "đếm", "ddeens": "đến", "ddieenj": "điện",
    "ddoof": "đồ", "ddooir": "đổi", "ddos": "đó", "ddown": "đơn", "dduowcj": "được",
    "dduwowcj": "được", "duocj": "được", "duowc": "được", "ees": "ế", "gaanf": "gần",
    "giamr": "giảm", "gias": "giá", "giumf": "giùm", "giups": "giúp", "hangf": "hàng",
    "heets": "hết", "hoir": "hỏi", "hoom": "hôm", "hown": "hơn", "huyr": "huỷ",
    "its": "ít", "khachs": "khách", "khoong": "không", "laf": "là", "laij": "lại",
    "laix": "lãi", "leej": "lệ", "leen": "lên", "luwowtj": "lượt", "maays": "mấy",
    "mawtj": "mặt", "mooix": "mỗi", "mootj": "một", "naof": "nào", "naoo": "nào",
    "nawm": "năm", "nayf": "này", "ngayf": "ngày", "nguwowif": "người",
    "nhaapj": "nhập", "nhaats": "nhất", "nhaf": "nhà", "nhieeu": "nhiều",
    "nhieeuf": "nhiều", "nhieuf": "nhiều", "nhir": "nhỉ", "nhoms": "nhóm", "nowj": "nợ",
    "phaanf": "phần", "quaan": "quân", "quys": "quý", "rooif": "rồi", "sangs": "sáng",
    "soos": "số", "tawng": "tăng", "thais": "thái", "thangs": "tháng", "thoaij": "thoại",
    "thuoocj": "thuộc", "thuoocs": "thuốc", "tieenf": "tiền", "tieeu": "tiêu",
    "tinhs": "tính", "tir": "tỉ", "tooi": "tôi", "toonf": "tồn", "toongr": "tổng",
    "trar": "trả", "trawm": "trăm", "treen": "trên", "trowr": "trở", "truwowcs": "trước",
    "tuwf": "từ", "vaayj": "vậy", "vaf": "và", "vaof": "vào", "veef": "về",
    "vowis": "với", "xuaats": "xuất",
    # Chat shorthand & keyboard typos
    "dthu": "doanh thu", "doanhthu": "doanh thu", "revenue": "doanh thu",
    "bnhiu": "bao nhiêu", "bnhui": "bao nhiêu", "bannhieu": "bao nhiêu", "bnhieu": "bao nhiêu",
    "nhiu": "nhiều", "nheiu": "nhiều", "nhìu": "nhiều",
    "caonhat": "cao nhất", "ngaynao": "ngày nào", "nhieudon": "nhiều đơn",
    "nhta": "nhất", "nhut": "nhất", "nhứt": "nhất",
    "hagn": "hàng", "bih": "bình", "tih": "tính", "tog": "tổng", "cog": "cộng",
    "trog": "trong", "dag": "đang", "nhug": "những", "nguog": "ngưỡng",
    "duco": "được", "duot": "được", "dc": "được", "đc": "được",
    "trc": "trước", "hnay": "hôm nay", "hqua": "hôm qua",
    "ko": "không", "hok": "không", "kon": "không",
    "kua": "của", "cko": "cho", "jum": "giùm", "jup": "giúp", "dup": "giúp",
    "lenn": "lên", "lne": "lên", "rui": "rồi", "tug": "từng",
    "mun": "muốn", "mún": "muốn", "bit": "biết", "bít": "biết",
    "zia": "về", "zo": "vào", "vs": "với",
}

_QUESTION_TOKEN_RE = re.compile(
    r"\b(" + "|".join(sorted(QUESTION_TOKEN_MAP.keys(), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def normalize_question_text(text: str) -> str:
    """Normalize un-composed Telex tokens and Vietnamese chat shorthand in a question string."""
    s = clean_str(text)
    if not s:
        return ""
    s = _QUESTION_TOKEN_RE.sub(lambda m: QUESTION_TOKEN_MAP.get(m.group(1).lower(), m.group(0)), s)
    # Phrase-level typo fixes
    s = re.sub(r"\btăng\s+dảm\b", "tăng giảm", s, flags=re.IGNORECASE)
    s = re.sub(r"\btang\s+dam\b", "tăng giảm", s, flags=re.IGNORECASE)
    s = re.sub(r"\bxo\s+với\b", "so với", s, flags=re.IGNORECASE)
    s = re.sub(r"\bxo\s+voi\b", "so với", s, flags=re.IGNORECASE)
    s = re.sub(r"\bdào\s+mấy\s+giờ\b", "vào mấy giờ", s, flags=re.IGNORECASE)
    s = re.sub(r"\bhông\s+kê\s+đơn\b", "không kê đơn", s, flags=re.IGNORECASE)
    s = re.sub(r"\bmí\s*%", "mấy %", s, flags=re.IGNORECASE)
    return s


def extract_phone_and_clean_name(raw_customer: Any) -> Tuple[str, str]:
    """Return (clean_name, phone_digits) from a raw customer cell."""
    s = clean_str(raw_customer)
    if not s:
        return "", ""
    m = re.search(r"(0\d[\d\s\.]{7,11})", s)
    phone = re.sub(r"\D", "", m.group(1)) if m else ""
    name = re.sub(r"[\s\-\(\)]*0\d[\d\s\.]{7,11}\)?", "", s).strip(" -()")
    name = re.sub(r"\s+", " ", name).strip()
    return name, phone


WALKIN_TOKENS = ("", "kl", "le", "khach le", "vang lai", "khach vang lai", "khach")


def is_walkin_customer(name: str, code: str = "") -> bool:
    if code and clean_str(code):
        nk_code = norm_key(code)
        if nk_code in WALKIN_TOKENS:
            return True
        # Valid customer code (e.g. KH0001): not walk-in unless name explicitly says walk-in
        nk_nm = norm_key(name)
        if nk_nm and nk_nm in ("kl", "le", "khach le", "vang lai", "khach vang lai"):
            return True
        return False
    nk = norm_key(name)
    if nk in WALKIN_TOKENS:
        return True
    return False


def extract_phone_model_and_brand(product_name: str, is_phone: bool) -> Tuple[Optional[str], Optional[str]]:
    if not is_phone or not product_name:
        return None, None
    s = clean_str(product_name)
    # Dòng máy: part before '<digits>GB' (Rulebook §6 Cửa hàng điện thoại)
    m = re.split(r"\s+\d+\s*GB\b", s, maxsplit=1, flags=re.IGNORECASE)
    model = m[0].strip() if m else s
    # Normalize casing of known phone brands/models
    first_word = s.split()[0] if s.split() else ""
    fw_lower = first_word.lower()
    if fw_lower == "iphone":
        brand = "Apple"
    elif fw_lower == "samsung":
        brand = "Samsung"
    elif fw_lower == "xiaomi":
        brand = "Xiaomi"
    elif fw_lower == "oppo":
        brand = "OPPO"
    elif fw_lower == "nokia":
        brand = "Nokia"
    elif fw_lower == "realme":
        brand = "realme"
    elif fw_lower == "vivo":
        brand = "vivo"
    else:
        brand = first_word
    return model, brand


def normalize_vehicle_model_and_plate(raw_xe: Any) -> Tuple[str, str]:
    """Return (canonical_vehicle_model, license_plate) from SoSuaXe['Xe']."""
    s = clean_str(raw_xe)
    if not s:
        return "", ""
    m = re.search(r"(\d{2}[A-Z]\d?-\d{3}\.\d{2})", s, flags=re.IGNORECASE)
    plate = m.group(1).upper() if m else ""
    raw_model = s[: m.start()].strip() if m else s
    mk = norm_key(raw_model)
    if mk in ("ab", "air blade", "airblade"):
        model = "Air Blade"
    elif mk in ("vision", "vision 110"):
        model = "Vision"
    elif mk in ("wave alpha", "alpha", "wave α", "wave a"):
        model = "Wave Alpha"
    elif mk in ("rsx", "wave rsx"):
        model = "Wave RSX"
    elif mk in ("lead",):
        model = "Lead"
    elif mk in ("sirius", "sirius fi"):
        model = "Sirius"
    elif mk in ("exciter", "ex"):
        model = "Exciter"
    elif mk in ("future", "future 125"):
        model = "Future"
    elif mk in ("sh mode", "shmode"):
        model = "SH Mode"
    elif mk in ("vario", "vario 125"):
        model = "Vario"
    elif mk in ("janus",):
        model = "Janus"
    elif mk in ("grande",):
        model = "Grande"
    elif mk in ("winner", "winner x"):
        model = "Winner"
    else:
        model = raw_model
    return model, plate


def normalize_payment_method(raw_val: Any) -> str:
    nk = norm_key(raw_val)
    if not nk:
        return ""
    if nk in ("tm", "tien mat", "cash"):
        return "cash"
    if nk in ("ck", "chuyen khoan", "transfer", "ngan hang"):
        return "transfer"
    if nk in ("vi dien tu", "ewallet", "momo", "zalopay", "vnpay", "vi"):
        return "ewallet"
    if nk in ("cod", "giao hang thu tien", "thu ho") or "cod" in nk or "thu ho" in nk:
        return "cod"
    if nk in ("no", "ghi no", "debt", "cong no", "mua chiu", "ban chiu", "tra sau"):
        return "debt"
    if nk in ("the", "card", "quet the"):
        return "card"
    return nk


def normalize_channel(raw_val: Any) -> str:
    nk = norm_key(raw_val)
    if not nk:
        return ""
    if nk in ("tai quay", "tai tiem", "tai quan", "truc tiep", "ban truc tiep", "pos", "store", "off", "offline"):
        return "in_store"
    if "zalo" in nk:
        return "zalo"
    if nk in ("dien thoai", "qua dien thoai", "goi dien", "phone", "tel"):
        return "phone"
    if any(k in nk for k in ("san", "tmdt", "thuong mai dien tu", "shopee", "lazada", "tiktok")):
        return "ecommerce"
    if any(k in nk for k in ("app", "ung dung", "grab", "befood", "shopeefood", "delivery")):
        return "delivery_app"
    if "web" in nk or "facebook" in nk or "fb" in nk:
        return "online"
    return nk


def normalize_customer_group(raw_val: Any) -> str:
    nk = norm_key(raw_val)
    if not nk:
        return ""
    if "than thiet" in nk or nk in ("vip", "member", "thanh vien", "khach than thiet"):
        return "than_thiet"
    if nk in ("si", "khach si", "ban si", "nhom si", "wholesale") or "khach si" in nk or "nhom si" in nk:
        return "si"
    if nk in ("le", "khach le", "ban le", "nhom le", "retail"):
        return "le"
    if "nha thau" in nk or "thau" in nk:
        return "nha_thau"
    if "cong trinh" in nk:
        return "cong_trinh"
    if nk in ("tho", "nhom tho"):
        return "tho"
    return nk


def normalize_gender(raw_val: Any) -> str:
    nk = norm_key(raw_val)
    if nk in ("nam", "male", "m"):
        return "nam"
    if nk in ("nu", "female", "f"):
        return "nu"
    return ""


@dataclass
class CanonicalBusiness:
    business_id: str
    split: str
    domain: str
    data_start_date: Optional[date]
    data_end_date: Optional[date]
    capabilities: Dict[str, bool] = field(default_factory=dict)

    # Canonical DataFrames
    products: pd.DataFrame = field(default_factory=pd.DataFrame)
    customers: pd.DataFrame = field(default_factory=pd.DataFrame)
    orders: pd.DataFrame = field(default_factory=pd.DataFrame)
    order_lines: pd.DataFrame = field(default_factory=pd.DataFrame)
    imports: pd.DataFrame = field(default_factory=pd.DataFrame)
    import_lines: pd.DataFrame = field(default_factory=pd.DataFrame)
    batches: pd.DataFrame = field(default_factory=pd.DataFrame)
    batch_sales: pd.DataFrame = field(default_factory=pd.DataFrame)
    payments: pd.DataFrame = field(default_factory=pd.DataFrame)
    installments: pd.DataFrame = field(default_factory=pd.DataFrame)
    installment_payments: pd.DataFrame = field(default_factory=pd.DataFrame)
    warranties: pd.DataFrame = field(default_factory=pd.DataFrame)
    deliveries: pd.DataFrame = field(default_factory=pd.DataFrame)
    due_dates: pd.DataFrame = field(default_factory=pd.DataFrame)
    repair_tickets: pd.DataFrame = field(default_factory=pd.DataFrame)
    mechanics: pd.DataFrame = field(default_factory=pd.DataFrame)
    price_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    expenses: pd.DataFrame = field(default_factory=pd.DataFrame)
    stock_snapshot: pd.DataFrame = field(default_factory=pd.DataFrame)
    stock_movements: pd.DataFrame = field(default_factory=pd.DataFrame)
    trade_ins: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Lookup dictionaries for fast entity linking & conversion
    unit_factors: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (prod_key, norm_unit) -> factor
    prod_base_unit: Dict[str, str] = field(default_factory=dict)              # prod_key -> base_unit
    prod_name_to_key: Dict[str, str] = field(default_factory=dict)            # norm_alias -> prod_key
    prod_key_to_name: Dict[str, str] = field(default_factory=dict)            # prod_key -> canonical display name
    prod_key_to_code: Dict[str, Optional[str]] = field(default_factory=dict)
    cust_name_to_key: Dict[str, str] = field(default_factory=dict)
    cust_key_to_name: Dict[str, str] = field(default_factory=dict)
    cust_key_to_code: Dict[str, Optional[str]] = field(default_factory=dict)


def _first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    col_map = {c.strip(): c for c in df.columns}
    for cand in candidates:
        if cand in col_map:
            return col_map[cand]
    # Case-insensitive fallback
    col_map_lower = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in col_map_lower:
            return col_map_lower[cand.lower()]
    return None


def _is_footer_str(val: Any) -> bool:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return False
    s = str(val).strip().lower()
    return s.startswith((
        "tổng ngày", "tong ngay", "tổng tháng", "tong thang",
        "tổng cộng", "tong cong", "tổng kết", "tong ket",
        "ghi chú:", "ghi chu:",
    ))


def _drop_footer_and_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mask = pd.Series(True, index=df.index)
    data_cols = [c for c in df.columns if not c.startswith("_")]
    for col in data_cols:
        s = df[col]
        # Repeated header check
        col_lower = col.strip().lower()
        is_hdr = s.astype(str).str.strip().str.lower() == col_lower
        is_ftr = s.apply(_is_footer_str)
        mask &= ~(is_hdr | is_ftr)
    return df[mask].copy()


def build_canonical_business(pkg: BusinessPackage) -> CanonicalBusiness:
    t_names = list(pkg.tables.keys())
    all_cols = [c for df in pkg.tables.values() for c in df.columns if not c.startswith("_")]
    domain = detect_business_domain(t_names, all_cols, pkg.readme_text)

    r_end = getattr(pkg, "readme_end_date", None) or getattr(pkg, "data_end_date", None)
    cb = CanonicalBusiness(
        business_id=pkg.business_id,
        split=pkg.split,
        domain=domain,
        data_start_date=None,
        data_end_date=parse_flexible_date(r_end) if r_end else None,
    )

    # Clean footer/repeated header rows from all tables upfront
    tables: Dict[str, pd.DataFrame] = {
        k: _drop_footer_and_header_rows(v) for k, v in pkg.tables.items()
    }
    if "BanHang" in tables and "BanHang_TongHop" in tables:
        tables["BanHang"] = pd.concat([tables["BanHang"], tables["BanHang_TongHop"]], ignore_index=True)
        del tables["BanHang_TongHop"]

    # 1. Build Products & Unit Conversions
    _build_products_and_units(cb, pkg, tables)

    # 2. Build Customers master table (if present)
    _build_customers(cb, tables)

    # 3. Special Auto Repair domain (dev_012)
    if "SoSuaXe" in tables:
        _build_auto_repair(cb, tables)
        _finalize_dates_and_capabilities(cb, tables)
        return cb

    # 4. Build Batch Sales & Returns first (needed to mark returned orders in pharmacies)
    _build_batches_and_batch_sales(cb, tables)

    # 5. Build Orders & Order Lines
    _build_orders_and_lines(cb, tables)

    # 6. Build Imports & Import Lines
    _build_imports(cb, tables)

    # 7. Build Stock Snapshot & Stock Movements
    _build_stock(cb, tables)

    # 8. Build Payments, Debts, Due Dates, Installments, Warranties, Deliveries
    _build_financials_and_aux(cb, tables)

    # 9. Finalize date bounds and capability flags
    _finalize_dates_and_capabilities(cb, tables)
    return cb


def _build_products_and_units(
    cb: CanonicalBusiness, pkg: BusinessPackage, tables: Dict[str, pd.DataFrame]
) -> None:
    prod_rows: List[Dict[str, Any]] = []
    alias_map: Dict[str, str] = {}
    code_to_key: Dict[str, str] = {}

    # Identify product catalog table
    cat_tname = None
    for cand in (
        "Danh sách hàng hóa", "DanhSachSanPham", "san_pham",
        "BanHang_HangHoa", "danh_muc_hang", "DanhMucHang",
    ):
        if cand in tables:
            cat_tname = cand
            break
    if cat_tname is None and "DanhMuc" in tables and "Số / Size" in tables["DanhMuc"].columns:
        cat_tname = "DanhMuc"
    if cat_tname is None and "Ghi tắt" in tables:
        cat_tname = "Ghi tắt"

    # Pharmacy drug info table
    drug_info: Dict[str, Tuple[str, bool]] = {}
    for dt_name in ("DanhSachThuoc", "thong_tin_thuoc", "BanHang_Thuoc"):
        if dt_name in tables:
            dt = tables[dt_name]
            c_code = _first_present(dt, ["Mã hàng", "Mã SKU", "MaHangHoa", "Mã HH"])
            c_ing = _first_present(dt, ["Hoạt chất", "ThanhPhanChinh"])
            c_rx = _first_present(dt, ["Thuốc kê đơn", "BanTheoDon", "Kê đơn"])
            if c_code:
                for _, r in dt.iterrows():
                    code = clean_str(r[c_code])
                    ing = clean_str(r[c_ing]) if c_ing else ""
                    rx_raw = str(r[c_rx]).strip().lower() if c_rx else ""
                    is_rx = rx_raw in ("1", "true", "có", "co", "kê đơn", "ke don", "yes")
                    if code:
                        drug_info[code] = (ing, is_rx)

    has_product_code = False
    has_product_category = False
    has_cost_price = False

    if cat_tname is not None:
        df_cat = tables[cat_tname]
        c_code = _first_present(
            df_cat,
            ["Mã hàng hóa", "Mã hàng", "Mã SKU", "MaHangHoa", "ma_sp", "Mã H.hóa", "Mã sản phẩm bán", "SKU", "Mã HH"],
        )
        c_name = _first_present(
            df_cat,
            ["Tên hàng hóa", "Tên hàng", "Tên sản phẩm", "TenGoi", "Món", "Hàng bán", "Hàng", "ten_hang", "Sản phẩm", "Tên SP"],
        )
        c_cat = _first_present(
            df_cat,
            ["Nhóm hàng hóa", "Nhóm hàng(3 Cấp)", "DanhMuc", "Phân loại", "Nhóm", "Loại hàng", "Nhóm hàng", "Nhom hang", "Danh mục", "Loại sản phẩm", "Loại"],
        )
        c_brand = _first_present(df_cat, ["Thương hiệu", "thuong_hieu", "Hãng"])
        c_unit = _first_present(df_cat, ["Đơn vị tính", "ĐVT", "Đơn vị", "DonViCoSo", "DVT", "Đơn vị cơ bản"])
        c_cost = _first_present(df_cat, ["Giá vốn", "GiaVonHang", "Giá vốn hiện tại"])
        c_price = _first_present(df_cat, ["Giá bán", "GiaBanLe", "Giá", "Giá lẻ", "Giá hiện tại", "Giá niêm yết"])
        c_size = _first_present(df_cat, ["Số / Size"])
        c_alias = _first_present(df_cat, ["Ghi tắt", "Tên gọi khác"])
        c_quydoi = _first_present(df_cat, ["Quy đổi"])

        if c_code is not None:
            has_product_code = True
        if c_cat is not None and df_cat[c_cat].dropna().astype(str).str.strip().replace("", np.nan).notna().any():
            has_product_category = True
        if c_cost is not None and df_cat[c_cost].dropna().notna().any():
            has_cost_price = True

        for _, r in df_cat.iterrows():
            name = clean_str(r[c_name]) if c_name else ""
            code = clean_str(r[c_code]) if c_code else ""
            if not name and not code:
                continue
            pkey = code if code else norm_key(name)
            cat_val = clean_str(r[c_cat]) if c_cat else ""
            brand_val = clean_str(r[c_brand]) if c_brand else ""
            unit_val = clean_str(r[c_unit]) if c_unit else ""
            cost_val = parse_vn_number(r[c_cost]) if c_cost else None
            price_val = parse_vn_number(r[c_price]) if c_price else None
            size_val = clean_str(r[c_size]) if c_size else ""
            ing_val, rx_val = drug_info.get(code, ("", False))

            is_phone = cb.domain == "phone_shop" and norm_key(cat_val) in ("may moi", "may cu")
            phone_model, phone_brand = extract_phone_model_and_brand(name, is_phone)
            if phone_brand and not brand_val:
                brand_val = phone_brand

            aliases = [name]
            if code:
                aliases.append(code)
                code_to_key[code] = pkey
            if c_alias and clean_str(r[c_alias]):
                for al in str(r[c_alias]).split(","):
                    al_s = clean_str(al)
                    if al_s:
                        aliases.append(al_s)

            prod_rows.append({
                "product_key": pkey,
                "product_code": code or None,
                "product_name": name,
                "category": cat_val or None,
                "category_norm": norm_key(cat_val),
                "brand": brand_val or None,
                "brand_norm": norm_key(brand_val),
                "phone_model": phone_model,
                "phone_model_norm": norm_key(phone_model) if phone_model else "",
                "milk_or_diaper_size": size_val or None,
                "size_norm": norm_key(size_val),
                "base_unit": unit_val or None,
                "base_unit_norm": norm_key(unit_val),
                "cost_price": cost_val,
                "sell_price": price_val,
                "active_ingredient": ing_val or None,
                "active_ingredient_norm": norm_key(ing_val),
                "is_prescription": rx_val,
                "is_service_fee": norm_key(name) == "phi phuc vu",
            })

            if unit_val:
                cb.prod_base_unit[pkey] = unit_val
                cb.unit_factors[(pkey, norm_key(unit_val))] = 1.0
            cb.prod_key_to_name[pkey] = name
            cb.prod_key_to_code[pkey] = code or None
            for al in aliases:
                nk = norm_key(al)
                if nk:
                    alias_map[nk] = pkey

            # Inline 'Quy đổi' in dev_013 DanhMuc (e.g. '1 thùng = 4 bịch')
            if c_quydoi and clean_str(r[c_quydoi]):
                qd_str = clean_str(r[c_quydoi])
                m_qd = re.search(r"1\s+([^\d=]+?)\s*=\s*(\d+(?:[\.,]\d+)?)\s+([^\d=]+)", qd_str)
                if m_qd:
                    lg_u = norm_key(m_qd.group(1))
                    factor = float(m_qd.group(2).replace(",", "."))
                    cb.unit_factors[(pkey, lg_u)] = factor

    # Load dedicated unit conversion tables
    raw_conv_list: List[Tuple[str, str, Any, str]] = []  # (pkey, large_unit_norm, raw_factor, base_unit_str)
    for ut_name in ("DanhSachQuyDoi", "DanhSachDonViTinh", "don_vi_sp", "BanHang_DonViQuyDoi", "DanhMuc"):
        if ut_name not in tables or ut_name == cat_tname:
            continue
        ut = tables[ut_name]
        c_pcode = _first_present(ut, ["Mã hàng", "MaHangHoa", "Mã sp", "Mã vạch"])
        c_pname = _first_present(ut, ["Tên sản phẩm", "Tên món", "Hàng"])
        c_lg = _first_present(ut, ["Đơn vị quy đổi", "ĐVT", "DonViTinh", "Đơn vị", "Quy cách", "ĐV nhập", "Loại ĐV", "ĐV lớn"])
        c_fac = _first_present(ut, ["Tỉ lệ quy đổi", "Quy đổi", "HeSoQuyDoi", "Bằng", "Hệ số", "Hệ số QĐ"])
        c_base = _first_present(ut, ["Đơn vị cơ bản", "Đơn vị gốc", "ĐV gốc", "ĐV nhỏ"])
        if not c_lg or not c_fac:
            continue
        for _, r in ut.iterrows():
            pkey = None
            if c_pcode and clean_str(r[c_pcode]):
                code = clean_str(r[c_pcode])
                pkey = code_to_key.get(code, code)
            elif c_pname and clean_str(r[c_pname]):
                pname = clean_str(r[c_pname])
                nk = norm_key(pname)
                pkey = alias_map.get(nk, nk)
                if pkey not in cb.prod_key_to_name:
                    cb.prod_key_to_name[pkey] = pname
                    alias_map[nk] = pkey
            if not pkey:
                continue
            lg_u = norm_key(r[c_lg])
            base_u = clean_str(r[c_base]) if c_base else ""
            if base_u:
                cb.prod_base_unit[pkey] = base_u
                cb.unit_factors[(pkey, norm_key(base_u))] = 1.0
            raw_conv_list.append((pkey, lg_u, r[c_fac], base_u))

    # First pass: numeric conversion factors
    chained_conv: List[Tuple[str, str, float, str]] = []
    for pkey, lg_u, raw_fac, _ in raw_conv_list:
        s_fac = clean_str(raw_fac)
        m_ch = re.match(r"^(\d+(?:[\.,]\d+)?)\s+([a-zA-ZÀ-ỹ]+)$", s_fac)
        if m_ch:
            chained_conv.append((pkey, lg_u, float(m_ch.group(1).replace(",", ".")), norm_key(m_ch.group(2))))
        else:
            fval = parse_vn_number(raw_fac)
            if fval and fval > 0:
                cb.unit_factors[(pkey, lg_u)] = float(fval)

    # Second pass: chained conversion factors (e.g. dev_005 '15 vỉ' -> 15 * 12 = 180)
    for pkey, lg_u, mult, mid_u in chained_conv:
        mid_fac = cb.unit_factors.get((pkey, mid_u), 1.0)
        cb.unit_factors[(pkey, lg_u)] = mult * mid_fac

    # Inline unit conversions from README.md
    for rule in getattr(pkg, "inline_unit_rules", []):
        prod_name = rule.product_name
        nk = norm_key(prod_name)
        pkey = alias_map.get(nk, nk)
        if pkey not in cb.prod_key_to_name:
            cb.prod_key_to_name[pkey] = prod_name
            alias_map[nk] = pkey
        lg_u = norm_key(rule.large_unit)
        base_u = clean_str(rule.base_unit)
        fac = float(rule.factor)
        cb.unit_factors[(pkey, lg_u)] = fac
        if base_u and pkey not in cb.prod_base_unit:
            cb.prod_base_unit[pkey] = base_u
            cb.unit_factors[(pkey, norm_key(base_u))] = 1.0

    # PVC pipe bundle rule (Rulebook & README: Ống PVC Bình Minh D32/D50/D90 cây 4m: 1 bó = 10 cây)
    for nk, pkey in list(alias_map.items()):
        if "ong pvc binh minh" in nk:
            cb.unit_factors[(pkey, "bo")] = 10.0
            if pkey not in cb.prod_base_unit:
                cb.prod_base_unit[pkey] = "cây"
                cb.unit_factors[(pkey, "cay")] = 1.0

    cb.prod_name_to_key = alias_map
    cb.products = pd.DataFrame(prod_rows)
    cb.capabilities["has_product_code"] = has_product_code
    cb.capabilities["has_product_category"] = has_product_category
    cb.capabilities["has_cost_price"] = has_cost_price


def _build_customers(cb: CanonicalBusiness, tables: Dict[str, pd.DataFrame]) -> None:
    cust_tname = None
    for cand in ("DanhSachKhachHang", "khach_hang", "BanHang_KhachHang", "KhachHang"):
        if cand in tables:
            cust_tname = cand
            break

    if cust_tname is None:
        return

    df_c = tables[cust_tname]
    c_code = _first_present(df_c, ["Mã khách hàng", "MaKhachHang", "Customer ID", "Mã KH mua", "Mã KH", "ma_kh", "Mã khách", "Mã khách lẻ/sỉ"])
    c_name = _first_present(df_c, ["Tên khách hàng", "TenGoi", "ten_kh", "Khách hàng mua", "Khách hàng (tên)", "KH", "Tên KH", "Khách lẻ/sỉ", "Loại khách"])
    c_phone = _first_present(df_c, ["Điện thoại", "SoDienThoai", "Số điện thoại", "SĐT", "Phone", "Di động"])
    c_grp = _first_present(df_c, ["Nhóm khách hàng", "LoaiKhachHang", "Kiểu khách", "Loại khách", "Hạng", "Nhóm"])
    c_gen = _first_present(df_c, ["Giới tính", "NamNu"])
    c_addr = _first_present(df_c, ["Địa chỉ", "DiaChiLienHe", "Addr"])
    c_reg = _first_present(df_c, ["Ngày đăng ký", "Ngày tạo", "TaoLuc"])

    rows = []
    for _, r in df_c.iterrows():
        code = clean_str(r[c_code]) if c_code else ""
        raw_name = clean_str(r[c_name]) if c_name else ""
        clean_nm, ph_from_nm = extract_phone_and_clean_name(raw_name)
        name = clean_nm or raw_name
        phone = re.sub(r"\D", "", clean_str(r[c_phone])) if c_phone else ph_from_nm
        if not code and not name:
            continue
        ckey = code if code else (f"{norm_key(name)}|{phone}" if phone else norm_key(name))
        grp_raw = clean_str(r[c_grp]) if c_grp else ""
        gen_raw = clean_str(r[c_gen]) if c_gen else ""
        addr_raw = clean_str(r[c_addr]) if c_addr else ""
        reg_dt = parse_flexible_date(r[c_reg]) if c_reg else None

        rows.append({
            "customer_key": ckey,
            "customer_code": code or None,
            "customer_name": name,
            "phone": phone,
            "customer_group": grp_raw or None,
            "customer_group_norm": normalize_customer_group(grp_raw),
            "gender": normalize_gender(gen_raw),
            "address": addr_raw or None,
            "address_norm": norm_key(addr_raw),
            "reg_date": reg_dt,
        })
        cb.cust_key_to_name[ckey] = name
        cb.cust_key_to_code[ckey] = code or None
        if code:
            cb.cust_name_to_key[norm_key(code)] = ckey
        if name:
            cb.cust_name_to_key[norm_key(name)] = ckey

    cb.customers = pd.DataFrame(rows)


def _build_auto_repair(cb: CanonicalBusiness, tables: Dict[str, pd.DataFrame]) -> None:
    # DanhSachTho
    mech_map: Dict[str, Tuple[str, float]] = {}
    if "DanhSachTho" in tables:
        df_m = tables["DanhSachTho"]
        m_rows = []
        for _, r in df_m.iterrows():
            full_name = clean_str(r.get("Họ tên"))
            if not full_name:
                continue
            comm = float(parse_vn_number(r.get("Hoa hồng (%)")) or 0.0)
            join_dt = parse_flexible_date(r.get("Ngày vào làm"))
            m_rows.append({
                "mechanic_name": full_name,
                "mechanic_norm": norm_key(full_name),
                "commission_pct": comm,
                "join_date": join_dt,
            })
            given = full_name.split()[-1]
            for alias in (full_name, given, f"a {given}", f"a.{given}", f"anh {given}"):
                mech_map[norm_key(alias)] = (full_name, comm)
        cb.mechanics = pd.DataFrame(m_rows)

    df_s = tables["SoSuaXe"]
    t_rows = []
    for idx, r in df_s.iterrows():
        s_year = int(r["_sheet_year"]) if pd.notna(r.get("_sheet_year")) else None
        dt = parse_flexible_date(r.get("Ngày"), default_year=s_year)
        raw_xe = clean_str(r.get("Xe"))
        if not dt or not raw_xe:
            continue
        model, plate = normalize_vehicle_model_and_plate(raw_xe)
        raw_tho = clean_str(r.get("Thợ"))
        mech_full, comm_pct = mech_map.get(norm_key(raw_tho), (raw_tho, 0.0))
        amt = float(parse_vn_number(r.get("Tiền")) or 0.0)
        note = clean_str(r.get("Ghi chú"))
        is_warranty = (amt == 0.0) or ("bao hanh" in norm_key(note))
        raw_work = clean_str(r.get("Làm gì"))
        work_items = [norm_key(p) for p in re.split(r"[,+]", raw_work) if clean_str(p)]

        t_rows.append({
            "ticket_id": f"TKT_{idx:06d}",
            "date": dt,
            "vehicle_raw": raw_xe,
            "vehicle_model": model,
            "vehicle_model_norm": norm_key(model),
            "license_plate": plate,
            "mechanic_name": mech_full,
            "mechanic_norm": norm_key(mech_full),
            "commission_pct": comm_pct,
            "amount": amt,
            "is_warranty": is_warranty,
            "is_paid": not is_warranty and amt > 0,
            "work_raw": raw_work,
            "work_items": work_items,
            "note": note,
        })
    cb.repair_tickets = pd.DataFrame(t_rows)


def _build_batches_and_batch_sales(cb: CanonicalBusiness, tables: Dict[str, pd.DataFrame]) -> None:
    # 1. Batch sales & returns (DanhSachHoaDonChiTietLo, DonHang_Lo, chi_tiet_don_lo, BanHang_BanTheoLo)
    bs_tname = None
    for cand in ("DanhSachHoaDonChiTietLo", "DonHang_Lo", "chi_tiet_don_lo", "BanHang_BanTheoLo"):
        if cand in tables:
            bs_tname = cand
            break

    if bs_tname is not None:
        df_bs = tables[bs_tname]
        c_oid = _first_present(df_bs, ["Mã hóa đơn", "Mã đơn hàng", "MaHoaDon"])
        c_lot = _first_present(df_bs, ["Lô", "Mã lô", "MaLo"])
        c_qty = _first_present(df_bs, ["Số lượng (ĐVT cơ bản)", "Số lượng", "SLQuyDoi", "SL quy ra lẻ"])
        c_ret = _first_present(df_bs, ["Số lượng trả (ĐVT cơ bản)", "Số lượng trả", "SLTraQuyDoi", "SL trả (ĐVT cơ bản)"])
        c_stt = _first_present(df_bs, ["STT", "SoThuTu", "No."])

        bs_rows = []
        for _, r in df_bs.iterrows():
            oid = clean_str(r[c_oid]) if c_oid else ""
            lot = clean_str(r[c_lot]) if c_lot else ""
            qty = float(parse_vn_number(r[c_qty]) or 0.0) if c_qty else 0.0
            ret = float(parse_vn_number(r[c_ret]) or 0.0) if c_ret else 0.0
            stt = clean_str(r[c_stt]) if c_stt else ""
            if oid or lot:
                bs_rows.append({
                    "order_id": oid,
                    "line_no": stt,
                    "batch_id": lot,
                    "sold_base_qty": qty,
                    "returned_base_qty": ret,
                })
        cb.batch_sales = pd.DataFrame(bs_rows)

    # 2. Batches master / import batch table (DanhSachLoHang, lo_hang, BanHang_LoHang, DonNhapHang_ChiTiet)
    b_tname = None
    for cand in ("DanhSachLoHang", "lo_hang", "BanHang_LoHang", "DonNhapHang_ChiTiet"):
        if cand in tables:
            b_tname = cand
            break

    # Import header dates if needed
    imp_dates: Dict[str, date] = {}
    for ih_name in ("DanhSachNhapHang", "phieu_nhap", "DonNhapHang", "SoTay_NhapHang"):
        if ih_name in tables:
            df_ih = tables[ih_name]
            c_id = _first_present(df_ih, ["Mã nhập hàng", "MaPhieuNhap", "Mã đơn nhập", "Số phiếu"])
            c_dt = _first_present(df_ih, ["Thời gian", "NgayNhapKho", "Ngày nhập"])
            if c_id and c_dt:
                for _, r in df_ih.iterrows():
                    iid = clean_str(r[c_id])
                    dt = parse_flexible_date(r[c_dt])
                    if iid and dt:
                        imp_dates[iid] = dt

    # Batch unit import prices from DanhSachChiTietNhapHang / chi_tiet_nhap / SoTay_NhapHang
    lot_unit_cost: Dict[str, float] = {}
    for ic_name in ("DanhSachChiTietNhapHang", "chi_tiet_nhap", "SoTay_NhapHang"):
        if ic_name in tables:
            df_ic = tables[ic_name]
            c_lot = _first_present(df_ic, ["Lô", "MaLo"])
            c_prc = _first_present(df_ic, ["Đơn giá", "DonGiaNhap", "Giá nhập"])
            c_u = _first_present(df_ic, ["ĐVT", "DonViTinh", "ĐVT nhập"])
            c_p = _first_present(df_ic, ["Mã hàng", "MaHangHoa", "Product"])
            if c_lot and c_prc:
                for _, r in df_ic.iterrows():
                    lot = clean_str(r[c_lot])
                    prc = parse_vn_number(r[c_prc])
                    if not lot or prc is None:
                        continue
                    raw_p = clean_str(r[c_p]) if c_p else ""
                    pkey = cb.prod_name_to_key.get(norm_key(raw_p), raw_p)
                    u_norm = norm_key(r[c_u]) if c_u else ""
                    fac = cb.unit_factors.get((pkey, u_norm), 1.0)
                    lot_unit_cost[lot] = float(prc) / fac if fac > 0 else float(prc)

    if b_tname is not None:
        df_b = tables[b_tname]
        c_imp = _first_present(df_b, ["Mã nhập hàng", "MaPhieuNhap", "Mã phiếu nhập", "Mã đơn nhập"])
        c_prd = _first_present(df_b, ["Mã hàng", "MaHangHoa", "Mã món", "Mã SKU"])
        c_lot = _first_present(df_b, ["Lô", "MaLo", "Mã lô"])
        c_exp = _first_present(df_b, ["Hạn sử dụng", "HanSuDung", "HSD"])
        c_qty = _first_present(df_b, ["Số lượng (ĐVT cơ bản)", "SLNhapQuyDoi", "SL nhập (ĐVT cơ bản)", "Số lượng nhập"])
        c_cost = _first_present(df_b, ["DonGiaNhap"])

        b_rows = []
        for _, r in df_b.iterrows():
            imp_id = clean_str(r[c_imp]) if c_imp else ""
            raw_p = clean_str(r[c_prd]) if c_prd else ""
            pkey = cb.prod_name_to_key.get(norm_key(raw_p), raw_p)
            lot = clean_str(r[c_lot]) if c_lot else ""
            exp_dt = parse_flexible_date(r[c_exp]) if c_exp else None
            qty = float(parse_vn_number(r[c_qty]) or 0.0) if c_qty else 0.0
            cost_base = float(parse_vn_number(r[c_cost])) if (c_cost and parse_vn_number(r[c_cost]) is not None) else lot_unit_cost.get(lot)
            imp_dt = imp_dates.get(imp_id)
            b_rows.append({
                "batch_id": lot,
                "import_id": imp_id,
                "import_date": imp_dt,
                "product_key": pkey,
                "expiry_date": exp_dt,
                "import_base_qty": qty,
                "unit_cost_base": cost_base,
            })
        cb.batches = pd.DataFrame(b_rows)
        cb.capabilities["has_batch_expiry"] = True
    else:
        cb.capabilities["has_batch_expiry"] = False


def _resolve_prod_key(cb: CanonicalBusiness, raw_code: str, raw_name: str) -> Tuple[str, str]:
    """Return (product_key, canonical_display_name)."""
    if raw_code:
        nk_c = norm_key(raw_code)
        if nk_c in cb.prod_name_to_key:
            pk = cb.prod_name_to_key[nk_c]
            return pk, cb.prod_key_to_name.get(pk, raw_name or raw_code)
        if raw_code in cb.prod_key_to_name:
            return raw_code, cb.prod_key_to_name[raw_code]
        pk = raw_code
        if raw_name and pk not in cb.prod_key_to_name:
            cb.prod_key_to_name[pk] = raw_name
            cb.prod_name_to_key[norm_key(raw_name)] = pk
        cb.prod_name_to_key[nk_c] = pk
        return pk, cb.prod_key_to_name.get(pk, raw_name or raw_code)

    nk_n = norm_key(raw_name)
    if nk_n in cb.prod_name_to_key:
        pk = cb.prod_name_to_key[nk_n]
        return pk, cb.prod_key_to_name.get(pk, raw_name)
    pk = nk_n
    if raw_name and pk not in cb.prod_key_to_name:
        cb.prod_key_to_name[pk] = raw_name
    if nk_n:
        cb.prod_name_to_key[nk_n] = pk
    return pk, cb.prod_key_to_name.get(pk, raw_name)


def _resolve_cust_key(
    cb: CanonicalBusiness, raw_code: str, raw_name: str, raw_phone: str, has_cust_code: bool
) -> Tuple[str, str, bool]:
    """Return (customer_key, display_name, is_walkin)."""
    clean_nm, ph_from_nm = extract_phone_and_clean_name(raw_name)
    name = clean_nm if clean_nm else clean_str(raw_name)
    phone = re.sub(r"\D", "", clean_str(raw_phone)) if raw_phone else ph_from_nm
    if is_walkin_customer(name, raw_code):
        return "", "Khách lẻ", True

    if has_cust_code and raw_code:
        ckey = raw_code
        disp = cb.cust_key_to_name.get(ckey, name or raw_code)
        if ckey not in cb.cust_key_to_name and name:
            cb.cust_key_to_name[ckey] = name
        return ckey, disp, False

    nk = norm_key(name)
    if nk in cb.cust_name_to_key and not phone:
        ckey = cb.cust_name_to_key[nk]
        return ckey, cb.cust_key_to_name.get(ckey, name), False

    if phone and cb.domain == "phone_shop":
        ckey = f"{nk}|{phone}"
    else:
        ckey = cb.cust_name_to_key.get(nk, nk)

    if ckey not in cb.cust_key_to_name and name:
        cb.cust_key_to_name[ckey] = name
    elif ckey in cb.cust_key_to_name and name:
        # Prefer accented spelling over unaccented spelling
        old_nm = cb.cust_key_to_name[ckey]
        if strip_accents(old_nm) == old_nm and strip_accents(name) != name:
            cb.cust_key_to_name[ckey] = name
    if nk and nk not in cb.cust_name_to_key:
        cb.cust_name_to_key[nk] = ckey
    return ckey, cb.cust_key_to_name.get(ckey, name), False


def _build_orders_and_lines(cb: CanonicalBusiness, tables: Dict[str, pd.DataFrame]) -> None:
    # Collect returned order IDs from tra_hang and batch_sales
    returned_oids: Set[str] = set()
    if "tra_hang" in tables:
        df_th = tables["tra_hang"]
        c_oid = _first_present(df_th, ["MaHoaDon", "Mã hóa đơn", "Mã đơn hàng"])
        if c_oid:
            for v in df_th[c_oid].dropna():
                s = clean_str(v)
                if s:
                    returned_oids.add(s)
    if not cb.batch_sales.empty:
        ret_bs = cb.batch_sales[cb.batch_sales["returned_base_qty"] > 0]
        for v in ret_bs["order_id"].dropna():
            if v:
                returned_oids.add(str(v))

    # Case A: Header-only sales table (danh_sach_hoa_don in biz_110, dev_009)
    if "danh_sach_hoa_don" in tables and len(tables) == 1:
        df_h = tables["danh_sach_hoa_don"]
        c_oid = _first_present(df_h, ["Mã hóa đơn", "Mã bill"])
        c_dt = _first_present(df_h, ["Ngày tháng", "Ngày giao dịch"])
        c_st = _first_present(df_h, ["Trạng thái đơn", "Status"])
        c_rev = _first_present(df_h, ["Giá trị hàng", "Tien hang"])
        c_tot = _first_present(df_h, ["TongTien"])
        c_paid = _first_present(df_h, ["Đã thu", "Da tra"])
        c_debt = _first_present(df_h, ["Nợ", "Chưa trả"])
        c_cust = _first_present(df_h, ["Đối tác", "Bên mua"])

        o_rows = []
        for oid_v, dt_v, st_v, rev_v, tot_v, paid_v, debt_v, cust_v in zip(
            df_h[c_oid], df_h[c_dt], df_h[c_st], df_h[c_rev],
            df_h[c_tot], df_h[c_paid], df_h[c_debt], df_h[c_cust],
        ):
            oid = clean_str(oid_v)
            dt = parse_flexible_date(dt_v)
            if not oid or not dt:
                continue
            status = normalize_order_status(clean_str(st_v))
            if oid in returned_oids:
                status = "returned"
            rev = float(parse_vn_number(rev_v) or 0.0)
            tot = float(parse_vn_number(tot_v) or rev)
            paid = float(parse_vn_number(paid_v) or 0.0)
            debt = float(parse_vn_number(debt_v) or 0.0)
            ckey, cname, is_wk = _resolve_cust_key(cb, "", clean_str(cust_v), "", False)
            o_rows.append({
                "order_id": oid,
                "date": dt,
                "hour": None,
                "status": status,
                "customer_key": ckey,
                "customer_name": cname,
                "is_walkin": is_wk,
                "revenue": rev,
                "order_discount": max(0.0, rev - tot),
                "voucher_discount": 0.0,
                "points_discount": 0.0,
                "shipping_fee": 0.0,
                "trade_in_amount": 0.0,
                "total_payable": tot,
                "paid_at_purchase": paid,
                "debt_at_end": debt,
                "payment_method": "debt" if debt > 0 else "",
                "channel": "",
                "staff_id": "",
                "room_area": "",
                "table_name": "",
            })
        cb.orders = pd.DataFrame(o_rows)
        cb.capabilities["has_order_id"] = True
        cb.capabilities["has_line_items"] = False
        return

    # Case B: Separate Header + Detail tables OR Combined Line Table
    hdr_tname = None
    for cand in ("DanhSachHoaDon", "don_hang", "BanHang_HoaDon"):
        if cand in tables:
            hdr_tname = cand
            break

    dtl_tname = None
    for cand in (
        "DanhSachHoaDonChiTiet", "chi_tiet_don", "BanHang_ChiTietHoaDon",
        "So_chi_tiet_ban_hang", "DonHang_ChiTiet", "BanHang", "SoBanHang",
        "ban_hang", "chi_tiet_hoa_don", "ChiTietHoaDon",
    ):
        if cand in tables:
            dtl_tname = cand
            break

    if dtl_tname is None:
        return

    df_d = tables[dtl_tname].copy()

    # Map detail columns
    c_oid = _first_present(df_d, ["Mã hóa đơn", "MaHoaDon", "Mã giao dịch", "Mã đơn hàng", "Đơn số", "Ma don", "Mã bill", "Bill", "HĐ", "Số phiếu", "Mã phiếu", "Số CT", "Mã GD"])
    c_dt = _first_present(df_d, ["Ngày", "Ngày đặt hàng", "Ngày HĐ", "Thời gian", "Ngày giao dịch", "Ngày xuất", "Ngày bán", "Ngày tháng", "ngay_ban", "Ngày mua", "Ngày lập"])
    c_hr = _first_present(df_d, ["Giờ tạo đơn", "Giờ", "Giờ bán"])
    c_st = _first_present(df_d, ["Trạng thái", "Trạng thái đơn hàng", "Trạng thái giao", "Trạng thái xử lý", "Tiến độ", "TT đơn", "Kết quả", "Status", "trang_thai", "Trạng thái đơn", "Ghi chú TT"])
    c_note = _first_present(df_d, ["Ghi chú", "Note"]) if c_st != "Ghi chú TT" else None
    c_ccode = _first_present(df_d, ["Mã khách", "Mã khách mua", "Mã người mua"])
    c_cname = _first_present(df_d, ["Khách hàng", "Tên khách hàng", "Khách hàng (tên)", "Họ tên khách", "Khách", "Đối tác", "Tên khách", "TenKH"])
    c_cphone = _first_present(df_d, ["SĐT", "Phone", "Số ĐT"])
    c_cgrp = _first_present(df_d, ["Nhóm khách hàng"])
    c_cgen = _first_present(df_d, ["Giới tính"])
    c_caddr = _first_present(df_d, ["Addr"])
    c_pcode = _first_present(df_d, ["Mã hàng", "MaHangHoa", "Ma hang", "Barcode", "Mã vạch", "Mã mặt hàng", "Mã hàng bán", "ID SP", "Mã H.hóa"])
    c_pname = _first_present(df_d, ["Tên hàng", "Chi tiết hàng", "Tên hàng hóa", "Tên sản phẩm", "Hàng hóa", "Hàng bán", "Mặt hàng", "Hàng", "Nội dung", "Sản phẩm", "Tên SP", "Ten hang"])
    c_pcat = _first_present(df_d, ["Phân loại", "Nhóm hàng"])
    c_unit = _first_present(df_d, ["Đơn vị tính", "DonViTinh", "Đ.vị", "ĐV", "dvt", "Đơn vị (tính)", "ĐVT", "Unit", "DonVi", "DVT", "ĐV tính"])
    c_qty = _first_present(df_d, ["Số lượng", "SL", "Slg", "Số lượng sản phẩm", "S.L", "SL bán", "Sluong", "SLg", "so_luong", "Số lượng hàng", "Số lượng bán", "SoLuong"])
    c_bqty = _first_present(df_d, ["Số lượng quy đổi", "SLQuyDoi", "SL cơ bản", "SL (ĐV cơ bản)", "Số lượng lẻ", "SL lẻ", "SL quy đổi", "Quy ra ĐV lẻ"])
    c_price = _first_present(df_d, ["Đơn giá", "GiaDonVi", "Giá niêm yết", "Giá sản phẩm", "Giá 1 đv", "Đơn giá (đ)", "Giá", "ĐG", "Giá bán", "Giá lẻ", "don_gia", "Đơn giá bán", "Giá bán lẻ", "Giá đơn vị", "Price"])
    c_ldisc = _first_present(df_d, ["Giảm giá", "ChietKhau", "Khuyến mãi", "Giảm (đ)", "Tiền giảm", "CK"])
    c_lamt = _first_present(df_d, ["Thành tiền", "TriGia", "Thanh tien", "Tổng tiền", "Thành tiền (đ)", "Amount", "TT", "Trị giá", "T.Tiền", "thanh_tien", "Tổng", "Tiền", "Cộng", "Tiền bán"])
    c_lcost = _first_present(df_d, ["Giá vốn", "GiaVonLucBan", "Giá vốn lúc bán"])
    c_pmeth = _first_present(df_d, ["Hình thức thanh toán", "Thanh toán bằng"])
    c_chan = _first_present(df_d, ["Nguồn", "Kênh bán", "Kênh"])
    c_staff = _first_present(df_d, ["Mã NV bán", "Nhân viên bán"])
    c_area = _first_present(df_d, ["Khu vực"])
    c_tbl = _first_present(df_d, ["Bàn"])
    c_odisc = _first_present(df_d, ["Tiền giảm", "Giảm giá HĐ", "Bớt"]) if c_ldisc != "Tiền giảm" else _first_present(df_d, ["Giảm giá HĐ", "Bớt"])
    c_ship = _first_present(df_d, ["Phí ship", "Phi ship"])
    c_tradein = _first_present(df_d, ["Thu máy cũ"])
    c_otot = _first_present(df_d, ["Tổng thanh toán", "Tổng phiếu", "Tong tien", "Khách trả"]) if cb.domain == "phone_shop" else _first_present(df_d, ["Tổng thanh toán", "Tổng phiếu", "Tong tien"])
    c_opaid = _first_present(df_d, ["Da tra", "Khách trả"])
    c_odebt = _first_present(df_d, ["Công nợ", "Còn nợ"])
    c_imei = _first_present(df_d, ["IMEI"])

    # Special case: STT as order ID in dev_011 and dev_014
    if c_oid is None and "STT" in df_d.columns and cb.domain == "phone_shop":
        c_oid = "STT"

    has_order_id = c_oid is not None
    cb.capabilities["has_order_id"] = has_order_id
    cb.capabilities["has_line_items"] = True

    # Drop subtotal/footer rows where product name is missing or where quantity is missing (except dev_013 Voucher/Điểm/Ship)
    if c_pname:
        df_d = df_d[df_d[c_pname].notna()].copy()
    if c_qty and cb.domain != "mother_baby":
        df_d = df_d[df_d[c_qty].notna()].copy()

    # Forward-fill order_id and order-level columns when blank on continuation rows of an order (Rulebook §8)
    if has_order_id:
        if df_d[c_oid].isna().any():
            # Forward-fill within each source file / sheet
            df_d[c_oid] = df_d.groupby(["_source_file", "_sheet_name"], dropna=False)[c_oid].ffill()
        # Convert float STT (e.g. 1.0) to clean int string
        if c_oid == "STT":
            df_d[c_oid] = df_d[c_oid].apply(lambda x: f"STT_{int(float(x))}" if pd.notna(x) and str(x).strip() != "" else "")
        else:
            df_d[c_oid] = df_d[c_oid].apply(clean_str)

        # Forward-fill order-level fields within each order_id (scoped by sheet if STT)
        grp_cols = ["_sheet_name", c_oid] if c_oid == "STT" and df_d["_sheet_name"].nunique() > 1 else [c_oid]
        if c_oid == "STT" and df_d["_sheet_name"].nunique() == 1:
            grp_cols = [c_oid]
        elif c_oid == "STT":
            # In dev_014, STT is unique across the whole book! Let's check: STT 33, 54, 158...
            grp_cols = [c_oid]

        for col_ff in (c_dt, c_hr, c_st, c_note, c_ccode, c_cname, c_cphone, c_chan, c_pmeth, c_staff, c_area, c_tbl):
            if col_ff and col_ff in df_d.columns and df_d[col_ff].isna().any():
                df_d[col_ff] = df_d.groupby(grp_cols, sort=False)[col_ff].ffill()

        # Rulebook §8: Deduplicate identical lines within the same order
        data_cols = [c for c in df_d.columns if not c.startswith("_")]
        df_d = df_d.drop_duplicates(subset=data_cols).copy()
    else:
        # No order ID (biz_107, biz_113, biz_126): forward-fill date if blank, do NOT deduplicate lines
        if c_dt and df_d[c_dt].isna().any():
            df_d[c_dt] = df_d[c_dt].ffill()

    # If separate header table exists, parse header table first
    hdr_info: Dict[str, Dict[str, Any]] = {}
    has_cust_code = bool(c_ccode) or (not cb.customers.empty and cb.customers["customer_code"].notna().any())
    cust_attr_map: Dict[str, Tuple[str, str, str]] = {}
    if not cb.customers.empty:
        for _, cr in cb.customers.iterrows():
            cust_attr_map[cr["customer_key"]] = (
                cr.get("customer_group_norm") or "",
                cr.get("gender") or "",
                cr.get("address_norm") or "",
            )

    if hdr_tname is not None:
        df_h = tables[hdr_tname]
        hc_oid = _first_present(df_h, ["Mã hóa đơn", "MaHoaDon", "Order ID", "Mã chứng từ", "Số CT", "Số phiếu"])
        hc_dt = _first_present(df_h, ["Thời gian", "NgayLap", "Ngày bán", "Ngày", "Date", "NgayBan"])
        hc_hr = _first_present(df_h, ["Giờ tạo đơn", "ThoiGianLap", "Thời gian tạo", "Thời gian"])
        hc_st = _first_present(df_h, ["Trạng thái", "TinhTrang", "Tình trạng", "Tình trạng đơn", "Xử lý"])
        hc_ccode = _first_present(df_h, ["Mã KH", "MaKhachHang", "Mã KH mua", "Customer ID", "Mã khách mua", "Mã thẻ", "Số KH"])
        hc_cname = _first_present(df_h, ["Khách hàng", "Tên KH"])
        hc_odisc = _first_present(df_h, ["Giảm giá", "ChietKhau", "CK", "Tiền giảm"])
        hc_ship = _first_present(df_h, ["PhiVanChuyen", "Phí vận chuyển"])
        hc_tot = _first_present(df_h, ["Tổng sau giảm giá", "TongThanhToan", "Tong tien", "Tổng đơn", "Phải thu", "Thành tiền", "Tổng bill"])
        hc_paid = _first_present(df_h, ["Khách đã trả", "KhachDaTra", "Đã thu", "Khách trả", "Thanh toán"])
        hc_debt = _first_present(df_h, ["Còn nợ", "ConPhaiThu", "Dư nợ", "Con no", "Còn lại"])
        hc_pmeth = _first_present(df_h, ["Phương thức thanh toán", "PhuongThucTT", "Phương thức TT", "Trả bằng", "Hình thức thanh toán"])
        hc_chan = _first_present(df_h, ["Kênh bán", "KenhBanHang", "Nguồn đơn", "Loại bán", "Kênh đặt", "Bán qua"])
        hc_staff = _first_present(df_h, ["Người bán", "MaNhanVien", "Mã NV bán"])

        if hc_ccode:
            has_cust_code = True

        # Room/Table table (DanhSachHoaDonPhongBan in biz_128)
        room_map: Dict[str, Tuple[str, str]] = {}
        if "DanhSachHoaDonPhongBan" in tables:
            df_rm = tables["DanhSachHoaDonPhongBan"]
            for r_oid, r_tbl, r_area in zip(df_rm["Mã hóa đơn"], df_rm["Phòng/Bàn"], df_rm["Khu vực"]):
                room_map[clean_str(r_oid)] = (clean_str(r_area), clean_str(r_tbl))

        for idx_h, r in df_h.iterrows():
            oid = clean_str(r[hc_oid]) if hc_oid else ""
            if not oid:
                continue
            s_yr = int(r["_sheet_year"]) if pd.notna(r.get("_sheet_year")) else None
            dt = parse_flexible_date(r[hc_dt], default_year=s_yr) if hc_dt else None
            hr = parse_flexible_hour(r[hc_hr]) if hc_hr else None
            st_raw = clean_str(r[hc_st]) if hc_st else ""
            status = normalize_order_status(st_raw)
            if oid in returned_oids:
                status = "returned"
            ccode = clean_str(r[hc_ccode]) if hc_ccode else ""
            cname_raw = clean_str(r[hc_cname]) if hc_cname else ""
            ckey, cname, is_wk = _resolve_cust_key(cb, ccode, cname_raw, "", has_cust_code)
            c_grp_n, c_gen_n, c_addr_n = cust_attr_map.get(ckey, ("", "", ""))

            odisc = float(parse_vn_number(r[hc_odisc]) or 0.0) if hc_odisc else 0.0
            ship = float(parse_vn_number(r[hc_ship]) or 0.0) if hc_ship else 0.0
            tot_val = parse_vn_number(r[hc_tot]) if hc_tot else None
            paid_val = parse_vn_number(r[hc_paid]) if hc_paid else None
            debt_val = parse_vn_number(r[hc_debt]) if hc_debt else None
            pmeth = normalize_payment_method(r[hc_pmeth]) if hc_pmeth else ""
            chan = normalize_channel(r[hc_chan]) if hc_chan else ""
            staff = clean_str(r[hc_staff]) if hc_staff else ""
            area, tbl_nm = room_map.get(oid, ("", ""))

            hdr_info[oid] = {
                "order_id": oid,
                "date": dt,
                "hour": hr,
                "status": status,
                "customer_key": ckey,
                "customer_name": cname,
                "is_walkin": is_wk,
                "customer_group_norm": c_grp_n,
                "customer_gender": c_gen_n,
                "customer_address_norm": c_addr_n,
                "order_discount": odisc,
                "voucher_discount": 0.0,
                "points_discount": 0.0,
                "shipping_fee": ship,
                "trade_in_amount": 0.0,
                "total_payable_hdr": float(tot_val) if tot_val is not None else None,
                "paid_at_purchase": float(paid_val) if paid_val is not None else None,
                "debt_at_end": float(debt_val) if debt_val is not None else None,
                "payment_method": pmeth,
                "channel": chan,
                "staff_id": staff,
                "room_area": area,
                "table_name": tbl_nm,
            }

    # Process detail lines
    line_rows: List[Dict[str, Any]] = []
    neg_qty_oids: Set[str] = set()
    order_agg: Dict[str, Dict[str, Any]] = {}

    # Trade-in map from ThuMayCu (dev_014)
    tradein_by_oid: Dict[str, float] = {}
    if "ThuMayCu" in tables:
        df_tmc = tables["ThuMayCu"]
        c_t_stt = _first_present(df_tmc, ["Trừ vào (STT sổ bán)"])
        c_t_amt = _first_present(df_tmc, ["Giá thu"])
        c_t_dt = _first_present(df_tmc, ["Ngày"])
        t_rows = []
        for _, r in df_tmc.iterrows():
            stt_v = r[c_t_stt] if c_t_stt else None
            oid_ref = f"STT_{int(float(stt_v))}" if pd.notna(stt_v) and str(stt_v).strip() != "" else ""
            amt_v = float(parse_vn_number(r[c_t_amt]) or 0.0) if c_t_amt else 0.0
            dt_v = parse_flexible_date(r[c_t_dt]) if c_t_dt else None
            if oid_ref:
                tradein_by_oid[oid_ref] = tradein_by_oid.get(oid_ref, 0.0) + amt_v
            t_rows.append({
                "order_id": oid_ref,
                "date": dt_v,
                "amount": amt_v,
            })
        cb.trade_ins = pd.DataFrame(t_rows)

    # Pre-build product cost/category lookup from cb.products
    prod_cat_map: Dict[str, str] = {}
    prod_cost_map: Dict[str, float] = {}
    if not cb.products.empty:
        for pk, pcat, pcost in zip(cb.products["product_key"], cb.products["category"], cb.products["cost_price"]):
            if pcat:
                prod_cat_map[pk] = pcat
            if pcost is not None and pd.notna(pcost):
                prod_cost_map[pk] = float(pcost)

    n_rows = len(df_d)
    s_oid = df_d[c_oid].tolist() if has_order_id else [f"ROW_{i:06d}" for i in range(n_rows)]
    s_dt = df_d[c_dt].tolist() if c_dt else [None] * n_rows
    s_yr = df_d["_sheet_year"].tolist() if "_sheet_year" in df_d.columns else [None] * n_rows
    s_hr = df_d[c_hr].tolist() if c_hr else [None] * n_rows
    s_st = df_d[c_st].tolist() if c_st else [None] * n_rows
    s_note = df_d[c_note].tolist() if c_note else [None] * n_rows
    s_ccode = df_d[c_ccode].tolist() if c_ccode else [""] * n_rows
    s_cname = df_d[c_cname].tolist() if c_cname else [""] * n_rows
    s_cphone = df_d[c_cphone].tolist() if c_cphone else [""] * n_rows
    s_cgrp = df_d[c_cgrp].tolist() if c_cgrp else [""] * n_rows
    s_cgen = df_d[c_cgen].tolist() if c_cgen else [""] * n_rows
    s_caddr = df_d[c_caddr].tolist() if c_caddr else [""] * n_rows
    s_pcode = df_d[c_pcode].tolist() if c_pcode else [""] * n_rows
    s_pname = df_d[c_pname].tolist() if c_pname else [""] * n_rows
    s_pcat = df_d[c_pcat].tolist() if c_pcat else [""] * n_rows
    s_unit = df_d[c_unit].tolist() if c_unit else [""] * n_rows
    s_qty = df_d[c_qty].tolist() if c_qty else [1.0] * n_rows
    s_bqty = df_d[c_bqty].tolist() if c_bqty else [None] * n_rows
    s_price = df_d[c_price].tolist() if c_price else [None] * n_rows
    s_ldisc = df_d[c_ldisc].tolist() if c_ldisc else [0.0] * n_rows
    s_lamt = df_d[c_lamt].tolist() if c_lamt else [0.0] * n_rows
    s_lcost = df_d[c_lcost].tolist() if c_lcost else [None] * n_rows
    s_pmeth = df_d[c_pmeth].tolist() if c_pmeth else [""] * n_rows
    s_chan = df_d[c_chan].tolist() if c_chan else [""] * n_rows
    s_staff = df_d[c_staff].tolist() if c_staff else [""] * n_rows
    s_area = df_d[c_area].tolist() if c_area else [""] * n_rows
    s_tbl = df_d[c_tbl].tolist() if c_tbl else [""] * n_rows
    s_odisc = df_d[c_odisc].tolist() if c_odisc else [None] * n_rows
    s_ship = df_d[c_ship].tolist() if c_ship else [None] * n_rows
    s_tradein = df_d[c_tradein].tolist() if c_tradein else [None] * n_rows
    s_otot = df_d[c_otot].tolist() if c_otot else [None] * n_rows
    s_opaid = df_d[c_opaid].tolist() if c_opaid else [None] * n_rows
    s_odebt = df_d[c_odebt].tolist() if c_odebt else [None] * n_rows
    s_imei = df_d[c_imei].tolist() if c_imei else [None] * n_rows

    if c_lcost is not None:
        cb.capabilities["has_cost_price"] = True
    if c_pcat is not None:
        cb.capabilities["has_product_category"] = True

    for i in range(n_rows):
        oid = clean_str(s_oid[i])
        if not oid:
            continue
        h = hdr_info.get(oid)
        yr_def = int(s_yr[i]) if s_yr[i] is not None and pd.notna(s_yr[i]) else None
        dt = h["date"] if h and h["date"] else parse_flexible_date(s_dt[i], default_year=yr_def)
        hr = h["hour"] if h and h["hour"] is not None else (parse_flexible_hour(s_hr[i]) if s_hr[i] is not None else parse_flexible_hour(s_dt[i]))

        if h:
            status = h["status"]
            ckey, cname, is_wk = h["customer_key"], h["customer_name"], h["is_walkin"]
            pmeth = h["payment_method"]
            chan = h["channel"]
            staff = h["staff_id"]
            area = h["room_area"]
            tbl_nm = h["table_name"]
        else:
            st_str = clean_str(s_st[i])
            note_str = clean_str(s_note[i])
            status = normalize_order_status(st_str, note_str)
            if oid in returned_oids:
                status = "returned"
            ckey, cname, is_wk = _resolve_cust_key(
                cb, clean_str(s_ccode[i]), clean_str(s_cname[i]), clean_str(s_cphone[i]), has_cust_code
            )
            pmeth = normalize_payment_method(s_pmeth[i])
            chan = normalize_channel(s_chan[i])
            staff = clean_str(s_staff[i])
            area = clean_str(s_area[i])
            tbl_nm = clean_str(s_tbl[i])

        # Record inline customer attributes (dev_007, dev_008)
        if ckey and not is_wk and (s_cgrp[i] or s_cgen[i] or s_caddr[i]):
            # Will be merged into cb.customers if cb.customers is empty
            pass

        if oid not in order_agg:
            if h:
                order_agg[oid] = dict(h)
                order_agg[oid]["revenue"] = 0.0
            else:
                odisc_v = float(parse_vn_number(s_odisc[i]) or 0.0) if s_odisc[i] is not None else 0.0
                ship_v = float(parse_vn_number(s_ship[i]) or 0.0) if s_ship[i] is not None else 0.0
                trd_v = float(parse_vn_number(s_tradein[i]) or 0.0) if s_tradein[i] is not None else tradein_by_oid.get(oid, 0.0)
                otot_v = parse_vn_number(s_otot[i]) if s_otot[i] is not None else None
                opaid_v = parse_vn_number(s_opaid[i]) if s_opaid[i] is not None else None
                odebt_v = parse_vn_number(s_odebt[i]) if s_odebt[i] is not None else None

                # Parse dev_008 Note debt if present
                note_raw = clean_str(s_note[i])
                if cb.business_id == "dev_008" and note_raw:
                    odebt_v = _parse_dev008_note_debt(note_raw, float(otot_v or 0.0))

                c_grp_fb, c_gen_fb, c_addr_fb = cust_attr_map.get(ckey, ("", "", ""))
                order_agg[oid] = {
                    "order_id": oid,
                    "date": dt,
                    "hour": hr,
                    "status": status,
                    "customer_key": ckey,
                    "customer_name": cname,
                    "is_walkin": is_wk,
                    "customer_group_norm": normalize_customer_group(s_cgrp[i]) or c_grp_fb,
                    "customer_gender": normalize_gender(s_cgen[i]) or c_gen_fb,
                    "customer_address_norm": norm_key(s_caddr[i]) or c_addr_fb,
                    "revenue": 0.0,
                    "order_discount": odisc_v,
                    "voucher_discount": 0.0,
                    "points_discount": 0.0,
                    "shipping_fee": ship_v,
                    "trade_in_amount": trd_v,
                    "total_payable_hdr": float(otot_v) if otot_v is not None else None,
                    "paid_at_purchase": float(opaid_v) if opaid_v is not None else None,
                    "debt_at_end": float(odebt_v) if odebt_v is not None else None,
                    "payment_method": pmeth,
                    "channel": chan,
                    "staff_id": staff,
                    "room_area": area,
                    "table_name": tbl_nm,
                }
        else:
            # Update order-level fields if they appeared on a later row (e.g. Bớt / Thu máy cũ / Khách trả)
            oa = order_agg[oid]
            if oa["date"] is None and dt is not None:
                oa["date"] = dt
            if not h:
                if s_odisc[i] is not None and oa["order_discount"] == 0.0:
                    oa["order_discount"] = float(parse_vn_number(s_odisc[i]) or 0.0)
                if s_tradein[i] is not None and oa["trade_in_amount"] == 0.0:
                    oa["trade_in_amount"] = float(parse_vn_number(s_tradein[i]) or 0.0)
                if s_otot[i] is not None and oa["total_payable_hdr"] is None:
                    v = parse_vn_number(s_otot[i])
                    if v is not None:
                        oa["total_payable_hdr"] = float(v)

        raw_pname = clean_str(s_pname[i])
        raw_pcode = clean_str(s_pcode[i])
        lamt = float(parse_vn_number(s_lamt[i]) or 0.0)

        # Mother & Baby (dev_013) special non-product rows: Voucher, Dùng <n> điểm, Phí ship
        if cb.domain == "mother_baby":
            nk_p = norm_key(raw_pname)
            if nk_p == "voucher":
                order_agg[oid]["voucher_discount"] += abs(lamt)
                order_agg[oid]["order_discount"] += abs(lamt)
                continue
            if nk_p.startswith("dung ") and "diem" in nk_p:
                order_agg[oid]["points_discount"] += abs(lamt)
                order_agg[oid]["order_discount"] += abs(lamt)
                continue
            if nk_p == "phi ship":
                order_agg[oid]["shipping_fee"] += abs(lamt)
                continue

        # Parse quantity and unit (handles '1 ly', '3 chai', etc.)
        qty_parsed, unit_from_qty = parse_qty_and_unit(s_qty[i])
        qty = float(qty_parsed if qty_parsed is not None else (parse_vn_number(s_qty[i]) or 0.0))
        if qty < 0:
            neg_qty_oids.add(oid)

        unit_str = clean_str(s_unit[i]) or (unit_from_qty or "")
        unit_norm = norm_key(unit_str)
        u_price = float(parse_vn_number(s_price[i]) or 0.0) if s_price[i] is not None else 0.0
        l_disc = float(parse_vn_number(s_ldisc[i]) or 0.0) if s_ldisc[i] is not None else 0.0
        l_cost = parse_vn_number(s_lcost[i]) if s_lcost[i] is not None else None

        # Check for inline toppings in dev_002 ("Trà đào cam sả (M) + Trân châu đen (7.000đ)")
        if cb.domain == "cafe" and " + " in raw_pname and "(" in raw_pname:
            parts = [p.strip() for p in raw_pname.split(" + ")]
            main_name = parts[0]
            topping_lines = []
            top_total_amt = 0.0
            for tp in parts[1:]:
                m_tp = re.match(r"^(?:(\d+)\s+)?(.+?)\s*\(([^)]+)\)$", tp)
                if m_tp:
                    t_qty = float(m_tp.group(1)) if m_tp.group(1) else 1.0
                    t_raw_nm = m_tp.group(2).strip()
                    if not norm_key(t_raw_nm).startswith("topping"):
                        t_full_nm = f"Topping {t_raw_nm}"
                    else:
                        t_full_nm = t_raw_nm
                    t_amt = float(parse_vn_number(m_tp.group(3)) or 0.0)
                    top_total_amt += t_amt
                    t_pk, t_disp = _resolve_prod_key(cb, "", t_full_nm)
                    cb.prod_base_unit.setdefault(t_pk, "phần")
                    topping_lines.append({
                        "order_id": oid,
                        "date": order_agg[oid]["date"],
                        "status": order_agg[oid]["status"],
                        "customer_key": order_agg[oid]["customer_key"],
                        "product_key": t_pk,
                        "product_name": t_disp,
                        "category": prod_cat_map.get(t_pk, ""),
                        "category_norm": norm_key(prod_cat_map.get(t_pk, "")),
                        "unit": "phần",
                        "unit_norm": "phan",
                        "qty": t_qty,
                        "base_qty": t_qty,
                        "unit_price": t_amt / t_qty if t_qty else t_amt,
                        "line_discount": 0.0,
                        "line_amount": t_amt,
                        "unit_cost": prod_cost_map.get(t_pk),
                        "gross_profit": None,
                        "is_service_fee": False,
                        "imei": None,
                    })
            raw_pname = main_name
            lamt = max(0.0, lamt - top_total_amt)
            for tl in topping_lines:
                line_rows.append(tl)
                order_agg[oid]["revenue"] += tl["line_amount"]

        pk, p_disp = _resolve_prod_key(cb, raw_pcode, raw_pname)
        if unit_str and pk not in cb.prod_base_unit and (pk, unit_norm) not in cb.unit_factors:
            cb.prod_base_unit[pk] = unit_str
            cb.unit_factors[(pk, unit_norm)] = 1.0

        # Base quantity resolution
        if s_bqty[i] is not None and pd.notna(s_bqty[i]):
            bqty_val = parse_vn_number(s_bqty[i])
            if bqty_val is not None:
                bqty = float(bqty_val)
                # Learn unit conversion factor from data if not yet known
                if qty > 0 and bqty > 0 and unit_norm:
                    ratio = round(bqty / qty, 4)
                    if ratio == 1.0 and pk not in cb.prod_base_unit:
                        cb.prod_base_unit[pk] = unit_str
                    cb.unit_factors.setdefault((pk, unit_norm), ratio)
            else:
                fac = cb.unit_factors.get((pk, unit_norm), 1.0)
                bqty = qty * fac
        else:
            fac = cb.unit_factors.get((pk, unit_norm), 1.0)
            bqty = qty * fac

        pcat = clean_str(s_pcat[i]) or prod_cat_map.get(pk, "")
        if pcat and pk not in prod_cat_map:
            prod_cat_map[pk] = pcat

        # Cost price and gross profit per line (Rulebook §5: qty * unit_cost_on_line)
        ucost = float(l_cost) if l_cost is not None else prod_cost_map.get(pk)
        if l_cost is not None:
            gprofit = lamt - (qty * float(l_cost))
        elif ucost is not None:
            # Catalog cost_price is per base_unit
            gprofit = lamt - (bqty * ucost)
        else:
            gprofit = None

        is_sf = norm_key(p_disp) == "phi phuc vu"
        line_rows.append({
            "order_id": oid,
            "date": order_agg[oid]["date"],
            "status": order_agg[oid]["status"],
            "customer_key": order_agg[oid]["customer_key"],
            "product_key": pk,
            "product_name": p_disp,
            "category": pcat,
            "category_norm": norm_key(pcat),
            "unit": unit_str,
            "unit_norm": unit_norm,
            "qty": qty,
            "base_qty": bqty,
            "unit_price": u_price,
            "line_discount": l_disc,
            "line_amount": lamt,
            "unit_cost": ucost,
            "gross_profit": gprofit,
            "is_service_fee": is_sf,
            "imei": clean_str(s_imei[i]) if s_imei[i] is not None else None,
        })
        order_agg[oid]["revenue"] += lamt

    # Apply negative-quantity return rule (biz_105, biz_117, dev_004)
    if neg_qty_oids:
        for oid in neg_qty_oids:
            if oid in order_agg:
                order_agg[oid]["status"] = "returned"
        for lr in line_rows:
            if lr["order_id"] in neg_qty_oids:
                lr["status"] = "returned"

    # Finalize orders table
    for oa in order_agg.values():
        rev = oa["revenue"]
        odisc = oa["order_discount"]
        ship = oa["shipping_fee"]
        trd = oa["trade_in_amount"]
        if oa["total_payable_hdr"] is not None:
            oa["total_payable"] = oa["total_payable_hdr"]
        else:
            oa["total_payable"] = max(0.0, rev - odisc + ship - trd)

    cb.orders = pd.DataFrame(list(order_agg.values()))
    cb.order_lines = pd.DataFrame(line_rows)

    # Backfill cb.products if catalog was absent or incomplete
    _backfill_products_from_lines(cb, prod_cat_map, prod_cost_map)
    # Backfill cb.customers if customer master table was absent
    _backfill_customers_from_orders(cb)


def _parse_dev008_note_debt(note: str, total_payable: float) -> float:
    s = strip_accents(str(note).lower()).strip()
    if not s or "tra du" in s or "ck du" in s or "tra het" in s:
        return 0.0
    if "no ca don" in s:
        return total_payable
    # Patterns: 'còn thiếu <X>', 'thiếu lại <X>', 'ghi nợ <X>', 'ghi sổ <X>', 'trả <Y>, nợ <X>', 'nợ <X>'
    m = re.search(r"(?:con thieu|thieu lai|ghi no|ghi so|no)\s*([\d\.,]+\s*(?:d|k|tr\d*|nghin|dong)?)", s)
    if m:
        val = parse_vn_number(m.group(1))
        if val is not None:
            return float(val)
    # Pattern: 'đưa trước <Y>, còn lại ghi sổ'
    m2 = re.search(r"(?:dua truoc|tra truoc|tra|ck)\s*([\d\.,]+\s*(?:d|k|tr\d*|nghin|dong)?)", s)
    if m2:
        paid = parse_vn_number(m2.group(1))
        if paid is not None:
            return max(0.0, total_payable - float(paid))
    return 0.0


def _backfill_products_from_lines(
    cb: CanonicalBusiness, prod_cat_map: Dict[str, str], prod_cost_map: Dict[str, float]
) -> None:
    existing_keys = set(cb.products["product_key"]) if not cb.products.empty else set()
    new_rows = []
    for pk, pname in cb.prod_key_to_name.items():
        if pk in existing_keys:
            continue
        pcat = prod_cat_map.get(pk, "")
        bunit = cb.prod_base_unit.get(pk, "")
        is_phone = cb.domain == "phone_shop" and norm_key(pcat) in ("may moi", "may cu")
        p_model, p_brand = extract_phone_model_and_brand(pname, is_phone)
        new_rows.append({
            "product_key": pk,
            "product_code": cb.prod_key_to_code.get(pk),
            "product_name": pname,
            "category": pcat or None,
            "category_norm": norm_key(pcat),
            "brand": p_brand,
            "brand_norm": norm_key(p_brand) if p_brand else "",
            "phone_model": p_model,
            "phone_model_norm": norm_key(p_model) if p_model else "",
            "milk_or_diaper_size": None,
            "size_norm": "",
            "base_unit": bunit or None,
            "base_unit_norm": norm_key(bunit),
            "cost_price": prod_cost_map.get(pk),
            "sell_price": None,
            "active_ingredient": None,
            "active_ingredient_norm": "",
            "is_prescription": False,
            "is_service_fee": norm_key(pname) == "phi phuc vu",
        })
    if new_rows:
        df_new = pd.DataFrame(new_rows)
        cb.products = pd.concat([cb.products, df_new], ignore_index=True) if not cb.products.empty else df_new
    if not cb.products.empty and "base_unit" in cb.products.columns:
        cb.products["base_unit"] = cb.products["product_key"].map(lambda k: cb.prod_base_unit.get(k, None)).fillna(cb.products["base_unit"])
        cb.products["base_unit_norm"] = cb.products["base_unit"].apply(norm_key)


def _backfill_customers_from_orders(cb: CanonicalBusiness) -> None:
    if cb.orders.empty:
        return
    if cb.customers.empty:
        c_rows = []
        seen = set()
        for _, r in cb.orders.iterrows():
            ck = r["customer_key"]
            if not ck or r["is_walkin"] or ck in seen:
                continue
            seen.add(ck)
            c_rows.append({
                "customer_key": ck,
                "customer_code": cb.cust_key_to_code.get(ck),
                "customer_name": cb.cust_key_to_name.get(ck, r["customer_name"]),
                "phone": ck.split("|")[1] if "|" in ck else "",
                "customer_group": r.get("customer_group_norm") or None,
                "customer_group_norm": r.get("customer_group_norm") or "",
                "gender": r.get("customer_gender") or "",
                "address": r.get("customer_address_norm") or None,
                "address_norm": r.get("customer_address_norm") or "",
                "reg_date": None,
            })
        cb.customers = pd.DataFrame(c_rows)



def _build_imports(cb: CanonicalBusiness, tables: Dict[str, pd.DataFrame]) -> None:
    # Supplier master lookup (dev_006, biz_123)
    sup_code_to_name: Dict[str, str] = {}
    if "nha_cung_cap" in tables:
        df_s = tables["nha_cung_cap"]
        c_sc = _first_present(df_s, ["MaNhaCungCap", "Mã NCC"])
        c_sn = _first_present(df_s, ["TenGoi", "Tên NCC"])
        if c_sc and c_sn:
            for sc, sn in zip(df_s[c_sc], df_s[c_sn]):
                sc_str, sn_str = clean_str(sc), clean_str(sn)
                if sc_str and sn_str:
                    sup_code_to_name[sc_str] = sn_str

    # Case 1: Separate Header + Detail import tables
    hdr_tname = None
    for cand in ("DanhSachNhapHang", "phieu_nhap", "DonNhapHang"):
        if cand in tables:
            hdr_tname = cand
            break

    dtl_tname = None
    for cand in ("DanhSachChiTietNhapHang", "chi_tiet_nhap", "DonNhapHang_ChiTiet"):
        if cand in tables:
            dtl_tname = cand
            break

    if hdr_tname is not None:
        df_h = tables[hdr_tname]
        hc_id = _first_present(df_h, ["Mã nhập hàng", "MaPhieuNhap", "Mã đơn nhập"])
        hc_dt = _first_present(df_h, ["Thời gian", "NgayNhapKho", "Ngày nhập"])
        hc_sup = _first_present(df_h, ["Mã NCC", "MaNhaCungCap", "Mã nhà cung cấp", "Nhà cung cấp"])
        hc_tot = _first_present(df_h, ["Cần trả NCC", "TongGiaTriNhap", "Tổng tiền hàng"])
        hc_st = _first_present(df_h, ["Trạng thái", "TinhTrang"])
        hc_note = _first_present(df_h, ["Ghi chú", "DienGiai"])

        imp_hdr: Dict[str, Dict[str, Any]] = {}
        for _, r in df_h.iterrows():
            iid = clean_str(r[hc_id]) if hc_id else ""
            if not iid:
                continue
            dt = parse_flexible_date(r[hc_dt]) if hc_dt else None
            raw_sup = clean_str(r[hc_sup]) if hc_sup else ""
            sup_name = sup_code_to_name.get(raw_sup, raw_sup)
            sup_code = raw_sup if raw_sup in sup_code_to_name or raw_sup.upper().startswith("NCC") else None
            tot_v = parse_vn_number(r[hc_tot]) if hc_tot else None
            st_raw = clean_str(r[hc_st]) if hc_st else ""
            st = "cancelled" if any(k in st_raw.lower() for k in ("huỷ", "hủy", "huy", "cancel")) else "completed"
            note_s = clean_str(r[hc_note]) if hc_note else ""
            imp_hdr[iid] = {
                "import_id": iid,
                "date": dt,
                "supplier_key": raw_sup or norm_key(sup_name),
                "supplier_code": sup_code,
                "supplier_name": sup_name,
                "status": st,
                "total_amount_hdr": float(tot_v) if tot_v is not None else None,
                "total_amount": 0.0,
                "is_opening": "đầu kỳ" in note_s.lower(),
            }

        il_rows = []
        if dtl_tname is not None:
            df_d = tables[dtl_tname]
            dc_id = _first_present(df_d, ["Mã nhập hàng", "MaPhieuNhap", "Mã đơn nhập"])
            dc_prd = _first_present(df_d, ["Mã hàng", "MaHangHoa", "Mã SKU"])
            dc_unit = _first_present(df_d, ["ĐVT", "DonViTinh"])
            dc_qty = _first_present(df_d, ["Số lượng", "SL", "Số lượng nhập"])
            dc_prc = _first_present(df_d, ["Đơn giá", "DonGiaNhap"])
            dc_lot = _first_present(df_d, ["Lô", "MaLo", "Mã lô"])

            for _, r in df_d.iterrows():
                iid = clean_str(r[dc_id]) if dc_id else ""
                if not iid:
                    continue
                h = imp_hdr.get(iid)
                dt = h["date"] if h else None
                st = h["status"] if h else "completed"
                raw_p = clean_str(r[dc_prd]) if dc_prd else ""
                pk, p_disp = _resolve_prod_key(cb, raw_p, "")
                u_str = clean_str(r[dc_unit]) if dc_unit else cb.prod_base_unit.get(pk, "")
                u_norm = norm_key(u_str)
                qty = float(parse_vn_number(r[dc_qty]) or 0.0) if dc_qty else 0.0
                fac = cb.unit_factors.get((pk, u_norm), 1.0)
                bqty = qty * fac
                prc = parse_vn_number(r[dc_prc]) if dc_prc else None
                u_prc = float(prc) if prc is not None else None
                lamt = (qty * u_prc) if u_prc is not None else None
                if h and lamt is not None:
                    h["total_amount"] += lamt
                il_rows.append({
                    "import_id": iid,
                    "date": dt,
                    "status": st,
                    "supplier_key": h["supplier_key"] if h else "",
                    "supplier_name": h["supplier_name"] if h else "",
                    "product_key": pk,
                    "product_name": p_disp,
                    "unit": u_str,
                    "unit_norm": u_norm,
                    "qty": qty,
                    "base_qty": bqty,
                    "unit_price": u_prc,
                    "line_amount": lamt,
                    "batch_id": clean_str(r[dc_lot]) if dc_lot else None,
                })

        for h in imp_hdr.values():
            if h["total_amount_hdr"] is not None:
                h["total_amount"] = h["total_amount_hdr"]

        cb.imports = pd.DataFrame(list(imp_hdr.values()))
        cb.import_lines = pd.DataFrame(il_rows)
        cb.capabilities["has_imports"] = not cb.imports.empty
        cb.capabilities["has_import_price"] = bool(not cb.import_lines.empty and cb.import_lines["unit_price"].notna().any())
        return

    # Case 2: Combined Single Import Table
    comb_tname = None
    for cand in ("ChiTietNhapHang", "chi_tiet_nhap_hang", "NhapHang", "SoNhapHang", "SoTay_NhapHang"):
        if cand in tables:
            comb_tname = cand
            break

    if comb_tname is None:
        cb.capabilities["has_imports"] = False
        cb.capabilities["has_import_price"] = False
        return

    df_c = tables[comb_tname].copy()
    c_id = _first_present(df_c, ["Mã phiếu nhập", "Số phiếu"])
    c_dt = _first_present(df_c, ["Ngày nhập", "Ngày"])
    c_sup = _first_present(df_c, ["Nhà cung cấp", "Mã NCC"])
    c_st = _first_present(df_c, ["Trạng thái phiếu", "Trạng thái"])
    c_note = _first_present(df_c, ["Ghi chú", "Ghi chu", "Lưu ý", "Note", "Nội dung", "Mô tả"])
    c_pcode = _first_present(df_c, ["Mã hàng bán", "Mã vạch"])
    c_pname = _first_present(df_c, ["Tên mặt hàng", "Hàng hóa", "Product", "Hàng"])
    c_unit = _first_present(df_c, ["ĐVT nhập", "ĐVT"])
    c_qty = _first_present(df_c, ["SL nhập", "SL"])
    c_prc = _first_present(df_c, ["Giá nhập", "Đơn giá"])
    c_lamt = _first_present(df_c, ["Thành tiền"])
    c_tot = _first_present(df_c, ["Tổng tiền phiếu"])
    c_lot = _first_present(df_c, ["Lô"])

    if c_id and df_c[c_id].isna().any():
        df_c[c_id] = df_c[c_id].ffill()
    for col_ff in (c_dt, c_sup, c_st, c_note, c_tot):
        if col_ff and col_ff in df_c.columns and df_c[col_ff].isna().any():
            df_c[col_ff] = df_c.groupby(c_id, sort=False)[col_ff].ffill() if c_id else df_c[col_ff].ffill()

    imp_agg: Dict[str, Dict[str, Any]] = {}
    il_rows = []
    for idx, r in df_c.iterrows():
        iid = clean_str(r[c_id]) if c_id else f"IMP_{idx:06d}"
        s_yr = int(r["_sheet_year"]) if pd.notna(r.get("_sheet_year")) else None
        dt = parse_flexible_date(r[c_dt], default_year=s_yr) if c_dt else None
        raw_sup = clean_str(r[c_sup]) if c_sup else ""
        sup_name = sup_code_to_name.get(raw_sup, raw_sup)
        sup_code = raw_sup if raw_sup.upper().startswith("NCC") else None
        sup_key = sup_code if sup_code else norm_key(sup_name)
        st_raw = clean_str(r[c_st]) if c_st else ""
        note_s = clean_str(r[c_note]) if c_note else ""
        st = "cancelled" if any(k in (st_raw + " " + note_s).lower() for k in ("huỷ", "hủy", "huy", "cancel")) else "completed"
        tot_hdr = parse_vn_number(r[c_tot]) if c_tot else None

        if iid not in imp_agg:
            imp_agg[iid] = {
                "import_id": iid,
                "date": dt,
                "supplier_key": sup_key,
                "supplier_code": sup_code,
                "supplier_name": sup_name,
                "status": st,
                "total_amount_hdr": float(tot_hdr) if tot_hdr is not None else None,
                "total_amount": 0.0,
                "is_opening": "đầu kỳ" in note_s.lower(),
            }

        raw_pc = clean_str(r[c_pcode]) if c_pcode else ""
        raw_pn = clean_str(r[c_pname]) if c_pname else ""
        pk, p_disp = _resolve_prod_key(cb, raw_pc, raw_pn)
        u_str = clean_str(r[c_unit]) if c_unit else cb.prod_base_unit.get(pk, "")
        u_norm = norm_key(u_str)
        qty_p, u_from_q = parse_qty_and_unit(r[c_qty]) if c_qty else (0.0, None)
        qty = float(qty_p or 0.0)
        if not u_str and u_from_q:
            u_str = u_from_q
            u_norm = norm_key(u_str)
        fac = cb.unit_factors.get((pk, u_norm), 1.0)
        bqty = qty * fac
        prc_v = parse_vn_number(r[c_prc]) if c_prc else None
        u_prc = float(prc_v) if prc_v is not None else None
        lamt_v = parse_vn_number(r[c_lamt]) if c_lamt else None
        lamt = float(lamt_v) if lamt_v is not None else ((qty * u_prc) if u_prc is not None else 0.0)
        imp_agg[iid]["total_amount"] += lamt

        il_rows.append({
            "import_id": iid,
            "date": imp_agg[iid]["date"],
            "status": st,
            "supplier_key": sup_key,
            "supplier_name": sup_name,
            "product_key": pk,
            "product_name": p_disp,
            "unit": u_str,
            "unit_norm": u_norm,
            "qty": qty,
            "base_qty": bqty,
            "unit_price": u_prc,
            "line_amount": lamt,
            "batch_id": clean_str(r[c_lot]) if c_lot else None,
        })

    for h in imp_agg.values():
        if h["total_amount_hdr"] is not None:
            h["total_amount"] = h["total_amount_hdr"]

    cb.imports = pd.DataFrame(list(imp_agg.values()))
    cb.import_lines = pd.DataFrame(il_rows)
    cb.capabilities["has_imports"] = not cb.imports.empty
    cb.capabilities["has_import_price"] = bool(not cb.import_lines.empty and cb.import_lines["unit_price"].notna().any())


def _build_stock(cb: CanonicalBusiness, tables: Dict[str, pd.DataFrame]) -> None:
    # 1. Finalize remaining_base_qty in cb.batches (subtracting non-cancelled batch_sales)
    if not cb.batches.empty:
        canc_oids: Set[str] = set()
        if not cb.orders.empty:
            canc_oids = set(cb.orders[cb.orders["status"] == "cancelled"]["order_id"])
        sold_by_lot: Dict[str, float] = {}
        if not cb.batch_sales.empty:
            valid_bs = cb.batch_sales[~cb.batch_sales["order_id"].isin(canc_oids)]
            for lot, sq, rq in zip(valid_bs["batch_id"], valid_bs["sold_base_qty"], valid_bs["returned_base_qty"]):
                sold_by_lot[lot] = sold_by_lot.get(lot, 0.0) + (float(sq) - float(rq))
        cb.batches["remaining_base_qty"] = [
            max(0.0, float(iq) - sold_by_lot.get(str(lot), 0.0))
            for lot, iq in zip(cb.batches["batch_id"], cb.batches["import_base_qty"])
        ]

    # 2. Stock movements from the_kho (dev_006, biz_123)
    if "the_kho" in tables:
        df_tk = tables["the_kho"]
        mv_rows = []
        last_stock: Dict[str, float] = {}
        for _, r in df_tk.iterrows():
            raw_p = clean_str(r.get("MaHangHoa"))
            pk, _ = _resolve_prod_key(cb, raw_p, "")
            dt_str = clean_str(r.get("NgayGioPhatSinh"))
            dt = parse_flexible_date(dt_str)
            q_chg = float(parse_vn_number(r.get("SLQuyDoi")) or 0.0)
            end_st = float(parse_vn_number(r.get("TonCuoi")) or 0.0)
            last_stock[pk] = end_st
            mv_rows.append({
                "product_key": pk,
                "date": dt,
                "datetime_str": dt_str,
                "tx_type": clean_str(r.get("LoaiGiaoDich")),
                "qty_change": q_chg,
                "ending_stock": end_st,
            })
        cb.stock_movements = pd.DataFrame(mv_rows)
        cb.stock_snapshot = pd.DataFrame([
            {
                "product_key": pk,
                "product_name": cb.prod_key_to_name.get(pk, pk),
                "base_unit": cb.prod_base_unit.get(pk, ""),
                "ending_base_qty": qty,
            }
            for pk, qty in last_stock.items()
        ])
        cb.capabilities["has_stock_data"] = not cb.stock_snapshot.empty
        return

    # 3. Dedicated Stock Snapshot Table (BaoCaoTonKho, TonKho, ton_kho)
    st_tname = None
    for cand in ("BaoCaoTonKho", "TonKho", "ton_kho"):
        if cand in tables:
            st_tname = cand
            break

    if st_tname is not None:
        df_s = tables[st_tname]
        c_pc = _first_present(df_s, ["Mã hàng", "ma_sp", "ID SP"])
        c_pn = _first_present(df_s, ["Tên hàng", "Tên sản phẩm", "Hàng hóa", "Product", "ten_hang"])
        c_u = _first_present(df_s, ["ĐVT", "ĐV gốc", "Đơn vị", "Đơn vị tính"])
        c_q = _first_present(df_s, ["Tồn cuối kỳ", "Tồn cuối", "Tồn thực", "Ton kho"])

        s_rows = []
        for _, r in df_s.iterrows():
            raw_pc = clean_str(r[c_pc]) if c_pc else ""
            raw_pn = clean_str(r[c_pn]) if c_pn else ""
            if not raw_pc and not raw_pn:
                continue
            pk, p_disp = _resolve_prod_key(cb, raw_pc, raw_pn)
            u_str = clean_str(r[c_u]) if c_u else cb.prod_base_unit.get(pk, "")
            qty = float(parse_vn_number(r[c_q]) or 0.0) if c_q else 0.0
            s_rows.append({
                "product_key": pk,
                "product_name": p_disp,
                "base_unit": u_str,
                "ending_base_qty": qty,
            })
        cb.stock_snapshot = pd.DataFrame(s_rows)
        cb.capabilities["has_stock_data"] = not cb.stock_snapshot.empty
        return

    # 4. Stock column inside product catalog (danh_muc_hang['Tồn cuối'] in dev_007, DanhMucHang['Tồn thực'] in dev_008)
    for cat_t in ("danh_muc_hang", "DanhMucHang"):
        if cat_t in tables:
            df_cat = tables[cat_t]
            c_q = _first_present(df_cat, ["Tồn cuối", "Tồn thực"])
            c_pc = _first_present(df_cat, ["Mã SKU", "Mã sản phẩm bán"])
            c_pn = _first_present(df_cat, ["Hàng bán", "Hàng"])
            c_u = _first_present(df_cat, ["ĐVT", "DVT"])
            if c_q:
                s_rows = []
                for _, r in df_cat.iterrows():
                    pk, p_disp = _resolve_prod_key(cb, clean_str(r[c_pc]) if c_pc else "", clean_str(r[c_pn]) if c_pn else "")
                    u_str = clean_str(r[c_u]) if c_u else cb.prod_base_unit.get(pk, "")
                    qty = float(parse_vn_number(r[c_q]) or 0.0)
                    s_rows.append({
                        "product_key": pk,
                        "product_name": p_disp,
                        "base_unit": u_str,
                        "ending_base_qty": qty,
                    })
                cb.stock_snapshot = pd.DataFrame(s_rows)
                cb.capabilities["has_stock_data"] = not cb.stock_snapshot.empty
                return

    # 5. Aggregate remaining_base_qty from batches (biz_121, biz_122, dev_005)
    if not cb.batches.empty and "remaining_base_qty" in cb.batches.columns:
        grp = cb.batches.groupby("product_key", as_index=False)["remaining_base_qty"].sum()
        s_rows = []
        for pk, qty in zip(grp["product_key"], grp["remaining_base_qty"]):
            s_rows.append({
                "product_key": pk,
                "product_name": cb.prod_key_to_name.get(pk, pk),
                "base_unit": cb.prod_base_unit.get(pk, ""),
                "ending_base_qty": float(qty),
            })
        cb.stock_snapshot = pd.DataFrame(s_rows)
        cb.capabilities["has_stock_data"] = not cb.stock_snapshot.empty
        return

    cb.capabilities["has_stock_data"] = False


def _build_financials_and_aux(cb: CanonicalBusiness, tables: Dict[str, pd.DataFrame]) -> None:
    pay_rows: List[Dict[str, Any]] = []

    # 1. Standard Payment Receipt Tables (SoQuy, BanHang_PhieuThu, thanh_toan)
    for pt_name in ("SoQuy", "BanHang_PhieuThu", "thanh_toan"):
        if pt_name in tables:
            df_p = tables[pt_name]
            c_pid = _first_present(df_p, ["Mã phiếu", "Mã thanh toán", "Phiếu thu", "MaPhieuThu", "Mã phiếu thu", "Mã TT", "Mã GD"])
            c_oid = _first_present(df_p, ["Mã hóa đơn", "Ma don", "Mã bill", "MaHoaDon", "Mã đơn", "Số HĐ", "Số CT", "Mã giao dịch"])
            c_dt = _first_present(df_p, ["Thời gian", "Ngày TT", "Thời gian thu", "NgayGioThu", "Ngày thu", "Ngay thu", "Thời điểm"])
            c_amt = _first_present(df_p, ["Giá trị", "Tiền thu", "Số tiền thu", "SoTienThu", "So tien", "Tiền"])
            c_pm = _first_present(df_p, ["Phương thức thanh toán", "PTTT", "PhuongThuc", "Phương thức", "HTTT", "Loại TT"])

            for idx, r in df_p.iterrows():
                oid = clean_str(r[c_oid]) if c_oid else ""
                dt = parse_flexible_date(r[c_dt]) if c_dt else None
                amt = float(parse_vn_number(r[c_amt]) or 0.0) if c_amt else 0.0
                if not oid or not dt or amt == 0.0:
                    continue
                pay_rows.append({
                    "payment_id": clean_str(r[c_pid]) if c_pid else f"PAY_{idx:06d}",
                    "order_id": oid,
                    "date": dt,
                    "amount": amt,
                    "payment_method": normalize_payment_method(r[c_pm]) if c_pm else "",
                    "is_installment": False,
                })

    # 2. Phone Shop Sổ nợ (dev_014)
    initial_debt_map: Dict[str, float] = {}
    if "Sổ nợ" in tables:
        df_sn = tables["Sổ nợ"].copy()
        if "Ghi chú" in df_sn.columns:
            df_sn["Ghi chú"] = df_sn["Ghi chú"].ffill()
        for idx, r in df_sn.iterrows():
            note_s = clean_str(r.get("Ghi chú"))
            m_stt = re.search(r"(\d+)", note_s)
            oid = f"STT_{int(m_stt.group(1))}" if m_stt else ""
            if not oid:
                continue
            no_v = parse_vn_number(r.get("Nợ"))
            if no_v is not None and no_v > 0:
                initial_debt_map[oid] = float(no_v)
            paid_v = parse_vn_number(r.get("Đã trả"))
            p_dt = parse_flexible_date(r.get("Ngày trả"))
            if paid_v is not None and paid_v > 0 and p_dt is not None:
                pay_rows.append({
                    "payment_id": f"SN_{idx:06d}",
                    "order_id": oid,
                    "date": p_dt,
                    "amount": float(paid_v),
                    "payment_method": "debt_repayment",
                    "is_installment": False,
                })

    # 3. Phone Shop Trả góp (dev_014)
    if "Trả góp" in tables:
        df_tg = tables["Trả góp"].copy()
        if "STT" in df_tg.columns:
            df_tg["STT"] = df_tg["STT"].ffill()
        inst_contracts: Dict[str, Dict[str, Any]] = {}
        inst_pay_rows: List[Dict[str, Any]] = []
        for idx, r in df_tg.iterrows():
            stt_val = r.get("STT")
            if pd.isna(stt_val) or str(stt_val).strip() == "":
                continue
            stt_int = int(float(stt_val))
            oid = f"STT_{stt_int}"
            if oid not in inst_contracts and pd.notna(r.get("Ngày")):
                dt_buy = parse_flexible_date(r.get("Ngày"))
                raw_c = clean_str(r.get("Khách"))
                raw_ph = clean_str(r.get("SĐT"))
                ckey, cname, _ = _resolve_cust_key(cb, "", raw_c, raw_ph, False)
                price_v = float(parse_vn_number(r.get("Giá")) or 0.0)
                down_v = float(parse_vn_number(r.get("Trả trước")) or 0.0)
                n_per = int(float(parse_vn_number(r.get("Số kỳ")) or 0))
                m_amt = float(parse_vn_number(r.get("Góp/tháng")) or 0.0)
                due_day = int(float(parse_vn_number(r.get("Hạn góp (ngày)")) or 1))
                init_debt = max(0.0, price_v - down_v)
                initial_debt_map[oid] = init_debt
                inst_contracts[oid] = {
                    "order_id": oid,
                    "contract_stt": stt_int,
                    "date": dt_buy,
                    "customer_key": ckey,
                    "customer_name": cname,
                    "phone": re.sub(r"\D", "", raw_ph),
                    "device_name": clean_str(r.get("Máy")),
                    "price": price_v,
                    "down_payment": down_v,
                    "initial_debt": init_debt,
                    "num_periods": n_per,
                    "monthly_amount": m_amt,
                    "due_day": due_day,
                }
            p_dt = parse_flexible_date(r.get("Ngày góp"))
            p_amt = parse_vn_number(r.get("Đã góp"))
            if p_dt is not None and p_amt is not None and p_amt > 0:
                inst_pay_rows.append({
                    "order_id": oid,
                    "date": p_dt,
                    "amount": float(p_amt),
                })
                pay_rows.append({
                    "payment_id": f"TG_{idx:06d}",
                    "order_id": oid,
                    "date": p_dt,
                    "amount": float(p_amt),
                    "payment_method": "installment",
                    "is_installment": True,
                })
        cb.installments = pd.DataFrame(list(inst_contracts.values()))
        cb.installment_payments = pd.DataFrame(inst_pay_rows)

    cb.payments = pd.DataFrame(pay_rows)

    # Update cb.orders with initial_debt_map (dev_014) or payment-based initial debt
    if not cb.orders.empty:
        if initial_debt_map:
            cb.orders["initial_debt"] = cb.orders["order_id"].map(lambda x: initial_debt_map.get(x, 0.0))
            cb.orders.loc[cb.orders["order_id"].isin(initial_debt_map), "payment_method"] = "debt"
        else:
            # Compute same-day payment vs post-order payment when cb.payments is available
            same_day_paid: Dict[str, float] = {}
            total_paid_all: Dict[str, float] = {}
            if not cb.payments.empty:
                oid_to_dt = dict(zip(cb.orders["order_id"], cb.orders["date"]))
                for p_oid, p_dt, p_amt in zip(cb.payments["order_id"], cb.payments["date"], cb.payments["amount"]):
                    total_paid_all[p_oid] = total_paid_all.get(p_oid, 0.0) + float(p_amt)
                    o_dt = oid_to_dt.get(p_oid)
                    if o_dt and p_dt and p_dt <= o_dt:
                        same_day_paid[p_oid] = same_day_paid.get(p_oid, 0.0) + float(p_amt)

            init_debts = []
            for _, r in cb.orders.iterrows():
                oid = r["order_id"]
                tot = float(r["total_payable"] or 0.0)
                pm = str(r.get("payment_method") or "")
                debt_end = r.get("debt_at_end")
                paid_pur = r.get("paid_at_purchase")
                if not cb.payments.empty:
                    # In businesses with payment tables (SoQuy, thanh_toan, BanHang_PhieuThu),
                    # an order is a credit order if payment_method == 'debt' or debt_at_end > 0
                    is_credit = (pm == "debt") or (debt_end is not None and pd.notna(debt_end) and float(debt_end) > 0)
                    if is_credit:
                        sd_p = same_day_paid.get(oid, 0.0)
                        init_debts.append(max(0.0, tot - sd_p))
                    else:
                        init_debts.append(0.0)
                else:
                    # No separate payment table (dev_007, dev_008, dev_009, biz_110)
                    if debt_end is not None and pd.notna(debt_end) and float(debt_end) > 0:
                        init_debts.append(float(debt_end))
                    elif pm == "debt" and paid_pur is not None and pd.notna(paid_pur):
                        init_debts.append(max(0.0, tot - float(paid_pur)))
                    else:
                        init_debts.append(0.0)
            cb.orders["initial_debt"] = init_debts
            cb.orders.loc[cb.orders["initial_debt"] > 0, "payment_method"] = "debt"

    # 4. Warranties (Bảo hành in dev_014)
    if "Bảo hành" in tables:
        df_bh = tables["Bảo hành"]
        w_rows = []
        for idx, r in df_bh.iterrows():
            r_dt = parse_flexible_date(r.get("Ngày nhận"))
            b_dt = parse_flexible_date(r.get("Ngày mua"))
            raw_c = clean_str(r.get("Khách"))
            raw_ph = clean_str(r.get("SĐT"))
            ckey, cname, _ = _resolve_cust_key(cb, "", raw_c, raw_ph, False)
            dev_m = clean_str(r.get("Máy"))
            w_rows.append({
                "warranty_id": f"WAR_{idx:06d}",
                "receive_date": r_dt,
                "purchase_date": b_dt,
                "customer_key": ckey,
                "customer_name": cname,
                "device_model": dev_m,
                "device_model_norm": norm_key(dev_m),
                "imei": clean_str(r.get("IMEI")),
                "issue": clean_str(r.get("Lỗi")),
                "resolution": clean_str(r.get("Xử lý")),
            })
        cb.warranties = pd.DataFrame(w_rows)

    # 5. Deliveries (SoGiaoHang, Bảng kê phiếu giao hàng)
    for del_t in ("SoGiaoHang", "Bảng kê phiếu giao hàng"):
        if del_t in tables:
            df_dl = tables[del_t]
            c_oid = _first_present(df_dl, ["Mã đơn hàng", "Số đơn", "Mã bill"])
            c_rnd = _first_present(df_dl, ["Lần giao", "Lượt giao"])
            c_dt = _first_present(df_dl, ["Ngày giao hàng", "Thời gian giao", "Ngày chở"])
            c_pn = _first_present(df_dl, ["Tên hàng hóa", "Hàng hóa", "Tên hàng"])
            c_u = _first_present(df_dl, ["Đơn vị tính", "ĐV gốc"])
            c_q = _first_present(df_dl, ["Số lượng giao", "SL thực giao"])
            c_plt = _first_present(df_dl, ["Biển số xe"])
            c_drv = _first_present(df_dl, ["Người giao hàng", "Lái xe", "Người chở"])

            d_rows = []
            for _, r in df_dl.iterrows():
                oid = clean_str(r[c_oid]) if c_oid else ""
                rnd = int(float(parse_vn_number(r[c_rnd]) or 1)) if c_rnd else 1
                dt = parse_flexible_date(r[c_dt]) if c_dt else None
                raw_pn = clean_str(r[c_pn]) if c_pn else ""
                pk, p_disp = _resolve_prod_key(cb, "", raw_pn)
                u_str = clean_str(r[c_u]) if c_u else cb.prod_base_unit.get(pk, "")
                u_norm = norm_key(u_str)
                qty = float(parse_vn_number(r[c_q]) or 0.0) if c_q else 0.0
                fac = cb.unit_factors.get((pk, u_norm), 1.0)
                bqty = qty * fac
                d_rows.append({
                    "order_id": oid,
                    "delivery_round": rnd,
                    "date": dt,
                    "product_key": pk,
                    "product_name": p_disp,
                    "unit": u_str,
                    "unit_norm": u_norm,
                    "qty": qty,
                    "base_qty": bqty,
                    "license_plate": clean_str(r[c_plt]) if c_plt else "",
                    "driver_name": clean_str(r[c_drv]) if c_drv else "",
                })
            cb.deliveries = pd.DataFrame(d_rows)
            break

    # 6. Due Dates (SoHanNo, Chi tiết công nợ theo hạn)
    for dd_t in ("SoHanNo", "Chi tiết công nợ theo hạn"):
        if dd_t in tables:
            df_dd = tables[dd_t]
            c_oid = _first_present(df_dd, ["Mã đơn hàng", "So hoa don"])
            c_days = _first_present(df_dd, ["Số ngày được nợ", "Kỳ hạn", "Hạn nợ (ngày)"])
            c_due = _first_present(df_dd, ["Hạn thanh toán", "Ngày phải trả", "Han tra"])

            dd_rows = []
            for _, r in df_dd.iterrows():
                oid = clean_str(r[c_oid]) if c_oid else ""
                days_v = parse_vn_number(r[c_days]) if c_days else None
                due_dt = parse_flexible_date(r[c_due]) if c_due else None
                if oid and due_dt:
                    dd_rows.append({
                        "order_id": oid,
                        "credit_days": int(days_v) if days_v is not None else None,
                        "due_date": due_dt,
                    })
            cb.due_dates = pd.DataFrame(dd_rows)
            break

    # 7. Price History (BangGia, lich_su_gia, BanHang_LichSuGia)
    for ph_t in ("BangGia", "lich_su_gia", "BanHang_LichSuGia"):
        if ph_t in tables:
            df_ph = tables[ph_t]
            c_pc = _first_present(df_ph, ["MaHangHoa", "Mã H.hóa", "Mã sản phẩm"])
            c_pn = _first_present(df_ph, ["Item", "Sản phẩm", "TenHang", "ten_hang", "Diễn giải"])
            c_dt = _first_present(df_ph, ["Ngày áp dụng", "NgayApDung"])
            c_old = _first_present(df_ph, ["Giá cũ"])
            c_new = _first_present(df_ph, ["Giá mới", "GiaBanLe"])

            ph_rows = []
            for _, r in df_ph.iterrows():
                raw_pc = clean_str(r[c_pc]) if c_pc else ""
                raw_pn = clean_str(r[c_pn]) if c_pn else ""
                pk, p_disp = _resolve_prod_key(cb, raw_pc, raw_pn)
                dt = parse_flexible_date(r[c_dt]) if c_dt else None
                old_p = parse_vn_number(r[c_old]) if c_old else None
                new_p = parse_vn_number(r[c_new]) if c_new else None
                if dt is not None and new_p is not None:
                    ph_rows.append({
                        "product_key": pk,
                        "product_name": p_disp,
                        "effective_date": dt,
                        "old_price": float(old_p) if old_p is not None else None,
                        "new_price": float(new_p),
                    })
            cb.price_history = pd.DataFrame(ph_rows)
            break

    # 8. Expenses (SoTay_ChiPhi in dev_003, biz_124)
    if "SoTay_ChiPhi" in tables:
        df_ex = tables["SoTay_ChiPhi"]
        c_dt = _first_present(df_ex, ["Ngày mua", "Ngày chi"])
        c_cat = _first_present(df_ex, ["Nhóm", "Mục"])
        c_amt = _first_present(df_ex, ["Tiền chi", "So tien"])
        c_nt = _first_present(df_ex, ["Lưu ý", "Ghi chú"])

        ex_rows = []
        for _, r in df_ex.iterrows():
            dt = parse_flexible_date(r[c_dt]) if c_dt else None
            cat = clean_str(r[c_cat]) if c_cat else ""
            amt = float(parse_vn_number(r[c_amt]) or 0.0) if c_amt else 0.0
            if dt and amt > 0:
                ex_rows.append({
                    "date": dt,
                    "category": cat,
                    "category_norm": norm_key(cat),
                    "amount": amt,
                    "note": clean_str(r[c_nt]) if c_nt else "",
                })
        cb.expenses = pd.DataFrame(ex_rows)


def _finalize_dates_and_capabilities(cb: CanonicalBusiness, tables: Dict[str, pd.DataFrame]) -> None:
    all_dates: List[date] = []
    for df in (cb.orders, cb.repair_tickets, cb.imports):
        if not df.empty and "date" in df.columns:
            valid_dts = [d for d in df["date"].dropna() if isinstance(d, date)]
            all_dates.extend(valid_dts)

    if all_dates:
        cb.data_start_date = min(all_dates)
        if cb.data_end_date is None:
            cb.data_end_date = max(all_dates)

    # Set capability flags accurately
    cb.capabilities["has_hour_of_day"] = bool(not cb.orders.empty and cb.orders["hour"].notna().any())
    cb.capabilities["has_staff"] = bool(
        (not cb.orders.empty and (cb.orders["staff_id"] != "").any())
        or not cb.repair_tickets.empty
    )
    cb.capabilities["has_channel"] = bool(not cb.orders.empty and (cb.orders["channel"] != "").any())
    cb.capabilities["has_payment_method"] = bool(not cb.orders.empty and (cb.orders["payment_method"] != "").any())
    cb.capabilities["has_room_area"] = bool(not cb.orders.empty and (cb.orders["room_area"] != "").any())
    cb.capabilities["has_customer_group"] = bool(
        not cb.customers.empty and "customer_group_norm" in cb.customers.columns and (cb.customers["customer_group_norm"] != "").any()
    )
    cb.capabilities["has_customer_gender"] = bool(
        not cb.customers.empty and "gender" in cb.customers.columns and (cb.customers["gender"] != "").any()
    )
    cb.capabilities["has_customer_address"] = bool(
        not cb.customers.empty and "address_norm" in cb.customers.columns and (cb.customers["address_norm"] != "").any()
    )
    cb.capabilities["has_debt_data"] = bool(
        not cb.payments.empty
        or (not cb.orders.empty and "initial_debt" in cb.orders.columns and (cb.orders["initial_debt"] > 0).any())
        or (
            not cb.orders.empty
            and "debt_at_end" in cb.orders.columns
            and pd.to_numeric(cb.orders["debt_at_end"], errors="coerce").fillna(0.0).gt(0.0).any()
        )
    )
    cb.capabilities["has_installments"] = not cb.installments.empty
    cb.capabilities["has_warranties"] = not cb.warranties.empty or not cb.repair_tickets.empty
    cb.capabilities["has_deliveries"] = not cb.deliveries.empty
    cb.capabilities["has_due_dates"] = not cb.due_dates.empty
    cb.capabilities["has_expenses"] = not cb.expenses.empty
    cb.capabilities["has_price_history"] = not cb.price_history.empty
    cb.capabilities["has_cancel_reason"] = False  # No business in train/public_test has a cancellation reason column


def load_canonical_business(
    split_dir: Path | str,
    business_id: str,
    cache_dir: Optional[Path | str] = "artifacts/cache",
    use_cache: bool = True,
) -> CanonicalBusiness:
    split_path = Path(split_dir)
    split_name = split_path.name
    if cache_dir:
        c_dir = Path(cache_dir)
        c_dir.mkdir(parents=True, exist_ok=True)
        c_file = c_dir / f"canon_{split_name}_{business_id}.pkl"
        if use_cache and c_file.exists():
            with open(c_file, "rb") as f:
                return pickle.load(f)

    pkg = load_business_package(split_path, business_id, cache_dir=cache_dir, use_cache=True)
    cb = build_canonical_business(pkg)

    if cache_dir:
        c_file = Path(cache_dir) / f"canon_{split_name}_{business_id}.pkl"
        with open(c_file, "wb") as f:
            pickle.dump(cb, f, protocol=pickle.HIGHEST_PROTOCOL)
    return cb
