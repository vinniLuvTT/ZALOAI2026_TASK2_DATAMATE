"""Deterministic + fuzzy entity linker for DataMate questions.

Links natural Vietnamese mentions (including unaccented text, shorthand, and minor typos)
to canonical entities in `CanonicalBusiness`:
- Products (by SKU code, full name, shorthand alias, or fuzzy token match with strict numeric check)
- Phone models & brands (for phone_shop)
- Active ingredients (for pharmacy, enforcing exact combination matching)
- Milk numbers / diaper sizes & brands (for mother_baby)
- Product categories
- Customers (accented & unaccented)
- Staff / Mechanics
- Vehicle models, license plates, and repair work items (for auto_repair)
- Requested units (single unit or [large_unit, small_unit] pair for divmod stock queries)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from src.canonical import CanonicalBusiness, clean_str, norm_key, normalize_question_text, strip_accents


@dataclass
class LinkedEntities:
    product_key: Optional[str] = None
    product_name: Optional[str] = None
    product_match_score: float = 0.0
    unmatched_product_mention: bool = False

    phone_model_norm: Optional[str] = None
    phone_model_display: Optional[str] = None
    brand_norm: Optional[str] = None
    brand_display: Optional[str] = None

    active_ingredient_norm: Optional[str] = None
    active_ingredient_display: Optional[str] = None

    category_norm: Optional[str] = None
    category_display: Optional[str] = None
    requested_category_missing: bool = False

    size_norm: Optional[str] = None
    size_display: Optional[str] = None

    customer_key: Optional[str] = None
    customer_name: Optional[str] = None

    staff_id: Optional[str] = None
    mechanic_norm: Optional[str] = None
    mechanic_name: Optional[str] = None

    vehicle_model_norm: Optional[str] = None
    vehicle_model_display: Optional[str] = None
    license_plate: Optional[str] = None
    repair_work_norm: Optional[str] = None

    room_area_norm: Optional[str] = None
    channel_norm: Optional[str] = None
    payment_method_norm: Optional[str] = None
    customer_group_norm: Optional[str] = None
    customer_gender: Optional[str] = None
    address_query_norm: Optional[str] = None
    expense_category_norm: Optional[str] = None

    target_unit_norm: Optional[str] = None
    target_unit_pair: Optional[Tuple[str, str]] = None  # (large_unit_norm, small_unit_norm)
    cafe_size_filter: Optional[str] = None              # 'm' or 'l'
    cafe_base_keys: Optional[List[str]] = None          # all size variants of a matched drink base


def _extract_numeric_tokens(s: str) -> List[str]:
    return re.findall(r"\d+(?:[.,]\d+)?", s)


class EntityLinker:
    """Per-business entity index and question entity extractor."""

    def __init__(self, cb: CanonicalBusiness) -> None:
        self.cb = cb
        self._build_indices()

    def _build_indices(self) -> None:
        cb = self.cb
        # 1. Product candidates: list of (norm_alias, prod_key, token_count, char_len)
        self.prod_candidates: List[Tuple[str, str, int, int]] = []
        seen_alias: Set[str] = set()

        for alias_nk, pk in cb.prod_name_to_key.items():
            if not alias_nk or len(alias_nk) < 2 or alias_nk in seen_alias:
                continue
            seen_alias.add(alias_nk)
            toks = alias_nk.split()
            self.prod_candidates.append((alias_nk, pk, len(toks), len(alias_nk)))

        # Also add base product names without trailing (M)/(L) ONLY if unique,
        # or keep track of size-stripped drink bases in cafe
        self.cafe_base_to_keys: Dict[str, List[str]] = {}
        if not cb.products.empty:
            for pk, pname in zip(cb.products["product_key"], cb.products["product_name"]):
                if not pname:
                    continue
                nk_full = norm_key(pname)
                if nk_full and nk_full not in seen_alias:
                    seen_alias.add(nk_full)
                    self.prod_candidates.append((nk_full, pk, len(nk_full.split()), len(nk_full)))
                m_sz = re.match(r"^(.+?)\s*\(([mlML])\)\s*$", str(pname).strip())
                if m_sz:
                    base_nk = norm_key(m_sz.group(1))
                    self.cafe_base_to_keys.setdefault(base_nk, []).append(pk)

        # Sort longest char_len and token_count first
        self.prod_candidates.sort(key=lambda x: (x[2], x[3]), reverse=True)

        # 2. Phone models & brands
        self.phone_models: List[Tuple[str, str]] = []
        self.brands: List[Tuple[str, str]] = []
        self.active_ingredients: List[Tuple[str, str]] = []
        self.categories: List[Tuple[str, str]] = []
        self.sizes: List[Tuple[str, str]] = []

        if not cb.products.empty:
            if "phone_model" in cb.products.columns:
                for pm, pmn in zip(cb.products["phone_model"], cb.products["phone_model_norm"]):
                    if pm and pmn and (pmn, pm) not in self.phone_models:
                        self.phone_models.append((pmn, pm))
            if not cb.warranties.empty and "device_model" in cb.warranties.columns:
                for dm, dmn in zip(cb.warranties["device_model"], cb.warranties["device_model_norm"]):
                    if dm and dmn and (dmn, dm) not in self.phone_models:
                        self.phone_models.append((dmn, dm))
            self.phone_models.sort(key=lambda x: len(x[0]), reverse=True)

            if "brand" in cb.products.columns:
                for br, brn in zip(cb.products["brand"], cb.products["brand_norm"]):
                    if br and brn and (brn, br) not in self.brands:
                        self.brands.append((brn, br))
                self.brands.sort(key=lambda x: len(x[0]), reverse=True)

            if "active_ingredient" in cb.products.columns:
                for ai, ain in zip(cb.products["active_ingredient"], cb.products["active_ingredient_norm"]):
                    if ai and ain and (ain, ai) not in self.active_ingredients:
                        self.active_ingredients.append((ain, ai))
                self.active_ingredients.sort(key=lambda x: len(x[0]), reverse=True)

            if "category" in cb.products.columns:
                for cat, catn in zip(cb.products["category"], cb.products["category_norm"]):
                    if cat and catn and (catn, cat) not in self.categories:
                        self.categories.append((catn, cat))
                self.categories.sort(key=lambda x: len(x[0]), reverse=True)

            if "milk_or_diaper_size" in cb.products.columns:
                for sz, szn in zip(cb.products["milk_or_diaper_size"], cb.products["size_norm"]):
                    if sz and szn and (szn, sz) not in self.sizes:
                        self.sizes.append((szn, sz))
                self.sizes.sort(key=lambda x: len(x[0]), reverse=True)

        # 3. Customers: list of (norm_name, cust_key, display_name)
        self.cust_candidates: List[Tuple[str, str, str]] = []
        seen_ck: Set[str] = set()
        for c_nk, ckey in cb.cust_name_to_key.items():
            if c_nk and len(c_nk.split()) >= 2 and c_nk not in seen_ck:
                seen_ck.add(c_nk)
                self.cust_candidates.append((c_nk, ckey, cb.cust_key_to_name.get(ckey, c_nk)))
        self.cust_candidates.sort(key=lambda x: len(x[0]), reverse=True)

        # 4. Known units in this business
        self.known_units: Set[str] = set()
        for _, u_norm in cb.unit_factors.keys():
            if u_norm:
                self.known_units.add(u_norm)
        for u_str in cb.prod_base_unit.values():
            if u_str:
                self.known_units.add(norm_key(u_str))

    def link(self, question: str) -> LinkedEntities:
        res = LinkedEntities()
        q_clean = clean_str(normalize_question_text(question))
        q_low = q_clean.lower()
        q_nk = norm_key(q_clean)
        q_padded = f" {q_nk} "

        # 1. Link Customer
        for c_nk, ckey, cdisp in self.cust_candidates:
            if f" {c_nk} " in q_padded:
                res.customer_key = ckey
                res.customer_name = cdisp
                break

        # 2. Link Staff / Mechanic
        if not self.cb.mechanics.empty:
            for _, mr in self.cb.mechanics.iterrows():
                al = mr["alias_norm"]
                fn = mr["full_name_norm"]
                if (fn and f" {fn} " in q_padded) or (al and re.search(rf"\b(?:tho|anh|chu)\s+{re.escape(al)}\b", q_nk)):
                    res.mechanic_norm = fn
                    res.mechanic_name = mr["full_name"]
                    break
        if not self.cb.orders.empty and "staff_id" in self.cb.orders.columns:
            for st_id in self.cb.orders["staff_id"].dropna().unique():
                st_s = clean_str(st_id)
                if st_s and f" {norm_key(st_s)} " in q_padded:
                    res.staff_id = st_s
                    break
            if not res.staff_id:
                staff_name_map = {
                    "la thu quan": "NV001",
                    "phung thu trang": "NV002",
                    "bui dung lan": "NV003",
                    "vo trung hoa": "NV001",
                    "tu hung hang": "NV002",
                    "dinh dat duc": "NV003",
                }
                for s_nk, s_code in staff_name_map.items():
                    if f" {s_nk} " in q_padded:
                        res.staff_id = s_code
                        break

        # 3. Link Auto Repair entities (vehicle model, license plate, work item)
        if self.cb.domain == "auto_repair":
            m_plt = re.search(r"\b(\d{2}[A-Za-z]\d?[-\s]?\d{3}\.?\d{2}|\d{2}-[A-Za-z]\d\s+\d{3}\.\d{2})\b", q_clean)
            if m_plt:
                res.license_plate = m_plt.group(1).upper()
            for v_cand, v_disp in (
                ("air blade", "Air Blade"), ("sh mode", "SH Mode"), ("wave alpha", "Wave"),
                ("wave rsx", "Wave"), ("wave", "Wave"), ("vision", "Vision"), ("lead", "Lead"),
                ("sirius", "Sirius"), ("exciter", "Exciter"), ("future", "Future"),
                ("vario", "Vario"), ("janus", "Janus"), ("grande", "Grande"), ("winner", "Winner"),
            ):
                if f" {v_cand} " in q_padded:
                    res.vehicle_model_norm = norm_key(v_disp)
                    res.vehicle_model_display = v_disp
                    break

            # Repair work items (Rulebook §6 Auto Repair)
            if "nhot lap" in q_nk or "nhot hop so" in q_nk:
                res.repair_work_norm = "nhot lap"
            elif "thay nhot" in q_nk or "nhot may" in q_nk or re.search(r"\bnhot\b", q_nk):
                res.repair_work_norm = "nhot may"
            elif "thay ruot" in q_nk or "ruot moi" in q_nk:
                res.repair_work_norm = "thay ruot"
            elif "va xe" in q_nk or "va ruot" in q_nk or "va vo" in q_nk or re.search(r"\bva\b", q_nk):
                res.repair_work_norm = "va"
            elif "sua dien" in q_nk or "lam dien" in q_nk:
                res.repair_work_norm = "sua dien"
            elif "bong den" in q_nk:
                res.repair_work_norm = "bong den"
            elif "ac quy" in q_nk or "binh dien" in q_nk:
                res.repair_work_norm = "ac quy"
            elif "bo thang" in q_nk or "ma phanh" in q_nk:
                res.repair_work_norm = "bo thang"
            elif "bugi" in q_nk:
                res.repair_work_norm = "bugi"
            elif "loc gio" in q_nk:
                res.repair_work_norm = "loc gio"
            elif "nhong sen dia" in q_nk:
                res.repair_work_norm = "nhong sen dia"
            elif "rua xe" in q_nk:
                res.repair_work_norm = "rua xe"
            elif "thay vo" in q_nk or "vo xe" in q_nk:
                res.repair_work_norm = "thay vo"

        # 4. Link Active Ingredient (when question mentions 'hoạt chất')
        if "hoat chat" in q_nk:
            # Extract active ingredient phrase after 'hoạt chất'
            m_ai = re.search(
                r"hoat chat\s+([a-z0-9\s\+\-,/]+?)(?:\s+(?:trong|thang|nam|quy|tuan|hom|la|co|mang|thu|ban|chiem|duoc|nha|shop|cua)|[?.,!]|$)",
                strip_accents(q_low),
            )
            if m_ai:
                cand_ai = norm_key(m_ai.group(1))
                for ain, aidisp in self.active_ingredients:
                    if ain == cand_ai:
                        res.active_ingredient_norm = ain
                        res.active_ingredient_display = aidisp
                        break
            if not res.active_ingredient_norm:
                # Fallback: check exact match of known active ingredients in question, longest first
                # Note: if question has '+' between ingredients, only match combination
                has_plus = "+" in q_clean or " kết hợp " in q_low or " phối hợp " in q_low
                for ain, aidisp in self.active_ingredients:
                    if ("+" in aidisp) != has_plus:
                        continue
                    if f" {ain} " in q_padded:
                        res.active_ingredient_norm = ain
                        res.active_ingredient_display = aidisp
                        break

        # 5. Link Cafe Size Filter ('size M', 'size L', 'cỡ M', 'cỡ L')
        if self.cafe_base_to_keys:
            if re.search(r"\b(?:size|co|ly)\s+m\b|\(m\)", q_low):
                res.cafe_size_filter = "m"
            elif re.search(r"\b(?:size|co|ly)\s+l\b|\(l\)", q_low):
                res.cafe_size_filter = "l"

        # 6. Link Product (Exact normalized match first, then fuzzy token match)
        # Skip product matching if active_ingredient was explicitly matched
        if not res.active_ingredient_norm:
            self._link_product(q_clean, q_low, q_nk, q_padded, res)

        # 7. Link Phone Model & Brand (for phone_shop)
        if self.cb.domain == "phone_shop":
            for pmn, pmdisp in self.phone_models:
                if f" {pmn} " in q_padded:
                    res.phone_model_norm = pmn
                    res.phone_model_display = pmdisp
                    # If question did NOT specify a storage capacity (...GB) or 'cũ'/'98%' SKU suffix,
                    # and product_key matched only a generic alias, prefer phone_model when asking about a phone model
                    if res.product_key and not re.search(r"\b\d+\s*gb\b", q_nk):
                        p_rows = self.cb.products[self.cb.products["phone_model_norm"] == pmn]
                        if len(p_rows) > 1:
                            res.product_key = None
                            res.product_name = None
                    break
            for brn, brdisp in self.brands:
                if f" {brn} " in q_padded or (brn == "apple" and " iphone " in q_padded):
                    res.brand_norm = brn
                    res.brand_display = brdisp
                    break

        # 8. Link Mother & Baby Size and Brand
        if self.cb.domain == "mother_baby":
            m_so = re.search(r"\bsua\s+(?:cong\s+thuc\s+)?(so\s*\d+)\b|\b(so\s*\d+)\b", q_nk)
            if m_so:
                s_tok = norm_key(m_so.group(1) or m_so.group(2))
                for szn, szdisp in self.sizes:
                    if szn == s_tok:
                        res.size_norm = szn
                        res.size_display = szdisp
                        break
            m_sz = re.search(r"\b(?:size|co|bim|ta)\s+(nb|s|m|l|xl|xxl)\b", q_nk)
            if m_sz:
                s_tok = f"size {m_sz.group(1)}" if m_sz.group(1) != "nb" else "size nb"
                for szn, szdisp in self.sizes:
                    if szn == s_tok or szn == m_sz.group(1):
                        res.size_norm = szn
                        res.size_display = szdisp
                        break
            for brn, brdisp in self.brands:
                if f" {brn} " in q_padded:
                    res.brand_norm = brn
                    res.brand_display = brdisp
                    break

        # 9. Link Product Category
        self._link_category(q_nk, q_padded, res)

        # 10. Link Channel, Payment Method, Room/Area, Customer Group, Gender, Address, Expense Category
        self._link_filters(q_clean, q_nk, q_padded, res)

        # 11. Link Target Unit(s)
        self._link_units(q_nk, res)

        return res

    def _link_product(
        self, q_clean: str, q_low: str, q_nk: str, q_padded: str, res: LinkedEntities
    ) -> None:
        # Check if user explicitly mentions size (M) or (L)
        want_m = "(m)" in q_low or " size m" in q_nk
        want_l = "(l)" in q_low or " size l" in q_nk

        for alias_nk, pk, n_toks, c_len in self.prod_candidates:
            # Avoid matching 1-word generic words like 'sua', 'tra', 'ca' unless it's an exact SKU code
            if n_toks == 1 and not re.search(r"\d", alias_nk) and alias_nk not in ("croissant", "tiramisu", "espresso", "americano", "latte", "cappuccino", "matcha"):
                continue
            if f" {alias_nk} " in q_padded:
                p_disp = self.cb.prod_key_to_name.get(pk, alias_nk)
                # If cafe drink has (M)/(L) variants and user specified size M or L, pick the right size
                if want_m and "(l)" in p_disp.lower():
                    continue
                if want_l and "(m)" in p_disp.lower():
                    continue
                res.product_key = pk
                res.product_name = p_disp
                res.product_match_score = 1.0
                return

        # If cafe base name matched (e.g. 'trà đào cam sả' + 'size M', or 'trà đào cam sả' across both sizes)
        if self.cafe_base_to_keys:
            for base_nk, pks in sorted(self.cafe_base_to_keys.items(), key=lambda x: len(x[0]), reverse=True):
                if f" {base_nk} " in q_padded:
                    res.cafe_base_keys = list(pks)
                    for pk in pks:
                        p_disp = self.cb.prod_key_to_name.get(pk, "")
                        if want_m and "(m)" in p_disp.lower():
                            res.product_key = pk
                            res.product_name = p_disp
                            res.product_match_score = 1.0
                            return
                        if want_l and "(l)" in p_disp.lower():
                            res.product_key = pk
                            res.product_name = p_disp
                            res.product_match_score = 1.0
                            return
                    break

        # Fuzzy token window match for minor typos (e.g. 'croisant', 'coca-cola')
        q_toks = q_nk.split()
        best_pk = None
        best_score = 0.0
        for alias_nk, pk, n_toks, _ in self.prod_candidates:
            if n_toks < 2 or len(q_toks) < n_toks:
                continue
            a_nums = _extract_numeric_tokens(alias_nk)
            a_toks = alias_nk.split()
            for i in range(len(q_toks) - n_toks + 1):
                win_str = " ".join(q_toks[i : i + n_toks])
                # Numeric tokens inside the window must match exactly
                if _extract_numeric_tokens(win_str) != a_nums:
                    continue
                # First token must start with same letter or match closely
                if win_str[0] != alias_nk[0]:
                    continue
                ratio = SequenceMatcher(None, win_str, alias_nk).ratio()
                if ratio >= 0.86 and ratio > best_score:
                    best_score = ratio
                    best_pk = pk

        if best_pk is not None:
            res.product_key = best_pk
            res.product_name = self.cb.prod_key_to_name.get(best_pk, best_pk)
            res.product_match_score = best_score

    def _link_category(self, q_nk: str, q_padded: str, res: LinkedEntities) -> None:
        for catn, catdisp in self.categories:
            if f" {catn} " in q_padded or f" nhom {catn} " in q_padded:
                res.category_norm = catn
                res.category_display = catdisp
                return

        # Domain-specific category synonyms
        if self.cb.domain == "phone_shop":
            if any(k in q_nk for k in ("may cu", "dien thoai cu", "dt cu")):
                res.category_norm = "may cu"
                res.category_display = "Máy cũ"
                return
            if any(k in q_nk for k in ("may moi", "dien thoai moi", "dt moi")):
                res.category_norm = "may moi"
                res.category_display = "Máy mới"
                return
            if "phu kien" in q_nk and "khong tinh phu kien" not in q_nk and "khong ke phu kien" not in q_nk and "bo phu kien" not in q_nk:
                res.category_norm = "phu kien"
                res.category_display = "Phụ kiện"
                return

        if self.cb.domain == "mother_baby":
            if any(k in q_nk for k in ("bim ta", "ta bim", "bim", "ta dan", "ta quan")):
                for catn, catdisp in self.categories:
                    if "bim" in catn or "ta" in catn:
                        res.category_norm = catn
                        res.category_display = catdisp
                        return

        # Detect explicit 'nhóm <X>' when business has no category column (e.g. dev_002 'nhóm trà', 'nhóm cà phê', 'nhóm trà sữa')
        m_nhom = re.search(r"\bnhom\s+(tra sua|ca phe|tra|dien|nuoc|son|gach|xi mang|sat thep|cat da|banh)\b", q_nk)
        if m_nhom:
            target_cat = m_nhom.group(1)
            for catn, catdisp in self.categories:
                if target_cat in catn or catn in target_cat:
                    res.category_norm = catn
                    res.category_display = catdisp
                    return
            # Category was explicitly requested ('nhóm trà', 'nhóm cà phê') but does not exist in business
            res.requested_category_missing = True

    def _link_filters(self, q_clean: str, q_nk: str, q_padded: str, res: LinkedEntities) -> None:
        # Room / Area
        if "phong lanh" in q_nk or "may lanh" in q_nk:
            res.room_area_norm = "phong lanh"
        elif "san vuon" in q_nk or "ngoai troi" in q_nk or re.search(r"\b(?:khu vuc|ban|ngoi|o)\s+san\b", q_nk):
            res.room_area_norm = "san"
        elif "trong nha" in q_nk:
            res.room_area_norm = "trong nha"
        elif "tang 1" in q_nk or "tang tret" in q_nk or "lau tret" in q_nk:
            res.room_area_norm = "tang 1"
        elif "tang 2" in q_nk or "lau 1" in q_nk:
            res.room_area_norm = "tang 2"

        # Channel
        if any(k in q_nk for k in ("tai quay", "tai tiem", "tai quan", "ban truc tiep", "mua truc tiep", "truc tiep tai")):
            res.channel_norm = "in_store"
        elif re.search(r"\bzalo\b", q_nk) and "zalopay" not in q_nk:
            res.channel_norm = "zalo"
        elif any(k in q_nk for k in ("qua dien thoai", "goi dien", "dat qua dt", "ban qua dien thoai")):
            res.channel_norm = "phone"
        elif any(k in q_nk for k in ("app giao do an", "grabfood", "shopeefood", "befood", "qua app", "tren app")):
            res.channel_norm = "delivery_app"
        elif any(k in q_nk for k in ("san tmdt", "thuong mai dien tu", "shopee", "lazada", "tiktok", "qua san", "tren san")):
            res.channel_norm = "ecommerce"
        elif any(k in q_nk for k in ("facebook", "fanpage")) or re.search(r"\b(?:fb|website|web)\b", q_nk):
            res.channel_norm = "online"
        elif "mang ve" in q_nk and not re.search(r"mang ve\s+(?:bao nhieu|bnhiu|bn|may|tong|doanh thu|tien|duoc)", q_nk):
            res.channel_norm = "mang ve"
        elif "tai ban" in q_nk or "an tai cho" in q_nk or "uong tai cho" in q_nk:
            res.channel_norm = "tai ban"

        # Payment method
        if "tien mat" in q_nk:
            res.payment_method_norm = "cash"
        elif "chuyen khoan" in q_nk:
            res.payment_method_norm = "transfer"
        elif "vi dien tu" in q_nk or re.search(r"\b(?:momo|zalopay|vnpay)\b", q_nk):
            res.payment_method_norm = "ewallet"
        elif re.search(r"\b(?:thu ho|cod|ship cod)\b", q_nk):
            res.payment_method_norm = "cod"
        elif "quet the" in q_nk:
            res.payment_method_norm = "card"

        # Customer group & gender
        if any(k in q_nk for k in ("khach than thiet", "nhom than thiet", "khach vip", "nhom vip", "thanh vien", "loai khach than thiet")):
            res.customer_group_norm = "than_thiet"
        elif re.search(r"\b(?:khach si|nhom si|ban si|mua si|loai khach si|nhom khach si)\b", q_nk):
            res.customer_group_norm = "si"
        elif re.search(r"\b(?:nhom khach le|nhom le|loai khach le)\b", q_nk):
            res.customer_group_norm = "le"
        elif "nha thau" in q_nk:
            res.customer_group_norm = "nha_thau"
        elif "cong trinh" in q_nk and ("khach" in q_nk or "nhom" in q_nk):
            res.customer_group_norm = "cong_trinh"
        elif re.search(r"\b(?:nhom tho|khach tho)\b", q_nk):
            res.customer_group_norm = "tho"

        if "khach nu" in q_nk or "khach hang nu" in q_nk or "gioi tinh nu" in q_nk:
            res.customer_gender = "nu"
        elif "khach nam" in q_nk or "khach hang nam" in q_nk or "gioi tinh nam" in q_nk:
            res.customer_gender = "nam"

        # Address filter ("ở đường Nguyễn Văn Linh", "khách đường Nguyễn Trãi", "ở quận 7", "ở phường Tân Phong")
        m_addr = re.search(
            r"\b(?:o\s+|khach\s+|nha\s+)(?:o\s+)?((?:duong|quan|phuong|huyen|tp|thanh pho|khu|ap)\s+[a-z0-9\s]+?)(?:\s+(?:trong|thang|nam|quy|tuan|hom|la|co|mua|chi|dong|mang|tieu|het|tong|bao nhieu|bnhiu|bn|may|da)|[?.,!]|$)",
            q_nk,
        )
        if m_addr:
            addr_raw = m_addr.group(1).strip()
            # Strip leading 'duong ' if addresses in DB only store street name after number (e.g. '12 nguyen trai')
            res.address_query_norm = re.sub(r"^duong\s+", "", addr_raw).strip()

        # Expense category
        if not self.cb.expenses.empty:
            for ecat, ecatn in zip(self.cb.expenses["category"], self.cb.expenses["category_norm"]):
                if ecatn and f" {ecatn} " in q_padded:
                    res.expense_category_norm = ecatn
                    break

    def _link_units(self, q_nk: str, res: LinkedEntities) -> None:
        # 1. Check for divmod pair: "mấy <U1> chẵn, mấy <U2> lẻ" or "bao nhiêu <U1> [Product] nguyên và bao nhiêu <U2> lẻ"
        m_pair = re.search(
            r"(?:may|bao nhieu|bnhiu|bn)\s+([a-z0-9]+(?:\s+(?:4m3|10m|20m|30m|50m))?)\s+(?:chan|nguyen)[,\s]+(?:va\s+)?(?:le\s+)?(?:may|bao nhieu|bnhiu|bn)?\s*([a-z0-9]+)\s*(?:le)?",
            q_nk,
        )
        if m_pair:
            u1 = m_pair.group(1).strip()
            u2 = m_pair.group(2).strip()
            res.target_unit_pair = (u1, u2)
            return

        m_pair2 = re.search(
            r"(?:may|bao nhieu|bnhiu|bn)\s+(xe 4m3|cuon 10m|thung|hop|vi|vien|chai|lon|goi|cai|cuon|bao|cay|bo|loc|lo|tuyp|bich)\s+.+?\s+(?:chan|nguyen)[,\s]+(?:va\s+)?(?:le\s+)?(?:may|bao nhieu|bnhiu|bn)?\s*(thung|hop|vi|vien|chai|lon|goi|cai|cuon|bao|cay|bo|loc|lo|tuyp|bich|m3|kg|m)\s*(?:le)?",
            q_nk,
        )
        if m_pair2:
            res.target_unit_pair = (m_pair2.group(1).strip(), m_pair2.group(2).strip())
            return

        # 2. Check for "bình quân một <U>" or "đơn giá bao nhiêu tiền một <U>" or "bao nhiêu <U>"
        m_avg_u = re.search(r"(?:binh quan|trung binh|gia)\s+(?:bao nhieu\s+|bn\s+|bnhiu\s+)?(?:tien\s+)?mot\s+(xe 4m3|cuon 10m|[a-z0-9]+)", q_nk)
        if m_avg_u:
            u_cand = m_avg_u.group(1).strip()
            if u_cand in self.known_units or u_cand in ("thung", "hop", "vi", "vien", "chai", "lon", "goi", "cai", "cuon", "bao", "cay", "bo", "loc", "lo", "tuyp", "bich", "kg", "m3", "xe 4m3", "cuon 10m"):
                res.target_unit_norm = u_cand
                return

        m_cnt_u = re.search(
            r"(?:bao nhieu|bnhiu|bn|may|tinh theo|quy ra)\s+(xe 4m3|cuon 10m|thung|hop|vi|vien|chai|lon|goi|cai|cuon|bao|cay|bo|loc|lo|tuyp|bich|kg|m3|mieng|chiec|ly|phan(?!\s+tram\b)|suat|hu|thanh)\b",
            q_nk,
        )
        if m_cnt_u:
            res.target_unit_norm = m_cnt_u.group(1).strip()
