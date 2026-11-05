"""Robust parsers for Vietnamese money, quantities, units, and sheet years."""

from __future__ import annotations

import math
import re
from typing import Any, Optional, Tuple


def parse_vn_number(val: Any) -> Optional[float]:
    """Parse any Vietnamese currency/number representation into a float.

    Examples handled:
    - 55000, 55000.0 -> 55000.0
    - '55.000đ', '294.770.300đ', '-10.000đ' -> 55000.0, 294770300.0, -10000.0
    - '38,000', '94,231,600' -> 38000.0, 94231600.0
    - '92,286.0000' -> 92286.0
    - '16 076', '50 000' -> 16076.0, 50000.0
    - '25k', '18400k', '105,5k' -> 25000.0, 18400000.0, 105500.0
    - '1,01tr', '1,6256tr', '4,38tr' -> 1010000.0, 1625600.0, 4380000.0
    - '19tr990', '1tr2' -> 19990000.0, 1200000.0
    - '3 chai', '1 ly', '2.5 kg' -> 3.0, 1.0, 2.5
    """
    if val is None:
        return None
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        fval = float(val)
        return fval if math.isfinite(fval) else None

    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "null", "-"):
        return None

    sign = 1.0
    if s.startswith("-"):
        sign = -1.0
        s = s[1:].strip()
    elif s.startswith("+"):
        s = s[1:].strip()

    s_low = s.lower()
    # Remove trailing currency symbols
    s_low = re.sub(r"\s*(?:đ|đồng|dong|d|vnd|vnđ)\s*$", "", s_low).strip()

    # Pattern 1: "19tr990", "1tr2"
    m_tr_mid = re.match(r"^(\d+)\s*tr(?:iệu|ieu)?\s*(\d+)$", s_low)
    if m_tr_mid:
        whole = int(m_tr_mid.group(1))
        frac_str = m_tr_mid.group(2)
        frac = float("0." + frac_str)
        return sign * round((whole + frac) * 1_000_000, 4)

    # Pattern 2: "<num>tr" or "<num> triệu" (e.g. "1,01tr", "1,6256tr", "16.56tr")
    m_tr = re.match(r"^([\d.,]+)\s*tr(?:iệu|ieu)?$", s_low)
    if m_tr:
        num_str = m_tr.group(1).replace(",", ".")
        try:
            return sign * round(float(num_str) * 1_000_000, 4)
        except ValueError:
            return None

    # Pattern 3: "<num>k" or "<num> nghìn/ngàn" (e.g. "25k", "25000k", "105,5k")
    m_k = re.match(r"^([\d.,]+)\s*(?:k|nghìn|nghin|ngàn|ngan)$", s_low)
    if m_k:
        num_str = m_k.group(1)
        if re.match(r"^\d{1,3}(?:\.\d{3})+$", num_str):
            num_str = num_str.replace(".", "")
        else:
            num_str = num_str.replace(",", ".")
        try:
            return sign * round(float(num_str) * 1_000, 4)
        except ValueError:
            return None

    # Pattern 4: Number with comma thousands and dot decimals: "92,286.0000"
    if re.match(r"^\d{1,3}(?:,\d{3})+\.\d+$", s_low):
        return sign * float(s_low.replace(",", ""))

    # Pattern 5: Number with dot thousands and comma decimals: "92.286,50"
    if re.match(r"^\d{1,3}(?:\.\d{3})+,\d+$", s_low):
        return sign * float(s_low.replace(".", "").replace(",", "."))

    # Pattern 6: Dot thousands separator: "55.000", "294.770.300"
    if re.match(r"^\d{1,3}(?:\.\d{3})+$", s_low):
        return sign * float(s_low.replace(".", ""))

    # Pattern 7: Comma thousands separator: "38,000", "94,231,600"
    if re.match(r"^\d{1,3}(?:,\d{3})+$", s_low):
        return sign * float(s_low.replace(",", ""))

    # Pattern 8: Space thousands separator: "16 076", "50 000"
    if re.match(r"^\d{1,3}(?:\s+\d{3})+$", s_low):
        return sign * float(re.sub(r"\s+", "", s_low))

    # Pattern 9: Plain number or number with leading quantity + unit ("3 chai", "1,5 kg")
    m_lead = re.match(r"^(\d+(?:[.,]\d+)?)(?:\s+[a-zA-ZÀ-ỹ]+.*)?$", s_low)
    if m_lead:
        num_s = m_lead.group(1).replace(",", ".")
        try:
            return sign * float(num_s)
        except ValueError:
            return None

    return None


def parse_quantity_and_unit(val: Any) -> Tuple[Optional[float], Optional[str]]:
    """Parse quantity cell that may include a unit suffix (e.g., '3 chai' -> (3.0, 'chai'))."""
    if val is None:
        return None, None
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val), None
    s = str(val).strip()
    if not s:
        return None, None
    m = re.match(r"^([+-]?\d+(?:[.,]\d+)?)\s*([a-zA-ZÀ-ỹ0-9_]*)$", s)
    if m:
        num_str = m.group(1).replace(",", ".")
        unit_str = m.group(2).strip() or None
        try:
            return float(num_str), unit_str
        except ValueError:
            pass
    return parse_vn_number(val), None


def extract_year_from_name(name: str) -> Optional[int]:
    """Extract 4-digit year (2024..2027) from sheet name or file name."""
    m = re.search(r"(202[3-8])", name)
    if m:
        return int(m.group(1))
    return None


parse_qty_and_unit = parse_quantity_and_unit

