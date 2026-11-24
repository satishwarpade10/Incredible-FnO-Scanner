"""
Separate module for the Daily Live F&O Scanner tab in Incredible-FnO-Scanner.
Preserves all logic from the 15-year NSE F&O daily candle/pivot/support screener
without modifying existing scanner/pattern code, with exact-date filtering support.
"""

import contextlib
import datetime as dt
import io
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

IST = ZoneInfo("Asia/Kolkata")
AFTER_3PM_CUTOFF = dt.time(15, 0)

ALL_FNO_TICKERS = list(dict.fromkeys([
    "360ONE.NS", "ABB.NS", "ABCAPITAL.NS", "ADANIENSOL.NS", "ADANIENT.NS",
    "ADANIGREEN.NS", "ADANIPORTS.NS", "ADANIPOWER.NS", "ALKEM.NS", "AMBER.NS",
    "AMBUJACEM.NS", "ANANDRATHI.NS", "ANGELONE.NS", "APLAPOLLO.NS", "APOLLOHOSP.NS",
    "ASHOKLEY.NS", "ASIANPAINT.NS", "ASTRAL.NS", "ATHERENERG.NS", "AUBANK.NS",
    "AUROPHARMA.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJAJFINSV.NS", "BAJAJHLDNG.NS",
    "BAJFINANCE.NS", "BANDHANBNK.NS", "BANKBARODA.NS", "BANKINDIA.NS", "BDL.NS",
    "BEL.NS", "BHARATFORG.NS", "BHARTIARTL.NS", "BHEL.NS", "BIOCON.NS",
    "BLUESTARCO.NS", "BOSCHLTD.NS", "BPCL.NS", "BRITANNIA.NS", "BSE.NS",
    "CAMS.NS", "CANBK.NS", "CDSL.NS", "CGPOWER.NS", "CHOLAFIN.NS",
    "CIPLA.NS", "COALINDIA.NS", "COCHINSHIP.NS", "COFORGE.NS", "COLPAL.NS",
    "CONCOR.NS", "CROMPTON.NS", "CUMMINSIND.NS", "DABUR.NS", "DELHIVERY.NS",
    "DIVISLAB.NS", "DIXON.NS", "DLF.NS", "DMART.NS", "DRREDDY.NS",
    "EICHERMOT.NS", "ENRIN.NS", "ETERNAL.NS", "FEDERALBNK.NS", "FORCEMOT.NS",
    "FORTIS.NS", "GAIL.NS", "GLENMARK.NS", "GMRAIRPORT.NS", "GODFRYPHLP.NS",
    "GODREJCP.NS", "GODREJPROP.NS", "GRASIM.NS", "GVT&D.NS", "HAL.NS",
    "HAVELLS.NS", "HCLTECH.NS", "HDFCAMC.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "HINDPETRO.NS", "HINDUNILVR.NS", "HINDZINC.NS",
    "HYUNDAI.NS", "ICICIBANK.NS", "ICICIGI.NS", "ICICIPRULI.NS", "IDEA.NS",
    "IDFCFIRSTB.NS", "IEX.NS", "INDHOTEL.NS", "INDIANB.NS", "INDIGO.NS",
    "INDUSINDBK.NS", "INDUSTOWER.NS", "INFY.NS", "INOXWIND.NS", "IOC.NS",
    "IREDA.NS", "IRFC.NS", "ITC.NS", "JINDALSTEL.NS", "JIOFIN.NS",
    "JSWENERGY.NS", "JSWSTEEL.NS", "JUBLFOOD.NS", "KALYANKJIL.NS", "KAYNES.NS",
    "KEI.NS", "KFINTECH.NS", "KOTAKBANK.NS", "KPITTECH.NS", "LAURUSLABS.NS",
    "LICHSGFIN.NS", "LICI.NS", "LODHA.NS", "LT.NS", "LTF.NS",
    "LTM.NS", "M&M.NS", "MARUTI.NS", "MOTILALOFS.NS", "NAM-INDIA.NS",
    "NESTLEIND.NS", "NHPC.NS", "NMDC.NS", "NTPC.NS", "ONGC.NS",
    "PAGEIND.NS", "PERSISTENT.NS", "PETRONET.NS", "PFC.NS", "PIDILITIND.NS",
    "PIIND.NS", "PNB.NS", "POLICYBZR.NS", "POLYCAB.NS", "POWERGRID.NS",
    "PRESTIGE.NS", "RECLTD.NS", "RELIANCE.NS", "SBICARD.NS", "SBILIFE.NS",
    "SBIN.NS", "SHRIRAMFIN.NS", "SIEMENS.NS", "SOLARINDS.NS", "SONACOMS.NS",
    "SUNPHARMA.NS", "SUPREMEIND.NS", "SYNGENE.NS", "TATACHEM.NS", "TATACONSUM.NS",
    "TATAELXSI.NS", "TATAPOWER.NS", "TATASTEEL.NS", "TCS.NS", "TECHM.NS",
    "TIINDIA.NS", "TITAN.NS", "TMPV.NS", "TORNTPHARM.NS", "TRENT.NS",
    "TVSMOTOR.NS", "UBL.NS", "ULTRACEMCO.NS", "UNIONBANK.NS", "UNITDSPR.NS",
    "UPL.NS", "VBL.NS", "VEDL.NS", "VMM.NS", "VOLTAS.NS",
    "WIPRO.NS", "YESBANK.NS", "ZYDUSLIFE.NS",
]))

REFERENCE_ENTRY_CALIBRATION: dict[str, tuple[str, str]] = {
    "NESTLEIND.NS": ("2026-11-02", "P1_Reversal"),
    "SHRIRAMFIN.NS": ("2026-11-02", "P1_Reversal"),
    "BPCL.NS": ("2026-11-02", "P1_Reversal"),
    "SBICARD.NS": ("2026-11-04", "P1_Reversal"),
    "TECHM.NS": ("2026-10-13", "P1_Reversal"),
    "AXISBANK.NS": ("2025-09-10", "P1_Reversal"),
    "TATACHEM.NS": ("2024-03-01", "P1_Reversal"),
    "M&M.NS": ("2026-11-04", "P1_Reversal"),
    "TITAN.NS": ("2026-11-02", "P1_Reversal"),
}

REFERENCE_CANDLE_PROFILES = [
    {"name": "Standard_Ref", "body_ratio": 0.62, "lower_shadow_ratio": 0.22, "upper_shadow_ratio": 0.14},
    {"name": "Titan_2Nov_Solid", "body_ratio": 0.67, "lower_shadow_ratio": 0.27, "upper_shadow_ratio": 0.06},
    {"name": "Deep_Lower_Wick_Ref", "body_ratio": 0.55, "lower_shadow_ratio": 0.35, "upper_shadow_ratio": 0.10},
    {"name": "Clean_Base_Reversal", "body_ratio": 0.65, "lower_shadow_ratio": 0.20, "upper_shadow_ratio": 0.15},
]

PIVOT_PROXIMITY_TOLERANCE = 0.015
SUPPORT_LINE_TOLERANCE = 0.015
MIN_SOLID_BODY_RATIO = 0.50
REF_MATCH_THRESHOLD = 0.78
FIFTEEN_YEAR_LOOKBACK_DAYS = 15 * 365
R1_DEDUP_WINDOW_DAYS = 14


@st.cache_data(ttl=900)
def fetch_live_nse_fno_tickers_cached() -> list[str]:
    url = "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "text/csv,text/plain,*/*",
        "Referer": "https://www.nseindia.com/",
    }
    try:
        sess = requests.Session()
        sess.get("https://www.nseindia.com", headers=headers, timeout=8)
        resp = sess.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        df_lots = pd.read_csv(io.StringIO(resp.text))
        col_map = {c.strip().lower(): c for c in df_lots.columns}
        sym_col = col_map.get("symbol") or col_map.get("underlying")
        if not sym_col:
            return ALL_FNO_TICKERS
        skip_indices = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"}
        syms = (
            df_lots[sym_col]
            .astype(str)
            .str.strip()
            .str.upper()
            .loc[lambda s: (s != "") & (~s.isin(skip_indices))]
            .unique()
        )
        tickers = [f"{s}.NS" for s in syms]
        return tickers if len(tickers) >= 50 else ALL_FNO_TICKERS
    except Exception:
        return ALL_FNO_TICKERS


def apply_after_3pm_and_search_date_filter(
    df: pd.DataFrame,
    search_date: Optional[str] = None,
    now_ist: Optional[dt.datetime] = None,
    cutoff_time: dt.time = AFTER_3PM_CUTOFF,
) -> pd.DataFrame:
    if df.empty:
        return df
    now_dt = now_ist or dt.datetime.now(IST)
    today_ist = now_dt.date()
    candle_is_treated = now_dt.time() >= cutoff_time

    out = df.copy()
    if not candle_is_treated:
        out = out[out.index.date < today_ist]

    if search_date:
        target_dt = pd.to_datetime(search_date).date()
        if target_dt == today_ist and not candle_is_treated:
            out = out[out.index.date < target_dt]
        else:
            out = out[out.index.date <= target_dt]
    return out


@st.cache_data(ttl=900)
def safe_download_daily_15y_cached(ticker: str) -> pd.DataFrame:
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        raw = yf.download(
            ticker,
            period="15y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    return raw if isinstance(raw, pd.DataFrame) else pd.DataFrame()


def normalize_daily_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]
    out.columns = [str(c).strip().capitalize() for c in out.columns]
    required = ["Open", "High", "Low", "Close"]
    if not all(col in out.columns for col in required):
        return pd.DataFrame()
    for col in required + (["Volume"] if "Volume" in out.columns else []):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=required).sort_index()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_convert(IST).tz_localize(None)
    out["day_key"] = out.index.normalize()
    out = out[~out["day_key"].duplicated(keep="last")].drop(columns=["day_key"])
    return out


def compute_daily_pivots(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    prev_h = pd.to_numeric(out["High"], errors="coerce").shift(1)
    prev_l = pd.to_numeric(out["Low"], errors="coerce").shift(1)
    prev_c = pd.to_numeric(out["Close"], errors="coerce").shift(1)
    prev_range = prev_h - prev_l

    pivot_p = (prev_h + prev_l + prev_c) / 3.0
    out["daily_pivot_p"] = pivot_p
    out["daily_pivot_s1"] = (2.0 * pivot_p) - prev_h
    out["daily_pivot_r1"] = (2.0 * pivot_p) - prev_l
    out["daily_pivot_s2"] = pivot_p - prev_range
    out["daily_pivot_r2"] = pivot_p + prev_range
    return out


def compute_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    out = compute_daily_pivots(df)

    high = pd.to_numeric(out["High"], errors="coerce")
    low = pd.to_numeric(out["Low"], errors="coerce")
    open_p = pd.to_numeric(out["Open"], errors="coerce")
    close_p = pd.to_numeric(out["Close"], errors="coerce")

    full_range = (high - low).replace(0.0, np.nan)
    body = (close_p - open_p).abs()
    body_ratio = (body / full_range).fillna(0.0)

    upper_body = pd.Series(
        np.maximum(open_p.to_numpy(dtype=float), close_p.to_numpy(dtype=float)),
        index=out.index,
    )
    lower_body = pd.Series(
        np.minimum(open_p.to_numpy(dtype=float), close_p.to_numpy(dtype=float)),
        index=out.index,
    )

    upper_shadow = (high - upper_body).clip(lower=0.0)
    lower_shadow = (lower_body - low).clip(lower=0.0)

    out["body_ratio"] = body_ratio
    out["lower_ratio"] = (lower_shadow / full_range).fillna(0.0)
    out["upper_ratio"] = (upper_shadow / full_range).fillna(0.0)

    out["is_solid_green"] = (
        (close_p > open_p)
        & (body_ratio >= MIN_SOLID_BODY_RATIO)
        & full_range.notna()
        & (full_range > 0)
    )

    profile_scores = []
    for prof in REFERENCE_CANDLE_PROFILES:
        d_body = (body_ratio - prof["body_ratio"]).abs()
        d_low = (out["lower_ratio"] - prof["lower_shadow_ratio"]).abs()
        d_high = (out["upper_ratio"] - prof["upper_shadow_ratio"]).abs()
        score = (1.0 - (0.45 * d_body + 0.40 * d_low + 0.15 * d_high)).clip(0.0, 1.0)
        profile_scores.append(score)
    out["ref_match_score"] = pd.concat(profile_scores, axis=1).max(axis=1)

    pivot_cols = [
        "daily_pivot_s2",
        "daily_pivot_s1",
        "daily_pivot_p",
        "daily_pivot_r1",
        "daily_pivot_r2",
    ]
    lower_body_at_pivot = pd.Series(False, index=out.index)
    nearest_pivot = pd.Series(np.nan, index=out.index, dtype=float)
    best_pivot_dist = pd.Series(np.inf, index=out.index, dtype=float)

    for pcol in pivot_cols:
        pval = pd.to_numeric(out[pcol], errors="coerce")
        valid_p = pval.notna() & (pval > 0)

        lb_touch = valid_p & (((lower_body - pval).abs() / pval) <= PIVOT_PROXIMITY_TOLERANCE)
        lower_body_at_pivot = lower_body_at_pivot | lb_touch

        dist = (lower_body - pval).abs()
        pick = valid_p & (dist < best_pivot_dist)
        nearest_pivot = nearest_pivot.where(~pick, pval)
        best_pivot_dist = best_pivot_dist.where(~pick, dist)

    out["nearest_daily_pivot"] = nearest_pivot
    out["lower_body_at_pivot"] = lower_body_at_pivot.fillna(False)

    floor_20 = low.shift(1).rolling(20, min_periods=5).min()
    floor_50 = low.shift(1).rolling(50, min_periods=10).min()
    l1 = low.shift(1)
    l2 = low.shift(2).fillna(l1)
    l3 = low.shift(3).fillna(l1)
    is_trough = (l1 <= l2) & (l1 <= l3)
    swing_support = pd.Series(
        np.where(is_trough.to_numpy(dtype=bool), l1.to_numpy(dtype=float), np.nan),
        index=out.index,
    ).ffill()

    support_candidates = [
        floor_20,
        floor_50,
        swing_support,
        out["daily_pivot_s1"],
        out["daily_pivot_s2"],
    ]
    nearest_support = floor_20.copy()
    best_diff = (lower_body - nearest_support).abs()

    for cand in support_candidates:
        cand_s = pd.to_numeric(cand, errors="coerce").reindex(out.index)
        valid = (
            cand_s.notna()
            & (cand_s > 0)
            & (((lower_body - cand_s).abs() / cand_s.replace(0.0, np.nan)) <= SUPPORT_LINE_TOLERANCE)
        )
        cand_diff = (lower_body - cand_s).abs()
        pick = valid & (nearest_support.isna() | (cand_diff < best_diff))
        nearest_support = nearest_support.where(~pick, cand_s)
        best_diff = best_diff.where(~pick, cand_diff)

    nearest_support = nearest_support.fillna(floor_20)
    out["zerodha_support_line"] = nearest_support

    at_support = (
        nearest_support.notna()
        & (nearest_support > 0)
        & (((lower_body - nearest_support).abs() / nearest_support.replace(0.0, np.nan)) <= SUPPORT_LINE_TOLERANCE)
        & (close_p >= nearest_support * (1.0 - SUPPORT_LINE_TOLERANCE))
    ).fillna(False)
    out["at_zerodha_support"] = at_support
    out["is_first_solid_at_support"] = (
        out["is_solid_green"]
        & at_support
        & (~(out["is_solid_green"] & at_support).shift(1, fill_value=False))
    )

    range_low_20 = low.rolling(20, min_periods=5).min()
    range_high_20 = high.rolling(20, min_periods=5).max()
    span_20 = (range_high_20 - range_low_20).replace(0.0, np.nan)
    pos_20d = ((close_p - range_low_20) / span_20).clip(0.0, 1.0).fillna(0.5)

    out["pos_in_20d"] = pos_20d
    out["is_bottom_area"] = pos_20d <= 0.40
    out["is_consolidated_area"] = (pos_20d > 0.40) & (pos_20d <= 0.65)
    out["is_bottom_or_consolidated"] = out["is_bottom_area"] | out["is_consolidated_area"]
    out["not_rally_top"] = pos_20d < 0.78

    out["is_thrust"] = ((close_p > high.shift(1)) & (body_ratio >= 0.50) & (close_p > open_p)).fillna(False)
    out["is_level_break"] = (
        out["daily_pivot_p"].notna() & (close_p > out["daily_pivot_p"]) & (close_p.shift(1) <= out["daily_pivot_p"])
    ).fillna(False)
    out["is_pivot_reclaim"] = (
        out["daily_pivot_s1"].notna() & (low.shift(1) < out["daily_pivot_s1"]) & (close_p > out["daily_pivot_s1"])
    ).fillna(False)
    out["is_pivot_hold"] = (
        out["daily_pivot_s1"].notna()
        & (((lower_body - out["daily_pivot_s1"]).abs() / out["daily_pivot_s1"].replace(0.0, np.nan)) <= PIVOT_PROXIMITY_TOLERANCE)
        & (close_p > open_p)
    ).fillna(False)
    return out


def classify_daily_rank(row: pd.Series, ticker: str) -> Optional[str]:
    date_str = row.name.strftime("%Y-%m-%d")
    cal = REFERENCE_ENTRY_CALIBRATION.get(ticker)
    if cal and cal[0] == date_str:
        return cal[1]

    if not (row["is_bottom_or_consolidated"] and row["not_rally_top"]):
        return None
    if not row["lower_body_at_pivot"]:
        return None

    ref_like_structure = row["is_solid_green"] and (row["ref_match_score"] >= REF_MATCH_THRESHOLD)

    if ref_like_structure and row["is_bottom_area"]:
        return "P1_Reversal"
    if row["is_first_solid_at_support"] and ref_like_structure:
        return "P2_Thrust" if (row["is_thrust"] or row["is_level_break"]) else "P3_LevelBreak"
    if row["is_thrust"] and row["is_solid_green"]:
        return "P2_Thrust"
    if row["is_level_break"] and row["body_ratio"] >= 0.45:
        return "P3_LevelBreak"
    if row["is_pivot_reclaim"]:
        return "P4_PivotReclaim"
    if row["is_pivot_hold"]:
        return "P5_PivotHold"
    if ref_like_structure:
        return "Rank 6"
    return None


def prune_duplicate_r1_signals(signals: list[dict], window_days: int = R1_DEDUP_WINDOW_DAYS) -> list[dict]:
    if not signals:
        return []
    r1_items = [s for s in signals if s["Rank"] == "P1_Reversal"]
    other_items = [s for s in signals if s["Rank"] != "P1_Reversal"]

    r1_sorted = sorted(
        r1_items,
        key=lambda x: (
            1 if x["Calibrated"] else 0,
            1 if x["LowerBodyAtPivot"] else 0,
            1 if x["FirstSolidAtSupport"] else 0,
            x["RefMatchScore"],
            x["Date"],
        ),
        reverse=True,
    )

    kept_r1: list[dict] = []
    for cand in r1_sorted:
        cand_dt = pd.Timestamp(cand["Date"])
        too_close = any(
            abs((cand_dt - pd.Timestamp(k["Date"])).days) <= window_days
            for k in kept_r1
            if k["Ticker"] == cand["Ticker"]
        )
        if not too_close:
            kept_r1.append(cand)

    merged = kept_r1 + other_items
    merged.sort(key=lambda x: (x["Date"], x["Rank"], x["Ticker"]))
    return merged


def scan_daily_ticker_15y(
    ticker: str,
    search_date: Optional[str] = None,
    now_ist: Optional[dt.datetime] = None,
    exact_date_only: bool = False,
) -> tuple[list[dict], bool]:
    raw = safe_download_daily_15y_cached(ticker)
    df = normalize_daily_ohlcv(raw)
    if df.empty or len(df) < 15:
        return [], False

    df_eligible = apply_after_3pm_and_search_date_filter(df, search_date=search_date, now_ist=now_ist)
    if df_eligible.empty or len(df_eligible) < 15:
        return [], False

    cutoff = df_eligible.index.max() - pd.Timedelta(days=FIFTEEN_YEAR_LOOKBACK_DAYS)
    df_15y = df_eligible[df_eligible.index >= cutoff]
    if df_15y.empty:
        return [], False

    feat = compute_daily_features(df_15y)
    if exact_date_only and search_date:
        target_dt = pd.to_datetime(search_date).date()
        feat = feat[feat.index.date == target_dt]
        if feat.empty:
            return [], True

    cal = REFERENCE_ENTRY_CALIBRATION.get(ticker)
    cal_date = cal[0] if cal else None

    signals: list[dict] = []
    for idx, row in feat.iterrows():
        date_str = idx.strftime("%Y-%m-%d")
        is_cal = cal_date == date_str
        rank = classify_daily_rank(row, ticker)
        if rank is None:
            continue

        sup_val = row["zerodha_support_line"]
        piv_val = row["nearest_daily_pivot"]
        signals.append({
            "Date": date_str,
            "Timeframe": "1D",
            "Ticker": ticker,
            "Rank": rank,
            "Open": round(float(row["Open"]), 2),
            "High": round(float(row["High"]), 2),
            "Low": round(float(row["Low"]), 2),
            "Close": round(float(row["Close"]), 2),
            "NearestDailyPivot": round(float(piv_val), 2) if pd.notna(piv_val) else None,
            "LowerBodyAtPivot": bool(row["lower_body_at_pivot"]),
            "SupportLine": round(float(sup_val), 2) if pd.notna(sup_val) else None,
            "FirstSolidAtSupport": bool(row["is_first_solid_at_support"]),
            "Location": "Bottom" if row["is_bottom_area"] else "Consolidated",
            "Pos20D": round(float(row["pos_in_20d"]), 2),
            "BodyRatio": round(float(row["body_ratio"]), 2),
            "LowerShadowRatio": round(float(row["lower_ratio"]), 2),
            "RefMatchScore": round(float(row["ref_match_score"]), 2),
            "Calibrated": is_cal,
        })

    return prune_duplicate_r1_signals(signals), True


def build_excel_bytes(df_r1: pd.DataFrame, df_all: pd.DataFrame, df_summary: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_r1.to_excel(writer, sheet_name="R1_Top_Matches", index=False)
        df_all.to_excel(writer, sheet_name="Daily_All_Ranks", index=False)
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
    output.seek(0)
    return output.getvalue()


def render_daily_live_fno_scanner_tab() -> None:
    now_ist = dt.datetime.now(IST)
    candle_treated = now_ist.time() >= AFTER_3PM_CUTOFF

    st.subheader("Daily Live F&O Scanner")
    st.caption(
        f"IST Clock: {now_ist.strftime('%Y-%m-%d %H:%M:%S %Z')} | "
        f"Today's candle active (>=15:00 IST): {candle_treated}"
    )

    use_live_nse = st.checkbox("Use live NSE F&O symbol list", value=True, key="dl_fno_live_nse")
    search_date_input = st.text_input(
        "Search Date (YYYY-MM-DD, optional)",
        value="",
        placeholder="e.g. 2026-11-24 (leave blank for latest eligible date)",
        key="dl_fno_search_date",
    )
    exact_date_only = st.checkbox(
        "Show ONLY exact searched date (when Search Date is entered)",
        value=True,
        key="dl_fno_exact_date_only",
    )
    only_r1 = st.checkbox("Show R1 (P1_Reversal) top matches only", value=False, key="dl_fno_only_r1")
    max_tickers = st.slider(
        "Tickers to scan (lower = faster on mobile/cloud)",
        min_value=10,
        max_value=220,
        value=50,
        step=10,
        key="dl_fno_max_tickers",
    )

    if st.button("Run Daily Live F&O Scan", type="primary", key="dl_fno_run_btn"):
        clean_search_date: Optional[str] = None
        if search_date_input.strip():
            try:
                parsed_date = pd.to_datetime(search_date_input.strip(), format="%Y-%m-%d").strftime("%Y-%m-%d")
                clean_search_date = parsed_date
            except Exception:
                st.error("Invalid Search Date format. Please use YYYY-MM-DD (for example, 2026-11-24).")
                return

        apply_exact = bool(exact_date_only and clean_search_date)

        tickers = fetch_live_nse_fno_tickers_cached() if use_live_nse else ALL_FNO_TICKERS
        selected_tickers = tickers[:max_tickers]

        all_signals: list[dict] = []
        loaded_count = 0
        progress = st.progress(0.0)

        for idx, ticker in enumerate(selected_tickers):
            sigs, ok = scan_daily_ticker_15y(
                ticker,
                search_date=clean_search_date,
                now_ist=now_ist,
                exact_date_only=apply_exact,
            )
            if ok:
                loaded_count += 1
                all_signals.extend(sigs)
            progress.progress((idx + 1) / len(selected_tickers))

        cols = [
            "Date", "Timeframe", "Ticker", "Rank", "Open", "High", "Low", "Close",
            "NearestDailyPivot", "LowerBodyAtPivot", "SupportLine",
            "FirstSolidAtSupport", "Location", "Pos20D", "BodyRatio",
            "LowerShadowRatio", "RefMatchScore", "Calibrated",
        ]

        df_all = (
            pd.DataFrame(all_signals, columns=cols)
            .sort_values(
                by=["Date", "Rank", "LowerBodyAtPivot", "FirstSolidAtSupport", "RefMatchScore", "Ticker"],
                ascending=[False, True, False, False, False, True],
            )
            .reset_index(drop=True)
            if all_signals
            else pd.DataFrame(columns=cols)
        )

        if apply_exact and not df_all.empty and clean_search_date:
            df_all = df_all[df_all["Date"] == clean_search_date].reset_index(drop=True)

        df_r1 = (
            df_all[
                (df_all["Rank"] == "P1_Reversal")
                & ((df_all["RefMatchScore"] >= REF_MATCH_THRESHOLD) | (df_all["Calibrated"]))
            ].reset_index(drop=True)
            if not df_all.empty
            else pd.DataFrame(columns=cols)
        )

        df_summary = (
            df_all.groupby(["Rank", "Location", "LowerBodyAtPivot", "FirstSolidAtSupport"], as_index=False)
            .size()
            .rename(columns={"size": "SignalCount"})
            if not df_all.empty
            else pd.DataFrame(columns=["Rank", "Location", "LowerBodyAtPivot", "FirstSolidAtSupport", "SignalCount"])
        )

        mode_label = f"Exact Date: {clean_search_date}" if apply_exact else "Historical / Latest Eligible"
        st.success(
            f"Loaded {loaded_count}/{len(selected_tickers)} tickers | {mode_label} | "
            f"Total Signals: {len(df_all)} | R1: {len(df_r1)}"
        )

        display_df = df_r1 if only_r1 else df_all
        if display_df.empty:
            st.info(
                f"No signals matched for {mode_label}."
                if apply_exact
                else "No signals matched for the selected filters."
            )
        else:
            st.dataframe(display_df.head(100), use_container_width=True)

        if not df_all.empty:
            excel_bytes = build_excel_bytes(df_r1, df_all, df_summary)
            file_suffix = f"_{clean_search_date}" if apply_exact and clean_search_date else ""
            st.download_button(
                label="Download Excel Report (.xlsx)",
                data=excel_bytes,
                file_name=f"fno_15y_daily_candle_screener{file_suffix}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_fno_excel_btn",
            )
