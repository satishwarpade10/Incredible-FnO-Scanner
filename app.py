
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from io import BytesIO
from numpy.lib.stride_tricks import sliding_window_view

st.set_page_config(page_title="F&O Pattern Scanner", page_icon="📈", layout="wide")

PATTERN_LENGTH = 60
TOP_MATCHES = 20
DTW_SHORTLIST = 100
HORIZONS = [5, 10, 20, 40, 60]

st.title("📈 F&O Pattern Scanner")
st.caption("Historical NSE F&O stock pattern matching using actual OHLC data")

@st.cache_data(ttl=86400)
def get_fno_symbols():
    url = "https://www.nseindia.com/static/products-services/equity-derivatives-contract-information"
    headers = {
        "User-Agent": "Mozilla/5.0 Chrome/130.0 Safari/537.36",
        "Referer": "https://www.nseindia.com/"
    }
    session = requests.Session()
    session.headers.update(headers)
    response = session.get(url, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    links = [
        a["href"] for a in soup.find_all("a", href=True)
        if "fo_mktlots" in a["href"].lower()
    ]
    if not links:
        raise RuntimeError("Could not find the NSE F&O stock list.")
    csv_url = links[0]
    if csv_url.startswith("/"):
        csv_url = "https://www.nseindia.com" + csv_url
    elif csv_url.startswith("//"):
        csv_url = "https:" + csv_url
    raw = session.get(csv_url, timeout=30)
    raw.raise_for_status()
    table = pd.read_csv(BytesIO(raw.content))
    table.columns = [str(c).strip().upper() for c in table.columns]
    symbol_col = next(c for c in table.columns if "SYMBOL" in c)
    section = table.apply(
        lambda row: row.astype(str).str.contains(
            "Individual Securities", case=False
        ).any(), axis=1
    )
    if not section.any():
        raise RuntimeError("NSE Individual Securities section not found.")
    start = section[section].index[0]
    rows = table.loc[start:].iloc[1:]
    symbols = rows[symbol_col].dropna().astype(str).str.strip()
    return sorted(set(
        s for s in symbols
        if s and s.upper() not in ("NAN", "SYMBOL")
    ))

@st.cache_data(ttl=3600, show_spinner=False)
def load_history(symbol):
    df = yf.download(
        symbol + ".NS", period="max", interval="1d",
        auto_adjust=True, progress=False, threads=False
    )
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()
    return df[required].dropna()

def make_timeframes(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    agg = {
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum"
    }
    return {
        "1D": df,
        "1W": df.resample("W-FRI").agg(agg).dropna(),
        "1M": df.resample("ME").agg(agg).dropna()
    }

def clean_index(df):
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    return out.sort_index()

def dtw_distance(a, b, window=8):
    n, m = len(a), len(b)
    w = max(window, abs(n - m))
    matrix = np.full((n + 1, m + 1), np.inf)
    matrix[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(max(1, i - w), min(m, i + w) + 1):
            cost = abs(a[i - 1] - b[j - 1])
            matrix[i, j] = cost + min(
                matrix[i - 1, j],
                matrix[i, j - 1],
                matrix[i - 1, j - 1]
            )
    return matrix[n, m] / (n + m)

def scan_timeframe(tf, reference, all_data, end_date):
    target_df = clean_index(all_data[reference][tf])
    today = pd.Timestamp.now().normalize()

    if end_date:
        target_df = target_df[
            target_df.index.normalize() <= pd.Timestamp(end_date)
        ]

    if tf == "1D":
        target_df = target_df[target_df.index.normalize() < today]
    elif tf == "1W":
        week_start = today - pd.Timedelta(days=today.weekday())
        target_df = target_df[target_df.index < week_start]
    else:
        target_df = target_df[
            (target_df.index.year < today.year) |
            ((target_df.index.year == today.year) &
             (target_df.index.month < today.month))
        ]

    if len(target_df) < PATTERN_LENGTH:
        raise ValueError("Reference stock has fewer than 60 completed candles.")

    target = target_df.iloc[-60:][
        ["Open", "High", "Low", "Close"]
    ].to_numpy(float)

    if target[0, 3] <= 0:
        raise ValueError("Invalid reference price.")

    target_norm = target / target[0, 3] - 1
    target_close = target_norm[:, 3]
    target_start = target_df.index[0]

    pieces = []
    total_windows = 0

    for symbol, frames in all_data.items():
        df = clean_index(frames[tf])
        if len(df) < 61:
            continue

        values = df[
            ["Open", "High", "Low", "Close"]
        ].to_numpy(float)

        if not np.isfinite(values).all():
            continue

        windows = sliding_window_view(
            values, window_shape=60, axis=0
        ).transpose(0, 2, 1)

        starts = np.arange(len(windows))
        ends = starts + 59

        eligible = np.asarray(df.index[59:] < target_start)
        windows = windows[eligible]
        starts = starts[eligible]
        ends = ends[eligible]

        if len(windows) == 0:
            continue

        total_windows += len(windows)

        base = windows[:, :1, 3:4]
        valid = base[:, 0, 0] != 0
        norm = windows / np.where(base == 0, np.nan, base) - 1
        valid &= np.isfinite(norm).all(axis=(1, 2))

        if not valid.any():
            continue

        norm = norm[valid]
        starts = starts[valid]
        ends = ends[valid]

        t = (
            target_close - target_close.mean()
        ) / (target_close.std() + 1e-9)

        wc = norm[:, :, 3]
        z = (
            wc - wc.mean(axis=1, keepdims=True)
        ) / (wc.std(axis=1, keepdims=True) + 1e-9)

        structure = np.mean(np.abs(z - t[None, :]), axis=1)
        candle = np.mean(
            np.abs(norm - target_norm[None, :, :]), axis=(1, 2)
        )
        fast_score = 0.40 * structure + 0.30 * candle

        take = np.argsort(fast_score)[:DTW_SHORTLIST]

        for ix in take:
            pieces.append((
                float(fast_score[ix]), symbol,
                int(starts[ix]), int(ends[ix]),
                float(structure[ix]), float(candle[ix])
            ))

    if not pieces:
        return pd.DataFrame(), total_windows

    pieces.sort(key=lambda x: x[0])
    candidates = pieces[:DTW_SHORTLIST]
    rows = []

    for _, symbol, start, end, se, ce in candidates:
        df = clean_index(all_data[symbol][tf])
        hist = df.iloc[start:end + 1][
            ["Open", "High", "Low", "Close"]
        ].to_numpy(float)

        hist_norm = hist / hist[0, 3] - 1
        de = dtw_distance(target_close, hist_norm[:, 3])
        score = 0.40 * se + 0.30 * ce + 0.30 * de

        row = {
            "Stock": symbol,
            "Pattern_Start": str(df.index[start].date()),
            "Pattern_End": str(df.index[end].date()),
            "Similarity_Score": round(score, 6),
            "Structure_Error": round(se, 6),
            "Candle_Error": round(ce, 6),
            "DTW_Error": round(de, 6)
        }

        base_close = float(df["Close"].iloc[end])
        for h in HORIZONS:
            future_index = end + h
            row[f"After_{h}"] = (
                round(
                    (float(df["Close"].iloc[future_index]) /
                     base_close - 1) * 100, 2
                )
                if future_index < len(df) and base_close else np.nan
            )
        rows.append(row)

    result = pd.DataFrame(rows)
    result = result.sort_values("Similarity_Score")
    return result.head(TOP_MATCHES).reset_index(drop=True), total_windows

try:
    symbols = get_fno_symbols()
    st.success(f"NSE F&O stocks loaded: {len(symbols)}")
except Exception as error:
    st.error(f"Could not load live NSE F&O list: {error}")
    st.stop()

reference = st.selectbox(
    "Enter or select reference stock",
    symbols,
    index=symbols.index("SBICARD") if "SBICARD" in symbols else 0
)

timeframes = st.multiselect(
    "Select timeframe",
    ["1D", "1W", "1M"],
    default=["1D", "1W", "1M"]
)

end_date = st.date_input(
    "Reference pattern ending date (optional)",
    value=None
)

st.info(
    "The scanner compares 60 completed candles with earlier "
    "historical windows. Historical similarity is not a prediction."
)

if st.button("🔍 FIND HISTORICAL MATCHES", type="primary",
             use_container_width=True):

    if not timeframes:
        st.error("Select at least one timeframe.")
        st.stop()

    all_data = {}
    errors = []
    progress = st.progress(0)

    with st.spinner("Downloading historical OHLC data. First scan may take time..."):
        for i, symbol in enumerate(symbols):
            try:
                data = load_history(symbol)
                if not data.empty and len(data) >= 100:
                    all_data[symbol] = make_timeframes(data)
            except Exception as error:
                errors.append(f"{symbol}: {str(error)[:80]}")
            progress.progress((i + 1) / len(symbols))

    if reference not in all_data:
        st.error(f"No usable history found for {reference}.")
        st.stop()

    st.success(f"Historical data ready for {len(all_data)} stocks.")

    combined = []

    for tf in timeframes:
        with st.spinner(f"Scanning {tf} historical patterns..."):
            try:
                result, total = scan_timeframe(
                    tf, reference, all_data, end_date
                )
                st.subheader(f"{tf} — Top historical matches")
                st.caption(f"{total:,} earlier windows screened")

                if result.empty:
                    st.warning("No eligible historical matches found.")
                    continue

                st.dataframe(
                    result, use_container_width=True, hide_index=True
                )

                st.download_button(
                    f"Download {tf} CSV",
                    result.to_csv(index=False).encode("utf-8"),
                    file_name=f"{reference}_{tf}_matches.csv",
                    mime="text/csv",
                    key=f"download_{tf}"
                )

                combined.append(result.assign(Timeframe=tf))

            except Exception as error:
                st.error(f"{tf} scan failed: {error}")

    if combined:
        final = pd.concat(combined, ignore_index=True)

        st.subheader("Combined results")
        st.download_button(
            "Download all timeframe results",
            final.to_csv(index=False).encode("utf-8"),
            file_name=f"{reference}_all_matches.csv",
            mime="text/csv"
        )

        st.subheader("Historical outcome summary")
        for tf in final["Timeframe"].unique():
            part = final[final["Timeframe"] == tf]
            summary = []
            for h in HORIZONS:
                values = pd.to_numeric(
                    part[f"After_{h}"], errors="coerce"
                ).dropna()
                summary.append({
                    "After candles": h,
                    "Average movement (%)": round(values.mean(), 2)
                    if len(values) else None,
                    "Positive outcomes (%)": round(
                        (values > 0).mean() * 100, 1
                    ) if len(values) else None,
                    "Samples": len(values)
                })
            st.write(f"**{tf}**")
            st.dataframe(
                pd.DataFrame(summary),
                use_container_width=True,
                hide_index=True
            )

    if errors:
        st.warning(
            f"{len(errors)} stocks could not be downloaded. "
            "Some historical matches may be missing."
        )

st.divider()
st.caption(
    "Data: Yahoo Finance adjusted historical OHLC; NSE F&O list "
    "where accessible. Research tool only, not investment advice."
)
