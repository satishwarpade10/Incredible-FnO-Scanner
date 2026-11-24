import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.graph_objects as go

from bs4 import BeautifulSoup
from io import BytesIO
from numpy.lib.stride_tricks import sliding_window_view

from daily_live_fno_scanner import render_daily_live_fno_scanner_tab

# =========================================================
# APP CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="F&O Pattern Scanner",
    page_icon="📈",
    layout="wide"
)

PATTERN_LENGTH = 60
TOP_MATCHES = 20
DTW_SHORTLIST = 100
HORIZONS = [5, 10, 20, 40, 60]

st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}
.stButton button {
    width: 100%;
    min-height: 48px;
    font-weight: bold;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# NSE F&O STOCK LIST
# =========================================================

@st.cache_data(ttl=86400, show_spinner=False)
def get_fno_symbols():

    url = (
        "https://www.nseindia.com/static/"
        "products-services/equity-derivatives-contract-information"
    )

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
        a["href"]
        for a in soup.find_all("a", href=True)
        if "fo_mktlots" in a["href"].lower()
    ]

    if not links:
        raise RuntimeError("NSE F&O stock list was not found.")

    csv_url = links[0]

    if csv_url.startswith("/"):
        csv_url = "https://www.nseindia.com" + csv_url
    elif csv_url.startswith("//"):
        csv_url = "https:" + csv_url

    raw = session.get(csv_url, timeout=30)
    raw.raise_for_status()

    table = pd.read_csv(BytesIO(raw.content))

    table.columns = [
        str(c).strip().upper()
        for c in table.columns
    ]

    symbol_col = next(
        c for c in table.columns
        if "SYMBOL" in c
    )

    section = table.apply(
        lambda row: row.astype(str).str.contains(
            "Individual Securities",
            case=False
        ).any(),
        axis=1
    )

    if not section.any():
        raise RuntimeError("NSE file format changed.")

    start = section[section].index[0]

    rows = table.loc[start:].iloc[1:]

    symbols = rows[symbol_col].dropna().astype(str).str.strip()

    return sorted(set(
        s for s in symbols
        if s and s.upper() not in ("NAN", "SYMBOL")
    ))


# =========================================================
# HISTORICAL OHLC DATA
# =========================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_history(symbol):

    df = yf.download(
        symbol + ".NS",
        period="max",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = [
        "Open", "High", "Low", "Close", "Volume"
    ]

    if not all(c in df.columns for c in required):
        return pd.DataFrame()

    df = df[required].copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    return df


def clean_index(df):

    out = df.copy()
    out.index = pd.to_datetime(out.index)

    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)

    return out.sort_index()


def make_timeframes(df):

    df = clean_index(df)

    aggregation = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }

    return {
        "1D": df,
        "1W": df.resample("W-FRI").agg(
            aggregation
        ).dropna(),
        "1M": df.resample("ME").agg(
            aggregation
        ).dropna()
    }


# =========================================================
# REFERENCE PATTERN
# =========================================================

def get_reference_pattern(df, timeframe, end_date=None):

    df = clean_index(df)

    today = pd.Timestamp.now().normalize()

    if end_date is not None:
        df = df[
            df.index.normalize() <= pd.Timestamp(end_date)
        ]

    # Exclude unfinished candles

    if timeframe == "1D":

        df = df[
            df.index.normalize() < today
        ]

    elif timeframe == "1W":

        week_start = today - pd.Timedelta(
            days=today.weekday()
        )

        df = df[df.index < week_start]

    elif timeframe == "1M":

        df = df[
            (df.index.year < today.year) |
            (
                (df.index.year == today.year) &
                (df.index.month < today.month)
            )
        ]

    if len(df) < PATTERN_LENGTH:
        raise ValueError(
            "Not enough completed candles for this timeframe."
        )

    return df.tail(PATTERN_LENGTH).copy()


# =========================================================
# DTW MATCHING
# =========================================================

def dtw_distance(a, b, window=8):

    n = len(a)
    m = len(b)

    w = max(window, abs(n - m))

    matrix = np.full(
        (n + 1, m + 1),
        np.inf
    )

    matrix[0, 0] = 0

    for i in range(1, n + 1):

        for j in range(
            max(1, i - w),
            min(m, i + w) + 1
        ):

            cost = abs(a[i - 1] - b[j - 1])

            matrix[i, j] = cost + min(
                matrix[i - 1, j],
                matrix[i, j - 1],
                matrix[i - 1, j - 1]
            )

    return matrix[n, m] / (n + m)


# =========================================================
# HISTORICAL PATTERN SCANNER
# =========================================================

def scan_timeframe(
    timeframe,
    reference,
    all_data,
    reference_pattern
):

    target = reference_pattern[
        ["Open", "High", "Low", "Close"]
    ].to_numpy(dtype=float)

    if len(target) != PATTERN_LENGTH:
        raise ValueError("Invalid reference pattern.")

    if target[0, 3] <= 0:
        raise ValueError("Invalid reference price.")

    target_norm = target / target[0, 3] - 1

    target_close = target_norm[:, 3]

    target_start = reference_pattern.index[0]

    candidates = []

    total_windows = 0

    for symbol, frames in all_data.items():

        df = clean_index(frames[timeframe])

        if len(df) < PATTERN_LENGTH:
            continue

        values = df[
            ["Open", "High", "Low", "Close"]
        ].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            continue

        windows = sliding_window_view(
            values,
            window_shape=PATTERN_LENGTH,
            axis=0
        ).transpose(0, 2, 1)

        starts = np.arange(len(windows))

        ends = starts + PATTERN_LENGTH - 1

        # Critical: historical windows must finish
        # before the reference pattern starts.

        eligible = np.asarray(
            df.index[PATTERN_LENGTH - 1:] < target_start
        )

        windows = windows[eligible]
        starts = starts[eligible]
        ends = ends[eligible]

        if len(windows) == 0:
            continue

        total_windows += len(windows)

        base = windows[:, :1, 3:4]

        valid = base[:, 0, 0] != 0

        normalized = (
            windows /
            np.where(base == 0, np.nan, base)
        ) - 1

        valid &= np.isfinite(normalized).all(
            axis=(1, 2)
        )

        if not valid.any():
            continue

        normalized = normalized[valid]
        starts = starts[valid]
        ends = ends[valid]

        # Structure similarity

        target_z = (
            target_close - target_close.mean()
        ) / (target_close.std() + 1e-9)

        window_close = normalized[:, :, 3]

        window_z = (
            window_close -
            window_close.mean(axis=1, keepdims=True)
        ) / (
            window_close.std(axis=1, keepdims=True) + 1e-9
        )

        structure_error = np.mean(
            np.abs(window_z - target_z[None, :]),
            axis=1
        )

        # OHLC candle similarity

        candle_error = np.mean(
            np.abs(
                normalized - target_norm[None, :, :]
            ),
            axis=(1, 2)
        )

        # Fast screening

        fast_score = (
            0.40 * structure_error +
            0.30 * candle_error
        )

        shortlist = np.argsort(fast_score)[
            :DTW_SHORTLIST
        ]

        for ix in shortlist:

            candidates.append({
                "Stock": symbol,
                "Start_Index": int(starts[ix]),
                "End_Index": int(ends[ix]),
                "Structure_Error": float(
                    structure_error[ix]
                ),
                "Candle_Error": float(
                    candle_error[ix]
                )
            })

    if not candidates:
        return pd.DataFrame(), total_windows

    candidates = sorted(
        candidates,
        key=lambda x: (
            0.40 * x["Structure_Error"] +
            0.30 * x["Candle_Error"]
        )
    )[:DTW_SHORTLIST]

    results = []

    for item in candidates:

        symbol = item["Stock"]

        df = clean_index(all_data[symbol][timeframe])

        start = item["Start_Index"]
        end = item["End_Index"]

        historical = df.iloc[
            start:end + 1
        ]

        hist_values = historical[
            ["Open", "High", "Low", "Close"]
        ].to_numpy(dtype=float)

        hist_norm = (
            hist_values / hist_values[0, 3]
        ) - 1

        dtw_error = dtw_distance(
            target_close,
            hist_norm[:, 3]
        )

        score = (
            0.40 * item["Structure_Error"] +
            0.30 * item["Candle_Error"] +
            0.30 * dtw_error
        )

        entry = float(
            historical["Close"].iloc[-1]
        )

        row = {
            "Stock": symbol,
            "Pattern_Start": str(
                historical.index[0].date()
            ),
            "Pattern_End": str(
                historical.index[-1].date()
            ),
            "Similarity_Score": round(score, 6),
            "Structure_Error": round(
                item["Structure_Error"], 6
            ),
            "Candle_Error": round(
                item["Candle_Error"], 6
            ),
            "DTW_Error": round(dtw_error, 6),
            "_Start_Index": start,
            "_End_Index": end
        }

        # Historical future close returns

        for horizon in HORIZONS:

            future_index = end + horizon

            if future_index < len(df) and entry > 0:

                future_close = float(
                    df["Close"].iloc[future_index]
                )

                row[f"After_{horizon}"] = round(
                    (future_close / entry - 1) * 100,
                    2
                )

            else:
                row[f"After_{horizon}"] = np.nan

        results.append(row)

    result = pd.DataFrame(results)

    result = result.sort_values(
        "Similarity_Score",
        ascending=True
    ).head(TOP_MATCHES).reset_index(drop=True)

    return result, total_windows


# =========================================================
# NORMALIZED OHLC FOR VISUAL COMPARISON
# =========================================================

def normalized_ohlc(df):

    out = df.copy()

    base = float(out["Close"].iloc[0])

    if base <= 0:
        return out

    for column in ["Open", "High", "Low", "Close"]:

        out[column] = (
            out[column] / base - 1
        ) * 100

    return out


# =========================================================
# CANDLESTICK CHART
# =========================================================

def candle_chart(df, title):

    chart_data = normalized_ohlc(df)

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=chart_data.index,
            open=chart_data["Open"],
            high=chart_data["High"],
            low=chart_data["Low"],
            close=chart_data["Close"],
            name="OHLC"
        )
    )

    fig.update_layout(
        title=title,
        height=380,
        xaxis_title="Date",
        yaxis_title="Change from first close (%)",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        margin=dict(l=10, r=10, t=55, b=10)
    )

    return fig


# =========================================================
# OVERLAY CHART
# =========================================================

def overlay_chart(reference_df, historical_df, title):

    ref = (
        reference_df["Close"] /
        reference_df["Close"].iloc[0] - 1
    ) * 100

    hist = (
        historical_df["Close"] /
        historical_df["Close"].iloc[0] - 1
    ) * 100

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(range(len(ref))),
            y=ref,
            mode="lines",
            name="Reference pattern"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(len(hist))),
            y=hist,
            mode="lines",
            name="Historical pattern"
        )
    )

    fig.update_layout(
        title=title,
        height=330,
        xaxis_title="Candle number",
        yaxis_title="Change from first close (%)",
        template="plotly_white"
    )

    return fig


# =========================================================
# HISTORICAL FUTURE MOVEMENT CHART
# =========================================================

def future_chart(df, end_index, title):

    start = end_index - PATTERN_LENGTH + 1

    segment = df.iloc[
        start:min(len(df), end_index + 61)
    ]

    if segment.empty:
        return go.Figure()

    base = float(segment["Close"].iloc[0])

    change = (
        segment["Close"] / base - 1
    ) * 100

    x = np.arange(len(change))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=change,
            mode="lines",
            name="Historical close movement"
        )
    )

    fig.add_vline(
        x=PATTERN_LENGTH - 1,
        line_dash="dash",
        annotation_text="Historical pattern ends"
    )

    fig.update_layout(
        title=title,
        height=300,
        xaxis_title="Candles",
        yaxis_title="Change from pattern start (%)",
        template="plotly_white"
    )

    return fig


# =========================================================
# APPLICATION INTERFACE
# =========================================================

st.title("📈 F&O PATTERN SCANNER")

tab_pattern, tab_daily_live = st.tabs([
    "Historical Pattern Scanner",
    "Daily Live F&O Scanner",
])

with tab_pattern:

    st.caption(
        "Historical NSE F&O pattern matching using real OHLC data"
    )

    st.info(
        "The scanner compares 60 completed candles against "
        "earlier historical NSE F&O patterns. "
        "Historical similarity is not a prediction."
    )

    try:

        symbols = get_fno_symbols()

        st.success(
            f"NSE F&O stocks loaded: {len(symbols)}"
        )

    except Exception as error:

        st.error(
            f"Unable to load the official NSE F&O list: {error}"
        )

        st.stop()


    # User inputs

    c1, c2 = st.columns(2)

    with c1:

        default_index = (
            symbols.index("SBICARD")
            if "SBICARD" in symbols else 0
        )

        reference = st.selectbox(
            "Reference stock",
            symbols,
            index=default_index
        )

    with c2:

        timeframes = st.multiselect(
            "Select timeframe",
            ["1D", "1W", "1M"],
            default=["1D"]
        )

    end_date = st.date_input(
        "Reference pattern ending date (optional)",
        value=None
    )

    st.caption(
        "Leave the date empty to use the latest completed candles."
    )

    run = st.button(
        "🔍 FIND HISTORICAL MATCHES",
        type="primary",
        use_container_width=True
    )


    # =========================================================
    # RUN SCANNER
    # =========================================================

    if run:

        if not timeframes:

            st.error("Please select at least one timeframe.")

            st.stop()

        all_data = {}
        errors = []

        progress = st.progress(0)

        status = st.empty()

        with st.spinner(
            "Downloading historical OHLC data..."
        ):

            for i, symbol in enumerate(symbols):

                status.write(
                    f"Downloading {symbol} ({i + 1}/{len(symbols)})"
                )

                try:

                    data = load_history(symbol)

                    if not data.empty and len(data) >= 100:

                        all_data[symbol] = make_timeframes(data)

                except Exception as error:

                    errors.append(
                        f"{symbol}: {str(error)[:100]}"
                    )

                progress.progress(
                    (i + 1) / len(symbols)
                )

        status.empty()

        if reference not in all_data:

            st.error(
                f"No usable historical data for {reference}."
            )

            st.stop()

        st.success(
            f"Historical data ready for {len(all_data)} stocks."
        )

        if errors:

            st.warning(
                f"{len(errors)} stocks could not be downloaded. "
                "Results may not include every F&O stock."
            )

        combined_results = []

        # Scan each selected timeframe

        for timeframe in timeframes:

            st.divider()

            st.header(
                f"{timeframe} Historical Pattern Results"
            )

            try:

                reference_pattern = get_reference_pattern(
                    all_data[reference][timeframe],
                    timeframe,
                    end_date
                )

                st.subheader(
                    f"Current reference: {reference}"
                )

                st.caption(
                    f"{reference_pattern.index[0].date()} "
                    f"to {reference_pattern.index[-1].date()} "
                    f"| {len(reference_pattern)} candles"
                )

                st.plotly_chart(
                    candle_chart(
                        reference_pattern,
                        f"{reference} — Current 60-Candle Pattern ({timeframe})"
                    ),
                    use_container_width=True
                )

                with st.spinner(
                    f"Searching historical {timeframe} patterns..."
                ):

                    result, total = scan_timeframe(
                        timeframe,
                        reference,
                        all_data,
                        reference_pattern
                    )

                st.metric(
                    "Historical windows screened",
                    f"{total:,}"
                )

                if result.empty:

                    st.warning(
                        "No eligible historical matches found."
                    )

                    continue

                st.subheader("Top 20 Historical Matches")

                display_columns = [
                    "Stock",
                    "Pattern_Start",
                    "Pattern_End",
                    "Similarity_Score",
                    "Structure_Error",
                    "Candle_Error",
                    "DTW_Error"
                ] + [
                    f"After_{h}" for h in HORIZONS
                ]

                st.dataframe(
                    result[display_columns],
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    f"Download {timeframe} results CSV",
                    result[display_columns].to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name=f"{reference}_{timeframe}_matches.csv",
                    mime="text/csv",
                    key=f"download_{timeframe}"
                )

                combined_results.append(
                    result.assign(Timeframe=timeframe)
                )

                # =============================================
                # VISUAL HISTORICAL COMPARISONS
                # =============================================

                st.divider()

                st.header(
                    f"📊 Visual Comparison — {timeframe}"
                )

                st.caption(
                    "Expand each historical match to compare its "
                    "actual OHLC candles with the reference."
                )

                for rank, row in result.iterrows():

                    symbol = row["Stock"]

                    historical_df = clean_index(
                        all_data[symbol][timeframe]
                    )

                    start = int(row["_Start_Index"])
                    end = int(row["_End_Index"])

                    match_pattern = historical_df.iloc[
                        start:end + 1
                    ].copy()

                    label = (
                        f"Match {rank + 1}: {symbol} | "
                        f"Error {row['Similarity_Score']:.6f} | "
                        f"{row['Pattern_Start']} to {row['Pattern_End']}"
                    )

                    with st.expander(
                        label,
                        expanded=(rank == 0)
                    ):

                        left, right = st.columns(2)

                        with left:

                            st.plotly_chart(
                                candle_chart(
                                    reference_pattern,
                                    f"Reference: {reference}"
                                ),
                                use_container_width=True,
                                key=f"ref_{timeframe}_{rank}"
                            )

                        with right:

                            st.plotly_chart(
                                candle_chart(
                                    match_pattern,
                                    f"Historical: {symbol}"
                                ),
                                use_container_width=True,
                                key=f"hist_{timeframe}_{rank}"
                            )

                        st.plotly_chart(
                            overlay_chart(
                                reference_pattern,
                                match_pattern,
                                "Normalized Pattern Shape Comparison"
                            ),
                            use_container_width=True,
                            key=f"overlay_{timeframe}_{rank}"
                        )

                        st.plotly_chart(
                            future_chart(
                                historical_df,
                                end,
                                f"{symbol} — Historical Pattern and Following Movement"
                            ),
                            use_container_width=True,
                            key=f"future_{timeframe}_{rank}"
                        )

                        st.subheader(
                            "Historical subsequent price movement"
                        )

                        outcomes = pd.DataFrame([{
                            "After candles": h,
                            "Close return (%)": row[f"After_{h}"]
                        } for h in HORIZONS])

                        st.dataframe(
                            outcomes,
                            use_container_width=True,
                            hide_index=True
                        )

            except Exception as error:

                st.error(
                    f"{timeframe} scan failed: {error}"
                )

        # =============================================
        # COMBINED DOWNLOAD
        # =============================================

        if combined_results:

            final_results = pd.concat(
                combined_results,
                ignore_index=True
            )

            st.divider()

            st.header("Complete Historical Results")

            st.download_button(
                "⬇️ Download all timeframe results",
                final_results.drop(
                    columns=["_Start_Index", "_End_Index"],
                    errors="ignore"
                ).to_csv(index=False).encode("utf-8"),
                file_name=f"{reference}_complete_history.csv",
                mime="text/csv"
            )

with tab_daily_live:
    render_daily_live_fno_scanner_tab()


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Data source: Yahoo Finance historical adjusted OHLC. "
    "F&O universe: NSE official stock list where accessible."
)

st.caption(
    "Similarity is a historical measurement, not a probability "
    "of future price movement. This application is a research tool, "
    "not investment advice."
)
