"""
Velos — Road Fatality Analysis
Run: python -m streamlit run app.py
Install: python -m pip install streamlit pandas duckdb httpx openpyxl
"""

import streamlit as st
import pandas as pd
import numpy as np
import duckdb, httpx, re, io
from datetime import datetime

st.set_page_config(page_title="Velos", page_icon=None, layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{background-color:#080C10!important;color:#C8D8E8!important;font-family:'DM Sans',sans-serif!important}
.main{background-color:#080C10!important}
.block-container{padding-top:1.8rem!important;padding-left:2rem!important;padding-right:2rem!important;max-width:100%!important}
[data-testid="stSidebar"]{background:#0D1117!important;border-right:1px solid #1A2535!important}
[data-testid="stSidebar"] *{color:#C8D8E8!important}
[data-testid="stSidebar"] label{color:#5A8FAA!important;font-family:'Space Mono',monospace!important;font-size:10px!important;letter-spacing:.12em!important;text-transform:uppercase!important}
h1{font-family:'Space Mono',monospace!important;font-size:1.4rem!important;font-weight:700!important;color:#00D4FF!important;letter-spacing:.06em!important;text-transform:uppercase!important;border-bottom:1px solid #1A2535!important;padding-bottom:.5rem!important;margin-bottom:1rem!important}
h2{font-family:'Space Mono',monospace!important;font-size:.75rem!important;color:#5A8FAA!important;letter-spacing:.15em!important;text-transform:uppercase!important;font-weight:400!important}
h3{font-family:'DM Sans',sans-serif!important;color:#00D4FF!important;font-size:1rem!important;font-weight:500!important}
.stCaption,[data-testid="stCaptionContainer"]{color:#3D5A70!important;font-family:'Space Mono',monospace!important;font-size:10px!important;letter-spacing:.06em!important}
[data-testid="stMetric"]{background:#0D1117!important;border:1px solid #1A2535!important;border-top:2px solid #00D4FF!important;border-radius:3px!important;padding:1rem 1.2rem!important}
[data-testid="stMetricLabel"]{font-family:'Space Mono',monospace!important;font-size:9px!important;color:#3D5A70!important;letter-spacing:.15em!important;text-transform:uppercase!important}
[data-testid="stMetricValue"]{font-family:'Space Mono',monospace!important;font-size:1.7rem!important;color:#00D4FF!important;font-weight:700!important}
[data-testid="stTabs"] [role="tablist"]{background:#0D1117!important;border-bottom:1px solid #1A2535!important}
[data-testid="stTabs"] button[role="tab"]{font-family:'Space Mono',monospace!important;font-size:10px!important;letter-spacing:.12em!important;text-transform:uppercase!important;color:#3D5A70!important;border:none!important;border-bottom:2px solid transparent!important;padding:.7rem 1.4rem!important;background:transparent!important;border-radius:0!important;transition:all .2s!important}
[data-testid="stTabs"] button[role="tab"]:hover{color:#00D4FF!important}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{color:#00D4FF!important;border-bottom:2px solid #00D4FF!important}
.stButton button{background:transparent!important;border:1px solid #00D4FF!important;color:#00D4FF!important;font-family:'Space Mono',monospace!important;font-size:10px!important;letter-spacing:.1em!important;text-transform:uppercase!important;border-radius:2px!important;padding:.4rem 1rem!important;transition:all .15s!important}
.stButton button:hover{background:#00D4FF!important;color:#080C10!important}
.stButton button:disabled{border-color:#1A2535!important;color:#1A2535!important}
.stTextInput input,.stTextArea textarea{background:#0D1117!important;border:1px solid #1A2535!important;border-radius:2px!important;color:#C8D8E8!important;font-family:'Space Mono',monospace!important;font-size:12px!important}
.stTextInput input:focus,.stTextArea textarea:focus{border-color:#00D4FF!important;box-shadow:0 0 0 1px #00D4FF22!important}
[data-testid="stDataFrame"]{border:1px solid #1A2535!important;border-radius:3px!important}
[data-testid="stDataFrame"] th{background:#0D1117!important;color:#5A8FAA!important;font-family:'Space Mono',monospace!important;font-size:9px!important;letter-spacing:.1em!important;text-transform:uppercase!important;border-bottom:1px solid #1A2535!important}
[data-testid="stDataFrame"] td{color:#C8D8E8!important;font-family:'Space Mono',monospace!important;font-size:11px!important;border-bottom:1px solid #0D1117!important}
[data-testid="stDataFrame"] tr:hover td{background:#111820!important}
.stSuccess{background:#071A10!important;border:1px solid #00FF8833!important;border-left:3px solid #00FF88!important;border-radius:2px!important}
.stError{background:#1A0808!important;border:1px solid #FF334433!important;border-left:3px solid #FF3344!important;border-radius:2px!important}
.stWarning{background:#1A1208!important;border:1px solid #FFB30033!important;border-left:3px solid #FFB300!important;border-radius:2px!important}
.stInfo{background:#071218!important;border:1px solid #00D4FF22!important;border-left:3px solid #00D4FF!important;border-radius:2px!important}
[data-testid="stChatMessage"]{background:#0D1117!important;border:1px solid #1A2535!important;border-radius:3px!important;margin-bottom:.5rem!important}
[data-testid="stChatInput"] textarea{background:#0D1117!important;color:#C8D8E8!important;font-family:'DM Sans',sans-serif!important}
[data-testid="stFileUploader"]{background:#0D1117!important;border:1px dashed #1A2535!important;border-radius:3px!important}
[data-testid="stMultiSelect"] span[data-baseweb="tag"]{background:#00D4FF15!important;border:1px solid #00D4FF33!important;border-radius:2px!important;color:#00D4FF!important;font-family:'Space Mono',monospace!important;font-size:10px!important}
hr{border-color:#1A2535!important;margin:1rem 0!important}
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:#080C10}
::-webkit-scrollbar-thumb{background:#1A2535;border-radius:2px}
::-webkit-scrollbar-thumb:hover{background:#00D4FF33}
[data-testid="stExpander"]{background:#0D1117!important;border:1px solid #1A2535!important;border-radius:3px!important}
[data-testid="stExpander"] summary{font-family:'Space Mono',monospace!important;font-size:10px!important;letter-spacing:.1em!important;color:#5A8FAA!important}
[data-testid="stDownloadButton"] button{background:transparent!important;border:1px solid #1A2535!important;color:#5A8FAA!important;font-family:'Space Mono',monospace!important;font-size:10px!important;letter-spacing:.1em!important;text-transform:uppercase!important}
[data-testid="stDownloadButton"] button:hover{border-color:#00D4FF!important;color:#00D4FF!important}
code,pre{background:#0D1117!important;border:1px solid #1A2535!important;color:#00D4FF!important;font-family:'Space Mono',monospace!important;font-size:11px!important;border-radius:2px!important}
</style>
""", unsafe_allow_html=True)

for k, v in [("df", None), ("filename", None), ("chat", []), ("load_note", "")]:
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _is_year_col(col):
    return bool(re.fullmatch(r"\d{4}", str(col).strip()))

def _guess_roles(df):
    """Sniff column names and content to find country / year / value columns."""
    country_col = year_col = value_col = None
    for c in df.columns:
        lc = c.lower().strip()
        if country_col is None and any(k in lc for k in ("country", "nation", "region")) and "code" not in lc:
            country_col = c
        if year_col is None and any(k in lc for k in ("year", "yr", "date", "period")):
            year_col = c
        if value_col is None and any(k in lc for k in ("value", "rate", "death", "fatal", "mortality", "count", "total")):
            value_col = c
    # content-based year fallback
    if year_col is None:
        for c in df.columns:
            s = df[c].dropna().astype(str)
            if not s.empty and s.str.fullmatch(r"\d{4}").mean() > 0.9:
                year_col = c
                break
    # first numeric column as value fallback
    if value_col is None:
        for c in df.columns:
            if c != year_col and pd.api.types.is_numeric_dtype(df[c]):
                value_col = c
                break
    return country_col, year_col, value_col

def _reshape_wide_to_long(df):
    """If columns contain 4-digit years, melt into long format."""
    year_cols = [c for c in df.columns if _is_year_col(c)]
    if not year_cols:
        return df, "Already in long format — no reshaping needed."
    id_cols = [c for c in df.columns if c not in year_cols]
    long = df.melt(id_vars=id_cols, value_vars=year_cols, var_name="Year", value_name="Value")
    long["Year"]  = pd.to_numeric(long["Year"],  errors="coerce")
    long["Value"] = pd.to_numeric(long["Value"], errors="coerce")
    long = long.dropna(subset=["Year", "Value"])
    return long, f"Reshaped wide→long: {len(year_cols)} year columns → {len(long):,} rows."

def _rename_to_standard(df):
    """Rename detected columns to canonical names."""
    country_col, year_col, value_col = _guess_roles(df)
    rename = {}
    if country_col and country_col != "Country Name": rename[country_col] = "Country Name"
    if year_col    and year_col    != "Year":         rename[year_col]    = "Year"
    if value_col   and value_col   != "Value":        rename[value_col]   = "Value"
    return df.rename(columns=rename) if rename else df

def load_file(uploaded_file):
    """
    Load any traffic CSV/Excel. Handles:
    - World Bank wide format (4 header rows, years as columns)
    - Plain long-format CSV
    - Excel files
    Returns (dataframe, note_string).
    """
    raw = uploaded_file.read()
    notes = []
    df = None

    if uploaded_file.name.endswith(".csv"):
        # Peek at row 4 to check for World Bank format
        try:
            peek = pd.read_csv(io.BytesIO(raw), skiprows=4, nrows=1, encoding_errors="replace")
            peek.columns = [str(c).strip() for c in peek.columns]
            is_wb = "Country Name" in peek.columns or any(_is_year_col(c) for c in peek.columns)
        except Exception:
            is_wb = False

        if is_wb:
            try:
                candidate = pd.read_csv(io.BytesIO(raw), skiprows=4, encoding_errors="replace")
                candidate = candidate.dropna(axis=1, how="all").dropna(axis=0, how="all")
                candidate.columns = [str(c).strip() for c in candidate.columns]
                year_cols = [c for c in candidate.columns if _is_year_col(c)]
                id_cols   = [c for c in candidate.columns if not _is_year_col(c)]
                if year_cols:
                    df = candidate.melt(
                        id_vars=id_cols, value_vars=year_cols,
                        var_name="Year", value_name="Value")
                    df["Year"]  = pd.to_numeric(df["Year"],  errors="coerce")
                    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
                    df = df.dropna(subset=["Year", "Value"]).reset_index(drop=True)
                    notes.append(f"World Bank format: {len(year_cols)} year columns melted to {len(df):,} rows.")
                else:
                    df = candidate
                    notes.append("World Bank format (no year columns found).")
            except Exception as e:
                df = None
                notes.append(f"WB parse failed ({e}), retrying as plain CSV.")

        if df is None:
            try:
                candidate = pd.read_csv(io.BytesIO(raw), encoding_errors="replace")
                candidate = candidate.dropna(axis=1, how="all").dropna(axis=0, how="all")
                candidate.columns = [str(c).strip() for c in candidate.columns]
                df, reshape_note = _reshape_wide_to_long(candidate)
                notes.append("Plain CSV. " + reshape_note)
            except Exception as e:
                raise ValueError(f"Could not read CSV: {e}")

    elif uploaded_file.name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(raw))
        df.columns = [str(c).strip() for c in df.columns]
        df, reshape_note = _reshape_wide_to_long(df)
        notes.append("Excel. " + reshape_note)
    else:
        raise ValueError("Unsupported file format.")

    if df is None:
        raise ValueError("Could not parse the file.")

    # Normalise column names (World Bank already has "Country Name")
    rename = {}
    for col in df.columns:
        if col.lower().strip() == "country name" and col != "Country Name":
            rename[col] = "Country Name"
        if col.lower().strip() in ("country code", "countrycode") and col != "Country Code":
            rename[col] = "Country Code"
    if rename:
        df = df.rename(columns=rename)

    return df.reset_index(drop=True), " ".join(notes)


# ══════════════════════════════════════════════════════════════════════════════
# AI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _ollama_alive():
    try:
        return httpx.get("http://localhost:11434/", timeout=2).status_code < 500
    except Exception:
        return False

def _build_context(df, filename):
    stats = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            stats.append(f"  {col} [numeric]: min={s.min():.2f}, max={s.max():.2f}, mean={s.mean():.2f}")
        else:
            top = s.value_counts().head(3).index.tolist()
            stats.append(f"  {col} [text]: {s.nunique()} unique, e.g. {top}")

    has_std = all(c in df.columns for c in ("Country Name", "Year", "Value"))
    exact_cols = ", ".join(f'"{c}"' for c in df.columns)
    sql_hint = f'SQL table is `dataset`. Exact column names (always double-quote them): {exact_cols}.'

    return f"""You are a traffic safety data analyst.
Dataset: {filename} — {df.shape[0]:,} rows x {df.shape[1]} columns.
Columns:
{chr(10).join(stats)}
Sample:
{df.head(5).to_csv(index=False)}
{sql_hint}

CRITICAL RULES:
1. To answer any data question write a SQL SELECT inside <sql>...</sql> tags. No exceptions.
2. Only use the exact column names listed above. Never invent column names.
3. Always double-quote column names that contain spaces e.g. "Country Name".
4. Do NOT show or explain the SQL in your reply. The app runs it automatically.
5. After results come back, reply in 2-3 plain sentences only."""

def _call_ai(context, question, provider, model, api_key):
    if provider == "Ollama (local)":
        try:
            r = httpx.post("http://localhost:11434/api/generate",
                json={"model": model, "prompt": f"{context}\n\nUser: {question}", "stream": False},
                timeout=90)
            if r.status_code != 200:
                return f" Ollama error {r.status_code}: {r.text[:200]}"
            return r.json().get("response", "No response.")
        except httpx.ConnectError:
            return " Ollama not running. Open a terminal and run: `ollama serve`"
        except Exception as e:
            return f" Ollama error: {e}"
    else:
        try:
            r = httpx.post("https://api.anthropic.com/v1/messages",
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": "claude-sonnet-4-20250514", "max_tokens": 1024,
                      "system": context, "messages": [{"role": "user", "content": question}]},
                timeout=60)
            r.raise_for_status()
            return r.json()["content"][0]["text"]
        except httpx.HTTPStatusError as e:
            return f" API error {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            return f" Anthropic error: {e}"

def _fix_sql(sql, df):
    for col in df.columns:
        if " " in col:
            sql = re.sub(r'(?<!["\'])' + re.escape(col) + r'(?!["\'])', '"' + col + '"', sql)
    sql = sql.rstrip("; \n").strip()
    return sql

def ask_ai(df, filename, question, provider, model, api_key):
    ctx = _build_context(df, filename)
    raw = _call_ai(ctx, question, provider, model, api_key)

    sql_match = re.search(r"<sql>(.*?)</sql>", raw, re.DOTALL | re.IGNORECASE)
    sql = sql_match.group(1).strip() if sql_match else None
    data = None

    if sql:
        sql = _fix_sql(sql, df)
        try:
            con = duckdb.connect()
            con.register("dataset", df)
            data = con.execute(sql).fetchdf()
            followup = f"SQL results:\n{data.head(20).to_csv(index=False)}\nSummarise in 2-3 plain sentences."
            raw = _call_ai(ctx, followup, provider, model, api_key)
        except Exception as e:
            sql_simple = re.sub(r"WHERE\s+.*?(GROUP BY|ORDER BY|LIMIT|$)", r"\1",
                                sql, flags=re.IGNORECASE | re.DOTALL).strip()
            try:
                con2 = duckdb.connect()
                con2.register("dataset", df)
                data = con2.execute(sql_simple).fetchdf()
                followup = f"SQL results:\n{data.head(20).to_csv(index=False)}\nSummarise in 2-3 plain sentences."
                raw = _call_ai(ctx, followup, provider, model, api_key)
                sql = sql_simple
            except Exception:
                raw += f"\n\n*(SQL failed: {e})*"

    answer = re.sub(r"<sql>.*?</sql>", "", raw, flags=re.DOTALL).strip()
    return {"answer": answer, "sql": sql, "data": data}


def run_sql(df, sql):
    if not sql.strip().upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries are permitted.")
    con = duckdb.connect()
    con.register("dataset", df)
    return con.execute(sql.strip()).fetchdf()

def ai_suggest_loader(df, filename, provider, model, api_key):
    cols   = df.columns.tolist()
    sample = df.head(5).to_csv(index=False)
    q = f"""The dataset '{filename}' loaded with columns: {cols}
Sample:
{sample}
This app needs columns called Country Name, Year, and Value (traffic fatality rate).
Write a short Python snippet that reshapes this dataframe (variable: `df`) to have those columns.
Return only runnable code, no explanation."""
    return _call_ai("You are a Python data expert. Be concise.", q, provider, model, api_key)




# ══════════════════════════════════════════════════════════════════════════════
# ECONOMETRICS HELPERS  (pure numpy/pandas — no external dependencies)
# ══════════════════════════════════════════════════════════════════════════════

def _ols(y, X):
    """
    OLS via normal equations. Returns dict with keys:
    coeffs, se, t_stats, p_values, r2, r2_adj, residuals, fitted, n, k
    Uses HC3 heteroskedasticity-robust standard errors.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    coeffs  = XtX_inv @ X.T @ y
    fitted  = X @ coeffs
    resid   = y - fitted
    # HC3 sandwich variance
    h       = np.einsum('ij,jk,ik->i', X, XtX_inv, X)   # leverage
    e_hc3   = resid / (1 - np.clip(h, 0, 0.9999))
    meat    = (X * e_hc3[:, None]).T @ (X * e_hc3[:, None])
    vcov    = XtX_inv @ meat @ XtX_inv
    se      = np.sqrt(np.diag(vcov))
    t_stats = coeffs / np.where(se > 0, se, np.nan)
    # two-sided p-values via t-distribution approximation (normal for large n)
    from math import erf, sqrt
    def _pval(t):
        # survival function of standard normal * 2
        x = abs(t) / sqrt(2)
        return (1 - erf(x))
    p_values = np.array([_pval(t) for t in t_stats])
    ss_res  = float(resid @ resid)
    ss_tot  = float(((y - y.mean()) ** 2).sum())
    r2      = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    r2_adj  = 1 - (1 - r2) * (n - 1) / max(n - k, 1)
    return dict(coeffs=coeffs, se=se, t_stats=t_stats, p_values=p_values,
                r2=r2, r2_adj=r2_adj, residuals=resid, fitted=fitted, n=n, k=k)

def _add_const(arr):
    """Prepend a column of ones."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    return np.hstack([np.ones((len(arr), 1)), arr])

def _prep_panel(df):
    d = df[["Country Name", "Year", "Value"]].dropna().copy()
    d = d[d["Value"] > 0].copy()
    d["log_value"] = np.log(d["Value"].astype(float))
    d["Year"]      = d["Year"].astype(int)
    return d.reset_index(drop=True)

def run_pooled_ols(df):
    """Pooled OLS: log(mortality) ~ intercept + Year"""
    d  = _prep_panel(df)
    X  = _add_const(d["Year"].values)
    r  = _ols(d["log_value"].values, X)
    return r, d

def run_twfe(df):
    """
    Two-way FE: remove country means from log(mortality), then compute
    year fixed effects on the demeaned outcome.
    With one obs per country-year, Year is collinear with year FEs so we
    report year FEs directly as the within-country global trend.
    Returns: year_fe DataFrame, within-residuals, R²-within, N.
    """
    d = _prep_panel(df).copy()
    # Step 1: remove country fixed effects (demean within country)
    d["country_mean"] = d.groupby("Country Name")["log_value"].transform("mean")
    d["y_within"]     = d["log_value"] - d["country_mean"]
    # Step 2: year FEs = mean of y_within by year
    year_fe = d.groupby("Year")["y_within"].mean().reset_index()
    year_fe.columns = ["Year", "Year FE (avg within)"]
    # Step 3: residuals after removing both FEs
    d["year_fe"]   = d.groupby("Year")["y_within"].transform("mean")
    d["residual"]  = d["y_within"] - d["year_fe"]
    ss_res  = float((d["residual"] ** 2).sum())
    ss_tot  = float((d["y_within"] ** 2).sum())
    r2_w    = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    n_countries = d["Country Name"].nunique()
    n_years     = d["Year"].nunique()
    # Linear trend through year FEs (slope = avg within-country annual change)
    yfe_vals = year_fe["Year FE (avg within)"].values
    yr_vals  = year_fe["Year"].values.astype(float)
    X_trend  = _add_const(yr_vals - yr_vals.mean())
    trend_r  = _ols(yfe_vals, X_trend)
    year_fe["Fitted trend"] = trend_r["fitted"]
    return {
        "year_fe":     year_fe,
        "r2_within":   r2_w,
        "n_countries": n_countries,
        "n_years":      n_years,
        "trend_slope":  float(trend_r["coeffs"][1]),
        "trend_p":      float(trend_r["p_values"][1]),
        "panel_d":      d,
    }

def run_convergence(df):
    """β-convergence: Δlog(mortality_t) ~ intercept + log(mortality_{t-1})"""
    d = _prep_panel(df).sort_values(["Country Name", "Year"])
    d["lag_log"] = d.groupby("Country Name")["log_value"].shift(1)
    d["delta"]   = d["log_value"] - d["lag_log"]
    d = d.dropna(subset=["lag_log", "delta"]).reset_index(drop=True)
    X = _add_const(d["lag_log"].values)
    r = _ols(d["delta"].values, X)
    r["col_names"] = ["const", "lag_log"]
    return r, d

def run_kuznets(df):
    """Quadratic time trend: log(mortality) ~ intercept + t + t²"""
    d     = _prep_panel(df).copy()
    t     = d["Year"].astype(float) - d["Year"].astype(float).mean()
    d["t"]  = t
    d["t2"] = t ** 2
    X = _add_const(np.column_stack([d["t"].values, d["t2"].values]))
    r = _ols(d["log_value"].values, X)
    r["col_names"] = ["const", "t", "t2"]
    return r, d

def country_trends(df):
    """Per-country OLS slope on log(mortality) ~ Year."""
    results = []
    d = _prep_panel(df)
    for country, g in d.groupby("Country Name"):
        if len(g) < 4:
            continue
        try:
            X = _add_const(g["Year"].values)
            r = _ols(g["log_value"].values, X)
            slope = float(r["coeffs"][1])
            pval  = float(r["p_values"][1])
            results.append({
                "Country":         country,
                "Annual % change": round(slope * 100, 2),
                "p-value":         round(pval, 4),
                "N years":         len(g),
                "Significant":     "✓" if pval < 0.05 else "",
                "Direction":       "▼ Improving" if slope < 0 else "▲ Worsening",
            })
        except Exception:
            pass
    out = pd.DataFrame(results)
    if out.empty:
        return out
    return out.sort_values("Annual % change").reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
<div style='padding:1.2rem 0 0.8rem 0'>
  <div style='font-family:Space Mono,monospace;font-size:9px;color:#3D5A70;letter-spacing:.25em;text-transform:uppercase;margin-bottom:6px'>Road Fatality Analysis</div>
  <div style='font-family:Space Mono,monospace;font-size:1.3rem;font-weight:700;color:#00D4FF;letter-spacing:.12em;text-transform:uppercase'>VELOS</div>
</div>
""", unsafe_allow_html=True)
    st.divider()

    st.subheader("Load Dataset")
    uploaded = st.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"])
    if uploaded:
        try:
            df_new, note = load_file(uploaded)
            st.session_state.df        = df_new
            st.session_state.filename  = uploaded.name
            st.session_state.chat      = []
            st.session_state.load_note = note
            st.success(f" {len(df_new):,} rows loaded")
            st.caption(note)
        except Exception as e:
            st.error(f"Could not load: {e}")

    st.divider()
    st.subheader("AI Configuration")
    provider = st.selectbox("Provider", ["Ollama (local)", "Anthropic (cloud)"])
    if provider == "Ollama (local)":
        model   = st.text_input("Model", "phi3:mini")
        api_key = ""
        if _ollama_alive():
            st.success("Ollama running")
        else:
            st.error("Ollama not running. Run: ollama serve")
    else:
        model   = "claude-sonnet-4-20250514"
        api_key = st.text_input("Anthropic API key", type="password", placeholder="sk-ant-...")

    # Filters
    if st.session_state.df is not None:
        fdf = st.session_state.df.copy()
        st.divider()
        st.subheader("Filters")
        if "Country Name" in fdf.columns:
            all_c = sorted(fdf["Country Name"].dropna().unique())
            sel_c = st.multiselect("Countries", all_c, default=all_c[:12])
            if sel_c:
                fdf = fdf[fdf["Country Name"].isin(sel_c)]
        if "Year" in fdf.columns:
            all_y = sorted(fdf["Year"].dropna().unique().astype(int))
            if len(all_y) > 1:
                yr = st.select_slider("Year range", options=all_y, value=(all_y[0], all_y[-1]))
                fdf = fdf[(fdf["Year"] >= yr[0]) & (fdf["Year"] <= yr[1])]
        st.caption(f"{len(fdf):,} / {len(st.session_state.df):,} rows")
    else:
        fdf = None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.df is None:
    st.title(" TrafficSight")
    st.markdown("### AI-powered traffic death data analysis")
    st.info(" Upload a CSV or Excel file to get started.\n\nHandles World Bank exports, long-format CSVs, and most other traffic data files automatically.")
    c1, c2, c3 = st.columns(3)
    c1.success("** Visualise**\nAuto-charts for any format")
    c2.success("** Trends**\nCountry comparisons over time")
    c3.success("** Ask AI**\nPlain-English questions")

else:
    df       = st.session_state.df
    filename = st.session_state.filename
    has_std  = all(c in fdf.columns for c in ("Country Name", "Year", "Value"))

    t1, t2, t3, t4, t5, t6 = st.tabs(["Overview", "Trends", "Compare", "Econometrics", "Query", "Data"])

    # OVERVIEW
    with t1:
        st.title(f"{filename}")
        if st.session_state.load_note:
            st.caption(f"ℹ️ {st.session_state.load_note}")
        if not has_std:
            st.warning("Columns (Country Name / Year / Value) not fully detected. Check your file format.")

        a, b, c, d = st.columns(4)
        a.metric("Rows", f"{len(fdf):,}")
        if "Country Name" in fdf.columns: b.metric("Countries", fdf["Country Name"].nunique())
        if "Value" in fdf.columns:
            c.metric("Avg rate", f"{fdf['Value'].mean():.1f}")
            d.metric("Highest",  f"{fdf['Value'].max():.1f}")
        st.divider()

        if has_std:
            latest = int(fdf["Year"].max())
            top20  = fdf[fdf["Year"] == latest].sort_values("Value", ascending=False).head(20)
            st.subheader(f"Top 20 countries — {latest}")
            st.bar_chart(top20.set_index("Country Name")["Value"])

    # TRENDS
    with t2:
        st.title("Trends")
        if not has_std:
            st.info("Needs Country Name, Year, and Value columns.")
        else:
            all_c  = sorted(fdf["Country Name"].dropna().unique())
            chosen = st.multiselect("Countries to compare", all_c, default=all_c[:6])
            if chosen:
                tdf   = fdf[fdf["Country Name"].isin(chosen)].copy()
                pivot = tdf.pivot_table(index="Year", columns="Country Name", values="Value", aggfunc="mean")
                st.line_chart(pivot)
                rows = []
                for country in chosen:
                    cdf = tdf[tdf["Country Name"] == country].sort_values("Year")
                    if len(cdf) < 2: continue
                    fv, lv = cdf.iloc[0]["Value"], cdf.iloc[-1]["Value"]
                    if pd.isna(fv) or pd.isna(lv) or fv == 0: continue
                    pct = (lv - fv) / fv * 100
                    rows.append({
                        "Country": country,
                        f"First ({int(cdf.iloc[0]['Year'])})":   f"{fv:.1f}",
                        f"Latest ({int(cdf.iloc[-1]['Year'])})": f"{lv:.1f}",
                        "Change": f"{'▼' if pct < 0 else '▲'} {abs(pct):.1f}%",
                        "Status": "Improved" if pct < 0 else "Worsened",
                    })
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("Select at least one country.")

    # COMPARE
    with t3:
        st.title("Country Comparison")
        if has_std:
            latest = int(fdf["Year"].max())
            cmp    = fdf[fdf["Year"] == latest].copy()
            show   = [c for c in ["Country Name", "Country Code", "Value"] if c in cmp.columns]
            cmp    = cmp[show].sort_values("Value", ascending=False)
            st.caption(f"Latest year: **{latest}**  —  Deaths per 100,000 population")
            st.dataframe(cmp.style.bar(subset=["Value"], color="#2E75B6"), use_container_width=True, hide_index=True)
        else:
            st.info("Needs Year and Value columns.")

        st.divider()
        st.subheader("Custom Query")
        st.caption("Table name is `dataset`.")
        default_q = ('SELECT "Country Name", AVG("Value") as avg_rate FROM dataset GROUP BY "Country Name" ORDER BY avg_rate DESC LIMIT 20'
                     if has_std else "SELECT * FROM dataset LIMIT 20")
        sql_in = st.text_area("SQL", value=default_q, height=80)
        if st.button("Run"):
            try:    st.dataframe(run_sql(fdf, sql_in), use_container_width=True)
            except Exception as e: st.error(str(e))


    # ECONOMETRICS
    with t4:
        st.title("Econometrics")
        if not has_std:
            st.info("Needs Country Name, Year, and Value columns.")
        else:
            econ_tabs = st.tabs([
                "Country Trends",
                "Pooled OLS",
                "Two-Way Fixed Effects",
                "Convergence",
                "Time Trend (Kuznets proxy)",
            ])

            # ── COUNTRY TRENDS ──────────────────────────────────────────────
            with econ_tabs[0]:
                st.subheader("Per-Country OLS Trend")
                st.caption(
                    "Annual % change in log(mortality rate) per country via OLS. "
                    "Negative = improving. HC3 heteroskedasticity-robust standard errors."
                )
                with st.spinner("Fitting country regressions..."):
                    ct = country_trends(fdf)
                if ct.empty:
                    st.warning("Not enough data.")
                else:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Countries analysed", len(ct))
                    c2.metric("Improving (sig.)",
                              int(((ct["Annual % change"] < 0) & (ct["Significant"] == "✓")).sum()))
                    c3.metric("Worsening (sig.)",
                              int(((ct["Annual % change"] > 0) & (ct["Significant"] == "✓")).sum()))
                    c4.metric("Median annual change", f"{ct['Annual % change'].median():+.2f}%")
                    st.dataframe(
                        ct.style.bar(subset=["Annual % change"], align="mid",
                                     color=["#FF4444", "#00BB55"]),
                        use_container_width=True, hide_index=True
                    )
                    st.download_button("⬇️ Download CSV",
                        ct.to_csv(index=False).encode(), "country_trends.csv", "text/csv")

            # ── POOLED OLS ───────────────────────────────────────────────────
            with econ_tabs[1]:
                st.subheader("Pooled OLS  —  log(mortality) ~ Year")
                st.caption(
                    "Baseline model ignoring panel structure. HC3 robust SEs. "
                    "Biased due to unobserved country heterogeneity — use TWFE for causal inference."
                )
                with st.spinner("Running OLS..."):
                    ols_r, ols_d = run_pooled_ols(fdf)

                coeff_year = float(ols_r["coeffs"][1])
                pval_year  = float(ols_r["p_values"][1])
                se_year    = float(ols_r["se"][1])
                annual_pct = coeff_year * 100

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("R²",         f"{ols_r['r2']:.4f}")
                c2.metric("Adj. R²",    f"{ols_r['r2_adj']:.4f}")
                c3.metric("Year coeff", f"{coeff_year:.6f}")
                c4.metric("p-value",    f"{pval_year:.4e}")

                st.info(
                    f"**Interpretation:** Each additional year is associated with a "
                    f"**{annual_pct:+.2f}%** change in road mortality on average "
                    f"({'statistically significant' if pval_year < 0.05 else 'not significant'} at 5%)."
                )

                with st.expander("Coefficient table"):
                    tbl = pd.DataFrame({
                        "Term":    ["Intercept", "Year"],
                        "Coeff":   [f"{ols_r['coeffs'][0]:.4f}", f"{coeff_year:.6f}"],
                        "Std Err": [f"{ols_r['se'][0]:.4f}",    f"{se_year:.6f}"],
                        "t-stat":  [f"{ols_r['t_stats'][0]:.3f}", f"{ols_r['t_stats'][1]:.3f}"],
                        "p-value": [f"{ols_r['p_values'][0]:.4f}", f"{pval_year:.4f}"],
                    })
                    st.dataframe(tbl, use_container_width=True, hide_index=True)

                st.subheader("Residuals vs Fitted")
                resid_df = pd.DataFrame({"Fitted": ols_r["fitted"], "Residual": ols_r["residuals"]})
                st.scatter_chart(resid_df.set_index("Fitted"), use_container_width=True)

            # ── TWFE ─────────────────────────────────────────────────────────
            with econ_tabs[2]:
                st.subheader("Two-Way Fixed Effects (TWFE)")
                st.caption(
                    "Demeans country + year effects iteratively, then OLS on residualised data. "
                    "Controls for all time-invariant country traits (α_i) and global year shocks (λ_t)."
                )
                with st.spinner("Fitting TWFE..."):
                    twfe = run_twfe(fdf)

                slope = twfe["trend_slope"]
                pval  = twfe["trend_p"]

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("R² (within)",    f"{twfe['r2_within']:.4f}")
                c2.metric("Countries",        str(twfe["n_countries"]))
                c3.metric("Year trend slope", f"{slope:.6f}")
                c4.metric("p-value",          f"{pval:.4e}")

                st.info(
                    "TWFE strips out country fixed effects (culture, geography, wealth) and "
                    "year fixed effects (global trends). The trend slope reflects "
                    "**within-country** change relative to the global average."
                )

                st.subheader("Year Fixed Effects over time")
                yfe_df = twfe["year_fe"].copy()
                st.line_chart(yfe_df.set_index("Year")[["Year FE (avg within)", "Fitted trend"]], use_container_width=True)

                with st.expander("Year fixed effects table"):
                    st.dataframe(twfe["year_fe"].round(6), use_container_width=True, hide_index=True)

            # ── CONVERGENCE ──────────────────────────────────────────────────
            with econ_tabs[3]:
                st.subheader("β-Convergence")
                st.caption(
                    "Regresses Δlog(mortality_t) on log(mortality_{t-1}). "
                    "β < 0 means convergence — high-mortality countries improve faster."
                )
                with st.spinner("Running convergence regression..."):
                    conv_r, conv_d = run_convergence(fdf)

                beta = float(conv_r["coeffs"][1])
                pval = float(conv_r["p_values"][1])
                half_life = -np.log(2) / np.log(1 + beta) if beta < 0 else None

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("β (convergence)", f"{beta:.4f}")
                c2.metric("p-value",         f"{pval:.4e}")
                c3.metric("R²",              f"{conv_r['r2']:.4f}")
                c4.metric("Half-life (yrs)", f"{half_life:.1f}" if half_life else "N/A")

                if beta < 0 and pval < 0.05:
                    st.success(
                        f"**Convergence detected** (β = {beta:.4f}, p < 0.05). "
                        f"High-mortality countries are closing the gap. "
                        f"Estimated half-life: **{half_life:.1f} years**."
                    )
                elif beta < 0:
                    st.warning(f"β = {beta:.4f} suggests convergence but is not significant (p = {pval:.3f}).")
                else:
                    st.error(f"No convergence — β = {beta:.4f} (p = {pval:.3f}).")

                st.subheader("Scatter: Lagged Level vs Change")
                sdf = conv_d[["lag_log", "delta"]].dropna().copy()
                sdf.columns = ["log(mortality) t-1", "Δlog(mortality)"]
                st.scatter_chart(sdf.set_index("log(mortality) t-1"), use_container_width=True)

                with st.expander("Coefficient table"):
                    tbl = pd.DataFrame({
                        "Term":    ["Intercept", "log(mortality) t-1"],
                        "Coeff":   [f"{conv_r['coeffs'][0]:.4f}", f"{beta:.4f}"],
                        "Std Err": [f"{conv_r['se'][0]:.4f}",     f"{conv_r['se'][1]:.4f}"],
                        "t-stat":  [f"{conv_r['t_stats'][0]:.3f}", f"{conv_r['t_stats'][1]:.3f}"],
                        "p-value": [f"{conv_r['p_values'][0]:.4f}", f"{pval:.4f}"],
                    })
                    st.dataframe(tbl, use_container_width=True, hide_index=True)

            # ── KUZNETS PROXY ────────────────────────────────────────────────
            with econ_tabs[4]:
                st.subheader("Nonlinear Time Trend  (Kuznets proxy)")
                st.caption(
                    "Fits log(mortality) = α + β₁·t + β₂·t² where t = centred year. "
                    "Inverted-U (β₂ < 0) = mortality rose then fell. "
                    "Without GDP data, time proxies income growth."
                )
                with st.spinner("Fitting quadratic trend..."):
                    kuz_r, kuz_d = run_kuznets(fdf)

                b1 = float(kuz_r["coeffs"][1])
                b2 = float(kuz_r["coeffs"][2])
                p1 = float(kuz_r["p_values"][1])
                p2 = float(kuz_r["p_values"][2])

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("β₁ (linear)",    f"{b1:.6f}")
                c2.metric("β₂ (quadratic)", f"{b2:.6f}")
                c3.metric("R²",             f"{kuz_r['r2']:.4f}")
                c4.metric("N",              f"{kuz_r['n']:,}")

                if b2 < 0 and p2 < 0.05:
                    turning  = -b1 / (2 * b2)
                    mean_yr  = float(kuz_d["Year"].astype(float).mean())
                    turn_yr  = int(mean_yr + turning)
                    st.info(
                        f"**Inverted-U detected.** Mortality peaked around **{turn_yr}** "
                        f"and has been declining since — consistent with a Kuznets safety transition."
                    )
                elif b2 > 0 and p2 < 0.05:
                    st.warning("U-shaped curve detected — mortality declined then rose.")
                else:
                    st.info("No significant nonlinear trend. Change is approximately linear over time.")

                # Fitted curve
                t_grid    = np.linspace(float(kuz_d["t"].min()), float(kuz_d["t"].max()), 200)
                X_grid    = _add_const(np.column_stack([t_grid, t_grid**2]))
                y_hat     = X_grid @ kuz_r["coeffs"]
                mean_yr   = float(kuz_d["Year"].astype(float).mean())
                curve_df  = pd.DataFrame({"Year": t_grid + mean_yr, "Fitted log(mortality)": y_hat})
                st.line_chart(curve_df.set_index("Year"), use_container_width=True)

                with st.expander("Coefficient table"):
                    tbl = pd.DataFrame({
                        "Term":    ["Intercept", "t (linear)", "t² (quadratic)"],
                        "Coeff":   [f"{kuz_r['coeffs'][0]:.4f}", f"{b1:.6f}", f"{b2:.6f}"],
                        "Std Err": [f"{kuz_r['se'][0]:.4f}",     f"{kuz_r['se'][1]:.6f}", f"{kuz_r['se'][2]:.6f}"],
                        "t-stat":  [f"{kuz_r['t_stats'][0]:.3f}", f"{kuz_r['t_stats'][1]:.3f}", f"{kuz_r['t_stats'][2]:.3f}"],
                        "p-value": [f"{kuz_r['p_values'][0]:.4f}", f"{p1:.4f}", f"{p2:.4f}"],
                    })
                    st.dataframe(tbl, use_container_width=True, hide_index=True)


    # ASK AI
    with t5:
        st.title("Query")
        st.caption(f"Provider: **{provider}**")
        suggestions = [
            "Highest fatality rates by country",
            "Global trend over time",
            "Greatest improvement by country",
            "Policy implications",
            "Dataset structure",
        ]
        sc = st.columns(len(suggestions))
        for i, q in enumerate(suggestions):
            if sc[i].button(q, key=f"sq{i}", use_container_width=True):
                st.session_state["pending_q"] = q
        st.divider()

        question = st.chat_input("Enter a question about the data...")
        if "pending_q" in st.session_state:
            question = st.session_state.pop("pending_q")

        if question:
            ready = (provider == "Ollama (local)" and _ollama_alive()) or \
                    (provider == "Anthropic (cloud)" and bool(api_key))
            if not ready:
                st.error("AI unavailable. Verify Ollama is running or enter an API key.")
            else:
                with st.spinner("Processing..."):
                    result = ask_ai(fdf, filename, question, provider, model, api_key)
                st.session_state.chat.append({**result, "question": question})

        for item in reversed(st.session_state.chat):
            with st.chat_message("user"):      st.write(item["question"])
            with st.chat_message("assistant"):
                st.write(item["answer"])
                if item["sql"]:
                    with st.expander("SQL"): st.code(item["sql"], language="sql")
                if item["data"] is not None and not item["data"].empty:
                    with st.expander("Data"):     st.dataframe(item["data"], use_container_width=True)

        if st.session_state.chat and st.button("Clear"):
            st.session_state.chat = []
            st.rerun()

    # RAW DATA
    with t6:
        st.title("Data")
        st.caption(f"{len(fdf):,} rows × {len(fdf.columns)} columns — {filename}")
        st.caption(f"Columns: `{'`, `'.join(fdf.columns.tolist())}`")
        search = st.text_input("Search all columns")
        disp   = fdf.copy()
        if search:
            mask = disp.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
            disp = disp[mask]
            st.caption(f"{len(disp):,} matching rows")
        st.dataframe(disp, use_container_width=True, height=500)
        st.download_button("⬇️ Download CSV", fdf.to_csv(index=False).encode(),
                           f"trafficsight_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")