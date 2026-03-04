"""
TrafficSight — AI Traffic Death Data Analyser
Run: python -m streamlit run app.py
"""
import streamlit as st
import pandas as pd
import duckdb, httpx, re
from datetime import datetime

st.set_page_config(page_title="TrafficSight", page_icon="🚦", layout="wide")

for key, val in [("df", None), ("filename", None), ("chat", [])]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Load & reshape World Bank wide CSV into long format ───────────────────────
def load_and_reshape(uploaded_file):
    raw = pd.read_csv(uploaded_file, skiprows=4, encoding_errors="replace")
    raw = raw.dropna(axis=1, how="all").dropna(axis=0, how="all")

    # ID columns (always present in World Bank format)
    id_cols = [c for c in ["Country Name", "Country Code", "Indicator Name", "Indicator Code"] if c in raw.columns]

    # Year columns = any column whose name is a 4-digit number
    year_cols = [c for c in raw.columns if str(c).strip().isdigit() and 1900 < int(c) < 2100]

    if year_cols:
        # Wide → long format
        df = raw.melt(id_vars=id_cols, value_vars=year_cols, var_name="Year", value_name="Value")
        df["Year"]  = pd.to_numeric(df["Year"],  errors="coerce")
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        df = df.dropna(subset=["Value"])
        df = df.sort_values(["Country Name", "Year"]).reset_index(drop=True)
    else:
        df = raw  # Already long format

    return df

# ── SQL helper ────────────────────────────────────────────────────────────────
def run_sql(df, sql):
    if not sql.strip().upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries allowed.")
    con = duckdb.connect()
    con.register("dataset", df)
    return con.execute(sql.strip()).fetchdf()

# ── AI helpers ────────────────────────────────────────────────────────────────
def build_context(df, filename):
    stats = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            stats.append(f"  {col} (number): min={s.min():.1f}, max={s.max():.1f}, mean={s.mean():.1f}")
        else:
            stats.append(f"  {col} (text): {s.nunique()} unique values")
    return f"""You are a traffic safety analyst. Dataset: {filename} — {df.shape[0]} rows.
Columns: {', '.join(df.columns.tolist())}
Stats:\n{chr(10).join(stats)}
Sample:\n{df.head(6).to_csv(index=False)}
The data shows mortality caused by road traffic injury per 100,000 population by country and year.
If you need to calculate something, write SQL inside <sql>...</sql> tags using table name `dataset`.
Column names are: Country Name, Country Code, Indicator Name, Indicator Code, Year, Value.
Keep answers under 200 words. Be practical and policy-focused."""

def ask_ollama(ctx, q, model):
    try:
        r = httpx.post("http://localhost:11434/api/generate",
            json={"model": model, "prompt": f"{ctx}\n\nUser: {q}", "stream": False}, timeout=60)
        r.raise_for_status()
        return r.json()["response"]
    except httpx.ConnectError:
        return "❌ Ollama not running. Run: `ollama serve` then `ollama pull llama3.1`"
    except Exception as e:
        return f"❌ Error: {e}"

def ask_anthropic(ctx, q, key):
    try:
        r = httpx.post("https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": "claude-sonnet-4-20250514", "max_tokens": 1024,
                  "system": ctx, "messages": [{"role": "user", "content": q}]}, timeout=60)
        r.raise_for_status()
        return r.json()["content"][0]["text"]
    except Exception as e:
        return f"❌ Anthropic error: {e}"

def ask_ai(df, filename, question, provider, model, key):
    ctx = build_context(df, filename)
    raw = ask_ollama(ctx, question, model) if provider == "Ollama (local)" else ask_anthropic(ctx, question, key)
    sql_m = re.search(r"<sql>(.*?)</sql>", raw, re.DOTALL | re.IGNORECASE)
    sql = sql_m.group(1).strip() if sql_m else None
    data = None
    if sql:
        try:
            data = run_sql(df, sql)
            followup = f"Results:\n{data.head(20).to_csv(index=False)}\nSummarise clearly."
            raw = ask_ollama(ctx, followup, model) if provider == "Ollama (local)" else ask_anthropic(ctx, followup, key)
        except Exception as e:
            raw += f"\n*(SQL failed: {e})*"
    return {"answer": re.sub(r"<sql>.*?</sql>", "", raw, flags=re.DOTALL).strip(), "sql": sql, "data": data}

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# 🚦 TrafficSight")
    st.divider()
    st.subheader("📂 Load Data")
    up = st.file_uploader("CSV or Excel", type=["csv","xlsx","xls"])
    if up:
        try:
            df_new = load_and_reshape(up)
            st.session_state.df = df_new
            st.session_state.filename = up.name
            st.session_state.chat = []
            st.success(f"✅ {len(df_new):,} rows loaded")
        except Exception as e:
            st.error(f"Error: {e}")

    fdf = st.session_state.df

    if fdf is not None:
        st.divider()
        st.subheader("🔍 Filters")

        all_countries = sorted(fdf["Country Name"].dropna().unique())
        sel_countries = st.multiselect("Countries", all_countries, default=all_countries[:12])

        all_years = sorted(fdf["Year"].dropna().unique().astype(int))
        year_range = st.select_slider("Year range", options=all_years, value=(all_years[0], all_years[-1]))

        fdf = fdf.copy()
        if sel_countries:
            fdf = fdf[fdf["Country Name"].isin(sel_countries)]
        fdf = fdf[(fdf["Year"] >= year_range[0]) & (fdf["Year"] <= year_range[1])]
        st.caption(f"{len(fdf):,} / {len(st.session_state.df):,} rows")

    st.divider()
    st.subheader("🤖 AI Settings")
    provider = st.selectbox("Provider", ["Ollama (local)", "Anthropic (cloud)"])
    model    = st.text_input("Model", "llama3.1") if provider == "Ollama (local)" else "claude-sonnet-4-20250514"
    api_key  = "" if provider == "Ollama (local)" else st.text_input("API key", type="password")
    if provider == "Ollama (local)":
        st.info("Start Ollama:\n`ollama serve`\n`ollama pull llama3.1`")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.df is None:
    st.title("🚦 TrafficSight")
    st.info("👈 Upload your World Bank traffic CSV using the sidebar.")
    c1, c2, c3 = st.columns(3)
    c1.success("**📊 Visualise**\nCharts across countries & years")
    c2.success("**📈 Trends**\nSee how rates change over time")
    c3.success("**🤖 Ask AI**\nPlain-English questions about your data")

else:
    df       = st.session_state.df
    filename = st.session_state.filename

    t1, t2, t3, t4, t5 = st.tabs(["📊 Overview","📈 Trends","🌍 Compare","🤖 Ask AI","📋 Raw Data"])

    # ── OVERVIEW ──────────────────────────────────────────────────────────────
    with t1:
        st.title("📊 Overview")
        st.caption("Mortality caused by road traffic injury (per 100,000 population)")

        a, b, c, d = st.columns(4)
        a.metric("Countries",   f"{fdf['Country Name'].nunique():,}")
        b.metric("Average rate", f"{fdf['Value'].mean():.1f}")
        c.metric("Highest",     f"{fdf['Value'].max():.1f}")
        d.metric("Lowest",      f"{fdf['Value'].min():.1f}")
        st.divider()

        latest_year = int(fdf["Year"].max())
        latest = fdf[fdf["Year"] == latest_year].copy()
        latest = latest.sort_values("Value", ascending=False).head(20)

        st.subheader(f"Top 20 countries — {latest_year}")
        st.bar_chart(latest.set_index("Country Name")["Value"])

    # ── TRENDS ────────────────────────────────────────────────────────────────
    with t2:
        st.title("📈 Trends Over Time")
        all_c = sorted(fdf["Country Name"].dropna().unique())
        chosen = st.multiselect("Select countries", all_c, default=all_c[:6])
        if chosen:
            tdf = fdf[fdf["Country Name"].isin(chosen)].copy()
            pivot = tdf.pivot_table(index="Year", columns="Country Name", values="Value", aggfunc="mean")
            st.line_chart(pivot)

            st.subheader("Change over period")
            rows = []
            for country in chosen:
                cdf = tdf[tdf["Country Name"] == country].sort_values("Year")
                if len(cdf) < 2: continue
                fv, lv = cdf.iloc[0]["Value"], cdf.iloc[-1]["Value"]
                if pd.isna(fv) or pd.isna(lv) or fv == 0: continue
                pct = (lv - fv) / fv * 100
                rows.append({
                    "Country": country,
                    f"First ({int(cdf.iloc[0]['Year'])})":  f"{fv:.1f}",
                    f"Latest ({int(cdf.iloc[-1]['Year'])})": f"{lv:.1f}",
                    "Change": f"{'▼' if pct < 0 else '▲'} {abs(pct):.1f}%",
                    "Status": "✅ Improved" if pct < 0 else "⚠️ Worsened",
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("Select at least one country above.")

    # ── COMPARE ───────────────────────────────────────────────────────────────
    with t3:
        st.title("🌍 Country Comparison")
        latest_year = int(fdf["Year"].max())
        cmp = fdf[fdf["Year"] == latest_year].copy()
        cmp = cmp.sort_values("Value", ascending=False)[["Country Name", "Country Code", "Value"]]
        st.caption(f"Latest year in selection: **{latest_year}**  |  Rate = deaths per 100,000 population")
        st.dataframe(
            cmp.style.bar(subset=["Value"], color="#2E75B6"),
            use_container_width=True, hide_index=True
        )
        st.caption("🔴 High risk  🟡 Moderate  🟢 Lower risk")

        st.subheader("🔎 Custom SQL query")
        st.caption("Table is called `dataset`. Columns: `Country Name`, `Country Code`, `Year`, `Value`")
        sql_in = st.text_area("SQL", 'SELECT "Country Name", AVG("Value") as avg_rate FROM dataset GROUP BY "Country Name" ORDER BY avg_rate DESC LIMIT 20', height=80)
        if st.button("▶ Run"):
            try:
                st.dataframe(run_sql(fdf, sql_in), use_container_width=True)
            except Exception as e:
                st.error(str(e))

    # ── ASK AI ────────────────────────────────────────────────────────────────
    with t4:
        st.title("🤖 Ask AI")
        st.caption(f"Using **{provider}**")
        suggestions = [
            "Which countries have the highest fatality rates?",
            "What is the global trend over time?",
            "Which countries improved the most?",
            "What does this data suggest for road safety policy?",
        ]
        cols = st.columns(len(suggestions))
        for i, q in enumerate(suggestions):
            if cols[i].button(q, key=f"sq{i}", use_container_width=True):
                st.session_state["pq"] = q
        st.divider()

        question = st.chat_input("Ask anything about your traffic data...")
        if "pq" in st.session_state:
            question = st.session_state.pop("pq")

        if question:
            if provider == "Anthropic (cloud)" and not api_key:
                st.error("Enter your Anthropic API key in the sidebar.")
            else:
                with st.spinner("Thinking..."):
                    res = ask_ai(fdf, filename, question, provider, model, api_key)
                st.session_state.chat.append({**res, "question": question})

        for item in reversed(st.session_state.chat):
            with st.chat_message("user"):
                st.write(item["question"])
            with st.chat_message("assistant"):
                st.write(item["answer"])
                if item["sql"]:
                    with st.expander("SQL used"): st.code(item["sql"], language="sql")
                if item["data"] is not None and not item["data"].empty:
                    with st.expander("Data"): st.dataframe(item["data"], use_container_width=True)

        if st.session_state.chat and st.button("🗑 Clear chat"):
            st.session_state.chat = []
            st.rerun()

    # ── RAW DATA ──────────────────────────────────────────────────────────────
    with t5:
        st.title("📋 Raw Data")
        st.caption(f"{len(fdf):,} rows — {filename}")
        search = st.text_input("Search all columns")
        disp = fdf.copy()
        if search:
            mask = disp.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
            disp = disp[mask]
            st.caption(f"{len(disp):,} matching rows")
        st.dataframe(disp, use_container_width=True, height=500)
        st.download_button(
            "⬇️ Download CSV",
            fdf.to_csv(index=False).encode(),
            f"trafficsight_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )
