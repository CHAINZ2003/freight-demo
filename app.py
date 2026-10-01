"""
Freight Margin & Dispatch Intelligence — live demo on simulated fleet data.

All loads, carriers and shippers below are generated. No client data is used.
Structure mirrors a real TMS export (load ID, lane, carrier, rate, cost,
dwell, detention), so the same views can run on a prospect's own CSV.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Freight Margin & Dispatch Intelligence (Demo)",
    page_icon="🚛",
    layout="wide",
)

# ---------- design tokens ----------
ASPHALT = "#1B2530"
AMBER = "#C77700"
GREEN = "#2F7D4F"
RED = "#B3402F"
MUTED = "#6B7480"
SPOT_COLOR = AMBER
CONTRACT_COLOR = ASPHALT

TARGET_MARGIN = 0.12      # gross margin target per load
DETENTION_PAID_RATE = 60  # $/hr paid to carriers after 2 free hours
DETENTION_BILL_RATE = 75  # $/hr billed to shippers when billed

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 2rem; }}
      .finding {{ border-left: 4px solid {AMBER}; padding: .6rem 1rem; margin: .6rem 0;
                  background: rgba(199,119,0,.07); border-radius: 0 6px 6px 0; }}
      .finding b {{ color: {ASPHALT}; }}
      .banner {{ padding: .5rem .9rem; border-radius: 6px; background: {ASPHALT};
                 color: #F7F6F3; font-size: .9rem; margin-bottom: 1rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- simulated fleet data ----------
LANES = [
    ("Dallas, TX", "Houston, TX", 240, 14),
    ("Dallas, TX", "Atlanta, GA", 780, 10),
    ("Houston, TX", "New Orleans, LA", 350, 9),
    ("Fort Worth, TX", "Chicago, IL", 925, 8),
    ("Dallas, TX", "Memphis, TN", 450, 9),
    ("Houston, TX", "San Antonio, TX", 200, 12),
    ("Dallas, TX", "Oklahoma City, OK", 205, 10),
    ("Laredo, TX", "Dallas, TX", 430, 11),
    ("Dallas, TX", "Denver, CO", 800, 6),
    ("Houston, TX", "Atlanta, GA", 790, 7),
]

# name: (fleet type, on-time rate, cost multiplier)
CARRIERS = {
    "Lone Star Haulers": ("Dedicated fleet", 0.96, 1.00),
    "Bluebonnet Freight": ("Dedicated fleet", 0.95, 1.01),
    "Red River Carriers": ("Owner-operator", 0.93, 0.99),
    "Gulf Coast Transport": ("Dedicated fleet", 0.94, 1.00),
    "Panhandle Express": ("Owner-operator", 0.91, 1.06),
    "Mesquite Motor Lines": ("Owner-operator", 0.78, 1.00),
    "Brazos Trucking": ("Dedicated fleet", 0.95, 0.98),
    "Alamo Cartage": ("Owner-operator", 0.92, 1.00),
    "Piney Woods Logistics": ("Dedicated fleet", 0.94, 1.02),
    "Rio Grande Hauling": ("Owner-operator", 0.88, 0.99),
    "Cap Rock Carriers": ("Dedicated fleet", 0.96, 1.00),
    "Caddo Lake Carriers": ("Owner-operator", 0.93, 1.01),
}

# shipper: extra average dock hours (planted detention problems)
SHIPPERS = {
    "Prairie Foods Co.": 0.2,
    "Summit Building Supply": 0.3,
    "Orbit Electronics": 0.1,
    "Ridgeline Chemical": 3.2,
    "Harbor Paper & Pack": 2.6,
    "Keystone Auto Parts": 0.4,
    "Sunbelt Beverage": 0.5,
    "Delta Textile Mills": 0.3,
}

# planted margin stories: (lane label, load type) -> mean margin
MARGIN_OVERRIDES = {
    ("Fort Worth → Chicago", "Spot"): 0.04,
    ("Dallas → Atlanta", "Spot"): 0.05,
    ("Houston → New Orleans", "Contract"): 0.09,
}


def lane_label(o, d):
    return f"{o.split(',')[0]} → {d.split(',')[0]}"


@st.cache_data(show_spinner=False)
def make_loads(end_date: str, n: int = 1400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(end_date)

    lane_idx = rng.choice(len(LANES), size=n, p=np.array([l[3] for l in LANES]) / sum(l[3] for l in LANES))
    carriers = list(CARRIERS)
    shippers = list(SHIPPERS)

    rows = []
    for i in range(n):
        o, d, base_miles, _ = LANES[lane_idx[i]]
        lane = lane_label(o, d)
        miles = round(base_miles * rng.uniform(0.98, 1.04))
        load_type = rng.choice(["Contract", "Spot"], p=[0.58, 0.42])
        mode = rng.choice(["Dry Van", "Reefer", "Flatbed"], p=[0.62, 0.23, 0.15])

        rpm = 3.35 if miles < 300 else 2.75 if miles < 600 else 2.40
        rpm *= {"Dry Van": 1.0, "Reefer": 1.12, "Flatbed": 1.08}[mode]
        rpm *= rng.normal(1.0, 0.10 if load_type == "Spot" else 0.03)
        revenue = miles * rpm

        mean_margin = MARGIN_OVERRIDES.get((lane, load_type), 0.19 if load_type == "Contract" else 0.15)
        margin = rng.normal(mean_margin, 0.045)

        carrier = carriers[rng.integers(len(carriers))]
        c_type, c_otd, c_mult = CARRIERS[carrier]
        carrier_cost = revenue * (1 - margin) * c_mult

        shipper = shippers[rng.integers(len(shippers))]
        dwell = rng.gamma(4, 0.6) + rng.gamma(2, SHIPPERS[shipper] / 2 + 0.01)
        detention_hours = round(max(0.0, dwell - 2) * 4) / 4
        det_paid = detention_hours * DETENTION_PAID_RATE
        det_billed_flag = bool(rng.random() < 0.35) and detention_hours > 0
        det_billed = detention_hours * DETENTION_BILL_RATE if det_billed_flag else 0.0

        pickup_late = bool(rng.random() < (0.10 + (0.05 if detention_hours > 2 else 0)))
        late_p = (1 - c_otd) + (0.04 if detention_hours > 2 else 0)
        delivery_late = bool(rng.random() < late_p)
        late_hours = round(rng.exponential(5), 1) if delivery_late else 0.0

        pickup = end - pd.Timedelta(days=int(rng.integers(0, 90)), hours=int(rng.integers(0, 12)))

        rows.append(
            dict(
                pickup_date=pickup.normalize(),
                lane=lane,
                origin=o,
                destination=d,
                miles=miles,
                mode=mode,
                load_type=load_type,
                carrier=carrier,
                carrier_type=c_type,
                shipper=shipper,
                revenue=round(revenue, 2),
                carrier_cost=round(carrier_cost, 2),
                dwell_hours=round(dwell, 2),
                detention_hours=detention_hours,
                detention_paid=round(det_paid, 2),
                detention_billed=round(det_billed, 2),
                detention_billed_flag=det_billed_flag,
                pickup_late=pickup_late,
                delivery_late=delivery_late,
                late_hours=late_hours,
            )
        )

    df = pd.DataFrame(rows).sort_values("pickup_date").reset_index(drop=True)
    df.insert(0, "load_id", [f"LD-{100200 + i}" for i in range(len(df))])
    df["rev_total"] = df["revenue"] + df["detention_billed"]
    df["gm"] = df["rev_total"] - df["carrier_cost"] - df["detention_paid"]
    df["margin_pct"] = df["gm"] / df["rev_total"]
    df["cost_per_mile"] = (df["carrier_cost"] + df["detention_paid"]) / df["miles"]
    df["on_time"] = ~df["delivery_late"]
    df["unbilled_detention"] = np.where(df["detention_billed_flag"], 0.0, df["detention_paid"])
    return df


def delay_reason(r):
    if r["detention_hours"] > 2:
        return "Dock delay at origin"
    if r["pickup_late"]:
        return "Late pickup"
    return "Carrier transit delay"


# ---------- filters ----------
today = pd.Timestamp.today().normalize()
data = make_loads(str(today.date()))

st.sidebar.header("Filters")
min_d, max_d = data["pickup_date"].min().date(), data["pickup_date"].max().date()
date_range = st.sidebar.date_input("Pickup date", (min_d, max_d), min_value=min_d, max_value=max_d)
modes = st.sidebar.multiselect("Equipment", sorted(data["mode"].unique()), default=sorted(data["mode"].unique()))
types = st.sidebar.multiselect("Load type", ["Contract", "Spot"], default=["Contract", "Spot"])
st.sidebar.caption("Margin target per load: 12%")

if isinstance(date_range, tuple) and len(date_range) == 2:
    start, stop = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start, stop = pd.Timestamp(min_d), pd.Timestamp(max_d)

df = data[
    data["pickup_date"].between(start, stop) & data["mode"].isin(modes) & data["load_type"].isin(types)
].copy()

# ---------- header ----------
st.title("Freight Margin & Dispatch Intelligence")
st.markdown(
    '<div class="banner">Live demo on an anonymized fleet simulation (Dallas–Houston, Atlanta, Chicago and other lanes). '
    "No client data. Built to run on a TMS load export.</div>",
    unsafe_allow_html=True,
)

if df.empty:
    st.warning("No loads match these filters. Widen the date range or equipment selection.")
    st.stop()

# ---------- KPIs ----------
rev = df["rev_total"].sum()
gm = df["gm"].sum()
below = df[df["margin_pct"] < TARGET_MARGIN]
leak = (TARGET_MARGIN * below["rev_total"] - below["gm"]).sum()

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Loads", f"{len(df):,}")
k2.metric("Revenue", f"${rev/1000:,.0f}K")
k3.metric("Gross margin", f"{gm/rev:.1%}", f"{(gm/rev - TARGET_MARGIN)*100:+.1f} pts vs target")
k4.metric("On-time delivery", f"{df['on_time'].mean():.1%}")
k5.metric("Margin leak", f"${leak/1000:,.0f}K", help="Gap to the 12% target on loads that fell below it.")
k6.metric("Unbilled detention", f"${df['unbilled_detention'].sum()/1000:,.1f}K", help="Detention paid to carriers that was never billed to the shipper.")


def style(fig, h=380):
    fig.update_layout(
        template="plotly_white",
        height=h,
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(color=ASPHALT),
        legend=dict(orientation="h", y=1.12, x=0),
    )
    return fig


tab_over, tab_lane, tab_car, tab_det, tab_find = st.tabs(
    ["Overview", "Lane margins", "Carrier scorecard", "Detention & exceptions", "Findings"]
)

# ---------- overview ----------
with tab_over:
    wk = df.copy()
    wk["week"] = wk["pickup_date"].dt.to_period("W").dt.start_time
    weekly = wk.groupby("week").agg(rev=("rev_total", "sum"), gm=("gm", "sum"), loads=("load_id", "count"), otd=("on_time", "mean")).reset_index()
    weekly["margin_pct"] = weekly["gm"] / weekly["rev"]

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=weekly["week"], y=weekly["margin_pct"], mode="lines+markers", line=dict(color=ASPHALT, width=3), name="Gross margin"))
        fig.add_hline(y=TARGET_MARGIN, line_dash="dash", line_color=AMBER, annotation_text="12% target", annotation_position="top left")
        fig.update_yaxes(tickformat=".0%", title="Gross margin")
        fig.update_layout(title="Weekly gross margin")
        st.plotly_chart(style(fig), width="stretch")
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=weekly["week"], y=weekly["loads"], marker_color="#C9CDD2", name="Loads"))
        fig.add_trace(go.Scatter(x=weekly["week"], y=weekly["otd"], yaxis="y2", mode="lines+markers", line=dict(color=GREEN, width=3), name="On-time %"))
        fig.update_layout(
            title="Weekly loads and on-time delivery",
            yaxis=dict(title="Loads"),
            yaxis2=dict(title="On-time", overlaying="y", side="right", tickformat=".0%", range=[0.7, 1.0], showgrid=False),
        )
        st.plotly_chart(style(fig), width="stretch")

    by_type = df.groupby("load_type").agg(rev=("rev_total", "sum"), gm=("gm", "sum"), loads=("load_id", "count")).reset_index()
    by_type["margin_pct"] = by_type["gm"] / by_type["rev"]
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Spot vs contract")
        for _, r in by_type.iterrows():
            st.metric(f"{r['load_type']} margin", f"{r['margin_pct']:.1%}", f"{int(r['loads'])} loads", delta_color="off")
    with c2:
        mix = df.groupby(["mode", "load_type"]).agg(rev=("rev_total", "sum"), gm=("gm", "sum")).reset_index()
        mix["margin_pct"] = mix["gm"] / mix["rev"]
        fig = px.bar(mix, x="mode", y="margin_pct", color="load_type", barmode="group",
                     color_discrete_map={"Spot": SPOT_COLOR, "Contract": CONTRACT_COLOR},
                     labels={"margin_pct": "Gross margin", "mode": "Equipment", "load_type": ""})
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(title="Margin by equipment and load type")
        st.plotly_chart(style(fig, 300), width="stretch")

# ---------- lane margins ----------
with tab_lane:
    lt = df.groupby(["lane", "load_type"]).agg(
        loads=("load_id", "count"), miles=("miles", "sum"), rev=("rev_total", "sum"), gm=("gm", "sum")
    ).reset_index()
    lt["margin_pct"] = lt["gm"] / lt["rev"]
    lt["margin_per_mile"] = lt["gm"] / lt["miles"]
    order = lt.groupby("lane")["gm"].sum().sort_values().index.tolist()

    fig = px.bar(lt, y="lane", x="margin_pct", color="load_type", barmode="group", orientation="h",
                 category_orders={"lane": order},
                 color_discrete_map={"Spot": SPOT_COLOR, "Contract": CONTRACT_COLOR},
                 labels={"margin_pct": "Gross margin", "lane": "", "load_type": ""})
    fig.add_vline(x=TARGET_MARGIN, line_dash="dash", line_color=RED, annotation_text="12% target")
    fig.update_xaxes(tickformat=".0%")
    fig.update_layout(title="Gross margin by lane, spot vs contract")
    st.plotly_chart(style(fig, 560), width="stretch")

    show = lt.sort_values("margin_pct")[["lane", "load_type", "loads", "rev", "gm", "margin_pct", "margin_per_mile"]]
    st.dataframe(
        show,
        hide_index=True,
        width="stretch",
        column_config={
            "lane": "Lane", "load_type": "Type", "loads": "Loads",
            "rev": st.column_config.NumberColumn("Revenue", format="$%.0f"),
            "gm": st.column_config.NumberColumn("Gross margin", format="$%.0f"),
            "margin_pct": st.column_config.NumberColumn("Margin %", format="percent"),
            "margin_per_mile": st.column_config.NumberColumn("Margin / mile", format="$%.2f"),
        },
    )

# ---------- carrier scorecard ----------
with tab_car:
    cs = df.groupby(["carrier", "carrier_type"]).agg(
        loads=("load_id", "count"),
        otd=("on_time", "mean"),
        cost_mile=("cost_per_mile", "mean"),
        avg_det=("detention_hours", "mean"),
        margin=("gm", "sum"),
        rev=("rev_total", "sum"),
    ).reset_index()
    cs["margin_pct"] = cs["margin"] / cs["rev"]

    fig = px.scatter(cs, x="otd", y="cost_mile", size="loads", color="carrier_type", hover_name="carrier",
                     color_discrete_map={"Dedicated fleet": ASPHALT, "Owner-operator": AMBER},
                     labels={"otd": "On-time delivery", "cost_mile": "Cost per mile", "carrier_type": ""})
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickprefix="$")
    fig.update_layout(title="Who is cheap and reliable, and who is neither")
    st.plotly_chart(style(fig, 440), width="stretch")

    st.dataframe(
        cs.sort_values("otd")[["carrier", "carrier_type", "loads", "otd", "cost_mile", "avg_det", "margin_pct"]],
        hide_index=True,
        width="stretch",
        column_config={
            "carrier": "Carrier", "carrier_type": "Type", "loads": "Loads",
            "otd": st.column_config.NumberColumn("On-time %", format="percent"),
            "cost_mile": st.column_config.NumberColumn("Cost / mile", format="$%.2f"),
            "avg_det": st.column_config.NumberColumn("Avg detention (hrs)", format="%.2f"),
            "margin_pct": st.column_config.NumberColumn("Margin % on their loads", format="percent"),
        },
    )

# ---------- detention & exceptions ----------
with tab_det:
    sh = df.groupby("shipper").agg(
        loads=("load_id", "count"),
        avg_dwell=("dwell_hours", "mean"),
        paid=("detention_paid", "sum"),
        billed=("detention_billed", "sum"),
        unbilled=("unbilled_detention", "sum"),
    ).reset_index().sort_values("unbilled", ascending=False)

    c1, c2 = st.columns([3, 2])
    with c1:
        fig = px.bar(sh, x="shipper", y="unbilled", labels={"unbilled": "Unbilled detention ($)", "shipper": ""},
                     color_discrete_sequence=[RED])
        fig.update_layout(title="Detention paid to carriers but never billed, by shipper")
        st.plotly_chart(style(fig, 360), width="stretch")
    with c2:
        st.dataframe(
            sh[["shipper", "avg_dwell", "paid", "billed"]],
            hide_index=True,
            width="stretch",
            column_config={
                "shipper": "Shipper",
                "avg_dwell": st.column_config.NumberColumn("Avg dock time (hrs)", format="%.1f"),
                "paid": st.column_config.NumberColumn("Paid out", format="$%.0f"),
                "billed": st.column_config.NumberColumn("Billed", format="$%.0f"),
            },
        )

    st.subheader("Most recent delivery exceptions")
    late = df[df["delivery_late"]].sort_values("pickup_date", ascending=False).head(15).copy()
    late["likely_cause"] = late.apply(delay_reason, axis=1)
    st.dataframe(
        late[["load_id", "pickup_date", "lane", "carrier", "shipper", "late_hours", "likely_cause"]],
        hide_index=True,
        width="stretch",
        column_config={
            "load_id": "Load", "pickup_date": st.column_config.DateColumn("Pickup"), "lane": "Lane",
            "carrier": "Carrier", "shipper": "Shipper",
            "late_hours": st.column_config.NumberColumn("Hours late", format="%.1f"),
            "likely_cause": "Likely cause",
        },
    )

# ---------- findings ----------
with tab_find:
    st.subheader("What this data says")
    items = []

    lt = df.groupby(["lane", "load_type"]).agg(loads=("load_id", "count"), rev=("rev_total", "sum"), gm=("gm", "sum")).reset_index()
    lt["m"] = lt["gm"] / lt["rev"]
    lt = lt[lt["loads"] >= 10]
    if len(lt):
        w = lt.sort_values("m").iloc[0]
        sub = df[(df["lane"] == w["lane"]) & (df["load_type"] == w["load_type"])]
        gap = TARGET_MARGIN * sub["rev_total"].sum() - sub["gm"].sum()
        items.append((
            f"{w['lane']} ({w['load_type'].lower()}) is the thinnest lane",
            f"{int(w['loads'])} loads earned {w['m']:.1%} against a {TARGET_MARGIN:.0%} target. That is about ${gap:,.0f} short over this period.",
        ))

    st_df = df.groupby("shipper")["unbilled_detention"].sum().sort_values(ascending=False)
    if st_df.sum() > 0:
        items.append((
            f"${st_df.sum():,.0f} of detention was paid out and never billed",
            f"{st_df.index[0]} alone accounts for ${st_df.iloc[0]:,.0f}. Billing detention on that shipper would recover it with no change to rates.",
        ))

    cr = df.groupby("carrier").agg(loads=("load_id", "count"), otd=("on_time", "mean"), late=("late_hours", "sum"), n_late=("delivery_late", "sum")).reset_index()
    cr = cr[cr["loads"] >= 30]
    if len(cr):
        c = cr.sort_values("otd").iloc[0]
        items.append((
            f"{c['carrier']} is the weakest carrier on reliability",
            f"{c['otd']:.1%} on-time across {int(c['loads'])} loads versus {df['on_time'].mean():.1%} for the whole book.",
        ))

    tm = df.groupby("load_type").apply(lambda g: g["gm"].sum() / g["rev_total"].sum(), include_groups=False)
    if {"Spot", "Contract"} <= set(tm.index):
        items.append((
            "Spot freight is earning less than contract",
            f"Spot margin is {tm['Spot']:.1%} versus {tm['Contract']:.1%} on contract, a gap of {(tm['Contract'] - tm['Spot'])*100:.1f} points.",
        ))

    for title, body in items:
        st.markdown(f'<div class="finding"><b>{title}</b><br>{body}</div>', unsafe_allow_html=True)

    st.download_button(
        "Download load data (CSV)",
        df.drop(columns=["rev_total", "detention_billed_flag"]).to_csv(index=False).encode(),
        file_name="simulated_loads.csv",
        mime="text/csv",
    )

st.caption("Built by Mubarak · Simulated data for demonstration · Works on a TMS export (CSV)")
