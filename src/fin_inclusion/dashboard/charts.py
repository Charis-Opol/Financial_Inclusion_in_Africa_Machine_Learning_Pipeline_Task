"""Interactive charts for the dashboard (Altair, rendered by Streamlit).

Same visual rules as the static report figures (`interpretability/plots.py`):
red pushes the score up, blue pushes it down, importance is a single blue,
and every chart has a hover tooltip carrying the exact values, so nothing is
readable from colour alone.
"""
from __future__ import annotations

import altair as alt
import pandas as pd

POSITIVE = "#e34948"
NEGATIVE = "#2a78d6"
MAGNITUDE = "#2a78d6"
SERIES = ["#2a78d6", "#eb6834"]
INK_MUTED = "#898781"


def contribution_chart(table: pd.DataFrame, title: str) -> alt.Chart:
    """One respondent's contribution per field, largest |contribution| at the top."""
    data = table.assign(direction=table["contribution"].map(lambda v: "raises the score" if v >= 0 else
                                                            "lowers the score"),
                        label=table["field"] + " = " + table["value"].astype(str))
    order = list(data["label"])
    return (
        alt.Chart(data, title=title)
        .mark_bar(cornerRadiusEnd=3, height=16)
        .encode(
            x=alt.X("contribution:Q", title="contribution (logits)"),
            y=alt.Y("label:N", sort=order, title=None),
            color=alt.Color("direction:N", scale=alt.Scale(domain=["raises the score", "lowers the score"],
                                                           range=[POSITIVE, NEGATIVE]), legend=alt.Legend(title=None)),
            tooltip=[alt.Tooltip("field:N"), alt.Tooltip("value:N"), alt.Tooltip("contribution:Q", format="+.3f")],
        )
        .properties(height=34 * len(data))
    )


def importance_chart(frame: pd.DataFrame, title: str) -> alt.Chart:
    """Mean |attribution| with 95% intervals, pooled and Uganda side by side."""
    order = list(frame[frame["sample"] == "pooled"].sort_values("mean_abs", ascending=False)["field"])
    base = alt.Chart(frame).encode(y=alt.Y("field:N", sort=order, title=None))
    bars = base.mark_bar(color=MAGNITUDE, height=14, cornerRadiusEnd=3).encode(
        x=alt.X("mean_abs:Q", title="mean |attribution| (logits)"),
        tooltip=[alt.Tooltip("field:N"), alt.Tooltip("mean_abs:Q", format=".3f"),
                 alt.Tooltip("ci_low:Q", title="95% CI low", format=".3f"),
                 alt.Tooltip("ci_high:Q", title="95% CI high", format=".3f"), alt.Tooltip("n:Q")],
    )
    whiskers = base.mark_rule(color="#0b0b0b").encode(x="ci_low:Q", x2="ci_high:Q")
    return (alt.layer(bars, whiskers, data=frame)
            .properties(width=280, height=30 * len(order))
            .facet(column=alt.Column("sample:N", title=None, sort=["pooled", "uganda"]), title=title))


def share_chart(frame: pd.DataFrame, title: str) -> alt.Chart:
    """Share of importance per field under each saved method; one small chart per method."""
    order = list(frame[frame["sample"] == "pooled"].groupby("field")["share"].mean().sort_values(ascending=False).index)
    return (
        alt.Chart(frame)
        .mark_bar(height=10, cornerRadiusEnd=2)
        .encode(
            x=alt.X("share:Q", title="share of importance", axis=alt.Axis(format="%")),
            y=alt.Y("field:N", sort=order, title=None),
            color=alt.Color("sample:N", scale=alt.Scale(domain=["pooled", "uganda"], range=SERIES),
                            legend=alt.Legend(title=None, orient="top")),
            yOffset=alt.YOffset("sample:N", sort=["pooled", "uganda"]),
            tooltip=[alt.Tooltip("method:N"), alt.Tooltip("sample:N"), alt.Tooltip("field:N"),
                     alt.Tooltip("share:Q", format=".1%")],
        )
        .properties(width=210, height=28 * len(order))
        .facet(facet=alt.Facet("method:N", title=None), columns=3, title=title)
    )


def what_if_chart(before: pd.DataFrame, after: pd.DataFrame, title: str) -> alt.Chart:
    """Each field's contribution before and after the change, joined by a line."""
    merged = before[["field", "contribution"]].merge(after[["field", "contribution"]], on="field",
                                                     suffixes=("_before", "_after"))
    order = list(merged.reindex(merged["contribution_before"].abs().sort_values(ascending=False).index)["field"])
    long = pd.concat([
        merged.assign(state="before", contribution=merged["contribution_before"]),
        merged.assign(state="after", contribution=merged["contribution_after"]),
    ])
    base = alt.Chart(long).encode(y=alt.Y("field:N", sort=order, title=None))
    lines = base.mark_line(color=INK_MUTED).encode(x="contribution:Q", detail="field:N")
    points = base.mark_circle(size=90).encode(
        x=alt.X("contribution:Q", title="contribution (logits)"),
        color=alt.Color("state:N", scale=alt.Scale(domain=["before", "after"], range=SERIES),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("field:N"), alt.Tooltip("state:N"), alt.Tooltip("contribution:Q", format="+.3f")],
    )
    return alt.layer(lines, points).properties(title=title, height=34 * len(order))


def method_heatmap(frame: pd.DataFrame, title: str) -> alt.Chart:
    """Fields x methods, each cell this respondent's contribution under that method."""
    long = frame.reset_index(names="field").melt(id_vars="field", var_name="method", value_name="contribution")
    limit = float(long["contribution"].abs().max()) or 1.0
    base = alt.Chart(long, title=title).encode(
        x=alt.X("method:N", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("field:N", sort=list(frame.index), title=None),
    )
    cells = base.mark_rect().encode(
        color=alt.Color("contribution:Q", title="logits",
                        scale=alt.Scale(domain=[-limit, 0, limit], range=[NEGATIVE, "#f0efec", POSITIVE])),
        tooltip=[alt.Tooltip("field:N"), alt.Tooltip("method:N"), alt.Tooltip("contribution:Q", format="+.3f")],
    )
    text = base.mark_text(fontSize=11).encode(text=alt.Text("contribution:Q", format="+.2f"))
    return alt.layer(cells, text).properties(height=32 * len(frame), width=150 * len(frame.columns))
