from typing import Optional
import pandas as pd
from streamlit_echarts import st_echarts

# Mapping of technical feature names to user-friendly labels
FEATURE_LABELS = {
    "text_frac_sensitive": "Sensitive Words Ratio",
    "text_mean_prob": "Average Risk of Text",
    "username_len": "Username Length",
    "username_digit_frac": "Digits in Username Ratio",
    "username_seg_count": "Username Segment Count",
    "username_has_year": "Username Contains Year",
    "username_has_date": "Username Contains Date",
    "total_activity": "Total Activity",
    "comment_post_ratio": "Comment/Post Ratio",
    "posts": "Number of Posts",
    "comments": "Number of Comments",
    "log_total_activity": "Log Total Activity",
    "log_comments": "Log Comments",
    "log_posts": "Log Posts",
    "post_comment_ratio": "Post/Comment Ratio",
}

def feature_pie3d(df_fi: Optional[pd.DataFrame], key: str = "pie"):
    """3D donut chart showing relative feature contribution"""
    if df_fi is None or df_fi.empty:
        return
    df = df_fi.copy()

    # Select importance-related column
    value_col = next((c for c in ["importance", "gain", "weight", "score", "value"] if c in df.columns), None)
    if not value_col:
        return

    df = df[["feature", value_col]].dropna().head(10)
    df[value_col] = df[value_col].abs()

    data = []
    for _, r in df.iterrows():
        name_raw = str(r["feature"])
        name = FEATURE_LABELS.get(name_raw, name_raw.replace("_", " ").title())
        data.append({"name": name[:24], "value": float(r[value_col])})

    option = {
        "tooltip": {"trigger": "item", "formatter": "{b}<br/>{d}%"},
        "series": [{
            "type": "pie",
            "radius": ["46%", "72%"],
            "avoidLabelOverlap": True,
            "label": {"show": True, "formatter": "{b}\n{d}%"},
            "labelLine": {"length": 18, "length2": 10},
            "emphasis": {"scale": True, "scaleSize": 6},
            "itemStyle": {
                "shadowBlur": 16,
                "shadowOffsetX": 0,
                "shadowColor": "rgba(0,0,0,0.15)"
            },
            "data": data,
        }],
        "color": [
            "#5B8FF9","#61DDAA","#65789B","#F6BD16","#7262fd",
            "#78D3F8","#F6903D","#9661BC","#F08BB4","#C2C8D5"
        ],
    }
    st_echarts(option, height="360px", key=key)