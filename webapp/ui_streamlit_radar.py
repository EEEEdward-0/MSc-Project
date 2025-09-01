from __future__ import annotations
from typing import Dict, List
from streamlit_echarts import st_echarts

def radar_chart(factors: Dict[str, float], labels: List[str], animate: bool = True):
    """Render a compact radar chart for factor scores (0–100)."""
    # Preserve incoming dict order
    keys = list(factors.keys())
    values = [float(factors[k]) for k in keys]
    indicator = [{"name": labels[i], "max": 100} for i in range(len(keys))]

    option = {
        "animation": bool(animate),
        "animationDuration": 300,
        "animationEasing": "quartOut",
        "tooltip": {"show": False},
        "radar": {
            "indicator": indicator,
            "radius": "72%",
            "startAngle": 90,
            "name": {"fontSize": 12, "color": "#334155"},
            "splitNumber": 5,
            "splitLine": {"lineStyle": {"color": "rgba(51,65,85,.25)"}},
            "splitArea": {"areaStyle": {"color": ["rgba(255,255,255,.0)"]}},
            "axisLine": {"lineStyle": {"color": "rgba(51,65,85,.35)"}},
        },
        "series": [{
            "type": "radar",
            "data": [{"value": values}],
            "symbol": "circle",
            "symbolSize": 4,
            "lineStyle": {"width": 2, "color": "rgba(37,99,235,1)"},
            "areaStyle": {"opacity": 0.15, "color": "rgba(37,99,235,1)"},
            "itemStyle": {"color": "rgba(37,99,235,1)", "borderWidth": 0},
        }],
    }

    # Use a stable key so updates replace the series in place
    st_echarts(options=option, height="320px", key="pa_radar")