"""
Streamlit front-end for the Reddit Privacy Audit demo.

This module focuses on presentational logic (layout, i18n, light data
loading). Heavy computation and model training live outside of this file.
Code and comments are written with an undergraduate CS audience in mind:
- functions are short and single-purpose
- IO helpers are defensive (return None on failure)
- UI helpers separate structure from content (i18n + CSS)
"""
from __future__ import annotations
import json
from pathlib import Path
 # Resolve repository root (…/privacy_audit_reddit)
ROOT = Path(__file__).resolve().parents[1]
from typing import List, Dict, Optional, Tuple

import streamlit as st
import pandas as pd

# 3D donut chart (local module)
from charts import feature_pie3d

# Minimal i18n dictionary for t().
# =================== i18n ===================
TRANSLATIONS: Dict[str, Dict] = {
    "zh": {
        "app": {
            "banner_title": "Privacy Audit · Reddit",
            "banner_sub": "基于模型的账户隐私风险可视化概览",
            "language": "语言",
            "data_source": "数据来源",
            "mode_auto": "自动检测",
            "mode_local": "本地训练结果",
            "mode_demo": "仅演示（滑块）",
            "user": "用户",
            "user_manual": "或手动输入用户名",
            "demo_box": "演示因子（0–100）",
            "import_user": "导入用户文件（csv/json）",
            "model_binary": "二分类模型",
            "model_semantic": "语义模型",
            "model_blend": "融合模型",
            "feature_pie": "关键因素贡献度",
            "radar_title": "风险因子",
            "by_model": "以下内容由模型计算得到",
            "overall_title": "综合风险估计",
            "footer_note": "本页内容仅用于学术演示，不代表平台立场；模型结果可能偏差。© 2025",
            "acc_label": "预估准确率",
            "note_locked": "已根据所选用户自动填充并锁定。切回 Demo 才可手动调整。",
        },
        "risk": {
            "labels": {"low": "低", "medium": "中", "high": "高"},
            "identity": {"name": "身份暴露", "brief": "账号资料或发言可能透露个人信息。"},
            "sensitive": {"name": "敏感话题", "brief": "涉及健康、政治、金融等敏感领域的参与程度。"},
            "exposure": {"name": "曝光度", "brief": "是否常在高流量社区活跃，容易被更多人看到。"},
            "activity": {"name": "活跃度", "brief": "发帖/评论的频率是否较高。"},
            "volume": {"name": "内容体量", "brief": "账户关联的历史内容是否较多。"},
            "concentration": {"name": "话题集中度", "brief": "是否长期聚焦少数话题/社区，容易被画像。"},
        },
        "unknown": "—",
    },
    "en": {
        "app": {
            "banner_title": "Privacy Audit · Reddit",
            "banner_sub": "Model-based privacy risk overview",
            "language": "Language",
            "data_source": "Data Source",
            "mode_auto": "Auto Detect",
            "mode_local": "Local (trained)",
            "mode_demo": "Demo only (sliders)",
            "user": "User",
            "user_manual": "Or type a username",
            "demo_box": "Demo factors (0–100)",
            "import_user": "Import user file (csv/json)",
            "model_binary": "Binary Model",
            "model_semantic": "Semantic/Text Model",
            "model_blend": "Blend/Ensemble",
            "feature_pie": "Key Factors Contribution",
            "radar_title": "Risk Factors",
            "by_model": "Content below is computed by models",
            "overall_title": "Estimated Overall Risk",
            "footer_note": "For academic demo only; results may be biased. © 2025",
            "acc_label": "Estimated Accuracy",
            "note_locked": "Values are auto-filled and locked for the selected user. Switch back to Demo to adjust manually.",
        },
        "risk": {
            "labels": {"low": "Low", "medium": "Medium", "high": "High"},
            "identity": {"name": "Identity", "brief": "Posts/profile may reveal personal info."},
            "sensitive": {"name": "Sensitive Topics", "brief": "Involvement with health/politics/finance."},
            "exposure": {"name": "Exposure", "brief": "Visibility from high-traffic communities."},
            "activity": {"name": "Activity", "brief": "How frequently you post and comment."},
            "volume": {"name": "Content Volume", "brief": "Total content tied to the account."},
            "concentration": {"name": "Focus", "brief": "Whether activity concentrates on a few topics."},
        },
        "unknown": "—",
    },
}
def t(lang: str, path: str) -> str:
    """Return a translated string from TRANSLATIONS by a dotted path.

    If the path is missing, return the original path to surface gaps during development.
    """
    d = TRANSLATIONS.get(lang, TRANSLATIONS["zh"])
    for k in path.split("."):
        d = d.get(k, {})
    return d if isinstance(d, str) else path

# Global styles (glass) and sidebar.
def inject_css():
    """Inject global CSS (glassmorphism, chips, tooltips) once per page.

    Keeping styles in one place makes it easier to reason about UI behaviour
    and avoids scattering magic values across components.
    """
    st.markdown(
        """
        <style>
        .stApp,[data-testid="stAppViewContainer"]{
          background:
            radial-gradient(1200px 700px at -10% -10%, rgba(255,255,255,.30) 0%, rgba(255,255,255,0) 60%),
            radial-gradient(1000px 600px at 120% -10%, rgba(200,220,255,.28) 0%, rgba(255,255,255,0) 60%),
            linear-gradient(135deg,#f6f9ff 0%,#eef2fb 40%,#eaf2ff 100%) !important;
        }
        [data-testid="block-container"]{max-width:1100px;padding-top:.6rem;padding-bottom:2rem;}
        [data-testid="stSidebar"]>div{
          background:rgba(255,255,255,.22);
          border:1px solid rgba(255,255,255,.55);
          box-shadow:0 12px 28px rgba(15,23,42,.14), inset 0 1px 0 rgba(255,255,255,.5);
          backdrop-filter:saturate(160%) blur(24px); -webkit-backdrop-filter:saturate(160%) blur(24px);
          border-radius:22px; margin:10px; padding-top:8px;
        }
        .pa-glass{
          position:relative; overflow:hidden;
          background:linear-gradient(180deg, rgba(255,255,255,.55), rgba(255,255,255,.32));
          border:1px solid rgba(255,255,255,.78);
          box-shadow:0 18px 40px rgba(15,23,42,.16), inset 0 1px 0 rgba(255,255,255,.72);
          backdrop-filter:saturate(200%) blur(18px); -webkit-backdrop-filter:saturate(200%) blur(18px);
          border-radius:22px;
        }
        /* Cards allow tooltips to overflow; this prevents clipping. */
        .pa-glass.pa-card{ overflow: visible; }
        .pa-card{ padding:16px; overflow: visible; }
        .pa-model-card{ min-height:380px; }
        .pa-title{ text-align:center; font-weight:800; font-size:22px; margin:6px 0 10px 0; }
        .pa-card-center{ display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:140px; }

        .pa-chip-row{display:flex; gap:8px; justify-content:center; flex-wrap:wrap; margin:6px 0 2px 0;}
        .pa-chip{ padding:4px 10px; border-radius:999px; font-weight:700; font-size:12px;
          border:1px solid rgba(255,255,255,.65); background:rgba(255,255,255,.7); color:#1f2937; box-shadow:inset 0 1px 0 rgba(255,255,255,.6); }

        .pa-riskchip{ padding:3px 10px; border-radius:999px; font-weight:700; font-size:12px;
          border:1px solid rgba(255,255,255,.6); background:rgba(255,255,255,.88); color:#0f172a; box-shadow: inset 0 1px 0 rgba(255,255,255,.55); }

        .stTable th{ background:rgba(255,255,255,.8) !important; font-weight:700 !important; color:#334155 !important; text-align:left !important; }
        .stTable td{ font-size:13px !important; }

        .pa-note{ display:inline-block;padding:2px 8px;border:1px solid rgba(255,255,255,.6); border-radius:10px;background:rgba(255,255,255,.6);font-size:12px;color:#475569; }
        .pa-footer{ text-align:center; color:#64748b; font-size:12px; margin-top:28px; }

        .pa-meter{ position: relative; width: 100%; height: 44px; border-radius: 9999px; padding: 3px; }
        .pa-meter-fill { height: 100%; border-radius: 9999px; box-shadow: inset 0 -1px 0 rgba(255,255,255,.6); transition: width .5s cubic-bezier(.22,1,.36,1); }
        .pa-meter-label { position:absolute; left:12px; top:50%; transform:translateY(-50%); font-weight:800; color:#0f172a; letter-spacing:.2px; }
        .pa-meter-badge { position:absolute; right:12px; top:50%; transform:translateY(-50%); padding: 4px 10px; border-radius:999px; font-weight:800; font-size:12px;
          border:1px solid rgba(255,255,255,.65); box-shadow: inset 0 1px 0 rgba(255,255,255,.55); background:rgba(255,255,255,.9); color:#0f172a; }

        /* Info icon + custom tooltip (flicker‑free) */
        .pa-tip{
          position:relative;
          display:inline-flex;
          align-items:center;
          gap:6px;
        }
        .pa-i{
          display:inline-flex; align-items:center; justify-content:center; width:18px;height:18px;border-radius:999px;
          background:#eef2ff; color:#475569; font-weight:900; font-size:11px; border:1px solid #c7d2fe; cursor:default; position:relative;
        }
        .pa-i:hover{ background:#c7d2fe; color:#111827; }
        .pa-tooltip {
          opacity:0; visibility:hidden; pointer-events:none; transition:opacity .12s ease;
          position:absolute; top:24px; left:0;
          background:rgba(255,255,255,.98); border:1px solid #cbd5e1; border-radius:10px;
          padding:8px 10px; font-size:12px; color:#111827;
          width:max-content; max-width:280px; white-space:normal; z-index:10;
          box-shadow:0 10px 24px rgba(15,23,42,.18);
        }
        .pa-tip:hover .pa-tooltip {
          opacity:1; visibility:visible; pointer-events:auto;
        }

        /* Make echarts use full column width */
        .st-echarts, div[data-testid="stECharts"], div.stEcharts {
          width: 100% !important;
        }

        section.main hr{ display:none !important; }
        div[data-testid="stMarkdownContainer"] > div:empty{ display:none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
# Blend AUC comparison (disabled).
# Disabled per request: blend model visualization is currently not shown.
def render_blend_compare_auc(bin_m, sem_m, blend_m, lang):
    """Disabled visualisation placeholder.

    We keep the function to document intent and to make re-enabling a
    one-line change (remove the early return and call the chart).
    """
    return  # ---- disabled: no visualization for blend ----


# Blend weights (disabled).
# Disabled per request: blend model visualization is currently not shown.
def render_blend_weights(P_BLEND, lang):
    """Disabled visualisation placeholder for blend weights (LR coefficients)."""
    return  # ---- disabled: no visualization for blend ----

# Cached IO.
@st.cache_data(show_spinner=False)
def load_csv_cached(p: str) -> Optional[pd.DataFrame]:
    """Read a CSV with caching. Returns None if the file is missing or unreadable.

    Using Optional[pd.DataFrame] prevents the common 'truth value of a
    DataFrame is ambiguous' error; callers must check for None/empty explicitly.
    """
    path = Path(p)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_json_cached(p: str) -> Optional[dict]:
    """Read a JSON file with caching and graceful failure (returns None)."""
    path = Path(p)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

# ---- Demo readiness check & threshold loader ----
FEATURES = ROOT / "data/processed/features.csv"
MODEL_DIR = ROOT / "models/cv_logreg"
THR_FILE  = MODEL_DIR / "threshold.txt"

def ensure_ready() -> str:
    """Prefer local assets; fall back to demo gracefully.

    Returns one of:
      - 'ok'          : features and model are available
      - 'no_model'    : features exist but model folder is missing
      - 'no_features' : model exists but features.csv is missing
      - 'demo'        : neither exists; run in pure demo (sliders only)

    This function never calls st.stop(); the page should always render so
    users can still play with the demo sliders or import a user JSON.
    """
    have_features = FEATURES.exists()
    have_model = MODEL_DIR.exists()

    if have_features and have_model:
        return "ok"

    if not have_features and not have_model:
        st.warning(
            "Demo mode: local features and models not found. You can still use the sliders or import a user JSON. "
            "To enable full functionality later, run your local pipeline (featurize/train-cv)."
        )
        return "demo"

    if not have_model:
        st.warning(
            "Model folder not found: models/cv_logreg/. The UI will run in slider/demo mode; "
            "accuracy and importance cards may be empty until you train a model."
        )
        return "no_model"

    # have_model is True but features.csv is missing
    st.warning(
        "data/processed/features.csv not found. Model cards can still show accuracy if metrics exist, "
        "but feature tables derived from features.csv will be hidden."
    )
    return "no_features"

def load_threshold(default: float = 0.5) -> float:
    """Load decision threshold from models/cv_logreg/threshold.txt if present."""
    try:
        return float(THR_FILE.read_text().strip())
    except Exception:
        return default

def first_non_empty_csv(paths: List[Path]) -> Optional[pd.DataFrame]:
    """Return the first existing, non-empty CSV from a list of candidate paths."""
    for p in paths:
        df = load_csv_cached(str(p))
        if df is not None and not df.empty:
            return df
    return None

# User candidates (from data/raw/users).
def load_user_options() -> List[str]:
    """List user candidates from data/raw/users (json/jsonl/ndjson).

    Returns ["Demo"] if the directory is absent or contains no files.
    """
    raw_dir = ROOT / "data/raw/users"
    if raw_dir.exists():
        names = sorted(
            {p.stem for p in raw_dir.glob("*.json")} |
            {p.stem for p in raw_dir.glob("*.jsonl")} |
            {p.stem for p in raw_dir.glob("*.ndjson")}
        )
        if names:
            return ["Demo"] + names[:1000]
    return ["Demo"]

# Page config.
st.set_page_config(page_title="Privacy Audit", layout="wide")
inject_css()
# --- Data source toggle: Auto / Local (trained) / Demo ---
ds_label = t("en", "app.data_source")  # label text localized later

# Probe availability once for rendering the choices
_have_features = FEATURES.exists()
_have_model = MODEL_DIR.exists()
_local_ok = _have_features and _have_model

# Build display labels (append '(unavailable)' when local assets are missing)
opt_auto_label  = t("en", "app.mode_auto")
opt_local_label = t("en", "app.mode_local") + ("  (unavailable)" if not _local_ok else "")
opt_demo_label  = t("en", "app.mode_demo")

# Present to user
DS_OPTIONS = [opt_auto_label, opt_local_label, opt_demo_label]
ds_choice_display = st.sidebar.radio(ds_label, DS_OPTIONS, index=0, key="ds_mode")

# Map selection back to internal codes (match by startswith to ignore the '(unavailable)' suffix)
label2code = {
    t("en", "app.mode_auto"): "auto",
    t("en", "app.mode_local"): "local",
    t("en", "app.mode_demo"): "demo",
}
# Recover the base label (strip suffix if present)
_base_label = next((k for k in label2code if ds_choice_display.startswith(k)), t("en", "app.mode_auto"))
_ds_choice = label2code[_base_label]

# Detect current availability for auto mode resolution
_detected = ensure_ready()

# Resolve final mode according to the user's toggle
if _ds_choice == "demo":
    _mode = "demo"
elif _ds_choice == "local":
    if _local_ok:
        _mode = "ok"  # full local experience
    elif _have_model and not _have_features:
        _mode = "no_features"
        st.info("Local model present but features.csv is missing. Some tables will be hidden.")
    elif _have_features and not _have_model:
        _mode = "no_model"
        st.info("Local features present but model folder is missing. Train a model to enable full local mode.")
    else:
        # Hard block: force back to demo if entirely unavailable
        _mode = "demo"
        st.warning("Local (trained) is unavailable — required files not found. Showing Demo instead.")
else:
    _mode = _detected  # auto

# Threshold depends on model presence
_current_thr = load_threshold(0.5) if _mode in ("ok", "no_features") else 0.5

# Visual cue when Local is unavailable but shown in the radio
if not _local_ok:
    st.sidebar.caption("Local (trained) is unavailable until you run featurize + train-cv.")

# Sidebar: language & user selection
# Show full names ("English", "Chinese") to the user, but translate them to internal codes ("en", "zh").
LANG_MAP = {"English": "en", "Chinese": "zh"}
lang_display = st.sidebar.selectbox(t("en","app.language"), list(LANG_MAP.keys()), index=0)
lang = LANG_MAP[lang_display]

user_opts = load_user_options()
user_sel = st.sidebar.selectbox(t(lang, "app.user"), user_opts, index=0)
user_manual = st.sidebar.text_input(t(lang, "app.user_manual"), value="" if user_sel == "Demo" else "")
final_user = user_manual.strip() or ("" if user_sel == "Demo" else user_sel)

# Top banner (compact).
st.markdown(
    f'''
    <div class="pa-glass" style="padding:18px; margin:6px 0 14px 0;">
      <h1 style="margin:0;font-size:20px;font-weight:900;color:#0f172a;">{t(lang,"app.banner_title")}</h1>
      <div style="margin-top:4px;color:#475569;font-size:13px;">{t(lang,"app.banner_sub")}</div>
    </div>
    ''',
    unsafe_allow_html=True,
)
st.caption(f'{t(lang,"app.by_model")}', unsafe_allow_html=True)
if _mode in ("ok", "no_features"):
    st.caption(f'Loaded model: models/cv_logreg  ·  threshold={_current_thr:.3f}')
else:
    st.caption('Demo mode: sliders are active (no local model).')

# Default demo factors.
factors = {"identity":78, "sensitive":52, "exposure":33, "activity":60, "volume":44, "concentration":20}
# Used when no concrete user is selected; values chosen for a balanced radar.

# If a user is selected, derive factors from raw JSON and lock sliders.
def derive_factors_from_user(username: str) -> Optional[Dict[str, float]]:
    """Heuristic factor estimation from a raw user JSON.

    This is intentionally simple and bounded to [0, 100] so the UI remains
    responsive even if real models are not loaded. Replace with proper
    feature extraction when integrating your ML pipeline.
    """
    if not username:
        return None
    raw = load_json_cached(str((ROOT/"data/raw/users")/f"{username}.json"))
    if not raw:
        return None
    # Heuristic: use post/comment counts and bio length (0–100)
    def nz(x): return x if isinstance(x,(int,float)) else 0
    posts = nz(len(raw.get("posts", [])))
    comments = nz(len(raw.get("comments", [])))
    bio = raw.get("bio") or raw.get("description") or ""

    def norm(x, hi):  # simple normalisation
        try:
            v = max(0.0, min(float(x)/float(hi), 1.0))
            return round(v*100, 0)
        except Exception:
            return 0.0

    total = posts + comments
    # For demo: identity uses bio length, sensitive is 50 (placeholder; replace with model output)
    out = {
        "identity": norm(len(bio), 120),         # based on bio length
        "sensitive": 50.0,                       # placeholder; replace with model output
        "exposure": norm(posts, 400) + norm(comments, 800) * 0.5,
        "activity": norm(total, 1200),
        "volume": norm(total, 2000),
        "concentration": 50.0,                   # placeholder; replace with topic stats
    }
    # clamp to [0, 100]
    for k in out:
        out[k] = max(0.0, min(100.0, float(out[k])))
    return out

locked_from_user = False
if final_user:
    guessed = derive_factors_from_user(final_user)
    if guessed:
        factors.update({k:int(v) for k,v in guessed.items()})
        locked_from_user = True

# Sidebar sliders (disabled when user is selected).
with st.sidebar.expander(t(lang, "app.demo_box"), expanded=(final_user=="")):
    for k in list(factors.keys()):
        label = TRANSLATIONS[lang]["risk"][k]["name"]
        factors[k] = st.slider(
            label, 0, 100, int(factors[k]), 1,
            key=f"slide_{k}",
            disabled=(locked_from_user and final_user!="")
        )
    if locked_from_user and final_user!="":
        st.caption(t(lang, "app.note_locked"))

# Overall risk meter.
def risk_level(score: float) -> str:
    """Map a 0–100 score to a discrete label: low / medium / high."""
    return "high" if score >= 70 else "medium" if score >= 40 else "low"

def render_overall_meter(score: float, lang: str):
    """Render the overall risk meter with a colour gradient and badges."""
    lvl = risk_level(score); pct = int(round(score))
    grad = "linear-gradient(90deg, #86efac, #16a34a)" if lvl=="low" else \
           "linear-gradient(90deg, #fde68a, #f59e0b)" if lvl=="medium" else \
           "linear-gradient(90deg, #fca5a5, #ef4444)"
    st.markdown(
        f"""
        <div class="pa-glass pa-meter">
          <div class="pa-meter-fill" style="width:{pct}%; background:{grad};"></div>
          <div class="pa-meter-label">{t(lang,"app.overall_title")} · {pct}%</div>
          <div class="pa-meter-badge">{TRANSLATIONS[lang]["risk"]["labels"][lvl]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def overall_score(factors: Dict[str, float]) -> float:
    """Compute a weighted mean of the six factors.

    The weights are empirical and can be tuned; we normalise by the sum of
    weights to keep the score in [0, 100].
    """
    W = {"identity":1.1,"sensitive":1.2,"exposure":0.9,"activity":1.0,"volume":0.8,"concentration":1.0}
    s = sum(factors.get(k,0)*W.get(k,1.0) for k in W); w = sum(W.values())
    return s/max(w, 1e-9)

render_overall_meter(overall_score(factors), lang)

# Radar + explanation cards.
col_radar, col_tags = st.columns([5,7])

with col_radar:
    st.markdown('<div class="pa-glass pa-card" style="padding:14px">', unsafe_allow_html=True)
    st.markdown(f'<div style="font-weight:800; font-size:18px; color:#0f172a; margin:0 0 10px 0;">{t(lang,"app.radar_title")}</div>', unsafe_allow_html=True)
    from ui_streamlit_radar import radar_chart
    display_labels = [TRANSLATIONS[lang]["risk"][k]["name"] for k in factors.keys()]
    radar_chart(factors, labels=display_labels, animate=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_tags:
    def brief_card(key: str, label: str, brief: str, val: float):
        lvl = risk_level(val)
        badge = {"low":"#86efac","medium":"#fde68a","high":"#fecaca"}[lvl]
        st.markdown(
            f"""
            <div class="pa-glass pa-card" style="margin:12px 0; position:relative;">
              <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="font-weight:800;display:flex;align-items:center;gap:6px;">
                  {label}
                  <span class="pa-tip"><span class="pa-i">i</span><span class="pa-tooltip">{brief}</span></span>
                </span>
                <span class="pa-riskchip" style="background:{badge};">
                  {TRANSLATIONS[lang]["risk"]["labels"][lvl]}
                </span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    for k in factors:
        brief_card(k, TRANSLATIONS[lang]["risk"][k]["name"], TRANSLATIONS[lang]["risk"][k]["brief"], float(factors[k]))

# Feature labels for charts/tables.
FEATURE_NAME_MAP = {
    "zh": {
        "text_frac_sensitive": "敏感词占比",
        "text_mean_prob": "文本平均风险分",
        "username_len": "用户名长度",
        "username_digit_frac": "用户名数字比例",
        "username_seg_count": "用户名分段数",
        "username_has_year": "用户名包含年份",
        "username_has_date": "用户名包含日期",
        "total_activity": "总活跃度",
        "comment_post_ratio": "评论/发帖比",
        "posts": "发帖数",
        "comments": "评论数",
        "log_total_activity": "活跃度（对数）",
        "log_comments": "评论数（对数）",
        "log_posts": "发帖数（对数）",
        "post_comment_ratio": "发帖/评论比"
    },
    "en": {
        "text_frac_sensitive": "Sensitive Word Ratio",
        "text_mean_prob": "Mean Risk Score",
        "username_len": "Username Length",
        "username_digit_frac": "Digits in Username",
        "username_seg_count": "Username Segments",
        "username_has_year": "Username Has Year",
        "username_has_date": "Username Has Date",
        "total_activity": "Total Activity",
        "comment_post_ratio": "Comment/Post Ratio",
        "posts": "Posts",
        "comments": "Comments",
        "log_total_activity": "Log(Total Activity)",
        "log_comments": "Log(Comments)",
        "log_posts": "Log(Posts)",
        "post_comment_ratio": "Post/Comment Ratio"
    }
}
def localize_feature_table(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    """Return a copy of df where the 'feature' column is translated for lang."""
    df = df.copy()
    if "feature" in df.columns:
        mapping = FEATURE_NAME_MAP["zh" if lang=="zh" else "en"]
        df["feature"] = df["feature"].apply(lambda x: mapping.get(str(x), str(x)))
    return df

# Model section: accuracy + 3D donut per model.
P_BIN   = ROOT / "models/cv_logreg"  # Logistic Regression (tabular)
P_SEM   = ROOT / "models/cv_lgbm"    # LightGBM (tabular)
P_BLEND = ROOT / "models/cv_rf"      # Random Forest (tabular)

def show_model_card(title: str, metrics: Optional[pd.DataFrame], fi_df: Optional[pd.DataFrame], pie_key: str):
    """Render a compact model card: title, accuracy chip and 3D donut chart.

    Parameters
    ----------
    title : str
        Section title (i18n already applied by caller).
    metrics : Optional[pd.DataFrame]
        Single-row table with at least a 'brier' column for accuracy.
    fi_df : Optional[pd.DataFrame]
        Feature-importance table with 'feature' and weight columns.
    pie_key : str
        Unique Streamlit key for the ECharts component.
    """
    st.markdown(f'<div style="font-weight:900;font-size:18px;color:#0f172a;margin:0 0 6px 0;">{title}</div>', unsafe_allow_html=True)

    # Accuracy chip from Brier: (1 - Brier) * 100
    if metrics is not None and not metrics.empty and "brier" in metrics.columns:
        try:
            b = float(metrics.iloc[0]["brier"])
            acc = max(0.0, min(1.0, 1.0 - b)) * 100.0
            st.markdown(
                f'<div class="pa-chip-row"><span class="pa-chip">{t(lang,"app.acc_label")}: {acc:.1f}%</span></div>',
                unsafe_allow_html=True,
            )
        except Exception:
            pass

    if fi_df is not None and not fi_df.empty:
        fi_df = localize_feature_table(fi_df, lang)
        st.caption(t(lang, "app.feature_pie"))
        try:
            feature_pie3d(fi_df, key=pie_key, height=320)
        except TypeError:
            feature_pie3d(fi_df, key=pie_key)

st.markdown("---")
c1, c2, c3 = st.columns(3)

with c1:
    df_m = first_non_empty_csv([P_BIN/"metrics_cv.csv", P_BIN/"cv_metrics.csv"])
    df_fi = first_non_empty_csv([P_BIN/"feature_importance.csv", P_BIN/"feature_importance_name.csv"])
    show_model_card(t(lang,"app.model_binary"), df_m, df_fi, "pie_bin")

with c2:
    df_m = first_non_empty_csv([P_SEM/"metrics_cv.csv", P_SEM/"cv_metrics.csv"])
    df_fi = first_non_empty_csv([P_SEM/"feature_importance.csv", P_SEM/"feature_importance_name.csv"])
    show_model_card(t(lang,"app.model_semantic"), df_m, df_fi, "pie_sem")

with c3:
    # --- Blend/Ensemble: only show title and accuracy chip (no note/caption) ---
    _rf_title = "随机森林" if lang=="zh" else "Random Forest"
    st.markdown(f'<div style="font-weight:900;font-size:18px;color:#0f172a;margin:0 0 6px 0;">{_rf_title}</div>', unsafe_allow_html=True)
    df_m_blend = first_non_empty_csv([P_BLEND/"metrics_cv.csv", P_BLEND/"cv_metrics.csv"])
    if df_m_blend is not None and not df_m_blend.empty and "brier" in df_m_blend.columns:
        try:
            b = float(df_m_blend.iloc[0]["brier"])
            acc = max(0.0, min(1.0, 1.0 - b)) * 100.0
            st.markdown(
                f'<div class="pa-chip-row"><span class="pa-chip">{t(lang,"app.acc_label")}: {acc:.1f}%</span></div>',
                unsafe_allow_html=True,
            )
        except Exception:
            pass

# Footer.
st.markdown(f'<div class="pa-footer">{t(lang,"app.footer_note")}</div>', unsafe_allow_html=True)