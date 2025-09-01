# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional
import streamlit as st

# Glassmorphism styles and layout
def inject_css():
    st.markdown(
        """
        <style>
        :root { --ink:#1f2937; --muted:#64748b; --glass: rgba(255,255,255,.65); }
        html,body,.stApp{ font-family: ui-sans-serif, Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; }
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
          background:linear-gradient(180deg, rgba(255,255,255,.58), rgba(255,255,255,.35));
          border:1px solid rgba(255,255,255,.78);
          box-shadow:0 18px 40px rgba(15,23,42,.12), inset 0 1px 0 rgba(255,255,255,.7);
          backdrop-filter:saturate(200%) blur(18px); -webkit-backdrop-filter:saturate(200%) blur(18px);
          border-radius:22px;
        }
        .pa-card{ padding:16px; }
        .pa-title{ text-align:center; font-weight:800; font-size:22px; margin:6px 0 10px 0; color:var(--ink); }
        .pa-note{ display:inline-block;padding:2px 8px;border:1px solid rgba(255,255,255,.6); border-radius:10px;background:rgba(255,255,255,.6);font-size:12px;color:#475569; }
        .pa-chip-row{display:flex; gap:8px; justify-content:center; flex-wrap:wrap; margin:8px 0 2px 0;}
        .pa-chip{ padding:4px 10px; border-radius:999px; font-weight:700; font-size:12px;
                  border:1px solid rgba(255,255,255,.65); background:rgba(255,255,255,.8); color:#1f2937;
                  box-shadow:inset 0 1px 0 rgba(255,255,255,.6); }
        .pa-riskchip{ padding:3px 10px; border-radius:999px; font-weight:700; font-size:12px;
                      border:1px solid rgba(255,255,255,.6); background:rgba(255,255,255,.88); color:#0f172a; box-shadow: inset 0 1px 0 rgba(255,255,255,.55); }
        .pa-footer{ text-align:center; color:#64748b; font-size:12px; margin-top:28px; }
        .pa-meter{ position: relative; width: 100%; height: 44px; border-radius: 9999px; padding: 3px; }
        .pa-meter-fill { height: 100%; border-radius: 9999px; box-shadow: inset 0 -1px 0 rgba(255,255,255,.6); transition: width .5s cubic-bezier(.22,1,.36,1); }
        .pa-meter-label { position:absolute; left:12px; top:50%; transform:translateY(-50%); font-weight:800; color:#0f172a; letter-spacing:.2px; }
        .pa-meter-badge { position:absolute; right:12px; top:50%; transform:translateY(-50%); padding: 4px 10px; border-radius:999px; font-weight:800; font-size:12px;
                          border:1px solid rgba(255,255,255,.65); box-shadow: inset 0 1px 0 rgba(255,255,255,.55); background:rgba(255,255,255,.9); color:#0f172a; }
        section.main hr{ display:none !important; }
        div[data-testid="stMarkdownContainer"] > div:empty{ display:none !important; }
        </style>
        """,
        unsafe_allow_html=True
    )

def banner(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="pa-glass" style="padding:18px; margin:6px 0 14px 0;">
          <h1 style="margin:0;font-size:20px;font-weight:900;color:#0f172a;">{title}</h1>
          <div style="margin-top:4px;color:#475569;font-size:13px;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def risk_chip(level_text: str, color: str):
    return f'<span class="pa-riskchip" style="background:{color};">{level_text}</span>'

def meter(title: str, pct: float, level_text: str, grad_css: str):
    pct = max(0, min(100, int(round(pct))))
    st.markdown(
        f"""
        <div class="pa-glass pa-meter">
          <div class="pa-meter-fill" style="width:{pct}%; background:{grad_css};"></div>
          <div class="pa-meter-label">{title} · {pct}%</div>
          <div class="pa-meter-badge">{level_text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def card_open_center(title: str, chip_html: Optional[str] = None):
    st.markdown(f'<div class="pa-glass pa-card" style="display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:140px;"><div class="pa-title">{title}</div>{chip_html or ""}', unsafe_allow_html=True)

def card_close():
    st.markdown("</div>", unsafe_allow_html=True)

def footer(note: str):
    st.markdown(f'<div class="pa-footer">{note}</div>', unsafe_allow_html=True)