#!/usr/bin/env bash
set -euo pipefail

echo "[1] 创建目标目录..."
mkdir -p models/{binary,semantic,blend}

echo "[2] 移动关键文件到 binary/ (二分类 LightGBM)"
# 指标
mv models/model/metrics_cv.csv models/cv_safe/cv_metrics.csv models/binary/ 2>/dev/null || true
# 特征重要性
mv models/model/feature_importance*.csv models/binary/ 2>/dev/null || true
# 特征定义
mv models/cv_safe/feature_columns.txt models/binary/ 2>/dev/null || true

echo "[3] 移动关键文件到 semantic/ (语义模型)"
mv models/model_text/metrics_cv.csv models/text_model/metrics_cv.csv models/model_text_tfidf/metrics_cv.csv models/semantic/ 2>/dev/null || true
mv models/model_text/feature_importance*.csv models/text_model/feature_importance*.csv models/model_text_tfidf/feature_importance*.csv models/semantic/ 2>/dev/null || true
mv models/model_text/feature_columns.txt models/text_model/feature_columns.txt models/model_text_tfidf/feature_columns.txt models/semantic/ 2>/dev/null || true

echo "[4] 移动关键文件到 blend/ (融合模型)"
mv models/blend_lr/final_lr.json models/blend/ 2>/dev/null || true
mv models/blend_lr/metrics_cv.csv models/blend/ 2>/dev/null || true
mv models/final_blend/final_blend_summary.json models/blend/ 2>/dev/null || true

echo "[5] 删除冗余文件 (预测/日志/折叠模型)"
find models -type f \( -name "oof_*.csv" -o -name "pred_test*.csv" -o -name "*_pred_*.csv" -o -name "train_log_fold*.csv" -o -name "lgbm_fold*.pkl" \) -print -delete

echo "[6] 清理空目录"
find models -type d -empty -delete

echo "[7] 整理完成，最终目录结构如下:"
find models -maxdepth 2 -print | sort