@@ -src/app.py
@@ def train_cv_impl(data_csv: str, outdir: str, folds: int = 5, random_state: int = 42):
-    df = _read_csv(data_csv)
-    feat_df, feat_cols = _prepare_features(df, columns=None)
-    _save_feature_columns(outdir, feat_cols)
-
-    X_all = feat_df.fillna(0.0).to_numpy(dtype=float)
-    if "y" not in df.columns:
-        raise KeyError("训练需要标签列 y，请先运行 auto-label 生成 train.csv。")
-    y_all = df["y"].astype(int).to_numpy()
+    df = _read_csv(data_csv)
+    feat_df, feat_cols = _prepare_features(df, columns=None)
+    _save_feature_columns(outdir, feat_cols)
+
+    # keep DataFrame with column names to avoid sklearn/lightgbm warnings
+    X_all_df = feat_df.fillna(0.0).astype(float)
+    if "y" not in df.columns:
+        raise KeyError("训练需要标签列 y，请先运行 auto-label 生成 train.csv。")
+    y_all = df["y"].astype(int).to_numpy()
@@ def train_cv_impl(data_csv: str, outdir: str, folds: int = 5, random_state: int = 42):
-        Xtr, Xva = X_all[tr], X_all[va]
-        ytr, yva = y_all[tr], y_all[va]
-
-        lgb_train = lgb.Dataset(Xtr, label=ytr)
-        lgb_valid = lgb.Dataset(Xva, label=yva, reference=lgb_train)
+        Xtr_df, Xva_df = X_all_df.iloc[tr], X_all_df.iloc[va]
+        ytr, yva = y_all[tr], y_all[va]
+
+        lgb_train = lgb.Dataset(Xtr_df.to_numpy(), label=ytr)
+        lgb_valid = lgb.Dataset(Xva_df.to_numpy(), label=yva, reference=lgb_train)
@@ def train_cv_impl(data_csv: str, outdir: str, folds: int = 5, random_state: int = 42):
-        clf.fit(Xtr, ytr)
-
-        proba = clf.predict_proba(Xva)[:, 1]
+        clf.fit(Xtr_df, ytr)
+
+        proba = clf.predict_proba(Xva_df)[:, 1]
