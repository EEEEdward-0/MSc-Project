# Data folder

## raw/
- `comments.csv`: Flatten from Reddit JSON (source of text). Columns: at least `username`, `text`, possibly others depending on extraction.

## processed/splits/
- `train.csv`: user-level features/labels for training (columns include `user_id`, `label`, ...).
- `dev_folds.csv`: user-level dev with `fold`.
- `test.csv`: user-level test users.

## processed/text/
- `comment_probs_userid.csv`: aggregated text-side features per `user_id` (e.g., `p_sensitive`, `text_len`).
- `legacy_comment_probs_username.csv` (if present): same idea but keyed by `username` (kept only for traceability).

## processed/blends/
- `dev_blend.csv`: dev set with `oof_pred_name`, `oof_pred_text` and label/fold, used for blending.

## Notes
- Raw JSON files remain under `data/raw/` (unchanged).
- Symlinks in `processed/` keep backward compatibility with existing scripts.
