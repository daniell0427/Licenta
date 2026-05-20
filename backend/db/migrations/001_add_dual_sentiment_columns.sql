-- Phase 2.1 — dual headline/summary sentiment columns on news_articles.
--
-- These four REAL columns hold sentiment scored on the headline and the
-- article summary SEPARATELY, which the narrative-consistency features
-- (signed / absolute / confidence-weighted divergence) are built from.
--
-- The legacy sentiment_label / sentiment_confidence / sentiment_signed columns
-- are kept untouched so existing read endpoints continue to work.
--
-- Apply with:  sqlite3 app.db < db/migrations/001_add_dual_sentiment_columns.sql
-- SQLite ALTER TABLE ADD COLUMN is non-destructive; existing rows get the default.

ALTER TABLE news_articles ADD COLUMN headline_score      REAL DEFAULT 0.0;
ALTER TABLE news_articles ADD COLUMN headline_confidence REAL DEFAULT 0.0;
ALTER TABLE news_articles ADD COLUMN summary_score       REAL DEFAULT 0.0;
ALTER TABLE news_articles ADD COLUMN summary_confidence  REAL DEFAULT 0.0;
