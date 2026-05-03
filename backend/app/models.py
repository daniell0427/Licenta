from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, DateTime, ForeignKey, UniqueConstraint, Index,
    Text, Float, Boolean,
)
from sqlalchemy.orm import relationship
from .db import Base


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    watchlist = relationship("WatchlistItem", back_populates="user", cascade="all, delete-orphan")


class WatchlistItem(Base):
    __tablename__ = "watchlist_items"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    ticker = Column(String, nullable=False)
    position = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="watchlist")

    __table_args__ = (
        UniqueConstraint("user_id", "ticker", name="uq_user_ticker"),
        Index("ix_watchlist_user", "user_id"),
    )


class Cache(Base):
    """Generic JSON cache: key → payload + updated_at. Replaces the in-memory dict
    so data survives restarts and TTLs can be longer for expensive ops."""
    __tablename__ = "cache"
    key = Column(String, primary_key=True)
    payload = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class Notification(Base):
    __tablename__ = "notifications"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    type = Column(String, nullable=False)             # "price_move" | "news"
    ticker = Column(String, nullable=False)
    title = Column(String, nullable=False)
    subtitle = Column(String, default="")
    severity = Column(String, default="medium")      # "low" | "medium" | "high"
    direction = Column(String, default="")            # "up" | "down" | ""
    label = Column(String, default="")                # "positive" | "negative" | ""
    url = Column(String, default="")
    fingerprint = Column(String, nullable=False)      # idempotency key
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    read_at = Column(DateTime, nullable=True)
    dismissed = Column(Boolean, default=False, nullable=False)

    __table_args__ = (
        UniqueConstraint("user_id", "fingerprint", name="uq_notif_user_fp"),
        Index("ix_notif_user_created", "user_id", "created_at"),
    )


# ---------- Research data ----------

class NewsArticle(Base):
    """Every news article we've ever seen, with timestamps and FinBERT sentiment.
    Stored once, scored once — research-friendly long-term archive."""
    __tablename__ = "news_articles"
    id = Column(Integer, primary_key=True)
    ticker = Column(String, nullable=False, index=True)
    title = Column(Text, nullable=False)
    summary = Column(Text, default="")
    publisher = Column(String, default="")
    url = Column(Text, default="")
    thumbnail = Column(Text, default="")
    published_at = Column(DateTime, index=True)              # timestamp from feed
    fetched_at = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)
    sentiment_label = Column(String, default="")            # "positive" | "neutral" | "negative"
    sentiment_confidence = Column(Float, default=0.0)
    sentiment_signed = Column(Float, default=0.0)            # P(pos) - P(neg)
    fingerprint = Column(String, nullable=False)             # dedup key

    __table_args__ = (
        UniqueConstraint("ticker", "fingerprint", name="uq_news_ticker_fp"),
        Index("ix_news_ticker_pub", "ticker", "published_at"),
    )


class PriceBar(Base):
    """OHLCV bar for a (ticker, interval, time) tuple. Used to keep a local
    historical archive that survives Yahoo outages and supports research."""
    __tablename__ = "price_bars"
    id = Column(Integer, primary_key=True)
    ticker = Column(String, nullable=False, index=True)
    interval = Column(String, nullable=False)                # "1d", "1h", "5m", ...
    bar_time = Column(DateTime, nullable=False)              # bar start, UTC
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    fetched_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("ticker", "interval", "bar_time", name="uq_bar"),
        Index("ix_bar_ticker_interval_time", "ticker", "interval", "bar_time"),
    )


class PredictionRecord(Base):
    """Every prediction we've ever generated — lets us evaluate accuracy
    against realized returns later."""
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True)
    ticker = Column(String, nullable=False, index=True)
    horizon_days = Column(Integer, nullable=False)           # 1 or 5
    direction = Column(String, nullable=False)               # "up" | "down"
    confidence = Column(Float, default=0.0)
    train_acc = Column(Float, default=0.0)
    val_acc = Column(Float, default=0.0)
    sentiment_score = Column(Float, default=0.0)             # aggregate sentiment used as input
    last_close = Column(Float, default=0.0)                  # price at prediction time
    generated_at = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)

    __table_args__ = (
        Index("ix_pred_ticker_gen", "ticker", "generated_at"),
    )
