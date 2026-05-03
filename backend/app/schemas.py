from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    email: EmailStr
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=6, max_length=128)


class UserLogin(BaseModel):
    email_or_username: str
    password: str


class UserOut(BaseModel):
    id: int
    email: str
    username: str
    created_at: datetime

    class Config:
        from_attributes = True


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class WatchlistAdd(BaseModel):
    ticker: str = Field(min_length=1, max_length=12)


class WatchlistRow(BaseModel):
    ticker: str
    price: Optional[float] = None
    change_pct: Optional[float] = None


class WatchlistOut(BaseModel):
    items: List[WatchlistRow]
