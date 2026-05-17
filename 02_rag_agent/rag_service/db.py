"""Database setup — SQLAlchemy async with pgvector."""
from __future__ import annotations

import os

from sqlalchemy import Column, Text, Integer, BigInteger, ForeignKey
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from pgvector.sqlalchemy import Vector

DATABASE_URL = os.environ["DATABASE_URL"]
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

engine = create_async_engine(DATABASE_URL, pool_size=10, max_overflow=5)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id           = Column(Text, primary_key=True)
    url          = Column(Text)
    title        = Column(Text, nullable=False, default="")
    contents     = Column(Text, nullable=False, default="")
    article_type = Column(Text)
    title_vec    = Column(Vector(EMBED_DIM))
    # fts column is GENERATED in SQL — not mapped


class Chunk(Base):
    __tablename__ = "chunks"

    id          = Column(BigInteger, primary_key=True, autoincrement=True)
    doc_id      = Column(Text, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_text  = Column(Text, nullable=False)
    chunk_vec   = Column(Vector(EMBED_DIM))
    # fts column is GENERATED in SQL — not mapped


async def get_session() -> AsyncSession:
    async with SessionLocal() as session:
        yield session
