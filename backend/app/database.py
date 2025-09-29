# app/database.py
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
import os

# Database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:2608@localhost:5432/db-proy-ml")
SYNC_DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:2608@localhost:5432/db-proy-ml").replace("+asyncpg", "")

# Create async engine
async_engine = create_async_engine(DATABASE_URL, echo=True)

# Create sync engine
sync_engine = create_engine(SYNC_DATABASE_URL, echo=True)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    async_engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

# Create sync session factory
SyncSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)

# Base class for models
Base = declarative_base()

# Dependency to get database session
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# Dependency to get sync database session
def get_sync_db():
    db = SyncSessionLocal()
    try:
        yield db
    finally:
        db.close()
