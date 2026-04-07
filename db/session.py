from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

engine = create_async_engine(settings.database_url, echo=settings.debug)
async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    """Initialize the database engine and create tables (called on app startup).

    Uses create_all which is safe for existing tables (no-op if table exists).
    For schema changes to existing tables, run migrations manually:
        psql -f db/migrations/versions/002_hybrid_search.sql
    """
    from db.models import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def dispose_db():
    """Dispose the database engine (called on app shutdown)."""
    await engine.dispose()
