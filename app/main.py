from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: init DB, create chat provider
    from db.session import init_db
    await init_db()

    # Initialize chat provider from config
    import providers  # triggers auto-registration
    from providers.registry import registry

    provider_kwargs = {}
    if settings.chat_provider == "gemini":
        provider_kwargs = {
            "project_id": settings.gcp_project_id,
            "region": settings.gcp_region,
            "default_model": settings.chat_model,
        }
        if settings.chat_api_key:
            provider_kwargs["api_key"] = settings.chat_api_key
    elif settings.chat_provider == "openai":
        provider_kwargs = {"api_key": settings.openai_api_key, "default_model": settings.chat_model}
    elif settings.chat_provider == "anthropic":
        provider_kwargs = {"api_key": settings.anthropic_api_key, "default_model": settings.chat_model}
    elif settings.chat_provider == "ollama":
        provider_kwargs = {"base_url": settings.ollama_base_url, "default_model": settings.chat_model}

    app.state.chat_provider = registry.create_chat(settings.chat_provider, **provider_kwargs)

    # Initialize embedding provider + service
    embedding_kwargs = {}
    if settings.embedding_provider == "gemini":
        embedding_kwargs = {
            "project_id": settings.gcp_project_id,
            "region": settings.gcp_region,
            "default_model": settings.embedding_model,
        }
    elif settings.embedding_provider == "openai":
        embedding_kwargs = {"api_key": settings.openai_api_key, "default_model": settings.embedding_model}

    try:
        embedding_provider = registry.create_embedding(settings.embedding_provider, **embedding_kwargs)
        from context_engine.embedding_service import EmbeddingService
        app.state.embedding_service = EmbeddingService(embedding_provider)
    except KeyError:
        app.state.embedding_service = None

    yield

    from db.session import dispose_db
    await dispose_db()


app = FastAPI(
    title="Ark Chatbot Context Engine",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from app.routes.chats import router as chats_router
from app.routes.messages import router as messages_router

app.include_router(chats_router)
app.include_router(messages_router)


from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# Serve static frontend
_static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

    @app.get("/")
    async def index():
        return FileResponse(os.path.join(_static_dir, "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok"}
