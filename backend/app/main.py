import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import auth, patient, triage, doctor, admin, system

logger = logging.getLogger(__name__)

# Session cleanup interval in seconds
_CLEANUP_INTERVAL = 300  # 5 minutes


async def _session_cleanup_loop():
    """Periodic background task to evict expired agent sessions."""
    from .services.session_manager import AgentSessionManager
    while True:
        await asyncio.sleep(_CLEANUP_INTERVAL)
        try:
            manager = AgentSessionManager.get_instance()
            removed = manager.cleanup_expired_sessions()
            if removed:
                logger.info("Session cleanup: removed %d expired sessions", removed)
        except Exception:
            logger.exception("Session cleanup error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — initialise and tear down the session manager."""
    # Startup: initialise the session manager singleton
    from .services.session_manager import AgentSessionManager
    manager = AgentSessionManager.get_instance()
    logger.info("AgentSessionManager ready (%d active sessions)", manager.active_session_count)

    # Preload heavyweight Triage models once at startup so the first patient
    # triage request does not stall while loading HuggingFace weights.
    await asyncio.to_thread(manager.preload_triage_models)
    logger.info("Triage models preloaded successfully; server startup is complete.")

    # Start the periodic cleanup task
    cleanup_task = asyncio.create_task(_session_cleanup_loop())

    yield

    # Shutdown: cancel cleanup task
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    logger.info("AgentSessionManager shutdown complete")


app = FastAPI(title="Medora API", version="1.0.0", lifespan=lifespan)

# CORS — allow the Vite frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(patient.router)
app.include_router(triage.router)
app.include_router(doctor.router)
app.include_router(admin.router)
app.include_router(system.router)


@app.get("/")
def root():
    return {"message": "Welcome to Medora API"}