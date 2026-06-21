from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.extension import _rate_limit_exceeded_handler
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from app.api.routes import router
from app.services.config import settings
from app.services.database import init_db


def create_app():
    app = FastAPI(
        title=settings.app_title,
        description=settings.app_description,
        version=settings.app_version
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=[settings.rate_limit]
    )
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    app.include_router(router)

    @app.on_event("startup")
    def startup():
        init_db()

    return app


api = create_app()
