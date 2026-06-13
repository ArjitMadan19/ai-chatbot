from fastapi import FastAPI
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.extension import _rate_limit_exceeded_handler
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from backend.app.api.routes import router
from backend.app.services.config import settings
from backend.app.services.database import init_db


def create_app():
    app = FastAPI(
        title=settings.app_title,
        description=settings.app_description,
        version=settings.app_version
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
