import hashlib
import json

from redis import Redis
from redis.exceptions import RedisError

from app.services.config import settings


redis_client = None

if settings.cache_enabled:
    redis_client = Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_connect_timeout=1,
        socket_timeout=1
    )


def build_cache_key(question, doc_type_filter=None, conversation_history=None):
    payload = {
        "question": question,
        "doc_type_filter": doc_type_filter,
        "conversation_history": conversation_history or []
    }
    normalized_payload = json.dumps(payload, sort_keys=True)
    payload_hash = hashlib.sha256(normalized_payload.encode("utf-8")).hexdigest()
    return f"ask:{payload_hash}"


def get_cached_answer(cache_key):
    if redis_client is None:
        return None

    try:
        cached_value = redis_client.get(cache_key)
    except RedisError:
        return None

    if cached_value is None:
        return None

    return json.loads(cached_value)


def set_cached_answer(cache_key, value):
    if redis_client is None:
        return

    try:
        redis_client.setex(
            cache_key,
            settings.cache_ttl_seconds,
            json.dumps(value)
        )
    except RedisError:
        return
