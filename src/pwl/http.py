"""HTTP client with retries and logging."""

import json
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import certifi

from .exceptions import IngestionError
from .logging_config import get_logger

logger = get_logger(__name__)

# Create SSL context with certifi's CA bundle
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def get_json(
    url: str,
    params: dict[str, Any] | None = None,
    retries: int = 3,
    timeout: int = 30,
    backoff: float = 2.0,
) -> dict[str, Any]:
    """GET a JSON document with retries. Error messages never include the URL,
    which may hold an API key."""
    if params:
        url = url + "?" + urllib.parse.urlencode(params, doseq=True)

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            logger.debug(
                "HTTP GET attempt %d/%d",
                attempt + 1,
                retries,
                extra={"extra_fields": {"url_host": urllib.parse.urlparse(url).netloc}},
            )
            req = urllib.request.Request(url, headers={"User-Agent": "vbl-economics-lab/0.1"})
            with urllib.request.urlopen(req, timeout=timeout, context=SSL_CONTEXT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                logger.debug(
                    "HTTP GET success",
                    extra={"extra_fields": {"status": resp.status}},
                )
                return data
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            wait_time = backoff**attempt
            logger.warning(
                "HTTP GET failed (attempt %d/%d): %s, retrying in %.1fs",
                attempt + 1,
                retries,
                type(exc).__name__,
                wait_time,
                extra={"extra_fields": {"error_type": type(exc).__name__, "attempt": attempt + 1}},
            )
            time.sleep(wait_time)

    parsed = urllib.parse.urlparse(url)
    raise IngestionError(
        f"GET failed after {retries} tries",
        source=parsed.netloc,
        original_error=last_error,
    )
