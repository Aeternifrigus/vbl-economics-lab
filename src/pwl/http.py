import json
import time
import urllib.error
import urllib.parse
import urllib.request


def get_json(url, params=None, retries=3, timeout=30, backoff=2.0):
    """GET a JSON document. Error messages never include the URL, which may hold an API key."""
    if params:
        url = url + "?" + urllib.parse.urlencode(params, doseq=True)
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            last = type(exc).__name__
            time.sleep(backoff**attempt)
    raise RuntimeError(f"GET failed after {retries} tries ({last})")
