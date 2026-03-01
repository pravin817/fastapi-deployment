"""HTTP utility functions for making requests."""

import json
import logging
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

def _http(url, method="GET", headers=None, body=None):
    """Make an HTTP request and return status, response, and headers.
    
    Args:
        url: The URL to request
        method: HTTP method (GET, POST, PUT, DELETE, etc.), defaults to GET
        headers: Optional dictionary of HTTP headers
        body: Optional request body (dict or string)
        
    Returns:
        Tuple of (status_code, response, response_headers)
        
    Raises:
        ConnectionError: If the request fails due to network issues
    """
    # Prepare request body and headers
    if isinstance(body, dict):
        try:
            body = json.dumps(body).encode("utf-8")
            headers = headers or {}
            headers.setdefault("Content-Type", "application/json")
        except (TypeError, ValueError) as e:
            logger.error("Failed to serialize body to JSON: %s", str(e))
            raise ValueError(f"Invalid JSON body: {e}") from e
    elif isinstance(body, str):
        body = body.encode("utf-8")
    elif body is not None:
        logger.warning("Body type %s is not supported, skipping", type(body).__name__)

    logger.debug("HTTP %s request to %s", method.upper(), url)
    
    req = urllib.request.Request(url, data=body, headers=headers or {}, method=method.upper())

    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read()
            status = resp.status
            resp_headers = dict(resp.headers)
            content_type = resp_headers.get("Content-Type", "")

            if "application/json" in content_type:
                response = json.loads(raw)
            else:
                response = raw.decode("utf-8", errors="replace")

            logger.debug("HTTP %s request successful with status %d", method.upper(), status)
            return status, response, resp_headers

    except urllib.error.HTTPError as e:
        raw = e.read()
        logger.warning("HTTP error %d for %s request to %s", e.code, method.upper(), url)
        return e.code, raw.decode("utf-8", errors="replace"), dict(e.headers)

    except urllib.error.URLError as e:
        logger.error("Request failed for %s: %s", url, e.reason)
        raise ConnectionError(f"Request failed: {e.reason}") from e
