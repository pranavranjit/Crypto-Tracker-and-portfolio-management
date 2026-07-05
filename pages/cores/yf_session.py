"""Shared yfinance HTTP session.

Yahoo Finance aggressively rate-limits shared cloud IPs (Streamlit Community
Cloud) when requests use the default python-requests user agent. Impersonating a
real browser via curl_cffi avoids the block. Falls back to the default session
(None) if curl_cffi is unavailable, so local runs keep working.
"""

try:
    from curl_cffi import requests as _cffi_requests

    YF_SESSION = _cffi_requests.Session(impersonate="chrome")
except Exception:
    YF_SESSION = None
