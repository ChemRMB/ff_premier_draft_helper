from pathlib import Path
from seleniumbase import SB


def fetch_fbref_html_with_browser(url: str, out_path: str | Path | None = None) -> str:
    """
    Fetch FBref HTML using a real browser in UC mode.
    Uses SB instead of Driver because SB is more reliable for UC/CAPTCHA flows.
    """
    out_path = Path(out_path) if out_path else None

    with SB(uc=True, test=False, headless=False, incognito=True) as sb:
        sb.uc_open_with_reconnect(url, 4)

        # Try CAPTCHA click if present. Safe to ignore if none exists.
        try:
            sb.uc_gui_click_captcha()
            sb.sleep(3)
        except Exception:
            pass

        sb.wait_for_element("body", timeout=15)
        html = sb.get_page_source()

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")

    return html
