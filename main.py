import base64
import logging
import mimetypes
import os
import re
import traceback
import uuid
from datetime import datetime
from typing import Optional, Tuple, Any, List

from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from google.genai import types


app = FastAPI()

APP_VERSION = "gemini-flyer-parser-prod-v2"

gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=gemini_api_key) if gemini_api_key else None


# =========================
# LOGGING
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("flyer_parser")


# =========================
# MODELS
# =========================

class FlyerRequest(BaseModel):
    subject: Optional[str] = ""
    body: Optional[str] = ""
    attachment_name: Optional[str] = ""
    attachment_content_base64: Optional[str] = ""
    content_type: Optional[str] = ""


class FlyerResponse(BaseModel):
    is_event: bool
    needs_review: bool
    title: str
    start_date: Optional[str]
    start_time: Optional[str]
    end_date: Optional[str]
    end_time: Optional[str]
    location: Optional[str]
    description: str
    confidence: float
    review_reason: Optional[str]


class GeminiFlyerExtraction(BaseModel):
    is_event: bool = False
    title: Optional[str] = None
    start_date: Optional[str] = None
    start_time: Optional[str] = None
    end_date: Optional[str] = None
    end_time: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None
    confidence: Optional[float] = 0.0
    review_reason: Optional[str] = None

    # New internal-only guidance fields
    is_non_event_notice: bool = False
    has_multiple_sessions: bool = False
    session_count: Optional[int] = None
    title_preserve_exact: Optional[str] = None


# =========================
# HEALTH / DEBUG
# =========================

@app.get("/")
async def root():
    return {"status": "ok", "app_version": APP_VERSION}


@app.get("/health")
async def health():
    return {"status": "ok", "app_version": APP_VERSION}


@app.get("/version")
async def version():
    return {"app_version": APP_VERSION}


@app.get("/gemini-test")
async def gemini_test():
    if not gemini_client:
        return {
            "success": False,
            "error": "Gemini client not initialized",
            "app_version": APP_VERSION,
        }

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Reply with exactly: Gemini is connected."
        )

        return {
            "success": True,
            "response": getattr(response, "text", ""),
            "app_version": APP_VERSION,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "app_version": APP_VERSION,
        }


# =========================
# HELPERS
# =========================

def make_request_id() -> str:
    return uuid.uuid4().hex[:12]


def log_info(request_id: str, message: str, **kwargs: Any) -> None:
    if kwargs:
        logger.info("[%s] %s | %s", request_id, message, kwargs)
    else:
        logger.info("[%s] %s", request_id, message)


def log_warning(request_id: str, message: str, **kwargs: Any) -> None:
    if kwargs:
        logger.warning("[%s] %s | %s", request_id, message, kwargs)
    else:
        logger.warning("[%s] %s", request_id, message)


def log_error(request_id: str, message: str, **kwargs: Any) -> None:
    if kwargs:
        logger.error("[%s] %s | %s", request_id, message, kwargs)
    else:
        logger.error("[%s] %s", request_id, message)


def compact_whitespace(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def strip_email_prefixes(text: str) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    cleaned = re.sub(r"^(?:(?:fw|fwd|re)\s*:\s*)+", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def is_generic_title(text: str) -> bool:
    if not text:
        return True

    normalized = strip_email_prefixes(text).strip().lower()

    generic_values = {
        "",
        "flyer",
        "test flyer",
        "attachment",
        "image",
        "photo",
        "scan",
        "document",
        "event",
        "notice",
        "announcement",
        "start_date",
        "<html><head>",
    }

    if normalized in generic_values:
        return True

    if normalized.endswith((".jpg", ".jpeg", ".png", ".pdf", ".gif", ".bmp", ".tif", ".tiff", ".webp")):
        return True

    return False


def normalize_time(time_str: Optional[str]) -> Optional[str]:
    if not time_str:
        return None

    cleaned = compact_whitespace(str(time_str))
    if not cleaned:
        return None

    cleaned = cleaned.replace(".", "")
    cleaned = cleaned.replace("–", "-").replace("—", "-")
    cleaned = re.sub(r"(?i)(\d)(am|pm)\b", r"\1 \2", cleaned)
    cleaned_upper = cleaned.upper().strip()

    special_map = {
        "NOON": "12:00:00",
        "MIDNIGHT": "00:00:00",
    }
    if cleaned_upper in special_map:
        return special_map[cleaned_upper]

    formats = [
        "%H:%M:%S",
        "%I:%M:%S %p",
        "%I:%M %p",
        "%I %p",
        "%H:%M",
        "%H%M",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(cleaned_upper, fmt)
            return dt.strftime("%H:%M:%S")
        except ValueError:
            continue

    simple_match = re.fullmatch(r"(\d{1,2})\s*([AP]M)", cleaned_upper)
    if simple_match:
        try:
            dt = datetime.strptime(f"{simple_match.group(1)} {simple_match.group(2)}", "%I %p")
            return dt.strftime("%H:%M:%S")
        except ValueError:
            pass

    return None


def normalize_date(date_str: Optional[str]) -> Optional[str]:
    if not date_str:
        return None

    cleaned = compact_whitespace(str(date_str))
    if not cleaned:
        return None

    formats = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m/%d/%y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%B %d %Y",
        "%b %d %Y",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(cleaned, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

    match = re.search(
        r"(?i)\b(Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)\b\s+(\d{1,2})(?:,)?\s+(\d{2,4})",
        cleaned,
    )
    if match:
        try:
            month_text = match.group(1)[:3].title()
            day = int(match.group(2))
            year = int(match.group(3))
            if year < 100:
                year += 2000
            month = datetime.strptime(month_text, "%b").month
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except Exception:
            return None

    return None


def extract_title(subject: str, body: str, attachment_name: str) -> str:
    subject_clean = strip_email_prefixes(subject or "")
    body_clean = (body or "").strip()
    attachment_clean = (attachment_name or "").strip()

    if subject_clean and not is_generic_title(subject_clean):
        return subject_clean[:150]

    if attachment_clean:
        attachment_base = re.sub(r"\.[A-Za-z0-9]+$", "", attachment_clean).strip()
        attachment_base = strip_email_prefixes(attachment_base)
        if attachment_base and not is_generic_title(attachment_base):
            return attachment_base[:150]

    # Better body title detection
    for line in body_clean.splitlines():
        line = strip_email_prefixes(line.strip())
        if not line:
            continue
        if len(line) < 4:
            continue
        if re.search(r"\b(am|pm|noon|midnight)\b", line, re.IGNORECASE):
            continue
        if re.search(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", line):
            continue
        if line.lower().startswith("<html"):
            continue
        if len(line) > 120:
            continue
        return line[:150]

    return "flyer"


def extract_date(text: str) -> Tuple[Optional[str], float]:
    if not text:
        return None, 0.0

    patterns = [
        r"(?P<month>\b(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)\b)\s+(?P<day>\d{1,2})(?:,)?\s+(?P<year>\d{2,4})",
        r"(?P<month>\d{1,2})/(?P<day>\d{1,2})/(?P<year>\d{2,4})",
        r"(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue

        try:
            parts = match.groupdict()
            month_raw = parts["month"]
            day = int(parts["day"])
            year = int(parts["year"])

            if isinstance(month_raw, str) and month_raw.isalpha():
                month_lookup = month_raw[:3].title()
                month = datetime.strptime(month_lookup, "%b").month
            else:
                month = int(month_raw)

            if year < 100:
                year += 2000

            dt = datetime(year, month, day)
            return dt.strftime("%Y-%m-%d"), 0.8

        except Exception:
            continue

    return None, 0.0


def extract_time_range(text: str) -> Tuple[Optional[str], Optional[str], float]:
    if not text:
        return None, None, 0.0

    patterns = [
        r"(?P<start>\b(?:Noon|Midnight)\b|\d{1,2}(?::\d{2})?\s*[APap][Mm])\s*(?:\-|–|—|to)\s*(?P<end>\b(?:Noon|Midnight)\b|\d{1,2}(?::\d{2})?\s*[APap][Mm])",
        r"(?P<start>\b(?:Noon|Midnight)\b|\d{1,2}(?::\d{2})?\s*[APap][Mm])\s*(?:\-|–|—|to)\s*(?P<end>\d{1,2}(?::\d{2})?)",
        r"(?P<start>\d{1,2}:\d{2})\s*(?:\-|–|—|to)\s*(?P<end>\d{1,2}:\d{2})",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue

        start_raw = match.group("start")
        end_raw = match.group("end")

        if re.fullmatch(r"\d{1,2}(?::\d{2})?", end_raw) and re.search(r"(?i)\b(am|pm)\b", start_raw):
            meridiem = re.search(r"(?i)\b(am|pm)\b", start_raw)
            if meridiem:
                end_raw = f"{end_raw} {meridiem.group(1)}"

        start_norm = normalize_time(start_raw)
        end_norm = normalize_time(end_raw)

        if start_norm and end_norm:
            return start_norm, end_norm, 0.75

    return None, None, 0.0


def extract_start_only_time(text: str) -> Tuple[Optional[str], float]:
    if not text:
        return None, 0.0

    patterns = [
        r"(?i)\bstarts?\s+at\s+(?P<start>\d{1,2}(?::\d{2})?\s*[ap]m|noon|midnight)\b",
        r"(?i)\bbegins?\s+at\s+(?P<start>\d{1,2}(?::\d{2})?\s*[ap]m|noon|midnight)\b",
        r"(?i)\bstart\s*time\s*[:\-]?\s*(?P<start>\d{1,2}(?::\d{2})?\s*[ap]m|noon|midnight)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            start_norm = normalize_time(match.group("start"))
            if start_norm:
                return start_norm, 0.65

    return None, 0.0


def extract_location(text: str) -> Tuple[Optional[str], float]:
    if not text:
        return None, 0.0

    label_patterns = [
        r"(?im)^\s*Location:\s*(.+)$",
        r"(?im)^\s*Where:\s*(.+)$",
        r"(?im)^\s*Venue:\s*(.+)$",
        r"(?im)^\s*Address:\s*(.+)$",
        r"(?im)^\s*Place:\s*(.+)$",
    ]

    for pattern in label_patterns:
        match = re.search(pattern, text)
        if match:
            location = compact_whitespace(match.group(1))
            if location:
                return location[:200], 0.85

    address_pattern = (
        r"\b\d{1,6}\s+[A-Za-z0-9.\-#'\s]+"
        r"\s(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|Way|Parkway|Pkwy|Trail|Trl)"
        r"(?:,\s*[A-Za-z.\-'\s]+)?"
        r"(?:,\s*[A-Z]{2})?"
        r"(?:\s+\d{5})?\b"
    )

    match = re.search(address_pattern, text, re.IGNORECASE)
    if match:
        return compact_whitespace(match.group(0))[:200], 0.8

    # Venue-only / room-only fallback
    venue_patterns = [
        r"(?im)^\s*(?:Where|Venue|Location|Place)\s*[:\-]?\s*(.+)$",
        r"(?im)^\s*(?:Room|Hall|Center|Lab|Library|Conference Room|Media Center|Training Hub)[^\n]{0,120}$",
    ]
    for pattern in venue_patterns:
        match = re.search(pattern, text)
        if match:
            value = compact_whitespace(match.group(1) if match.groups() else match.group(0))
            if value and len(value) >= 3:
                return value[:200], 0.6

    return None, 0.0


def looks_like_event(text: str) -> bool:
    event_keywords = [
        "event",
        "workshop",
        "training",
        "meeting",
        "class",
        "seminar",
        "webinar",
        "job fair",
        "resource fair",
        "hiring event",
        "hiring day",
        "open house",
        "clinic",
        "bootcamp",
        "session",
        "orientation",
        "fair",
    ]

    text_lower = (text or "").lower()
    return any(keyword in text_lower for keyword in event_keywords)


def looks_like_non_event_notice(text: str) -> bool:
    if not text:
        return False

    text_lower = text.lower()

    non_event_phrases = [
        "internal awareness only",
        "no public event",
        "no registration",
        "no attendees",
        "office closed",
        "building access unavailable",
        "maintenance is in progress",
        "notice only",
        "for internal use only",
        "closure notice",
    ]

    hits = sum(1 for phrase in non_event_phrases if phrase in text_lower)
    return hits >= 2


def count_distinct_date_mentions(text: str) -> int:
    if not text:
        return 0

    patterns = [
        r"\b(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)\s+\d{1,2}(?:,\s*\d{2,4})?\b",
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        r"\b\d{4}-\d{1,2}-\d{1,2}\b",
    ]

    matches = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, text, flags=re.IGNORECASE))

    normalized = set(compact_whitespace(m).lower() for m in matches if m)
    return len(normalized)


def looks_like_multi_session(text: str) -> bool:
    if not text:
        return False

    text_lower = text.lower()

    trigger_terms = [
        "orientation",
        "session",
        "session 1",
        "session 2",
        "day 1",
        "day 2",
        "day 3",
        "workshop days",
        "attendance required all days",
        "multiple sessions",
        "schedule",
    ]

    has_trigger_term = any(term in text_lower for term in trigger_terms)
    date_count = count_distinct_date_mentions(text)

    return has_trigger_term and date_count >= 2


def looks_like_real_file_bytes(data: bytes) -> bool:
    if not data or len(data) < 4:
        return False

    signatures = [
        b"\xff\xd8\xff",
        b"\x89PNG\r\n\x1a\n",
        b"%PDF",
        b"GIF87a",
        b"GIF89a",
        b"BM",
        b"II*\x00",
        b"MM\x00*",
        b"RIFF",
    ]

    return any(data.startswith(sig) for sig in signatures)


def looks_like_base64_text(data: bytes) -> bool:
    if not data:
        return False

    try:
        text = data.decode("utf-8", errors="strict").strip()
    except Exception:
        return False

    if not text:
        return False

    compact = re.sub(r"\s+", "", text)

    if re.fullmatch(r"[A-Za-z0-9+/=]+", compact) is None:
        return False

    return len(compact) > 100


def detect_mime_from_bytes(data: Optional[bytes]) -> Optional[str]:
    if not data or len(data) < 4:
        return None

    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"%PDF"):
        return "application/pdf"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    if data.startswith(b"BM"):
        return "image/bmp"
    if data.startswith(b"II*\x00") or data.startswith(b"MM\x00*"):
        return "image/tiff"
    if data.startswith(b"RIFF") and b"WEBP" in data[:32]:
        return "image/webp"

    return None


def guess_mime_type(attachment_name: str, content_type: str, attachment_bytes: Optional[bytes]) -> str:
    byte_mime = detect_mime_from_bytes(attachment_bytes)
    if byte_mime:
        return byte_mime

    if content_type:
        return content_type

    guessed, _ = mimetypes.guess_type(attachment_name or "")
    return guessed or "application/octet-stream"


def decode_attachment_base64(data: Optional[str], request_id: str) -> Tuple[Optional[bytes], str]:
    if not data:
        return None, "missing"

    try:
        working = data.strip()

        if not working:
            return None, "empty"

        if working.startswith('"') and working.endswith('"'):
            working = working[1:-1]

        if working.lower().startswith("data:") and "," in working:
            working = working.split(",", 1)[1]

        first_pass = base64.b64decode(working, validate=False)
        log_info(request_id, "first base64 decode complete", first_pass_len=len(first_pass))

        if looks_like_real_file_bytes(first_pass):
            return first_pass, "single_decode_real_bytes"

        if looks_like_base64_text(first_pass):
            log_info(request_id, "first decode still looks like base64 text, attempting second decode")
            compact = re.sub(r"\s+", "", first_pass.decode("utf-8", errors="strict"))
            second_pass = base64.b64decode(compact, validate=False)
            log_info(request_id, "second base64 decode complete", second_pass_len=len(second_pass))

            if looks_like_real_file_bytes(second_pass):
                return second_pass, "double_decode_real_bytes"

            return second_pass, "double_decode_non_signature_bytes"

        return first_pass, "single_decode_non_signature_bytes"

    except Exception as e:
        log_warning(request_id, "decode_attachment_base64 failed", error=str(e))
        return None, f"decode_failed: {str(e)}"


def parse_gemini_response(parsed: Any) -> Optional[GeminiFlyerExtraction]:
    if parsed is None:
        return None

    if isinstance(parsed, GeminiFlyerExtraction):
        return parsed

    if isinstance(parsed, dict):
        try:
            return GeminiFlyerExtraction(**parsed)
        except Exception:
            return None

    try:
        if hasattr(parsed, "model_dump"):
            return GeminiFlyerExtraction(**parsed.model_dump())
    except Exception:
        pass

    return None


def simplify_gemini_error(error_text: str) -> str:
    if not error_text:
        return "Gemini extraction failed."

    upper = error_text.upper()

    if "429" in upper or "RESOURCE_EXHAUSTED" in upper or "QUOTA" in upper:
        return "Gemini quota exceeded; manual review required."

    if "DEADLINE_EXCEEDED" in upper or "TIMEOUT" in upper:
        return "Gemini timed out; manual review required."

    return "Gemini extraction failed; manual review required."


def extract_event_with_gemini(
    request_id: str,
    subject: str,
    body: str,
    attachment_name: str,
    attachment_bytes: Optional[bytes],
    mime_type: str,
) -> Tuple[Optional[GeminiFlyerExtraction], Optional[str], Optional[str]]:
    if not gemini_client:
        log_warning(request_id, "Gemini client missing")
        return None, "Gemini client not initialized.", "Gemini unavailable; manual review required."

    if not attachment_bytes:
        log_warning(request_id, "No attachment bytes available for Gemini")
        return None, "No attachment bytes available.", "Attachment could not be decoded."

    prompt = f"""
Extract event information from this flyer attachment and the surrounding email context.

Return only structured data matching the provided schema.

Important rules:
- Prefer the flyer attachment over the email subject.
- Ignore email prefixes like FW:, FWD:, and RE: when choosing the title.
- Preserve the visible flyer title as exactly as practical.
- If the flyer title is bilingual, keep both languages when clearly shown.
- Do not use generic headers like "flyer", "event", or a file name as the event title if a better title exists on the flyer.
- If the flyer says "10AM - 2PM", return start_time as "10:00:00" and end_time as "14:00:00".
- If the flyer says "Noon", return "12:00:00".
- Return start_date and end_date in YYYY-MM-DD format when possible.
- Return times in HH:MM:SS 24-hour format when possible.
- For location, include the venue and street address if clearly visible. Venue-only locations are acceptable if that is all the flyer provides.
- Be conservative. Do not invent details.
- confidence must be between 0.0 and 1.0.
- description should be short and useful.
- If this is not actually an event flyer, set is_event to false and is_non_event_notice to true when appropriate.
- If the flyer appears to contain multiple distinct sessions, dates, or schedule blocks, set has_multiple_sessions to true.
- A flyer with multiple distinct sessions should not be flattened into one continuous event.
- If key details are unclear, explain briefly in review_reason.
- If the image is blank or contains no useful flyer content, set is_event to false.

Email subject:
{subject}

Email body:
{body}

Attachment name:
{attachment_name}
""".strip()

    try:
        log_info(
            request_id,
            "calling Gemini",
            attachment_name=attachment_name,
            mime_type=mime_type,
            byte_len=len(attachment_bytes),
        )

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                prompt,
                types.Part.from_bytes(
                    data=attachment_bytes,
                    mime_type=mime_type,
                ),
            ],
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_schema=GeminiFlyerExtraction,
            ),
        )

        parsed = parse_gemini_response(getattr(response, "parsed", None))
        if parsed:
            log_info(
                request_id,
                "Gemini parsed successfully",
                is_event=parsed.is_event,
                title=parsed.title,
                start_date=parsed.start_date,
                start_time=parsed.start_time,
                end_date=parsed.end_date,
                end_time=parsed.end_time,
                location=parsed.location,
                confidence=parsed.confidence,
                review_reason=parsed.review_reason,
                is_non_event_notice=parsed.is_non_event_notice,
                has_multiple_sessions=parsed.has_multiple_sessions,
                session_count=parsed.session_count,
            )
            return parsed, None, None

        raw_text = getattr(response, "text", None)
        log_warning(
            request_id,
            "Gemini returned no parsed result",
            raw_text_preview=(raw_text[:500] if raw_text else None),
        )
        return None, "Gemini returned no structured result.", "Gemini returned no structured result; manual review required."

    except Exception as e:
        raw_error = str(e)
        clean_error = simplify_gemini_error(raw_error)
        log_warning(request_id, "Gemini extraction failed", error=raw_error)
        return None, raw_error, clean_error


def build_review_response(
    *,
    title: str = "flyer",
    description: str = "",
    is_event: bool = False,
    start_date: str = "",
    start_time: str = "",
    end_date: str = "",
    end_time: str = "",
    location: str = "",
    confidence: float = 0.0,
    review_reason: Optional[str] = None,
) -> FlyerResponse:
    return FlyerResponse(
        is_event=is_event,
        needs_review=True,
        title=title or "flyer",
        start_date=start_date or "",
        start_time=start_time or "",
        end_date=end_date or "",
        end_time=end_time or "",
        location=location or "",
        description=description or "",
        confidence=round(max(0.0, min(confidence, 1.0)), 2),
        review_reason=review_reason or "Needs manual review.",
    )


def dedupe_review_reasons(reasons: List[str]) -> str:
    seen = set()
    ordered = []
    for reason in reasons:
        cleaned = compact_whitespace(reason)
        if not cleaned:
            continue
        lower = cleaned.lower()
        if lower not in seen:
            seen.add(lower)
            ordered.append(cleaned)
    return " ".join(ordered)


# =========================
# MAIN PARSER
# =========================

@app.post("/parse-flyer", response_model=FlyerResponse)
async def parse_flyer(payload: FlyerRequest):
    request_id = make_request_id()

    try:
        # Request-local values only
        subject = payload.subject or ""
        body = payload.body or ""
        attachment_name = payload.attachment_name or ""
        attachment_content_base64 = payload.attachment_content_base64 or ""
        content_type = payload.content_type or ""

        log_info(
            request_id,
            "/parse-flyer called",
            subject_preview=subject[:200],
            attachment_name=attachment_name,
            content_type=content_type,
            attachment_b64_len=len(attachment_content_base64),
        )

        subject_clean = strip_email_prefixes(subject)
        description_default = compact_whitespace(body) or subject_clean or ""
        title = extract_title(subject_clean, body, attachment_name)

        attachment_bytes, decode_status = decode_attachment_base64(attachment_content_base64, request_id)
        mime_type = guess_mime_type(attachment_name, content_type, attachment_bytes)

        log_info(
            request_id,
            "attachment processing complete",
            decode_status=decode_status,
            decoded_byte_len=(len(attachment_bytes) if attachment_bytes else 0),
            resolved_mime_type=mime_type,
        )

        full_text = "\n".join(
            part
            for part in [
                subject_clean.strip(),
                body.strip(),
                attachment_name.strip(),
            ]
            if part
        ).strip()

        # Fallback extraction from email/body context only
        date_value, date_conf = extract_date(full_text)
        start_time, end_time, time_conf = extract_time_range(full_text)

        if not start_time:
            start_only_time, start_only_conf = extract_start_only_time(full_text)
            if start_only_time:
                start_time = start_only_time
                time_conf = max(time_conf, start_only_conf)

        location, location_conf = extract_location(full_text)

        gemini_result = None
        gemini_error_raw = None
        gemini_error_clean = None

        if attachment_bytes:
            gemini_result, gemini_error_raw, gemini_error_clean = extract_event_with_gemini(
                request_id=request_id,
                subject=subject_clean,
                body=body,
                attachment_name=attachment_name,
                attachment_bytes=attachment_bytes,
                mime_type=mime_type,
            )
        else:
            gemini_error_clean = "Attachment could not be decoded."

        gemini_conf = 0.0
        gemini_review_reason = None
        gemini_end_date = None

        # Internal review flags
        multiple_sessions_detected = looks_like_multi_session(full_text)
        non_event_notice_detected = looks_like_non_event_notice(full_text)
        blank_or_junk_detected = False

        if gemini_result:
            if gemini_result.title_preserve_exact and not is_generic_title(gemini_result.title_preserve_exact):
                title = compact_whitespace(gemini_result.title_preserve_exact)[:150]
            elif gemini_result.title and not is_generic_title(gemini_result.title):
                title = compact_whitespace(gemini_result.title)[:150]

            description = compact_whitespace(gemini_result.description or description_default)

            gem_date = normalize_date(gemini_result.start_date)
            if gem_date:
                date_value = gem_date
                date_conf = max(date_conf, 0.95)

            gem_end_date = normalize_date(gemini_result.end_date)
            if gem_end_date:
                gemini_end_date = gem_end_date

            gem_start_time = normalize_time(gemini_result.start_time)
            if gem_start_time:
                start_time = gem_start_time
                time_conf = max(time_conf, 0.95)

            gem_end_time = normalize_time(gemini_result.end_time)
            if gem_end_time:
                end_time = gem_end_time
                time_conf = max(time_conf, 0.95)

            if gemini_result.location:
                location = compact_whitespace(gemini_result.location)[:200]
                # Venue-only locations are acceptable
                location_conf = max(location_conf, 0.6 if "," not in location else 0.9)

            gemini_conf = float(gemini_result.confidence or 0.0)
            gemini_review_reason = gemini_result.review_reason

            multiple_sessions_detected = multiple_sessions_detected or bool(gemini_result.has_multiple_sessions)
            non_event_notice_detected = non_event_notice_detected or bool(gemini_result.is_non_event_notice)

            is_event = bool(gemini_result.is_event) or bool(date_value) or looks_like_event(full_text)
        else:
            description = description_default
            is_event = looks_like_event(full_text) or bool(date_value)

        # Blank / junk protection
        if not gemini_result and not body.strip() and not subject_clean.strip():
            if attachment_name and re.search(r"\b(blank|empty|white|scan)\b", attachment_name, re.IGNORECASE):
                blank_or_junk_detected = True

        # Non-event notice override
        if non_event_notice_detected:
            is_event = False

        title = strip_email_prefixes(title)
        if is_generic_title(title):
            attachment_base = re.sub(r"\.[A-Za-z0-9]+$", "", attachment_name or "").strip()
            attachment_base = strip_email_prefixes(attachment_base)
            if attachment_base and not is_generic_title(attachment_base):
                title = attachment_base[:150]
            elif gemini_result and gemini_result.title and not is_generic_title(gemini_result.title):
                title = compact_whitespace(gemini_result.title)[:150]
            else:
                title = "flyer"

        review_reasons: List[str] = []

        if decode_status in {"missing", "empty"}:
            review_reasons.append("Attachment content is missing or empty.")
        elif decode_status.startswith("decode_failed"):
            review_reasons.append("Attachment could not be decoded.")
        elif decode_status in {"single_decode_non_signature_bytes", "double_decode_non_signature_bytes"}:
            review_reasons.append("Attachment decoded, but file signature could not be confirmed.")

        if gemini_error_clean:
            review_reasons.append(gemini_error_clean)

        if multiple_sessions_detected:
            review_reasons.append("Multiple distinct event sessions detected; review before calendar creation.")

        if is_event and not date_value:
            review_reasons.append("No date confidently extracted.")

        if is_event and not start_time:
            review_reasons.append("No start time confidently extracted.")

        if is_event and not location:
            review_reasons.append("No location confidently extracted.")

        if is_event and start_time and not end_time:
            review_reasons.append("No end time confidently extracted.")

        if gemini_review_reason:
            review_reasons.append(gemini_review_reason)

        start_date_final = date_value or ""
        end_date_final = gemini_end_date or start_date_final or ""
        start_time_final = start_time or ""
        end_time_final = end_time or ""
        location_final = location or ""

        # Confidence handling
        confidence_candidates = [
            date_conf,
            time_conf,
            location_conf,
            gemini_conf,
            0.6 if is_event else 0.2,
        ]
        confidence = round(min(max(confidence_candidates), 0.99), 2)

        # Final routing policy
        if blank_or_junk_detected and not is_event:
            needs_review = False
            review_reason = None
        elif non_event_notice_detected and not is_event and not gemini_error_clean:
            needs_review = False
            review_reason = None
        else:
            needs_review = len(review_reasons) > 0
            review_reason = dedupe_review_reasons(review_reasons) if review_reasons else None

        # If no signal at all and looks like junk, suppress event status
        if not date_value and not start_time and not location and not looks_like_event(full_text) and non_event_notice_detected:
            is_event = False

        # If Gemini unavailable and we only know nothing, be conservative
        if gemini_error_clean and not date_value and not start_time and not location and not looks_like_event(full_text):
            is_event = False

        log_info(
            request_id,
            "final result",
            is_event=is_event,
            title=title,
            start_date=start_date_final,
            start_time=start_time_final,
            end_date=end_date_final,
            end_time=end_time_final,
            location=location_final,
            needs_review=needs_review,
            confidence=confidence,
            review_reason=review_reason,
            multiple_sessions_detected=multiple_sessions_detected,
            non_event_notice_detected=non_event_notice_detected,
        )

        return FlyerResponse(
            is_event=is_event,
            needs_review=needs_review,
            title=title,
            start_date=start_date_final,
            start_time=start_time_final,
            end_date=end_date_final,
            end_time=end_time_final,
            location=location_final,
            description=description,
            confidence=confidence,
            review_reason=review_reason,
        )

    except Exception as e:
        log_error(request_id, "ERROR in /parse-flyer", error=str(e))
        traceback.print_exc()

        safe_title = "flyer"
        safe_description = compact_whitespace(getattr(payload, "body", "") or getattr(payload, "subject", "") or "")

        return build_review_response(
            title=safe_title,
            description=safe_description or "Parsing failed.",
            is_event=False,
            start_date="",
            start_time="",
            end_date="",
            end_time="",
            location="",
            confidence=0.0,
            review_reason="Parser exception; manual review required.",
        )
