import base64
import mimetypes
import os
import re
import traceback
from datetime import datetime
from typing import Optional, Tuple

from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from google.genai import types


app = FastAPI()

APP_VERSION = "gemini-ocr-v2"

gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=gemini_api_key) if gemini_api_key else None


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
            "response": response.text,
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
    }

    if normalized in generic_values:
        return True

    if normalized.endswith(".jpg") or normalized.endswith(".jpeg") or normalized.endswith(".png") or normalized.endswith(".pdf"):
        return True

    return False


def normalize_time(time_str: Optional[str]) -> Optional[str]:
    if not time_str:
        return None

    cleaned = time_str.strip()
    cleaned = cleaned.replace(".", "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"(?i)(\d)(am|pm)\b", r"\1 \2", cleaned)
    cleaned = cleaned.upper().strip()

    formats = [
        "%I:%M %p",
        "%I %p",
        "%H:%M",
        "%H%M",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(cleaned, fmt)
            return dt.strftime("%H:%M:%S")
        except ValueError:
            continue

    return None


def normalize_date(date_str: Optional[str]) -> Optional[str]:
    if not date_str:
        return None

    cleaned = date_str.strip()

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

    # Month name regex fallback
    match = re.search(
        r"(?i)\b(Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)\b\s+(\d{1,2})(?:,)?\s+(\d{2,4})",
        cleaned,
    )
    if match:
        month_text = match.group(1)[:3].title()
        day = int(match.group(2))
        year = int(match.group(3))
        if year < 100:
            year += 2000
        month = datetime.strptime(month_text, "%b").month
        return datetime(year, month, day).strftime("%Y-%m-%d")

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

    for line in body_clean.splitlines():
        line = strip_email_prefixes(line.strip())
        if not line:
            continue
        if len(line) < 4:
            continue
        if re.search(r"\b(am|pm)\b", line, re.IGNORECASE):
            continue
        if re.search(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", line):
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
        # 10AM - 2PM / 10 AM - 2 PM / 10:00AM - 2:00PM
        r"(?P<start>\d{1,2}(?::\d{2})?\s*[APap][Mm])\s*(?:\-|–|—|to)\s*(?P<end>\d{1,2}(?::\d{2})?\s*[APap][Mm])",
        # 10:00 - 14:00
        r"(?P<start>\d{1,2}:\d{2})\s*(?:\-|–|—|to)\s*(?P<end>\d{1,2}:\d{2})",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue

        start_raw = match.group("start")
        end_raw = match.group("end")

        start_norm = normalize_time(start_raw)
        end_norm = normalize_time(end_raw)

        if start_norm and end_norm:
            return start_norm, end_norm, 0.75

    return None, None, 0.0


def extract_location(text: str) -> Tuple[Optional[str], float]:
    if not text:
        return None, 0.0

    label_patterns = [
        r"(?im)^\s*Location:\s*(.+)$",
        r"(?im)^\s*Where:\s*(.+)$",
        r"(?im)^\s*Venue:\s*(.+)$",
        r"(?im)^\s*Address:\s*(.+)$",
    ]

    for pattern in label_patterns:
        match = re.search(pattern, text)
        if match:
            location = match.group(1).strip()
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
        return match.group(0).strip()[:200], 0.8

    virtual_patterns = [
        r"\bzoom\b",
        r"\bvirtual\b",
        r"\bonline\b",
        r"\bgoogle meet\b",
        r"\bmicrosoft teams\b",
        r"\bteams\b",
        r"\bwebex\b",
    ]

    for pattern in virtual_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return "Virtual", 0.75

    room_patterns = [
        r"\bRoom\s+\w+\b",
        r"\bSuite\s+\w+\b",
        r"\bBuilding\s+\w+\b",
        r"\bConference Room\s+\w+\b",
    ]

    for pattern in room_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()[:200], 0.6

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
        "open house",
        "celebration",
        "conference",
        "networking",
        "join us",
        "register",
        "fair",
    ]

    text_lower = (text or "").lower()
    return any(keyword in text_lower for keyword in event_keywords)


def guess_mime_type(attachment_name: str, content_type: str) -> str:
    if content_type:
        return content_type

    guessed, _ = mimetypes.guess_type(attachment_name or "")
    return guessed or "application/octet-stream"


def decode_attachment_base64(data: Optional[str]) -> Optional[bytes]:
    if not data:
        return None

    try:
        # Handle possible data URL prefix
        if "," in data and data.lower().startswith("data:"):
            data = data.split(",", 1)[1]
        return base64.b64decode(data)
    except Exception:
        return None


def extract_event_with_gemini(
    subject: str,
    body: str,
    attachment_name: str,
    attachment_bytes: Optional[bytes],
    mime_type: str,
) -> Optional[GeminiFlyerExtraction]:
    if not gemini_client or not attachment_bytes:
        return None

    prompt = f"""
Extract event information from this flyer attachment and the surrounding email context.

Return only structured data matching the provided schema.

Important rules:
- Prefer the flyer attachment over the email subject.
- Ignore email prefixes like FW:, FWD:, and RE: when choosing the title.
- The title should be the actual event title from the flyer, not the email subject.
- If the flyer says "10AM - 2PM", return start_time as "10:00:00" and end_time as "14:00:00".
- Return start_date and end_date in YYYY-MM-DD format when possible.
- Return times in HH:MM:SS 24-hour format when possible.
- For location, include the venue and street address if clearly visible.
- Be conservative. Do not invent details.
- confidence must be between 0.0 and 1.0.
- description should be short and useful.

Email subject:
{subject}

Email body:
{body}

Attachment name:
{attachment_name}
""".strip()

    try:
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

        if getattr(response, "parsed", None):
            return response.parsed

        return None

    except Exception as e:
        print("Gemini extraction failed:", str(e))
        return None


# =========================
# MAIN PARSER
# =========================

@app.post("/parse-flyer", response_model=FlyerResponse)
async def parse_flyer(payload: FlyerRequest):
    try:
        subject = payload.subject or ""
        body = payload.body or ""
        attachment_name = payload.attachment_name or ""
        attachment_content_base64 = payload.attachment_content_base64 or ""
        content_type = payload.content_type or ""

        subject_clean = strip_email_prefixes(subject)
        attachment_bytes = decode_attachment_base64(attachment_content_base64)
        mime_type = guess_mime_type(attachment_name, content_type)

        gemini_result = extract_event_with_gemini(
            subject=subject_clean,
            body=body,
            attachment_name=attachment_name,
            attachment_bytes=attachment_bytes,
            mime_type=mime_type,
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

        # Local fallback values from plain text/email context
        title = extract_title(subject_clean, body, attachment_name)
        description = body.strip() if body.strip() else subject_clean.strip()

        date_value, date_conf = extract_date(full_text)
        start_time, end_time, time_conf = extract_time_range(full_text)
        location, location_conf = extract_location(full_text)

        gemini_conf = 0.0
        gemini_review_reason = None
        gemini_end_date = None

        if gemini_result:
            if gemini_result.title and not is_generic_title(gemini_result.title):
                title = gemini_result.title.strip()[:150]

            if gemini_result.description:
                description = gemini_result.description.strip()

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
                location = gemini_result.location.strip()[:200]
                location_conf = max(location_conf, 0.9)

            is_event = bool(gemini_result.is_event) or bool(date_value) or looks_like_event(full_text)
            gemini_conf = float(gemini_result.confidence or 0.0)
            gemini_review_reason = gemini_result.review_reason
        else:
            is_event = looks_like_event(full_text) or bool(date_value)

        # Final cleanup on title in case subject was junk
        title = strip_email_prefixes(title)
        if is_generic_title(title):
            attachment_base = re.sub(r"\.[A-Za-z0-9]+$", "", attachment_name or "").strip()
            if attachment_base and not is_generic_title(attachment_base):
                title = attachment_base[:150]
            elif gemini_result and gemini_result.title:
                title = strip_email_prefixes(gemini_result.title)[:150]
            else:
                title = "flyer"

        review_reasons = []

        if not date_value:
            review_reasons.append("No date confidently extracted.")

        if not start_time:
            review_reasons.append("No start time confidently extracted.")

        if gemini_review_reason:
            review_reasons.append(gemini_review_reason)

        # Stable fallbacks for schema consistency
        fallback_date = "2026-03-30"
        fallback_start_time = "10:00:00"
        fallback_end_time = "12:00:00"
        fallback_location = "Orlando"

        start_date = date_value if date_value else fallback_date
        end_date = gemini_end_date if gemini_end_date else start_date

        start_time_final = start_time if start_time else fallback_start_time
        end_time_final = end_time if end_time else fallback_end_time
        location_final = location if location else fallback_location

        confidence = max(date_conf, time_conf, location_conf, gemini_conf, 0.6 if is_event else 0.2)
        confidence = round(min(confidence, 1.0), 2)

        needs_review = len(review_reasons) > 0
        review_reason = " ".join(review_reasons) if review_reasons else None

        return FlyerResponse(
            is_event=is_event,
            needs_review=needs_review,
            title=title,
            start_date=start_date,
            start_time=start_time_final,
            end_date=end_date,
            end_time=end_time_final,
            location=location_final,
            description=description,
            confidence=confidence,
            review_reason=review_reason,
        )

    except Exception as e:
        print("ERROR in /parse-flyer")
        print(str(e))
        traceback.print_exc()

        return FlyerResponse(
            is_event=False,
            needs_review=True,
            title="flyer",
            start_date="2026-03-30",
            start_time="10:00:00",
            end_date="2026-03-30",
            end_time="12:00:00",
            location="Orlando",
            description="Parsing failed.",
            confidence=0.0,
            review_reason=f"Parser exception: {str(e)}",
        )
