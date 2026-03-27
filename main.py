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


app = FastAPI()


# =========================
# GEMINI SETUP
# =========================

gemini_api_key = os.getenv("GEMINI_API_KEY")

gemini_client = None
if gemini_api_key:
    gemini_client = genai.Client(api_key=gemini_api_key)


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


# =========================
# HEALTH CHECK
# =========================

@app.get("/health")
async def health():
    return {"status": "ok"}


# =========================
# GEMINI TEST ENDPOINT
# =========================

@app.get("/gemini-test")
async def gemini_test():
    if not gemini_client:
        return {"error": "Gemini client not initialized"}

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Say hello from Gemini"
        )

        return {
            "success": True,
            "response": response.text
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# =========================
# HELPERS
# =========================

def normalize_time(time_str: Optional[str]) -> Optional[str]:
    if not time_str:
        return None

    cleaned = time_str.strip().lower().replace(".", "")

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


def extract_title(subject: str, body: str, attachment_name: str) -> str:
    generic_subjects = {
        "",
        "flyer",
        "fw: flyer",
        "fwd: flyer",
        "attachment",
    }

    subject_clean = (subject or "").strip()
    attachment_clean = (attachment_name or "").strip()
    body_clean = (body or "").strip()

    if subject_clean.lower() not in generic_subjects:
        return subject_clean[:150]

    if attachment_clean:
        attachment_base = re.sub(r"\.[A-Za-z0-9]+$", "", attachment_clean).strip()
        if attachment_base:
            return attachment_base[:150]

    for line in body_clean.splitlines():
        line = line.strip()
        if len(line) >= 4:
            return line[:150]

    return "flyer"


def extract_date(text: str) -> Tuple[Optional[str], float]:
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
    patterns = [
        r"(?P<start>\d{1,2}(?::\d{2})?\s*[APap][Mm])\s*(?:\-|–|—|to)\s*(?P<end>\d{1,2}(?::\d{2})?\s*[APap][Mm])",
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
            return start_norm, end_norm, 0.7

    return None, None, 0.0


def extract_location(text: str) -> Tuple[Optional[str], float]:
    patterns = [
        r"(?i)\bLocation:\s*(.+)",
        r"(?i)\bWhere:\s*(.+)",
        r"(?i)\bVenue:\s*(.+)",
        r"(?i)\bAddress:\s*(.+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            location = match.group(1).strip()
            location = location.splitlines()[0].strip()
            if location:
                return location[:200], 0.6

    return None, 0.0


def looks_like_event(text: str) -> bool:
    keywords = [
        "event",
        "workshop",
        "training",
        "meeting",
        "class",
        "seminar",
        "job fair",
        "hiring event",
        "open house",
    ]

    text_lower = (text or "").lower()
    return any(k in text_lower for k in keywords)


# =========================
# MAIN PARSER
# =========================

@app.post("/parse-flyer", response_model=FlyerResponse)
async def parse_flyer(payload: FlyerRequest):
    try:
        subject = payload.subject or ""
        body = payload.body or ""
        attachment_name = payload.attachment_name or ""

        full_text = "\n".join(
            part for part in [subject.strip(), body.strip(), attachment_name.strip()] if part
        ).strip()

        title = extract_title(subject, body, attachment_name)
        description = body.strip() if body.strip() else subject.strip()

        date_value, date_conf = extract_date(full_text)
        start_time, end_time, time_conf = extract_time_range(full_text)
        location, location_conf = extract_location(full_text)

        is_event = looks_like_event(full_text) or bool(date_value)

        review_reasons = []

        if not date_value:
            review_reasons.append("No date confidently extracted.")

        if not start_time:
            review_reasons.append("No start time confidently extracted.")

        fallback_date = "2026-03-30"
        fallback_start_time = "10:00:00"
        fallback_end_time = "12:00:00"
        fallback_location = "Orlando"

        start_date = date_value if date_value else fallback_date
        end_date = start_date

        start_time_final = start_time if start_time else fallback_start_time
        end_time_final = end_time if end_time else fallback_end_time
        location_final = location if location else fallback_location

        confidence = max(date_conf, time_conf, location_conf, 0.6 if is_event else 0.2)
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
