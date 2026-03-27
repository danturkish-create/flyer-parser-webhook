from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import traceback

app = FastAPI()


# -----------------------------
# Request Model
# -----------------------------
class FlyerRequest(BaseModel):
    subject: Optional[str] = None
    body: Optional[str] = None
    attachment_name: Optional[str] = None


# -----------------------------
# Health Check
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -----------------------------
# Helper: Normalize time safely
# -----------------------------
def normalize_time(t: Optional[str]) -> Optional[str]:
    if not t:
        return None

    try:
        parts = t.split(":")
        if len(parts) == 2:
            return t + ":00"
        return t
    except Exception:
        return t


# -----------------------------
# Helper: Extract Title
# -----------------------------
def extract_title(subject: Optional[str], body: Optional[str]) -> Optional[str]:
    if body:
        lines = [line.strip() for line in body.splitlines() if line.strip()]
        for line in lines:
            if len(line) > 3:
                return line

    if subject:
        return subject.strip()

    return None


# -----------------------------
# Main Endpoint
# -----------------------------
@app.post("/parse-flyer")
async def parse_flyer(req: FlyerRequest):
    try:
        text = f"{req.subject or ''} {req.body or ''}".lower()

        parsed = {
            "is_event": False,
            "needs_review": False,
            "title": None,
            "start_date": None,
            "start_time": None,
            "end_date": None,
            "end_time": None,
            "location": None,
            "description": req.body,
            "confidence": 0.0,
            "review_reason": None,
        }

        # -----------------------------
        # Basic event detection
        # -----------------------------
        if "am" in text or "pm" in text or ":" in text:
            parsed["is_event"] = True
            parsed["confidence"] = 0.6

        # -----------------------------
        # Populate fields (IMPROVED TITLE)
        # -----------------------------
        if parsed["is_event"]:
            parsed["title"] = extract_title(req.subject, req.body)

            # Placeholder values (next step will replace these)
            parsed["start_date"] = "2026-03-30"
            parsed["start_time"] = "10:00"
            parsed["end_date"] = "2026-03-30"
            parsed["end_time"] = "12:00"
            parsed["location"] = "Orlando"

        # -----------------------------
        # Normalize times safely
        # -----------------------------
        parsed["start_time"] = normalize_time(parsed.get("start_time"))
        parsed["end_time"] = normalize_time(parsed.get("end_time"))

        # -----------------------------
        # Needs review check
        # -----------------------------
        if parsed["is_event"]:
            if not parsed["start_date"] or not parsed["start_time"]:
                parsed["needs_review"] = True
                parsed["review_reason"] = "Missing date or time"
                parsed["confidence"] = 0.4

        return parsed

    except Exception as e:
        print("🔥 PARSE ERROR:", str(e))
        print(traceback.format_exc())

        return {
            "is_event": False,
            "needs_review": True,
            "title": None,
            "start_date": None,
            "start_time": None,
            "end_date": None,
            "end_time": None,
            "location": None,
            "description": None,
            "confidence": 0.0,
            "review_reason": f"Server error: {str(e)}",
        }
