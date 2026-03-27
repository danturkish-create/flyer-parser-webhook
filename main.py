from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import json
from google import genai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

@app.get("/")
async def root():
    return {"status": "ok"}

@app.post("/parse-flyer")
async def parse_flyer(request: Request):
    try:
        data = await request.json()

        subject = data.get("subject", "")
        body = data.get("body", "")
        attachment_name = data.get("attachment_name", "")

        prompt = f"""
You extract structured event data from emails about public events.

Return ONLY valid JSON with this exact schema:

{{
  "is_event": true,
  "title": "",
  "start_date": "",
  "start_time": "",
  "end_date": "",
  "end_time": "",
  "all_day": false,
  "location": "",
  "description": "",
  "confidence": 0,
  "needs_review": false,
  "review_reason": ""
}}

Rules:
- If this is not clearly an event, set "is_event" to false.
- If details are missing or ambiguous, set "needs_review" to true.
- Confidence must be an integer from 0 to 100.
- Do not include markdown fences.
- Do not include any text before or after the JSON.

INPUT:
Subject: {subject}
Body: {body}
Attachment: {attachment_name}
"""

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
        )

        content = response.text.strip()

        # Validate that Gemini returned JSON
        parsed = json.loads(content)

       if parsed.get("start_time") and ":" in parsed["start_time"]:
    if len(parsed["start_time"].split(":")) == 2:
        parsed["start_time"] = parsed["start_time"] + ":00"
        
if parsed.get("end_time") and len(parsed["end_time"]) == 5:
    parsed["end_time"] = parsed["end_time"] + ":00"

        return JSONResponse(content=parsed)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
