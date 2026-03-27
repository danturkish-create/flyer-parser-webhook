from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

Return ONLY valid JSON.

Schema:
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

INPUT:
Subject: {subject}
Body: {body}
Attachment: {attachment_name}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        content = response.choices[0].message.content

        return JSONResponse(content={"raw_response": content})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
