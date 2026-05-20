"""
RFE Extractor Service
Extracts pipe takeoff data from American Stainless RFE PDFs via Gemini Flash 2.5
and appends rows to the TCG Master Google Sheet.
"""
import os
import re
import json
import base64
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import google.generativeai as genai
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("rfe-extractor")

# ---------- Config from environment ----------
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
SHEET_ID = os.environ["SHEET_ID"]
DRIVE_INBOX_FOLDER_ID = os.environ["DRIVE_INBOX_FOLDER_ID"]
DRIVE_PROCESSED_FOLDER_ID = os.environ["DRIVE_PROCESSED_FOLDER_ID"]
DRIVE_QUARANTINE_FOLDER_ID = os.environ["DRIVE_QUARANTINE_FOLDER_ID"]
SERVICE_ACCOUNT_JSON = os.environ["SERVICE_ACCOUNT_JSON"]  # full JSON as a string

# ---------- Initialize clients ----------
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = genai.GenerativeModel("gemini-2.5-flash")

creds_info = json.loads(SERVICE_ACCOUNT_JSON)
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
credentials = service_account.Credentials.from_service_account_info(creds_info, scopes=SCOPES)
sheets_service = build("sheets", "v4", credentials=credentials)
drive_service = build("drive", "v3", credentials=credentials)

# ---------- Extraction Prompt ----------
EXTRACTION_PROMPT = """You are a precise data extraction system for industrial pipe takeoff sheets from American Stainless.

The PDF you are analyzing is one of TWO possible formats. First, classify which format it is, then extract accordingly.

**FORMAT A — "Detailed Takeoff"**
Identifying signs: A table with these column headers: PL QTY | Unit | Size | Description | Type | Foreman
Header context typically shows: "NSWV", "Pipe Takeoff", "CVE Project No. XXXXX", "Estimated Bills of Material"

For Format A, extract each row:
- pl_qty (integer, preserve 0 explicitly)
- unit (string: "ea", "ft", "lb", etc.)
- size (number as string, e.g. "12.00", "0.50")
- description (full text from Description column)
- type (e.g. "tee", "elbow-90", "valve-ball", "valve-butterfly", "flange-RF", "NBGs", "bolting", "nipple")
- foreman (e.g. "Jacob Berry - HSM CW/R Bypass")
- pl_number (if present in row, e.g. "PL-0496")

**FORMAT B — "Pivot Summary"**
Identifying signs: Two columns labeled "Row Labels" and "Sum of PL QTY". Hierarchical structure where description rows are followed by indented size sub-rows.

For Format B, INHERIT the parent description down to each numeric sub-row. Each output row must be self-contained.
- pl_qty (from "Sum of PL QTY" column, integer)
- unit (leave empty "")
- size (the indented number under the description, as string)
- description (inherited from the parent description row)
- type (leave empty "" unless clearly inferable from description keywords)
- foreman (leave empty "")
- pl_number (leave empty "")

Also extract this metadata if visible:
- project_number (e.g. "21180" from "CVE Project No. 21180"; empty if not present)
- grand_total (Format B only — the Grand Total number at the bottom; null if not present)

IGNORE all handwritten content (RFE numbers in margins, dates, signatures, checkmarks, "Loaded" notes, substitution notes). RFE number and date come from the filename and are added separately.

Return ONLY valid JSON in this exact shape (no markdown, no code fences, no commentary):

{
  "format": "detailed" | "pivot" | "unknown",
  "project_number": "",
  "grand_total": null,
  "rows": [
    {
      "pl_qty": 0,
      "unit": "",
      "size": "",
      "description": "",
      "type": "",
      "foreman": "",
      "pl_number": ""
    }
  ]
}

If you cannot confidently classify the format, return "format": "unknown" and an empty rows array."""

# ---------- FastAPI app ----------
app = FastAPI(title="RFE Extractor")


class ProcessRequest(BaseModel):
    file_id: Optional[str] = None  # if None, process whole inbox folder


@app.get("/")
def health():
    return {"status": "ok", "service": "rfe-extractor", "time": datetime.now(timezone.utc).isoformat()}


@app.post("/process")
def process(req: ProcessRequest, background_tasks: BackgroundTasks):
    """Trigger processing of one file or the entire inbox folder."""
    if req.file_id:
        background_tasks.add_task(process_single_file, req.file_id)
        return {"status": "queued", "file_id": req.file_id}
    else:
        background_tasks.add_task(process_inbox)
        return {"status": "queued", "scope": "inbox"}


@app.get("/process-inbox")
def process_inbox_get(background_tasks: BackgroundTasks):
    """GET-friendly trigger — visit this URL in a browser to process all PDFs in the inbox."""
    background_tasks.add_task(process_inbox)
    return {"status": "queued", "scope": "inbox", "message": "Processing started — check sheet in 30-60 seconds"}


# ---------- Filename parser ----------
FILENAME_PATTERN = re.compile(
    r"^AS_RFE_(?P<rfe>\d+)_(?P<month>\d{2})-(?P<day>\d{2})-(?P<year>\d{4})\.pdf$",
    re.IGNORECASE,
)


def parse_filename(filename: str):
    """Parse 'AS_RFE_5903_05-17-2026.pdf' into structured fields."""
    m = FILENAME_PATTERN.match(filename)
    if not m:
        return None
    return {
        "vendor": "AS",
        "rfe_number": m.group("rfe"),
        "date_received": f"{m.group('year')}-{m.group('month')}-{m.group('day')}",
    }


# ---------- Drive helpers ----------
def list_inbox_pdfs():
    """List PDFs directly in the inbox folder (excludes Processed and Quarantine subfolders)."""
    q = (
        f"'{DRIVE_INBOX_FOLDER_ID}' in parents "
        f"and mimeType='application/pdf' "
        f"and trashed=false"
    )
    results = drive_service.files().list(q=q, fields="files(id, name)").execute()
    return results.get("files", [])


def download_pdf(file_id: str) -> bytes:
    """Download a PDF from Drive into memory."""
    request = drive_service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()


def move_file(file_id: str, dest_folder_id: str):
    """Move a Drive file to a different folder."""
    file = drive_service.files().get(fileId=file_id, fields="parents").execute()
    prev_parents = ",".join(file.get("parents", []))
    drive_service.files().update(
        fileId=file_id,
        addParents=dest_folder_id,
        removeParents=prev_parents,
        fields="id, parents",
    ).execute()


# ---------- Gemini extraction ----------
def extract_with_gemini(pdf_bytes: bytes) -> dict:
    """Send PDF to Gemini Flash 2.5 and parse the JSON response."""
    pdf_part = {"mime_type": "application/pdf", "data": pdf_bytes}
    response = GEMINI_MODEL.generate_content([EXTRACTION_PROMPT, pdf_part])
    text = response.text.strip()
    # Strip any accidental code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


# ---------- Sheets writer ----------
def append_rows_to_sheet(tab: str, rows: list):
    """Append rows to a tab in TCG Master."""
    body = {"values": rows}
    sheets_service.spreadsheets().values().append(
        spreadsheetId=SHEET_ID,
        range=f"'{tab}'!A:A",
        valueInputOption="USER_ENTERED",
        insertDataOption="INSERT_ROWS",
        body=body,
    ).execute()


def log_quarantine(filename: str, reason: str, notes: str = ""):
    """Log a problem file to the Quarantine tab."""
    append_rows_to_sheet(
        "Quarantine",
        [[filename, reason, datetime.now(timezone.utc).isoformat(), notes]],
    )


# ---------- Main processing ----------
def process_single_file(file_id: str):
    """Process one file by ID."""
    try:
        meta = drive_service.files().get(fileId=file_id, fields="id, name").execute()
        filename = meta["name"]
        log.info(f"Processing {filename} ({file_id})")

        parsed = parse_filename(filename)
        if not parsed:
            log.warning(f"Bad filename: {filename}")
            log_quarantine(filename, "bad_filename", "Did not match AS_RFE_####_MM-DD-YYYY.pdf")
            move_file(file_id, DRIVE_QUARANTINE_FOLDER_ID)
            return

        pdf_bytes = download_pdf(file_id)
        extracted = extract_with_gemini(pdf_bytes)

        if extracted.get("format") == "unknown" or not extracted.get("rows"):
            log.warning(f"Unknown format or empty rows: {filename}")
            log_quarantine(filename, "unknown_format", "Gemini could not classify or returned no rows")
            move_file(file_id, DRIVE_QUARANTINE_FOLDER_ID)
            return

        # Build sheet rows
        extracted_at = datetime.now(timezone.utc).isoformat()
        sheet_rows = []
        for row in extracted["rows"]:
            sheet_rows.append([
                parsed["vendor"],
                parsed["rfe_number"],
                parsed["date_received"],
                row.get("pl_number", ""),
                row.get("pl_qty", ""),
                row.get("unit", ""),
                row.get("size", ""),
                row.get("description", ""),
                row.get("type", ""),
                row.get("foreman", ""),
                extracted.get("project_number", ""),
                extracted.get("format", ""),
                filename,
                extracted_at,
            ])

        append_rows_to_sheet("American Stainless", sheet_rows)
        log.info(f"Wrote {len(sheet_rows)} rows from {filename}")

        move_file(file_id, DRIVE_PROCESSED_FOLDER_ID)
        log.info(f"Moved {filename} to Processed")

    except Exception as e:
        log.exception(f"Failure on file {file_id}: {e}")
        try:
            meta = drive_service.files().get(fileId=file_id, fields="name").execute()
            log_quarantine(meta["name"], "exception", str(e)[:500])
            move_file(file_id, DRIVE_QUARANTINE_FOLDER_ID)
        except Exception as inner:
            log.exception(f"Could not quarantine {file_id}: {inner}")


def process_inbox():
    """Process every PDF in the inbox folder."""
    files = list_inbox_pdfs()
    log.info(f"Inbox scan found {len(files)} PDF(s)")
    for f in files:
        process_single_file(f["id"])
