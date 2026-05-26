"""
RFE Extractor Service v4.1
==========================

Same as v4 but RENDER_SCALE dropped from 200/72 to 150/72.
Reason: 200 DPI causes 504 Deadline Exceeded even on the per-page classify
call. 150 DPI is the operational ceiling for Gemini Flash 2.5 on these PDFs.

Priority order (unchanged from v4):
  Tier 1 (must be correct):
    - RFE number (from filename)
    - Date received (RAW mode)
    - Size — must be correct, no hallucinated/dropped sizes
    - Description — must be correct, no truncation
  Tier 2 (acceptable as-is):
    - Quantities — manual correction in master sheet is accepted workflow
  Tier 3 (new in v4):
    - Connex + Shelf location columns
    - AI-assisted bulk location assignment endpoints
    - Reprocess-quarantine endpoint
    - OCR/AI diagnostic columns pushed to far right of sheet

Architectural notes:
  - Format B Pass 1 is page-by-page with parent-description carryover
  - Anti-hallucination prompt rules + post-process filter on invalid sizes
  - Two-pass cross-check validation (Pass 1 sums vs Pass 2 subtotals)
  - Sheet schema: item-critical block | location block | meta block

Manual Google Sheets setup notes at the bottom of this file.
"""

import os
import re
import json
import uuid
import logging
import io
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

import google.generativeai as genai
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

import pypdfium2 as pdfium

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("rfe-extractor")

# ---------- Config from environment ----------
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
SHEET_ID = os.environ["SHEET_ID"]
DRIVE_INBOX_FOLDER_ID = os.environ["DRIVE_INBOX_FOLDER_ID"]
DRIVE_PROCESSED_FOLDER_ID = os.environ["DRIVE_PROCESSED_FOLDER_ID"]
DRIVE_QUARANTINE_FOLDER_ID = os.environ["DRIVE_QUARANTINE_FOLDER_ID"]
SERVICE_ACCOUNT_JSON = os.environ["SERVICE_ACCOUNT_JSON"]

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

# ---------- Rasterization config ----------
# 150 DPI. 200 DPI causes 504s even on per-page classify calls.
# Lower DPI loses some pixel disambiguation for handwriting but processes reliably.
RENDER_SCALE = 150 / 72  # ≈ 2.08

# ---------- Domain whitelists ----------
ASME_STANDARDS = {"B16.5", "B16.9", "B16.11", "B16.25", "B16.47"}
VALID_GRADES = {"WPB", "WPC", "WPHY", "WP11", "WP22", "WP91", "WP304", "WP316", "WP316L", "WP304L"}
INVALID_TO_VALID_GRADE = {
    "WPE": "WPB",
}

# ---------- Sheet schema (v4) ----------
ITEM_COLUMNS = [
    "Vendor", "RFE Number", "Date Received", "PL Number", "Quantity", "Unit",
    "Size", "Description", "Type", "Foreman", "Project Number",
]
LOCATION_COLUMNS = ["Connex", "Shelf"]
META_COLUMNS_AMERICAN_STAINLESS = ["Source Format", "Source File", "Extracted At"]
META_COLUMNS_NEEDS_REVIEW = META_COLUMNS_AMERICAN_STAINLESS + [
    "Subtotal Match", "Grand Total Match", "Validation Method", "Validation Notes",
]
AMERICAN_STAINLESS_HEADERS = ITEM_COLUMNS + LOCATION_COLUMNS + META_COLUMNS_AMERICAN_STAINLESS
NEEDS_REVIEW_HEADERS = ITEM_COLUMNS + LOCATION_COLUMNS + META_COLUMNS_NEEDS_REVIEW


def _col_letter(idx_0: int) -> str:
    letters = ""
    n = idx_0
    while True:
        letters = chr(ord("A") + (n % 26)) + letters
        n = n // 26 - 1
        if n < 0:
            break
    return letters


COL_IDX_RFE_NUMBER = ITEM_COLUMNS.index("RFE Number")
COL_IDX_SIZE = ITEM_COLUMNS.index("Size")
COL_IDX_DESCRIPTION = ITEM_COLUMNS.index("Description")
COL_IDX_CONNEX = len(ITEM_COLUMNS)
COL_IDX_SHELF = len(ITEM_COLUMNS) + 1

# ---------- Prompts ----------

CLASSIFY_PROMPT = """Classify this American Stainless RFE PDF.

FORMAT A — "Detailed Takeoff":
- Printed table with columns: PL QTY | Unit | Size | Description | Type | Foreman
- Header context: "NSWV", "Pipe Takeoff", "CVE Project No. XXXXX"
- Many rows, each with a foreman name

FORMAT B — "Pivot Summary":
- Two columns: "Row Labels" (left) and "Sum of PL QTY" (right)
- Parent descriptions in left column with indented sizes below each
- Bold subtotal numbers on parent rows; "Grand Total" at the bottom

UNKNOWN — neither.

Return ONLY this JSON:
{"format": "detailed" | "pivot" | "unknown"}"""


FORMAT_A_PROMPT = """You are a precise data extraction system for American Stainless industrial pipe takeoff sheets.

This document is FORMAT A — "Detailed Takeoff" — a printed table with columns:
PL QTY | Unit | Size | Description | Type | Foreman

═══════════════════════════════════════════════════════════════════
HANDWRITING — STRICT RULES
═══════════════════════════════════════════════════════════════════
IGNORE all handwritten content:
- Handwritten RFE numbers, dates, signatures
- Handwritten checkmarks (✓), tick marks, slashes, pen strokes
- "Loaded" / "Substituted" / status notes
- Strikethroughs, circles, underlines added in pen

CHECKMARK INTERFERENCE WITH QUANTITIES:
A handwritten checkmark may sit in or next to a printed PL QTY cell.
The checkmark is NOT a digit.
- Two-digit qtys (15, 39, 14) may have a checkmark overlapping the leading
  digit. Do NOT drop the leading digit. Re-examine.
- Any 1-2 character mark with curves, hooks, or slashes that is not clearly
  a printed digit is handwriting — ignore it.

═══════════════════════════════════════════════════════════════════
SIZE & DESCRIPTION — ANTI-HALLUCINATION
═══════════════════════════════════════════════════════════════════
- Sizes and descriptions MUST come directly from printed text on this page.
- Do NOT infer sizes from context. If a row's size is unclear, return "" for
  that row's size field — we will drop the row rather than guess.
- Do NOT truncate descriptions. Read the FULL Description column text per row,
  including the last row near the page bottom.

═══════════════════════════════════════════════════════════════════
CHARACTER DISAMBIGUATION
═══════════════════════════════════════════════════════════════════
- Pressure ratings: 150, 300, 600, 900, 1500 — never 160.
- Flange face types: FF, RF, RTJ — never PF.
- Grades: WPB is real; WPE is NOT (read as WPB).
- ASME standards: B16.5, B16.9, B16.11, B16.25, B16.47 — never B16.111.

═══════════════════════════════════════════════════════════════════
EXTRACTION
═══════════════════════════════════════════════════════════════════
For each row, extract:
- pl_qty (integer; preserve 0)
- unit (string)
- size (string — exact printed value; "" if unclear)
- description (FULL Description column text; "" if unclear)
- type (e.g. "tee", "elbow-90", "flange-RF", "NBGs")
- foreman (e.g. "Jacob Berry - HSM CW/R Bypass")
- pl_number (e.g. "PL-0496"; "" if absent)
- project_number (e.g. "21180"; "" if absent)

═══════════════════════════════════════════════════════════════════
OUTPUT
═══════════════════════════════════════════════════════════════════
Return ONLY this JSON:
{
  "project_number": "",
  "rows": [
    {"pl_qty": 0, "unit": "", "size": "", "description": "",
     "type": "", "foreman": "", "pl_number": ""}
  ]
}"""


def format_b_line_items_prompt(last_parent_description: str, page_number: int, total_pages: int) -> str:
    """Pass 1 prompt for ONE page of a Format B PDF. Threads parent-description carryover."""
    if last_parent_description:
        carryover = (
            f"This is page {page_number} of {total_pages}.\n"
            f"The previous page ended in the middle of a group whose parent description was:\n"
            f'    "{last_parent_description}"\n'
            f"If the first indented size rows on THIS page appear BEFORE any new parent description, "
            f"they belong to that previous group. Use exactly that parent_description string for those rows."
        )
    else:
        carryover = (
            f"This is page {page_number} of {total_pages}.\n"
            f"There is no carryover group from a previous page."
        )

    return f"""You are a precise data extraction system for American Stainless RFE pivot summary tables.

This is ONE PAGE of a multi-page pivot table export with two columns:
- LEFT ("Row Labels"): parent descriptions with indented sizes below each
- RIGHT ("Sum of PL QTY"): numbers

═══════════════════════════════════════════════════════════════════
PAGE CONTINUITY
═══════════════════════════════════════════════════════════════════
{carryover}

═══════════════════════════════════════════════════════════════════
YOUR TASK — LINE ITEMS ONLY
═══════════════════════════════════════════════════════════════════
Extract ONLY the indented size rows and their corresponding quantities on
THIS PAGE.

DO NOT extract:
- Bold subtotal numbers (on parent description rows)
- The "Grand Total" number
- Page headers, footers, or "RFE XXXX" stamps

═══════════════════════════════════════════════════════════════════
ANTI-HALLUCINATION — CRITICAL
═══════════════════════════════════════════════════════════════════
- Every `size` you return MUST be a number that VISUALLY APPEARS as an indented
  value in the left column on this page. Do NOT invent sizes.
- If a row in the right column has a number but you cannot find a matching
  indented size in the left column for it, DO NOT MAKE ONE UP. Skip that row.
- If a size IS visible but the quantity is unclear (overlapped by handwriting),
  STILL capture the row — set pl_qty to your best read. Quantity errors are
  acceptable; phantom sizes are not.
- A real size with an unclear quantity is BETTER than skipping the row.

═══════════════════════════════════════════════════════════════════
PARENT DESCRIPTION ASSIGNMENT
═══════════════════════════════════════════════════════════════════
For each indented size, parent_description is the most recent non-indented
description above it on this page — OR the carryover parent_description
above (if the size appears before any new parent description on this page).

Copy parent_description text EXACTLY as printed, including punctuation and the
full description (do not truncate).

═══════════════════════════════════════════════════════════════════
HANDWRITING & DISAMBIGUATION
═══════════════════════════════════════════════════════════════════
- Ignore all handwritten marks.
- Pressure ratings: 150, 300, 600, 900, 1500 — never 160.
- Flange faces: FF, RF, RTJ — never PF.
- Grades: WPB real, WPE not (read as WPB).
- ASME: B16.5, B16.9, B16.11, B16.25, B16.47 — never B16.111.

═══════════════════════════════════════════════════════════════════
OUTPUT
═══════════════════════════════════════════════════════════════════
Return ONLY this JSON:
{{
  "project_number": "",
  "last_parent_description_on_page": "",
  "rows": [
    {{"parent_description": "", "size": "", "pl_qty": 0}}
  ]
}}

`last_parent_description_on_page` = the most recent parent description that
appeared on THIS page (used to thread continuity to the next page). If no new
parent description appeared on this page, return the carryover value above."""


FORMAT_B_SUBTOTALS_PROMPT = """You are extracting validation anchors from an American Stainless RFE pivot summary.

YOUR TASK — SUBTOTALS AND GRAND TOTAL ONLY.

For each PARENT DESCRIPTION (non-indented row in the left column) across ALL pages, capture:
- The parent_description text (full, exactly as printed)
- The BOLD subtotal number on the same row in the right column

Also capture:
- The "Grand Total" number printed at the very bottom of the document.

DO NOT extract indented size rows or individual line item quantities.
Ignore all handwritten marks.

Return ONLY this JSON:
{
  "grand_total": 0,
  "subtotals": [
    {"parent_description": "", "subtotal": 0}
  ]
}"""


LOCATION_PARSE_PROMPT = """You are parsing a natural-language inventory location assignment from a foreman.

The foreman is unpacking items from one or more RFE (Request For Equipment)
shipments into storage containers called "Connex" (numbered Connex 1 through
Connex 20) with named shelves (e.g. "A14", "B12").

The foreman will describe in plain English which items went where. You will
output a structured JSON list of assignments.

INPUT EXAMPLES
"RFE 5414, all the Garlock 150# Ring gaskets sizes 3 through 8, Connex 7,
shelf B12. The Flexitallic CGI gaskets sizes 8 and 10, Connex 7, shelf B14."

"5903 PL-0496 all 12in stuff to Connex 3 shelf A1"

"Lap joint flanges from 5414, Connex 12 shelf top"

OUTPUT SCHEMA
{
  "assignments": [
    {
      "rfe_number": "5414",
      "description_filter": "Garlock 150# Ring gasket",
      "size_filter": ["3", "4", "6", "8"],
      "connex": "Connex 7",
      "shelf": "B12"
    }
  ]
}

Rules:
- rfe_number: digits only, no "RFE" prefix.
- description_filter: short text string to substring-match against the
  Description column. Most distinguishing words from the foreman's phrasing.
  Empty "" means "match any description for this RFE".
- size_filter: array of sizes as strings ("3", "0.5", "1.5"). Expand ranges
  like "3 through 8" to standard pipe sizes in that range: usually 3, 4, 6, 8.
  Skip 5 and 7 unless foreman explicitly says them. Empty [] means "match any".
- connex: must be "Connex N" where N is 1-20. Normalize "C7", "con 7",
  "connex seven" → "Connex 7".
- shelf: short string as foreman said it (e.g. "B12", "top"). Empty "" if not specified.

If ambiguous, include the partial assignment with empty fields — the matching
will surface 0 rows and the user can refine.

Return ONLY the JSON."""


# ---------- FastAPI ----------
app = FastAPI(title="RFE Extractor v4.1")

LOCATION_PREVIEWS: Dict[str, Dict[str, Any]] = {}


class ProcessRequest(BaseModel):
    file_id: Optional[str] = None


class LocationPreviewRequest(BaseModel):
    text: str


class LocationCommitRequest(BaseModel):
    preview_id: str


@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "rfe-extractor",
        "version": "v4.1",
        "time": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/process")
def process(req: ProcessRequest, background_tasks: BackgroundTasks):
    if req.file_id:
        background_tasks.add_task(process_single_file, req.file_id)
        return {"status": "queued", "file_id": req.file_id}
    background_tasks.add_task(process_inbox)
    return {"status": "queued", "scope": "inbox"}


@app.get("/process-inbox")
def process_inbox_get(background_tasks: BackgroundTasks):
    background_tasks.add_task(process_inbox)
    return {
        "status": "queued",
        "scope": "inbox",
        "message": "Processing started — check sheet in 30-90 seconds per file",
    }


@app.get("/reprocess-quarantine")
def reprocess_quarantine_get(background_tasks: BackgroundTasks):
    background_tasks.add_task(reprocess_quarantine)
    return {
        "status": "queued",
        "scope": "quarantine",
        "message": "Quarantined files being moved back to inbox and reprocessed",
    }


@app.post("/assign-locations/preview")
def assign_locations_preview(req: LocationPreviewRequest):
    try:
        return assign_locations_preview_impl(req.text)
    except Exception as e:
        log.exception(f"Location preview failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/assign-locations/commit")
def assign_locations_commit(req: LocationCommitRequest):
    try:
        return assign_locations_commit_impl(req.preview_id)
    except Exception as e:
        log.exception(f"Location commit failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Filename parser ----------
FILENAME_PATTERN = re.compile(
    r"^AS_RFE_(?P<rfe>\d+)_(?P<month>\d{2})-(?P<day>\d{2})-(?P<year>\d{4})\.pdf$",
    re.IGNORECASE,
)


def parse_filename(filename: str):
    m = FILENAME_PATTERN.match(filename)
    if not m:
        return None
    return {
        "vendor": "AS",
        "rfe_number": m.group("rfe"),
        "date_received": f"{m.group('year')}-{m.group('month')}-{m.group('day')}",
    }


# ---------- Drive helpers ----------
def list_folder_pdfs(folder_id: str):
    q = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
    results = drive_service.files().list(q=q, fields="files(id, name)").execute()
    return results.get("files", [])


def list_inbox_pdfs():
    return list_folder_pdfs(DRIVE_INBOX_FOLDER_ID)


def download_pdf(file_id: str) -> bytes:
    request = drive_service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()


def move_file(file_id: str, dest_folder_id: str):
    file = drive_service.files().get(fileId=file_id, fields="parents").execute()
    prev_parents = ",".join(file.get("parents", []))
    drive_service.files().update(
        fileId=file_id,
        addParents=dest_folder_id,
        removeParents=prev_parents,
        fields="id, parents",
    ).execute()


# ---------- Rasterization ----------
def rasterize_pdf_pages(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """One PNG per page. A single bad page is skipped, not fatal.

    Each page render is isolated in its own try/except so a corrupt or
    unrenderable page (which can happen with skewed phone-scan PDFs) does
    not crash the whole file. Only pages that actually render are kept.
    If every page fails, an empty list is returned and the caller
    quarantines the file.
    """
    pdf = pdfium.PdfDocument(pdf_bytes)
    pages = []
    try:
        for page_index in range(len(pdf)):
            page = None
            try:
                page = pdf[page_index]
                pil_image = page.render(scale=RENDER_SCALE).to_pil()
                buf = io.BytesIO()
                pil_image.save(buf, format="PNG", optimize=True)
                pages.append({"mime_type": "image/png", "data": buf.getvalue()})
            except Exception as e:
                log.warning(f"Skipping unrenderable page {page_index}: {e}")
            finally:
                if page is not None:
                    try:
                        page.close()
                    except Exception:
                        pass
    finally:
        try:
            pdf.close()
        except Exception:
            pass
    return pages


# ---------- Gemini calls ----------
def _gemini_json_call(prompt: str, image_parts: list) -> dict:
    response = GEMINI_MODEL.generate_content([prompt] + image_parts)
    text = response.text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def classify_format(image_parts: list) -> str:
    # First page only is enough to classify and keeps the call cheap.
    first_page_only = image_parts[:1]
    result = _gemini_json_call(CLASSIFY_PROMPT, first_page_only)
    return result.get("format", "unknown")


def extract_format_a(image_parts: list) -> dict:
    return _gemini_json_call(FORMAT_A_PROMPT, image_parts)


def extract_format_b_line_items_per_page(image_parts: List[dict]) -> dict:
    """Pass 1 — page-by-page with parent-description carryover."""
    all_rows = []
    last_parent = ""
    project_number = ""
    total_pages = len(image_parts)

    for i, page_part in enumerate(image_parts, start=1):
        prompt = format_b_line_items_prompt(last_parent, i, total_pages)
        try:
            result = _gemini_json_call(prompt, [page_part])
        except Exception as e:
            log.exception(f"Pass 1 failed on page {i}/{total_pages}: {e}")
            continue

        if not project_number:
            project_number = result.get("project_number", "") or ""
        page_rows = result.get("rows", []) or []
        all_rows.extend(page_rows)
        new_last = result.get("last_parent_description_on_page", "") or ""
        if new_last:
            last_parent = new_last
        log.info(f"Pass 1 page {i}/{total_pages}: {len(page_rows)} row(s), carryover='{last_parent[:60]}'")

    return {"project_number": project_number, "rows": all_rows}


def extract_format_b_subtotals(image_parts: list) -> dict:
    """Pass 2 — all pages together. Sparse content, fits in one call."""
    return _gemini_json_call(FORMAT_B_SUBTOTALS_PROMPT, image_parts)


# ---------- Post-process ----------
ASME_PATTERN = re.compile(r"\bB16\.\d{1,4}\b")
WORD_PATTERN = re.compile(r"\b[A-Z]{2,5}\d{0,4}[A-Z]?\b")


def _snap_asme(token: str) -> str:
    if token in ASME_STANDARDS:
        return token
    match = re.match(r"^B16\.(\d+)$", token)
    if not match:
        return token
    digits = match.group(1)
    for n in range(len(digits), 0, -1):
        candidate = f"B16.{digits[:n]}"
        if candidate in ASME_STANDARDS:
            return candidate
    return token


def _snap_grade(token: str) -> Optional[str]:
    return INVALID_TO_VALID_GRADE.get(token)


def apply_whitelist_snaps(text: str) -> str:
    if not text:
        return text
    text = ASME_PATTERN.sub(lambda m: _snap_asme(m.group(0)), text)
    text = WORD_PATTERN.sub(lambda m: _snap_grade(m.group(0)) or m.group(0), text)
    return text


def is_size_valid(size: Any) -> bool:
    """Valid = any non-empty size string.

    The anti-hallucination filter only needs to catch rows where Gemini
    invented a row with NO size (blank left column). It must NOT reject
    valid compound sizes like '5/8"x3"' or '1 1/8"-8x10 1/2"' used on bolt
    and stud RFEs. So: a size is valid if it is simply non-empty after
    trimming. Format B's two-pass subtotal cross-check remains the real
    anti-hallucination safety net.
    """
    if size is None:
        return False
    return bool(str(size).strip())


def filter_rows_with_valid_size(rows: List[dict]) -> Tuple[List[dict], int]:
    """Drop rows where size is empty/invalid. Returns (kept, dropped_count)."""
    kept = []
    dropped = 0
    for row in rows:
        if is_size_valid(row.get("size")):
            kept.append(row)
        else:
            dropped += 1
    return kept, dropped


def snap_row_fields(row: dict) -> dict:
    out = dict(row)
    for field in ("description", "type"):
        if field in out and isinstance(out[field], str):
            out[field] = apply_whitelist_snaps(out[field])
    return out


# ---------- Sheets I/O ----------
def ensure_sheet_headers():
    """Write v4 headers if row 1 of a tab is empty. Idempotent."""
    targets = [
        ("American Stainless", AMERICAN_STAINLESS_HEADERS),
        ("Needs Review", NEEDS_REVIEW_HEADERS),
        ("Quarantine", ["Filename", "Reason", "Detected At", "Notes"]),
    ]
    for tab, headers in targets:
        try:
            resp = sheets_service.spreadsheets().values().get(
                spreadsheetId=SHEET_ID,
                range=f"'{tab}'!1:1",
            ).execute()
            row = resp.get("values", [[]])
            row_is_empty = not row or not row[0] or all(not c for c in row[0])
            if row_is_empty:
                sheets_service.spreadsheets().values().update(
                    spreadsheetId=SHEET_ID,
                    range=f"'{tab}'!A1",
                    valueInputOption="RAW",
                    body={"values": [headers]},
                ).execute()
                log.info(f"Wrote v4 headers to '{tab}'")
        except Exception as e:
            log.warning(f"Could not ensure headers on '{tab}': {e}")


def append_rows_to_sheet(tab: str, rows: list):
    body = {"values": rows}
    sheets_service.spreadsheets().values().append(
        spreadsheetId=SHEET_ID,
        range=f"'{tab}'!A:A",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body=body,
    ).execute()


def log_quarantine(filename: str, reason: str, notes: str = ""):
    append_rows_to_sheet(
        "Quarantine",
        [[filename, reason, datetime.now(timezone.utc).isoformat(), notes]],
    )


def read_tab(tab: str) -> List[List[str]]:
    resp = sheets_service.spreadsheets().values().get(
        spreadsheetId=SHEET_ID,
        range=f"'{tab}'!A:ZZ",
    ).execute()
    return resp.get("values", [])


# ---------- Format B cross-pass validation ----------
def validate_format_b_cross_pass(line_items: list, subtotals_payload: dict) -> dict:
    sums_by_parent = {}
    for row in line_items:
        parent = row.get("parent_description", "")
        try:
            qty = int(row.get("pl_qty", 0) or 0)
        except (ValueError, TypeError):
            qty = 0
        sums_by_parent[parent] = sums_by_parent.get(parent, 0) + qty

    subtotals = subtotals_payload.get("subtotals", []) or []
    grand_total = subtotals_payload.get("grand_total")

    subtotal_by_parent = {}
    for s in subtotals:
        parent = s.get("parent_description", "")
        try:
            v = int(s.get("subtotal", 0) or 0)
        except (ValueError, TypeError):
            v = 0
        subtotal_by_parent[parent] = v

    all_parents = set(sums_by_parent.keys()) | set(subtotal_by_parent.keys())
    subtotal_results = []
    all_passed = True

    for parent in all_parents:
        expected = subtotal_by_parent.get(parent)
        actual = sums_by_parent.get(parent, 0)

        if expected is None:
            subtotal_results.append({
                "parent_description": parent,
                "expected": None, "actual": actual, "delta": None,
                "passed": False, "failure_reason": "no_matching_subtotal_in_pass2",
            })
            all_passed = False
            continue

        if parent not in sums_by_parent:
            subtotal_results.append({
                "parent_description": parent,
                "expected": expected, "actual": 0, "delta": -expected,
                "passed": False, "failure_reason": "no_matching_line_items_in_pass1",
            })
            all_passed = False
            continue

        passed = (actual == expected)
        if not passed:
            all_passed = False
        subtotal_results.append({
            "parent_description": parent,
            "expected": expected, "actual": actual, "delta": actual - expected,
            "passed": passed,
            "failure_reason": "" if passed else "sum_mismatch",
        })

    total_extracted = sum(sums_by_parent.values())
    grand_total_int = None
    if grand_total is not None:
        try:
            grand_total_int = int(grand_total)
        except (ValueError, TypeError):
            grand_total_int = None

    if grand_total_int is None:
        gt_passed = False
        gt_delta = None
    else:
        gt_passed = (total_extracted == grand_total_int)
        gt_delta = total_extracted - grand_total_int

    grand_total_result = {
        "expected": grand_total_int, "actual": total_extracted,
        "delta": gt_delta, "passed": gt_passed,
    }
    if not gt_passed:
        all_passed = False

    return {
        "passed": all_passed,
        "subtotal_results": subtotal_results,
        "grand_total_result": grand_total_result,
        "method": "two-pass-cross-check",
    }


def build_subtotal_match_note(parent_description: str, subtotal_results: list) -> str:
    for sub in subtotal_results:
        if sub["parent_description"] == parent_description:
            if sub["passed"]:
                return f"OK ({sub['expected']})"
            reason = sub.get("failure_reason", "")
            if reason == "no_matching_subtotal_in_pass2":
                return f"FAIL — line items sum to {sub['actual']} but no matching bold subtotal"
            if reason == "no_matching_line_items_in_pass1":
                return f"FAIL — subtotal {sub['expected']} but no matching line items"
            return f"FAIL — expected {sub['expected']}, got {sub['actual']}, delta {sub['delta']:+d}"
    return ""


def build_grand_total_note(grand_total_result: dict) -> str:
    if grand_total_result["expected"] is None:
        return "No grand total in PDF"
    if grand_total_result["passed"]:
        return f"OK ({grand_total_result['expected']})"
    delta = grand_total_result.get("delta")
    delta_str = f", delta {delta:+d}" if isinstance(delta, int) else ""
    return f"WARN — expected {grand_total_result['expected']}, got {grand_total_result['actual']}{delta_str}"


# ---------- Main processing ----------
def process_single_file(file_id: str):
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

        try:
            image_parts = rasterize_pdf_pages(pdf_bytes)
        except Exception as e:
            log.exception(f"Rasterization failed: {e}")
            log_quarantine(filename, "rasterization_failed", str(e)[:500])
            move_file(file_id, DRIVE_QUARANTINE_FOLDER_ID)
            return

        if not image_parts:
            log_quarantine(filename, "no_pages", "PDF rasterized to zero pages")
            move_file(file_id, DRIVE_QUARANTINE_FOLDER_ID)
            return

        fmt = classify_format(image_parts)
        log.info(f"{filename}: classified as '{fmt}' ({len(image_parts)} page(s))")

        extracted_at = datetime.now(timezone.utc).isoformat()
        destination_tab = "American Stainless"
        validation = None
        grand_total_note = ""
        project_number = ""
        rows = []

        if fmt == "detailed":
            payload = extract_format_a(image_parts)
            project_number = payload.get("project_number", "") or ""
            raw_rows = payload.get("rows", []) or []
            kept, dropped = filter_rows_with_valid_size(raw_rows)
            if dropped:
                log.warning(f"{filename}: dropped {dropped} Format A row(s) with empty/invalid size")
            rows = [snap_row_fields(r) for r in kept]

        elif fmt == "pivot":
            pass1 = extract_format_b_line_items_per_page(image_parts)
            pass2 = extract_format_b_subtotals(image_parts)
            project_number = pass1.get("project_number", "") or ""
            raw_line_items = pass1.get("rows", []) or []

            line_items, dropped_count = filter_rows_with_valid_size(raw_line_items)
            if dropped_count:
                log.warning(f"{filename}: dropped {dropped_count} Format B row(s) with invalid size (anti-hallucination)")

            normalized_line_items = []
            for li in line_items:
                parent_desc = apply_whitelist_snaps(li.get("parent_description", "") or "")
                normalized_line_items.append({
                    "parent_description": parent_desc,
                    "size": str(li.get("size", "") or "").strip(),
                    "pl_qty": li.get("pl_qty", 0),
                })

            normalized_pass2 = {
                "grand_total": pass2.get("grand_total"),
                "subtotals": [
                    {"parent_description": apply_whitelist_snaps(s.get("parent_description", "") or ""),
                     "subtotal": s.get("subtotal", 0)}
                    for s in (pass2.get("subtotals", []) or [])
                ],
            }

            validation = validate_format_b_cross_pass(normalized_line_items, normalized_pass2)
            grand_total_note = build_grand_total_note(validation["grand_total_result"])

            rows = []
            for li in normalized_line_items:
                rows.append({
                    "pl_qty": li["pl_qty"],
                    "unit": "",
                    "size": li["size"],
                    "description": li["parent_description"],
                    "type": "",
                    "foreman": "",
                    "pl_number": "",
                })

            if not validation["passed"]:
                destination_tab = "Needs Review"
                failed_groups = sum(1 for s in validation["subtotal_results"] if not s["passed"])
                log.warning(
                    f"Format B validation FAILED for {filename}: "
                    f"{failed_groups}/{len(validation['subtotal_results'])} groups failed; "
                    f"grand_total passed={validation['grand_total_result']['passed']}"
                )

        else:
            log.warning(f"Unknown format for {filename}")
            log_quarantine(filename, "unknown_format", "Classifier returned 'unknown'")
            move_file(file_id, DRIVE_QUARANTINE_FOLDER_ID)
            return

        if not rows:
            log.warning(f"No rows after filtering for {filename}")
            log_quarantine(filename, "empty_rows", f"format={fmt}, all rows dropped or none extracted")
            move_file(file_id, DRIVE_QUARANTINE_FOLDER_ID)
            return

        sheet_rows = []
        for row in rows:
            item_block = [
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
                project_number,
            ]
            location_block = ["", ""]  # Connex, Shelf — filled later
            meta_block = [fmt, filename, extracted_at]
            base_row = item_block + location_block + meta_block

            if destination_tab == "Needs Review":
                subtotal_match = build_subtotal_match_note(
                    row.get("description", ""), validation["subtotal_results"]
                )
                failed_count = sum(1 for s in validation["subtotal_results"] if not s["passed"])
                validation_notes = (
                    f"{failed_count} group(s) failed validation"
                    if not validation["passed"] else ""
                )
                base_row.extend([
                    subtotal_match,
                    grand_total_note,
                    validation.get("method", ""),
                    validation_notes,
                ])
            sheet_rows.append(base_row)

        append_rows_to_sheet(destination_tab, sheet_rows)
        log.info(f"Wrote {len(sheet_rows)} rows from {filename} to '{destination_tab}'")

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
    ensure_sheet_headers()
    files = list_inbox_pdfs()
    log.info(f"Inbox scan found {len(files)} PDF(s)")
    for f in files:
        process_single_file(f["id"])


def reprocess_quarantine():
    """Move every PDF from Quarantine back to Inbox, then process inbox."""
    files = list_folder_pdfs(DRIVE_QUARANTINE_FOLDER_ID)
    log.info(f"Quarantine scan found {len(files)} PDF(s) to reprocess")
    for f in files:
        try:
            move_file(f["id"], DRIVE_INBOX_FOLDER_ID)
            log.info(f"Moved {f['name']} from Quarantine back to Inbox")
        except Exception as e:
            log.exception(f"Could not move {f['name']}: {e}")
    process_inbox()


# ---------- Location assignment ----------
def parse_location_text(text: str) -> dict:
    response = GEMINI_MODEL.generate_content([LOCATION_PARSE_PROMPT, text])
    response_text = response.text.strip()
    response_text = re.sub(r"^```(?:json)?\s*", "", response_text)
    response_text = re.sub(r"\s*```$", "", response_text)
    return json.loads(response_text)


def find_matching_rows(tab: str, assignment: dict) -> List[Dict[str, Any]]:
    """Find sheet rows matching an assignment's filters."""
    rfe_filter = str(assignment.get("rfe_number", "")).strip()
    desc_filter = str(assignment.get("description_filter", "")).strip().lower()
    size_filter = assignment.get("size_filter", []) or []
    size_filter_normalized = [str(s).strip() for s in size_filter]
    connex = assignment.get("connex", "")
    shelf = assignment.get("shelf", "")

    all_rows = read_tab(tab)
    if not all_rows or len(all_rows) < 2:
        return []

    matches = []
    for i, row in enumerate(all_rows[1:], start=2):  # row 1 is headers; data starts at sheet row 2
        if len(row) <= COL_IDX_DESCRIPTION:
            continue
        row_rfe = (row[COL_IDX_RFE_NUMBER] if len(row) > COL_IDX_RFE_NUMBER else "").strip()
        row_size = (row[COL_IDX_SIZE] if len(row) > COL_IDX_SIZE else "").strip()
        row_desc = (row[COL_IDX_DESCRIPTION] if len(row) > COL_IDX_DESCRIPTION else "").strip().lower()

        if rfe_filter and row_rfe != rfe_filter:
            continue
        if desc_filter and desc_filter not in row_desc:
            continue
        if size_filter_normalized and row_size not in size_filter_normalized:
            continue

        matches.append({
            "row_number": i,
            "rfe_number": row_rfe,
            "size": row_size,
            "description": row[COL_IDX_DESCRIPTION] if len(row) > COL_IDX_DESCRIPTION else "",
            "proposed_connex": connex,
            "proposed_shelf": shelf,
        })
    return matches


def assign_locations_preview_impl(text: str) -> dict:
    parsed = parse_location_text(text)
    assignments = parsed.get("assignments", []) or []

    preview = {
        "preview_id": str(uuid.uuid4()),
        "parsed_assignments": assignments,
        "matches_per_tab": {},
        "total_matches": 0,
    }

    all_updates = []
    for tab in ("American Stainless", "Needs Review"):
        tab_matches = []
        for assignment in assignments:
            tab_matches.extend(find_matching_rows(tab, assignment))
        preview["matches_per_tab"][tab] = tab_matches
        preview["total_matches"] += len(tab_matches)
        for m in tab_matches:
            all_updates.append({
                "tab": tab,
                "row_number": m["row_number"],
                "connex": m["proposed_connex"],
                "shelf": m["proposed_shelf"],
            })

    LOCATION_PREVIEWS[preview["preview_id"]] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updates": all_updates,
        "parsed_assignments": assignments,
    }
    return preview


def assign_locations_commit_impl(preview_id: str) -> dict:
    preview = LOCATION_PREVIEWS.get(preview_id)
    if not preview:
        raise HTTPException(status_code=404, detail="preview_id not found (expired or invalid)")

    updates = preview.get("updates", [])
    if not updates:
        return {"status": "no_updates", "applied": 0}

    data = []
    connex_col_letter = _col_letter(COL_IDX_CONNEX)
    shelf_col_letter = _col_letter(COL_IDX_SHELF)
    for u in updates:
        range_str = f"'{u['tab']}'!{connex_col_letter}{u['row_number']}:{shelf_col_letter}{u['row_number']}"
        data.append({
            "range": range_str,
            "values": [[u["connex"], u["shelf"]]],
        })

    body = {"valueInputOption": "RAW", "data": data}
    sheets_service.spreadsheets().values().batchUpdate(
        spreadsheetId=SHEET_ID,
        body=body,
    ).execute()

    applied = len(updates)
    del LOCATION_PREVIEWS[preview_id]
    log.info(f"Applied {applied} location update(s) from preview {preview_id}")
    return {"status": "ok", "applied": applied}


# ════════════════════════════════════════════════════════════════════════════
# ONE-TIME GOOGLE SHEETS SETUP (do this after first deploy)
# ════════════════════════════════════════════════════════════════════════════
#
# 1. Hit /process-inbox once with no files in the inbox. This triggers
#    ensure_sheet_headers() which writes v4 column headers to the three tabs
#    if row 1 is empty.
#
# 2. American Stainless tab headers (16 cols):
#      Vendor | RFE Number | Date Received | PL Number | Quantity | Unit |
#      Size | Description | Type | Foreman | Project Number |
#      Connex | Shelf |
#      Source Format | Source File | Extracted At
#
# 3. Needs Review tab headers (20 cols):
#      [same 16 above] + Subtotal Match | Grand Total Match |
#      Validation Method | Validation Notes
#
# 4. Quarantine tab headers (4 cols):
#      Filename | Reason | Detected At | Notes
#
# 5. Add Connex dropdown (data validation) on column L of both data tabs:
#    - Cell range: 'American Stainless'!L:L (then repeat for Needs Review)
#    - Criteria: List of items
#    - Items: Connex 1,Connex 2,Connex 3,Connex 4,Connex 5,Connex 6,
#             Connex 7,Connex 8,Connex 9,Connex 10,Connex 11,Connex 12,
#             Connex 13,Connex 14,Connex 15,Connex 16,Connex 17,Connex 18,
#             Connex 19,Connex 20
#    - On invalid data: Reject input
#
# 6. (Optional) Freeze header row: View → Freeze → 1 row.
#
# 7. Shelf column (M) is free text — no dropdown.
#
# ════════════════════════════════════════════════════════════════════════════
