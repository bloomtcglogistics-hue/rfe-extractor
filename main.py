"""
RFE Extractor Service v3
Extracts pipe takeoff data from American Stainless RFE PDFs via Gemini Flash 2.5
and appends rows to the TCG Master Google Sheet.

v3 changes vs v2:
- PDFs rasterized to PNG at 200 DPI before sending to Gemini (more pixels for
  digit disambiguation; targets Format A handwritten two-digit qty truncation
  caused by checkmarks adjacent to qty cells).
- Format B uses TWO-PASS extraction:
    Pass 1: line items only, NO subtotal awareness ("ignore bold numbers, extract
            only indented quantities"). Gemini cannot fudge values to match a
            target it does not see.
    Pass 2: bold subtotals + grand total ONLY, anchored by parent description.
    Validation cross-checks pass-1 sums vs pass-2 subtotals — Gemini's
    self-report is no longer trusted.
- Format classification is now a separate cheap pre-pass. Single round-trip
  prompts that classified AND extracted let Gemini "see" the whole document
  including subtotals; separating them keeps each call narrowly scoped.
- Sheets append uses valueInputOption="RAW" so ISO date strings no longer get
  coerced to Excel serial numbers (e.g. 46162 → 2026-05-20).
- Post-process whitelists snap fabricated values to valid domain values:
    * ASME standards: B16.5, B16.9, B16.11, B16.25, B16.47 (catches B16.111)
    * Grade designations: WPB is real, WPE is not (snap WPE → WPB)
- Format A prompt explicitly addresses the checkmark/two-digit qty failure mode.
- Needs Review tab gets a 'validation_method' diagnostic column so failure modes
  are distinguishable at a glance.
"""

import os
import re
import json
import logging
import io
from datetime import datetime, timezone
from typing import Optional

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
# 200 DPI: pypdfium2 takes a scale factor relative to its base 72 DPI.
# 150 / 72 ≈ 2.08. Reduced from 200 DPI after Format B 504 timeouts on multi-page PDFs.
RENDER_SCALE = 150 / 72  # ≈ 2.08

# ---------- Domain whitelists for post-process snapping ----------
ASME_STANDARDS = {"B16.5", "B16.9", "B16.11", "B16.25", "B16.47"}
# Common ASTM grades for fittings/pipe. WPE is NOT a real grade — common
# misread of WPB. Add others here as they come up.
VALID_GRADES = {"WPB", "WPC", "WPHY", "WP11", "WP22", "WP91", "WP304", "WP316", "WP316L", "WP304L"}
INVALID_TO_VALID_GRADE = {
    "WPE": "WPB",   # WPE is not a real ASTM grade
}

# ---------- Prompts ----------

CLASSIFY_PROMPT = """Classify this American Stainless RFE PDF into ONE of three categories.

FORMAT A — "Detailed Takeoff":
- A printed table with these column headers: PL QTY | Unit | Size | Description | Type | Foreman
- Header context shows: "NSWV", "Pipe Takeoff", "CVE Project No. XXXXX"
- Many rows, each with a foreman name (e.g. "Jacob Berry - HSM CW/R Bypass")

FORMAT B — "Pivot Summary":
- Two columns labeled "Row Labels" (left) and "Sum of PL QTY" (right)
- Parent descriptions in left column with indented sizes below each
- Bold subtotal numbers on parent rows in the right column
- A "Grand Total" row at the bottom

UNKNOWN — neither of the above.

Return ONLY this JSON, nothing else:
{"format": "detailed" | "pivot" | "unknown"}"""


FORMAT_A_PROMPT = """You are a precise data extraction system for American Stainless industrial pipe takeoff sheets.

This document is FORMAT A — "Detailed Takeoff" — a printed table with columns:
PL QTY | Unit | Size | Description | Type | Foreman

═══════════════════════════════════════════════════════════════════
HANDWRITING — STRICT RULES
═══════════════════════════════════════════════════════════════════
IGNORE all handwritten content. Specifically:
- Handwritten RFE numbers, dates, signatures in the margins
- Handwritten checkmarks (✓), tick marks, slashes, or pen strokes
- Handwritten "Loaded" / "Substituted" / status notes
- Strikethroughs, circles, underlines added in pen
- ANY ink markings adjacent to or overlapping printed numbers

CRITICAL — CHECKMARK INTERFERENCE WITH QUANTITIES:
A common pattern on these sheets is a handwritten checkmark or tick mark
placed in the same cell as (or directly adjacent to) a printed PL QTY value.
The checkmark is NOT a digit. Specifically:
- If a printed quantity is two digits (e.g. "15", "39", "14"), a checkmark to
  its left or right may visually overlap and make the leading digit harder to
  see. DO NOT drop the leading digit. Re-examine carefully.
- Treat any 1-2 character mark with curves, hooks, or slashes that is not a
  clear printed digit as a handwritten mark — ignore it.
- After reading a quantity, sanity-check: PL QTY values on these sheets are
  typically 1-200. If you extracted a single digit and there is ambiguous ink
  next to it, look again for a leading digit before committing.

═══════════════════════════════════════════════════════════════════
CHARACTER DISAMBIGUATION
═══════════════════════════════════════════════════════════════════
- Pressure ratings are ALWAYS 150, 300, 600, 900, or 1500. NEVER 160.
  If you see "160#" you are misreading "150#".
- Flange face types are FF (Flat Face), RF (Raised Face), or RTJ. NEVER "PF".
- ASTM grade designations: WPB is a real grade. WPE is NOT a real ASTM grade —
  if you see "WPE" you are misreading "WPB". (Post-processing will also catch
  this, but read it correctly the first time.)
- ASME standards in descriptions are one of: B16.5, B16.9, B16.11, B16.25,
  B16.47. There is no "B16.111" or "B16.1111" — those are misreads of B16.11.

═══════════════════════════════════════════════════════════════════
EXTRACTION
═══════════════════════════════════════════════════════════════════
For each row in the printed table, extract:
- pl_qty (integer; preserve 0 explicitly; apply checkmark rule above)
- unit (string: "ea", "ft", "lb", etc.)
- size (number as string, e.g. "12.00", "0.50")
- description (FULL text from Description column — do not truncate; this
  includes the last row, which is often near the page bottom)
- type (e.g. "tee", "elbow-90", "valve-ball", "valve-butterfly", "flange-RF",
  "NBGs", "bolting", "nipple")
- foreman (e.g. "Jacob Berry - HSM CW/R Bypass")
- pl_number (e.g. "PL-0496", or "" if not present)

Also extract:
- project_number (e.g. "21180" from "CVE Project No. 21180"; "" if absent)

═══════════════════════════════════════════════════════════════════
OUTPUT
═══════════════════════════════════════════════════════════════════
Return ONLY this JSON, no markdown, no commentary:
{
  "project_number": "",
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
}"""


# Pass 1 of Format B: LINE ITEMS ONLY. Subtotals are intentionally hidden from
# this prompt's instructions so Gemini cannot "adjust" line items to match a
# target it does not see.
FORMAT_B_LINE_ITEMS_PROMPT = """You are a precise data extraction system for American Stainless RFE pivot summary tables.

This document is a PIVOT TABLE EXPORT with two columns:
- LEFT ("Row Labels"): parent descriptions with indented sizes below each
- RIGHT ("Sum of PL QTY"): numbers

═══════════════════════════════════════════════════════════════════
YOUR TASK — LINE ITEMS ONLY
═══════════════════════════════════════════════════════════════════
Extract ONLY the indented size rows and their corresponding quantities.

DO NOT extract:
- Bold subtotal numbers (those on the same row as a parent description)
- The "Grand Total" number at the bottom

Focus exclusively on:
- Each indented size value in the left column
- Its corresponding line item quantity in the right column (the non-bold
  number directly to the right of that indented size, or the next non-bold
  number in the right column following the indented size)
- The parent description text that introduces the group the size belongs to

═══════════════════════════════════════════════════════════════════
GROUP STRUCTURE
═══════════════════════════════════════════════════════════════════
LEFT                                              RIGHT
─────────────────────────────────────────────────────────────────
Gasket, 150# FF 1/8" thick Garlock Blue-Gard...  76  ← IGNORE (bold subtotal)
  3                                              38  ← extract: size 3, qty 38
  4                                              25  ← extract: size 4, qty 25
  6                                              13  ← extract: size 6, qty 13
Gasket, 150# Ring 1/8" thick Flexitallic...     148  ← IGNORE (bold subtotal)
  3                                              50  ← extract: size 3, qty 50
  4                                               5  ← extract: size 4, qty 5
  ...

For each indented size, the parent_description is the most recent non-indented
description above it in the left column.

═══════════════════════════════════════════════════════════════════
HANDWRITING — IGNORE
═══════════════════════════════════════════════════════════════════
Ignore all handwritten marks: checkmarks, dates, signatures, status notes.
Extract only PRINTED values.

═══════════════════════════════════════════════════════════════════
CHARACTER DISAMBIGUATION
═══════════════════════════════════════════════════════════════════
- Pressure ratings: 150, 300, 600, 900, 1500 — never 160.
- Flange faces: FF, RF, RTJ — never PF.
- Grades: WPB is real, WPE is not (read as WPB).
- ASME: B16.5, B16.9, B16.11, B16.25, B16.47 — never B16.111.

═══════════════════════════════════════════════════════════════════
OUTPUT
═══════════════════════════════════════════════════════════════════
Return ONLY this JSON, no markdown, no commentary:
{
  "project_number": "",
  "rows": [
    {
      "parent_description": "",
      "size": "",
      "pl_qty": 0
    }
  ]
}

Capture EVERY indented size row across ALL groups, including groups that span
page breaks. A group continues across pages until a new parent description
appears."""


# Pass 2 of Format B: SUBTOTALS + GRAND TOTAL ONLY. Line items are intentionally
# hidden so Gemini focuses solely on the bold numbers.
FORMAT_B_SUBTOTALS_PROMPT = """You are extracting validation anchors from an American Stainless RFE pivot summary.

This document is a PIVOT TABLE EXPORT. Your task is narrow:

═══════════════════════════════════════════════════════════════════
YOUR TASK — SUBTOTALS AND GRAND TOTAL ONLY
═══════════════════════════════════════════════════════════════════
For each PARENT DESCRIPTION (non-indented row in the left column), capture:
- The parent_description text (full, exactly as printed)
- The BOLD subtotal number on the same row in the right column

Also capture:
- The "Grand Total" number printed at the very bottom of the document

DO NOT extract:
- Indented size rows
- Individual line item quantities

═══════════════════════════════════════════════════════════════════
LAYOUT REMINDER
═══════════════════════════════════════════════════════════════════
LEFT                                              RIGHT
─────────────────────────────────────────────────────────────────
Gasket, 150# FF 1/8" thick Garlock Blue-Gard...  76   ← capture this
  3                                              38   ← ignore (line item)
  4                                              25   ← ignore (line item)
  ...
Gasket, 150# Ring 1/8" thick Flexitallic...     148   ← capture this
  ...
Grand Total                                     942   ← capture as grand_total

═══════════════════════════════════════════════════════════════════
HANDWRITING — IGNORE
═══════════════════════════════════════════════════════════════════
Ignore all handwritten marks.

═══════════════════════════════════════════════════════════════════
OUTPUT
═══════════════════════════════════════════════════════════════════
Return ONLY this JSON, no markdown, no commentary:
{
  "grand_total": 0,
  "subtotals": [
    {
      "parent_description": "",
      "subtotal": 0
    }
  ]
}

The parent_description must match exactly the full printed text so it can be
joined with line items extracted in a separate pass."""


# ---------- FastAPI app ----------
app = FastAPI(title="RFE Extractor v3")


class ProcessRequest(BaseModel):
    file_id: Optional[str] = None


@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "rfe-extractor",
        "version": "v3",
        "time": datetime.now(timezone.utc).isoformat(),
    }


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
    return {"status": "queued", "scope": "inbox", "message": "Processing started — check sheet in 30-90 seconds"}


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
    q = (
        f"'{DRIVE_INBOX_FOLDER_ID}' in parents "
        f"and mimeType='application/pdf' "
        f"and trashed=false"
    )
    results = drive_service.files().list(q=q, fields="files(id, name)").execute()
    return results.get("files", [])


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
def rasterize_pdf_to_pngs(pdf_bytes: bytes) -> list:
    """
    Rasterize all pages of a PDF to PNG bytes at ~200 DPI.
    Returns a list of {"mime_type": "image/png", "data": <bytes>} dicts ready
    for Gemini multimodal input.
    """
    pdf = pdfium.PdfDocument(pdf_bytes)
    parts = []
    try:
        for page_index in range(len(pdf)):
            page = pdf[page_index]
            pil_image = page.render(scale=RENDER_SCALE).to_pil()
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG", optimize=True)
            parts.append({"mime_type": "image/png", "data": buf.getvalue()})
            page.close()
    finally:
        pdf.close()
    return parts


# ---------- Gemini calls ----------
def _gemini_json_call(prompt: str, image_parts: list) -> dict:
    """Call Gemini with a prompt and image parts, parse JSON response."""
    response = GEMINI_MODEL.generate_content([prompt] + image_parts)
    text = response.text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def classify_format(image_parts: list) -> str:
    """Pass 0: classify document format."""
    result = _gemini_json_call(CLASSIFY_PROMPT, image_parts)
    return result.get("format", "unknown")


def extract_format_a(image_parts: list) -> dict:
    """Format A single-call extraction."""
    return _gemini_json_call(FORMAT_A_PROMPT, image_parts)


def extract_format_b_line_items(image_parts: list) -> dict:
    """Format B pass 1: line items only, no subtotal awareness."""
    return _gemini_json_call(FORMAT_B_LINE_ITEMS_PROMPT, image_parts)


def extract_format_b_subtotals(image_parts: list) -> dict:
    """Format B pass 2: subtotals + grand total only, no line item awareness."""
    return _gemini_json_call(FORMAT_B_SUBTOTALS_PROMPT, image_parts)


# ---------- Post-process whitelists ----------
ASME_PATTERN = re.compile(r"\bB16\.\d{1,4}\b")
WORD_PATTERN = re.compile(r"\b[A-Z]{2,5}\d{0,4}[A-Z]?\b")


def _snap_asme(token: str) -> str:
    """Snap a B16.XXX token to the nearest valid ASME standard."""
    if token in ASME_STANDARDS:
        return token
    # Strip extra trailing digits: B16.111 → try B16.11, then B16.1
    match = re.match(r"^B16\.(\d+)$", token)
    if not match:
        return token
    digits = match.group(1)
    # Try progressively shorter suffixes
    for n in range(len(digits), 0, -1):
        candidate = f"B16.{digits[:n]}"
        if candidate in ASME_STANDARDS:
            return candidate
    return token  # No match — leave as-is


def _snap_grade(token: str) -> Optional[str]:
    """Snap a grade token to its corrected value, or None if no rule applies."""
    return INVALID_TO_VALID_GRADE.get(token)


def apply_whitelist_snaps(text: str) -> str:
    """Apply post-process whitelist corrections to a text field."""
    if not text:
        return text
    # ASME standards
    def _asme_repl(m):
        return _snap_asme(m.group(0))
    text = ASME_PATTERN.sub(_asme_repl, text)
    # Grade tokens — only replace known bad ones
    def _grade_repl(m):
        tok = m.group(0)
        corrected = _snap_grade(tok)
        return corrected if corrected else tok
    text = WORD_PATTERN.sub(_grade_repl, text)
    return text


def snap_row_fields(row: dict) -> dict:
    """Apply whitelist snaps to text fields in an extracted row."""
    out = dict(row)
    for field in ("description", "type"):
        if field in out and isinstance(out[field], str):
            out[field] = apply_whitelist_snaps(out[field])
    return out


# ---------- Sheets writer ----------
def append_rows_to_sheet(tab: str, rows: list):
    """Append rows using RAW mode so date strings are not coerced to serials."""
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


# ---------- Format B cross-pass validation ----------
def validate_format_b_cross_pass(line_items: list, subtotals_payload: dict) -> dict:
    """
    Validates Format B by comparing pass-1 line item sums against pass-2 subtotals.
    Gemini cannot fudge this because the two passes never see each other's targets.

    Returns:
      {
        "passed": bool,
        "subtotal_results": [...],
        "grand_total_result": {...},
        "method": "two-pass-cross-check"
      }
    """
    # Sum line items per parent_description from pass 1
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

    # Build lookup of pass-2 subtotals
    subtotal_by_parent = {}
    for s in subtotals:
        parent = s.get("parent_description", "")
        try:
            subtotal_val = int(s.get("subtotal", 0) or 0)
        except (ValueError, TypeError):
            subtotal_val = 0
        subtotal_by_parent[parent] = subtotal_val

    # Cross-check: every parent in either pass must appear in both, and sums must match
    all_parents = set(sums_by_parent.keys()) | set(subtotal_by_parent.keys())
    subtotal_results = []
    all_passed = True

    for parent in all_parents:
        expected = subtotal_by_parent.get(parent)
        actual = sums_by_parent.get(parent, 0)

        if expected is None:
            # Line items found but no matching subtotal — parent_description
            # text mismatch between passes (anchor failure)
            subtotal_results.append({
                "parent_description": parent,
                "expected": None,
                "actual": actual,
                "delta": None,
                "passed": False,
                "failure_reason": "no_matching_subtotal_in_pass2",
            })
            all_passed = False
            continue

        if parent not in sums_by_parent:
            # Subtotal found but no line items — anchor failure other direction
            subtotal_results.append({
                "parent_description": parent,
                "expected": expected,
                "actual": 0,
                "delta": -expected,
                "passed": False,
                "failure_reason": "no_matching_line_items_in_pass1",
            })
            all_passed = False
            continue

        passed = (actual == expected)
        if not passed:
            all_passed = False
        subtotal_results.append({
            "parent_description": parent,
            "expected": expected,
            "actual": actual,
            "delta": actual - expected,
            "passed": passed,
            "failure_reason": "" if passed else "sum_mismatch",
        })

    # Grand total check
    total_extracted = sum(sums_by_parent.values())
    if grand_total is None:
        gt_passed = False
        gt_delta = None
    else:
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
            grand_total = grand_total_int  # normalize for downstream

    grand_total_result = {
        "expected": grand_total,
        "actual": total_extracted,
        "delta": gt_delta,
        "passed": gt_passed,
    }

    # Grand total mismatch alone fails the document
    if not gt_passed:
        all_passed = False

    return {
        "passed": all_passed,
        "subtotal_results": subtotal_results,
        "grand_total_result": grand_total_result,
        "method": "two-pass-cross-check",
    }


def build_subtotal_match_note(parent_description: str, subtotal_results: list) -> str:
    """Build a per-row note describing the subtotal match status for that row's parent group."""
    for sub in subtotal_results:
        if sub["parent_description"] == parent_description:
            if sub["passed"]:
                return f"OK ({sub['expected']})"
            reason = sub.get("failure_reason", "")
            if reason == "no_matching_subtotal_in_pass2":
                return f"FAIL — line items sum to {sub['actual']} but no matching bold subtotal found"
            if reason == "no_matching_line_items_in_pass1":
                return f"FAIL — subtotal {sub['expected']} found but no matching line items"
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

        # Rasterize once, reuse across all Gemini calls
        try:
            image_parts = rasterize_pdf_to_pngs(pdf_bytes)
        except Exception as e:
            log.exception(f"Rasterization failed for {filename}: {e}")
            log_quarantine(filename, "rasterization_failed", str(e)[:500])
            move_file(file_id, DRIVE_QUARANTINE_FOLDER_ID)
            return

        if not image_parts:
            log.warning(f"No pages rendered from {filename}")
            log_quarantine(filename, "no_pages", "PDF rasterized to zero pages")
            move_file(file_id, DRIVE_QUARANTINE_FOLDER_ID)
            return

        # Pass 0: classify
        fmt = classify_format(image_parts)
        log.info(f"{filename}: classified as '{fmt}'")

        extracted_at = datetime.now(timezone.utc).isoformat()
        destination_tab = "American Stainless"
        validation = None
        grand_total_note = ""
        project_number = ""
        rows = []

        if fmt == "detailed":
            # Format A: single extraction call
            payload = extract_format_a(image_parts)
            project_number = payload.get("project_number", "") or ""
            raw_rows = payload.get("rows", []) or []
            rows = [snap_row_fields(r) for r in raw_rows]

        elif fmt == "pivot":
            # Format B: two-pass extraction with cross-validation
            pass1 = extract_format_b_line_items(image_parts)
            pass2 = extract_format_b_subtotals(image_parts)
            project_number = pass1.get("project_number", "") or ""
            line_items_raw = pass1.get("rows", []) or []

            # Normalize Format B line items into the standard row shape, applying
            # whitelist snaps to inherited parent_description (which becomes the row's description).
            rows = []
            for li in line_items_raw:
                parent_desc = apply_whitelist_snaps(li.get("parent_description", "") or "")
                rows.append({
                    "pl_qty": li.get("pl_qty", 0),
                    "unit": "",
                    "size": li.get("size", "") or "",
                    "description": parent_desc,
                    "type": "",
                    "foreman": "",
                    "pl_number": "",
                })

            # Run cross-pass validation against the SAME (snapped) descriptions
            # used in pass-1 normalization. We rebuild a comparable pass1 view
            # using snapped descriptions so the parent keys join correctly with
            # pass2 subtotal descriptions (which we also snap).
            normalized_pass1 = [
                {
                    "parent_description": apply_whitelist_snaps(li.get("parent_description", "") or ""),
                    "pl_qty": li.get("pl_qty", 0),
                }
                for li in line_items_raw
            ]
            normalized_pass2 = {
                "grand_total": pass2.get("grand_total"),
                "subtotals": [
                    {
                        "parent_description": apply_whitelist_snaps(s.get("parent_description", "") or ""),
                        "subtotal": s.get("subtotal", 0),
                    }
                    for s in (pass2.get("subtotals", []) or [])
                ],
            }

            validation = validate_format_b_cross_pass(normalized_pass1, normalized_pass2)
            grand_total_note = build_grand_total_note(validation["grand_total_result"])

            if not validation["passed"]:
                destination_tab = "Needs Review"
                failed_groups = sum(1 for s in validation["subtotal_results"] if not s["passed"])
                log.warning(
                    f"Format B validation FAILED for {filename}: "
                    f"{failed_groups}/{len(validation['subtotal_results'])} groups failed; "
                    f"grand_total passed={validation['grand_total_result']['passed']}"
                )

        else:
            # Unknown format
            log.warning(f"Unknown format for {filename}")
            log_quarantine(filename, "unknown_format", "Classifier returned 'unknown'")
            move_file(file_id, DRIVE_QUARANTINE_FOLDER_ID)
            return

        if not rows:
            log.warning(f"No rows extracted from {filename}")
            log_quarantine(filename, "empty_rows", f"format={fmt}, extraction returned no rows")
            move_file(file_id, DRIVE_QUARANTINE_FOLDER_ID)
            return

        # Build sheet rows. Column order matches v2 schema for backward compat
        # of the American Stainless tab. Needs Review gets 4 extra diagnostic columns
        # (subtotal_match, grand_total_match, validation_method, validation_notes).
        sheet_rows = []
        for row in rows:
            base_row = [
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
                fmt,
                filename,
                extracted_at,
            ]
            if destination_tab == "Needs Review":
                subtotal_match = build_subtotal_match_note(
                    row.get("description", ""), validation["subtotal_results"]
                )
                failed_count = sum(1 for s in validation["subtotal_results"] if not s["passed"])
                validation_notes = (
                    f"{failed_count} group(s) failed validation"
                    if not validation["passed"]
                    else ""
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
    files = list_inbox_pdfs()
    log.info(f"Inbox scan found {len(files)} PDF(s)")
    for f in files:
        process_single_file(f["id"])
