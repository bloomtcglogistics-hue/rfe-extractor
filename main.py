"""
RFE Extractor Service v2
Extracts pipe takeoff data from American Stainless RFE PDFs via Gemini Flash 2.5
and appends rows to the TCG Master Google Sheet.

v2 changes:
- Format B (pivot table) properly handles bold subtotals as validation anchors
- Per-group subtotal validation: line item sums must equal the bold subtotal
- Grand total cross-check
- Failed validation routes rows to "Needs Review" tab instead of "American Stainless"
- Character disambiguation rules (150 vs 160, FF vs PF, etc.)
- Explicit handwriting-ignore rules
"""
import os
import re
import json
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

# ---------- Extraction Prompt v2 ----------
EXTRACTION_PROMPT = """You are a precise data extraction system for industrial pipe takeoff sheets from American Stainless.

The PDF you are analyzing is one of TWO possible formats. First, classify which format it is, then extract accordingly.

═══════════════════════════════════════════════════════════════════
GLOBAL RULES (apply to both formats):
═══════════════════════════════════════════════════════════════════

IGNORE ALL HANDWRITTEN CONTENT. Specifically ignore:
- Handwritten RFE numbers in the top margin
- Handwritten dates anywhere on the page
- Handwritten signatures
- Handwritten checkmarks (✓) next to printed values
- Handwritten "Loaded" or status notes
- Handwritten substitution notes (e.g. "offered as WNRF")
- Handwritten numbers written near or next to printed numbers
- Strikethroughs, circles, or other ink markings
- Anything not printed by a computer

ONLY extract PRINTED values from the document.

CHARACTER DISAMBIGUATION (critical for accuracy):
- Pressure ratings in this industry are ALWAYS 150, 300, 600, 900, or 1500. NEVER 160. If you see "160#" you are misreading "150#" — output "150#".
- Flange face types are FF (Flat Face), RF (Raised Face), or RTJ. NEVER "PF". If you see "PF" you are misreading "FF" — output "FF".
- Common pressure values: 150# (most common), 300#, 600#. If unsure between 1 and 6 in a pressure rating, it is almost always 1.

═══════════════════════════════════════════════════════════════════
FORMAT A — "Detailed Takeoff"
═══════════════════════════════════════════════════════════════════

Identifying signs: A table with these column headers: PL QTY | Unit | Size | Description | Type | Foreman
Header context typically shows: "NSWV", "Pipe Takeoff", "CVE Project No. XXXXX", "Estimated Bills of Material"

For Format A, extract each row:
- pl_qty (integer, preserve 0 explicitly)
- unit (string: "ea", "ft", "lb", etc.)
- size (number as string, e.g. "12.00", "0.50")
- description (full text from Description column, captured completely)
- type (e.g. "tee", "elbow-90", "valve-ball", "valve-butterfly", "flange-RF", "NBGs", "bolting", "nipple")
- foreman (e.g. "Jacob Berry - HSM CW/R Bypass")
- pl_number (if present in row, e.g. "PL-0496")

═══════════════════════════════════════════════════════════════════
FORMAT B — "Pivot Summary" (CRITICAL — READ CAREFULLY)
═══════════════════════════════════════════════════════════════════

Identifying signs: Two columns labeled "Row Labels" (left) and "Sum of PL QTY" (right). At the bottom there is a "Grand Total" row.

═══ FORMAT B STRUCTURE ═══

This is a PIVOT TABLE EXPORT, not a flat list. Understand the structure carefully:

LEFT COLUMN ("Row Labels"):
- Contains PARENT DESCRIPTIONS (e.g. "Gasket, 150# FF 1/8" thick Garlock Blue-Gard 3000")
- Below each parent description, indented, are the SIZES for that item (e.g. "3", "4", "6")
- The next parent description starts a new group

RIGHT COLUMN ("Sum of PL QTY"):
- On the SAME ROW as each parent description is a BOLD SUBTOTAL for that item
- Below the subtotal are the LINE ITEM QUANTITIES, one per size, in matching order
- The subtotal equals the sum of its line items

═══ VISUAL EXAMPLE ═══

LEFT                                                  RIGHT
─────────────────────────────────────────────────────────────────
Gasket, 150# FF 1/8" thick Garlock Blue-Gard 3000  →  76   (BOLD SUBTOTAL)
  3                                                →  38   (line item: size 3 → 38 each)
  4                                                →  25   (line item: size 4 → 25 each)
  6                                                →  13   (line item: size 6 → 13 each)
Gasket, 150# Ring 1/8" thick Flexitallic CGI...    →  148  (BOLD SUBTOTAL)
  3                                                →  50
  4                                                →  5
  6                                                →  20
  8                                                →  50
  12                                               →  7
  14                                               →  4
  10                                               →  12

═══ FORMAT B EXTRACTION RULES ═══

1. For each parent description, identify the BOLD SUBTOTAL on the same row in the right column. Do NOT include subtotals as extracted line items — they are validation anchors.

2. Below the parent description, count how many indented SIZES appear before the next parent description starts. That count equals N.

3. Below the bold subtotal in the right column, the next N values are the line item QUANTITIES — one per size, in matching order.

4. Pair them strictly 1:1:
   - 1st indented size  ↔  1st quantity below subtotal
   - 2nd indented size  ↔  2nd quantity below subtotal
   - …Nth indented size  ↔  Nth quantity below subtotal

5. The sum of those N quantities MUST equal the bold subtotal. If your extraction doesn't math, you are misreading values — re-examine the right column carefully.

6. SPECIAL CASE — single-size parents: If a parent has only ONE indented size, the bold subtotal equals the single line item quantity. The right column may show the value twice (once as subtotal, once as line item) or once (as a single value).

═══ FORMAT B OUTPUT ═══

For each extracted line item (NOT subtotals), output:
- pl_qty (integer, the line item quantity)
- unit (leave empty "")
- size (the indented size value as string, e.g. "0.5", "12")
- description (inherited from the parent description above; full text, captured completely)
- type (leave empty "" — Format B does not have a Type column)
- foreman (leave empty "" — Format B does not have a Foreman column)
- pl_number (leave empty "")

ALSO output, in the top-level JSON:
- group_subtotals: An array of objects, one per parent description group, each with:
    - parent_description: the parent description text
    - expected_subtotal: the bold subtotal you read for this group
    - line_item_count: how many line items you extracted for this group
- grand_total: The "Grand Total" number printed at the bottom of the document (integer or null)

═══════════════════════════════════════════════════════════════════
SHARED METADATA (both formats)
═══════════════════════════════════════════════════════════════════

Also extract if visible:
- project_number (e.g. "21180" from "CVE Project No. 21180"; empty if not present)

═══════════════════════════════════════════════════════════════════
SELF-VALIDATION BEFORE RETURNING
═══════════════════════════════════════════════════════════════════

For Format B specifically, before returning your JSON:

1. For each group_subtotals entry, mentally sum the extracted line items for that parent_description. Verify the sum equals expected_subtotal. If not, re-examine the right column values for that group.

2. Sum ALL extracted line item qtys across all groups. Verify the total equals grand_total. If not, re-examine.

3. Only return your JSON after these checks pass to the best of your ability.

═══════════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════

Return ONLY valid JSON in this exact shape (no markdown, no code fences, no commentary):

{
  "format": "detailed" | "pivot" | "unknown",
  "project_number": "",
  "grand_total": null,
  "group_subtotals": [
    {
      "parent_description": "",
      "expected_subtotal": 0,
      "line_item_count": 0
    }
  ],
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

For Format A, group_subtotals can be an empty array.

If you cannot confidently classify the format, return "format": "unknown" and empty arrays for group_subtotals and rows."""

# ---------- FastAPI app ----------
app = FastAPI(title="RFE Extractor v2")


class ProcessRequest(BaseModel):
    file_id: Optional[str] = None


@app.get("/")
def health():
    return {"status": "ok", "service": "rfe-extractor", "version": "v2", "time": datetime.now(timezone.utc).isoformat()}


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


# ---------- Gemini extraction ----------
def extract_with_gemini(pdf_bytes: bytes) -> dict:
    pdf_part = {"mime_type": "application/pdf", "data": pdf_bytes}
    response = GEMINI_MODEL.generate_content([EXTRACTION_PROMPT, pdf_part])
    text = response.text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


# ---------- Sheets writer ----------
def append_rows_to_sheet(tab: str, rows: list):
    body = {"values": rows}
    sheets_service.spreadsheets().values().append(
        spreadsheetId=SHEET_ID,
        range=f"'{tab}'!A:A",
        valueInputOption="USER_ENTERED",
        insertDataOption="INSERT_ROWS",
        body=body,
    ).execute()


def log_quarantine(filename: str, reason: str, notes: str = ""):
    append_rows_to_sheet(
        "Quarantine",
        [[filename, reason, datetime.now(timezone.utc).isoformat(), notes]],
    )


# ---------- Validation ----------
def validate_format_b(extracted: dict) -> dict:
    """
    Validates Format B extraction:
    - Sums line items per group, compares to expected subtotal
    - Sums all line items, compares to grand total
    Returns dict with: passed (bool), subtotal_results (per-group), grand_total_result
    """
    rows = extracted.get("rows", [])
    group_subtotals = extracted.get("group_subtotals", [])
    grand_total = extracted.get("grand_total")

    # Build map: parent_description -> sum of line item qtys
    sums_by_parent = {}
    for row in rows:
        parent = row.get("description", "")
        try:
            qty = int(row.get("pl_qty", 0) or 0)
        except (ValueError, TypeError):
            qty = 0
        sums_by_parent[parent] = sums_by_parent.get(parent, 0) + qty

    # Compare each group's sum to its expected subtotal (STRICT — exact match)
    subtotal_results = []
    all_groups_passed = True
    for sub in group_subtotals:
        parent = sub.get("parent_description", "")
        expected = sub.get("expected_subtotal", 0)
        actual = sums_by_parent.get(parent, 0)
        passed = (actual == expected)
        if not passed:
            all_groups_passed = False
        subtotal_results.append({
            "parent_description": parent,
            "expected": expected,
            "actual": actual,
            "delta": actual - expected,
            "passed": passed,
        })

    # Grand total check (informational only — doesn't block)
    total_extracted = sum(sums_by_parent.values())
    grand_total_result = {
        "expected": grand_total,
        "actual": total_extracted,
        "delta": (total_extracted - grand_total) if grand_total is not None else None,
        "passed": (grand_total is not None and total_extracted == grand_total),
    }

    return {
        "passed": all_groups_passed,
        "subtotal_results": subtotal_results,
        "grand_total_result": grand_total_result,
    }


def build_subtotal_match_note(parent_description: str, subtotal_results: list) -> str:
    """Build a per-row note describing the subtotal match status for that row's parent group."""
    for sub in subtotal_results:
        if sub["parent_description"] == parent_description:
            if sub["passed"]:
                return f"OK ({sub['expected']})"
            return f"FAIL — expected {sub['expected']}, got {sub['actual']}, delta {sub['delta']:+d}"
    return ""


def build_grand_total_note(grand_total_result: dict) -> str:
    """Build a note describing the grand total match status."""
    if grand_total_result["expected"] is None:
        return "No grand total in PDF"
    if grand_total_result["passed"]:
        return f"OK ({grand_total_result['expected']})"
    return f"WARN — expected {grand_total_result['expected']}, got {grand_total_result['actual']}, delta {grand_total_result['delta']:+d}"


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
        extracted = extract_with_gemini(pdf_bytes)

        if extracted.get("format") == "unknown" or not extracted.get("rows"):
            log.warning(f"Unknown format or empty rows: {filename}")
            log_quarantine(filename, "unknown_format", "Gemini could not classify or returned no rows")
            move_file(file_id, DRIVE_QUARANTINE_FOLDER_ID)
            return

        extracted_at = datetime.now(timezone.utc).isoformat()
        fmt = extracted.get("format", "")

        # Determine destination tab based on validation
        destination_tab = "American Stainless"
        validation = None
        grand_total_note = ""

        if fmt == "pivot":
            validation = validate_format_b(extracted)
            grand_total_note = build_grand_total_note(validation["grand_total_result"])

            if not validation["passed"]:
                destination_tab = "Needs Review"
                log.warning(
                    f"Format B validation FAILED for {filename}: "
                    f"{sum(1 for s in validation['subtotal_results'] if not s['passed'])} "
                    f"groups out of {len(validation['subtotal_results'])} have subtotal mismatches"
                )

        # Build sheet rows
        sheet_rows = []
        for row in extracted["rows"]:
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
                extracted.get("project_number", ""),
                fmt,
                filename,
                extracted_at,
            ]

            # If going to Needs Review, add the 3 extra diagnostic columns
            if destination_tab == "Needs Review":
                subtotal_match = build_subtotal_match_note(row.get("description", ""), validation["subtotal_results"])
                validation_notes = ""
                if not validation["passed"]:
                    failed_count = sum(1 for s in validation['subtotal_results'] if not s['passed'])
                    validation_notes = f"{failed_count} group(s) failed subtotal validation"
                base_row.extend([subtotal_match, grand_total_note, validation_notes])

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
