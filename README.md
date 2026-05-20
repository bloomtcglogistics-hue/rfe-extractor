# RFE Extractor

FastAPI service that extracts pipe takeoff data from American Stainless RFE PDFs
via Gemini Flash 2.5 and appends rows to the TCG Master Google Sheet.

## Endpoints
- `GET /` — health check
- `POST /process` — process inbox folder or a specific file ID

## Environment variables
- `GEMINI_API_KEY` — Gemini API key
- `SHEET_ID` — TCG Master spreadsheet ID
- `DRIVE_INBOX_FOLDER_ID` — WV RFEs folder ID
- `DRIVE_PROCESSED_FOLDER_ID` — Processed subfolder ID
- `DRIVE_QUARANTINE_FOLDER_ID` — Quarantine subfolder ID
- `SERVICE_ACCOUNT_JSON` — Full service account JSON as a single-line string
