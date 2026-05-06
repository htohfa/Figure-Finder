import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime


SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

HEADERS = ["timestamp", "query", "n_papers", "n_matches", "cost", "rating"]


def _get_sheet():
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=SCOPES
    )
    client = gspread.authorize(creds)
    sheet_id = st.secrets["sheets"]["sheet_id"]
    return client.open_by_key(sheet_id).sheet1


def load_stats() -> dict:
    try:
        sheet = _get_sheet()
        rows = sheet.get_all_records()
        ratings = [r["rating"] for r in rows if r.get("rating")]
        return {
            "searches": len(rows),
            "ratings": ratings,
        }
    except Exception:
        return {"searches": 0, "ratings": []}


def log_search(query: str, n_papers: int, n_matches: int, cost: float):
    try:
        sheet = _get_sheet()
        if sheet.row_count == 1 and not sheet.row_values(1):
            sheet.append_row(HEADERS)
        sheet.append_row([
            datetime.utcnow().isoformat(),
            query, n_papers, n_matches, round(cost, 4), "",
        ])
    except Exception:
        pass


def log_rating(rating: int):
    try:
        sheet = _get_sheet()
        all_rows = sheet.get_all_values()
        # Skip header row, find last data row with empty rating
        for i in range(len(all_rows) - 1, 0, -1):
            row = all_rows[i]
            # Check if this row has search data but no rating
            if row[0] and (len(row) < 6 or row[5] == ""):
                sheet.update_cell(i + 1, 6, rating)
                return
    except Exception:
        pass
