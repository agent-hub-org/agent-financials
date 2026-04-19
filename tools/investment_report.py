"""Generate downloadable investment thesis PDF/markdown reports."""
import logging
import os
import re
import uuid
from datetime import datetime, timezone

from langchain_core.tools import tool

logger = logging.getLogger("agent_financials.tools.investment_report")

_BASE_URL = (os.getenv("BACKEND_URL") or os.getenv("PUBLIC_URL") or "").rstrip("/")

_UNICODE_TO_ASCII = str.maketrans({
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta",
    "≤": "<=", "≥": ">=", "≠": "!=", "→": "->",
    "·": ".", "•": "*", "₹": "Rs.", "€": "EUR", "£": "GBP",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2014": "--", "\u2013": "-",
})


def _sanitize(text: str) -> str:
    return text.translate(_UNICODE_TO_ASCII)


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text[:60]


def _create_pdf_bytes(title: str, content: str) -> bytes:
    from fpdf import FPDF

    title = _sanitize(title)
    content = _sanitize(content)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header bar
    pdf.set_fill_color(30, 50, 120)
    pdf.rect(0, 0, 210, 18, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_xy(10, 4)
    pdf.cell(0, 10, "Investment Report — Agent Hub", ln=False)

    # Title
    pdf.set_text_color(20, 20, 20)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_xy(10, 24)
    pdf.multi_cell(190, 10, title, align="C")
    pdf.ln(4)

    # Divider
    pdf.set_draw_color(30, 50, 120)
    pdf.set_line_width(0.6)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    # Content
    pdf.set_font("Helvetica", size=11)
    pdf.set_text_color(20, 20, 20)

    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# "):
            pdf.set_font("Helvetica", "B", 16)
            pdf.ln(4)
            pdf.multi_cell(0, 10, stripped[2:])
            pdf.set_font("Helvetica", size=11)
        elif stripped.startswith("## "):
            pdf.set_font("Helvetica", "B", 14)
            pdf.ln(3)
            pdf.multi_cell(0, 9, stripped[3:])
            pdf.set_font("Helvetica", size=11)
        elif stripped.startswith("### "):
            pdf.set_font("Helvetica", "B", 12)
            pdf.ln(2)
            pdf.multi_cell(0, 8, stripped[4:])
            pdf.set_font("Helvetica", size=11)
        elif stripped.startswith("#### "):
            pdf.set_font("Helvetica", "BI", 11)
            pdf.ln(2)
            pdf.multi_cell(0, 7, stripped[5:])
            pdf.set_font("Helvetica", size=11)
        elif stripped.startswith("> "):
            # Blockquote — light blue background
            quote_text = re.sub(r"\*\*(.*?)\*\*", r"\1", stripped[2:])
            quote_text = re.sub(r"\*(.*?)\*", r"\1", quote_text)
            pdf.set_fill_color(230, 240, 255)
            pdf.set_font("Helvetica", "I", 10)
            pdf.set_x(14)
            pdf.multi_cell(180, 6, quote_text, fill=True)
            pdf.set_font("Helvetica", size=11)
            pdf.ln(1)
        elif stripped.startswith("- ") or stripped.startswith("* "):
            bullet = re.sub(r"\*\*(.*?)\*\*", r"\1", stripped[2:])
            bullet = re.sub(r"\*(.*?)\*", r"\1", bullet)
            pdf.set_x(14)
            pdf.multi_cell(0, 6, f"• {bullet}")
        elif stripped.startswith("|"):
            # Simple table row
            plain = re.sub(r"\*\*(.*?)\*\*", r"\1", stripped)
            plain = re.sub(r"\*(.*?)\*", r"\1", plain)
            pdf.set_font("Courier", size=9)
            pdf.multi_cell(0, 5, plain)
            pdf.set_font("Helvetica", size=11)
        elif stripped:
            plain = re.sub(r"\*\*(.*?)\*\*", r"\1", stripped)
            plain = re.sub(r"\*(.*?)\*", r"\1", plain)
            pdf.multi_cell(0, 6, plain)
        else:
            pdf.ln(3)

    # Footer
    pdf.set_y(-18)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 5, "This report is for educational purposes only and does not constitute SEBI-registered investment advice.", align="C")

    return pdf.output()


@tool
async def generate_investment_report(title: str, content: str, ticker: str = "", format: str = "pdf") -> str:
    """Generate a downloadable investment analysis report (PDF or markdown).

    Args:
        title: Report title, e.g. "RELIANCE.NS Investment Thesis — Q4 2026".
        content: Full markdown content of the report. Use ## for sections,
                 > for mentor takeaways, #### for metric groups.
        ticker: The main ticker symbol (optional, used in filename).
        format: "pdf" or "markdown". Defaults to "pdf".
    """
    from database.mongo import MongoDB

    file_id = uuid.uuid4().hex
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = _slugify(ticker or title)

    if format == "pdf":
        filename = f"{timestamp}_{slug}_report.pdf"
    else:
        filename = f"{timestamp}_{slug}_report.md"

    try:
        if format == "pdf":
            file_bytes = _create_pdf_bytes(title, content)
        else:
            file_bytes = f"# {title}\n\n{content}".encode("utf-8")

        await MongoDB.store_file(
            file_id=file_id,
            filename=filename,
            data=file_bytes,
            file_type="investment_report",
        )

        logger.info("Generated investment report: file_id='%s', format='%s', size=%d bytes",
                    file_id, format, len(file_bytes))

        return (
            f"Investment report generated!\n\n"
            f"**Title:** {title}\n"
            f"**Format:** {format.upper()}\n"
            f"**Download:** [Download Report: {title}]({_BASE_URL}/download/{file_id})"
        )

    except Exception as e:
        logger.error("Failed to generate investment report: %s", e)
        return (
            f"Error generating report ({format}): {e}. "
            "If format was 'pdf', retry with format='markdown'."
        )
