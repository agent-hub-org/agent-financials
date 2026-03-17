import asyncio
import logging
from typing import List, Dict, Any
import pandas as pd
import yfinance as yf
from langchain_core.tools import tool

logger = logging.getLogger("agent_financials.bse_nse_reports")


def _get_financial_reports_sync(ticker: str) -> List[Dict[str, Any]]:
    logger.info("Fetching financial reports for ticker='%s'", ticker)
    results = []
    
    try:
        t = yf.Ticker(ticker)
        
        # Yearly Income Statement
        try:
            inc_stmt = t.income_stmt
            if not inc_stmt.empty:
                markdown_str = inc_stmt.to_markdown()
                results.append({
                    "title": f"{ticker} Yearly Income Statement",
                    "content": f"Yearly Income Statement for {ticker}:\n\n{markdown_str}",
                    "period": "Yearly",
                    "type": "Income Statement"
                })
        except Exception as e:
            logger.warning("Failed to fetch yearly income statement for %s: %s", ticker, e)

        # Quarterly Income Statement
        try:
            q_inc_stmt = t.quarterly_income_stmt
            if not q_inc_stmt.empty:
                markdown_str = q_inc_stmt.to_markdown()
                results.append({
                    "title": f"{ticker} Quarterly Income Statement",
                    "content": f"Quarterly Income Statement for {ticker}:\n\n{markdown_str}",
                    "period": "Quarterly",
                    "type": "Income Statement"
                })
        except Exception as e:
            logger.warning("Failed to fetch quarterly income statement for %s: %s", ticker, e)

        # Yearly Balance Sheet
        try:
            bal_sheet = t.balance_sheet
            if not bal_sheet.empty:
                markdown_str = bal_sheet.to_markdown()
                results.append({
                    "title": f"{ticker} Yearly Balance Sheet",
                    "content": f"Yearly Balance Sheet for {ticker}:\n\n{markdown_str}",
                    "period": "Yearly",
                    "type": "Balance Sheet"
                })
        except Exception as e:
            logger.warning("Failed to fetch yearly balance sheet for %s: %s", ticker, e)

        # Quarterly Balance Sheet
        try:
            q_bal_sheet = t.quarterly_balance_sheet
            if not q_bal_sheet.empty:
                markdown_str = q_bal_sheet.to_markdown()
                results.append({
                    "title": f"{ticker} Quarterly Balance Sheet",
                    "content": f"Quarterly Balance Sheet for {ticker}:\n\n{markdown_str}",
                    "period": "Quarterly",
                    "type": "Balance Sheet"
                })
        except Exception as e:
            logger.warning("Failed to fetch quarterly balance sheet for %s: %s", ticker, e)

        # Yearly Cash Flow
        try:
            cash_flow = t.cashflow
            if not cash_flow.empty:
                markdown_str = cash_flow.to_markdown()
                results.append({
                    "title": f"{ticker} Yearly Cash Flow",
                    "content": f"Yearly Cash Flow for {ticker}:\n\n{markdown_str}",
                    "period": "Yearly",
                    "type": "Cash Flow"
                })
        except Exception as e:
            logger.warning("Failed to fetch yearly cashflow for %s: %s", ticker, e)

        # Quarterly Cash Flow
        try:
            q_cash_flow = t.quarterly_cashflow
            if not q_cash_flow.empty:
                markdown_str = q_cash_flow.to_markdown()
                results.append({
                    "title": f"{ticker} Quarterly Cash Flow",
                    "content": f"Quarterly Cash Flow for {ticker}:\n\n{markdown_str}",
                    "period": "Quarterly",
                    "type": "Cash Flow"
                })
        except Exception as e:
            logger.warning("Failed to fetch quarterly cashflow for %s: %s", ticker, e)

        return results
    except Exception as e:
        logger.error("Error fetching financial reports for ticker='%s': %s", ticker, e)
        return [{"error": f"Failed to fetch reports. Error: {str(e)}"}]


@tool("get_bse_nse_reports", return_direct=True)
async def get_bse_nse_reports(ticker: str) -> List[Dict[str, Any]]:
    """
    Fetch raw quarterly and yearly financial reports (Income Statement, Balance Sheet, Cash Flow) 
    for a given ticker (use .NS or .BO suffix for Indian stocks).
    Returns a list of dictionaries with markdown-formatted tabular data.
    """
    return await asyncio.to_thread(_get_financial_reports_sync, ticker)
