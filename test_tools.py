import asyncio
import os
import sys

# Add current directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.yfinance_tool import get_ticker_data

async def test_yfinance():
    print("Testing YFinance Tool...")
    # Test an Indian Stock (Reliance)
    # Using the sync fallback manually if asyncio fails or just calling the tool directly
    res = await get_ticker_data.ainvoke({"ticker": "RELIANCE.NS"})
    print(f"Result for RELIANCE.NS => Price: {res.get('currentPrice')} | Market Cap: {res.get('marketCap')}")
    
    # Test a US stock
    res2 = await get_ticker_data.ainvoke({"ticker": "AAPL"})
    print(f"Result for AAPL => Price: {res2.get('currentPrice')} | Market Cap: {res2.get('marketCap')}")
    print("---")

async def main():
    try:
        await test_yfinance()
        print("Basic tests passed. Skipping LLM/Pinecone to avoid requiring API keys locally in tests.")
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
