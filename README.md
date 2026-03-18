# agent-financials

Financial analysis agent that uses MCP servers for tools and exposes itself via both a REST API and an A2A protocol server.

## What It Does

- Expert financial analyst and investing mentor for Indian BSE/NSE and global markets
- Fetches real-time market data, interprets financial statements
- Provides actionable investment insights with clear "Bottom Line" recommendations
- Long-term user memory via Mem0 across sessions
- Conversation persistence via MongoDB

## Architecture

- **Tools** served remotely via 3 MCP servers (web-search:8010, finance-data:8011, vector-db:8012)
- **Agent** built on agent-sdk (`BaseAgent` with LangGraph)
- **LLM Provider:** NVIDIA
- **A2A server** mounted at `/a2a` for agent-to-agent communication
- **REST API** for direct usage

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/ask` | Send a financial query |
| GET | `/history/{session_id}` | Retrieve conversation history |
| GET | `/health` | Health check |
| — | `/a2a` | A2A protocol endpoint (used by marketplace) |

## A2A Agent Card

- **Skills:** stock-analysis, financial-reports, financial-research
- **URL:** `http://localhost:9001`

## Structure

```
agent-financials/
├── app.py              # FastAPI + A2A server (port 8080)
├── pyproject.toml
├── agents/
│   └── agent.py        # Financial agent with MCP config + Mem0 integration
├── a2a/
│   ├── agent_card.py   # A2A Agent Card definition
│   ├── executor.py     # FinancialAgentExecutor
│   └── server.py       # A2A Starlette app builder
├── database/
│   ├── mongo.py        # MongoDB conversation storage
│   └── memory.py       # Mem0 long-term memory
└── agent-sdk/          # Shared agent framework (submodule)
```

## Prerequisites

MCP tool servers must be running on ports 8010, 8011, 8012.

## Running

```bash
infisical run -- uvicorn app:app --host 0.0.0.0 --port 8080
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `NVIDIA_API_KEY` | NVIDIA API key (LLM + embeddings) |
| `MONGO_URI` | MongoDB connection string |
| `MEM0_API_KEY` | Mem0 API key for long-term memory |

## Dependencies

`agent-sdk`, `motor`, `fastapi`, `uvicorn`, `mem0ai`, `a2a-sdk`, `langchain-mcp-adapters`
