import logging

import os
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from agent_sdk.a2a.server.mongodb_task_store import AsyncMongoDBTaskStore

from .agent_card import FINANCIAL_AGENT_CARD
from .executor import FinancialAgentExecutor

logger = logging.getLogger("agent_financials.a2a_server")


def create_a2a_app() -> A2AStarletteApplication:
    """Build the A2A Starlette application for the financial agent."""
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    task_store = AsyncMongoDBTaskStore(
        conn_string=mongo_uri,
        db_name=os.getenv("MONGO_DB_NAME", "agent_financials"),
        collection_name="a2a_tasks"
    )
    executor = FinancialAgentExecutor()
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
    )
    a2a_app = A2AStarletteApplication(
        agent_card=FINANCIAL_AGENT_CARD,
        http_handler=request_handler,
    )
    logger.info("A2A application created for Financial Agent")
    return a2a_app
