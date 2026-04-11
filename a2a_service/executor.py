import logging

from agent_sdk.a2a.executor import StreamingAgentExecutor
from agents.agent import run_query, stream_for_a2a

logger = logging.getLogger("agent_financials.a2a_executor")

class FinancialAgentExecutor(StreamingAgentExecutor):
    """A2A executor that streams financial agent responses chunk-by-chunk.

    Uses StreamingAgentExecutor so the A2A SSE connection carries live data
    (progress markers + synthesis text) throughout the pipeline run, preventing
    Cloudflare 524 timeouts on long-running financial queries.
    """

    def __init__(self):
        super().__init__(run_query_fn=run_query, stream_fn=stream_for_a2a)
