import logging
import asyncio
from core.models import PipelineContext, SearchResult, MCPServerConfig
from typing import List
try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    ClientSession = None
    sse_client = None

async def call_mcp_server(server_config: MCPServerConfig, query: str) -> List[SearchResult]:
    """
    Calls an MCP server's search/retrieve tool using the official SSE client.
    This supports modern streamable HTTP MCP servers.
    """
    if not HAS_MCP:
        logging.warning(f"Skipping MCP call for {server_config.name}: 'mcp' library not installed.")
        return []
    try:
        # Connect strictly via Server-Sent Events (SSE) which is standard for HTTP MCP
        async with asyncio.timeout(5.0): # Global MCP Timeout
            async with sse_client(server_config.url) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    # Initialize the session
                    await session.initialize()
                    
                    # Execute the 'search' tool with the user's query
                    # Assumes the standard MCP tools/call convention
                    result = await session.call_tool("search", {"query": query})
                
                mapped = []
                # MCP 'call_tool' typically returns a CallToolResult whose 'content' 
                # is a list of TextContent or ImageContent objects.
                if result and hasattr(result, "content"):
                    for i, item in enumerate(result.content):
                        # item is usually text content defined by the MCP spec
                        # if item has a .text property, grab it
                        content_text = getattr(item, 'text', str(item))
                        
                        mapped.append(SearchResult(
                            id=f"mcp-{server_config.name}-{i}",
                            content=content_text,
                            score=1.0, # Semantic Reranker and RRF normalize these downstream
                            source_index=server_config.name,
                            source_type="MCP"
                        ))
                return mapped
                
    except Exception as e:
        logging.error(f"MCP Server {server_config.name} SSE call failed: {e}")
        return []

async def retrieve_from_all_mcp(context: PipelineContext) -> PipelineContext:
    """Orchestrates parallel calls to routed MCP servers."""
    import asyncio
    
    if not context.routed_mcp_servers:
        return context
        
    tasks = [call_mcp_server(mcp, context.query) for mcp in context.routed_mcp_servers]
    mcp_results = await asyncio.gather(*tasks)
    
    # Flatten and add to context
    for res_list in mcp_results:
        context.all_results.extend(res_list)
        
    return context
