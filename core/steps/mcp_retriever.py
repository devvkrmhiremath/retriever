import json
import os
import httpx
import logging
from core.models import PipelineContext, SearchResult, MCPServerConfig
from typing import List

async def call_mcp_server(server_config: MCPServerConfig, query: str) -> List[SearchResult]:
    """
    Calls an MCP server's search/retrieve tool. 
    This is a generic implementation using the MCP HTTP protocol patterns.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Note: For real MCP servers, we'd use the mcp-python-sdk to interact 
            # with specific resources/tools. Here we simulate the pattern.
            payload = {
                "method": "tools/call",
                "params": {
                    "name": "search",
                    "arguments": {"query": query}
                }
            }
            
            # Placeholder URL - would be the specialized MCP bridge or direct SSE endpoint
            response = await client.post(server_config.url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            results = data.get("result", {}).get("content", [])
            
            mapped = []
            for i, item in enumerate(results):
                mapped.append(SearchResult(
                    id=f"mcp-{server_config.name}-{i}",
                    content=item.get("text", str(item)),
                    score=1.0, # MCP servers often don't have normalized relevance scores
                    source_index=server_config.name,
                    source_type="MCP"
                ))
            return mapped
            
    except Exception as e:
        logging.error(f"MCP Server {server_config.name} call failed: {e}")
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
