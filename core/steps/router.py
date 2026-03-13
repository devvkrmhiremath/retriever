import json
import os
from core.models import PipelineContext, SearchIndexConfig, MCPServerConfig
from utils.azure_clients import clients

async def select_indexes(context: PipelineContext) -> PipelineContext:
    """
    Intelligent Router to identify core indexes and MCP servers.
    Handles 'Index Overload' (15-20+ indexes) by selecting only the top 3-5 most relevant.
    """
    
    # 1. Prepare system prompt with all index and server metadata
    index_meta = [{
        "name": idx.name, 
        "description": idx.description, 
        "category": idx.category
    } for idx in context.indexes]
    
    mcp_meta = [{
        "name": srv.name, 
        "description": srv.description
    } for srv in context.mcp_servers]

    system_prompt = """You are a High-Performance Retrieval Router.
Your goal is to optimize precision and recall for a huge-scale RAG system (20+ sources).

Task:
From the provided source lists, select only the most relevant ones to answer the user's query.
Avoid selecting more than 5 total sources to keep the retrieval window high-quality and fast.

Rules:
1. Prioritize 'Core' indexes if the query is broad.
2. Select 'Specialized' indexes for technical/specific queries.
3. Include MCP servers only if the query likely needs their domain knowledge.

Return JSON format:
{
  "selected_indexes": ["name1", "name2"],
  "selected_mcp_servers": ["server1"]
}"""

    input_data = f"Query: {context.rewritten.search_query}\n\nAvailable Indexes:\n{json.dumps(index_meta)}\n\nAvailable MCP Servers:\n{json.dumps(mcp_meta)}"

    client = clients.openai_client
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
    
    response = await client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_data}
        ],
        response_format={"type": "json_object"},
        temperature=0.0
    )
    
    data = json.loads(response.choices[0].message.content)
    sel_idx = data.get("selected_indexes", [])
    sel_mcp = data.get("selected_mcp_servers", [])
    
    # Map back to full objects
    context.routed_indexes = [idx for idx in context.indexes if idx.name in sel_idx]
    context.routed_mcp_servers = [srv for srv in context.mcp_servers if srv.name in sel_mcp]
    
    # Safety Fallback
    if not context.routed_indexes and context.indexes:
        context.routed_indexes = context.indexes[:1]
        
    return context

async def determine_retrieval_depth(context: PipelineContext) -> PipelineContext:
    """Determines retrieval depth (k) based on query and index count."""
    
    prompt = """Determine per-index retrieval depth (k).
Simple question → 10
Broad research → 30
Deep technical inquiry → 50
Return ONLY the number."""

    client = clients.openai_client
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
    
    response = await client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Query: {context.query}"}
        ]
    )
    
    try:
        context.retrieval_k = int(response.choices[0].message.content.strip())
    except ValueError:
        context.retrieval_k = 20
        
    return context
