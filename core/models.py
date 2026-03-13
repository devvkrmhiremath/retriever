from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class SearchIndexConfig(BaseModel):
    name: str
    content_field: str
    vector_field: str
    description: str = "Enterprise data index"
    category: str = "General"

class MCPServerConfig(BaseModel):
    name: str
    url: str
    description: str

class RewrittenQuery(BaseModel):
    search_query: str
    semantic_query: str
    keywords: List[str]

class SearchResult(BaseModel):
    id: str
    content: str
    score: float
    rerank_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_index: str
    source_type: str = "AzureSearch" # AzureSearch or MCP

class PipelineContext(BaseModel):
    query: str
    indexes: List[SearchIndexConfig]
    mcp_servers: List[MCPServerConfig] = Field(default_factory=list)
    max_context_chunks: int = 15
    
    # Internal intermediate state
    rewritten: Optional[RewrittenQuery] = None
    embedding: Optional[List[float]] = None
    routed_indexes: List[SearchIndexConfig] = Field(default_factory=list)
    routed_mcp_servers: List[MCPServerConfig] = Field(default_factory=list)
    retrieval_k: int = 20
    all_results: List[SearchResult] = Field(default_factory=list)
    final_context: List[SearchResult] = Field(default_factory=list)
    html_answer: Optional[str] = None
