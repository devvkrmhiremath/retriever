from typing import List, Dict, Any
from core.models import PipelineContext, SearchIndexConfig, MCPServerConfig
from core.steps.query_rewriter import rewrite_query
from core.steps.embedder import generate_embedding
from core.steps.router import select_indexes, determine_retrieval_depth
from core.steps.retriever import hybrid_search_all
from core.steps.mcp_retriever import retrieve_from_all_mcp
from core.steps.ranker import apply_rrf, remove_duplicates, maximal_marginal_relevance
from core.steps.synthesizer import generate_answer
from utils.azure_clients import clients

class RAGOrchestrator:
    """Orchestrates the Elite Azure RAG Pipeline with MCP + Smart Routing."""
    
    async def run(self, 
                  query: str, 
                  indexes: List[Dict[str, Any]], 
                  mcp_servers: List[Dict[str, Any]] = [], 
                  max_context_chunks: int = 15) -> str:
        
        # 0. Context Setup
        idx_configs = [SearchIndexConfig(**idx) for idx in indexes]
        mcp_configs = [MCPServerConfig(**srv) for srv in mcp_servers]
        
        context = PipelineContext(
            query=query, 
            indexes=idx_configs, 
            mcp_servers=mcp_configs, 
            max_context_chunks=max_context_chunks
        )
        
        try:
            # 1. Query Rewrite (Precision starts with optimized parameters)
            await rewrite_query(context)
            
            # 2. Embedding Generation (High-Dimensional text-embedding-3-large)
            await generate_embedding(context)
            
            # 3-4. Smart Routing & Scaling (Identifying Core Indexes vs MCP Servers)
            await select_indexes(context)
            await determine_retrieval_depth(context)
            
            # 5a. Parallel Hybrid Search (Azure AI Search - Semantic Reranking tier)
            import asyncio
            search_task = hybrid_search_all(context)
            
            # 5b. Parallel MCP Retrieval (Open-Source Knowledge retrieval)
            mcp_task = retrieve_from_all_mcp(context)
            
            # Parallel execution of both retrieval clusters
            await asyncio.gather(search_task, mcp_task)
            
            # 6-8. Intelligence Cluster (Fuse + Deduplicate + Diversify)
            apply_rrf(context) # Reciprocal Rank Fusion for broad cross-source relevance
            remove_duplicates(context)
            maximal_marginal_relevance(context, lambda_param=0.6) # High diversity for 20+ sources
            
            # 9. Final Answer Synthesis (Strictly Grounded HTML with Citations)
            await generate_answer(context)
            
            return context.html_answer
            
        except Exception as e:
            import logging
            logging.error(f"Pipeline flow error: {str(e)}")
            raise
