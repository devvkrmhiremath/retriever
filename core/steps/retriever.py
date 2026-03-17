import asyncio
import logging
from typing import List
from azure.search.documents.models import VectorizedQuery
from core.models import PipelineContext, SearchResult, SearchIndexConfig
from utils.azure_clients import clients

# Max concurrent search requests to avoid socket exhaustion in Azure Functions
RETRIEVAL_SEMAPHORE = asyncio.Semaphore(5)
# Max time to wait for a single index before dropping it (SLA Protection)
INDEX_TIMEOUT_SECONDS = 5.0

async def search_index(index_config: SearchIndexConfig, context: PipelineContext) -> List[SearchResult]:
    """
    Performs high-precision Hybrid Search (Vector + BM25) on an Azure AI Search index.
    """
    client = clients.get_search_client(index_config.name)
    
    # 1. Configure Vector Query (text-embedding-3-large)
    vector_query = VectorizedQuery(
        vector=context.embedding,
        k_nearest_neighbors=context.retrieval_k,
        fields=index_config.vector_field,
        exhaustive=False # Set to False for HNSW; massive latency improvement for large indexes
    )
    
    async with RETRIEVAL_SEMAPHORE:
        try:
            # 2. Execute Hybrid Search with Semantic Reranking (with absolute timeout)
            results = await asyncio.wait_for(
                client.search(
                    search_text=context.rewritten.search_query,
                    vector_queries=[vector_query],
                    query_type="semantic",
                    semantic_configuration_name=index_config.semantic_config,
                    top=context.retrieval_k,
                    select=index_config.select_fields if index_config.select_fields else ["id", index_config.content_field],
                    include_total_count=True
                ),
                timeout=INDEX_TIMEOUT_SECONDS
            )
            
            mapped_results = []
            async for item in results:
                mapped_results.append(SearchResult(
                    id=str(item["id"]),
                    content=item[index_config.content_field],
                    score=item.get("@search.score", 0.0),
                    rerank_score=item.get("@search.reranker_score", 0.0),
                    source_index=index_config.name,
                    metadata=item
                ))
                
            return mapped_results
            
        except asyncio.TimeoutError:
            logging.warning(f"Index {index_config.name} timed out after {INDEX_TIMEOUT_SECONDS}s. Skipping.")
            return []
        except Exception as e:
            logging.error(f"Error searching index {index_config.name}: {e}")
            return []

async def hybrid_search_all(context: PipelineContext) -> PipelineContext:
    """Executes parallel hybrid search across all routed indexes."""
    
    tasks = [search_index(idx, context) for idx in context.routed_indexes]
    all_index_results = await asyncio.gather(*tasks)
    
    # Flatten the results
    context.all_results = [res for sublist in all_index_results for res in sublist]
    return context
