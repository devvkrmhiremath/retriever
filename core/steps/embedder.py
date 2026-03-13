import os
from typing import List
from core.models import PipelineContext
from utils.azure_clients import clients

async def generate_embedding(context: PipelineContext) -> PipelineContext:
    """Generates embeddings for the semantic rewrite of the query."""
    if not context.rewritten:
        raise ValueError("Context must have rewritten query before embedding.")
        
    client = clients.openai_client
    deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
    
    response = await client.embeddings.create(
        input=context.rewritten.semantic_query,
        model=deployment
    )
    
    context.embedding = response.data[0].embedding
    return context
