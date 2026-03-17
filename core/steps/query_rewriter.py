import json
import os
from openai import AsyncAzureOpenAI
from core.models import PipelineContext, RewrittenQuery
from utils.azure_clients import clients

async def rewrite_query(context: PipelineContext):
    """
    Optimizes the user query for hybrid retrieval.
    Generates:
    - Precise search query (Natural Language)
    - Semantic query (Optimized for Embeddings)
    - Keywords (Boolean/Lexical search)
    """
    
    prompt = """You are an expert Retrieval Engineer. 
Decompose the user question into optimized search parameters for an Azure AI Search hybrid pipeline.

Tasks:
1. search_query: A refined version of the question for semantic search.
2. semantic_query: A dense representation of the core intent.
3. keywords: A list of 5-8 specific technical terms, entities, or acronyms for keyword matching.

Return valid JSON:
{
  "search_query": "...",
  "semantic_query": "...",
  "keywords": ["...", "..."]
}"""

    client: AsyncAzureOpenAI = clients.openai_client
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
    
    response = await client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": context.query}
        ],
        response_format={"type": "json_object"},
        temperature=0.0
    )
    
    data = json.loads(response.choices[0].message.content)
    context.rewritten = RewrittenQuery(**data)
    return context
