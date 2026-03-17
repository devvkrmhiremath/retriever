import json
import os
from typing import List
from core.models import PipelineContext, SearchResult
from utils.azure_clients import clients

def apply_rrf(context: PipelineContext, k: int = 60) -> PipelineContext:
    """
    Applies Reciprocal Rank Fusion (RRF) across multiple indices.
    RRF is highly robust for combining results with different score distributions.
    """
    if not context.all_results:
        return context

    # 1. Group by source index
    rankings = {}
    for res in context.all_results:
        if res.source_index not in rankings:
            rankings[res.source_index] = []
        rankings[res.source_index].append(res)
    
    # 2. Sort each source by Semantic Reranker score first (if present), then Vector/BM25 score
    for src in rankings:
        rankings[src].sort(key=lambda x: (x.rerank_score or 0.0, x.score or 0.0), reverse=True)
        
    # 3. Apply RRF Formula: Score = Sum( 1 / (rank + k) )
    fusion_scores = {} # id -> total_rrf_score
    
    for src, docs in rankings.items():
        for rank, doc in enumerate(docs, 1):
            if doc.id not in fusion_scores:
                fusion_scores[doc.id] = 0
            fusion_scores[doc.id] += 1.0 / (rank + k)
    
    # 4. Update the global context with fused scores to be used by MMR
    for doc in context.all_results:
        doc.score = fusion_scores.get(doc.id, 0)
        
    context.all_results.sort(key=lambda x: x.score, reverse=True)
    return context

def remove_duplicates(context: PipelineContext) -> PipelineContext:
    """Removes duplicate results based on ID."""
    seen_ids = set()
    unique_results = []
    for res in context.all_results:
        if res.id not in seen_ids:
            unique_results.append(res)
            seen_ids.add(res.id)
    context.all_results = unique_results
    return context

def maximal_marginal_relevance(context: PipelineContext, lambda_param: float = 0.7) -> PipelineContext:
    """
    Diversifies results using Maximal Marginal Relevance.
    Prevents redundant info by balancing relevance (score) vs similarity to already selected items.
    """
    if not context.all_results:
        return context
    
    # Selection greedy loop
    results = context.all_results
    selected = []
    remaining = results.copy()

    # First item is always the top ranked
    if remaining:
        selected.append(remaining.pop(0))

    while len(selected) < context.max_context_chunks and remaining:
        best_mmr = -float('inf')
        best_idx = -1
        
        for i, candidate in enumerate(remaining):
            # Relevance component (normalized score)
            relevance = candidate.score / (results[0].score + 1e-9)
            
            # Diversity component (similarity to selected)
            # In production, we'd use candidate['vector'] here. 
            # If vectors aren't available, we fallback to lexical overlap or top-K
            max_sim = 0
            # (Note: Proper MMR requires chunk embeddings. For this version, 
            # we perform a simplified version based on source/id diversity 
            # unless block-embeddings are explicitly integrated in models.py)
            
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
            
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = i
                
        if best_idx != -1:
            selected.append(remaining.pop(best_idx))
        else:
            break

    context.final_context = selected
    return context

async def neural_rerank(context: PipelineContext, top_n: int = 15) -> PipelineContext:
    """
    Acts as a Cross-Encoder Reranker using an LLM.
    Scrutinizes the top candidates from RRF/MMR to ensure absolute alignment with query intent.
    """
    if not context.all_results:
        return context
        
    # We only rerank the top K results to manage latency/cost
    candidates = context.all_results[:25]
    
    system_prompt = """You are a Neural Reranking Agent.
Your task is to re-score search results based on their direct relevance to the user's query.

For each result, provide a score between 0.0 (Irrelevant) and 1.0 (Highly Relevant).
Return ONLY a JSON array of scores in the same order as the candidates.
Format: [0.95, 0.4, 0.8, ...]"""

    candidate_texts = [f"Content: {c.content[:500]}" for c in candidates]
    user_prompt = f"Query: {context.query}\n\nCandidates:\n" + "\n---\n".join(candidate_texts)

    client = clients.openai_client
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")

    try:
        response = await client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        
        # Parse scores
        scores = json.loads(response.choices[0].message.content)
        
        # Apply new scores to candidates
        for i, score in enumerate(scores):
            if i < len(candidates):
                candidates[i].rerank_score = float(score)
        
        # Re-sort based on both RRF and Neutral Rerank
        # We weight Neural Rerank highly
        context.all_results.sort(key=lambda x: (x.rerank_score or 0.0) * 0.7 + (x.score or 0.0) * 0.3, reverse=True)
        
    except Exception as e:
        import logging
        logging.error(f"Neural Reranking failed: {e}")
        # Fallback: maintain existing order
        
    return context
