# Elite Azure RAG Retriever 🚀

Enterprise-grade RAG pipeline for Azure AI Search and Model Context Protocol (MCP).

## 🌟 Key Features
- **Smart Index Routing**: Intelligent selection for massive index scales (20+).
- **Hybrid Retrieval**: Vector (text-embedding-3-large) + BM25 + Semantic Reranking.
- **Cross-Source Fusion (RRF)**: Reciprocal Rank Fusion for combining results.
- **MMR Diversity**: Maximal Marginal Relevance for context diversification.
- **MCP Integration**: Real-time retrieval from MCP-compliant servers.
- **Grounded HTML**: Final synthesis with strict grounding and citations.

## 🛠️ Architecture
- **Engine**: Python / Asyncio
- **Serverless**: Azure Functions v2 (Python)
- **Search**: Azure AI Search (Hybrid + Semantic tiers)
- **Intelligence**: Azure OpenAI (GPT-4o)

## 🚀 Getting Started
1. Configure `local.settings.json` (as per `local.settings.example.json`).
2. Run `pip install -r requirements.txt`.
3. Start local function with `func start`.

## 📡 API Spec
**POST `/api/elite_rag`**
```json
{
  "query": "What are the latest compliance changes?",
  "indexes": [
    {
      "name": "idx-compliance",
      "description": "Regulatory and legal documents",
      "category": "Core",
      "content_field": "text",
      "vector_field": "vector"
    }
  ],
  "mcp_servers": [
    {
      "name": "realtime-news",
      "url": "https://mcp-server-url",
      "description": "Real-time news retrieval"
    }
  ]
}
```
