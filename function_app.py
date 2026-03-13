import azure.functions as func
import logging
import json
from core.orchestrator import RAGOrchestrator

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="elite_rag")
async def elite_rag_pipeline(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Elite Azure RAG + MCP Pipeline triggered.')

    try:
        req_body = req.get_json()
    except ValueError:
        return func.HttpResponse(
            json.dumps({"error": "Invalid JSON body"}),
            status_code=400,
            mimetype="application/json"
        )

    query = req_body.get('query')
    indexes = req_body.get('indexes') # Expected: List[Dict] with description, category etc.
    mcp_servers = req_body.get('mcp_servers', []) # Optional: List[Dict]
    max_context_chunks = req_body.get('max_context_chunks', 15)

    if not query or not (indexes or mcp_servers):
        return func.HttpResponse(
            json.dumps({"error": "Missing 'query', 'indexes', or 'mcp_servers' in request"}),
            status_code=400,
            mimetype="application/json"
        )

    try:
        orchestrator = RAGOrchestrator()
        html_answer = await orchestrator.run(
            query=query, 
            indexes=indexes, 
            mcp_servers=mcp_servers,
            max_context_chunks=max_context_chunks
        )

        return func.HttpResponse(
            json.dumps({"answer": html_answer}),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": "Internal pipeline error", "details": str(e)}),
            status_code=500,
            mimetype="application/json"
        )
