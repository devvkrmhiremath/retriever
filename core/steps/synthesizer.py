import os
from typing import List
from core.models import PipelineContext, SearchResult
from utils.azure_clients import clients

async def generate_answer(context: PipelineContext) -> PipelineContext:
    """
    Generates a grounding-driven, citation-rich answer in HTML format.
    """
    
    system_prompt = """You are a highly precise Enterprise Knowledge Assistant.
Your task is to synthesize an answer to the user's question based EXCLUSIVELY on the provided context.

STRICT CONSTRAINTS:
1. GROUNDING: If the answer is not in the context, state "I do not have enough information."
2. CITATIONS: Every claim MUST have a citation to the specific [REF X] source.
3. FORMAT: Output valid semantic HTML only (h1, h2, ul, li, p, strong).
4. NO MARKDOWN: Do not use ```html or markdown syntax. Just the raw HTML tags.
5. STYLE: Professional, concise, and structured.
"""

    client = clients.openai_client
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
    
    # Efficient context representation
    context_data = []
    for i, res in enumerate(context.final_context, 1):
        context_data.append(f"[REF {i}] (Source: {res.source_index})\nContent: {res.content}")
    
    full_context_str = "\n\n---\n\n".join(context_data)
    
    user_prompt = f"USER QUESTION: {context.query}\n\nPROVIDED CONTEXT:\n{full_context_str}"
    
    response = await client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2 # Lower temperature for higher precision
    )
    
    context.html_answer = response.choices[0].message.content
    return context

async def generate_answer_stream(context: PipelineContext):
    """
    Streams the grounding-driven answer back to the UI.
    """
    system_prompt = """You are a highly precise Enterprise Knowledge Assistant.
Your task is to synthesize an answer to the user's question based EXCLUSIVELY on the provided context.

STRICT CONSTRAINTS:
1. GROUNDING: If the answer is not in the context, state "I do not have enough information."
2. CITATIONS: Every claim MUST have a citation to the specific [REF X] source.
3. FORMAT: Output valid semantic HTML only (h1, h2, ul, li, p, strong).
4. NO MARKDOWN: Do not use ```html or markdown syntax. Just the raw HTML tags.
5. STYLE: Professional, concise, and structured.
"""

    client = clients.openai_client
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
    
    context_data = []
    for i, res in enumerate(context.final_context, 1):
        context_data.append(f"[REF {i}] (Source: {res.source_index})\nContent: {res.content}")
    
    full_context_str = "\n\n---\n\n".join(context_data)
    user_prompt = f"USER QUESTION: {context.query}\n\nPROVIDED CONTEXT:\n{full_context_str}"
    
    response = await client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        stream=True
    )
    
    full_answer = ""
    async for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            full_answer += content
            yield content
            
    context.html_answer = full_answer
