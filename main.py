import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import uuid
import os
import datetime
from google import genai
from google.genai import types
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult

load_dotenv()

# --- Configuration ---
inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer() 
)

# --- Function 1: Ingest with 10-Minute Auto-Deletion ---
@inngest_client.create_function(
    fn_id="RAG: Ingest PDF with TTL",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context):
    # Extract event data
    pdf_path = ctx.event.data["pdf_path"]
    source_id = ctx.event.data["source_id"]

    # Step 1: Load and Chunk
    def _load():
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    chunks_and_src = await ctx.step.run("load_and_chunk", _load, output_type=RAGChunkAndSrc)

    # Step 2: Embed and Upsert to Vector DB
    def _upsert(data: RAGChunkAndSrc):
        store = QdrantStorage()
        vecs = embed_texts(data.chunks)
        
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{data.source_id}:{i}")) for i in range(len(data.chunks))]
        payloads = [{"source": data.source_id, "text": t} for t in data.chunks]

        store.upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(data.chunks))

    ingested = await ctx.step.run("embed_and_upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)

    # Step 3: Wait for 10 Minutes (FIXED)
    # datetime.timedelta is used instead of a string to avoid TypeError
    await ctx.step.sleep("wait_10_min", datetime.timedelta(minutes=10))

    # Step 4: Cleanup
    def _cleanup():
        QdrantStorage().delete_by_source(source_id)
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        return "Cleanup successful"

    await ctx.step.run("cleanup_data", _cleanup)

    return {"status": "Session expired, data wiped.", "ingested_count": ingested.ingested}


# --- Function 2: Query with Isolation ---
@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))
    source_id = ctx.event.data.get("source_id")

    if not source_id:
        return {"answer": "Error: No active document session found.", "sources": [], "num_contexts": 0}

    # Step 1: Search Vector DB
    def _search(q, k, sid):
        query_vec = embed_texts([q])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k=k, source_filter=sid)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])
    
    found = await ctx.step.run("embed_and_search", lambda: _search(question, top_k, source_id), output_type=RAGSearchResult)

    if not found.contexts:
        return {
            "answer": "I cannot answer this question because the document context is missing or the session has expired.",
            "sources": [],
            "num_contexts": 0
        }

    # Step 2: Generate Answer
    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    
    def _generate_answer(q, context):
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        
        system_instruction = "You are a helpful assistant. Answer questions using only the provided context."
        
        user_prompt = (
            "Use the following context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {q}\n"
            "Answer concisely using the context above."
        )

        response = client.models.generate_content(
            model="gemma-3-27b", 
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1,
                max_output_tokens=1500,
            )
        )

        return response.text
    
    answer = await ctx.step.run("llm_answer", lambda: _generate_answer(question, context_block))

    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}

# --- FastAPI App Definition ---
app = FastAPI()

inngest.fast_api.serve(app, inngest_client, functions=[rag_ingest_pdf, rag_query_pdf_ai])
