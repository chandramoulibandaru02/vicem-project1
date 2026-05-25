import asyncio
import logging
from typing import AsyncIterator, Iterable

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pydantic import SecretStr
from sentence_transformers import SentenceTransformer



class SentenceTransformerEmbeddingFunction(Embeddings):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("ecm_ai_backend")
        self.model = SentenceTransformer(model_name, device="cpu")

    def embed_documents(self, texts: Iterable[str]) -> list[list[float]]:
        vectors = self.model.encode(list(texts), normalize_embeddings=True)
        return vectors.astype(float).tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode([text], normalize_embeddings=True)
        return vector.astype(float)[0].tolist()

    async def aembed_documents(self, texts: Iterable[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


class RAGPipeline:
    def __init__(
        self,
        persist_directory: str = "chroma_db",
        collection_name: str = "ecm_documents",
        groq_api_key: str | None = None,
        groq_model: str = "llama3-8b-8192",
        top_k: int = 4,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or logging.getLogger("ecm_ai_backend")
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.top_k = top_k
        self.groq_api_key = groq_api_key
        self.groq_model = groq_model

        self.embedding_function = SentenceTransformerEmbeddingFunction(logger=self.logger)
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
        )

        secret_key = SecretStr(self.groq_api_key) if self.groq_api_key else None
        self.llm = ChatGroq(
            api_key=secret_key,
            model=self.groq_model,
            temperature=0.2,
            max_tokens=512,
            streaming=True,
        )

        # Simplified prompt structure relying on custom injected text headers
        self.prompt = PromptTemplate(
            template=(
                "You are an ECM AI assistant. Use only the retrieved context to answer the user's question. \n"
                "If the answer is not in the retrieved context, say you do not have enough information to answer confidently. \n"
                "Cite every source you use using the format [filename: page_number].\n\n"
                "Context:\n{context}\n\n"
                "Question:\n{question}\n\n"
                "Answer:"
            ),
            input_variables=["context", "question"],
        )

    def _format_sources(self, documents: list[Document]) -> list[dict[str, object]]:
        sources = []
        for document in documents:
            metadata = document.metadata or {}
            sources.append(
                {
                    "filename": metadata.get("filename"),
                    "page_number": metadata.get("page_number"),
                    "chunk_id": metadata.get("chunk_id"),
                    "parent_id": metadata.get("parent_id"),
                }
            )
        return sources

    def _format_context(self, documents: list[Document]) -> str:
        """Injects metadata into the text block so the LLM can see filenames and page numbers."""
        context_parts = []
        for index, document in enumerate(documents, start=1):
            metadata = document.metadata or {}
            clean_text = document.page_content.strip().replace("\n", " ")
            source_tag = f"[Source {index} | {metadata.get('filename', 'unknown')} | page {metadata.get('page_number', 'unknown')}]"
            context_parts.append(f"{source_tag}\n{clean_text}")
        return "\n\n".join(context_parts)

    async def retrieve_context(self, query: str, top_k: int | None = None, filters: dict | None = None) -> list[Document]:
        requested_k = top_k or self.top_k
        # Utilizing native async method from Chroma
        documents = await self.vector_store.asimilarity_search(
            query,
            k=requested_k,
            filter=filters,
        )
        self.logger.info("Retrieved %s context documents for query: %s", len(documents), query)
        return documents

    async def search_documents(self, query: str, top_k: int | None = None, filters: dict | None = None) -> list[tuple[Document, float]]:
        requested_k = top_k or self.top_k
        # Utilizing native async method from Chroma
        results = await self.vector_store.asimilarity_search_with_score(
            query,
            k=requested_k,
            filter=filters,
        )
        self.logger.info("Semantic search returned %s matches for query: %s", len(results), query)
        return results

    def build_context_prompt(self, query: str, documents: list[Document]) -> str:
        context = self._format_context(documents)
        return self.prompt.format(context=context, question=query)

    async def answer_query(self, query: str, top_k: int | None = None, filters: dict | None = None) -> dict[str, object]:
        documents = await self.retrieve_context(query, top_k=top_k, filters=filters)
        
        # Build prompt string with explicit custom context headers manually
        formatted_prompt = self.build_context_prompt(query, documents)
        
        # Invoke via LLM directly since we've packed the document info into the context string manually
        response = await self.llm.ainvoke(formatted_prompt)
        
        answer = response.content if isinstance(response.content, str) else ""
        sources = self._format_sources(documents)

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": len(documents),
            "query": query,
        }

    async def stream_answer(self, query: str, top_k: int | None = None, filters: dict | None = None) -> AsyncIterator[str]:
        documents = await self.retrieve_context(query, top_k=top_k, filters=filters)
        prompt = self.build_context_prompt(query, documents)

        self.logger.info("Streaming answer for query: %s", query)
        async for chunk in self.llm.astream(prompt):
            content = getattr(chunk, "content", None)
            if isinstance(content, str):
                yield content
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        yield item
                    elif isinstance(item, dict):
                        value = item.get("content")
                        if isinstance(value, str):
                            yield value
            elif isinstance(content, dict):
                value = content.get("content")
                if isinstance(value, str):
                    yield value

        sources = self._format_sources(documents)
        if sources:
            yield "\n\nSources:\n"
            for source in sources:
                filename = source.get("filename") or "unknown"
                page_number = source.get("page_number") or "unknown"
                yield f"- {filename} (page {page_number})\n"