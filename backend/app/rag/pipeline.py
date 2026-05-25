import logging
from typing import AsyncIterator

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pydantic import SecretStr

from app.rag.embeddings import (
    EmbeddingGenerator,
)
from app.rag.vector_store import (
    create_vectorstore,
)


class RAGPipeline:
    def __init__(
        self,
        groq_api_key: str | None = None,
        groq_model: str = "llama3-8b-8192",
        top_k: int = 4,
        namespace: str = "default",
        logger: logging.Logger | None = None,
    ) -> None:

        self.logger = logger or logging.getLogger(
            "ecm_ai_backend"
        )

        self.top_k = top_k
        self.namespace = namespace

        # Embeddings
        self.embedding_generator = (
            EmbeddingGenerator(
                logger=self.logger
            )
        )

        # Pinecone Vector Store
        self.vector_store = create_vectorstore(
            logger=self.logger,
            namespace=namespace,
        )

        # Groq
        secret_key = (
            SecretStr(groq_api_key)
            if groq_api_key
            else None
        )

        self.llm = ChatGroq(
            api_key=secret_key,
            model=groq_model,
            temperature=0.2,
            max_tokens=1024,
            streaming=True,
        )

        # Prompt
        self.prompt = PromptTemplate(
            template="""
You are an ECM enterprise AI assistant.

Use ONLY the provided context.

If answer is not found,
say you do not have enough information.

Always cite sources:
[filename: page_number]

Context:
{context}

Question:
{question}

Answer:
""",
            input_variables=[
                "context",
                "question",
            ],
        )

    def _format_sources(
        self,
        documents: list[Document],
    ) -> list[dict]:

        sources = []

        for document in documents:

            metadata = document.metadata or {}

            sources.append(
                {
                    "filename": metadata.get(
                        "filename"
                    ),
                    "page_number": metadata.get(
                        "page_number"
                    ),
                    "chunk_id": metadata.get(
                        "chunk_id"
                    ),
                }
            )

        return sources

    def _format_context(
        self,
        documents: list[Document],
    ) -> str:

        context_parts = []

        for idx, document in enumerate(
            documents,
            start=1,
        ):

            metadata = (
                document.metadata or {}
            )

            source = (
                f"[Source {idx} | "
                f"{metadata.get('filename', 'unknown')} | "
                f"page {metadata.get('page_number', 'unknown')}]"
            )

            text = (
                document.page_content
                .strip()
                .replace("\n", " ")
            )

            context_parts.append(
                f"{source}\n{text}"
            )

        return "\n\n".join(
            context_parts
        )

    async def retrieve_context(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> list[Document]:

        requested_k = (
            top_k or self.top_k
        )

        query_embedding = (
            self.embedding_generator
            .generate_single_embedding(
                query
            )
            .tolist()
        )

        results = (
            self.vector_store.search_documents(
                query_embedding=query_embedding,
                top_k=requested_k,
                filters=filters,
            )
        )

        documents = []

        for result in results:

            metadata = result.get(
                "metadata",
                {},
            )

            documents.append(
                Document(
                    page_content=result.get(
                        "document",
                        "",
                    ),
                    metadata=metadata,
                )
            )

        self.logger.info(
            "Retrieved %s documents",
            len(documents),
        )

        return documents

    async def answer_query(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> dict:

        documents = (
            await self.retrieve_context(
                query=query,
                top_k=top_k,
                filters=filters,
            )
        )

        context = self._format_context(
            documents
        )

        formatted_prompt = (
            self.prompt.format(
                context=context,
                question=query,
            )
        )

        response = (
            await self.llm.ainvoke(
                formatted_prompt
            )
        )

        return {
            "answer": response.content,
            "sources": self._format_sources(
                documents
            ),
            "retrieved_chunks": len(
                documents
            ),
        }

    async def stream_answer(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> AsyncIterator[str]:

        documents = (
            await self.retrieve_context(
                query=query,
                top_k=top_k,
                filters=filters,
            )
        )

        context = self._format_context(
            documents
        )

        prompt = self.prompt.format(
            context=context,
            question=query,
        )

        async for chunk in self.llm.astream(
            prompt
        ):

            if hasattr(
                chunk,
                "content",
            ):

                content = chunk.content

                if isinstance(
                    content,
                    str,
                ):
                    yield content

        yield "\n\nSources:\n"

        for source in self._format_sources(
            documents
        ):

            yield (
                f"- "
                f"{source.get('filename')} "
                f"(page "
                f"{source.get('page_number')})\n"
            )