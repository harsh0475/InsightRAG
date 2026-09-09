"""PostgreSQL + pgvector vector store implementation with HNSW cosine indexing."""
import json
import logging
from typing import List, Optional
import psycopg
from pgvector.psycopg import register_vector

from backend.app.core.config import get_settings
from backend.app.schemas.document import ChunkMetadata, DocumentChunk
from backend.app.schemas.retrieval import VectorSearchResult
from backend.app.services.vector_store.base import BaseVectorStore

logger = logging.getLogger("insightrag.vector_store.pgvector")


class PGVectorStore(BaseVectorStore):
    """Production vector store using PostgreSQL and the pgvector extension with HNSW."""

    def __init__(self, connection_url: Optional[str] = None, dimension: Optional[int] = None):
        settings = get_settings()
        self.connection_url = connection_url or settings.DATABASE_URL
        # Normalize psycopg url format
        if self.connection_url.startswith("postgresql+psycopg://"):
            self.connection_url = self.connection_url.replace("postgresql+psycopg://", "postgresql://")
        elif self.connection_url.startswith("postgresql+asyncpg://"):
            self.connection_url = self.connection_url.replace("postgresql+asyncpg://", "postgresql://")

        self.dimension = dimension or settings.EMBEDDING_DIMENSION

    def _get_connection(self) -> psycopg.Connection:
        """Open a database connection with pgvector registered."""
        conn = psycopg.connect(self.connection_url, autocommit=True)
        register_vector(conn)
        return conn

    def initialize_schema(self) -> None:
        """Create the vector extension, document_chunks table, and HNSW index."""
        logger.info(f"Initializing pgvector schema (dimension={self.dimension})")
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # 1. Enable pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                # 2. Create chunks table
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        chunk_id VARCHAR(64) PRIMARY KEY,
                        document_id VARCHAR(64) NOT NULL,
                        document_name VARCHAR(255) NOT NULL,
                        source TEXT NOT NULL,
                        page_number INTEGER,
                        section TEXT,
                        chunk_position INTEGER NOT NULL,
                        token_count INTEGER NOT NULL,
                        char_count INTEGER NOT NULL,
                        text TEXT NOT NULL,
                        embedding vector({self.dimension}) NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )

                # 3. Create HNSW index for sub-millisecond approximate nearest-neighbor search
                # Cosine distance operator is vector_cosine_ops (<=>)
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding_hnsw
                    ON document_chunks
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64);
                    """
                )

                # 4. Create B-Tree index on document_id for fast metadata filtering and deletes
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_document_chunks_doc_id
                    ON document_chunks (document_id);
                    """
                )
        logger.info("pgvector schema and HNSW index initialized successfully.")

    def index_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> int:
        """Insert or update chunks and their embedding vectors in PostgreSQL."""
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks count does not match embeddings count.")

        upsert_sql = """
            INSERT INTO document_chunks (
                chunk_id, document_id, document_name, source, page_number,
                section, chunk_position, token_count, char_count, text, embedding
            ) VALUES (
                %(chunk_id)s, %(document_id)s, %(document_name)s, %(source)s, %(page_number)s,
                %(section)s, %(chunk_position)s, %(token_count)s, %(char_count)s, %(text)s, %(embedding)s
            )
            ON CONFLICT (chunk_id) DO UPDATE SET
                text = EXCLUDED.text,
                embedding = EXCLUDED.embedding,
                page_number = EXCLUDED.page_number,
                section = EXCLUDED.section;
        """

        rows = []
        for chunk, emb in zip(chunks, embeddings):
            meta = chunk.metadata
            rows.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": meta.document_id,
                    "document_name": meta.document_name,
                    "source": meta.source,
                    "page_number": meta.page_number,
                    "section": meta.section,
                    "chunk_position": meta.chunk_position,
                    "token_count": meta.token_count,
                    "char_count": meta.char_count,
                    "text": chunk.text,
                    "embedding": emb,
                }
            )

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(upsert_sql, rows)

        logger.info(f"Indexed {len(rows)} chunks in PostgreSQL pgvector.")
        return len(rows)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
    ) -> List[VectorSearchResult]:
        """Query top-K nearest neighbors using cosine distance (<=>) and HNSW."""
        # 1 - (embedding <=> query_embedding) converts cosine distance to cosine similarity
        search_sql = """
            SELECT
                chunk_id,
                document_id,
                document_name,
                source,
                page_number,
                section,
                chunk_position,
                token_count,
                char_count,
                text,
                1 - (embedding <=> %(qvec)s) AS score
            FROM document_chunks
            WHERE (%(doc_ids)s::text[] IS NULL OR document_id = ANY(%(doc_ids)s))
            ORDER BY embedding <=> %(qvec)s
            LIMIT %(limit)s;
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    search_sql,
                    {
                        "qvec": query_embedding,
                        "doc_ids": document_ids if document_ids else None,
                        "limit": top_k,
                    },
                )
                records = cur.fetchall()

        results: List[VectorSearchResult] = []
        for r in records:
            (
                chunk_id,
                doc_id,
                doc_name,
                source,
                page_num,
                section,
                chunk_pos,
                token_cnt,
                char_cnt,
                text,
                raw_score,
            ) = r
            # Score is cosine similarity [0.0, 1.0]
            score = max(0.0, float(raw_score))
            meta = ChunkMetadata(
                document_id=doc_id,
                document_name=doc_name,
                source=source,
                page_number=page_num,
                section=section,
                chunk_id=chunk_id,
                chunk_position=chunk_pos,
                token_count=token_cnt,
                char_count=char_cnt,
            )
            results.append(
                VectorSearchResult(
                    chunk_id=chunk_id,
                    text=text,
                    document_id=doc_id,
                    document_name=doc_name,
                    metadata=meta,
                    score=round(score, 4),
                )
            )

        return results

    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        sql = """
            SELECT chunk_id, document_id, document_name, source, page_number,
                   section, chunk_position, token_count, char_count, text
            FROM document_chunks WHERE chunk_id = %(cid)s;
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, {"cid": chunk_id})
                row = cur.fetchone()
                if not row:
                    return None
                meta = ChunkMetadata(
                    document_id=row[1],
                    document_name=row[2],
                    source=row[3],
                    page_number=row[4],
                    section=row[5],
                    chunk_id=row[0],
                    chunk_position=row[6],
                    token_count=row[7],
                    char_count=row[8],
                )
                return DocumentChunk(chunk_id=row[0], text=row[9], metadata=meta)

    def delete_document(self, document_id: str) -> int:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM document_chunks WHERE document_id = %(did)s;", {"did": document_id})
                deleted = cur.rowcount
        return deleted

    def count_chunks(self) -> int:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM document_chunks;")
                return cur.fetchone()[0]

