"""CLI tool to test document ingestion and inspect chunk provenance metadata."""
import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.services.ingestion_service import IngestionService


def main():
    parser = argparse.ArgumentParser(
        description="InsightRAG Document Ingestion CLI - Parse, clean, chunk, and inspect documents."
    )
    parser.add_argument("file_path", type=str, help="Path to document file (.pdf, .md, .txt)")
    parser.add_argument("--chunk-size", type=int, default=None, help="Target chunk size in tokens")
    parser.add_argument("--overlap", type=int, default=None, help="Chunk overlap in tokens")
    parser.add_argument("--preview-chunks", type=int, default=3, help="Number of chunks to display in detail")
    parser.add_argument("--json", action="store_true", help="Output full IngestionResult as JSON")

    args = parser.parse_args()
    path = Path(args.file_path)
    if not path.exists():
        print(f"Error: File not found: {args.file_path}", file=sys.stderr)
        sys.exit(1)

    with open(path, "rb") as f:
        file_bytes = f.read()

    service = IngestionService(
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
    )

    result = service.ingest_document(
        file_bytes=file_bytes,
        filename=path.name,
        source_path=str(path.resolve()),
    )

    if args.json:
        print(json.dumps(result.model_dump(), indent=2, default=str))
        return

    print("=" * 80)
    print(f" INSIGHTRAG DOCUMENT INGESTION SUMMARY: {result.document_name}")
    print("=" * 80)
    print(f" Document ID   : {result.document_id}")
    print(f" Source Type   : {result.source_type.value.upper()}")
    print(f" Total Chunks  : {result.total_chunks}")
    print(f" Total Tokens  : {result.total_tokens}")
    print(f" Chunk Size    : {service.chunk_size} tokens")
    print(f" Chunk Overlap : {service.chunk_overlap} tokens")
    print("-" * 80)

    num_preview = min(args.preview_chunks, len(result.chunks))
    print(f" Displaying first {num_preview} chunk(s):\n")

    for i, chunk in enumerate(result.chunks[:num_preview]):
        meta = chunk.metadata
        print(f" [{i + 1}/{num_preview}] CHUNK ID: {chunk.chunk_id}")
        print(f"     Page Number   : {meta.page_number}")
        print(f"     Section Title : {meta.section}")
        print(f"     Chunk Position: {meta.chunk_position}")
        print(f"     Token Count   : {meta.token_count} | Char Count: {meta.char_count}")
        print(f"     Preview Text  :\n     \"{chunk.text[:250]}...\"")
        print("-" * 80)


if __name__ == "__main__":
    main()

