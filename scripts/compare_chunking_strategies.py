"""Academic document chunking strategy comparison script for ResearchMate.

Compares fixed-size word chunking vs recursive character chunking
across multiple academic PDF documents.
"""

import io
import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, List
from pypdf import PdfReader, PdfWriter

from backend.app.services.documents.chunker import ChunkingStrategy, DocumentChunker
from backend.app.services.documents.loader import DocumentLoader


def _create_minimal_pdf_page(text: str) -> bytes:
    """Generate a single-page PDF with text content."""
    stream = f"BT /F1 12 Tf 72 712 Td ({text}) Tj ET"
    stream_len = len(stream)
    pdf = f"""%PDF-1.4
1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj
2 0 obj <</Type /Pages /Kids [3 0 R] /Count 1>> endobj
3 0 obj <</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R>> endobj
4 0 obj <</Type /Font /Subtype /Type1 /BaseFont /Helvetica>> endobj
5 0 obj <</Length {stream_len}>> stream
{stream}
endstream endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000248 00000 n 
0000000315 00000 n 
trailer <</Size 6 /Root 1 0 R>>
startxref
410
%%EOF"""
    return pdf.encode("latin-1")


def create_multipage_pdf(pages: List[str]) -> bytes:
    """Assemble multiple pages into a multi-page PDF document."""
    writer = PdfWriter()
    for text in pages:
        single = _create_minimal_pdf_page(text)
        reader = PdfReader(io.BytesIO(single))
        writer.add_page(reader.pages[0])
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def run_comparison():
    """Execute comparative benchmark on multiple academic PDF documents."""
    papers = {
        "transformer_architecture.pdf": [
            "Abstract: We propose the Transformer model based solely on attention mechanisms [1].",
            "Section 2: The scaled dot-product attention computes Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V.",
            "Section 3: Multi-Head Attention allows the model to jointly attend to information from different representation subspaces (Vaswani et al., 2017).",
        ],
        "distributed_consensus.pdf": [
            "Abstract: Paxos and Raft are foundational protocols for consensus in distributed state machines [2].",
            "Section 2: Raft decomposes consensus into leader election, log replication, and safety guarantees.",
            "Section 3: Empirical evaluations show Raft matches Paxos throughput while reducing operational misunderstandings (Ongaro & Ousterhout, 2014).",
        ],
        "federated_learning.pdf": [
            "Abstract: Federated Learning trains machine learning models across decentralized edge devices [3].",
            "Section 2: Secure aggregation protocols guarantee user differential privacy with math bound epsilon < 0.5.",
            "Section 3: Experiments demonstrate fast convergence even under non-IID data distributions (McMahan et al., 2017).",
        ],
    }

    print("=" * 80)
    print(" ResearchMate: Academic Document Chunking Strategy Benchmark")
    print("=" * 80)

    for filename, pages in papers.items():
        pdf_bytes = create_multipage_pdf(pages)
        loaded_doc = DocumentLoader.load_bytes(pdf_bytes, filename=filename)

        print(f"\nDocument: {filename} ({loaded_doc.page_count} pages, {loaded_doc.char_count} chars)")
        print("-" * 70)

        comparison = DocumentChunker.compare_strategies(loaded_doc.raw_text, chunk_size=300, chunk_overlap=60)
        fixed = comparison["fixed_size"]
        rec = comparison["recursive"]

        print(f"{'Metric':<32} | {'Fixed-Word Chunking':<20} | {'Recursive Text Chunking':<22}")
        print("-" * 80)
        print(f"{'Total Chunks':<32} | {fixed['total_chunks']:<20} | {rec['total_chunks']:<22}")
        print(f"{'Avg Chunk Size (chars)':<32} | {fixed['avg_chunk_chars']:<20} | {rec['avg_chunk_chars']:<22}")
        print(f"{'Sentence Boundary Cleanliness':<32} | {fixed['sentence_boundary_integrity']}%{'':<15} | {rec['sentence_boundary_integrity']}%{'':<17}")
        print(f"{'Citations Preserved':<32} | {fixed['citations_detected']:<20} | {rec['citations_detected']:<22}")

    print("\n" + "=" * 80)
    print(" Selected Academic Strategy:")
    print(f" Strategy: {DocumentChunker.RECOMMENDED_ACADEMIC_STRATEGY.value.upper()}")
    print(" Rationale: Recursive text chunking respects paragraph and sentence boundaries,")
    print(" preventing truncation of citations, equations, and critical academic context.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_comparison()
