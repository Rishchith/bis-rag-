"""
BIS Standards RAG Pipeline
Core retrieval-augmented generation pipeline for BIS standard recommendations.
"""

import os
import time
import json
import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Optional heavy imports (graceful degradation)
# ─────────────────────────────────────────────
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss-cpu not installed – falling back to numpy cosine search")

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    logger.warning("sentence-transformers not installed")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("anthropic SDK not installed")

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("pdfplumber not installed – PDF ingestion disabled")


# ─────────────────────────────────────────────
#  BIS SP-21 Knowledge Base (built-in seed)
# ─────────────────────────────────────────────
BIS_STANDARDS_DB = [
    {
        "standard_id": "IS 269",
        "title": "Ordinary Portland Cement – Specification",
        "category": "Cement",
        "keywords": ["cement", "OPC", "portland cement", "ordinary portland cement",
                     "concrete binder", "mortar binder", "grey cement"],
        "description": (
            "Specifies requirements for ordinary portland cement (OPC) used in general "
            "civil construction, concrete structures, masonry mortar, plastering, and "
            "other building applications. Covers chemical composition, physical properties "
            "like fineness, setting time, soundness, and compressive strength grades 33, 43, 53."
        ),
        "applicability": "Structural concrete, masonry, plastering, general construction",
        "part": None,
    },
    {
        "standard_id": "IS 8112",
        "title": "43 Grade Ordinary Portland Cement – Specification",
        "category": "Cement",
        "keywords": ["43 grade cement", "OPC 43", "medium strength cement",
                     "general purpose cement", "rcc", "reinforced concrete"],
        "description": (
            "Covers 43-grade OPC for reinforced concrete structures, precast elements, "
            "and general masonry. Specifies minimum 28-day compressive strength of 43 MPa, "
            "initial and final setting times, fineness by Blaine method, and soundness by "
            "Le Chatelier expansion."
        ),
        "applicability": "RCC slabs, beams, columns, precast elements",
        "part": None,
    },
    {
        "standard_id": "IS 12269",
        "title": "53 Grade Ordinary Portland Cement – Specification",
        "category": "Cement",
        "keywords": ["53 grade cement", "OPC 53", "high strength cement",
                     "prestressed concrete", "high performance concrete"],
        "description": (
            "Specifies requirements for 53-grade OPC offering higher early strength for "
            "prestressed concrete, high-rise structures, and rapid construction. Minimum "
            "28-day strength of 53 MPa. Suitable for bridges, flyovers, and industrial floors."
        ),
        "applicability": "Prestressed concrete, high-rise buildings, bridges, industrial floors",
        "part": None,
    },
    {
        "standard_id": "IS 455",
        "title": "Portland Slag Cement – Specification",
        "category": "Cement",
        "keywords": ["slag cement", "PSC", "blast furnace slag", "sulphate resistant",
                     "marine structures", "underground construction"],
        "description": (
            "Specifies Portland slag cement (PSC) manufactured by blending OPC clinker with "
            "granulated blast-furnace slag. Provides better resistance to sulphates and chlorides. "
            "Ideal for marine environments, sewage treatment plants, and foundations in aggressive soils."
        ),
        "applicability": "Marine structures, sulphate-rich soils, sewage works, retaining walls",
        "part": None,
    },
    {
        "standard_id": "IS 1489",
        "title": "Portland Pozzolana Cement – Specification",
        "category": "Cement",
        "keywords": ["PPC", "pozzolana cement", "flyash cement", "blended cement",
                     "low heat cement", "mass concrete"],
        "description": (
            "Covers Portland Pozzolana Cement (PPC) made by blending OPC clinker with "
            "fly ash or calcined clay. Generates lower heat of hydration, improving durability "
            "in mass concrete. Part 1 covers fly-ash based; Part 2 covers calcined clay based."
        ),
        "applicability": "Dams, mass concrete, foundations, general building construction",
        "part": "Part 1 (Fly Ash Based), Part 2 (Calcined Clay Based)",
    },
    {
        "standard_id": "IS 516",
        "title": "Methods of Tests for Strength of Concrete",
        "category": "Concrete",
        "keywords": ["concrete strength test", "compressive strength", "cube test",
                     "flexural strength", "concrete testing", "cube crushing"],
        "description": (
            "Specifies methods for determining compressive strength of concrete using 150mm "
            "cube specimens, flexural strength by beam tests, and indirect tensile strength. "
            "Covers curing, age of testing, and acceptance criteria for concrete grades M10–M80."
        ),
        "applicability": "Quality control of concrete, structural assessment, mix design validation",
        "part": None,
    },
    {
        "standard_id": "IS 456",
        "title": "Plain and Reinforced Concrete – Code of Practice",
        "category": "Concrete",
        "keywords": ["RCC design", "reinforced concrete", "plain concrete",
                     "concrete code", "structural concrete", "concrete mix design",
                     "concrete cover", "durability"],
        "description": (
            "The primary Indian code for design and construction of plain and reinforced concrete "
            "structures. Covers material specifications, mix design, workability, durability, "
            "structural analysis, and detailing. Mandatory reference for all RC structures in India."
        ),
        "applicability": "All reinforced and plain concrete structural elements",
        "part": None,
    },
    {
        "standard_id": "IS 10262",
        "title": "Concrete Mix Proportioning – Guidelines",
        "category": "Concrete",
        "keywords": ["mix design", "concrete proportioning", "water cement ratio",
                     "mix proportion", "target strength", "grade of concrete"],
        "description": (
            "Provides a systematic procedure for proportioning concrete mixes to achieve target "
            "mean strength. Covers selection of water-cement ratio, cement content, aggregate "
            "proportions, and admixture dosage for grades M10 to M80."
        ),
        "applicability": "Mix design for all grades of concrete",
        "part": None,
    },
    {
        "standard_id": "IS 383",
        "title": "Coarse and Fine Aggregates for Concrete – Specification",
        "category": "Aggregates",
        "keywords": ["aggregate", "sand", "gravel", "crushed stone", "fine aggregate",
                     "coarse aggregate", "river sand", "manufactured sand", "M-sand"],
        "description": (
            "Specifies requirements for natural and crushed aggregates for use in concrete. "
            "Covers grading zones for fine aggregate, nominal sizes for coarse aggregate, "
            "limits on deleterious materials, alkali-silica reactivity, and flakiness index."
        ),
        "applicability": "All concrete production requiring coarse and fine aggregates",
        "part": None,
    },
    {
        "standard_id": "IS 2386",
        "title": "Methods of Test for Aggregates for Concrete",
        "category": "Aggregates",
        "keywords": ["aggregate test", "sieve analysis", "flakiness", "elongation",
                     "impact value", "crushing value", "water absorption aggregate"],
        "description": (
            "Multi-part standard covering test methods for aggregates including particle size "
            "analysis (sieve), specific gravity, water absorption, impact value, crushing value, "
            "10% fines value, soundness, and alkali-aggregate reactivity."
        ),
        "applicability": "Laboratory testing of aggregates for quality assurance",
        "part": "Parts 1–8",
    },
    {
        "standard_id": "IS 2430",
        "title": "Methods of Sampling of Aggregates for Concrete",
        "category": "Aggregates",
        "keywords": ["aggregate sampling", "sampling procedure", "sample reduction",
                     "representative sample", "quarry sampling"],
        "description": (
            "Describes procedures for collecting representative samples of coarse and fine "
            "aggregates from stockpiles, conveyor belts, and quarry faces for testing purposes."
        ),
        "applicability": "Sampling of aggregates at quarry, stockpile, or site",
        "part": None,
    },
    {
        "standard_id": "IS 1786",
        "title": "High Strength Deformed Steel Bars and Wires for Concrete Reinforcement",
        "category": "Steel",
        "keywords": ["TMT bar", "HYSD bar", "rebars", "Fe415", "Fe500", "Fe550", "Fe600",
                     "reinforcement steel", "deformed bars", "steel for RCC"],
        "description": (
            "Specifies requirements for high strength deformed (HSD) steel bars and wires used "
            "as reinforcement in concrete structures. Covers grades Fe415, Fe500, Fe500D, Fe550, "
            "Fe550D, and Fe600. Includes tensile strength, yield strength, elongation, and bend tests."
        ),
        "applicability": "Steel reinforcement in all RC structural members",
        "part": None,
    },
    {
        "standard_id": "IS 2062",
        "title": "Hot Rolled Medium and High Tensile Structural Steel",
        "category": "Steel",
        "keywords": ["structural steel", "mild steel", "hot rolled steel", "steel plates",
                     "steel sections", "angles", "channels", "I-beams", "H-beams"],
        "description": (
            "Specifies requirements for hot-rolled structural steel plates, strips, sheets, "
            "flats, and sections. Grades E250, E300, E350, E410, E450, E550 covering chemical "
            "composition, tensile properties, impact energy, and weldability."
        ),
        "applicability": "Steel structures, industrial buildings, bridges, fabrication",
        "part": None,
    },
    {
        "standard_id": "IS 432",
        "title": "Mild Steel and Medium Tensile Steel Bars for Concrete Reinforcement",
        "category": "Steel",
        "keywords": ["mild steel bar", "plain round bar", "MS rod", "Fe250",
                     "plain bar reinforcement"],
        "description": (
            "Specifies requirements for plain mild steel (Fe250) and medium tensile steel bars "
            "used as reinforcement. Now largely replaced by IS 1786 HSD bars in new construction "
            "but still referenced for repair and legacy structures."
        ),
        "applicability": "Plain bar reinforcement, stirrups, links in older construction",
        "part": "Parts 1 & 2",
    },
    {
        "standard_id": "IS 2204",
        "title": "Construction of Reinforced Concrete Shell Roof",
        "category": "Concrete",
        "keywords": ["shell roof", "thin shell", "barrel vault", "folded plate",
                     "hyperbolic paraboloid", "curved roof"],
        "description": (
            "Covers design and construction of reinforced concrete thin shells, barrel vaults, "
            "folded plates, and hyperbolic paraboloid roofs for industrial and public buildings."
        ),
        "applicability": "Curved RC roofs, industrial sheds, large-span structures",
        "part": None,
    },
    {
        "standard_id": "IS 1343",
        "title": "Prestressed Concrete – Code of Practice",
        "category": "Concrete",
        "keywords": ["prestressed concrete", "PSC", "post-tensioning", "pre-tensioning",
                     "prestressing steel", "long span beams", "bridges prestressed"],
        "description": (
            "Code of practice for design and construction of prestressed concrete structures. "
            "Covers material requirements, loss of prestress, serviceability and ultimate limit "
            "state design, and construction requirements for pre- and post-tensioned systems."
        ),
        "applicability": "Bridges, long-span beams, railway sleepers, prestressed slabs",
        "part": None,
    },
    {
        "standard_id": "IS 4031",
        "title": "Methods of Physical Tests for Hydraulic Cement",
        "category": "Cement",
        "keywords": ["cement test", "fineness test", "setting time test", "soundness test",
                     "consistency test", "cement strength test", "Vicat test"],
        "description": (
            "Multi-part standard specifying physical test methods for hydraulic cements including "
            "fineness by Blaine, standard consistency, initial and final setting time by Vicat "
            "apparatus, soundness by Le Chatelier and autoclave, and compressive strength."
        ),
        "applicability": "Cement quality testing at plant, site, or laboratory",
        "part": "Parts 1–15",
    },
    {
        "standard_id": "IS 3812",
        "title": "Specification for Pulverised Fuel Ash",
        "category": "Cement",
        "keywords": ["fly ash", "PFA", "pulverised fuel ash", "coal ash",
                     "supplementary cementitious material", "SCM", "pozzolanа"],
        "description": (
            "Specifies two grades of fly ash (Class F and Class C) for use in concrete and "
            "cement manufacture. Covers fineness, loss on ignition, chemical composition, "
            "pozzolanity, and lime reactivity requirements."
        ),
        "applicability": "Blended concrete, PPC manufacture, geotechnical fill",
        "part": "Parts 1 & 2",
    },
    {
        "standard_id": "IS 9103",
        "title": "Concrete Admixtures – Specification",
        "category": "Concrete",
        "keywords": ["admixture", "plasticiser", "superplasticiser", "water reducer",
                     "retarder", "accelerator", "air entraining agent", "concrete additive"],
        "description": (
            "Specifies performance requirements and test methods for chemical admixtures "
            "including plasticisers, superplasticisers, retarding admixtures, accelerating "
            "admixtures, and air-entraining agents for concrete."
        ),
        "applicability": "Ready-mix concrete, high-performance concrete, self-compacting concrete",
        "part": None,
    },
    {
        "standard_id": "IS 875",
        "title": "Code of Practice for Design Loads (other than Earthquake)",
        "category": "Structural",
        "keywords": ["dead load", "live load", "wind load", "snow load", "imposed load",
                     "design loads", "load combinations", "building loads"],
        "description": (
            "Five-part code specifying design loads for buildings and structures. Part 1: Dead "
            "loads (unit weights of materials). Part 2: Imposed loads. Part 3: Wind loads. "
            "Part 4: Snow loads. Part 5: Special loads and combinations."
        ),
        "applicability": "Structural design of all buildings and civil structures",
        "part": "Parts 1–5",
    },
    {
        "standard_id": "IS 13920",
        "title": "Ductile Detailing of Reinforced Concrete Structures",
        "category": "Concrete",
        "keywords": ["seismic detailing", "ductile detailing", "earthquake resistant",
                     "confinement", "seismic zone", "lateral ties", "hoops"],
        "description": (
            "Provides ductile detailing requirements for RC structures in seismic zones. "
            "Covers confinement of columns, lap splices, anchorage, beam-column joints, "
            "and shear walls for earthquake-resistant construction."
        ),
        "applicability": "RC buildings in seismic zones II, III, IV, V",
        "part": None,
    },
]


# ─────────────────────────────────────────────
#  Chunker
# ─────────────────────────────────────────────
class BISChunker:
    """Chunk BIS standards records into overlapping text windows for embedding."""

    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_standard(self, standard: dict) -> list[dict]:
        """Create rich text chunks from a standard record."""
        chunks = []

        # Chunk 1 – dense summary
        summary = (
            f"Standard {standard['standard_id']}: {standard['title']}. "
            f"Category: {standard['category']}. "
            f"{standard['description']} "
            f"Applicable to: {standard['applicability']}."
        )
        if standard.get("part"):
            summary += f" Covers: {standard['part']}."

        chunks.append({
            "text": summary,
            "standard_id": standard["standard_id"],
            "title": standard["title"],
            "category": standard["category"],
            "chunk_type": "summary",
        })

        # Chunk 2 – keyword-focused
        kw_chunk = (
            f"{standard['standard_id']} {standard['title']} "
            f"is used for {', '.join(standard['keywords'][:8])}. "
            f"This standard belongs to the {standard['category']} category."
        )
        chunks.append({
            "text": kw_chunk,
            "standard_id": standard["standard_id"],
            "title": standard["title"],
            "category": standard["category"],
            "chunk_type": "keywords",
        })

        return chunks

    def chunk_all(self, standards: list[dict]) -> list[dict]:
        all_chunks = []
        for std in standards:
            all_chunks.extend(self.chunk_standard(std))
        return all_chunks


# ─────────────────────────────────────────────
#  Vector Store (FAISS or numpy fallback)
# ─────────────────────────────────────────────
class VectorStore:
    def __init__(self):
        self.chunks: list[dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index = None  # FAISS index if available

    def build(self, chunks: list[dict], model) -> None:
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        logger.info(f"Encoding {len(texts)} chunks …")
        embs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        self.embeddings = np.array(embs, dtype="float32")

        if FAISS_AVAILABLE:
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)  # inner-product on normalised vectors = cosine
            self.index.add(self.embeddings)
            logger.info("FAISS index built.")
        else:
            logger.info("Using numpy cosine search.")

    def search(self, query_emb: np.ndarray, top_k: int = 10) -> list[tuple[float, dict]]:
        q = np.array(query_emb, dtype="float32").reshape(1, -1)
        if FAISS_AVAILABLE and self.index is not None:
            scores, indices = self.index.search(q, top_k)
            return [(float(scores[0][i]), self.chunks[indices[0][i]])
                    for i in range(len(indices[0])) if indices[0][i] >= 0]
        else:
            # Cosine similarity via numpy
            sims = (self.embeddings @ q.T).flatten()
            top_idx = np.argsort(sims)[::-1][:top_k]
            return [(float(sims[i]), self.chunks[i]) for i in top_idx]


# ─────────────────────────────────────────────
#  BIS RAG Pipeline
# ─────────────────────────────────────────────
class BISRAGPipeline:
    """
    End-to-end RAG pipeline:
      Retriever (dense + keyword) → Reranker (MMR dedup) → LLM Generator
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k_retrieve: int = 10,
        top_k_return: int = 5,
        use_llm: bool = True,
    ):
        self.top_k_retrieve = top_k_retrieve
        self.top_k_return = top_k_return
        self.use_llm = use_llm

        # Load embedding model
        if not ST_AVAILABLE:
            raise ImportError("sentence-transformers is required. pip install sentence-transformers")
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embed_model = SentenceTransformer(embedding_model)

        # Build vector store
        chunker = BISChunker()
        chunks = chunker.chunk_all(BIS_STANDARDS_DB)
        self.store = VectorStore()
        self.store.build(chunks, self.embed_model)

        # Anthropic client
        self.client = None
        if use_llm and ANTHROPIC_AVAILABLE:
            self.client = anthropic.Anthropic()

        logger.info("BIS RAG Pipeline ready.")

    # ── Keyword boost ──────────────────────────────────────────────────────────
    def _keyword_boost(self, query: str, standard_id: str) -> float:
        """Return a small additive boost if query words appear in standard keywords."""
        q_words = set(query.lower().split())
        for std in BIS_STANDARDS_DB:
            if std["standard_id"] == standard_id:
                kw_words = set(" ".join(std["keywords"]).lower().split())
                overlap = len(q_words & kw_words)
                return overlap * 0.02
        return 0.0

    # ── MMR deduplication ──────────────────────────────────────────────────────
    def _mmr_dedup(
        self, candidates: list[tuple[float, dict]], top_n: int
    ) -> list[tuple[float, dict]]:
        """Return top_n unique standard IDs by highest score (simple dedup)."""
        seen_ids: set[str] = set()
        result = []
        for score, chunk in candidates:
            sid = chunk["standard_id"]
            if sid not in seen_ids:
                seen_ids.add(sid)
                result.append((score, chunk))
            if len(result) >= top_n:
                break
        return result

    # ── LLM Generation ────────────────────────────────────────────────────────
    def _generate_rationale(self, query: str, standards: list[dict]) -> str:
        if self.client is None:
            return ""
        context_parts = []
        for std in standards:
            context_parts.append(
                f"- {std['standard_id']}: {std['title']}\n  {std['description']}"
            )
        context = "\n".join(context_parts)

        prompt = f"""You are a BIS (Bureau of Indian Standards) compliance expert helping Indian MSEs.

Product/Query: {query}

Relevant BIS Standards:
{context}

For each standard above, write ONE concise sentence explaining WHY it is relevant to the given product description.
Format your response as a JSON array:
[
  {{"standard_id": "IS XXX", "rationale": "...one sentence..."}},
  ...
]
Return ONLY the JSON array, no other text."""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    # ── Main query method ──────────────────────────────────────────────────────
    def query(self, product_description: str) -> dict:
        t0 = time.time()

        # 1. Embed query
        q_emb = self.embed_model.encode(
            [product_description], normalize_embeddings=True, show_progress_bar=False
        )[0]

        # 2. Retrieve
        candidates = self.store.search(q_emb, top_k=self.top_k_retrieve)

        # 3. Keyword boost
        boosted = [(score + self._keyword_boost(product_description, chunk["standard_id"]), chunk)
                   for score, chunk in candidates]
        boosted.sort(key=lambda x: x[0], reverse=True)

        # 4. MMR dedup → top results
        top_results = self._mmr_dedup(boosted, self.top_k_return)

        # 5. Build output standards list
        std_ids = [r[1]["standard_id"] for r in top_results]
        std_map = {s["standard_id"]: s for s in BIS_STANDARDS_DB}
        retrieved = [std_map[sid] for sid in std_ids if sid in std_map]

        # 6. LLM rationale (best-effort)
        rationales = {}
        if self.use_llm and self.client:
            try:
                raw = self._generate_rationale(product_description, retrieved)
                import re
                json_match = re.search(r"\[.*\]", raw, re.DOTALL)
                if json_match:
                    for item in json.loads(json_match.group()):
                        rationales[item["standard_id"]] = item["rationale"]
            except Exception as e:
                logger.warning(f"LLM rationale failed: {e}")

        latency = time.time() - t0

        # 7. Format response
        results = []
        for i, std in enumerate(retrieved):
            results.append({
                "rank": i + 1,
                "standard_id": std["standard_id"],
                "title": std["title"],
                "category": std["category"],
                "relevance_score": round(top_results[i][0], 4),
                "rationale": rationales.get(
                    std["standard_id"],
                    f"Relevant to {std['applicability'].split(',')[0].strip()}.",
                ),
                "applicability": std["applicability"],
            })

        return {
            "query": product_description,
            "retrieved_standards": [r["standard_id"] for r in results],
            "detailed_results": results,
            "latency_seconds": round(latency, 3),
        }
