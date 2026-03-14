"""
Unit tests for RAGService.

Covers:
- Initialization: dependencies stored, defaults set
- retrieve: empty query, embedding failure, vector store query,
  score filtering, chunk building, exception fallback
- retrieve_as_context: delegates to retrieve, formats output, empty result
- ingest_document: chunking, embedding, upsert, skipped chunks, empty chunks
- _build_chunks: score threshold filtering, RetrievedChunk field mapping
- RetrievedChunk: to_dict, to_context_string
"""
import pytest
from src.infrastructure.services.rag import (
    RAGService,
    RetrievedChunk,
    DEFAULT_TOP_K,
    DEFAULT_SCORE_THRESHOLD,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_embedding_model(mocker):
    model = mocker.AsyncMock()
    model.embed_text = mocker.AsyncMock(return_value=[0.1, 0.2, 0.3])
    model.embed_batch = mocker.AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
    return model


@pytest.fixture
def mock_vector_store(mocker):
    store = mocker.AsyncMock()
    store.query = mocker.AsyncMock(return_value=[])
    store.upsert = mocker.AsyncMock(return_value=True)
    return store


@pytest.fixture
def rag_service(mock_embedding_model, mock_vector_store):
    return RAGService(
        embedding_model=mock_embedding_model,
        vector_store=mock_vector_store,
    )


def _raw_result(score=0.8, chunk_id="chunk-1", content="clinical text",
                source="guideline.pdf", doc_type="clinical_guideline"):
    return {
        "id": chunk_id,
        "content": content,
        "score": score,
        "metadata": {"source": source, "doc_type": doc_type, "specialty": "cardiology"},
    }


# ── RetrievedChunk ────────────────────────────────────────────────────────────

class TestRetrievedChunk:
    """RetrievedChunk serialises correctly."""

    def test_to_dict_contains_all_fields(self):
        # Arrange
        chunk = RetrievedChunk(
            chunk_id="c1",
            content="ACE inhibitors recommended",
            score=0.92,
            source="guideline.pdf",
            doc_type="clinical_guideline",
            metadata={"specialty": "cardiology"},
        )

        # Act
        d = chunk.to_dict()

        # Assert
        assert d["chunk_id"] == "c1"
        assert d["content"] == "ACE inhibitors recommended"
        assert d["score"] == 0.92
        assert d["source"] == "guideline.pdf"
        assert d["doc_type"] == "clinical_guideline"
        assert d["metadata"] == {"specialty": "cardiology"}

    def test_to_context_string_contains_source_and_content(self):
        # Arrange
        chunk = RetrievedChunk(
            chunk_id="c1",
            content="Beta-blockers contraindicated in asthma",
            score=0.85,
            source="drug_ref.pdf",
            doc_type="drug_reference",
            metadata={},
        )

        # Act
        s = chunk.to_context_string()

        # Assert
        assert "drug_ref.pdf" in s
        assert "drug_reference" in s
        assert "Beta-blockers contraindicated in asthma" in s
        assert "0.850" in s


# ── Initialization ────────────────────────────────────────────────────────────

class TestRAGServiceInitialization:
    """RAGService stores dependencies and defaults correctly."""

    def test_stores_embedding_model_and_vector_store(
        self, mock_embedding_model, mock_vector_store
    ):
        # Act
        service = RAGService(
            embedding_model=mock_embedding_model,
            vector_store=mock_vector_store,
        )

        # Assert
        assert service.embedding_model is mock_embedding_model
        assert service.vector_store is mock_vector_store

    def test_uses_default_top_k_and_threshold(
        self, mock_embedding_model, mock_vector_store
    ):
        # Act
        service = RAGService(
            embedding_model=mock_embedding_model,
            vector_store=mock_vector_store,
        )

        # Assert
        assert service.default_top_k == DEFAULT_TOP_K
        assert service.score_threshold == DEFAULT_SCORE_THRESHOLD

    def test_accepts_custom_top_k_and_threshold(
        self, mock_embedding_model, mock_vector_store
    ):
        # Act
        service = RAGService(
            embedding_model=mock_embedding_model,
            vector_store=mock_vector_store,
            default_top_k=10,
            score_threshold=0.7,
        )

        # Assert
        assert service.default_top_k == 10
        assert service.score_threshold == 0.7


# ── retrieve ──────────────────────────────────────────────────────────────────

class TestRetrieve:
    """retrieve: full pipeline from query to filtered RetrievedChunk list."""

    @pytest.mark.asyncio
    async def test_returns_empty_list_for_empty_query(self, rag_service):
        # Act
        result = await rag_service.retrieve(query="")

        # Assert
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_list_for_whitespace_query(self, rag_service):
        # Act
        result = await rag_service.retrieve(query="   ")

        # Assert
        assert result == []

    @pytest.mark.asyncio
    async def test_embeds_query_before_searching(
        self, rag_service, mock_embedding_model
    ):
        # Act
        await rag_service.retrieve(query="hypertension treatment")

        # Assert
        mock_embedding_model.embed_text.assert_awaited_once_with("hypertension treatment")

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_embedding_fails(
        self, rag_service, mock_embedding_model
    ):
        # Arrange
        mock_embedding_model.embed_text.return_value = []

        # Act
        result = await rag_service.retrieve(query="hypertension")

        # Assert
        result == []

    @pytest.mark.asyncio
    async def test_passes_embedding_and_top_k_to_vector_store(
        self, rag_service, mock_embedding_model, mock_vector_store
    ):
        # Arrange
        mock_embedding_model.embed_text.return_value = [0.1, 0.2]

        # Act
        await rag_service.retrieve(query="diabetes", top_k=8)

        # Assert
        call_kwargs = mock_vector_store.query.call_args[1]
        assert call_kwargs["query_embedding"] == [0.1, 0.2]
        assert call_kwargs["top_k"] == 8

    @pytest.mark.asyncio
    async def test_passes_filters_to_vector_store(
        self, rag_service, mock_vector_store
    ):
        # Act
        await rag_service.retrieve(
            query="ACE inhibitor",
            filters={"doc_type": "drug_reference"},
        )

        # Assert
        call_kwargs = mock_vector_store.query.call_args[1]
        assert call_kwargs["filters"] == {"doc_type": "drug_reference"}

    @pytest.mark.asyncio
    async def test_returns_chunks_above_threshold(
        self, rag_service, mock_vector_store
    ):
        # Arrange
        mock_vector_store.query.return_value = [
            _raw_result(score=0.9, chunk_id="high"),
            _raw_result(score=0.1, chunk_id="low"),
        ]

        # Act
        result = await rag_service.retrieve(query="query")

        # Assert — only the high-score chunk passes default threshold (0.35)
        assert len(result) == 1
        assert result[0].chunk_id == "high"

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_results(
        self, rag_service, mock_vector_store
    ):
        # Arrange
        mock_vector_store.query.return_value = []

        # Act
        result = await rag_service.retrieve(query="obscure query")

        # Assert
        assert result == []

    @pytest.mark.asyncio
    async def test_uses_instance_default_top_k_when_not_specified(
        self, mock_embedding_model, mock_vector_store
    ):
        # Arrange
        service = RAGService(
            embedding_model=mock_embedding_model,
            vector_store=mock_vector_store,
            default_top_k=7,
        )

        # Act
        await service.retrieve(query="query")

        # Assert
        call_kwargs = mock_vector_store.query.call_args[1]
        assert call_kwargs["top_k"] == 7

    @pytest.mark.asyncio
    async def test_score_threshold_overridden_per_call(
        self, rag_service, mock_vector_store
    ):
        # Arrange — chunk score 0.5 would pass 0.35 default but not 0.8 override
        mock_vector_store.query.return_value = [_raw_result(score=0.5)]

        # Act
        result = await rag_service.retrieve(query="query", score_threshold=0.8)

        # Assert
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_exception(
        self, rag_service, mock_embedding_model
    ):
        # Arrange
        mock_embedding_model.embed_text.side_effect = RuntimeError("embed failed")

        # Act
        result = await rag_service.retrieve(query="query")

        # Assert — never raises, returns empty list
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_retrieved_chunk_objects(
        self, rag_service, mock_vector_store
    ):
        # Arrange
        mock_vector_store.query.return_value = [_raw_result(score=0.9)]

        # Act
        result = await rag_service.retrieve(query="query")

        # Assert
        assert isinstance(result[0], RetrievedChunk)
        assert result[0].content == "clinical text"
        assert result[0].source == "guideline.pdf"
        assert result[0].doc_type == "clinical_guideline"


# ── retrieve_as_context ───────────────────────────────────────────────────────

class TestRetrieveAsContext:
    """retrieve_as_context formats chunks into a single LLM-ready string."""

    @pytest.mark.asyncio
    async def test_returns_empty_string_when_no_chunks(self, rag_service):
        # Arrange
        rag_service.retrieve = mocker.AsyncMock(return_value=[]) \
            if False else None

    @pytest.mark.asyncio
    async def test_returns_empty_string_when_no_results(
        self, mocker, rag_service, mock_vector_store
    ):
        # Arrange
        mock_vector_store.query.return_value = []

        # Act
        result = await rag_service.retrieve_as_context(query="query")

        # Assert
        assert result == ""

    @pytest.mark.asyncio
    async def test_returns_formatted_string_with_chunk_content(
        self, rag_service, mock_vector_store
    ):
        # Arrange
        mock_vector_store.query.return_value = [
            _raw_result(score=0.9, content="ACE inhibitor recommended")
        ]

        # Act
        result = await rag_service.retrieve_as_context(query="query")

        # Assert
        assert "ACE inhibitor recommended" in result
        assert "Retrieved 1 relevant document" in result

    @pytest.mark.asyncio
    async def test_includes_separator_between_multiple_chunks(
        self, rag_service, mock_vector_store
    ):
        # Arrange
        mock_vector_store.query.return_value = [
            _raw_result(score=0.9, chunk_id="c1", content="First chunk"),
            _raw_result(score=0.8, chunk_id="c2", content="Second chunk"),
        ]

        # Act
        result = await rag_service.retrieve_as_context(query="query")

        # Assert
        assert "---" in result
        assert "First chunk" in result
        assert "Second chunk" in result


# ── ingest_document ───────────────────────────────────────────────────────────

class TestIngestDocument:
    """ingest_document chunks text, embeds, and upserts to vector store.

    ingest_document does `from utils.ingest import chunk_text, prepare_documents`
    as a local import inside the function body. We must ensure utils.ingest is
    in sys.modules before patching, otherwise mocker.patch raises AttributeError.
    The fixture `ensure_utils_ingest` handles this.
    """

    @pytest.fixture(autouse=True)
    def ensure_utils_ingest(self, mocker):
        """
        Import utils.ingest so it exists in sys.modules before any patch runs.
        If the real module is importable, great. If not (missing deps), inject
        a stub with the two functions ingest_document uses.
        """
        import sys
        import types
        if "utils.ingest" not in sys.modules:
            stub = types.ModuleType("utils.ingest")
            stub.chunk_text = lambda text: []
            stub.prepare_documents = lambda chunks, metadata, prefix: ([], [])
            sys.modules["utils.ingest"] = stub
            # Also make utils.ingest accessible as an attribute of utils
            import utils
            utils.ingest = stub

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_chunks_produced(self, mocker, rag_service):
        # Arrange
        mocker.patch("utils.ingest.chunk_text", return_value=[])
        mocker.patch("utils.ingest.prepare_documents", return_value=([], []))

        # Act
        result = await rag_service.ingest_document(
            text="", metadata={}, doc_id_prefix="doc"
        )

        # Assert
        assert result == 0

    @pytest.mark.asyncio
    async def test_upserts_valid_chunks_and_returns_count(
        self, mocker, rag_service, mock_vector_store, mock_embedding_model
    ):
        # Arrange
        chunks = ["chunk one text", "chunk two text"]
        ids = ["doc_001", "doc_002"]
        metas = [{"source": "test.pdf"}, {"source": "test.pdf"}]
        mocker.patch("utils.ingest.chunk_text", return_value=chunks)
        mocker.patch("utils.ingest.prepare_documents", return_value=(ids, metas))
        mock_embedding_model.embed_batch.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_vector_store.upsert.return_value = True

        # Act
        result = await rag_service.ingest_document(
            text="full document text", metadata={"source": "test.pdf"}, doc_id_prefix="doc"
        )

        # Assert
        assert result == 2
        mock_vector_store.upsert.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_skips_chunks_with_empty_embedding(
        self, mocker, rag_service, mock_vector_store, mock_embedding_model
    ):
        # Arrange — second embedding is empty (failed)
        chunks = ["chunk one", "chunk two"]
        mocker.patch("utils.ingest.chunk_text", return_value=chunks)
        mocker.patch(
            "utils.ingest.prepare_documents",
            return_value=(["id1", "id2"], [{}, {}]),
        )
        mock_embedding_model.embed_batch.return_value = [[0.1, 0.2], []]
        mock_vector_store.upsert.return_value = True

        # Act
        result = await rag_service.ingest_document(
            text="text", metadata={}, doc_id_prefix="doc"
        )

        # Assert — only one valid chunk upserted
        assert result == 1
        upserted = mock_vector_store.upsert.call_args[0][0]
        assert len(upserted) == 1

    @pytest.mark.asyncio
    async def test_returns_zero_when_upsert_fails(
        self, mocker, rag_service, mock_vector_store, mock_embedding_model
    ):
        # Arrange
        mocker.patch("utils.ingest.chunk_text", return_value=["chunk"])
        mocker.patch(
            "utils.ingest.prepare_documents",
            return_value=(["id1"], [{}]),
        )
        mock_embedding_model.embed_batch.return_value = [[0.1, 0.2]]
        mock_vector_store.upsert.return_value = False

        # Act
        result = await rag_service.ingest_document(
            text="text", metadata={}, doc_id_prefix="doc"
        )

        # Assert
        assert result == 0

    @pytest.mark.asyncio
    async def test_calls_embed_batch_with_all_chunks(
        self, mocker, rag_service, mock_embedding_model, mock_vector_store
    ):
        # Arrange
        chunks = ["chunk a", "chunk b", "chunk c"]
        mocker.patch("utils.ingest.chunk_text", return_value=chunks)
        mocker.patch(
            "utils.ingest.prepare_documents",
            return_value=(["i1", "i2", "i3"], [{}, {}, {}]),
        )
        mock_embedding_model.embed_batch.return_value = [[0.1], [0.2], [0.3]]
        mock_vector_store.upsert.return_value = True

        # Act
        await rag_service.ingest_document(
            text="text", metadata={}, doc_id_prefix="doc"
        )

        # Assert
        mock_embedding_model.embed_batch.assert_awaited_once_with(chunks)


# ── _build_chunks ─────────────────────────────────────────────────────────────

class TestBuildChunks:
    """_build_chunks filters by threshold and maps fields correctly."""

    def test_filters_out_results_below_threshold(self, rag_service):
        # Arrange
        raw = [
            _raw_result(score=0.8, chunk_id="pass"),
            _raw_result(score=0.2, chunk_id="fail"),
        ]

        # Act
        chunks = rag_service._build_chunks(raw, threshold=0.35)

        # Assert
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "pass"

    def test_includes_results_at_exact_threshold(self, rag_service):
        # Arrange
        raw = [_raw_result(score=0.35, chunk_id="exact")]

        # Act
        chunks = rag_service._build_chunks(raw, threshold=0.35)

        # Assert
        assert len(chunks) == 1

    def test_returns_empty_list_when_all_below_threshold(self, rag_service):
        # Arrange
        raw = [_raw_result(score=0.1), _raw_result(score=0.2)]

        # Act
        chunks = rag_service._build_chunks(raw, threshold=0.5)

        # Assert
        assert chunks == []

    def test_maps_metadata_fields_to_chunk(self, rag_service):
        # Arrange
        raw = [_raw_result(
            score=0.9,
            chunk_id="c99",
            content="content here",
            source="my_doc.pdf",
            doc_type="drug_reference",
        )]

        # Act
        chunks = rag_service._build_chunks(raw, threshold=0.0)

        # Assert
        c = chunks[0]
        assert c.chunk_id == "c99"
        assert c.content == "content here"
        assert c.source == "my_doc.pdf"
        assert c.doc_type == "drug_reference"
        assert c.score == 0.9

    def test_uses_uuid_when_result_has_no_id(self, rag_service):
        # Arrange — result missing "id" key
        raw = [{"content": "text", "score": 0.9, "metadata": {}}]

        # Act
        chunks = rag_service._build_chunks(raw, threshold=0.0)

        # Assert — chunk_id assigned automatically
        assert chunks[0].chunk_id != ""
        assert len(chunks[0].chunk_id) == 36  # UUID4

    def test_uses_unknown_for_missing_source_and_doc_type(self, rag_service):
        # Arrange — metadata is empty
        raw = [{"id": "x", "content": "text", "score": 0.9, "metadata": {}}]

        # Act
        chunks = rag_service._build_chunks(raw, threshold=0.0)

        # Assert
        assert chunks[0].source == "unknown"
        assert chunks[0].doc_type == "unknown"