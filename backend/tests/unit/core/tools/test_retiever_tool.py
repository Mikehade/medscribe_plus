"""
Unit tests for RetrieverTools

"""
import json
import pytest


from src.core.tools.retriever import RetrieverTools


@pytest.fixture
def mock_rag_service(mocker):
    service = mocker.AsyncMock()
    service.retrieve = mocker.AsyncMock()
    return service


@pytest.fixture
def retriever_tools(mock_rag_service):
    return RetrieverTools(
        rag_service=mock_rag_service,
        enabled_tools=[
            "retrieve_clinical_documents_context",
            "retrieve_clinical_documents_by_document_type",
        ],
    )


def _make_chunks(mocker, n=2):
    chunks = []
    for i in range(n):
        c = mocker.Mock()
        c.to_dict.return_value = {"id": i, "text": f"chunk {i}"}
        chunks.append(c)
    return chunks


class TestRetrieverToolsInitialization:
    """RetrieverTools stores dependencies correctly."""

    def test_stores_rag_service(self, mock_rag_service):
        # Act
        tools = RetrieverTools(rag_service=mock_rag_service)

        # Assert
        assert tools.rag_service is mock_rag_service

    def test_stores_extra_kwargs(self, mock_rag_service):
        # Act
        tools = RetrieverTools(rag_service=mock_rag_service, user_id="u1")

        # Assert
        assert tools.kwargs["user_id"] == "u1"


class TestRetrieverToolsExecute:
    """execute routes and normalises results."""

    @pytest.mark.asyncio
    async def test_executes_retrieve_clinical_documents_context(
        self, retriever_tools, mock_rag_service, mocker
    ):
        # Arrange
        mock_rag_service.retrieve.return_value = _make_chunks(mocker)

        # Act
        result = await retriever_tools.execute(
            "retrieve_clinical_documents_context", {"query": "hypertension treatment"}
        )

        # Assert
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_returns_error_for_unknown_tool(self, retriever_tools):
        # Act
        result = await retriever_tools.execute("nonexistent", {})

        # Assert
        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_catches_unexpected_exception(self, retriever_tools, mock_rag_service):
        # Arrange
        mock_rag_service.retrieve.side_effect = RuntimeError("vector store down")

        # Act
        result = await retriever_tools.execute(
            "retrieve_clinical_documents_context", {"query": "query"}
        )

        # Assert
        assert result["success"] is False
        assert "vector store down" in result["error"]


class TestRetrieveClinicalDocumentsContext:
    """_retrieve_clinical_documents_context searches and formats results."""

    @pytest.mark.asyncio
    async def test_returns_chunks_on_success(self, retriever_tools, mock_rag_service, mocker):
        # Arrange
        chunks = _make_chunks(mocker, n=3)
        mock_rag_service.retrieve.return_value = chunks

        # Act
        result = await retriever_tools._retrieve_clinical_documents_context(
            query="first-line treatment for hypertension"
        )

        # Assert
        assert result["success"] is True
        assert result["results_found"] == 3
        assert len(result["context"]) == 3

    @pytest.mark.asyncio
    async def test_passes_query_and_top_k_to_service(self, retriever_tools, mock_rag_service, mocker):
        # Arrange
        mock_rag_service.retrieve.return_value = _make_chunks(mocker)

        # Act
        await retriever_tools._retrieve_clinical_documents_context(
            query="diabetes management", top_k=8
        )

        # Assert
        call_kwargs = mock_rag_service.retrieve.call_args[1]
        assert call_kwargs["query"] == "diabetes management"
        assert call_kwargs["top_k"] == 8

    @pytest.mark.asyncio
    async def test_passes_specialty_filter_when_provided(
        self, retriever_tools, mock_rag_service, mocker
    ):
        # Arrange
        mock_rag_service.retrieve.return_value = _make_chunks(mocker)

        # Act
        await retriever_tools._retrieve_clinical_documents_context(
            query="cardiac drugs", specialty="cardiology"
        )

        # Assert
        call_kwargs = mock_rag_service.retrieve.call_args[1]
        assert call_kwargs["filters"] == {"specialty": "cardiology"}

    @pytest.mark.asyncio
    async def test_passes_no_filter_when_specialty_empty(
        self, retriever_tools, mock_rag_service, mocker
    ):
        # Arrange
        mock_rag_service.retrieve.return_value = _make_chunks(mocker)

        # Act
        await retriever_tools._retrieve_clinical_documents_context(
            query="general query", specialty=""
        )

        # Assert
        call_kwargs = mock_rag_service.retrieve.call_args[1]
        assert call_kwargs["filters"] is None

    @pytest.mark.asyncio
    async def test_returns_no_results_message_when_empty(
        self, retriever_tools, mock_rag_service
    ):
        # Arrange
        mock_rag_service.retrieve.return_value = []

        # Act
        result = await retriever_tools._retrieve_clinical_documents_context(query="obscure query")

        # Assert
        assert result["success"] is True
        assert result["results_found"] == 0
        assert result["context"] == ""

    @pytest.mark.asyncio
    async def test_returns_failure_on_service_exception(self, retriever_tools, mock_rag_service):
        # Arrange
        mock_rag_service.retrieve.side_effect = ConnectionError("chroma down")

        # Act
        result = await retriever_tools._retrieve_clinical_documents_context(query="query")

        # Assert
        assert result["success"] is False
        assert "chroma down" in result["error"]


class TestRetrieveClinicalDocumentsByDocumentType:
    """_retrieve_clinical_documents_by_document_type filters by doc_type."""

    @pytest.mark.asyncio
    async def test_returns_chunks_for_valid_doc_type(
        self, retriever_tools, mock_rag_service, mocker
    ):
        # Arrange
        chunks = _make_chunks(mocker, n=2)
        mock_rag_service.retrieve.return_value = chunks

        # Act
        result = await retriever_tools._retrieve_clinical_documents_by_document_type(
            query="ACE inhibitors", doc_type="drug_reference"
        )

        # Assert
        assert result["success"] is True
        assert result["results_found"] == 2

    @pytest.mark.asyncio
    async def test_passes_doc_type_filter_to_service(
        self, retriever_tools, mock_rag_service, mocker
    ):
        # Arrange
        mock_rag_service.retrieve.return_value = _make_chunks(mocker)

        # Act
        await retriever_tools._retrieve_clinical_documents_by_document_type(
            query="BP guidelines", doc_type="clinical_guideline"
        )

        # Assert
        call_kwargs = mock_rag_service.retrieve.call_args[1]
        assert call_kwargs["filters"] == {"doc_type": "clinical_guideline"}

    @pytest.mark.asyncio
    async def test_returns_failure_when_doc_type_empty(self, retriever_tools):
        # Act
        result = await retriever_tools._retrieve_clinical_documents_by_document_type(
            query="query", doc_type=""
        )

        # Assert
        assert result["success"] is False
        assert "doc_type is required" in result["message"]

    @pytest.mark.asyncio
    async def test_returns_no_results_message_when_empty(
        self, retriever_tools, mock_rag_service
    ):
        # Arrange
        mock_rag_service.retrieve.return_value = []

        # Act
        result = await retriever_tools._retrieve_clinical_documents_by_document_type(
            query="rare condition", doc_type="clinical_guideline"
        )

        # Assert
        assert result["success"] is True
        assert result["results_found"] == 0

    @pytest.mark.asyncio
    async def test_returns_failure_on_service_exception(
        self, retriever_tools, mock_rag_service
    ):
        # Arrange
        mock_rag_service.retrieve.side_effect = RuntimeError("index error")

        # Act
        result = await retriever_tools._retrieve_clinical_documents_by_document_type(
            query="query", doc_type="drug_reference"
        )

        # Assert
        assert result["success"] is False
        assert "index error" in result["error"]