from dependency_injector import containers, providers
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from contextlib import asynccontextmanager

from src.config.base import get_settings

# Middleware imports
# from src.infrastructure.middleware.auth import JWTAuthMiddleware
# from src.infrastructure.middleware.socket import WebSocketAuthMiddleware

# Repository imports
# from src.infrastructure.repository.auth import AuthRepository

# Embedding model
from src.infrastructure.embedding_models.bedrock import BedrockEmbeddingModel

# Service Imports
from src.infrastructure.services.transcription import TranscriptionService
from src.infrastructure.services.patient import PatientService
from src.infrastructure.services.soap import SOAPService
from src.infrastructure.services.evaluation import EvaluationService
from src.infrastructure.services.rag import RAGService
from src.infrastructure.language_model_service.bedrock import BedrockModelService


# Agent Imports
from src.core.agents.scribe import ScribeAgent
from src.core.agents.evaluation import EvaluationAgent

# Language Model Imports
from src.infrastructure.language_models.bedrock import BedrockModel
from src.infrastructure.language_models.sonic import SonicModel

# Guided Json imports (if needed)

# Tools imports
from src.core.tools.base import ToolRegistry
from src.core.tools.evaluation import EvaluationTools
from src.core.tools.patient import PatientTools
from src.core.tools.soap import SOAPTools
from src.core.tools.scribe_evaluation import ScribeEvaluationTools
from src.core.tools.retriever import RetrieverTools

# Prompts imports
from src.core.prompts.scribe import ScribePrompt
from src.infrastructure.prompts.patient import extract_ehr_fields_prompt
from src.core.prompts.evaluation import EvaluationPrompt

# Redis imports
from src.infrastructure.cache.redis.client import RedisClient
from src.infrastructure.cache.redis.manager import RedisCacheManager
from src.infrastructure.cache.service import CacheService

# Vector Store imports
from src.infrastructure.vector_store.chroma import ChromaVectorStore

# Consumers
from src.api.scribe.consumer import ScribeConsumer


# Model Schema Imports
from src.infrastructure.model_schemas.patient import ExtractEHRFieldsGuidedJson


class Container(containers.DeclarativeContainer):
    """
    DI container for creating factories and singletons used by services.
    """
    config = providers.Singleton(get_settings)

    # --- Database ---
    # db_engine = providers.Singleton(
    #     create_async_engine,
    #     providers.Callable(
    #         lambda cfg: cfg.SQLALCHEMY_DATABASE_URI.replace("postgresql://", "postgresql+asyncpg://"),
    #         config
    #     ),
    #     pool_size=20,
    #     max_overflow=10,
    #     pool_timeout=30,
    #     # pool_recycle=3600,
    #     pool_recycle=300, # reduce recycle time to 5 minutes
    #     pool_pre_ping=True,

    #     # Additional settings for stale connections
    #     connect_args={
    #         "timeout": 60,  # Connection timeout
    #         "command_timeout": 60,  # Command execution timeout
    #         "server_settings": {
    #             "application_name": "elle_fastapi",
    #             "jit": "off"
    #         }
    #     },
        
    #     # Reset connections on return to pool
    #     pool_reset_on_return='rollback',

    #     # Enable echo for debugging (disable in production)
    #     # echo=True,
    # )

    # session_factory = providers.Singleton(
    #     async_sessionmaker,
    #     bind=db_engine,
    #     expire_on_commit=False,
    #     class_=AsyncSession,

    #     autoflush=False,  # Don't auto-flush
    #     autocommit=False,  # Don't auto-commit
    # )

    # AsyncSession per request
    # Database Session Resource - Creates AsyncSession instances with automatic lifecycle management.
    # The Resource provider automatically closes sessions and returns connections to the pool, preventing connection leaks.
    # db_session = providers.Resource(
    #     lambda sf: sf(),
    #     session_factory
    # )

    # --- Cache ---
    redis_client = providers.Singleton(
        RedisClient,
        url=providers.Callable(lambda c: c.REDIS_LOCATION, config),
        redis_name=providers.Callable(lambda c: c.REDIS_NAME, config),
        redis_password=providers.Callable(lambda c: c.REDIS_PASSWORD, config),
        redis_host=providers.Callable(lambda c: c.REDIS_HOST, config),
        redis_port=providers.Callable(lambda c: c.REDIS_PORT, config),
        redis_db=providers.Callable(lambda c: c.REDIS_DB, config),
    )

    cache_manager = providers.Singleton(
        RedisCacheManager,
        client=redis_client,
    )

    cache_service = providers.Singleton(
        CacheService,
        manager=cache_manager,
    )
    
    # --- LLM Models ---
    # --- Primary LLM Model (for agent conversations) ---
    llm_model = providers.Factory(
        BedrockModel,
        aws_access_key=providers.Callable(lambda c: c.AWS_ACCESS_KEY, config),
        aws_secret_key=providers.Callable(lambda c: c.AWS_SECRET_KEY, config),
    )

    # --- Analysis LLM Model (for other stuffs like document extraction/analysis) ---
    llm_model_analysis = providers.Factory(
        BedrockModel,
        aws_access_key=providers.Callable(lambda c: c.AWS_ACCESS_KEY, config),
        aws_secret_key=providers.Callable(lambda c: c.AWS_SECRET_KEY, config),
        max_tokens=4096,
    )


    sonic_model = providers.Factory(
        SonicModel,
        aws_access_key=providers.Callable(lambda c: c.AWS_ACCESS_KEY, config),
        aws_secret_key=providers.Callable(lambda c: c.AWS_SECRET_KEY, config),
    )

    # ── Embedding Model ────────────────────────────────────────────────────
    # Singleton — one Bedrock client shared across all RAG calls.
    embedding_model = providers.Singleton(
        BedrockEmbeddingModel,
        aws_access_key=providers.Callable(lambda c: c.AWS_ACCESS_KEY, config),
        aws_secret_key=providers.Callable(lambda c: c.AWS_SECRET_KEY, config),
        # region_name=providers.Callable(lambda c: c.AWS_REGION_NAME, config),
    )

    # ── Vector Store ───────────────────────────────────────────────────────
    vector_store = providers.Singleton(
        ChromaVectorStore,
        collection_name=providers.Callable(lambda c: c.CHROMA_COLLECTION, config),
        persist_directory=providers.Callable(lambda c: c.CHROMA_PERSIST_DIR, config),
    )

    # Services
    # ── Services ──────────────────────────────────────────────────────────────
    # All implementation lives here. Tools call services. Agents call services
    # directly only for post-loop data fetching.
    bedrock_service = providers.Factory(
        BedrockModelService,
        bedrock_model=llm_model_analysis
    )

    # ── RAG Service ────────────────────────────────────────────────────────
    # Depends on the abstract embedding model and vector store — not on
    # their concrete implementations.
    rag_service = providers.Singleton(
        RAGService,
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    transcription_service = providers.Factory(
        TranscriptionService,
        sonic_model=sonic_model,
        cache_service=cache_service,
    )

    patient_service = providers.Singleton(
        PatientService,
        cache_service=cache_service,
        llm_service=bedrock_service,
        ehr_fields_extraction_prompt=extract_ehr_fields_prompt,
        ehr_fields_guided_json=ExtractEHRFieldsGuidedJson,
    )

    soap_service = providers.Singleton(
        SOAPService,
        llm_model=llm_model_analysis,
        cache_service=cache_service,
    )

    evaluation_service = providers.Singleton(
        EvaluationService,
        llm_model=llm_model_analysis,
        cache_service=cache_service,
    )

    # ------ PROMPTS ---------
    scribe_prompt_template = providers.Factory(
        ScribePrompt,
    )

    evaluation_prompt_template = providers.Factory(
        ScribePrompt,
    )

    evaluation_tools_for_eval_agent = providers.Factory(
        EvaluationTools,
        evaluation_service=evaluation_service,
        cache_service=cache_service,
        enabled_tools=[
            "check_hallucinations", 
            "check_drug_interactions", 
            "check_guideline_alignment", 
            "aggregate_scores"
            ],
    )
    evaluation_tool_registry = providers.Factory(
        ToolRegistry,
        tool_classes=providers.List(
            evaluation_tools_for_eval_agent,
            ),
    )
    evaluation_agent = providers.Factory(
        EvaluationAgent,
        llm_model=llm_model,
        tool_registry=evaluation_tool_registry,
        prompt_template=evaluation_prompt_template,
        cache_service=cache_service,
    )


    patient_tools_for_scribe_agent = providers.Factory(
        PatientTools,
        patient_service=patient_service,
        enabled_tools=[
            "get_patient_history",
            "insert_ehr_note",
            "flag_missing_ehr_fields"
        ]  # Only enable specific tools
    )

    soap_tools_for_scribe_agent = providers.Factory(
        SOAPTools,
        soap_service=soap_service,
        enabled_tools=[
            "generate_soap_note",
            # "save_session_transcript",
            "get_session_transcript"
        ]  # Only enable specific tools
    )

    evaluation_tools_for_scribe_agent = providers.Factory(
        ScribeEvaluationTools,
        evaluation_agent=evaluation_agent,
        cache_service=cache_service,
        enabled_tools=[
            "evaluate_consultation",
        ],
    )

    retriver_tools_for_scribe_agent = providers.Factory(
        RetrieverTools,
        rag_service=rag_service,
        enabled_tools=[
            "retrieve_clinical_documents_by_document_type",
            "retrieve_clinical_documents_context",
        ],
    )

    # Tool Registry for Scribe Agent
    scribe_agent_tool_registry = providers.Factory(
        ToolRegistry,
        tool_classes=providers.List(
            patient_tools_for_scribe_agent,
            soap_tools_for_scribe_agent,
            evaluation_tools_for_scribe_agent,
            retriver_tools_for_scribe_agent,
        )
    )

    # Scribe Agent
    scribe_agent = providers.Factory(
        ScribeAgent,
        llm_model=llm_model,
        tool_registry=scribe_agent_tool_registry,  
        prompt_template=scribe_prompt_template,
        transcription_service=transcription_service,
        soap_service=soap_service,
        evaluation_service=evaluation_service,
        cache_service=cache_service,
    )

    # ── Consumers (Factory — one per WebSocket connection) ────────────────────
    # websocket and user are injected per-connection at the route level

    # scribe_consumer = providers.Factory(
    #     ScribeConsumer,
    #     agent=scribe_agent,
    #     # websocket and user passed at call time:
    #     # container.scribe_consumer(websocket=ws, user=current_user)
    # )