"""
AWS Bedrock embedding model using Amazon Nova Multimodal Embeddings v2.

Prerequisites (from AWS docs):
    - The Nova Multimodal Embeddings model must be ENABLED on your AWS account.
    - Go to: AWS Console → Bedrock → Model access (us-east-1) → Enable it.

Request format:
    {
        "taskType": "SINGLE_EMBEDDING",
        "singleEmbeddingParams": {
            "embeddingPurpose": "GENERIC_INDEX",
            "embeddingDimension": 1024,
            "text": {"truncationMode": "END", "value": "<text>"}
        }
    }

Response format (confirmed from AWS docs):
    {
        "embeddings": [
            {
                "embeddingType": "TEXT",
                "embedding": [0.031, 0.032, ...]
            }
        ]
    }

Available dimensions: 3072 (default), 1024, 384, 256.
"""
import json
from typing import List

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from src.infrastructure.embedding_models.base import BaseEmbeddingModel
from utils.logger import get_logger

logger = get_logger()

NOVA_EMBED_MODEL_ID = "amazon.nova-2-multimodal-embeddings-v1:0"
DEFAULT_EMBEDDING_DIM = 1024  # Supported: 3072, 1024, 384, 256


class BedrockEmbeddingModel(BaseEmbeddingModel):
    """
    Amazon Nova Multimodal Embeddings v2 via AWS Bedrock InvokeModel.

    IMPORTANT: This model must be explicitly enabled in your AWS account before use.
    AWS Console → Amazon Bedrock → Model access (us-east-1) → Enable Nova Multimodal Embeddings.

    Args:
        aws_access_key: AWS access key ID
        aws_secret_key: AWS secret access key
        region_name: AWS region — Nova Embed v2 is only available in us-east-1
        model_id: Bedrock model ID
        embedding_dimension: Output vector size — 3072 | 1024 | 384 | 256
        embedding_purpose: "GENERIC_INDEX" for RAG/search (default),
                           "CLASSIFICATION" or "CLUSTERING" for those tasks
    """

    def __init__(
        self,
        aws_access_key: str,
        aws_secret_key: str,
        region_name: str = "us-east-1",
        model_id: str = NOVA_EMBED_MODEL_ID,
        embedding_dimension: int = DEFAULT_EMBEDDING_DIM,
        embedding_purpose: str = "GENERIC_INDEX",
        **kwargs,
    ):
        super().__init__(model_id=model_id, **kwargs)

        self.region_name = region_name
        self._embedding_dimension = embedding_dimension
        self.embedding_purpose = embedding_purpose

        self._client = boto3.client(
            service_name="bedrock-runtime",
            region_name=region_name,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            config=Config(
                signature_version="v4",
                retries={"max_attempts": 5, "mode": "adaptive"},
            ),
        )

        logger.info(
            f"BedrockEmbeddingModel initialised | model={model_id} | "
            f"region={region_name} | dim={embedding_dimension} | "
            f"purpose={embedding_purpose}"
        )

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dimension

    async def embed_text(self, text: str) -> List[float]:
        """
        Embed a single string via Nova Multimodal Embeddings v2.

        Returns:
            Dense vector of length `embedding_dimension`, or [] on failure.
        """
        if not text or not text.strip():
            logger.warning("embed_text called with empty string — returning []")
            return []

        request_body = {
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": self.embedding_purpose,
                "embeddingDimension": self._embedding_dimension,
                "text": {
                    "truncationMode": "END",
                    "value": text,
                },
            },
        }

        try:
            response = self._client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json",
            )

            response_body = json.loads(response["body"].read())

            # Response: {"embeddings": [{"embeddingType": "TEXT", "embedding": [...]}]}
            embeddings_list = response_body.get("embeddings", [])
            if not embeddings_list:
                logger.error(
                    f"Nova Embed returned no embeddings. "
                    f"Response keys: {list(response_body.keys())}"
                )
                return []

            embedding = embeddings_list[0].get("embedding", [])
            if not embedding:
                logger.error("Nova Embed embedding field was empty")
                return []

            logger.debug(f"Embedded ({len(embedding)}-dim): '{text[:60]}'")
            return embedding

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "ValidationException" and "model identifier" in str(e).lower():
                logger.error(
                    "ValidationException: model identifier invalid. "
                    "The Nova Multimodal Embeddings model is likely not enabled on your account. "
                    "Enable it at: AWS Console → Bedrock → Model access (us-east-1)."
                )
            else:
                logger.error(
                    f"Bedrock ClientError [{error_code}] during embedding: {e}",
                    exc_info=True,
                )
            return []

        except Exception as e:
            logger.error(f"Unexpected embedding error: {e}", exc_info=True)
            return []

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts sequentially.

        Nova Embed InvokeModel doesn't support true batching — each text needs
        its own request. For large offline jobs consider Bedrock's async batch
        inference API (start_async_invoke) instead.

        Returns:
            List of vectors in input order. Failed ones are empty lists.
        """
        embeddings: List[List[float]] = []

        for i, text in enumerate(texts):
            embeddings.append(await self.embed_text(text))
            if (i + 1) % 10 == 0:
                logger.info(f"Embedded {i + 1}/{len(texts)} chunks...")

        logger.info(f"Batch complete: {len(embeddings)} vectors generated")
        return embeddings