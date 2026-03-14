import os
import io
import json
from typing import Dict, Any, List
from pdf2image import convert_from_bytes
from PIL import Image

from src.infrastructure.language_models.bedrock import BedrockModel

from utils.logger import get_logger
from utils.helpers import (
    extract_dict_from_string,
    fix_malformed_json,
    take_annotation_from,
    fix_stringified_lists,
)

logger = get_logger()

class BedrockModelService:
    """ Bedrock model service """
    def __init__(
        self, 
        bedrock_model: BedrockModel
    ) -> None:
        self.bedrock_model = bedrock_model

    @take_annotation_from(BedrockModel.prompt)
    async def async_call_prompt(self, **kwargs):
        """A helper method to asynchronously prompt"""
        async for result in self.bedrock_model.prompt(**kwargs):
            logger.info("result: ", result)
            result = result.get('output').get('message').get('content')[0]

            #clean dictionary
            result = fix_malformed_json(result)
            if isinstance(result, dict):
                if "text" in result.keys():
                    # logger.info("In text")
                    result = result.get("text")
            if isinstance(result, str):
                # logger.info("In string")
                result = extract_dict_from_string(result)

            if result:
                if "toolUse" in result.keys():
                    result = result.get("toolUse")

            if result:
                if "input" in result.keys():
                    result = result.get("input")
            if result:
                if "parameters" in result.keys():
                    result = result.get("parameters")

            if result:
                result = await fix_stringified_lists(result)

            return result

    async def async_call_prompt_claude(self, **kwargs):
        """A helper method to asynchronously call cls.prompt from sync methods using asyncio.run"""
        async for result in self.bedrock_model.prompt(**kwargs):
            # logger.info("result: ", result)
            result = fix_malformed_json(result)
            # print("first result: ", result)
            
            result = result.get("output", {}).get("message", {}).get("content")
            result = result[1] if len(result) > 1 else result[0]
            result = fix_malformed_json(result)
            # print("second result: ", result)
            if isinstance(result, dict):
                if "text" in result.keys():
                    # logger.info("In text")
                    result = result.get("text")
                    result = fix_malformed_json(result)

            # print("Third result: ", result)
            if isinstance(result, str):
                # logger.info("In string")
                str_result = extract_dict_from_string(result)
                result = str_result if isinstance(str_result, dict) else result

            if isinstance(result, str):
                # logger.info("In string Fix Malformed")
                str_result = fix_malformed_json(result)
                result = str_result if isinstance(str_result, dict) else result

            if result and isinstance(result, dict):
                if "toolUse" in result.keys():
                    result = result.get("toolUse")

            if result and isinstance(result, dict):
                if "input" in result.keys():
                    result = result.get("input")
            if result and isinstance(result, dict):
                if "parameters" in result.keys():
                    result = result.get("parameters")

            if result and isinstance(result, dict):
                if "content" in result.keys():
                    result = result.get("content")


            if isinstance(result, str):
                # logger.info("In string Fix Malformed")
                str_result = fix_malformed_json(result)
                result = str_result if isinstance(str_result, dict) else result

            if result and isinstance(result, dict):
                result = await fix_stringified_lists(result)

            # logger.info(f"\n Async call result: {type(result)} \n")
            # logger.info(f"\n Async call result: {result} \n")
            return result

    async def prompt_llm_for_image(self, 
        prompt: str, 
        image: bytes, 
        text: str, 
        response_schema: dict,
        temperature: float = 0.90
    ) -> Dict[str, Any]:
        """
        Prompt LLM for text.

        Args:
            prompt: - prompt for model
            image: - bytes representation of image
            text: - text context for model when available
            response_schema: - response_schema for model
            temperature: - temperature

        Returns:
            dict: A dictionary containing the result of the LLM prompt.
        """
        try:

            # logger.info("In Bedrock model service image execution")

            # prepare message
            messages = [
                        {
                            "role": "user",
                            "content": [
                                {"image": {"format": "png", "source": {"bytes": image}}},
                                {"text": prompt},
                            ],
                        }
                    ]

            # get llm response
            response = await self.async_call_prompt(
                # text=text,
                grammar=response_schema,
                is_function_call=True,
                message_history=messages,
                temperature=temperature,
            )
            # logger.info(f"\n Response in prompt for image: {response} \n")
            # logger.info(f"\n Response Type in prompt for image: {type(response)} \n")
            
            return response
        
        except Exception as e:
            logger.error(f'Unable to process request and prompt LLM for image: {e}')
            raise e

    async def prompt_llm_for_text(
        self, 
        prompt: str, 
        text: str, 
        response_schema: dict, 
        model_id: str = None, 
        model_type: str = "llama", 
        temperature: float = None,
        reasoning: bool = False,
    ) -> Dict[str, Any]:
        """
        Prompt LLM for text.

        Args:
            prompt: - prompt for model
            text: - text context for model when available
            model_id: - model id to use
            model_type: - llama or claude
            response_schema: - response_schema for model

        Returns:
            dict: A dictionary containing the result of the LLM prompt.
        """
        try:
            # logger.info(f"In Bedrock model service text execution, Prompt: {prompt} \n text: {text} \n Schema: {response_schema}")

            if model_type == "llama":

                # get llm response
                response = await self.async_call_prompt(
                    system_prompt=prompt,
                    # prompt=prompt,
                    text=text,
                    grammar=response_schema,
                    is_function_call=False,
                    model_id=model_id,
                    temperature=temperature
                )
            else:
                response = await self.async_call_prompt_claude(
                    system_prompt=prompt,
                    # prompt=prompt,
                    text=text,
                    grammar=response_schema,
                    is_function_call=True,
                    temperature=temperature,
                    reasoning=reasoning,
                )
            logger.info(f"\n Response in prompt for text: {response} \n")
            # logger.info(f"\n Response Type in prompt for text: {type(response)} \n")
            
            return response
        
        except Exception as e:
            logger.error(f'Unable to process request and prompt LLM: {e}')
            raise e

    async def prompt_llm_for_document(
        self, 
        prompt: str, 
        document: bytes, 
        text: str, 
        response_schema: dict,
        temperature: float = 0.90
    ) -> Dict[str, Any]:
        """
        Prompt LLM for text.

        Args:
            prompt: - prompt for model
            document: - bytes representation of document
            text: - text context for model when available
            response_schema: - response_schema for model
            temperature: - temperature

        Returns:
            dict: A dictionary containing the result of the LLM prompt.
        """
        try:

            # logger.info("In Bedrock model service image execution")

            # prepare message
            messages = await self.prepare_document_for_bedrock(
                document_bytes=document,
                prompt=prompt,
                image_format="png",  # or "jpeg" for smaller size
                dpi=150  # adjust for quality vs size tradeoff
            )

            # get llm response
            response = await self.async_call_prompt_claude(
                # text=text,
                grammar=response_schema,
                is_function_call=True,
                message_history=messages,
                temperature=temperature,
                reasoning=True
            )
            # logger.info(f"\n Response in prompt for image: {response} \n")
            # logger.info(f"\n Response Type in prompt for image: {type(response)} \n")
            
            return response
        
        except Exception as e:
            logger.error(f'Unable to process request and prompt LLM for image: {e}')
            raise e

    async def prepare_document_for_bedrock(
        self,
        document_bytes: bytes,
        prompt: str,
        image_format: str = "png",
        dpi: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Convert document bytes to LLM message format with all pages as images.
        
        Args:
            document_bytes: Raw bytes of the PDF or image document
            prompt: Your analysis prompt text
            image_format: Output image format ('png', 'jpeg', 'webp')
            dpi: DPI for PDF conversion (higher = better quality, larger size)
        
        Returns:
            List of message dictionaries formatted for LLM
        """
        content = []
        
        try:
            # Try to convert as PDF first
            images = convert_from_bytes(document_bytes, dpi=dpi)
            
            # Convert each page to bytes
            for page_num, image in enumerate(images, 1):
                img_byte_arr = io.BytesIO()
                
                # Convert to RGB if necessary (for JPEG)
                if image_format.lower() == 'jpeg' and image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Save image to bytes
                image.save(img_byte_arr, format=image_format.upper())
                img_bytes = img_byte_arr.getvalue()
                
                # Add to content array
                content.append({
                    "image": {
                        "format": image_format.lower(),
                        "source": {"bytes": img_bytes}
                    }
                })
                
        except Exception as e:
            # If PDF conversion fails, assume it's already an image
            try:
                image = Image.open(io.BytesIO(document_bytes))
                
                img_byte_arr = io.BytesIO()
                
                # Convert to RGB if necessary (for JPEG)
                if image_format.lower() == 'jpeg' and image.mode != 'RGB':
                    image = image.convert('RGB')
                
                image.save(img_byte_arr, format=image_format.upper())
                img_bytes = img_byte_arr.getvalue()
                
                content.append({
                    "image": {
                        "format": image_format.lower(),
                        "source": {"bytes": img_bytes}
                    }
                })
                
            except Exception as img_error:
                raise ValueError(f"Could not process document as PDF or image: {img_error}")
        
        # Add the text prompt at the end
        content.append({"text": prompt})
        
        # Return formatted messages
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        return messages
