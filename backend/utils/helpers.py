import io
import re
import ast
import json
import base64
import asyncio
import aiohttp
# import fitz
# import PyPDF2
from pypdf import PdfReader
# import magic
import tempfile
from PIL import Image
from io import BytesIO
from fastapi import UploadFile
from json_repair import repair_json
from typing import Dict, Any, cast, ParamSpec, TypeVar, Union, Optional
from collections.abc import Callable
from docx2pdf import convert

from utils.logger import get_logger

logger = get_logger()

T = TypeVar("T")
P = ParamSpec("P")

def take_annotation_from(
    _origin: Callable[P, T],
) -> Callable[[Callable[..., Any]], Callable[P, T]]:
    """
    Decorator that copies the signature of one typing-related function to another.

    This decorator is useful for maintaining consistency in function signatures, especially
    when working with type hints and annotations. It allows you to create a wrapper function
    with the same signature as an original function, preserving the parameter types and return type.

    Args:
        _origin: The original function whose signature will be copied.

    Returns:
        A decorator that takes another function as input and returns a wrapped version
        of that function with the same signature as the original function.

    Example usage:

    >>> def f(foo: str, bar: int, *, baz: bool) -> None:
    ...     print(foo, bar, baz)
    ...
    >>> @take_annotation_from(f)
    ... def f_wrapper(*args: Any, **kwargs: Any) -> Any:
    ...     return f(*args, **kwargs)
    """

    def decorator(target: Callable[..., Any]) -> Callable[P, T]:
        return cast(Callable[P, T], target)

    return decorator

def fix_malformed_json(
    input_data
    ) -> dict or None:
    """ Function to fix malformed json """
    try:
        # If input_data is a string, attempt to repair it
        if isinstance(input_data, str):
            repaired_json = repair_json(input_data)
            return repaired_json
        # If input_data is already a valid Python object, return it as-is
        elif isinstance(input_data, (dict, list)):
            return input_data
        # For other types, return the original input
        else:
            return input_data

    except Exception as e:
        print(f"Error processing input in API Config: {e}")
        return None


def extract_dict_from_string(
    text: str
    ) -> Dict[str, Any]:
    """
    Extract the first JSON dictionary found in the given text.
    
    Args:
        text (str): The string that may contain a JSON dictionary.
        
    Returns:
        dict or None: The extracted dictionary if found and valid, otherwise None.
    """
    # Find the first occurrence of '{'
    start = text.find('{')
    if start == -1:
        return None

    # Track the braces to extract the complete dictionary even if nested
    brace_count = 0
    end = start
    for i in range(start, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1

        # Once all opened braces have been closed, mark the end position.
        if brace_count == 0:
            end = i + 1
            break

    # Extract the substring that should represent the JSON dictionary
    json_str = text[start:end]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # If the JSON is malformed or cannot be decoded, return None.
        return None

async def is_scanned_pdf(
    uploaded_file, 
    text_threshold=100
    ) -> bool:
    """ Method to distinguish between scanned and text-based pdf 
    
    Args:
        uploaded_file: file
        text_threshold: threshold of text to use in detection
    """
    try:
        # Convert InMemoryUploadedFile to BytesIO for PyPDF2
        file_stream = io.BytesIO(uploaded_file.read())
        # reader = PyPDF2.PdfReader(file_stream)
        reader = PdfReader(file_stream)

        total_text_length = 0
        for page in reader.pages:
            text = page.extract_text()
            if text:
                total_text_length += len(text.strip())

        return total_text_length < text_threshold  # Return True if likely scanned
    except Exception as e:
        logger.error(f"Error processing PDF to check if scanned: {e}")
        return False  # Default to scanned if there's an error


def fix_double_quoted_lists( 
    obj
    ):
    """
    Recursively traverse the object and convert lists with double quotes to actual lists.

    Args:
        obj: - object to fix
    Returns:
        dict: The fixed object.
    """
    for key, value in obj.items():
        if isinstance(value, dict):
            obj[key] = fix_double_quoted_lists(value)
        elif isinstance(value, str):
            if value.startswith('[') and value.endswith(']'):
                try:
                    obj[key] = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    pass  # If it's not a valid literal structure, leave it as is
    return obj

async def fix_stringified_lists(
    data: dict
    ) -> dict:
    """
    Recursively goes through a dictionary and converts any list formatted as a string back to a proper list.
    """
    for key, value in data.items():
        if isinstance(value, str):
            try:
                parsed_value = ast.literal_eval(value)
                if isinstance(parsed_value, list):
                    data[key] = parsed_value
            except (ValueError, SyntaxError):
                pass  # Ignore if it's not a stringified list

        elif isinstance(value, dict):
            data[key] = await fix_stringified_lists(value)  # Recursively fix nested dictionaries

    return data


async def extract_texts_from_pdf(
    document_bytes: bytes, 
    file_type: str = "pdf"
) -> str:
        """
        Determines extraction method based on file type and returns extracted text.
        """
        if file_type.lower() == "pdf":
            return await asyncio.to_thread(extract_pdf_text, document_bytes)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

def extract_pdf_text(
    document_bytes: bytes
) -> str:
    """
    Synchronous helper to extract text from PDF bytes using PyMuPDF.

    :param document_bytes: Raw PDF bytes.
    :return: Extracted text as a single string.
    """
    # Open the PDF from bytes
    doc = fitz.open(stream=document_bytes, filetype="pdf")
    # Iterate pages and gather text
    text_chunks = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(text_chunks)


async def convert_image_to_bytes(
    image_data: Union[bytes, bytearray, str, UploadFile, Image.Image]
):
    """
    convert image to bytes

    Args:
        image_data
    """
    MAX_PIXELS = (1024, 1024)

    try:
        # 1. Normalize input to raw bytes
        if isinstance(image_data, (bytes, bytearray)):
            data = image_data

        elif isinstance(image_data, UploadFile):
            data = await image_data.read()

        elif isinstance(image_data, Image.Image):
            buffer = io.BytesIO()
            image_data.save(buffer, format="PNG")
            data = buffer.getvalue()

        elif isinstance(image_data, str):
            # Could be Base64 or file path
            try:
                data = base64.b64decode(image_data)
            except Exception:
                # Treat as file path
                with open(image_data, "rb") as f:
                    data = f.read()

        else:
            # File-like object
            data = image_data.read()

        # 2. Load image in Pillow
        img = Image.open(io.BytesIO(data))

        # 3. Resize if needed
        if img.size[0] > MAX_PIXELS[0] or img.size[1] > MAX_PIXELS[1]:
            img.thumbnail(MAX_PIXELS)

        # 4. Convert non-RGB images to RGB
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")

        # 5. Save as PNG and return bytes
        output_buffer = io.BytesIO()
        img.save(output_buffer, format="PNG")
        return output_buffer.getvalue()

    except Exception as e:
        print(f"\n Error in prepare_image_for_bedrock: {e} \n")
        return None


def format_date_string(
    date_str
):
    """
    Converts a date string like "18 MAR / MARS 23" to "2023-03-18".
    """

    MONTH_MAP = {
        "JAN": "01", "JANV": "01",
        "FEB": "02", "FÉV": "02", "FEV": "02",
        "MAR": "03", "MARS": "03",
        "APR": "04", "AVR": "04",
        "MAY": "05", "MAI": "05",
        "JUN": "06", "JUIN": "06",
        "JUL": "07", "JUIL": "07",
        "AUG": "08", "AOÛT": "08", "AOUT": "08",
        "SEP": "09", "SEPT": "09",
        "OCT": "10",
        "NOV": "11",
        "DEC": "12", "DÉC": "12"
    }
    pattern = r"(\d{1,2})\s([A-ZÉÛÎÔÂ]+)\s/\s([A-ZÉÛÎÔÂ]+)\s(\d{2})"
    match = re.search(pattern, date_str, re.IGNORECASE)

    if match:
        day, month_en, month_fr, year_suffix = match.groups()
        month = MONTH_MAP.get(month_en.upper()) or MONTH_MAP.get(month_fr.upper())

        if month:
            # Convert 2-digit year to 4-digit (assuming 1900s or 2000s based on logic)
            year = f"20{year_suffix}" if int(year_suffix) <= 30 else f"19{year_suffix}"
            return f"{year}-{month}-{day.zfill(2)}"

    return date_str  # Return unchanged if no match

async def update_date_fields_in_llm_response(
    data
):
    """
    Recursively updates date fields in a dictionary.
    """
    for key, value in data.items():
        if isinstance(value, str):
            formatted_value = format_date_string(value)
            data[key] = formatted_value
        elif isinstance(value, dict):
            await update_date_fields_in_llm_response(value)  # Recursive call for nested dictionaries
            
    return data


async def bytes_to_base64(image_bytes: bytes) -> str:
    """
    Convert image bytes to a Base64-encoded string.

    Args:
        image_bytes (bytes): Raw PNG/JPEG bytes.

    Returns:
        str: Base64 encoded string (UTF-8).
    """
    if not image_bytes:
        return None

    return base64.b64encode(image_bytes).decode("utf-8")


async def download_url_to_bytes(
    url: str, 
    timeout: int = 30
) -> Optional[bytes]:
    """
    Asynchronously downloads the content from a given URL and returns it as bytes.

    Args:
        url (str): The URL of the object to download.
        timeout (int, optional): Maximum number of seconds to wait for the download. Default is 30 seconds.

    Returns:
        Optional[bytes]: The content of the URL as bytes if successful, None if download fails.

    Raises:
        ValueError: If the URL is invalid.
        aiohttp.ClientError: For network-related errors during download.
        asyncio.TimeoutError: If the download exceeds the given timeout.
    
    Example usage:
        data = await download_url_to_bytes("https://example.com/file.png")
    """
    if not url or not isinstance(url, str):
        logger.error("Invalid URL provided")
        return None

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=timeout) as response:
                response.raise_for_status()  # Raise exception for 4xx/5xx responses
                content = await response.read()  # Read content as bytes
                return content
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.info(f"Failed to download {url}: {e}")
        return None

async def docx_to_pdf(docx_bytes: bytes) -> bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_docx:
        tmp_docx.write(docx_bytes)
        tmp_docx.flush()
        pdf_path = tmp_docx.name.replace(".docx", ".pdf")
        convert(tmp_docx.name, pdf_path)
        with open(pdf_path, "rb") as f:
            return f.read()

async def process_incoming_document(
    base64_doc: str
) -> dict:
    """
    Process and validate an incoming Base64-encoded document received over a WebSocket
    connection before sending it to the LLM for analysis.

    If the document is a supported conversion type (like DOCX), it is converted to PDF
    before being returned.

    Args:
        base64_doc (str):
            The Base64-encoded document string received from the client.

    Returns:
        dict:
            A dictionary containing the document bytes (as PDF if converted) and the
            original file extension.
            Example: {"file": b'pdf_bytes', "extension": ".docx"}

    Raises:
        ValueError:
            If the detected MIME type is not one of the supported document formats.
        binascii.Error:
            If the Base64 string is invalid or cannot be decoded.
        Exception:
            Any unexpected error raised while processing the document.
    """
    # Mapping for common MIME types to short file extensions
    MIME_TO_EXTENSIONS = {
        "image/jpeg": ".jpeg",
        "image/png": ".png",
        "application/pdf": ".pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    }
    # 1. Convert Base64 to bytes
    file_bytes = base64.b64decode(base64_doc)

    # 2. Detect MIME type
    mime = magic.from_buffer(file_bytes, mime=True)

    # 3. Validate supported types
    allowed_mime = set(MIME_TO_EXTENSIONS.keys())

    if mime not in allowed_mime:
        # Returning a dictionary to match the return signature for unsupported files
        return {
            "file": None,
            "extension": "invalid"
        }

    final_bytes = file_bytes
    # 4. Use the mapping to get the correct, short file extension
    ext = MIME_TO_EXTENSIONS.get(mime, ".unknown")
    final_mime = mime

    # Handle DOCX conversion (if applicable)
    if mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        # The extension remains the original (.docx) as expected by the test, but the content is converted.
        try:
            # Convert DOCX → PDF
            # Note: We must ensure the temporary file is closed before conversion can access it.
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_docx:
                tmp_docx.write(file_bytes)
                # The file is flushed and closed automatically by the 'with' block 
                # before we try to convert it.
            
            # tmp_docx.name is the path where the DOCX file was saved
            pdf_path = tmp_docx.name.replace(ext, ".pdf")
            convert(tmp_docx.name, pdf_path)
            
            # Read the converted PDF content
            with open(pdf_path, "rb") as f:
                final_bytes = f.read()
            
            final_mime = "application/pdf"
            
        finally:
            # Clean up temporary files (important for systems like Linux)
            import os
            try:
                os.remove(tmp_docx.name) # Remove the temporary DOCX file
                os.remove(pdf_path)      # Remove the temporary PDF file
            except OSError as e:
                # logger.warning(f"Error removing temporary files: {e}")
                pass # Suppress cleanup errors during test
            

    return {
        "file": final_bytes,
        "extension": ext
    }