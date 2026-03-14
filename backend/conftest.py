"""
conftest.py — place at backend/ next to pytest.ini.

Loaded by pytest before any test module is collected, so sys.modules
is patched before any src/ import runs.

When you hit a new collection-time error:

    ModuleNotFoundError: No module named 'foo.bar'
        → add "foo" and "foo.bar" to _MODULES

    ImportError: cannot import name 'Baz' from 'foo.bar'
        → add "Baz" to _ATTRS["foo.bar"]
"""
import sys
import types
from unittest.mock import MagicMock


def _stub(name: str) -> types.ModuleType:
    """Empty module — all attribute access returns a MagicMock."""
    mod = types.ModuleType(name)
    mod.__path__ = []
    mod.__package__ = name
    mod.__getattr__ = lambda attr: MagicMock()
    return mod


# ── 1. Module paths (parent before child) ────────────────────────────────────

_MODULES = [
    # AWS Bedrock bidirectional streaming SDK  (sonic.py)
    "aws_sdk_bedrock_runtime",
    "aws_sdk_bedrock_runtime.client",
    "aws_sdk_bedrock_runtime.config",
    "aws_sdk_bedrock_runtime.models",

    # Smithy AWS credential helper  (sonic.py)
    "smithy_aws_core",
    "smithy_aws_core.identity",
    "smithy_aws_core.identity.environment",

    # PDF utilities  (bedrock_model_service.py, patient.py)
    "pdf2image",
    "pdf2image.exceptions",

    # Pillow / PIL  (bedrock_model_service.py)
    "PIL",
    "PIL.Image",

    # PDF reading/writing  (utils/helpers.py)
    "pypdf",

    # Word → PDF conversion  (utils/helpers.py)
    "docx2pdf",

    # python-docx — Word document manipulation  (utils/helpers.py likely)
    "docx",
    "docx.shared",
    "docx.enum",
    "docx.enum.text",
    "docx.oxml",
    "docx.oxml.ns",
]

for _name in _MODULES:
    if _name not in sys.modules:
        sys.modules[_name] = _stub(_name)


# ── 2. Named attributes for explicit "from X import Y" statements ─────────────

_ATTRS: dict[str, list[str]] = {
    # sonic.py
    "aws_sdk_bedrock_runtime.client": [
        "BedrockRuntimeClient",
        "InvokeModelWithBidirectionalStreamOperationInput",
    ],
    "aws_sdk_bedrock_runtime.config": [
        "Config",
    ],
    "aws_sdk_bedrock_runtime.models": [
        "BidirectionalInputPayloadPart",
        "InvokeModelWithBidirectionalStreamInputChunk",
    ],
    "smithy_aws_core.identity.environment": [
        "EnvironmentCredentialsResolver",
    ],

    # bedrock_model_service.py / patient.py
    "pdf2image": [
        "convert_from_bytes",
        "convert_from_path",
    ],
    "pdf2image.exceptions": [
        "PDFInfoNotInstalledError",
        "PDFPageCountError",
        "PDFSyntaxError",
    ],

    # bedrock_model_service.py
    "PIL": [
        "Image",
    ],
    "PIL.Image": [
        "open",
        "Image",
        "ANTIALIAS",
        "LANCZOS",
    ],

    # utils/helpers.py
    "pypdf": [
        "PdfReader",
        "PdfWriter",
        "PdfMerger",
    ],
    "docx2pdf": [
        "convert",
    ],
    "docx": [
        "Document",
    ],
    "docx.shared": [
        "Inches",
        "Pt",
        "RGBColor",
        "Cm",
        "Emu",
    ],
    "docx.enum.text": [
        "WD_ALIGN_PARAGRAPH",
        "WD_LINE_SPACING",
    ],
    "docx.oxml.ns": [
        "qn",
    ],
}

for _mod_name, _attrs in _ATTRS.items():
    _mod = sys.modules[_mod_name]
    for _attr in _attrs:
        if not hasattr(_mod, _attr):
            setattr(_mod, _attr, MagicMock())