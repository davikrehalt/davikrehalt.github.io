import google.generativeai as genai
import google.ai.generativelanguage as glm
import openai
import anthropic

import os
import argparse
import logging
import subprocess
from dotenv import load_dotenv
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path
from io import BytesIO
import base64
import re
import time
import pypdf
import json
import shutil
import random
import tempfile
from datetime import datetime
import math
import httpx # For fetching files from URLs if needed later
import asyncio # ADDED

# --- Configuration ---

# Load environment variables
load_dotenv()

# --- Logging Setup ---
# Configure logger instance
logger = logging.getLogger("pdf_translator")
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s] - %(message)s')

# Console Handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

# File Handler will be added later once WORKING_DIR is set

# --- Model Configuration ---
def get_model_configs():
    """Return available model configurations suitable for PDF processing."""
    # Note: Verify model names and capabilities against provider documentation
    # as they change frequently. Especially check PDF input methods and caching.
    return {
        # === Gemini Models ===
        "gemini-2.5-pro-preview": {
            "provider": "gemini",
            "name": "gemini-2.5-pro-preview-03-25",
            "nickname": "Gemini 2.5 Pro Preview",
            "supports_pdf_native": True,
            "supports_cache": True,
            "context_window_tokens": 2097152,
        },
        "gemini-2.5-pro-exp": { # ADDED
            "provider": "gemini",
            "name": "gemini-2.5-pro-exp-03-25",
            "nickname": "Gemini 2.5 Pro Experimental",
            "supports_pdf_native": True, # Assuming same capabilities
            "supports_cache": True,      # Assuming same capabilities
            "context_window_tokens": 2097152, # Assuming same capabilities
        },
        "gemini-2.0-flash": {
            "provider": "gemini",
            "name": "gemini-2.0-flash-latest",
            "nickname": "Gemini 2.0 Flash",
            "supports_pdf_native": True,
            "supports_cache": False,
            "context_window_tokens": 1048576,
        },

        # === OpenAI Models ===
        "gpt-4o": {
            "provider": "openai",
            "name": "gpt-4o",
            "nickname": "GPT-4o",
            "supports_pdf_native": True,
            "supports_cache": False,
            "context_window_tokens": 128000,
        },
        "gpt-4o-mini": {
            "provider": "openai",
            "name": "gpt-4o-mini",
            "nickname": "GPT-4o mini",
            "supports_pdf_native": True,
            "supports_cache": False,
            "context_window_tokens": 128000,
        },

         # === Claude Models ===
        "claude-3.7-sonnet": {
             "provider": "claude",
             "name": "claude-3-7-sonnet-20250219",
             "nickname": "Claude 3.7 Sonnet",
             "supports_pdf_native": True,
             "supports_cache": True,
             "context_window_tokens": 200000,
        },
        "claude-3.5-sonnet": {
            "provider": "claude",
            "name": "claude-3-5-sonnet-20240620",
            "nickname": "Claude 3.5 Sonnet",
            "supports_pdf_native": True,
            "supports_cache": True,
            "context_window_tokens": 200000,
        },
         # Add 3.5 Haiku if needed and supported
         # "claude-3.5-haiku": {
         #     "provider": "claude",
         #     "name": "claude-3-5-haiku-20241022",
         #     "nickname": "Claude 3.5 Haiku",
         #     "supports_pdf_native": True, # Check documentation
         #     "supports_cache": True, # Check documentation
         #     "context_window_tokens": 200000, # Example value
         # },
    }

# Default provider and model if not specified
DEFAULT_PROVIDER = "gemini"
DEFAULT_MODEL_KEY_MAP = {
    "gemini": "gemini-2.5-pro-preview",
    "openai": "gpt-4o",
    "claude": "claude-3.7-sonnet",
}

# Global variables for the chosen provider/model and clients
TARGET_PROVIDER: Optional[str] = None
TARGET_MODEL_INFO: Optional[Dict[str, Any]] = None
gemini_client: Optional[Any] = None # MODIFIED: Use Any for type hint
openai_client: Optional[openai.OpenAI] = None
anthropic_client: Optional[anthropic.Anthropic] = None

# Max retries for a single chunk (API or Compile failure)
MAX_CHUNK_RETRIES = 3
# Target number of pages per chunk
TARGET_PAGES_PER_CHUNK = 10 # User can override via CLI

# --- Path and Processing Settings ---
INPUT_PDF_PATH = "input.pdf"
WORKING_DIR = Path("latex_processing_llm") # More generic name
SAVED_CHUNKS_DIR = WORKING_DIR / "saved_pdf_chunks" # DIRECTORY FOR SAVED CHUNKS
STATE_FILE = WORKING_DIR / "state.json"
FINAL_TEX_FILE = WORKING_DIR / "final_document.tex"
FINAL_PDF_FILE = WORKING_DIR / "final_document.pdf"
LATEX_AUX_DIR = WORKING_DIR / "latex_aux"
# REMOVED FAILED_CHUNKS_DIR
PROMPT_LOG_FILE = WORKING_DIR / "prompts_and_responses.log"

# --- System Prompts ---
# Added a dedicated system prompt for LaTeX tasks
SYSTEM_PROMPT_LATEX = "You are an expert in LaTeX document generation and translation. Follow instructions precisely, maintain mathematical accuracy, and ensure the output is valid, compilable LaTeX code. Respond ONLY with the requested LaTeX code unless otherwise specified."

# --- Placeholder Fallback Preamble ---
fallback_preamble = """\\documentclass[11pt]{amsart}
\\usepackage[utf8]{inputenc}
\\usepackage[T1]{fontenc}
\\usepackage{textcomp}
\\usepackage{amsmath, amssymb, amsthm}
\\usepackage{mathtools}
\\usepackage{mathrsfs}
\\usepackage[mathcal]{euscript}
\\usepackage{amsfonts}
\\usepackage{bm}
\\usepackage{graphicx}
\\usepackage{xcolor}
\\usepackage[verbose,letterpaper,tmargin=3cm,bmargin=3cm,lmargin=2.5cm,rmargin=2.5cm]{geometry}
\\usepackage{microtype}
\\usepackage{csquotes}
\\usepackage{hyperref}
\\hypersetup{
    colorlinks=true,
    linkcolor={red!50!black},
    citecolor={blue!50!black},
    urlcolor={blue!80!black}
}
\\usepackage[capitalise]{cleveref}
\\usepackage{tikz}
\\usepackage{tikz-cd}
\\usetikzlibrary{matrix, arrows, decorations.markings}
\\usepackage[shortlabels]{enumitem}
\\usepackage{booktabs}
\\theoremstyle{plain}
\\newtheorem{theorem}{Theorem}[section]
\\newtheorem{lemma}[theorem]{Lemma}
\\newtheorem{proposition}[theorem]{Proposition}
\\newtheorem{corollary}[theorem]{Corollary}
\\theoremstyle{definition}
\\newtheorem{definition}[theorem]{Definition}
\\newtheorem{example}[theorem]{Example}
\\newtheorem{remark}[theorem]{Remark}
\\DeclareMathOperator{\\Hom}{Hom}
\\DeclareMathOperator{\\End}{End}
\\DeclareMathOperator{\\Spec}{Spec}
\\DeclareMathOperator{\\Proj}{Proj}
\\title{Translated Document (Fallback Preamble)}
\\author{AI Translator Script}
\\date{\\today}
"""

# --- API Key Handling and Client Initialization ---
def setup_clients(): # Keep sync for now, init async clients inside if needed
    """Initializes API clients based on environment variables."""
    global gemini_client, openai_client, anthropic_client
    success = True

    # Gemini
    google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if google_api_key:
        try:
            genai.configure(api_key=google_api_key)
            # MODIFIED: Store the configured genai module
            gemini_client = genai 
            logger.info("Google AI (Gemini) client configured successfully.")
        except ImportError:
            logger.error("Google AI libraries not installed. Run: pip install google-generativeai")
            gemini_client = None # Ensure it's None if import fails
        except Exception as e:
            logger.error(f"Failed to configure Google AI: {e}")
            gemini_client = None
    else:
        logger.warning("GOOGLE_API_KEY/GEMINI_API_KEY not set. Gemini models unavailable.")
        gemini_client = None

    # OpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        try:
            openai_client = openai.OpenAI(api_key=openai_api_key)
            # Test connection slightly
            openai_client.models.list()
            logger.info("OpenAI (GPT) client initialized.")
        except ImportError:
             logger.error("OpenAI library not installed. Run: pip install openai")
             openai_client = None
        except openai.AuthenticationError:
            logger.error("OpenAI API key is invalid or expired.")
            openai_client = None
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            openai_client = None
    else:
        logger.warning("OPENAI_API_KEY not set. OpenAI models unavailable.")
        openai_client = None

    # Anthropic
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_api_key:
        try:
            anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
            # Test connection slightly (no simple list models, maybe check account?)
            # Let's assume init success means key is likely valid for now
            logger.info("Anthropic (Claude) client initialized.")
        except ImportError:
             logger.error("Anthropic library not installed. Run: pip install anthropic")
             anthropic_client = None
        except anthropic.AuthenticationError:
            logger.error("Anthropic API key is invalid or expired.")
            anthropic_client = None
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            anthropic_client = None
    else:
        logger.warning("ANTHROPIC_API_KEY not set. Claude models unavailable.")
        anthropic_client = None

    # Check if at least one client is available
    if not gemini_client and not openai_client and not anthropic_client:
        logger.critical("CRITICAL: No API clients could be initialized. Check API keys and library installations.")
        success = False

    return success

async def setup_target_provider_and_model(requested_provider=None, requested_model_key=None): # Make async
    """Sets the target provider and model based on CLI args or defaults."""
    global TARGET_PROVIDER, TARGET_MODEL_INFO, DEFAULT_PROVIDER, DEFAULT_MODEL_KEY_MAP
    all_models = get_model_configs()

    # Determine Provider
    if requested_provider and requested_provider in ["gemini", "openai", "claude"]:
        TARGET_PROVIDER = requested_provider
        logger.info(f"Using specified provider: {TARGET_PROVIDER}")
    else:
        TARGET_PROVIDER = DEFAULT_PROVIDER
        logger.info(f"Using default provider: {TARGET_PROVIDER}")
        if requested_provider:
            logger.warning(f"Invalid provider '{requested_provider}' specified, falling back to default.")

    # Check if client for target provider is available
    if TARGET_PROVIDER == "gemini" and not gemini_client:
        logger.critical(f"CRITICAL: Gemini provider selected, but client is not available. Check API key.")
        return False
    if TARGET_PROVIDER == "openai" and not openai_client:
        logger.critical(f"CRITICAL: OpenAI provider selected, but client is not available. Check API key.")
        return False
    if TARGET_PROVIDER == "claude" and not anthropic_client:
        logger.critical(f"CRITICAL: Claude provider selected, but client is not available. Check API key.")
        return False

    # Determine Model Key
    target_key = requested_model_key
    if not target_key:
        target_key = DEFAULT_MODEL_KEY_MAP.get(TARGET_PROVIDER)
        logger.info(f"Using default model for {TARGET_PROVIDER}: {target_key}")
    else:
        # Validate if the requested model belongs to the target provider
        if target_key not in all_models or all_models[target_key]['provider'] != TARGET_PROVIDER:
             logger.error(f"Error: Requested model key '{target_key}' not found or does not belong to provider '{TARGET_PROVIDER}'.")
             logger.info(f"Available models for {TARGET_PROVIDER}: {[k for k, v in all_models.items() if v['provider'] == TARGET_PROVIDER]}")
             target_key = DEFAULT_MODEL_KEY_MAP.get(TARGET_PROVIDER)
             logger.warning(f"Falling back to default model for {TARGET_PROVIDER}: {target_key}")

    if not target_key or target_key not in all_models:
        logger.critical(f"CRITICAL: Could not determine a valid model key for provider {TARGET_PROVIDER}. Cannot proceed.")
        TARGET_MODEL_INFO = None
        return False

    TARGET_MODEL_INFO = all_models[target_key]

    # Final check: Does the selected model support native PDF processing?
    if not TARGET_MODEL_INFO.get("supports_pdf_native"):
        logger.critical(f"CRITICAL: Selected model '{TARGET_MODEL_INFO['nickname']}' ({TARGET_MODEL_INFO['name']}) does not support the required PDF input method according to config. Cannot proceed.")
        TARGET_MODEL_INFO = None
        return False

    logger.info(f"Selected model: {TARGET_MODEL_INFO['nickname']} ({TARGET_MODEL_INFO['name']}) from provider '{TARGET_PROVIDER}'")
    return True

def get_provider_client(provider: str) -> Optional[Any]:
    """Returns the initialized client instance for the given provider."""
    if provider == "gemini":
        return gemini_client
    elif provider == "openai":
        return openai_client
    elif provider == "claude":
        return anthropic_client
    else:
        logger.error(f"Attempted to get client for unknown provider: {provider}")
        return None

# --- Helper Functions ---

def clean_llm_latex_output(raw_latex, is_first_chunk=False):
    """Clean up LLM LaTeX output, removing markdown and optionally preamble/doc wrappers."""
    if not raw_latex:
        return ""

    # 1. Extract from ```latex ... ``` block if present
    latex_block_match = re.search(r"```(?:latex)?\s*(.*?)\s*```", raw_latex, re.DOTALL | re.IGNORECASE)
    if latex_block_match:
        cleaned_latex = latex_block_match.group(1).strip()
        logger.debug("Extracted LaTeX from markdown block.")
    else:
        cleaned_latex = raw_latex.strip()
        logger.debug("No markdown block detected, using raw stripped output.")

    # 2. Remove \end{document} always
    cleaned_latex = re.sub(r'\\end\{document\}', '', cleaned_latex, flags=re.IGNORECASE).strip()

    # 3. If it's the first chunk, remove potential preamble/metadata aggressively
    if is_first_chunk:
        logger.debug("Aggressive cleaning for First Chunk.")
        cleaned_latex = re.sub(r'^\s*\\documentclass.*?\{.*?\}\s*', '', cleaned_latex, flags=re.MULTILINE | re.IGNORECASE)
        cleaned_latex = re.sub(r'^\s*\\usepackage.*?\{.*?\}\s*', '', cleaned_latex, flags=re.MULTILINE | re.IGNORECASE)
        cleaned_latex = re.sub(r'^\s*\\title\{.*?\}\s*', '', cleaned_latex, flags=re.DOTALL | re.IGNORECASE | re.MULTILINE)
        cleaned_latex = re.sub(r'^\s*\\author\{.*?\}\s*', '', cleaned_latex, flags=re.DOTALL | re.IGNORECASE | re.MULTILINE)
        cleaned_latex = re.sub(r'^\s*\\date\{.*?\}\s*', '', cleaned_latex, flags=re.DOTALL | re.IGNORECASE | re.MULTILINE)
        cleaned_latex = re.sub(r'^\s*\\maketitle\s*', '', cleaned_latex, flags=re.IGNORECASE | re.MULTILINE)
        cleaned_latex = re.sub(r'^\s*\\begin\{document\}\s*', '', cleaned_latex, flags=re.IGNORECASE | re.MULTILINE)
        cleaned_latex = cleaned_latex.strip()

    # 4. For subsequent chunks, only remove \documentclass or \begin{document} if mistakenly added
    elif not is_first_chunk:
        cleaned_latex = re.sub(r'^\s*\\documentclass.*?\{.*?\}\s*', '', cleaned_latex, flags=re.MULTILINE | re.IGNORECASE).strip()
        cleaned_latex = re.sub(r'^\s*\\begin\{document\}\s*', '', cleaned_latex, flags=re.IGNORECASE | re.MULTILINE).strip()

    return cleaned_latex

def get_total_pages(pdf_path):
    """Get the total number of pages in the PDF using pypdf."""
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        count = len(reader.pages)
        if count > 0:
            return count
        else:
            logger.error(f"pypdf reported 0 pages for {pdf_path}. Check PDF validity.")
            return 0
    except FileNotFoundError:
         logger.error(f"Input PDF not found: {pdf_path}")
         return -1
    except Exception as e:
        logger.error(f"Error getting page count with pypdf for {pdf_path}: {e}")
        return 0

def upload_pdf_to_provider(pdf_path: Path, provider: str) -> Optional[Union[genai.types.File, str]]: # ENSURE SYNC
    """Uploads the full PDF to the specified provider's file storage."""
    if not pdf_path.exists():
        logger.error(f"PDF file not found for upload: {pdf_path}")
        return None

    logger.info(f"Uploading {pdf_path.name} ({pdf_path.stat().st_size / 1024**2:.2f} MB) via {provider} API...")
    start_time = time.time()

    if provider == "gemini":
        if not gemini_client: return None
        try:
            # Use sync upload_file and get_file
            uploaded_file = genai.upload_file(path=str(pdf_path), display_name=pdf_path.name)
            while uploaded_file.state.name == "PROCESSING":
                logger.debug(f"Gemini file '{uploaded_file.name}' processing...")
                time.sleep(5) # Use sync sleep
                uploaded_file = genai.get_file(name=uploaded_file.name) # Use sync get_file
            if uploaded_file.state.name == "FAILED":
                logger.error(f"Gemini file upload failed processing: {uploaded_file.name}")
                return None
            elapsed = time.time() - start_time
            logger.info(f"Gemini: Upload successful. File: {uploaded_file.name}, State: {uploaded_file.state.name}, URI: {uploaded_file.uri} ({elapsed:.2f}s)")
            return uploaded_file # Return the File object
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Gemini: Error uploading file after {elapsed:.2f}s: {e}", exc_info=True)
            return None

    elif provider == "openai":
        if not openai_client: return None
        try:
            # REMOVE async/await/to_thread for sync client
            with open(pdf_path, "rb") as f:
                 response = openai_client.files.create(file=f, purpose="user_data") # Use sync create
            elapsed = time.time() - start_time
            logger.info(f"OpenAI: Upload successful. File ID: {response.id}, Status: {response.status} ({elapsed:.2f}s)")
            return response.id
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"OpenAI: Error uploading file after {elapsed:.2f}s: {e}", exc_info=True)
            return None

    elif provider == "claude":
        # Claude doesn't have a persistent file upload API like the others.
        # We send the file content (base64) directly with each relevant request.
        logger.info("Claude: No separate upload step needed. File content will be sent with requests.")
        return "claude_requires_inline_content" # Return a placeholder indicating no ID/object needed

    else:
        logger.error(f"Unsupported provider for PDF upload: {provider}")
        return None

def delete_provider_file(file_ref: Union[genai.types.File, str], provider: str): # ENSURE SYNC
    """Deletes an uploaded file from the specified provider."""
    if not file_ref:
        logger.debug(f"{provider}: No file reference provided for deletion.")
        return

    if provider == "gemini":
        if not gemini_client: return
        # Determine file name correctly
        if isinstance(file_ref, genai.types.File):
            file_name = file_ref.name
        elif isinstance(file_ref, str):
             file_name = file_ref # Assume string is the name
        else:
             logger.warning(f"Gemini: Invalid file_ref type for deletion: {type(file_ref)}")
             return
        logger.info(f"Gemini: Attempting to delete file: {file_name}")
        try:
            genai.delete_file(name=file_name) # Use sync delete (REMOVE await)
            logger.info(f"Gemini: Successfully deleted file: {file_name}")
        except Exception as e:
            logger.warning(f"Gemini: Could not delete file {file_name}: {e}")

    elif provider == "openai":
        if not openai_client or not isinstance(file_ref, str): return
        file_id = file_ref
        logger.info(f"OpenAI: Attempting to delete file: {file_id}")
        try:
            openai_client.files.delete(file_id=file_id) # Use sync delete (REMOVE await)
            logger.info(f"OpenAI: Successfully deleted file: {file_id}")
        except Exception as e:
            logger.warning(f"OpenAI: Could not delete file {file_id}: {e}")

    elif provider == "claude":
        # No file to delete as it's sent inline
        logger.debug("Claude: No file deletion necessary.")
        pass

    else:
        logger.warning(f"Unsupported provider for file deletion: {provider}")

def create_pdf_chunk_file(
    original_pdf_path: Path,
    start_page: int,
    end_page: int,
    output_dir: Path # CHANGED: Now SAVED_CHUNKS_DIR
) -> Optional[Path]:
    """Extracts a page range into a PDF file in the specified directory.
    Returns the path to the file or None on error.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # Use a consistent naming scheme based on original page numbers
    chunk_file_name = f"chunk_p{start_page}-p{end_page}.pdf"
    chunk_file_path = output_dir / chunk_file_name

    # Avoid recreation if exists?
    # if chunk_file_path.exists():
    #     logger.debug(f"Chunk PDF already exists: {chunk_file_path.name}")
    #     return chunk_file_path

    try:
        reader = pypdf.PdfReader(str(original_pdf_path))
        writer = pypdf.PdfWriter()

        start_idx = start_page - 1
        end_idx = end_page

        if start_idx < 0 or end_idx > len(reader.pages):
            logger.error(f"Invalid page range for chunk file: {start_page}-{end_page} (Total: {len(reader.pages)})")
            return None

        for i in range(start_idx, end_idx):
            writer.add_page(reader.pages[i])

        with open(chunk_file_path, "wb") as fp:
            writer.write(fp)
        
        logger.debug(f"Created PDF chunk file: {chunk_file_path.name} (Orig. Pages {start_page}-{end_page}) saved to {output_dir.name}")
        return chunk_file_path
    except Exception as e:
        logger.error(f"Error creating PDF chunk file {chunk_file_path.name}: {e}")
        return None

def estimate_token_count(text: Optional[str] = None, num_pages: int = 0, provider: str = "unknown") -> int:
    """Roughly estimate token count."""
    text_tokens = 0
    page_tokens = 0
    # Simple approx: 3-4 chars per token
    if text:
        text_tokens = math.ceil(len(text) / 3.5)

    # Provider-specific page cost (very approximate, check docs)
    if provider == "gemini":
        page_tokens = num_pages * 258 # From Gemini docs example
    elif provider == "claude":
        page_tokens = num_pages * 1500 # Placeholder based on Claude docs text/image estimate
    elif provider == "openai":
        # OpenAI pricing based on text + image per page is complex. Let's use a rough estimate.
        page_tokens = num_pages * 1000 # Very rough guess

    return text_tokens + page_tokens

def log_prompt_and_response(
    task_id: Union[int, str], # Chunk number or task name like 'preamble_synthesis'
    context_id: str, # Page range or other context identifier
    provider: str,
    model_name: str,
    prompt: str,
    response: Any,
    success: bool,
    error_msg: Optional[str] = None
):
    """Logs prompt/response details to a separate file, generalized for tasks."""
    try:
        with open(PROMPT_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"--- Task: {task_id} ({context_id}) | {datetime.now().isoformat()} | Provider: {provider} | Model: {model_name} ---\\n")
            # Limit prompt logging size if needed
            log_prompt = prompt
            if len(log_prompt) > 5000: # Log first/last 2.5k chars
                log_prompt = log_prompt[:2500] + "\\n... (prompt truncated) ...\\n" + log_prompt[-2500:]
            f.write("--- PROMPT --->\\n")
            f.write(log_prompt + "\\n")
            f.write("<--- END PROMPT ---\\n")
            f.write("--- RESPONSE --->\\n")
            if success:
                if isinstance(response, str):
                    log_resp = response
                    if len(log_resp) > 2000: # Log first/last 1k chars
                         log_resp = log_resp[:1000] + "\\n... (response truncated) ...\\n" + log_resp[-1000:]
                    f.write(f"Status: SUCCESS\\nOutput:\\n{log_resp}\\n")
                else:
                     f.write(f"Status: SUCCESS\\nRaw Output (type {type(response)}):\\n{response!r}\\n")
            else:
                f.write(f"Status: FAILED\\nError: {error_msg}\\n")
                if response:
                    f.write(f"Raw Error Response (type {type(response)}):\\n{response!r}\\n")
            f.write("<--- END RESPONSE ---\\n\\n")
    except Exception as e:
        logger.error(f"Error writing to prompt log file: {e}")

# --- NEW: General Prompt Sending Function ---
def send_prompt_to_provider( # ENSURE SYNC
    prompt: Union[str, list],
    provider: str,
    model_info: Dict[str, Any],
    client: Any,
    system_prompt: Optional[str] = None,
    files: Optional[List[Union[genai.types.File, Path]]] = None, # Use correct type hint genai.types.File
    temperature: float = 0.3,
    max_retries: int = 2,
    timeout_seconds: int = 300
) -> Dict[str, Any]:
    """Sends a text prompt (optionally with files for Gemini) to the specified provider.

    Handles basic retry logic.
    Returns a dictionary: {'status': 'success', 'content': text} or 
                          {'status': 'error', 'error_message': msg}
    """
    model_name = model_info['name']
    nickname = model_info.get('nickname', model_name)
    logger.debug(f"Sending general prompt to {provider} model {nickname}...")

    for attempt in range(1, max_retries + 1):
        logger.debug(f"Attempt {attempt}/{max_retries} for {provider} call...")
        response_data = {"status": "error", "error_message": f"Attempt {attempt} failed for {provider}"}
        start_time = time.time()
        temp_files_to_delete_locally = [] # Track temp files created here

        try:
            if provider == "gemini":
                if not client: raise ValueError("Gemini client (genai module) not provided.")
                # MODIFIED: Instantiate model directly from the genai module (stored in 'client')
                model_instance = client.GenerativeModel(model_name)
                generation_config = genai.types.GenerationConfig(temperature=temperature)
                safety_settings = [{"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_DANGEROUS_CONTENT"]]

                # Process files: Use existing File or upload Paths
                processed_files = []
                if files:
                    for file_item in files:
                        if isinstance(file_item, Path):
                            logger.debug(f"Gemini (send_prompt): Uploading temp file {file_item.name}...")
                            # Use sync upload
                            uploaded_file = upload_pdf_to_provider(file_item, provider)
                            if not uploaded_file:
                                raise RuntimeError(f"Temporary file upload failed in send_prompt: {file_item.name}")
                            processed_files.append(uploaded_file)
                            temp_files_to_delete_locally.append(uploaded_file) # Mark for deletion later in this function
                        elif isinstance(file_item, genai.types.File): # Use correct type hint
                            processed_files.append(file_item) # Use existing file reference
                        else:
                             logger.warning(f"Gemini (send_prompt): Unsupported file type: {type(file_item)}")
                
                content_parts = processed_files + ([prompt] if isinstance(prompt, str) else prompt)

                # Use sync generate_content (REMOVE await)
                response = model_instance.generate_content(
                     content_parts,
                     generation_config=generation_config,
                     safety_settings=safety_settings,
                     request_options={'timeout': timeout_seconds}
                )
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    raise RuntimeError(f"Gemini prompt blocked: {response.prompt_feedback.block_reason}")
                if response.parts:
                    response_text = "".join(p.text for p in response.parts if hasattr(p, 'text')).strip()
                    if response_text:
                        response_data = {"status": "success", "content": response_text}
                        break # Success!
                    else:
                        response_data["error_message"] = "Gemini returned empty response."
                else:
                    finish_reason = response.candidates[0].finish_reason.name if response.candidates else "UNKNOWN"
                    response_data["error_message"] = f"Gemini returned no usable parts. Finish Reason: {finish_reason}"

            elif provider == "openai":
                if not client: raise ValueError("OpenAI client not provided.")
                # OpenAI API generally takes files via IDs in the prompt, not as separate args here.
                # This function assumes the caller has embedded file IDs if needed.
                messages = []
                if system_prompt: messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                # Use sync completions.create (REMOVE await)
                response = client.chat.completions.create(
                     model=model_name,
                     messages=messages,
                     max_tokens=4096,
                     temperature=temperature,
                     timeout=float(timeout_seconds)
                 )
                if response.choices and response.choices[0].message and response.choices[0].message.content:
                    response_text = response.choices[0].message.content.strip()
                    if response_text:
                        response_data = {"status": "success", "content": response_text}
                        break # Success!
                    else:
                        response_data["error_message"] = "OpenAI returned empty content."
                else:
                    finish_reason = response.choices[0].finish_reason if response.choices else "N/A"
                    response_data["error_message"] = f"OpenAI returned no usable content. Finish Reason: {finish_reason}"

            elif provider == "claude":
                if not client: raise ValueError("Anthropic client not provided.")
                # Claude also generally takes files inline in the prompt content blocks.
                # This function assumes the caller prepared the prompt string correctly.
                messages = [{"role": "user", "content": prompt}]
                # Use sync messages.create (REMOVE await)
                response = client.messages.create(
                     model=model_name,
                     max_tokens=4096,
                     messages=messages,
                     system=system_prompt if system_prompt else "You are a helpful assistant.",
                     temperature=temperature
                 )
                if response.content and isinstance(response.content, list) and len(response.content) > 0 and response.content[0].type == "text":
                    response_text = response.content[0].text.strip()
                    if response_text:
                        response_data = {"status": "success", "content": response_text}
                        break # Success!
                    else:
                         response_data["error_message"] = "Claude returned empty text content."
                elif response.stop_reason == 'error':
                    response_data["error_message"] = f"Claude API error. Stop Reason: {response.stop_reason}"
                else:
                     response_data["error_message"] = f"Claude returned unexpected content structure or stop reason: {response.stop_reason}"
            
            else:
                response_data["error_message"] = f"Unsupported provider in send_prompt: {provider}"
                break # Don't retry unsupported provider

        except Exception as e:
            logger.warning(f"Error during {provider} call (Attempt {attempt}/{max_retries}): {e}")
            response_data["error_message"] = f"Exception during {provider} call: {e}"
        
        finally:
            # Clean up temporary files uploaded by *this function call*
            for file_ref in temp_files_to_delete_locally: # Correct variable name
                delete_provider_file(file_ref, "gemini") # Call sync delete

            elapsed = time.time() - start_time
            logger.debug(f"{provider} attempt {attempt} finished in {elapsed:.2f}s. Status: {response_data['status']}")

        # If not successful and retries remain, wait before next attempt
        if response_data["status"] != "success" and attempt < max_retries:
            wait_time = random.uniform(2, 5) * attempt # Exponential backoff (simple)
            logger.info(f"Waiting {wait_time:.1f}s before retry...")
            time.sleep(wait_time) # Use sync sleep

    return response_data

# --- Provider-Specific LaTeX Generation ---

async def get_latex_from_chunk_gemini( # CHANGE to async def
    chunk_pdf_path: Path, # Path to the chunk PDF file
    chunk_num: int,
    total_chunks: int,
    start_page: int,
    end_page: int,
    start_desc: str, # ADDED
    end_desc: str, # ADDED
    model_info: Dict[str, Any], # ADDED model_info for the specific model
    valid_bib_keys: List[str], # ADDED
    gemini_cache_name: Optional[str] = None # ADDED cache name
) -> Tuple[Optional[str], bool, Optional[str]]:
    """Gets LaTeX for specific pages using Gemini, uploading the chunk PDF."""
    global gemini_client # Use the initialized client
    if not gemini_client: # Removed check for TRANSLATION_MODEL_PRIMARY, use model_info
        return None, False, "Gemini client not available."

    model_name = model_info['name']
    nickname = model_info['nickname']
    page_range_str = f"Pages {start_page}-{end_page}"
    desc_range_str = f'("{start_desc}" to "{end_desc}")'

    logger.info(f"Gemini: Requesting LaTeX for Chunk {chunk_num} {desc_range_str} from {nickname} using file {chunk_pdf_path.name}...")

    # --- Prompt Engineering (REVISED - Add citation instructions) ---
    bib_key_info = "No valid citation keys were extracted." # Default message
    if valid_bib_keys:
         limited_keys_str = ", ".join(valid_bib_keys[:20]) + ("..." if len(valid_bib_keys) > 20 else "")
         bib_key_info = f"The following BibTeX keys are known to be valid from the full document bibliography: [{limited_keys_str}]. Use the standard \\cite{{key}} command ONLY for these keys."

    prompt_parts = [
        f"# Task: LaTeX Translation (Chunk {chunk_num}/{total_chunks})",
        f"Analyze the **entirety** of the provided PDF document chunk ('{chunk_pdf_path.name}').",
        f'This chunk represents the content from a larger document starting around "{start_desc}" and ending around "{end_desc}".', # Use single quotes for outer f-string or escape inner quotes correctly
        f'Generate a **complete and compilable English LaTeX document** corresponding *only* to the content starting from "{start_desc}" (inclusive, likely near the start of this PDF chunk) and ending at "{end_desc}" (inclusive, likely near the end of this PDF chunk).',
        "\\n# Core Requirements:",
        "- **Include Preamble:** Your output MUST start with `\\documentclass` and include a suitable preamble with common packages (inputenc, fontenc, amsmath, amssymb, graphicx, hyperref, etc.) and any theorem environments detected.",
        "- **Include Metadata:** Include `\\title`, `\\author`, `\\date` based on the content, using placeholders if unclear.",
        "- **Include Body Wrappers:** Include `\\begin{{document}}`, `\\maketitle` (if applicable) at the start, and ensure the final output includes `\\end{{document}}` at the very end.", # MODIFIED: Used literal string
        "- **Translate:** All non-English text to English.",
        "- **Math Accuracy:** Preserve all mathematical content accurately.",
        "- **Structure:** Maintain semantic structure.",
        "- **Figures/Tables:** Use placeholders like `\\includegraphics[...]{{placeholder_chunk{chunk_num}_imgX.png}}`.",
        "- **Citations:** " + bib_key_info + " If a citation refers to a key NOT in this list, represent it textually in brackets, e.g., `[Author, Year]` or `[Reference Name]`, do NOT use the \\cite command for unknown keys.",
        "- **Escaping:** Correctly escape special LaTeX characters in translated text only.",
        f'- **Focus:** Generate LaTeX **only** for the content described, from "{start_desc}" to "{end_desc}". Exclude any bibliography section (e.g., \\begin{{thebibliography}}) that might appear within this chunk.',
        "\\n# Output Format:",
        "Provide the complete LaTeX document directly, without any surrounding markdown like ```latex ... ```."
    ]
    prompt = "\\n".join(prompt_parts)

    # --- API Call Implementation (Upload, Call, Process, Delete) ---
    start_time = time.time()
    latex_output = None
    success = False
    error_msg = None
    chunk_file_ref: Optional[genai.types.File] = None # Correct type hint

    try:
        # 1. Upload chunk (wrap sync upload_file in to_thread)
        logger.debug(f"Gemini (Chunk {chunk_num}): Uploading...")
        # Wrap sync upload_file
        chunk_file_ref = await asyncio.to_thread(
            genai.upload_file, path=str(chunk_pdf_path), display_name=f"chunk_{chunk_num}_{chunk_pdf_path.name}"
        )
        while chunk_file_ref.state.name == "PROCESSING":
            # Use await asyncio.sleep
            await asyncio.sleep(5)
            # Wrap sync get_file
            chunk_file_ref = await asyncio.to_thread(genai.get_file, name=chunk_file_ref.name)
        if chunk_file_ref.state.name == "FAILED":
            raise RuntimeError(f"Gemini file upload failed processing: {chunk_file_ref.name}")
        logger.debug(f"Gemini (Chunk {chunk_num}): Upload OK: {chunk_file_ref.name}")

        # 2. Prep model & config
        model_instance = genai.GenerativeModel(model_name)
        generation_config=genai.types.GenerationConfig(temperature=0.3)
        safety_settings = [{"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_DANGEROUS_CONTENT"]]

        # 3. Call API (wrap sync generate_content)
        logger.debug(f"Gemini (Chunk {chunk_num}): Sending prompt...")
        # Wrap sync generate_content
        response = await asyncio.to_thread(
            model_instance.generate_content,
            [chunk_file_ref, prompt],
            generation_config=generation_config,
            safety_settings=safety_settings,
            request_options={'timeout': 300}
        )

        # 4. Process response
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            error_msg = f"Gemini prompt blocked: {response.prompt_feedback.block_reason}"
            success = False
        elif response.parts:
            latex_output = "".join(part.text for part in response.parts if hasattr(part, 'text')).strip()
            success = bool(latex_output)
            if not success: error_msg = "Gemini returned empty response."
        else:
            finish_reason = response.candidates[0].finish_reason.name if response.candidates else "UNKNOWN"
            error_msg = f"Gemini returned no usable parts. Finish Reason: {finish_reason}"
            success = False

    except Exception as e:
        error_msg = f"Error during Gemini API call for chunk {chunk_num}: {e}"
        logger.error(error_msg, exc_info=True)
        success = False
        latex_output = None

    finally:
        # 5. Delete uploaded chunk (use async delete helper)
        if chunk_file_ref:
            # KEEP await asyncio.to_thread for the sync delete helper
            await asyncio.to_thread(delete_provider_file, chunk_file_ref, "gemini")

    elapsed = time.time() - start_time
    logger.info(f"Gemini (Chunk {chunk_num}): API call finished in {elapsed:.2f}s. Success: {success}")
    log_prompt_and_response(chunk_num, page_range_str, "gemini", f"{nickname} ({model_name})", prompt, latex_output if success else error_msg, success, error_msg if not success else None)
    return latex_output, success, error_msg

async def get_latex_from_chunk_openai(
    chunk_pdf_path: Path,
    chunk_num: int,
    total_chunks: int,
    start_page: int,
    end_page: int,
    start_desc: str,
    end_desc: str,
    model_info: Dict[str, Any],
    valid_bib_keys: List[str] # ADDED
) -> Tuple[Optional[str], bool, Optional[str]]:
    """Gets LaTeX for specific pages using OpenAI, uploading the chunk PDF."""
    global openai_client # Use initialized client
    if not openai_client:
        return None, False, "OpenAI client not available."

    model_name = model_info['name']
    nickname = model_info['nickname']
    page_range_str = f"Pages {start_page}-{end_page}"
    desc_range_str = f'("{start_desc}" to "{end_desc}")'

    logger.info(f"OpenAI: Requesting LaTeX for Chunk {chunk_num} {desc_range_str} from {nickname} using file {chunk_pdf_path.name}...")

    # --- Prompt Engineering --- (Add citation instructions)
    bib_key_info = "No valid citation keys were extracted." # Default message
    if valid_bib_keys:
         limited_keys_str = ", ".join(valid_bib_keys[:20]) + ("..." if len(valid_bib_keys) > 20 else "")
         bib_key_info = f"Valid BibTeX keys: [{limited_keys_str}]. Use \\cite{{key}} ONLY for these keys."

    # Update template to include bib_key_info dynamically
    user_prompt_text_template = f"""# Task: LaTeX Translation (Chunk {{chunk_num}}/{{total_chunks}})
Analyze the **entirety** of the provided PDF document chunk (uploaded as file ID {{CHUNK_FILE_ID_PLACEHOLDER}}).
This chunk represents the content from a larger document starting around "{{start_desc}}" and ending around "{{end_desc}}".
Generate a **complete and compilable English LaTeX document** corresponding *only* to the content starting from "{{start_desc}}" (inclusive, likely near the start of this PDF chunk) and ending at "{{end_desc}}" (inclusive, likely near the end of this PDF chunk).

# Core Requirements:
- **Include Preamble:** Start with `\\documentclass` and include a suitable preamble (common packages, theorems).
- **Include Metadata:** Include `\\title`, `\\author`, `\\date` based on content.
- **Include Body Wrappers:** Include `\\begin{{document}} `, `\\maketitle` (if applicable), and `\\end{{document}}` at the very end. # MODIFIED: Used literal string
- **Translate:** Non-English text to English.
- **Math Accuracy:** Preserve math accurately.
- **Structure:** Maintain semantic structure.
- **Figures/Tables:** Use placeholders like `placeholder_chunk{{chunk_num}}_imgX.png`.
- **Citations:** {bib_key_info} If citing a key NOT in this list, use a textual placeholder like `[Author, Year]` instead of \\cite.
- **Escaping:** Escape special LaTeX characters in text.
- **Focus:** Generate LaTeX **only** for content from "{{start_desc}}" to "{{end_desc}}". Exclude any bibliography section (e.g., \\begin{{thebibliography}}).

# Output Format:
Provide the complete LaTeX document directly, without markdown formatting like ```latex ... ```."""

    # Format the template with actual chunk info
    formatted_prompt_template = user_prompt_text_template.format(
        chunk_num=chunk_num,
        total_chunks=total_chunks,
        start_desc=start_desc,
        end_desc=end_desc,
        bib_key_info=bib_key_info,
        CHUNK_FILE_ID_PLACEHOLDER="{CHUNK_FILE_ID_PLACEHOLDER}" # Keep placeholder
    )

    # --- API Call Implementation ---
    start_time = time.time()
    latex_output = None
    success = False
    error_msg = None
    chunk_file_id: Optional[str] = None # OpenAI uses file ID string

    try:
        # 1. Upload the chunk PDF (wrap sync create in to_thread)
        logger.debug(f"OpenAI (Chunk {chunk_num}): Uploading chunk file {chunk_pdf_path.name}...")
        with open(chunk_pdf_path, "rb") as f:
            # Assuming client.files.create is sync based on typical SDKs
            response_upload = await asyncio.to_thread(openai_client.files.create, file=f, purpose="user_data")
        chunk_file_id = response_upload.id
        logger.debug(f"OpenAI (Chunk {chunk_num}): Chunk upload successful: {chunk_file_id}")

        # 2. Finalize prompt with file ID
        final_user_prompt = formatted_prompt_template.replace("{CHUNK_FILE_ID_PLACEHOLDER}", chunk_file_id)

        # 3. Construct messages payload
        messages = [
             {"role": "system", "content": "You are an expert LaTeX translator. Generate complete, compilable LaTeX documents based on the provided PDF content and instructions. Output only the raw LaTeX code."},
             {"role": "user", "content": final_user_prompt}
         ]

        # 4. Make the API call (use await completions.create)
        logger.debug(f"OpenAI (Chunk {chunk_num}): Sending prompt to {nickname}...")
        response = await openai_client.chat.completions.create(
             model=model_name,
             messages=messages,
             max_tokens=4096,
             temperature=0.3,
             timeout=300.0
         )

        # 5. Process the response
        if response.choices and response.choices[0].message and response.choices[0].message.content:
             latex_output = response.choices[0].message.content.strip()
             if latex_output:
                 logger.info(f"OpenAI (Chunk {chunk_num}): Received {len(latex_output)} chars of LaTeX.")
                 success = True
             else:
                 error_msg = "OpenAI returned empty content."
                 logger.warning(f"OpenAI (Chunk {chunk_num}): {error_msg}")
                 success = False
        else:
             finish_reason = response.choices[0].finish_reason if response.choices else "N/A"
             error_msg = f"OpenAI returned no usable content. Finish Reason: {finish_reason}"
             logger.warning(f"OpenAI (Chunk {chunk_num}): {error_msg}")
             success = False

    except Exception as e:
        error_msg = f"Error during OpenAI API call for chunk {chunk_num}: {e}"
        logger.error(error_msg, exc_info=True)
        success = False
        latex_output = None

    finally:
        # 6. Delete the uploaded chunk PDF (use async delete helper)
        if chunk_file_id:
            await delete_provider_file(chunk_file_id, "openai")

    elapsed = time.time() - start_time
    logger.info(f"OpenAI (Chunk {chunk_num}): API call finished in {elapsed:.2f}s. Success: {success}")
    log_prompt_and_response(chunk_num, page_range_str, "openai", f"{nickname} ({model_name})", formatted_prompt_template.replace("{CHUNK_FILE_ID_PLACEHOLDER}", chunk_file_id or "<upload failed>"), latex_output if success else error_msg, success, error_msg if not success else None)
    return latex_output, success, error_msg

async def get_latex_from_chunk_claude(
    chunk_pdf_path: Path,
    chunk_num: int,
    total_chunks: int,
    start_page: int,
    end_page: int,
    start_desc: str,
    end_desc: str,
    model_info: Dict[str, Any],
    valid_bib_keys: List[str] # ADDED
) -> Tuple[Optional[str], bool, Optional[str]]:
    """Gets LaTeX for a specific PDF chunk using Claude, sending chunk as base64."""
    global anthropic_client # Use initialized client
    if not anthropic_client:
        return None, False, "Anthropic client not available."

    model_name = model_info['name']
    nickname = model_info['nickname']
    page_range_str = f"Pages {start_page}-{end_page}"
    desc_range_str = f'("{start_desc}" to "{end_desc}")'

    logger.info(f"Claude: Requesting LaTeX for Chunk {chunk_num} {desc_range_str} from {nickname}...using chunk file {chunk_pdf_path.name}...")

    # --- Read chunk PDF and base64 encode (use to_thread) ---
    pdf_base64 = None
    media_type = "application/pdf"
    try:
        pdf_bytes = await asyncio.to_thread(chunk_pdf_path.read_bytes)
        pdf_base64_bytes = await asyncio.to_thread(base64.b64encode, pdf_bytes)
        pdf_base64 = pdf_base64_bytes.decode('utf-8')
        logger.debug(f"Claude (Chunk {chunk_num}): PDF encoded ({len(pdf_bytes) / 1024:.1f} KB).")
        if len(pdf_base64) > 15 * 1024 * 1024: # Increased limit slightly, check Claude docs
             logger.warning(f"Claude (Chunk {chunk_num}): Base64 encoded PDF is large ({len(pdf_base64)/1024**2:.1f} MB). May hit limits.")
    except Exception as e:
        logger.error(f"Claude (Chunk {chunk_num}): Failed to read/encode PDF chunk: {e}")
        return None, False, f"Failed to read/encode PDF chunk: {e}"

    # --- Prompt Engineering --- (Add citation instructions)
    bib_key_info = "No valid citation keys were extracted."
    if valid_bib_keys:
         limited_keys_str = ", ".join(valid_bib_keys[:20]) + ("..." if len(valid_bib_keys) > 20 else "")
         bib_key_info = f"Valid BibTeX keys: [{limited_keys_str}]. Use \\cite{{key}} ONLY for these keys."

    user_prompt_text = f"""# Task: LaTeX Translation (Chunk {chunk_num}/{total_chunks})
Analyze the **entirety** of the provided PDF document chunk (content included in this message).
This chunk represents the content from a larger document starting around "{start_desc}" and ending around "{end_desc}".
Generate a **complete and compilable English LaTeX document** corresponding *only* to the content starting from "{start_desc}" (inclusive, likely near the start of this PDF chunk) and ending at "{end_desc}" (inclusive, likely near the end of this PDF chunk).

# Core Requirements:
- **Include Preamble:** Your output MUST start with `\\documentclass` and include a suitable preamble (common packages, theorems).
- **Include Metadata:** Include `\\title`, `\\author`, `\\date` based on content.
- **Include Body Wrappers:** Include `\\begin{{document}}`, `\\maketitle` (if applicable), and `\\end{{document}}` at the very end. # MODIFIED: Used literal string
- **Translate:** Non-English text to English.
- **Math Accuracy:** Preserve math accurately.
- **Structure:** Maintain semantic structure.
- **Figures/Tables:** Use placeholders like `placeholder_chunk{chunk_num}_imgX.png`.
- **Citations:** {bib_key_info} If citing a key NOT in this list, use a textual placeholder like `[Author, Year]` instead of \\cite.
- **Escaping:** Escape special LaTeX characters in text.
- **Focus:** Generate LaTeX **only** for content from "{start_desc}" to "{end_desc}". Exclude any bibliography section (e.g., \\begin{{thebibliography}}).

# Output Format:
Provide the complete LaTeX document directly, without any surrounding markdown like ```latex ... ```."""

    # --- API Call Implementation ---
    start_time = time.time()
    latex_output = None
    success = False
    error_msg = None

    try:
        # 1. Construct messages payload
        # Claude vision models expect image type for PDFs
        content_blocks = [
             {
                 "type": "image",
                 "source": {"type": "base64", "media_type": media_type, "data": pdf_base64}
             },
             {
                 "type": "text",
                 "text": user_prompt_text
             }
         ]
        messages = [{"role": "user", "content": content_blocks}]

        # 2. Make the API call (use await messages.create)
        logger.debug(f"Claude (Chunk {chunk_num}): Sending prompt to {nickname}...")
        response = await asyncio.to_thread(
            anthropic_client.messages.create,
             model=model_name,
             max_tokens=4096,
             messages=messages,
             system="You are an expert LaTeX translator. Generate complete, compilable LaTeX documents based on the provided PDF content and instructions. Output only the raw LaTeX code.",
             temperature=0.3
         ) # KEEP await/to_thread

        # 3. Process the response
        if response.content and isinstance(response.content, list) and response.content[0].type == "text":
            latex_output = response.content[0].text.strip()
            if latex_output:
                logger.info(f"Claude (Chunk {chunk_num}): Received {len(latex_output)} chars of LaTeX.")
                success = True
            else:
                error_msg = "Claude returned empty text content."
                logger.warning(f"Claude (Chunk {chunk_num}): {error_msg}")
                success = False
        elif response.stop_reason == 'error':
            api_error_details = getattr(response, 'error', None)
            error_msg = f"Claude API error. Stop Reason: {response.stop_reason}. Details: {api_error_details}"
            logger.error(f"Claude (Chunk {chunk_num}): {error_msg}")
            success = False
        else:
            error_msg = f"Claude returned unexpected content structure or stop reason: {response.stop_reason}"
            logger.warning(f"Claude (Chunk {chunk_num}): {error_msg}")
            success = False

    except Exception as e:
        error_msg = f"Error during Claude API call for chunk {chunk_num}: {e}"
        logger.error(error_msg, exc_info=True)
        success = False
        latex_output = None

    elapsed = time.time() - start_time
    logger.info(f"Claude (Chunk {chunk_num}): API call finished in {elapsed:.2f}s. Success: {success}")
    log_prompt_and_response(chunk_num, page_range_str, "claude", f"{nickname} ({model_name})", user_prompt_text, latex_output if success else error_msg, success, error_msg if not success else None)
    return latex_output, success, error_msg

# --- Compilation and State ---
def read_state(state_file_path: Path) -> Dict[str, Any]: # MODIFIED to take path
    """Reads the processing state, converting keys back to int."""
    default_state = {
        "last_completed_chunk_index": 0,
        "provider": None,
        "original_pdf_upload_ref": None,
        "successful_preambles": {}, # Dict[int, Optional[str]]
        "successful_bodies": {}    # Dict[int, str]
    }
    if state_file_path.exists():
        try:
            with open(state_file_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            # Basic validation
            if isinstance(state.get("last_completed_chunk_index"), int):
                 logger.info(f"Loaded state: Last completed chunk index = {state['last_completed_chunk_index']}, Provider = {state.get('provider', 'N/A')}, FileRef = {state.get('original_pdf_upload_ref', 'N/A')}")
                 # Ensure keys exist and convert back to int
                 loaded_state = default_state.copy() # Start with defaults
                 loaded_state.update(state) # Overwrite with loaded values
                 # Convert keys for dicts back to int
                 loaded_state["successful_preambles"] = {int(k): v for k, v in loaded_state.get("successful_preambles", {}).items()}
                 loaded_state["successful_bodies"] = {int(k): v for k, v in loaded_state.get("successful_bodies", {}).items()}
                 return loaded_state
            else:
                logger.warning(f"State file {state_file_path} invalid format. Resetting.")
                return default_state.copy()
        except Exception as e:
            logger.error(f"Error reading state file {state_file_path}: {e}. Resetting.")
            return default_state.copy()
    logger.info("No state file found. Starting fresh.")
    return default_state.copy()

def write_state(state_dict):
    """Writes the current processing state."""
    try:
        if STATE_FILE.exists(): shutil.copy2(STATE_FILE, STATE_FILE.with_suffix(".json.bak"))
        with open(STATE_FILE, 'w', encoding='utf-8') as f: json.dump(state_dict, f, indent=2)
        logger.info(f"State updated: Last completed chunk = {state_dict.get('last_completed_chunk_index')}, Provider = {state_dict.get('provider')}")
    except Exception as e: logger.error(f"Could not write state file {STATE_FILE}: {e}")

# --- NEW Helper: Get Chunk Definitions via AI --- 
def get_chunk_definitions_from_ai(
    original_pdf_path: Path,
    provider: str,
    model_info: Dict[str, Any]
) -> Optional[str]:
    """Calls the specified cheap AI model to analyze the PDF and suggest chunk definitions.
    Returns the raw string response from the AI, expecting JSON.
    """
    logger.info(f"Calling {provider} model {model_info['nickname']} for chunk definition analysis...")
    model_name = model_info['name']
    raw_response = None
    start_time = time.time()

    # Placeholder Prompt (Needs refinement!)
    prompt = f"""
Analyze the structure of the provided PDF document ('{original_pdf_path.name}'). 
Identify logical sections (like chapters, major sections, distinct topics, long proofs) suitable for translation chunks. 
Aim for roughly 8-12 pages per chunk, but prioritize semantic boundaries over exact page count.

Respond ONLY with a valid JSON list, where each item represents a chunk and has the following keys:
- "chunk_num": Integer, starting from 1.
- "start_page": Integer, the starting page number of the chunk.
- "end_page": Integer, the ending page number of the chunk.
- "start_desc": A short (max 10 words) textual description of where the chunk starts (e.g., "Start of Section 3.1", "Beginning of Theorem 4 proof", "Top of page 5").
- "end_desc": A short (max 10 words) textual description of where the chunk ends (e.g., "End of Section 3.1", "Conclusion of Theorem 4 proof", "Bottom of page 12").

Example JSON list item:
{{
  "chunk_num": 1,
  "start_page": 1,
  "end_page": 9,
  "start_desc": "Start of Introduction",
  "end_desc": "End of Section 1.2"
}}

Ensure the page ranges cover the entire document without gaps or overlaps. Output ONLY the JSON list.
"""

    # --- Implement provider-specific logic --- 
    temp_file_ref: Optional[Union[glm.File, str]] = None # Type hint for clarity
    try:
        if provider == "gemini":
            if not gemini_client: raise ValueError("Gemini client not initialized.")
            # Upload original PDF temporarily
            logger.info("Gemini (Chunk Def): Uploading original PDF...")
            temp_file_ref = genai.upload_file(path=str(original_pdf_path), display_name=f"chunk_def_{original_pdf_path.name}")
            while temp_file_ref.state.name == "PROCESSING": time.sleep(3); temp_file_ref = genai.get_file(name=temp_file_ref.name)
            if temp_file_ref.state.name == "FAILED": raise RuntimeError("Gemini chunk def PDF upload failed.")
            logger.info(f"Gemini (Chunk Def): Upload complete: {temp_file_ref.name}")
            
            # Make API call
            # MODIFIED: Instantiate model directly from genai module
            model_instance = genai.GenerativeModel(model_name)
            generation_config=genai.types.GenerationConfig(
                temperature=0.1, 
                response_mime_type="application/json" # Request JSON output directly
            )
            response = model_instance.generate_content(
                 [temp_file_ref, prompt], 
                 generation_config=generation_config,
                 safety_settings=[{"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_DANGEROUS_CONTENT"]], 
                 request_options={'timeout': 180} 
            )
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                logger.error(f"Gemini (Chunk Def): Blocked - {response.prompt_feedback.block_reason}")
            elif response.parts:
                 # Assuming JSON is directly in the text part
                 raw_response = "".join(p.text for p in response.parts if hasattr(p, 'text'))
            else: logger.warning("Gemini (Chunk Def): No text parts in response.")

        elif provider == "openai":
            if not openai_client: raise ValueError("OpenAI client not initialized.")
            # Upload original PDF temporarily
            logger.info("OpenAI (Chunk Def): Uploading original PDF...")
            with open(original_pdf_path, "rb") as f:
                 response_upload = openai_client.files.create(file=f, purpose="user_data")
            temp_file_ref = response_upload.id # Store File ID
            logger.info(f"OpenAI (Chunk Def): Upload complete: {temp_file_ref}")
            
            # Construct messages payload - Use simpler prompt asking for JSON
            simple_prompt = "Analyze the provided PDF and generate chunk definitions as a JSON list based on logical sections (approx 8-12 pages each). Format: [{'chunk_num':int, 'start_page':int, 'end_page':int, 'start_desc':str, 'end_desc':str},...]"
            messages = [
                 {"role": "system", "content": "You are an expert document analyzer. Respond ONLY with a valid JSON list matching the specified format. Keys must be double-quoted."}, 
                 {"role": "user", "content": [
                     {"type": "input_file", "file_id": temp_file_ref},
                     {"type": "text", "text": simple_prompt}
                 ]}
             ]
            
            # Make API call
            response = openai_client.chat.completions.create(
                 model=model_name,
                 messages=messages,
                 max_tokens=3072, 
                 temperature=0.1,
                 response_format={"type": "json_object"}, 
                 timeout=180.0
             )
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                 raw_response = response.choices[0].message.content
            else: logger.warning(f"OpenAI (Chunk Def): Invalid response structure. Finish reason: {response.choices[0].finish_reason if response.choices else 'N/A'}")

        elif provider == "claude":
            if not anthropic_client: raise ValueError("Anthropic client not initialized.")
            # Read and Base64 encode the original PDF
            logger.info("Claude (Chunk Def): Reading and encoding original PDF...")
            try:
                 pdf_bytes = original_pdf_path.read_bytes()
                 pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
                 logger.info(f"Claude (Chunk Def): PDF encoded ({len(pdf_bytes) / 1024**2:.2f} MB).")
                 # TODO: Add size check here if needed
            except Exception as e:
                 logger.error(f"Claude (Chunk Def): Failed to read/encode PDF: {e}")
                 raise 

            # Construct messages payload - Use simpler prompt asking for JSON
            simple_prompt = "Analyze the provided PDF and generate chunk definitions as a JSON list based on logical sections (approx 8-12 pages each). Respond ONLY with the JSON list. Keys must be double-quoted. Format: [{'chunk_num':int, 'start_page':int, 'end_page':int, 'start_desc':str, 'end_desc':str},...]"
            content_blocks = [
                 {
                     "type": "document",
                     "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_base64}
                 },
                 {
                     "type": "text",
                     "text": simple_prompt
                 }
             ]
            messages = [{"role": "user", "content": content_blocks}]

            # Make API call
            response = anthropic_client.messages.create(
                 model=model_name,
                 max_tokens=3072, 
                 messages=messages,
                 system="You are an expert document analyzer. Respond ONLY with the requested JSON list.", 
                 temperature=0.1,
             )
            if response.content and isinstance(response.content, list) and response.content[0].type == "text":
                 # Attempt to strip potential markdown fences if JSON is inside
                 resp_text = response.content[0].text.strip()
                 json_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", resp_text, re.DOTALL)
                 if json_match:
                     raw_response = json_match.group(1) # Extract JSON from fence
                 else:
                     raw_response = resp_text # Assume raw text is JSON
            elif response.stop_reason == 'error':
                 logger.error(f"Claude (Chunk Def): API error reported. Stop Reason: {response.stop_reason}")
            else:
                 logger.warning(f"Claude (Chunk Def): Invalid response structure. Stop Reason: {response.stop_reason}")

        else:
            logger.error(f"Unsupported provider for AI chunk definition: {provider}")

    except Exception as e:
        logger.error(f"Error during AI chunk definition ({provider}): {e}", exc_info=True)
        raw_response = None

    elapsed = time.time() - start_time
    if raw_response:
        logger.info(f"AI chunk definition successful ({elapsed:.2f}s).")
        return raw_response
    else:
        logger.error(f"AI chunk definition failed ({elapsed:.2f}s).")
        return None

# --- Function: Determine Chunk Definitions (Moved Here) ---
def determine_and_save_chunk_definitions(
    pdf_path: Path,
    total_pages: int,
    target_chunk_size: int,
    provider: str,
    chunk_def_model_info: Dict[str, Any], # Keep this, needed for fallback msg
    output_file: Path,
    ai_chunk_data: Optional[List[Dict[str, Any]]] = None # ADDED Optional pre-fetched data
) -> Optional[List[Dict[str, Any]]]:
    """Determines chunk boundaries from pre-fetched AI data or falls back to fixed size.
    Saves definitions to output_file and returns the structured list.
    """
    logger.info("Determining and saving chunk definitions...")
    chunk_ranges_struct = None

    # --- Use AI Data or Fallback ---
    if ai_chunk_data:
        # Basic validation of AI data format
        try:
            if isinstance(ai_chunk_data, list) and \
               all(isinstance(item, dict) for item in ai_chunk_data) and \
               all(all(k in item for k in ['chunk_num', 'start_page', 'end_page', 'start_desc', 'end_desc']) for item in ai_chunk_data):
                logger.info(f"Using pre-fetched AI chunk data ({len(ai_chunk_data)} chunks).")
                chunk_ranges_struct = ai_chunk_data
            else:
                logger.warning("AI chunk data has invalid format. Falling back to fixed size.")
                ai_chunk_data = None # Force fallback
        except Exception as json_e:
            logger.warning(f"Error validating AI chunk data structure: {json_e}. Falling back to fixed size.")
            ai_chunk_data = None # Force fallback
    
    # --- Fallback logic (if AI data was None or invalid) ---
    if not chunk_ranges_struct:
        logger.warning(f"AI chunk definition failed or not provided/valid. Falling back to fixed-size chunks.")
        num_chunks = math.ceil(total_pages / target_chunk_size)
        chunk_ranges_struct = []
        for i in range(num_chunks):
            start_p = i * target_chunk_size + 1
            end_p = min((i + 1) * target_chunk_size, total_pages)
            chunk_ranges_struct.append({
                'chunk_num': i + 1,
                'start_page': start_p,
                'end_page': end_p,
                'start_desc': f"Start of Page {start_p}", # Generic description
                'end_desc': f"End of Page {end_p}" # Generic description
            })
        logger.info(f"Calculated {len(chunk_ranges_struct)} fixed-size chunks (Target: {target_chunk_size} pages)." )

    # Save definitions to file (if we have a structure)
    if chunk_ranges_struct:
        try:
            if output_file.exists():
                 backup_file = output_file.with_suffix(".txt.bak")
                 shutil.copy2(output_file, backup_file)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(chunk_ranges_struct, f, indent=2)
            logger.info(f"Chunk definitions saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save chunk definitions to {output_file}: {e}")
    else:
         logger.error("No chunk definitions (AI or fallback) could be determined.")

    return chunk_ranges_struct

# --- Function: Generate Final Preamble with AI (Moved Here) ---
def generate_final_preamble_with_ai(
    successful_preambles: List[str],
    provider: str,
    model_info: Dict[str, Any],
    pdf_path_for_context: Optional[Path] = None # Optional: Might help AI?
) -> Optional[str]:
    # ... (Function implementation as defined previously) ...
    """
    Attempts to synthesize a single, clean LaTeX preamble from a list of preambles
    extracted from successfully processed chunks using an AI model.

    Args:
        successful_preambles: A list of preamble strings.
        provider: The AI provider ('gemini', 'openai', 'claude').
        model_info: Configuration dictionary for the target AI model.
        pdf_path_for_context: Optional path to the original PDF for potential context.

    Returns:
        A synthesized preamble string, or None if synthesis fails or is invalid.
    """
    logger.info(f"Starting AI preamble synthesis using {provider} model {model_info.get('nickname', 'default')}.")

    valid_preambles = [p for p in successful_preambles if not p.strip().startswith("% Preamble for")]
    if not valid_preambles:
        logger.warning("No valid preambles found to synthesize from.")
        return None

    # --- Prepare Input for Prompt ---
    combined_preambles_text = "\\n\\n% --- Next Preamble Snippet ---\\n\\n".join(valid_preambles)
    max_input_chars = 50000
    if len(combined_preambles_text) > max_input_chars:
        logger.warning(f"Combined preamble text exceeds {max_input_chars} chars, truncating for AI prompt.")
        combined_preambles_text = combined_preambles_text[:max_input_chars] + "\\n... (truncated)"

    # --- Craft the Prompt ---
    prompt = f"""Analyze the following LaTeX preamble snippets extracted from different, successfully compiled chunks of a single original document.
Your task is to synthesize a single, clean, non-redundant, and valid LaTeX preamble that incorporates all *necessary* packages and definitions found in the snippets.

Guidelines:
1. Identify the document class (e.g., \\\\documentclass{{amsart}}) and use it ONCE at the beginning. If multiple classes are present, choose the most frequent or appropriate one (often 'amsart' or 'article' for math papers).
2. Include all necessary \\\\usepackage commands found in the snippets. Remove duplicate package inclusions.
3. Include relevant custom command definitions (\\\\newcommand, \\\\def, \\\\DeclareMathOperator, etc.) and theorem environments (\\\\newtheorem). Remove duplicates.
4. Try to maintain a logical order (documentclass, packages, definitions).
5. Remove any comments that are just markers like "% --- Next Preamble Snippet ---". Keep meaningful comments if they exist.
6. Ensure the output is ONLY the preamble content, ending *before* \\\\begin{{document}}. Do not include \\\\begin{{document}} or any body text.
6. Ensure the output is ONLY the preamble content, ending *before* \\begin{{document}}. Do not include \\begin{{document}} or any body text.
7. If the snippets are contradictory or nonsensical, do your best to create a reasonable standard math article preamble including common packages like amsmath, amssymb, graphicx, hyperref, amsthm.

Preamble Snippets:
```latex
{combined_preambles_text}
```

Synthesized Preamble:
"""

    # --- Call AI Provider (Corrected) ---
    ai_response_text = None
    try:
        # Pass pdf_path if Gemini potentially uses it for context
        files_for_prompt = [pdf_path_for_context] if provider == "gemini" and pdf_path_for_context else None
        provider_client = get_provider_client(provider)
        if not provider_client:
            raise ValueError(f"Could not get client for provider: {provider}")

        # Use the central function to handle API calls
        response_data = send_prompt_to_provider(
            prompt=prompt,
            provider=provider,
            model_info=model_info,
            client=provider_client,
            max_retries=2, # Maybe fewer retries for synthesis?
            temperature=0.2, # Lower temperature for more deterministic synthesis
            system_prompt=SYSTEM_PROMPT_LATEX,
            files=files_for_prompt,
            # No page_num needed
            timeout_seconds=180 # Shorter timeout for synthesis?
        )

        if response_data and response_data.get("status") == "success":
            ai_response_text = response_data.get("content")
            logger.info("AI preamble synthesis successful.")
            # Log the successful synthesis prompt/response (Corrected call)
            log_prompt_and_response(
                task_id="preamble_synthesis",
                context_id="-", # No specific page range
                provider=provider,
                model_name=model_info.get('nickname', 'default'),
                prompt=prompt,
                response=ai_response_text,
                success=True
            )
        else:
            err_msg = response_data.get("error_message", "Unknown error during AI call")
            logger.error(f"AI preamble synthesis failed: {err_msg}")
            # Log failure (Corrected call)
            log_prompt_and_response(
                task_id="preamble_synthesis",
                context_id="-",
                provider=provider,
                model_name=model_info.get('nickname', 'default'),
                prompt=prompt,
                response=None,
                success=False,
                error_msg=err_msg
            )
            return None

    except Exception as e:
        logger.error(f"Exception during AI preamble synthesis call: {e}", exc_info=True)
        # Log exception (Corrected call)
        log_prompt_and_response(
            task_id="preamble_synthesis",
            context_id="-",
            provider=provider,
            model_name=model_info.get('nickname', 'default'),
            prompt=prompt,
            response=None,
            success=False,
            error_msg=f"EXCEPTION: {e}"
        )
        return None

    # --- Validate and Clean Response ---
    if not ai_response_text or not isinstance(ai_response_text, str):
        logger.warning("AI returned empty or invalid response for preamble synthesis.")
        return None

    # Basic validation: Check for common preamble elements and absence of body
    # Refine this validation as needed
    cleaned_preamble = ai_response_text.strip()
    if not cleaned_preamble.startswith("\\documentclass"):
        # Sometimes AI might add introductory text. Try to find the start.
        start_index = cleaned_preamble.find("\\documentclass")
        if start_index != -1:
            logger.warning("AI response had leading text before \\documentclass. Attempting to clean.")
            cleaned_preamble = cleaned_preamble[start_index:]
        else:
             logger.warning("Synthesized preamble does not start with \\documentclass. Result might be invalid.")
             # Decide whether to return it anyway or fail
             # return None # Stricter approach
             # For now, return it and let LaTeX compilation catch errors

    if "\\begin{document}" in cleaned_preamble:
        logger.warning("Synthesized preamble appears to contain \\begin{document}. Truncating.")
        cleaned_preamble = cleaned_preamble.split("\\begin{document}")[0].strip()

    # Optional: Add a comment indicating AI synthesis
    final_preamble = cleaned_preamble
    
    # Log the final synthesized preamble (first/last few lines for brevity)
    preview_lines = final_preamble.splitlines()
    preview = "\\n".join(preview_lines[:5]) + ("\\n..." if len(preview_lines) > 5 else "")
    logger.debug(f"Final synthesized preamble preview:\\n{preview}")

    return final_preamble

# --- NEW: Bibliography Extraction Helper ---
async def extract_bibliography_with_ai( # CHANGE to async def
    original_pdf_path: Path,
    provider: str,
    model_info: Dict[str, Any]
) -> Optional[str]:
    """Calls the specified AI model to extract bibliography data as BibTeX.
    Returns the raw BibTeX string or None on failure.
    """
    logger.info(f"Calling {provider} model {model_info['nickname']} for BibTeX extraction...")
    model_name = model_info['name']
    raw_response = None
    start_time = time.time()

    # Prompt asking ONLY for BibTeX output
    prompt = f"""
Analyze the provided PDF document ('{original_pdf_path.name}').
Identify all bibliographic references cited or listed within the document.
Format these references strictly as a BibTeX (.bib) file.

Respond ONLY with the raw BibTeX content. Do not include any explanations, greetings, or markdown formatting like ```bibtex ... ```.

Example BibTeX entry format:
@article{{key,
  author  = {{Author Name}},
  title   = {{Article Title}},
  journal = {{Journal Name}},
  year    = {{Year}},
  volume  = {{Volume Number}},
  pages   = {{Page Range}}
}}

Start your response directly with the first BibTeX entry (e.g., `@article{{...`).
"""

    temp_file_ref: Optional[Union[genai.types.File, str]] = None
    uploaded_for_this_call = False
    try:
        provider_client = get_provider_client(provider)
        if not provider_client:
            raise ValueError(f"Could not get client for provider: {provider}")

        files_for_prompt = []
        # Call SYNC upload_pdf_to_provider
        if provider == "gemini" or provider == "openai":
            logger.info(f"{provider.capitalize()} (BibTeX): Uploading original PDF...")
            # REMOVE await from call to sync helper
            temp_file_ref = upload_pdf_to_provider(original_pdf_path, provider)
            if not temp_file_ref:
                raise RuntimeError(f"{provider.capitalize()} BibTeX PDF upload failed.")
            logger.info(f"{provider.capitalize()} (BibTeX): Upload complete.")
            if provider == "gemini":
                # Ensure temp_file_ref is the File object here if needed by send_prompt
                files_for_prompt.append(temp_file_ref)
            uploaded_for_this_call = True
        elif provider == "claude":
            logger.info("Claude (BibTeX): Reading and encoding original PDF...")
            try:
                # KEEP await asyncio.to_thread for blocking I/O
                pdf_bytes = await asyncio.to_thread(original_pdf_path.read_bytes)
                pdf_base64_bytes = await asyncio.to_thread(base64.b64encode, pdf_bytes)
                pdf_base64 = pdf_base64_bytes.decode('utf-8')
                content_blocks = [
                     {"type": "image", "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_base64}},
                     {"type": "text", "text": prompt}
                 ]
                prompt = content_blocks # Update prompt for Claude
            except Exception as e_claude_encode:
                 logger.error(f"Claude (BibTeX): Failed to read/encode PDF: {e_claude_encode}")
                 raise

        # Call SYNC send_prompt_to_provider
        # REMOVE await from call to sync helper
        response_data = send_prompt_to_provider(
            prompt=prompt,
            provider=provider,
            model_info=model_info,
            client=provider_client,
            max_retries=2,
            temperature=0.1,
            system_prompt="You are a bibliographic assistant. Respond ONLY with valid BibTeX.",
            files=files_for_prompt,
            timeout_seconds=240
        )

        # Process response_data dictionary (sync)
        if response_data and response_data.get("status") == "success":
            raw_response = response_data.get("content")
            if raw_response and "@" in raw_response[:20]:
                logger.info(f"BibTeX extraction successful.")
            else:
                log_message_prefix = "AI response received, but may not be valid BibTeX:\n"
                response_preview = (raw_response[:200] + "...") if raw_response else "(Empty Response)"
                logger.warning(log_message_prefix + response_preview)
            log_prompt_and_response(
                task_id="bibliography_extraction", context_id=original_pdf_path.name,
                provider=provider, model_name=model_info.get('nickname', 'default'),
                prompt=str(prompt), response=raw_response, success=True
            )
        else:
            err_msg = response_data.get("error_message", "Unknown error during AI call")
            logger.error(f"AI BibTeX extraction failed: {err_msg}")
            log_prompt_and_response(
                task_id="bibliography_extraction", context_id=original_pdf_path.name,
                provider=provider, model_name=model_info.get('nickname', 'default'),
                prompt=str(prompt), response=None, success=False, error_msg=err_msg
            )
            raw_response = None

    except Exception as e:
        logger.error(f"Exception during AI BibTeX extraction ({provider}): {e}", exc_info=True)
        log_prompt_and_response(
            task_id="bibliography_extraction", context_id=original_pdf_path.name,
            provider=provider, model_name=model_info.get('nickname', 'default'),
            prompt=str(prompt), response=None, success=False, error_msg=f"EXCEPTION: {e}"
        )
        raw_response = None
    finally:
        # Call SYNC delete_provider_file
        if uploaded_for_this_call and temp_file_ref:
             logger.info(f"{provider.capitalize()} (BibTeX): Cleaning up temporary PDF upload...")
             # REMOVE await from call to sync helper
             delete_provider_file(temp_file_ref, provider)

    elapsed = time.time() - start_time
    if raw_response:
        logger.info(f"AI BibTeX extraction finished ({elapsed:.2f}s).")
    else:
        logger.error(f"AI BibTeX extraction failed ({elapsed:.2f}s).")
    return raw_response

# --- NEW: BibTeX Key Extraction Helper ---
def extract_bib_keys(bibtex_content: str) -> List[str]:
    """Extracts citation keys (e.g., 'AuthorYear') from a BibTeX string."""
    if not bibtex_content:
        return []
    # Regex to find patterns like @article{key,... or @book{key,...
    # It captures the content between the first { and the first ,
    keys = re.findall(r"^\s*@.*?\{(.+?),", bibtex_content, re.MULTILINE)
    extracted_keys = [key.strip() for key in keys if key.strip()]
    logger.debug(f"Extracted {len(extracted_keys)} BibTeX keys: {extracted_keys[:10]}...")
    return extracted_keys

# --- NEW: Parse LLM LaTeX Output ---
def parse_llm_latex_output(raw_latex: Optional[str], chunk_num: int, total_chunks: int) -> Tuple[Optional[str], Optional[str]]:
    """Extracts preamble and body from raw LLM LaTeX output."""
    if not raw_latex or not isinstance(raw_latex, str):
        logger.warning(f"Chunk {chunk_num}/{total_chunks}: Received invalid or empty raw output for parsing.")
        return None, None

    # 1. Remove potential markdown fences
    cleaned_latex = raw_latex.strip()
    md_match = re.search(r"```(?:latex)?\s*(.*?)\s*```", cleaned_latex, re.DOTALL | re.IGNORECASE)
    if md_match:
        cleaned_latex = md_match.group(1).strip()
        logger.debug(f"Chunk {chunk_num}: Removed markdown fences before parsing preamble/body.")

    # 2. Extract Preamble (everything before \begin{document})
    preamble: Optional[str] = None
    body: Optional[str] = None
    begin_doc_match = re.search(r"\\begin\{document\}", cleaned_latex, re.IGNORECASE)

    if begin_doc_match:
        preamble_end_index = begin_doc_match.start()
        preamble = cleaned_latex[:preamble_end_index].strip()
        # Extract Body (between \begin{document} and \end{document})
        end_doc_match = re.search(r"\\end\{document\}", cleaned_latex[preamble_end_index:], re.IGNORECASE)
        if end_doc_match:
            body_start_index = begin_doc_match.end()
            # Adjust end_doc_match index relative to the start of the body search
            body_end_index = preamble_end_index + end_doc_match.start()
            body = cleaned_latex[body_start_index:body_end_index].strip()
        else:
            # If no \end{document}, take everything after \begin{document}
            logger.warning(f"Chunk {chunk_num}: Could not find \\\\end{{document}}. Taking all content after \\\\begin{{document}} as body.")
            body = cleaned_latex[begin_doc_match.end():].strip()
    else:
        # If no \begin{document}, assume the whole thing might be body content (or just preamble?)
        # Let's treat it as body for now, assuming preamble is less likely without begin{doc}
        logger.warning(f"Chunk {chunk_num}: Could not find \\\\begin{{document}}. Treating entire cleaned output as body.")
        body = cleaned_latex
        preamble = None # Explicitly set preamble to None

    # Log results
    preamble_len = len(preamble) if preamble else 0
    body_len = len(body) if body else 0
    logger.debug(f"Chunk {chunk_num}: Parsed preamble ({preamble_len} chars), body ({body_len} chars).")

    # Return None for body if it's empty after parsing, as it indicates failure.
    return preamble, body if body else None

# --- State Saving (Modified for Async Results) ---
def save_state(last_completed_chunk_index: int,
               # These dictionaries should contain ALL successful results up to this point
               all_successful_preambles: Dict[int, Optional[str]],
               all_successful_bodies: Dict[int, str],
               state_file_path: Path,
               # Pass the currently known provider and ref explicitly
               current_provider: Optional[str],
               current_upload_ref: Optional[str]): # Ref should be the string ID/Name
    """Writes the current processing state, including ALL successful chunks' content."""
    # Convert dict keys to strings for JSON compatibility
    preambles_to_save = {str(k): v for k, v in all_successful_preambles.items()}
    bodies_to_save = {str(k): v for k, v in all_successful_bodies.items()}

    # REMOVED: Re-reading current_state here was causing overwrites.
    # Construct state dictionary ONLY from passed arguments.
    state_dict = {
        "last_completed_chunk_index": last_completed_chunk_index,
        "provider": current_provider, # Use passed provider
        "original_pdf_upload_ref": current_upload_ref, # Use passed ref (should be string name/ID)
        "successful_preambles": preambles_to_save,
        "successful_bodies": bodies_to_save
    }
    try:
        backup_path = state_file_path.with_suffix(".json.bak")
        if state_file_path.exists():
            shutil.copy2(state_file_path, backup_path)
        with open(state_file_path, 'w', encoding='utf-8') as f:
            json.dump(state_dict, f, indent=2)
    except Exception as e:
        logger.error(f"Could not write state file {state_file_path}: {e}")

# --- Provider-Specific LaTeX Generation (MODIFIED TO ASYNC) ---

async def get_latex_from_chunk_gemini( # MODIFIED to async
    chunk_pdf_path: Path, # Path to the chunk PDF file
    chunk_num: int,
    total_chunks: int,
    start_page: int,
    end_page: int,
    start_desc: str, # ADDED
    end_desc: str, # ADDED
    model_info: Dict[str, Any], # ADDED model_info for the specific model
    valid_bib_keys: List[str], # ADDED
    gemini_cache_name: Optional[str] = None # ADDED cache name
) -> Tuple[Optional[str], bool, Optional[str]]:
    # ... (Inner logic remains the same, but uses await for genai calls) ...
    global gemini_client # Use the initialized client
    if not gemini_client: # Removed check for TRANSLATION_MODEL_PRIMARY, use model_info
        return None, False, "Gemini client not available."

    model_name = model_info['name']
    nickname = model_info['nickname']
    page_range_str = f"Pages {start_page}-{end_page}"
    desc_range_str = f'("{start_desc}" to "{end_desc}")'

    logger.info(f"Gemini: Requesting LaTeX for Chunk {chunk_num} {desc_range_str} from {nickname} using file {chunk_pdf_path.name}...")

    # --- Prompt Engineering (REVISED - Add citation instructions) ---
    bib_key_info = "No valid citation keys were extracted." # Default message
    if valid_bib_keys:
         limited_keys_str = ", ".join(valid_bib_keys[:20]) + ("..." if len(valid_bib_keys) > 20 else "")
         bib_key_info = f"The following BibTeX keys are known to be valid from the full document bibliography: [{limited_keys_str}]. Use the standard \\cite{{key}} command ONLY for these keys."

    prompt_parts = [
        f"# Task: LaTeX Translation (Chunk {chunk_num}/{total_chunks})",
        f"Analyze the **entirety** of the provided PDF document chunk ('{chunk_pdf_path.name}').",
        f'This chunk represents the content from a larger document starting around "{start_desc}" and ending around "{end_desc}".', # Use single quotes for outer f-string or escape inner quotes correctly
        f'Generate a **complete and compilable English LaTeX document** corresponding *only* to the content starting from "{start_desc}" (inclusive, likely near the start of this PDF chunk) and ending at "{end_desc}" (inclusive, likely near the end of this PDF chunk).',
        "\\n# Core Requirements:",
        "- **Include Preamble:** Your output MUST start with `\\documentclass` and include a suitable preamble with common packages (inputenc, fontenc, amsmath, amssymb, graphicx, hyperref, etc.) and any theorem environments detected.",
        "- **Include Metadata:** Include `\\title`, `\\author`, `\\date` based on the content, using placeholders if unclear.",
        "- **Include Body Wrappers:** Include `\\begin{{document}}`, `\\maketitle` (if applicable) at the start, and ensure the final output includes `\\end{{document}}` at the very end.", # MODIFIED: Used literal string
        "- **Translate:** All non-English text to English.",
        "- **Math Accuracy:** Preserve all mathematical content accurately.",
        "- **Structure:** Maintain semantic structure.",
        "- **Figures/Tables:** Use placeholders like `\\includegraphics[...]{{placeholder_chunk{chunk_num}_imgX.png}}`.",
        "- **Citations:** " + bib_key_info + " If a citation refers to a key NOT in this list, represent it textually in brackets, e.g., `[Author, Year]` or `[Reference Name]`, do NOT use the \\cite command for unknown keys.",
        "- **Escaping:** Correctly escape special LaTeX characters in translated text only.",
        f'- **Focus:** Generate LaTeX **only** for the content described, from "{start_desc}" to "{end_desc}". Exclude any bibliography section (e.g., \\begin{{thebibliography}}) that might appear within this chunk.',
        "\\n# Output Format:",
        "Provide the complete LaTeX document directly, without any surrounding markdown like ```latex ... ```."
    ]
    prompt = "\\n".join(prompt_parts)

    # --- API Call Implementation (Upload, Call, Process, Delete) ---
    start_time = time.time()
    latex_output = None
    success = False
    error_msg = None
    chunk_file_ref: Optional[genai.types.File] = None # Correct type hint

    try:
        # 1. Upload chunk (wrap sync upload_file in to_thread)
        logger.debug(f"Gemini (Chunk {chunk_num}): Uploading...")
        # Wrap sync upload_file
        chunk_file_ref = await asyncio.to_thread(
            genai.upload_file, path=str(chunk_pdf_path), display_name=f"chunk_{chunk_num}_{chunk_pdf_path.name}"
        )
        while chunk_file_ref.state.name == "PROCESSING":
            # Use await asyncio.sleep
            await asyncio.sleep(5)
            # Wrap sync get_file
            chunk_file_ref = await asyncio.to_thread(genai.get_file, name=chunk_file_ref.name)
        if chunk_file_ref.state.name == "FAILED":
            raise RuntimeError(f"Gemini file upload failed processing: {chunk_file_ref.name}")
        logger.debug(f"Gemini (Chunk {chunk_num}): Upload OK: {chunk_file_ref.name}")

        # 2. Prep model & config
        model_instance = genai.GenerativeModel(model_name)
        generation_config=genai.types.GenerationConfig(temperature=0.3)
        safety_settings = [{"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_DANGEROUS_CONTENT"]]

        # 3. Call API (wrap sync generate_content)
        logger.debug(f"Gemini (Chunk {chunk_num}): Sending prompt...")
        # Wrap sync generate_content
        response = await asyncio.to_thread(
            model_instance.generate_content,
            [chunk_file_ref, prompt],
            generation_config=generation_config,
            safety_settings=safety_settings,
            request_options={'timeout': 300}
        )

        # 4. Process response
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            error_msg = f"Gemini prompt blocked: {response.prompt_feedback.block_reason}"
            success = False
        elif response.parts:
            latex_output = "".join(part.text for part in response.parts if hasattr(part, 'text')).strip()
            success = bool(latex_output)
            if not success: error_msg = "Gemini returned empty response."
        else:
            finish_reason = response.candidates[0].finish_reason.name if response.candidates else "UNKNOWN"
            error_msg = f"Gemini returned no usable parts. Finish Reason: {finish_reason}"
            success = False

    except Exception as e:
        error_msg = f"Error during Gemini API call for chunk {chunk_num}: {e}"
        logger.error(error_msg, exc_info=True)
        success = False
        latex_output = None

    finally:
        # 5. Delete uploaded chunk (use async delete helper)
        if chunk_file_ref:
            # KEEP await asyncio.to_thread for the sync delete helper
            await asyncio.to_thread(delete_provider_file, chunk_file_ref, "gemini")

    elapsed = time.time() - start_time
    logger.info(f"Gemini (Chunk {chunk_num}): API call finished in {elapsed:.2f}s. Success: {success}")
    log_prompt_and_response(chunk_num, page_range_str, "gemini", f"{nickname} ({model_name})", prompt, latex_output if success else error_msg, success, error_msg if not success else None)
    return latex_output, success, error_msg

async def get_latex_from_chunk_openai( # MODIFIED to async
    chunk_pdf_path: Path,
    chunk_num: int,
    total_chunks: int,
    start_page: int,
    end_page: int,
    start_desc: str,
    end_desc: str,
    model_info: Dict[str, Any],
    valid_bib_keys: List[str] # ADDED
) -> Tuple[Optional[str], bool, Optional[str]]:
    # ... (Inner logic remains the same, but uses await for openai calls) ...
    global openai_client # Use initialized client
    if not openai_client:
        return None, False, "OpenAI client not available."

    model_name = model_info['name']
    nickname = model_info['nickname']
    page_range_str = f"Pages {start_page}-{end_page}"
    desc_range_str = f'("{start_desc}" to "{end_desc}")'

    logger.info(f"OpenAI: Requesting LaTeX for Chunk {chunk_num} {desc_range_str} from {nickname} using file {chunk_pdf_path.name}...")

    # --- Prompt Engineering --- (Add citation instructions)
    bib_key_info = "No valid citation keys were extracted." # Default message
    if valid_bib_keys:
         limited_keys_str = ", ".join(valid_bib_keys[:20]) + ("..." if len(valid_bib_keys) > 20 else "")
         bib_key_info = f"Valid BibTeX keys: [{limited_keys_str}]. Use \\cite{{key}} ONLY for these keys."

    # Update template to include bib_key_info dynamically
    user_prompt_text_template = f"""# Task: LaTeX Translation (Chunk {{chunk_num}}/{{total_chunks}})
Analyze the **entirety** of the provided PDF document chunk (uploaded as file ID {{CHUNK_FILE_ID_PLACEHOLDER}}).
This chunk represents the content from a larger document starting around "{{start_desc}}" and ending around "{{end_desc}}".
Generate a **complete and compilable English LaTeX document** corresponding *only* to the content starting from "{{start_desc}}" (inclusive, likely near the start of this PDF chunk) and ending at "{{end_desc}}" (inclusive, likely near the end of this PDF chunk).

# Core Requirements:
- **Include Preamble:** Start with `\\documentclass` and include a suitable preamble (common packages, theorems).
- **Include Metadata:** Include `\\title`, `\\author`, `\\date` based on content.
- **Include Body Wrappers:** Include `\\begin{{document}} `, `\\maketitle` (if applicable), and `\\end{{document}}` at the very end. # MODIFIED: Used literal string
- **Translate:** Non-English text to English.
- **Math Accuracy:** Preserve math accurately.
- **Structure:** Maintain semantic structure.
- **Figures/Tables:** Use placeholders like `placeholder_chunk{{chunk_num}}_imgX.png`.
- **Citations:** {bib_key_info} If citing a key NOT in this list, use a textual placeholder like `[Author, Year]` instead of \\cite.
- **Escaping:** Escape special LaTeX characters in text.
- **Focus:** Generate LaTeX **only** for content from "{{start_desc}}" to "{{end_desc}}". Exclude any bibliography section (e.g., \\begin{{thebibliography}}).

# Output Format:
Provide the complete LaTeX document directly, without markdown formatting like ```latex ... ```."""

    # Format the template with actual chunk info
    formatted_prompt_template = user_prompt_text_template.format(
        chunk_num=chunk_num,
        total_chunks=total_chunks,
        start_desc=start_desc,
        end_desc=end_desc,
        bib_key_info=bib_key_info,
        CHUNK_FILE_ID_PLACEHOLDER="{CHUNK_FILE_ID_PLACEHOLDER}" # Keep placeholder
    )

    # --- API Call Implementation ---
    start_time = time.time()
    latex_output = None
    success = False
    error_msg = None
    chunk_file_id: Optional[str] = None # OpenAI uses file ID string

    try:
        # 1. Upload the chunk PDF (wrap sync create in to_thread)
        logger.debug(f"OpenAI (Chunk {chunk_num}): Uploading chunk file {chunk_pdf_path.name}...")
        with open(chunk_pdf_path, "rb") as f:
            # Assuming client.files.create is sync based on typical SDKs
            response_upload = await asyncio.to_thread(openai_client.files.create, file=f, purpose="user_data")
        chunk_file_id = response_upload.id
        logger.debug(f"OpenAI (Chunk {chunk_num}): Chunk upload successful: {chunk_file_id}")

        # 2. Finalize prompt with file ID
        final_user_prompt = formatted_prompt_template.replace("{CHUNK_FILE_ID_PLACEHOLDER}", chunk_file_id)

        # 3. Construct messages payload
        messages = [
             {"role": "system", "content": "You are an expert LaTeX translator. Generate complete, compilable LaTeX documents based on the provided PDF content and instructions. Output only the raw LaTeX code."},
             {"role": "user", "content": final_user_prompt}
         ]

        # 4. Make the API call (use await completions.create)
        logger.debug(f"OpenAI (Chunk {chunk_num}): Sending prompt to {nickname}...")
        response = await openai_client.chat.completions.create(
             model=model_name,
             messages=messages,
             max_tokens=4096,
             temperature=0.3,
             timeout=300.0
         )

        # 5. Process the response
        if response.choices and response.choices[0].message and response.choices[0].message.content:
             latex_output = response.choices[0].message.content.strip()
             if latex_output:
                 logger.info(f"OpenAI (Chunk {chunk_num}): Received {len(latex_output)} chars of LaTeX.")
                 success = True
             else:
                 error_msg = "OpenAI returned empty content."
                 logger.warning(f"OpenAI (Chunk {chunk_num}): {error_msg}")
                 success = False
        else:
             finish_reason = response.choices[0].finish_reason if response.choices else "N/A"
             error_msg = f"OpenAI returned no usable content. Finish Reason: {finish_reason}"
             logger.warning(f"OpenAI (Chunk {chunk_num}): {error_msg}")
             success = False

    except Exception as e:
        error_msg = f"Error during OpenAI API call for chunk {chunk_num}: {e}"
        logger.error(error_msg, exc_info=True)
        success = False
        latex_output = None

    finally:
        # 6. Delete the uploaded chunk PDF (use async delete helper)
        if chunk_file_id:
            await delete_provider_file(chunk_file_id, "openai")

    elapsed = time.time() - start_time
    logger.info(f"OpenAI (Chunk {chunk_num}): API call finished in {elapsed:.2f}s. Success: {success}")
    log_prompt_and_response(chunk_num, page_range_str, "openai", f"{nickname} ({model_name})", formatted_prompt_template.replace("{CHUNK_FILE_ID_PLACEHOLDER}", chunk_file_id or "<upload failed>"), latex_output if success else error_msg, success, error_msg if not success else None)
    return latex_output, success, error_msg

async def get_latex_from_chunk_claude( # MODIFIED to async
    chunk_pdf_path: Path,
    chunk_num: int,
    total_chunks: int,
    start_page: int,
    end_page: int,
    start_desc: str,
    end_desc: str,
    model_info: Dict[str, Any],
    valid_bib_keys: List[str] # ADDED
) -> Tuple[Optional[str], bool, Optional[str]]:
    # ... (Inner logic remains the same, but uses await for anthropic calls) ...
    global anthropic_client # Use initialized client
    if not anthropic_client:
        return None, False, "Anthropic client not available."

    model_name = model_info['name']
    nickname = model_info['nickname']
    page_range_str = f"Pages {start_page}-{end_page}"
    desc_range_str = f'("{start_desc}" to "{end_desc}")'

    logger.info(f"Claude: Requesting LaTeX for Chunk {chunk_num} {desc_range_str} from {nickname}...using chunk file {chunk_pdf_path.name}...")

    # --- Read chunk PDF and base64 encode (use to_thread) ---
    pdf_base64 = None
    media_type = "application/pdf"
    try:
        pdf_bytes = await asyncio.to_thread(chunk_pdf_path.read_bytes)
        pdf_base64_bytes = await asyncio.to_thread(base64.b64encode, pdf_bytes)
        pdf_base64 = pdf_base64_bytes.decode('utf-8')
        logger.debug(f"Claude (Chunk {chunk_num}): PDF encoded ({len(pdf_bytes) / 1024:.1f} KB).")
        if len(pdf_base64) > 15 * 1024 * 1024: # Increased limit slightly, check Claude docs
             logger.warning(f"Claude (Chunk {chunk_num}): Base64 encoded PDF is large ({len(pdf_base64)/1024**2:.1f} MB). May hit limits.")
    except Exception as e:
        logger.error(f"Claude (Chunk {chunk_num}): Failed to read/encode PDF chunk: {e}")
        return None, False, f"Failed to read/encode PDF chunk: {e}"

    # --- Prompt Engineering --- (Add citation instructions)
    bib_key_info = "No valid citation keys were extracted."
    if valid_bib_keys:
         limited_keys_str = ", ".join(valid_bib_keys[:20]) + ("..." if len(valid_bib_keys) > 20 else "")
         bib_key_info = f"Valid BibTeX keys: [{limited_keys_str}]. Use \\cite{{key}} ONLY for these keys."

    user_prompt_text = f"""# Task: LaTeX Translation (Chunk {chunk_num}/{total_chunks})
Analyze the **entirety** of the provided PDF document chunk (content included in this message).
This chunk represents the content from a larger document starting around "{start_desc}" and ending around "{end_desc}".
Generate a **complete and compilable English LaTeX document** corresponding *only* to the content starting from "{start_desc}" (inclusive, likely near the start of this PDF chunk) and ending at "{end_desc}" (inclusive, likely near the end of this PDF chunk).

# Core Requirements:
- **Include Preamble:** Your output MUST start with `\\documentclass` and include a suitable preamble (common packages, theorems).
- **Include Metadata:** Include `\\title`, `\\author`, `\\date` based on content.
- **Include Body Wrappers:** Include `\\begin{{document}}`, `\\maketitle` (if applicable), and `\\end{{document}}` at the very end. # MODIFIED: Used literal string
- **Translate:** Non-English text to English.
- **Math Accuracy:** Preserve math accurately.
- **Structure:** Maintain semantic structure.
- **Figures/Tables:** Use placeholders like `placeholder_chunk{chunk_num}_imgX.png`.
- **Citations:** {bib_key_info} If citing a key NOT in this list, use a textual placeholder like `[Author, Year]` instead of \\cite.
- **Escaping:** Escape special LaTeX characters in text.
- **Focus:** Generate LaTeX **only** for content from "{start_desc}" to "{end_desc}". Exclude any bibliography section (e.g., \\begin{{thebibliography}}).

# Output Format:
Provide the complete LaTeX document directly, without any surrounding markdown like ```latex ... ```."""

    # --- API Call Implementation ---
    start_time = time.time()
    latex_output = None
    success = False
    error_msg = None

    try:
        # 1. Construct messages payload
        # Claude vision models expect image type for PDFs
        content_blocks = [
             {
                 "type": "image",
                 "source": {"type": "base64", "media_type": media_type, "data": pdf_base64}
             },
             {
                 "type": "text",
                 "text": user_prompt_text
             }
         ]
        messages = [{"role": "user", "content": content_blocks}]

        # 2. Make the API call (use await messages.create)
        logger.debug(f"Claude (Chunk {chunk_num}): Sending prompt to {nickname}...")
        response = await asyncio.to_thread(
            anthropic_client.messages.create,
             model=model_name,
             max_tokens=4096,
             messages=messages,
             system="You are an expert LaTeX translator. Generate complete, compilable LaTeX documents based on the provided PDF content and instructions. Output only the raw LaTeX code.",
             temperature=0.3
         ) # KEEP await/to_thread

        # 3. Process the response
        if response.content and isinstance(response.content, list) and response.content[0].type == "text":
            latex_output = response.content[0].text.strip()
            if latex_output:
                logger.info(f"Claude (Chunk {chunk_num}): Received {len(latex_output)} chars of LaTeX.")
                success = True
            else:
                error_msg = "Claude returned empty text content."
                logger.warning(f"Claude (Chunk {chunk_num}): {error_msg}")
                success = False
        elif response.stop_reason == 'error':
            api_error_details = getattr(response, 'error', None)
            error_msg = f"Claude API error. Stop Reason: {response.stop_reason}. Details: {api_error_details}"
            logger.error(f"Claude (Chunk {chunk_num}): {error_msg}")
            success = False
        else:
            error_msg = f"Claude returned unexpected content structure or stop reason: {response.stop_reason}"
            logger.warning(f"Claude (Chunk {chunk_num}): {error_msg}")
            success = False

    except Exception as e:
        error_msg = f"Error during Claude API call for chunk {chunk_num}: {e}"
        logger.error(error_msg, exc_info=True)
        success = False
        latex_output = None

    elapsed = time.time() - start_time
    logger.info(f"Claude (Chunk {chunk_num}): API call finished in {elapsed:.2f}s. Success: {success}")
    log_prompt_and_response(chunk_num, page_range_str, "claude", f"{nickname} ({model_name})", user_prompt_text, latex_output if success else error_msg, success, error_msg if not success else None)
    return latex_output, success, error_msg

# ... (compile_latex, create_chunk_failure_placeholder, save_failed_chunk_data remain synchronous) ...
# ... (read_state is already updated) ...

def compile_latex(
    tex_filepath: Path,
    aux_dir: Path,
    main_bib_file_path: Optional[Path] = None, # ADDED argument
    num_runs: int = 2,
    timeout: int = 120
) -> Tuple[bool, str, Optional[Path]]:
    """Compile a LaTeX file using pdflatex, including bibtex run if main_bib_file_path exists. Returns success, log, and path to PDF if successful."""
    tex_filepath = Path(tex_filepath)
    if not tex_filepath.is_file():
        return False, f"Error: LaTeX source file not found: {tex_filepath}", None

    aux_dir = Path(aux_dir)
    aux_dir.mkdir(parents=True, exist_ok=True)
    base_name = tex_filepath.stem
    log_file = aux_dir / f"{base_name}.log"
    pdf_file = aux_dir / f"{base_name}.pdf"
    aux_file = aux_dir / f"{base_name}.aux" # Path to aux file

    # Use the provided main_bib_file_path directly to check existence
    bib_file_exists = main_bib_file_path and main_bib_file_path.exists()

    # Clean previous auxiliary files for this specific compilation run
    logger.debug(f"Cleaning aux files for {base_name} in {aux_dir}")
    for pattern in [f"{base_name}.aux", f"{base_name}.log", f"{base_name}.toc", f"{base_name}.out", f"{base_name}.synctex.gz", f"{base_name}.pdf", f"{base_name}.fls", f"{base_name}.fdb_latexmk", f"{base_name}.bbl", f"{base_name}.blg"]: # Added bbl/blg
        try:
            p = aux_dir / pattern
            if p.exists(): p.unlink()
        except OSError as e: logger.warning(f"Could not delete potential aux file {p}: {e}")

    full_log = f"--- Compiling: {tex_filepath.name} in {aux_dir.name} ---\n"
    success = False
    final_pdf_path = None
    bibtex_run_occurred = False # Renamed from bibtex_run_needed for clarity

    # Determine required runs (need 1 initial, 1 bibtex if needed, 1-2 final)
    max_passes = num_runs + (1 if bib_file_exists else 0)

    for i in range(max_passes + 1): # Iterate enough times
        pass_num = i + 1
        is_bibtex_run = False # Flag to check if this pass is bibtex
        run_cwd = None # Default CWD

        # --- Decide action for this pass ---
        if pass_num == 1:
             command = [ "pdflatex", "-interaction=nonstopmode", "-file-line-error", "-halt-on-error", f"-output-directory={str(aux_dir.resolve())}", str(tex_filepath.resolve()) ]
             current_pass_desc = f"Initial pdflatex (Pass {pass_num})"
        elif pass_num == 2 and aux_file.exists() and bib_file_exists and not bibtex_run_occurred:
             command = [ "bibtex", str(aux_file.resolve().stem) ]
             run_cwd = aux_dir.resolve()
             is_bibtex_run = True
             bibtex_run_occurred = True
             current_pass_desc = f"BibTeX (Pass {pass_num})"
        elif pass_num >= 2:
             command = [ "pdflatex", "-interaction=nonstopmode", "-file-line-error", "-halt-on-error", f"-output-directory={str(aux_dir.resolve())}", str(tex_filepath.resolve()) ]
             current_pass_desc = f"Post-BibTeX pdflatex (Pass {pass_num})"
             if pass_num > max_passes:
                 full_log += f"\n--- Completed {max_passes} compilation passes. Stopping. ---"
                 break
        else:
             full_log += f"\n--- Skipping unexpected pass logic state {pass_num} ---"
             continue

        full_log += f"\n--- {current_pass_desc} ---\nRun: {' '.join(command)}\nCWD: {run_cwd or '.'}\n"
        start_time = time.time()
        process = None
        try:
            process = subprocess.run(
                 command, capture_output=True, text=True, encoding='utf-8', errors='replace',
                 timeout=timeout, check=False, cwd=run_cwd
            )
            elapsed = time.time() - start_time
            full_log += f"Pass {pass_num} completed in {elapsed:.2f}s. Return Code: {process.returncode}\n"
            stdout_snippet = process.stdout[-1000:].strip() if process.stdout else ""
            stderr_snippet = process.stderr.strip() if process.stderr else ""
            if stdout_snippet: full_log += f"Stdout (last 1k):\n{stdout_snippet}\n...\n"
            if stderr_snippet: full_log += f"Stderr:\n{stderr_snippet}\n"

            log_content = ""
            errors_found_in_log = []
            if log_file.exists() and not is_bibtex_run:
                try:
                    log_content = log_file.read_text(encoding='utf-8', errors='replace')
                    fatal_errors = re.findall(r"^! ", log_content, re.MULTILINE)
                    undefined_errors = re.findall(r"Undefined control sequence", log_content)
                    missing_file_errors = re.findall(r"LaTeX Error: File `.*?' not found", log_content)
                    errors_found_in_log = fatal_errors + undefined_errors + missing_file_errors
                except Exception as log_read_e:
                    full_log += f"\n--- Error reading log file {log_file}: {log_read_e} ---"
            elif process.returncode != 0 and not is_bibtex_run:
                 full_log += f"\n--- Log file not found after {current_pass_desc} exited with code {process.returncode}. Assuming failure. ---"
                 success = False
                 break

            if errors_found_in_log:
                full_log += f"\n--- LaTeX Error(s) detected in log ({current_pass_desc}): {errors_found_in_log[:5]}... ---"
                success = False
                break
            elif process.returncode != 0:
                if is_bibtex_run:
                     blg_file = aux_dir / f"{base_name}.blg"
                     blg_content = ""
                     if blg_file.exists():
                          try: blg_content = blg_file.read_text(encoding='utf-8', errors='replace')[-1000:]
                          except Exception as e_blg: blg_content = f"(Error reading .blg: {e_blg})"
                     full_log += f"\n--- Warning: BibTeX exited with code {process.returncode}. Log:\n{stdout_snippet}\n{stderr_snippet}\n{blg_content}\n---"
                else:
                    if pass_num >= max_passes:
                        full_log += f"\n--- {current_pass_desc} exited with code {process.returncode} on final pass. Failure. ---"
                        success = False
                        break
                    else:
                        full_log += f"\n--- {current_pass_desc} exit code {process.returncode} on pass {pass_num}. Continuing. ---"
            elif not is_bibtex_run:
                 full_log += f"\n--- {current_pass_desc} completed successfully (Exit code 0). ---"
                 if pass_num >= max_passes:
                      if pdf_file.is_file() and pdf_file.stat().st_size > 100:
                           success = True
                           final_pdf_path = pdf_file
                           full_log += f"\n--- PDF file {pdf_file.name} generated successfully. ---"
                      else:
                           full_log += f"\n--- Compilation finished, but PDF file {pdf_file.name} is missing or empty. ---"
                           success = False
                      break

        except FileNotFoundError:
            full_log += f"\nError: '{command[0]}' command not found. Is LaTeX installed and in PATH?\n"
            success = False
            break
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            full_log += f"\nError: {command[0]} timed out after {timeout}s ({current_pass_desc}, Runtime {elapsed:.1f}s).\n"
            success = False
            break
        except Exception as e:
            elapsed = time.time() - start_time
            full_log += f"\nError running {command[0]} ({current_pass_desc}, Runtime {elapsed:.1f}s): {e}\n"
            success = False
            break

        if not success and not is_bibtex_run and (errors_found_in_log or (process and process.returncode != 0 and pass_num >= max_passes)):
             full_log += f"\n--- Stopping compilation due to failure in pass {pass_num} ({current_pass_desc}) ---"
             break

    if success and final_pdf_path:
        logger.info(f"Compilation successful: {tex_filepath.name} -> {final_pdf_path.name}")
    else:
        logger.error(f"Compilation FAILED: {tex_filepath.name}")
        fail_log_path = aux_dir / f"{base_name}_compile_fail.log"
        try:
            fail_log_path.write_text(full_log, encoding='utf-8')
            logger.error(f"Detailed compilation log saved to: {fail_log_path}")
        except Exception as e_write:
            logger.error(f"Could not write detailed failure log to {fail_log_path}: {e_write}")

    return success, full_log, final_pdf_path

def create_chunk_failure_placeholder(chunk_num: int, start_page: int, end_page: int, reason: str) -> str:
    """Generates a LaTeX placeholder for a failed chunk."""
    reason_escaped = reason.replace('_', '\\_').replace('%', '\\%').replace('&', '\\&')
    placeholder = f"""
% ========== CHUNK {chunk_num} (Pages {start_page}-{end_page}): FAILED ==========
\\clearpage
\\begin{{center}} \\vspace*{{1cm}}
\\fbox{{ \\begin{{minipage}}{{0.85\\textwidth}} \\centering
\\Large\\textbf{{Chunk {chunk_num} (Pages {start_page}-{end_page}): Translation/Compilation Failed}}\\\\[1em]
\\large\\textcolor{{red}}{{Reason: {reason_escaped}}}\\\\[1em]
(Original content omitted)
\\end{{minipage}} }}
\\vspace*{{1cm}} \\end{{center}}
\\clearpage
% ========== END OF CHUNK {chunk_num} PLACEHOLDER ==========
"""
    return placeholder


# --- Custom Exceptions ---
class ChunkProcessingError(Exception):
    """Custom exception for errors during chunk processing."""
    pass

async def main(input_pdf_cli=None, requested_provider=None, requested_model_key=None):
    global TARGET_PROVIDER, TARGET_MODEL_INFO, fallback_preamble, gemini_cache, gemini_cache_name, state # Make state global? Or pass around? Let's pass for now.

    start_time_main = time.time()
    logger.info("="*60)
    logger.info(f"--- Starting PDF to LaTeX Conversion (Async) ---")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("="*60)

    # --- Setup ---
    # Ensure clients are set up first (still sync for now)
    if not setup_clients(): return
    # Setup target provider (needs to be sync or awaited)
    if not await setup_target_provider_and_model(requested_provider, requested_model_key): return

    INPUT_PDF_PATH_ACTUAL = Path(input_pdf_cli if input_pdf_cli else INPUT_PDF_PATH).resolve()
    logger.info(f"Input PDF: {INPUT_PDF_PATH_ACTUAL}")
    logger.info(f"Working Directory: {WORKING_DIR.resolve()}")
    logger.info(f"Provider: {TARGET_PROVIDER}, Model: {TARGET_MODEL_INFO['nickname']}")

    if not INPUT_PDF_PATH_ACTUAL.is_file():
        logger.critical(f"CRITICAL ERROR: Input PDF not found at {INPUT_PDF_PATH_ACTUAL}")
        return

    total_pages = get_total_pages(INPUT_PDF_PATH_ACTUAL)
    if total_pages <= 0:
        logger.critical(f"CRITICAL ERROR: Could not get page count or PDF is invalid/empty: {INPUT_PDF_PATH_ACTUAL}")
        return
    logger.info(f"PDF '{INPUT_PDF_PATH_ACTUAL.name}' has {total_pages} pages.")

    # Define final output paths early
    final_output_filename_base = f"{INPUT_PDF_PATH_ACTUAL.stem}_translated"
    FINAL_TEX_FILE = WORKING_DIR / f"{final_output_filename_base}.tex"
    FINAL_PDF_FILE = WORKING_DIR / f"{final_output_filename_base}.pdf"
    STATE_FILE = WORKING_DIR / "state.json" # Ensure STATE_FILE is defined before read_state

    # --- State and Chunk Management ---
    state = read_state(STATE_FILE) # Correct call
    successful_preambles = state.get("successful_preambles", {})
    successful_bodies = state.get("successful_bodies", {})
    last_completed_chunk_index = state.get("last_completed_chunk_index", 0)
    # ... other setup ...

    # --- RE-INSERTED: Step 1: Upload Full PDF & Handle State --- 
    original_pdf_upload_ref = state.get("original_pdf_upload_ref", None)
    provider_in_state = state.get("provider")
    gemini_cache = None
    gemini_cache_name = None

    # Ensure provider consistency or reset
    if provider_in_state and provider_in_state != TARGET_PROVIDER:
        logger.warning(f"Provider changed from '{provider_in_state}' to '{TARGET_PROVIDER}'. Resetting state and potentially orphaned file ref.")
        original_pdf_upload_ref = None # Clear ref as it belongs to old provider
        last_completed_chunk_index = 0
        successful_preambles = {}
        successful_bodies = {}
        state = {"last_completed_chunk_index": 0, "provider": TARGET_PROVIDER, "original_pdf_upload_ref": None, "successful_preambles": {}, "successful_bodies": {}}
        save_state(last_completed_chunk_index, successful_preambles, successful_bodies, STATE_FILE, TARGET_PROVIDER, original_pdf_upload_ref) # Save reset state
    else:
        # Re-establish Gemini file object from state if resuming with Gemini
        if TARGET_PROVIDER == "gemini" and isinstance(original_pdf_upload_ref, str):
             try:
                 logger.info(f"Gemini: Attempting to retrieve file object for {original_pdf_upload_ref}...")
                 # Use sync get_file, no await needed as main is async but get_file is sync
                 original_pdf_upload_ref = genai.get_file(name=original_pdf_upload_ref)
                 logger.info(f"Gemini: Retrieved file object: {original_pdf_upload_ref.name}")
             except Exception as e_get:
                 logger.error(f"Gemini: Could not retrieve file object '{original_pdf_upload_ref}' from state. Will re-upload. Error: {e_get}")
                 original_pdf_upload_ref = None # Reset if retrieval fails

    # Upload if needed
    upload_needed = False
    if TARGET_PROVIDER == "gemini" and not isinstance(original_pdf_upload_ref, genai.types.File):
        upload_needed = True
    elif TARGET_PROVIDER == "openai" and not isinstance(original_pdf_upload_ref, str):
        upload_needed = True
    elif TARGET_PROVIDER == "claude":
        # Claude never stores a persistent ref
        if original_pdf_upload_ref: original_pdf_upload_ref = None # Clear any old ref

    if upload_needed and TARGET_PROVIDER != "claude":
        logger.info(f"Uploading full PDF for {TARGET_PROVIDER}...")
        # Call sync upload function directly, no await
        uploaded_ref = upload_pdf_to_provider(INPUT_PDF_PATH_ACTUAL, TARGET_PROVIDER)
        if not uploaded_ref:
            logger.critical(f"Failed to upload PDF via {TARGET_PROVIDER}. Cannot proceed.")
            return
        original_pdf_upload_ref = uploaded_ref
        # Update state immediately after successful upload
        state["provider"] = TARGET_PROVIDER
        ref_to_store = None
        if isinstance(original_pdf_upload_ref, genai.types.File): ref_to_store = original_pdf_upload_ref.name
        elif isinstance(original_pdf_upload_ref, str): ref_to_store = original_pdf_upload_ref
        state["original_pdf_upload_ref"] = ref_to_store
        # CORRECTED CALL: Pass ref_to_store (string) instead of original_pdf_upload_ref (object)
        save_state(last_completed_chunk_index, successful_preambles, successful_bodies, STATE_FILE, TARGET_PROVIDER, ref_to_store)
    elif TARGET_PROVIDER == "claude" and state.get("provider") != "claude":
         # Ensure state reflects Claude provider if starting fresh or switching
         state["provider"] = TARGET_PROVIDER
         state["original_pdf_upload_ref"] = None # Claude has no ref to store
         # CORRECTED CALL: Pass None for ref_to_store
         save_state(last_completed_chunk_index, successful_preambles, successful_bodies, STATE_FILE, TARGET_PROVIDER, None)

    # --- RE-INSERTED: Step 2: Determine Chunk Definitions --- 
    chunk_definition_file = WORKING_DIR / "chunk-division.txt"
    chunk_definitions = None
    if chunk_definition_file.exists() and last_completed_chunk_index > 0:
        logger.info(f"Resuming: Loading chunk definitions from {chunk_definition_file}")
        try:
            with open(chunk_definition_file, 'r', encoding='utf-8') as f: chunk_definitions = json.load(f)
            if not isinstance(chunk_definitions, list) or not all(isinstance(cd, dict) for cd in chunk_definitions): raise ValueError("Invalid format")
        except Exception as e:
            logger.error(f"Failed to load/validate chunk definitions: {e}. Redetermining chunks.")
            chunk_definitions = None

    if not chunk_definitions:
        logger.info("Attempting AI-driven chunk definition...")
        ai_chunk_definitions_raw = get_chunk_definitions_from_ai(INPUT_PDF_PATH_ACTUAL, TARGET_PROVIDER, TARGET_MODEL_INFO)
        parsed_ai_data = None
        if ai_chunk_definitions_raw:
            try:
                json_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", ai_chunk_definitions_raw, re.DOTALL)
                if json_match: json_str = json_match.group(1)
                else: json_str = ai_chunk_definitions_raw
                parsed_ai_data = json.loads(json_str)
            except json.JSONDecodeError as json_e:
                logger.error(f"AI returned invalid JSON for chunk definitions: {json_e}")
                logger.debug(f"Raw AI response for chunk def:\n{ai_chunk_definitions_raw}")

        chunk_definitions = determine_and_save_chunk_definitions(
            INPUT_PDF_PATH_ACTUAL, total_pages, TARGET_PAGES_PER_CHUNK,
            TARGET_PROVIDER, TARGET_MODEL_INFO, chunk_definition_file,
            ai_chunk_data=parsed_ai_data
        )
        if not chunk_definitions:
            logger.critical("Failed to determine chunk definitions via AI or fallback. Cannot proceed.")
            return
        # Reset progress if chunks were redetermined
        last_completed_chunk_index = 0
        successful_preambles = {}
        successful_bodies = {}
        state["last_completed_chunk_index"] = 0
        state["successful_preambles"] = {}
        state["successful_bodies"] = {}
        # CORRECTED CALL: Pass ref_to_store (string)
        save_state(last_completed_chunk_index, successful_preambles, successful_bodies, STATE_FILE, TARGET_PROVIDER, original_pdf_upload_ref)

    if not chunk_definitions: # Final check
        logger.critical("CRITICAL ERROR: No chunk definitions available. Cannot proceed.")
        return

    total_chunks = len(chunk_definitions)
    logger.info(f"Using {total_chunks} chunk definitions.")

    # --- RE-INSERTED: Step 2.5: Extract Bibliography --- 
    bib_file_path = WORKING_DIR / f"{INPUT_PDF_PATH_ACTUAL.stem}.bib"
    bib_content = None
    valid_bib_keys: List[str] = []

    if not bib_file_path.exists() or last_completed_chunk_index == 0:
        logger.info(f"--- Extracting Bibliography --- ")
        # ADD await back to call the now async extract_bibliography_with_ai
        bib_content_extracted = await extract_bibliography_with_ai(INPUT_PDF_PATH_ACTUAL, TARGET_PROVIDER, TARGET_MODEL_INFO)
        if bib_content_extracted:
            try:
                if "@" in bib_content_extracted[:50]:
                    bib_file_path.write_text(bib_content_extracted, encoding='utf-8')
                    logger.info(f"Bibliography saved to {bib_file_path}")
                    bib_content = bib_content_extracted
                else:
                     logger.warning("AI response for bibliography did not look like BibTeX. Discarding.")
                     bib_content = None
            except Exception as e:
                logger.error(f"Failed to write bibliography file {bib_file_path}: {e}")
                bib_content = None
        else:
            logger.warning("AI failed to extract bibliography. Citations may not work.")
            if bib_file_path.exists():
                 try: bib_file_path.unlink()
                 except OSError: pass
    else:
         logger.info(f"Skipping BibTeX extraction, file already exists: {bib_file_path}")
         try:
             bib_content = bib_file_path.read_text(encoding='utf-8')
         except Exception as e:
             logger.error(f"Failed to read existing bibliography file {bib_file_path}: {e}")
             bib_content = None

    if bib_content:
        valid_bib_keys = extract_bib_keys(bib_content)
    logger.info(f"Proceeding to chunk processing with {len(valid_bib_keys)} BibTeX keys known.")

    # --- Step 3: Process Chunks (Async) ---
    if last_completed_chunk_index >= total_chunks:
        logger.info("All chunks already processed based on loaded state.")
    else:
        logger.info(f"Starting/Resuming chunk processing from chunk {last_completed_chunk_index + 1}...")

        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(5)

        # Define the asynchronous function to process a single chunk
        async def process_chunk_async(
            chunk_idx: int,
            chunk_defs: List[Dict[str, Any]],
            total_chunks_arg: int
        ) -> Tuple[int, Optional[str], Optional[str]]: # CHANGED: Remove bool return
            # REMOVED: nonlocal all_chunk_tasks_successful

            # Get chunk details from the main list
            try:
                # Use passed chunk_defs
                chunk_info = chunk_defs[chunk_idx - 1]
                if chunk_info['chunk_num'] != chunk_idx:
                    err_msg = f"Mismatch between chunk index {chunk_idx} and definition number {chunk_info['chunk_num']}."
                    logger.error(err_msg + " Aborting chunk.")
                    raise ChunkProcessingError(err_msg)
                start_page = chunk_info['start_page']
                end_page = chunk_info['end_page']
                start_desc = chunk_info['start_desc']
                end_desc = chunk_info['end_desc']
            except IndexError:
                err_msg = f"Could not find definition for chunk index {chunk_idx}."
                logger.error(err_msg + " Aborting chunk.")
                raise ChunkProcessingError(err_msg)
            except KeyError as e:
                 err_msg = f"Missing key {e} in definition for chunk index {chunk_idx}."
                 logger.error(err_msg + " Aborting chunk.")
                 raise ChunkProcessingError(err_msg)

            chunk_pdf_path = None
            # Use passed total_chunks_arg
            logger.info(f"--- Starting processing for Chunk {chunk_idx}/{total_chunks_arg} (Pages {start_page}-{end_page}) ---")

            try:
                # Create chunk PDF
                chunk_pdf_path = await asyncio.to_thread(
                    create_pdf_chunk_file,
                    INPUT_PDF_PATH_ACTUAL,
                    start_page, end_page,
                    SAVED_CHUNKS_DIR
                )
                if not chunk_pdf_path:
                    err_msg = f"Chunk {chunk_idx}: Failed to create PDF chunk file. Skipping."
                    logger.error(err_msg)
                    raise ChunkProcessingError(err_msg)
                logger.info(f"Chunk {chunk_idx}: Created chunk PDF: {chunk_pdf_path.name}")

                raw_output = None
                max_retries = 3
                base_delay = 2
                api_success = False
                api_error_msg = "No response received"

                # Acquire semaphore before making API call
                async with semaphore:
                    for attempt in range(max_retries):
                        logger.debug(f"Chunk {chunk_idx}: Attempt {attempt + 1}/{max_retries} for API call...")
                        common_args = {
                            "chunk_pdf_path": chunk_pdf_path, "chunk_num": chunk_idx,
                            "total_chunks": total_chunks_arg, "start_page": start_page,
                            "end_page": end_page, "start_desc": start_desc,
                            "end_desc": end_desc, "model_info": TARGET_MODEL_INFO,
                            "valid_bib_keys": valid_bib_keys
                        }
                        try:
                            if TARGET_PROVIDER == "gemini":
                                # Pass gemini_cache_name if needed by the function
                                raw_output, api_success, api_error_msg = await get_latex_from_chunk_gemini(
                                    **common_args #, gemini_cache_name=gemini_cache_name # Uncomment if cache used
                                )
                            elif TARGET_PROVIDER == "openai":
                                raw_output, api_success, api_error_msg = await get_latex_from_chunk_openai(**common_args)
                            elif TARGET_PROVIDER == "claude": # Ensure this matches the function name
                                raw_output, api_success, api_error_msg = await get_latex_from_chunk_claude(**common_args)
                            else:
                                err_msg = f"Invalid TARGET_PROVIDER '{TARGET_PROVIDER}' during chunk processing."
                                logger.error(err_msg)
                                raise ChunkProcessingError(err_msg)

                            if api_success:
                                break # Exit retry loop if successful
                            else:
                                logger.warning(f"Chunk {chunk_idx}: API Attempt {attempt + 1}/{max_retries} failed. Reason: {api_error_msg}")

                        except ChunkProcessingError: # Re-raise specific errors
                            raise
                        except Exception as e:
                            logger.warning(f"Chunk {chunk_idx}: Attempt {attempt + 1}/{max_retries} failed with exception: {e}", exc_info=False) # Less verbose logging for retries
                            api_error_msg = str(e)
                            api_success = False # Ensure failure

                        if not api_success and attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            logger.info(f"Chunk {chunk_idx}: Waiting {delay:.1f}s before retry...")
                            await asyncio.sleep(delay)

                if not api_success:
                    err_msg = f"Chunk {chunk_idx}: API calls failed after {max_retries} attempts. Final error: {api_error_msg}"
                    logger.error(err_msg)
                    raise ChunkProcessingError(err_msg)

                logger.info(f"Chunk {chunk_idx}: Received successful response from {TARGET_PROVIDER}. Parsing...")
                extracted_preamble, extracted_body = parse_llm_latex_output(raw_output, chunk_idx, total_chunks_arg)

                if extracted_body:
                    logger.info(f"Chunk {chunk_idx}: Successfully parsed LaTeX content.")
                    return chunk_idx, extracted_preamble, extracted_body # CHANGED return
                else:
                    err_msg = f"Chunk {chunk_idx}: Failed to parse LaTeX content from response (body was null/empty after parsing)."
                    logger.warning(err_msg)
                    # REMOVED save_failed_chunk_data call
                    raise ChunkProcessingError(err_msg)

            except ChunkProcessingError: # Re-raise to stop processing
                raise
            except Exception as e:
                err_msg = f"Chunk {chunk_idx}: Unexpected error during processing: {e}"
                logger.exception(err_msg)
                raise ChunkProcessingError(err_msg) # Raise custom error
            finally:
                # ... (finally block remains same) ...
                # if chunk_pdf_path and chunk_pdf_path.exists():
                # try:
                        # await asyncio.to_thread(chunk_pdf_path.unlink)
                        # logger.debug(f"Chunk {chunk_idx}: Deleted temporary file {chunk_pdf_path.name}")
                    # except Exception as e_unlink:
                         # logger.warning(f"Chunk {chunk_idx}: Could not delete temp file {chunk_pdf_path.name}. Error: {e_unlink}")
                pass

        # --- End of process_chunk_async ---

        # Create tasks for chunks that need processing
        tasks = []
        indices_to_process = []
        for i in range(last_completed_chunk_index + 1, total_chunks + 1):
             if i not in successful_bodies:
                 indices_to_process.append(i)
                 # MODIFIED: Pass chunk_definitions and total_chunks to the task
                 tasks.append(asyncio.create_task(process_chunk_async(i, chunk_definitions, total_chunks)))
             else:
                 logger.info(f"Skipping task creation for chunk {i}, already present in loaded state.")

        # Run tasks concurrently and gather results
        if tasks:
            logger.info(f"Processing chunks {indices_to_process} concurrently ({len(tasks)} tasks)...")
            try:
                results = await asyncio.gather(*tasks)
            except ChunkProcessingError as e_gather:
                 logger.critical(f"CRITICAL: Chunk processing failed: {e_gather}. Aborting further processing.")
                 # Clean up full PDF upload before exiting if possible
                 if original_pdf_upload_ref and state.get("provider"):
                     logger.info("Attempting cleanup of uploaded PDF before exiting due to chunk error...")
                     ref_to_del = state.get("original_pdf_upload_ref")
                     delete_provider_file(ref_to_del, state["provider"])
                 return # Exit main

            # Process results - updates IN-MEMORY dictionaries (only if gather succeeded)
            for chunk_num, preamble, body in results:
                # No need to check success flag, gather would have raised exception
                successful_preambles[chunk_num] = preamble
                successful_bodies[chunk_num] = body
                if chunk_num > last_completed_chunk_index:
                     last_completed_chunk_index = chunk_num
                # REMOVED placeholder logic

            # Save state AFTER processing ALL results from the gather call
            save_state(last_completed_chunk_index,
                       successful_preambles,
                       successful_bodies,
                       STATE_FILE,
                       TARGET_PROVIDER,
                       state.get("original_pdf_upload_ref"))
            logger.info(f"State saved after processing batch. Last completed chunk: {last_completed_chunk_index}")
        else:
            logger.info("No new chunk tasks were created (already processed or loaded from state).")

    # --- Post Chunk Processing Checks ---

    # --- Final LaTeX Assembly ---
    # (Preamble synthesis logic remains synchronous for now)
    final_preamble = None
    if last_completed_chunk_index == total_chunks: # Only synth if all chunks attempted
        logger.info("All chunks processed. Attempting final preamble synthesis...")
        valid_preambles_for_synthesis = [p for p in successful_preambles.values() if p and not p.strip().startswith("% Preamble for")] # Filter None/empty too

        if not valid_preambles_for_synthesis:
             logger.warning("No successful preambles extracted. Using basic fallback preamble.")
             final_preamble = fallback_preamble
        else:
            logger.info(f"Attempting to synthesize final preamble using AI from {len(valid_preambles_for_synthesis)} successful chunks...")
            # Assuming generate_final_preamble_with_ai is sync for now
            final_preamble = generate_final_preamble_with_ai(
                valid_preambles_for_synthesis,
                TARGET_PROVIDER, TARGET_MODEL_INFO,
                INPUT_PDF_PATH_ACTUAL if TARGET_PROVIDER == "gemini" else None
            )
            if not final_preamble:
                 logger.warning("AI preamble synthesis failed. Using fallback concatenation.")
                 # Fallback concatenation logic...
                 all_preamble_lines = set()
                 doc_class_line = ""
                 for p in valid_preambles_for_synthesis:
                     for line in p.splitlines():
                          stripped_line = line.strip()
                          if stripped_line.startswith("\\documentclass"):
                              if not doc_class_line: doc_class_line = stripped_line
                          elif stripped_line: all_preamble_lines.add(stripped_line)
                 if not doc_class_line: doc_class_line = fallback_preamble.splitlines()[0]
                 other_lines = sorted(list(all_preamble_lines))
                 final_preamble = doc_class_line + "\n" + "\n".join(other_lines)
                 final_preamble = "% Preamble created using fallback concatenation\n" + final_preamble
            else:
                 logger.info("AI successfully synthesized the final preamble.")
                 final_preamble = final_preamble.strip()

    # --- Step 5: Final Document Assembly ---
    if final_preamble:
        logger.info("Concatenating final document...")
        bib_commands = ""
        if bib_file_path.exists():
            bib_stem = bib_file_path.stem
            bib_style = "amsplain"
            bib_commands = f"\n\\bibliographystyle{{{bib_style}}}\n\\bibliography{{{bib_stem}}}\n"
            logger.info(f"Adding bibliography commands for {bib_stem}.bib with style {bib_style}")
        else:
            logger.info("No .bib file found, skipping bibliography commands.")

        # Ensure bodies are joined in the correct order
        final_body_content = "\n\n% ========================================\n\n".join(
            successful_bodies[i] for i in range(1, total_chunks + 1)
        )

        final_combined_latex = (
            final_preamble.strip() + "\n\n" +
            "\\begin{document}\n\n" +
            final_body_content + # Use ordered bodies
            bib_commands +
            "\n\n\\end{document}" # Use double backslash
        )

        try:
            FINAL_TEX_FILE.write_text(final_combined_latex, encoding='utf-8')
            logger.info(f"Final concatenated LaTeX saved to {FINAL_TEX_FILE}")

        except Exception as e:
             logger.error(f"Error writing final LaTeX file {FINAL_TEX_FILE.name}: {e}", exc_info=True)
             all_chunk_tasks_successful = False # Mark as overall failure
    else:
        # This case happens if preamble synth wasn't attempted (e.g., chunks didn't finish)
        logger.error("Final document assembly skipped because preamble synthesis did not run (likely due to incomplete chunk processing).")
        # REMOVED: all_chunk_tasks_successful = False # Mark as overall failure

    # --- Final Summary --- 
    end_time_main = time.time()
    total_time_main = end_time_main - start_time_main
    logger.info("="*60)

    # Check for PDF existence now
    final_pdf_exists = FINAL_PDF_FILE.exists() and FINAL_PDF_FILE.stat().st_size > 100

    if final_pdf_exists:
        logger.info(f"--- PDF to LaTeX Conversion Script Finished Successfully --- ")
        logger.info(f"Final PDF generated: {FINAL_PDF_FILE.resolve()}")
        # REMOVED placeholder warning
    else:
        logger.error(f"--- PDF to LaTeX Script Finished With Errors --- ")
        if FINAL_TEX_FILE.exists():
            logger.error(f"Final LaTeX file generated: {FINAL_TEX_FILE.resolve()}")
            logger.error("However, the final PDF compilation failed or was skipped.")
        else:
            logger.error("The final .tex file was not generated.")
        # REMOVED check for incomplete processing, as script should error out earlier

    logger.info(f"Total script execution time: {total_time_main:.2f} seconds ({total_time_main/60:.2f} minutes).")
    logger.info(f"Logs and intermediate files are in: {WORKING_DIR.resolve()}")
    logger.info(f"Prompt/Response Log: {PROMPT_LOG_FILE.resolve()}")
    logger.info("="*60)

    # --- Cleanup ---
    # Clean up the original full PDF upload using state info
    provider_in_state = state.get("provider")
    file_ref_in_state = state.get("original_pdf_upload_ref") # This should be the string name/ID
    if provider_in_state in ["gemini", "openai"] and file_ref_in_state:
        logger.info(f"Cleaning up uploaded file reference ({provider_in_state}: {file_ref_in_state})...")
        # Call sync delete_provider_file directly (no await needed from async main)
        delete_provider_file(file_ref_in_state, provider_in_state)

# --- Command-Line Interface ---
if __name__ == "__main__":
    # ... (Argument parsing remains the same) ...
    parser = argparse.ArgumentParser(
        description='Translate a PDF document to LaTeX using AI (Gemini, OpenAI, or Claude), processing in chunks.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument( 'input_pdf', nargs='?', default=INPUT_PDF_PATH, help=f'Path to the input PDF file.\\n(Default: {INPUT_PDF_PATH})' )
    parser.add_argument( '--provider', '-p', choices=["gemini", "openai", "claude"], default=None, help=f'AI provider to use.\\n(Default: {DEFAULT_PROVIDER})')
    parser.add_argument( '--model', '-m', choices=list(get_model_configs().keys()), default=None, help='Specific model key to use (must match provider).\\n(Default: Provider default)')
    parser.add_argument( '--list-models', '-l', action='store_true', help='List configured model keys and exit.' )
    parser.add_argument( '--working-dir', '-w', default=str(WORKING_DIR), help=f'Directory for intermediate files, logs, and output.\\n(Default: {WORKING_DIR})' )
    parser.add_argument( '--chunk-size', '-c', type=int, default=TARGET_PAGES_PER_CHUNK, help=f'Target number of pages per processing chunk.\\n(Default: {TARGET_PAGES_PER_CHUNK})' )
    parser.add_argument( '--clean', action='store_true', help='Delete the working directory before starting (force fresh run).' )

    args = parser.parse_args()

    if args.list_models:
        print("\nAvailable Model Keys:")
        print("-" * 60)
        for provider in ["gemini", "openai", "claude"]:
            print(f"--- {provider.upper()} ---")
            models = {k: v for k, v in get_model_configs().items() if v['provider'] == provider}
            if models:
                for key, config in sorted(models.items()):
                    print(f"  {key:<25} ({config['nickname']})")
            else:
                print("  (No models configured)")
        print("-" * 60)
        exit(0)

    # Update globals based on CLI args
    TARGET_PAGES_PER_CHUNK = args.chunk_size
    WORKING_DIR = Path(args.working_dir)

    # --- Re-init paths and logging based on potentially changed WORKING_DIR ---
    STATE_FILE = WORKING_DIR / "state.json" # Define before use in main
    # FINAL_TEX_FILE/FINAL_PDF_FILE are now set inside main based on input PDF stem
    LATEX_AUX_DIR = WORKING_DIR / "latex_aux"
    PROMPT_LOG_FILE = WORKING_DIR / "prompts_and_responses.log"
    SAVED_CHUNKS_DIR = WORKING_DIR / "saved_pdf_chunks" # Ensure this is defined globally
    log_file_path = WORKING_DIR / "processing.log"

    if args.clean:
        if WORKING_DIR.exists():
            confirm = input(f"WARNING: Delete working directory '{WORKING_DIR.resolve()}'? (y/N): ")
            if confirm.lower() == 'y':
                logger.warning(f"Deleting working directory: {WORKING_DIR}")
                try:
                    shutil.rmtree(WORKING_DIR)
                except Exception as e:
                    logger.error(f"Failed to delete working directory: {e}")
                    exit(1)
            else:
                print("Clean operation cancelled.")
                exit(0)

    # (Re)create directories
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_AUX_DIR.mkdir(parents=True, exist_ok=True)

    # Add File Handler to logger
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler): handler.close(); logger.removeHandler(handler)
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    logger.info(f"Logging to console and file: {log_file_path.resolve()}")

    # Call main function
    # MODIFIED: Use asyncio.run to execute the async main function
    asyncio.run(main(input_pdf_cli=args.input_pdf, requested_provider=args.provider, requested_model_key=args.model))

