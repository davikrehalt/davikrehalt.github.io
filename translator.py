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
FAILED_CHUNKS_DIR = WORKING_DIR / "failed_chunks" # Store failed chunk LaTeX/compile logs
PROMPT_LOG_FILE = WORKING_DIR / "prompts_and_responses.log"

# LaTeX Tag Constant
END_DOCUMENT_TAG = "\\end{document}"

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
def setup_clients():
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

def setup_target_provider_and_model(requested_provider=None, requested_model_key=None):
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

def upload_pdf_to_provider(pdf_path: Path, provider: str) -> Optional[Union[glm.File, str]]:
    """Uploads the full PDF to the specified provider's file storage."""
    if not pdf_path.exists():
        logger.error(f"PDF file not found for upload: {pdf_path}")
        return None

    logger.info(f"Uploading {pdf_path.name} ({pdf_path.stat().st_size / 1024**2:.2f} MB) via {provider} API...")
    start_time = time.time()

    if provider == "gemini":
        if not gemini_client: return None
        try:
            # Ensure genai is used for upload/get/delete
            uploaded_file = genai.upload_file(path=str(pdf_path), display_name=pdf_path.name)
            while uploaded_file.state.name == "PROCESSING":
                logger.debug(f"Gemini file '{uploaded_file.name}' processing...")
                time.sleep(5)
                uploaded_file = genai.get_file(name=uploaded_file.name)
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
            with open(pdf_path, "rb") as f:
                # Use "user_data" purpose as recommended
                response = openai_client.files.create(file=f, purpose="user_data")
            elapsed = time.time() - start_time
            # Wait briefly for processing? OpenAI API doesn't explicitly require polling like Gemini's.
            # Let's assume it's ready quickly.
            logger.info(f"OpenAI: Upload successful. File ID: {response.id}, Status: {response.status} ({elapsed:.2f}s)")
            return response.id # Return the File ID string
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

def delete_provider_file(file_ref: Union[glm.File, str], provider: str):
    """Deletes an uploaded file from the specified provider."""
    if not file_ref:
        logger.debug(f"{provider}: No file reference provided for deletion.")
        return

    if provider == "gemini":
        if not gemini_client or not isinstance(file_ref, glm.File): return
        file_name = file_ref.name
        logger.info(f"Gemini: Attempting to delete file: {file_name}")
        try:
            genai.delete_file(name=file_name)
            logger.info(f"Gemini: Successfully deleted file: {file_name}")
        except Exception as e:
            logger.warning(f"Gemini: Could not delete file {file_name}: {e}")

    elif provider == "openai":
        if not openai_client or not isinstance(file_ref, str): return
        file_id = file_ref
        logger.info(f"OpenAI: Attempting to delete file: {file_id}")
        try:
            openai_client.files.delete(file_id=file_id)
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
def send_prompt_to_provider(
    prompt: str,
    provider: str,
    model_info: Dict[str, Any],
    client: Any, # The initialized client instance
    system_prompt: Optional[str] = None,
    files: Optional[List[Union[glm.File, Path]]] = None, # List of Gemini files or paths for upload
    temperature: float = 0.3,
    max_retries: int = 2,
    # page_num: Optional[int] = None, # Removed page_num as this is general
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
        temp_files_to_delete = [] # Keep track of files uploaded just for this call

        try:
            if provider == "gemini":
                if not client: raise ValueError("Gemini client (genai module) not provided.")
                # MODIFIED: Instantiate model directly from the genai module (stored in 'client')
                model_instance = client.GenerativeModel(model_name)
                generation_config = genai.types.GenerationConfig(temperature=temperature)
                safety_settings = [{"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_DANGEROUS_CONTENT"]]

                # Process files: Use existing glm.File or upload Paths
                processed_files = []
                if files:
                    for file_item in files:
                        if isinstance(file_item, Path):
                            logger.debug(f"Gemini (send_prompt): Uploading temp file {file_item.name}...")
                            uploaded_file = genai.upload_file(path=str(file_item), display_name=f"temp_{file_item.name}")
                            while uploaded_file.state.name == "PROCESSING": time.sleep(3); uploaded_file = genai.get_file(uploaded_file.name)
                            if uploaded_file.state.name == "FAILED": raise RuntimeError(f"Temporary file upload failed: {file_item.name}")
                            processed_files.append(uploaded_file)
                            temp_files_to_delete.append(uploaded_file) # Mark for deletion
                        elif isinstance(file_item, genai.types.File): # CORRECTED: Check against genai.types.File
                            processed_files.append(file_item) # Use existing file reference
                        else:
                             logger.warning(f"Gemini (send_prompt): Unsupported file type: {type(file_item)}")
                
                content_parts = processed_files + [prompt] # Files first, then prompt

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
            for file_ref in temp_files_to_delete:
                delete_provider_file(file_ref, "gemini")
            
            elapsed = time.time() - start_time
            logger.debug(f"{provider} attempt {attempt} finished in {elapsed:.2f}s. Status: {response_data['status']}")

        # If not successful and retries remain, wait before next attempt
        if response_data["status"] != "success" and attempt < max_retries:
            wait_time = random.uniform(2, 5) * attempt # Exponential backoff (simple)
            logger.info(f"Waiting {wait_time:.1f}s before retry...")
            time.sleep(wait_time)

    return response_data

# --- Provider-Specific LaTeX Generation ---

def get_latex_from_chunk_gemini(
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
        "- **Include Body Wrappers:** Include `\\begin{document}`, `\\maketitle` (if applicable) at the start, and ensure the final output includes `{END_DOCUMENT_TAG}` at the very end.", # REVERTED: Put back the request for END_DOCUMENT_TAG
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
    # ... (Gemini API call logic remains largely the same as previously attempted) ...
    start_time = time.time()
    latex_output = None
    success = False
    error_msg = None
    chunk_file_ref: Optional[glm.File] = None

    try:
        # 1. Upload chunk
        logger.debug(f"Gemini (Chunk {chunk_num}): Uploading...")
        chunk_file_ref = genai.upload_file(path=str(chunk_pdf_path), display_name=f"chunk_{chunk_num}_{chunk_pdf_path.name}")
        while chunk_file_ref.state.name == "PROCESSING": time.sleep(5); chunk_file_ref = genai.get_file(name=chunk_file_ref.name)
        if chunk_file_ref.state.name == "FAILED": raise RuntimeError(f"Gemini file upload failed processing: {chunk_file_ref.name}")
        logger.debug(f"Gemini (Chunk {chunk_num}): Upload OK: {chunk_file_ref.name}")

        # 2. Prep model & config
        # MODIFIED: Instantiate model directly from genai module
        model_instance = genai.GenerativeModel(model_name)
        generation_config=genai.types.GenerationConfig(temperature=0.3)
        safety_settings = [{"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_DANGEROUS_CONTENT"]]

        # 3. Call API
        logger.debug(f"Gemini (Chunk {chunk_num}): Sending prompt...")
        response = model_instance.generate_content(
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
        # 5. Delete uploaded chunk
        if chunk_file_ref:
            delete_provider_file(chunk_file_ref, "gemini")

    elapsed = time.time() - start_time
    logger.info(f"Gemini (Chunk {chunk_num}): API call finished in {elapsed:.2f}s. Success: {success}")
    log_prompt_and_response(chunk_num, page_range_str, "gemini", f"{nickname} ({model_name})", prompt, latex_output if success else error_msg, success, error_msg if not success else None)
    return latex_output, success, error_msg

def get_latex_from_chunk_openai(
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

    user_prompt_text_template = f"""# Task: LaTeX Translation (Chunk {{chunk_num}}/{{total_chunks}})
Analyze the **entirety** of the provided PDF document chunk (uploaded as file ID {{CHUNK_FILE_ID_PLACEHOLDER}}).
This chunk represents the content from a larger document starting around "{{start_desc}}" and ending around "{{end_desc}}".
Generate a **complete and compilable English LaTeX document** corresponding *only* to the content starting from "{{start_desc}}" (inclusive, likely near the start of this PDF chunk) and ending at "{{end_desc}}" (inclusive, likely near the end of this PDF chunk).

# Core Requirements:
- **Include Preamble:** Start with `\\documentclass` and include a suitable preamble (common packages, theorems).
- **Include Metadata:** Include `\\title`, `\\author`, `\\date` based on content.
- **Include Body Wrappers:** Include `\\begin{{document}} `, `\\maketitle` (if applicable), and `{{END_DOCUMENT_TAG}}`. # REVERTED: Put back the request for END_DOCUMENT_TAG
- **Translate:** Non-English text to English.
- **Math Accuracy:** Preserve math accurately.
- **Structure:** Maintain semantic structure.
- **Figures/Tables:** Use placeholders like `placeholder_chunk{{chunk_num}}_imgX.png`.
- **Citations:** {{bib_key_info}} If citing a key NOT in this list, use a textual placeholder like `[Author, Year]` instead of \\cite.
- **Escaping:** Escape special LaTeX characters in text.
- **Focus:** Generate LaTeX only for content from "{{start_desc}}" to "{{end_desc}}".

# Output Format:
Provide the complete LaTeX document directly, without markdown formatting like ```latex ... ```."""

    # --- API Call Implementation ---
    start_time = time.time()
    latex_output = None
    success = False
    error_msg = None
    chunk_file_id: Optional[str] = None # OpenAI uses file ID string

    try:
        # 1. Upload the chunk PDF
        logger.debug(f"OpenAI (Chunk {chunk_num}): Uploading chunk file {chunk_pdf_path.name}...")
        with open(chunk_pdf_path, "rb") as f:
            response_upload = openai_client.files.create(file=f, purpose="user_data")
        chunk_file_id = response_upload.id
        logger.debug(f"OpenAI (Chunk {chunk_num}): Chunk upload successful: {chunk_file_id}")

        # 2. Finalize prompt with file ID
        final_user_prompt = user_prompt_text_template.replace("{CHUNK_FILE_ID_PLACEHOLDER}", chunk_file_id)

        # 3. Construct messages payload
        messages = [
             {"role": "system", "content": "You are an expert LaTeX translator. Generate complete, compilable LaTeX documents based on the provided PDF content and instructions. Output only the raw LaTeX code."}, # System prompt
             {"role": "user", "content": final_user_prompt}
         ]

        # 4. Make the API call
        logger.debug(f"OpenAI (Chunk {chunk_num}): Sending prompt to {nickname}...")
        response = openai_client.chat.completions.create(
             model=model_name,
             messages=messages,
             max_tokens=4096, # Adjust as needed, depends on model
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
        # 6. Delete the uploaded chunk PDF
        if chunk_file_id:
            delete_provider_file(chunk_file_id, "openai")

    elapsed = time.time() - start_time
    logger.info(f"OpenAI (Chunk {chunk_num}): API call finished in {elapsed:.2f}s. Success: {success}")
    # Log using the template string before ID replacement for readability
    log_prompt_and_response(chunk_num, page_range_str, "openai", f"{nickname} ({model_name})", user_prompt_text_template.replace("{CHUNK_FILE_ID_PLACEHOLDER}", chunk_file_id or "<upload failed>"), latex_output if success else error_msg, success, error_msg if not success else None)
    return latex_output, success, error_msg

def get_latex_from_chunk_claude(
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

    # --- Read chunk PDF and base64 encode ---
    pdf_base64 = None
    media_type = "application/pdf"
    try:
        pdf_bytes = chunk_pdf_path.read_bytes()
        pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
        logger.debug(f"Claude (Chunk {chunk_num}): PDF encoded ({len(pdf_bytes) / 1024:.1f} KB).")
        # Simple size check (e.g., 5MB limit for base64 string? Claude limits vary)
        if len(pdf_base64) > 5 * 1024 * 1024:
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
- **Include Body Wrappers:** Include `\\begin{{document}}`, `\\maketitle` (if applicable), and `{END_DOCUMENT_TAG}`. # REVERTED: Put back the request for END_DOCUMENT_TAG
- **Translate:** Non-English text to English.
- **Math Accuracy:** Preserve math accurately.
- **Structure:** Maintain semantic structure.
- **Figures/Tables:** Use placeholders like `placeholder_chunk{chunk_num}_imgX.png`.
- **Citations:** {bib_key_info} If citing a key NOT in this list, use a textual placeholder like `[Author, Year]` instead of \\cite.
- **Escaping:** Escape special LaTeX characters in text.
- **Focus:** Generate LaTeX only for content from "{start_desc}" to "{end_desc}".

# Output Format:
Provide the complete LaTeX document directly, without any surrounding markdown like ```latex ... ```."""

    # --- API Call Implementation ---
    start_time = time.time()
    latex_output = None
    success = False
    error_msg = None

    try:
        # 1. Construct messages payload
        content_blocks = [
             {
                 "type": "document",
                 "source": {"type": "base64", "media_type": media_type, "data": pdf_base64}
             },
             {
                 "type": "text",
                 "text": user_prompt_text
             }
         ]
        messages = [{"role": "user", "content": content_blocks}]

        # 2. Make the API call
        logger.debug(f"Claude (Chunk {chunk_num}): Sending prompt to {nickname}...")
        response = anthropic_client.messages.create(
             model=model_name,
             max_tokens=4096, # Adjust as needed
             messages=messages,
             system="You are an expert LaTeX translator. Generate complete, compilable LaTeX documents based on the provided PDF content and instructions. Output only the raw LaTeX code.",
             temperature=0.3,
             # timeout=... # httpx timeout can be configured on the client
         )

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
            error_msg = f"Claude API error. Stop Reason: {response.stop_reason}"
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

    # No file cleanup needed for Claude as it's inline

    elapsed = time.time() - start_time
    logger.info(f"Claude (Chunk {chunk_num}): API call finished in {elapsed:.2f}s. Success: {success}")
    # Log prompt text only (don't log large base64 pdf)
    log_prompt_and_response(chunk_num, page_range_str, "claude", f"{nickname} ({model_name})", user_prompt_text, latex_output if success else error_msg, success, error_msg if not success else None)
    return latex_output, success, error_msg

# --- Compilation and State ---

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
        # Run pdflatex first.
        # If aux file exists and bib file exists, run bibtex (only once).
        # Run pdflatex again (potentially multiple times) to resolve refs.
        if pass_num == 1:
             command = [ "pdflatex", "-interaction=nonstopmode", "-file-line-error", "-halt-on-error", f"-output-directory={str(aux_dir.resolve())}", str(tex_filepath.resolve()) ]
             current_pass_desc = f"Initial pdflatex (Pass {pass_num})"
        # Run BibTeX on the *second* pass IF needed and not already run
        elif pass_num == 2 and aux_file.exists() and bib_file_exists and not bibtex_run_occurred:
             command = [ "bibtex", str(aux_file.resolve().stem) ] # Bibtex often just needs the stem
             # Bibtex needs to run *in* the aux directory to find files (.aux, potentially .bst)
             run_cwd = aux_dir.resolve()
             is_bibtex_run = True
             bibtex_run_occurred = True # Mark bibtex as having run
             current_pass_desc = f"BibTeX (Pass {pass_num})"
        # Run pdflatex on subsequent passes
        elif pass_num >= 2:
             command = [ "pdflatex", "-interaction=nonstopmode", "-file-line-error", "-halt-on-error", f"-output-directory={str(aux_dir.resolve())}", str(tex_filepath.resolve()) ]
             current_pass_desc = f"Post-BibTeX pdflatex (Pass {pass_num})"
             # Stop if we've run enough passes after bibtex (or enough if no bibtex)
             if pass_num > max_passes:
                 full_log += f"\n--- Completed {max_passes} compilation passes. Stopping. ---\n"
                 break
        else: # Safety break for unexpected state
             full_log += f"\n--- Skipping unexpected pass logic state {pass_num} ---\n"
             continue

        full_log += f"\n--- {current_pass_desc} ---\nRun: {' '.join(command)}\nCWD: {run_cwd or '.'}\n" # Log CWD
        start_time = time.time()
        process = None # Define process outside try
        try:
            # Run the command
            process = subprocess.run(
                 command,
                 capture_output=True,
                 text=True,
                 encoding='utf-8',
                 errors='replace',
                 timeout=timeout,
                 check=False, # Don't raise exception on non-zero exit
                 cwd=run_cwd # Specify CWD if needed (for bibtex)
            )
            elapsed = time.time() - start_time
            full_log += f"Pass {pass_num} completed in {elapsed:.2f}s. Return Code: {process.returncode}\n"
            # Capture relevant output snippets
            stdout_snippet = process.stdout[-1000:].strip() if process.stdout else ""
            stderr_snippet = process.stderr.strip() if process.stderr else ""
            if stdout_snippet: full_log += f"Stdout (last 1k):\n{stdout_snippet}\n...\n"
            if stderr_snippet: full_log += f"Stderr:\n{stderr_snippet}\n"

            log_content = ""
            errors_found_in_log = []
            # Check log file *after* pdflatex run
            if log_file.exists() and not is_bibtex_run:
                try:
                    log_content = log_file.read_text(encoding='utf-8', errors='replace')
                    # More specific error checks
                    fatal_errors = re.findall(r"^! ", log_content, re.MULTILINE) # Any line starting with ! is usually fatal
                    undefined_errors = re.findall(r"Undefined control sequence", log_content)
                    missing_file_errors = re.findall(r"LaTeX Error: File `.*?' not found", log_content)
                    errors_found_in_log = fatal_errors + undefined_errors + missing_file_errors
                except Exception as log_read_e: 
                    full_log += f"\n--- Error reading log file {log_file}: {log_read_e} ---\n"
            elif process.returncode != 0 and not is_bibtex_run:
                 full_log += f"\n--- Log file not found after {current_pass_desc} exited with code {process.returncode}. Assuming failure. ---\n"
                 success = False
                 break # Exit loop on definite failure

            # --- Decide Outcome of This Pass ---
            if errors_found_in_log:
                full_log += f"\n--- LaTeX Error(s) detected in log ({current_pass_desc}): {errors_found_in_log[:5]}... ---\n"
                success = False
                break # Exit loop on fatal LaTeX error
            elif process.returncode != 0:
                if is_bibtex_run:
                     # Read bibtex log (.blg) if it exists
                     blg_file = aux_dir / f"{base_name}.blg"
                     blg_content = ""
                     if blg_file.exists():
                          try: blg_content = blg_file.read_text(encoding='utf-8', errors='replace')[-1000:]
                          except Exception as e_blg: blg_content = f"(Error reading .blg: {e_blg})"
                     full_log += f"\n--- Warning: BibTeX exited with code {process.returncode}. Citations might be unresolved. Log (stdout/stderr/blg tail):\n{stdout_snippet}\n{stderr_snippet}\n{blg_content}\n---"
                     # Don't break, let subsequent pdflatex runs proceed
                else: # pdflatex failed, but no obvious log errors yet
                    # If it's the last *intended* pdflatex run, treat as failure
                    if pass_num >= max_passes:
                        full_log += f"\n--- {current_pass_desc} exited with code {process.returncode} on final pass ({pass_num}/{max_passes}), no specific errors in log. Treating as failure. ---\n"
                        success = False
                        break
                    else: 
                        # Might be temporary (e.g., undefined refs), let it run again
                        full_log += f"\n--- {current_pass_desc} exit code {process.returncode} on pass {pass_num}, no critical errors in log yet. Continuing. ---\n"
            # If this pass ran pdflatex successfully
            elif not is_bibtex_run: 
                 full_log += f"\n--- {current_pass_desc} completed successfully (Exit code 0). ---\n"
                 # Check for PDF only after the final intended pass
                 if pass_num >= max_passes:
                      if pdf_file.is_file() and pdf_file.stat().st_size > 100: # Basic check for non-empty PDF
                           success = True
                           final_pdf_path = pdf_file
                           full_log += f"\n--- PDF file {pdf_file.name} generated successfully after {max_passes} passes. ---\n"
                      else:
                           full_log += f"\n--- Compilation finished ({max_passes} passes), but PDF file {pdf_file.name} is missing or empty. ---\n"
                           success = False
                      break # Stop after final check

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

        # If a pdflatex run explicitly failed (errors in log or non-zero exit on last run), stop.
        if not success and not is_bibtex_run and (errors_found_in_log or (process and process.returncode != 0 and pass_num >= max_passes)):
             full_log += f"\n--- Stopping compilation due to failure in pass {pass_num} ({current_pass_desc}) ---\n"
             break

    # --- Final Logging ---
    if success and final_pdf_path:
        logger.info(f"Compilation successful: {tex_filepath.name} -> {final_pdf_path.name}")
    else:
        logger.error(f"Compilation FAILED: {tex_filepath.name}")
        # Save the detailed log to a failure file
        fail_log_path = aux_dir / f"{base_name}_compile_fail.log"
        try:
            fail_log_path.write_text(full_log, encoding='utf-8')
            logger.error(f"Detailed compilation log saved to: {fail_log_path}")
        except Exception as e_write: 
            logger.error(f"Could not write detailed failure log to {fail_log_path}: {e_write}")

    return success, full_log, final_pdf_path # Return final PDF path only on success

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

def save_failed_chunk_data(chunk_num, start_page, end_page, latex_content, compile_log):
    """Saves the LaTeX and compile log for a failed chunk."""
    try:
        base_name = f"failed_chunk_{chunk_num}_p{start_page}-p{end_page}"
        failed_tex_path = FAILED_CHUNKS_DIR / f"{base_name}.tex"
        failed_log_path = FAILED_CHUNKS_DIR / f"{base_name}.log"

        if latex_content: failed_tex_path.write_text(latex_content, encoding='utf-8')
        if compile_log: failed_log_path.write_text(compile_log, encoding='utf-8')
        logger.info(f"Saved failed data for chunk {chunk_num} to {FAILED_CHUNKS_DIR}")
    except Exception as e: logger.error(f"Error saving failed chunk data for chunk {chunk_num}: {e}")

def read_state():
    """Reads the processing state."""
    # Added provider, file_ref
    default_state = {
        "last_completed_chunk_index": 0,
        "chunk_ranges": None,
        "provider": None,
        "original_pdf_upload_ref": None, # URI (Gemini) or File ID (OpenAI) or None (Claude)
        "preamble": None,
    }
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, 'r', encoding='utf-8') as f: state = json.load(f)
            # Basic validation
            if isinstance(state.get("last_completed_chunk_index"), int):
                 logger.info(f"Loaded state: Last completed chunk index = {state['last_completed_chunk_index']}, Provider = {state.get('provider', 'N/A')}, FileRef = {state.get('original_pdf_upload_ref', 'N/A')}")
                 # Ensure keys exist
                 for key in default_state: state.setdefault(key, None)
                 return state
            else: logger.warning(f"State file {STATE_FILE} invalid format. Resetting."); return default_state
        except Exception as e: logger.error(f"Error reading state file {STATE_FILE}: {e}. Resetting."); return default_state
    logger.info("No state file found. Starting fresh.")
    return default_state

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
    final_preamble = f"% Preamble synthesized by AI ({provider} {model_info.get('nickname', '')})\\n" + cleaned_preamble
    
    # Log the final synthesized preamble (first/last few lines for brevity)
    preview_lines = final_preamble.splitlines()
    preview = "\\n".join(preview_lines[:5]) + ("\\n..." if len(preview_lines) > 5 else "")
    logger.debug(f"Final synthesized preamble preview:\\n{preview}")

    return final_preamble

# --- NEW: Bibliography Extraction Helper ---
def extract_bibliography_with_ai(
    original_pdf_path: Path,
    provider: str,
    model_info: Dict[str, Any]
) -> Optional[str]:
    """Calls the specified AI model to extract bibliography data as BibTeX.
    Returns the raw BibTeX string or None on failure.
    """
    logger.info(f"Calling {provider} model {model_info['nickname']} for BibTeX extraction..."
    )
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
@article{{key, # Escaped outer braces
  author  = {{Author Name}}, # Escaped inner and outer braces
  title   = {{Article Title}},
  journal = {{Journal Name}},
  year    = {{Year}},
  volume  = {{Volume Number}},
  pages   = {{Page Range}}
}} # Escaped outer braces

Start your response directly with the first BibTeX entry (e.g., `@article{{...`).
"""

    # Use the central prompt sending function
    # Requires uploading the full PDF for context
    temp_file_ref: Optional[Union[glm.File, str]] = None
    uploaded_for_this_call = False
    try:
        provider_client = get_provider_client(provider)
        if not provider_client:
            raise ValueError(f"Could not get client for provider: {provider}")

        # Upload the original PDF temporarily if not Claude
        files_for_prompt = []
        if provider == "gemini" or provider == "openai":
            logger.info(f"{provider.capitalize()} (BibTeX): Uploading original PDF..."
            )
            temp_file_ref = upload_pdf_to_provider(original_pdf_path, provider)
            if not temp_file_ref:
                raise RuntimeError(f"{provider.capitalize()} BibTeX PDF upload failed.")
            logger.info(f"{provider.capitalize()} (BibTeX): Upload complete.")
            files_for_prompt.append(temp_file_ref)
            uploaded_for_this_call = True # Mark for deletion
        elif provider == "claude":
            # Claude requires inline base64, prepare prompt accordingly
            logger.info("Claude (BibTeX): Reading and encoding original PDF...")
            try:
                pdf_bytes = original_pdf_path.read_bytes()
                pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
                # Construct Claude-specific content block
                content_blocks = [
                     {
                         "type": "document",
                         "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_base64}
                     },
                     {
                         "type": "text",
                         "text": prompt # The BibTeX request prompt
                     }
                 ]
                # Override the simple text prompt for Claude
                prompt = content_blocks
            except Exception as e_claude_encode:
                 logger.error(f"Claude (BibTeX): Failed to read/encode PDF: {e_claude_encode}")
                 raise

        # Use the central function to handle API calls
        response_data = send_prompt_to_provider(
            prompt=prompt, # Can be string or Claude content blocks
            provider=provider,
            model_info=model_info,
            client=provider_client,
            max_retries=2,
            temperature=0.1, # Low temp for factual extraction
            system_prompt="You are a bibliographic assistant. Respond ONLY with valid BibTeX.",
            files=files_for_prompt, # Only used by Gemini currently in send_prompt
            timeout_seconds=240 # Allow more time for full doc analysis
        )

        if response_data and response_data.get("status") == "success":
            raw_response = response_data.get("content")
            # Basic validation: check if it looks like BibTeX
            if raw_response and "@" in raw_response[:20]:
                logger.info(f"BibTeX extraction successful.")
            else:
                # Rewritten logging to avoid embedding raw_response slice directly in f-string
                log_message_prefix = "AI response received, but may not be valid BibTeX:\n"
                response_preview = (raw_response[:200] + "...") if raw_response else "(Empty Response)"
                logger.warning(log_message_prefix + response_preview)
                # Decide whether to return potentially invalid response or None
                # For now, let's return it and see if it causes issues downstream

            log_prompt_and_response(
                task_id="bibliography_extraction",
                context_id=original_pdf_path.name,
                provider=provider,
                model_name=model_info.get('nickname', 'default'),
                prompt=str(prompt), # Log the text version of prompt
                response=raw_response,
                success=True
            )
        else:
            err_msg = response_data.get("error_message", "Unknown error during AI call")
            logger.error(f"AI BibTeX extraction failed: {err_msg}")
            log_prompt_and_response(
                task_id="bibliography_extraction",
                context_id=original_pdf_path.name,
                provider=provider,
                model_name=model_info.get('nickname', 'default'),
                prompt=str(prompt),
                response=None,
                success=False,
                error_msg=err_msg
            )
            raw_response = None # Ensure None is returned on failure

    except Exception as e:
        logger.error(f"Exception during AI BibTeX extraction ({provider}): {e}", exc_info=True)
        log_prompt_and_response(
            task_id="bibliography_extraction",
            context_id=original_pdf_path.name,
            provider=provider,
            model_name=model_info.get('nickname', 'default'),
            prompt=str(prompt),
            response=None,
            success=False,
            error_msg=f"EXCEPTION: {e}"
        )
        raw_response = None
    finally:
        # Delete the temporarily uploaded full PDF if we uploaded it here
        if uploaded_for_this_call and temp_file_ref:
             logger.info(f"{provider.capitalize()} (BibTeX): Cleaning up temporary PDF upload...")
             delete_provider_file(temp_file_ref, provider)

    elapsed = time.time() - start_time
    if raw_response:
        logger.info(f"AI BibTeX extraction finished ({elapsed:.2f}s)."
        )
    else:
        logger.error(f"AI BibTeX extraction failed ({elapsed:.2f}s)."
        )

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

def main(input_pdf_cli=None, requested_provider=None, requested_model_key=None):
    global TARGET_PROVIDER, TARGET_MODEL_INFO, fallback_preamble, gemini_cache, gemini_cache_name # Ensure cache vars are accessible

    start_time_main = time.time()
    logger.info("="*60)
    logger.info(f"--- Starting PDF to LaTeX Conversion ---")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("="*60)

    # --- Setup ---
    if not setup_clients(): return
    if not setup_target_provider_and_model(requested_provider, requested_model_key): return

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

    # --- Provider-Specific Page Limit Check ---
    page_limit = None # Currently only used for informative error message
    if TARGET_PROVIDER == "gemini": page_limit = 150 # Rough estimate, check docs
    elif TARGET_PROVIDER == "openai": page_limit = 80 # Rough estimate, check docs
    elif TARGET_PROVIDER == "claude": page_limit = 100 # Rough estimate, check docs

    # Note: This check is currently only informative. The script processes chunks,
    # so the main limit is per-chunk API limits, not total pages for the script itself.
    if page_limit and total_pages > page_limit:
         logger.warning(f"Warning: Input PDF has {total_pages} pages, which might exceed typical single-call limits for {TARGET_PROVIDER}, but script will process in chunks.")
         # return # Don't exit, just warn

    # --- State and Chunk Management ---
    state = read_state()
    chunk_definition_file = WORKING_DIR / "chunk-division.txt"

    last_completed_chunk_index = state.get("last_completed_chunk_index", 0)
    chunk_definitions = None
    original_pdf_upload_ref = state.get("original_pdf_upload_ref", None)

    # --- Step 1: Upload Full PDF & Create Cache (If needed) --- 
    # Caching logic is simplified/commented out as it needs more provider-specific implementation details
    gemini_cache = None
    gemini_cache_name = None
    # Re-establish Gemini file object from state if resuming
    if TARGET_PROVIDER == "gemini" and isinstance(original_pdf_upload_ref, str):
         try:
             logger.info(f"Gemini: Attempting to retrieve file object for {original_pdf_upload_ref}...")
             original_pdf_upload_ref = genai.get_file(name=original_pdf_upload_ref) # Try to get the File object
             logger.info(f"Gemini: Retrieved file object: {original_pdf_upload_ref.name}")
         except Exception as e_get:
             logger.error(f"Gemini: Could not retrieve file object '{original_pdf_upload_ref}' from state. Upload may be orphaned: {e_get}")
             original_pdf_upload_ref = None # Reset if retrieval fails

    if TARGET_PROVIDER in ["gemini", "openai"] and not original_pdf_upload_ref:
        logger.info(f"Uploading full PDF for {TARGET_PROVIDER}...")
        uploaded_ref = upload_pdf_to_provider(INPUT_PDF_PATH_ACTUAL, TARGET_PROVIDER)
        if not uploaded_ref:
            logger.critical(f"Failed to upload PDF via {TARGET_PROVIDER}. Cannot proceed.")
            return
        original_pdf_upload_ref = uploaded_ref # Store the actual object/ID
        # Store appropriate ref in state
        if isinstance(original_pdf_upload_ref, glm.File): state["original_pdf_upload_ref"] = original_pdf_upload_ref.name
        elif isinstance(original_pdf_upload_ref, str): state["original_pdf_upload_ref"] = original_pdf_upload_ref
        state["provider"] = TARGET_PROVIDER
        write_state(state)

    # Simplified Cache Handling (Commented Out - requires genai library updates/review)
    # if TARGET_PROVIDER == "gemini" and isinstance(original_pdf_upload_ref, glm.File) and TARGET_MODEL_INFO.get("supports_cache"):
    #     logger.info(f"Gemini: Checking/Creating cache for file {original_pdf_upload_ref.name}...")
    #     # Implement proper cache checking and creation here using latest genai methods
    #     # gemini_cache_name = find_or_create_cache(...)
    #     pass # Placeholder
    elif TARGET_PROVIDER == "claude":
         # Ensure state reflects Claude provider if starting fresh
         if last_completed_chunk_index == 0 and state.get("provider") != "claude":
             state["provider"] = TARGET_PROVIDER
             state["original_pdf_upload_ref"] = None
             write_state(state)

    # --- Step 2: Determine Chunk Definitions --- 
    ai_chunk_definitions_raw = None
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
                # Handle potential markdown fences around JSON
                json_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", ai_chunk_definitions_raw, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = ai_chunk_definitions_raw # Assume raw response is JSON
                parsed_ai_data = json.loads(json_str)
            except json.JSONDecodeError as json_e:
                logger.error(f"AI returned invalid JSON for chunk definitions: {json_e}")
                logger.debug(f"Raw AI response for chunk def:\n{ai_chunk_definitions_raw}")
        # Pass potentially parsed data (or None) to saving/fallback function
        chunk_definitions = determine_and_save_chunk_definitions(
            INPUT_PDF_PATH_ACTUAL, total_pages, TARGET_PAGES_PER_CHUNK,
            TARGET_PROVIDER, TARGET_MODEL_INFO, chunk_definition_file,
            ai_chunk_data=parsed_ai_data
        )
        if not chunk_definitions: 
            logger.critical("Failed to determine chunk definitions via AI or fallback. Cannot proceed.")
            return
        last_completed_chunk_index = 0 # Reset progress if chunks were redetermined
        state["last_completed_chunk_index"] = 0
        write_state(state)

    if not chunk_definitions: # Final check
        logger.critical("CRITICAL ERROR: No chunk definitions available. Cannot proceed.")
        return

    total_chunks = len(chunk_definitions)
    logger.info(f"Using {total_chunks} chunk definitions.")

    # --- Step 2.5: Extract Bibliography --- 
    bib_file_path = WORKING_DIR / f"{INPUT_PDF_PATH_ACTUAL.stem}.bib"
    bib_content = None # Initialize here to guarantee scope
    valid_bib_keys: List[str] = [] # Initialize empty list for keys

    if not bib_file_path.exists() or last_completed_chunk_index == 0:
        logger.info(f"--- Extracting Bibliography --- ")
        bib_content_extracted = extract_bibliography_with_ai(INPUT_PDF_PATH_ACTUAL, TARGET_PROVIDER, TARGET_MODEL_INFO)
        if bib_content_extracted:
            try:
                # Simple validation: does it look like BibTeX?
                if "@" in bib_content_extracted[:50]: 
                    bib_file_path.write_text(bib_content_extracted, encoding='utf-8')
                    logger.info(f"Bibliography saved to {bib_file_path}")
                    bib_content = bib_content_extracted # Assign to main variable
                    valid_bib_keys = extract_bib_keys(bib_content) # Extract keys now
                else:
                     logger.warning("AI response for bibliography did not look like BibTeX. Discarding.")
                     bib_content = None
            except Exception as e:
                logger.error(f"Failed to write bibliography file {bib_file_path}: {e}")
                bib_content = None # Ensure None if save fails
        else:
            logger.warning("AI failed to extract bibliography. Citations may not work.")
            # Clean up potentially empty/invalid file if it exists
            if bib_file_path.exists():
                 try: bib_file_path.unlink()
                 except OSError: pass 
    else:
         logger.info(f"Skipping BibTeX extraction, file already exists: {bib_file_path}")
         try:
             bib_content = bib_file_path.read_text(encoding='utf-8') # Read into main variable
             valid_bib_keys = extract_bib_keys(bib_content) # Extract keys now
         except Exception as e:
             logger.error(f"Failed to read existing bibliography file {bib_file_path}: {e}")
             bib_content = None # Ensure None on read error
             valid_bib_keys = [] # Ensure empty list on read error
    
    # This check should no longer be needed as valid_bib_keys is set above
    # if bib_content and not valid_bib_keys:
    #      valid_bib_keys = extract_bib_keys(bib_content) 

    # Log the status before starting the loop
    logger.info(f"Proceeding to chunk processing with {len(valid_bib_keys)} BibTeX keys known.")

    # --- Step 3: Process Chunks --- 
    start_chunk_index = last_completed_chunk_index
    successful_preambles = []
    successful_bodies = []
    processing_error_occurred = False

    for chunk_index in range(start_chunk_index, total_chunks):
        chunk_info = chunk_definitions[chunk_index]
        chunk_num = chunk_info['chunk_num']
        start_page = chunk_info['start_page']
        end_page = chunk_info['end_page']
        start_desc = chunk_info['start_desc']
        end_desc = chunk_info['end_desc']
        page_range_str = f"Pages {start_page}-{end_page}"
        desc_str = f'("{start_desc}" to "{end_desc}")' # Corrected quote escaping
        logger.info(f"--- Processing Chunk {chunk_num}/{total_chunks} ({page_range_str}) {desc_str} --- ")

        # Create chunk PDF
        SAVED_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
        chunk_pdf_path = create_pdf_chunk_file(INPUT_PDF_PATH_ACTUAL, start_page, end_page, SAVED_CHUNKS_DIR)
        if not chunk_pdf_path: # Handle failure to create chunk PDF
            logger.error(f"Failed to create PDF for chunk {chunk_num}. Skipping.")
            successful_bodies.append(create_chunk_failure_placeholder(chunk_num, start_page, end_page, "Failed to create chunk PDF"))
            successful_preambles.append(f"% Preamble for failed chunk {chunk_num} omitted")
            # Update state even if chunk creation fails, so we don't retry indefinitely
            last_completed_chunk_index = chunk_index + 1 
            state["last_completed_chunk_index"] = last_completed_chunk_index
            write_state(state)
            processing_error_occurred = True # Mark that an error occurred
            continue

        chunk_full_latex, chunk_compile_success = None, False
        extracted_preamble, extracted_body = None, None
        chunk_tex_path = WORKING_DIR / f"attempt_chunk_{chunk_num}.tex"
        compile_log = "" # Initialize compile log string

        # --- Retry Loop --- 
        for attempt in range(1, MAX_CHUNK_RETRIES + 1):
            logger.info(f"Attempt {attempt}/{MAX_CHUNK_RETRIES} for Chunk {chunk_num}...")
            api_output = None
            api_success = False
            api_error_msg = "Attempt Failed"

            # --- Call Provider API (Updated) ---
            common_args = {
                "chunk_pdf_path": chunk_pdf_path,
                "chunk_num": chunk_num,
                "total_chunks": total_chunks,
                "start_page": start_page,
                "end_page": end_page,
                "start_desc": start_desc,
                "end_desc": end_desc,
                "model_info": TARGET_MODEL_INFO,
                "valid_bib_keys": valid_bib_keys # Pass the extracted keys
            }
            if TARGET_PROVIDER == "gemini":
                api_output, api_success, api_error_msg = get_latex_from_chunk_gemini(
                    **common_args,
                    gemini_cache_name=gemini_cache_name # Pass cache name if available
                )
            elif TARGET_PROVIDER == "openai":
                 api_output, api_success, api_error_msg = get_latex_from_chunk_openai(**common_args)
            elif TARGET_PROVIDER == "claude":
                 api_output, api_success, api_error_msg = get_latex_from_chunk_claude(**common_args)
            else:
                logger.critical(f"Unsupported TARGET_PROVIDER '{TARGET_PROVIDER}' in processing loop.")
                processing_error_occurred = True
                break # Break retry loop

            if processing_error_occurred: break # Break chunk loop if provider unsupported

            if not api_success:
                logger.error(f"API call failed for chunk {chunk_num} (Attempt {attempt}): {api_error_msg}")
                if attempt < MAX_CHUNK_RETRIES:
                    logger.info(f"Retrying chunk {chunk_num} API call...")
                    time.sleep(random.uniform(3, 7) * attempt) # Exponential backoff
                    continue # Next attempt
                else:
                    logger.error(f"Chunk {chunk_num} failed all API attempts.")
                    chunk_compile_success = False # Ensure marked as failed
                    break # Exit retry loop for this chunk

            # --- Compile Individual Chunk ---
            logger.info(f"API successful for chunk {chunk_num}. Compiling individually...")
            chunk_full_latex = api_output # Store the successful output

            try:
                if isinstance(chunk_full_latex, str):
                    # Remove potential markdown fences
                    md_match = re.search(r"```(?:latex)?\s*(.*?)\s*```", chunk_full_latex, re.DOTALL | re.IGNORECASE)
                    if md_match: 
                        logger.debug(f"Chunk {chunk_num}: Removed markdown fences from output.")
                        chunk_full_latex = md_match.group(1).strip()
                    else: 
                        chunk_full_latex = chunk_full_latex.strip()
                    
                    # Ensure it starts and ends correctly (basic check)
                    if not chunk_full_latex.startswith("\\documentclass"):
                         logger.warning(f"Chunk {chunk_num} output missing \\documentclass. Prepending fallback.")
                         # Try prepending a basic documentclass - might still fail
                         chunk_full_latex = fallback_preamble.splitlines()[0] + "\n" + chunk_full_latex 
                    if not chunk_full_latex.strip().endswith(END_DOCUMENT_TAG):
                         logger.warning(f"Chunk {chunk_num} output missing \\\\end{{document}}. Appending it.")
                         chunk_full_latex += f"\n\n{END_DOCUMENT_TAG}"
                else: 
                    # Should not happen if API success is True, but safety check
                    raise TypeError("API reported success but output was not a string.")
                
                chunk_tex_path.write_text(chunk_full_latex, encoding='utf-8')
            except Exception as e:
                logger.error(f"Error preparing/writing chunk LaTeX file {chunk_tex_path.name}: {e}. Skipping compile attempt.")
                compile_log = f"Error preparing/writing LaTeX file: {e}"
                if attempt < MAX_CHUNK_RETRIES: 
                    logger.info("Retrying chunk due to write error...")
                    time.sleep(random.uniform(2, 5))
                    continue # Retry API call
                else: 
                    chunk_compile_success = False
                    break # Exit retry loop

            # Compile (calls pdflatex, possibly bibtex if needed)
            # Note: Individual chunk compile doesn't use the main .bib file usually
            compile_ok, compile_log, _ = compile_latex(
                chunk_tex_path,
                LATEX_AUX_DIR,
                main_bib_file_path=bib_file_path # Pass the main bib file path
            )

            if compile_ok:
                logger.info(f"Individual compilation successful for chunk {chunk_num}.")
                try:
                    # Read the *successfully compiled* content back for parsing
                    compiled_tex_content = chunk_tex_path.read_text(encoding='utf-8')
                    # Corrected regex strings (removed trailing backslash, fixed escaping)
                    preamble_match = re.match(r"(.*?)(?:\\begin\{document\})", compiled_tex_content, re.DOTALL | re.IGNORECASE)
                    body_match = re.search(r"\\begin\{document\}(.*?)(?:\\end\{document\})?", compiled_tex_content, re.DOTALL | re.IGNORECASE)
                    
                    if preamble_match and body_match:
                        extracted_preamble = preamble_match.group(1).strip()
                        extracted_body = body_match.group(1).strip()
                        # Basic check if preamble/body look reasonable
                        if extracted_preamble and extracted_body: 
                            logger.debug(f"Extracted preamble ({len(extracted_preamble)} chars) and body ({len(extracted_body)} chars).")
                            chunk_compile_success = True
                            break # Success! Exit retry loop for this chunk
                        else:
                             logger.error("Parsed empty preamble or body from successfully compiled chunk. Treating as failure.")
                             compile_log += "\nError: Parsed empty preamble/body after successful compile."
                             chunk_compile_success = False # Ensure marked as failed
                    else:
                         logger.error("Could not parse preamble/body from successfully compiled chunk using regex. Treating as failure.")
                         compile_log += "\nError: Failed to parse preamble/body regex after successful compile."
                         chunk_compile_success = False # Ensure marked as failed
                         
                except Exception as parse_e: 
                     logger.error(f"Error parsing preamble/body: {parse_e}. Treating as failure.", exc_info=True)
                     compile_log += f"\nError: Exception during preamble/body parsing: {parse_e}"
                     chunk_compile_success = False # Ensure failure on exception

                # If parsing failed, break retry loop (already marked as failure)
                if not chunk_compile_success:
                    break 
                    
            else: # Compile failed
                logger.warning(f"Individual compilation failed for chunk {chunk_num} (Attempt {attempt}).")
                # Save data for this failed attempt
                save_failed_chunk_data(chunk_num, start_page, end_page, chunk_full_latex, compile_log) # Pass current latex/log
                if attempt < MAX_CHUNK_RETRIES:
                     logger.info("Retrying chunk API call after compile failure...")
                     time.sleep(random.uniform(3, 7) * attempt) # Exponential backoff before API retry
                     continue # Continue to the next attempt in the retry loop (will call API again)
                else:
                     logger.error(f"Chunk {chunk_num} failed compilation after all retries.")
                     chunk_compile_success = False # Ensure marked as failed
                     break # Exit retry loop, chunk failed all attempts
            
        # --- End Retry Loop --- 
        
        # If a critical error like unsupported provider occurred, break main loop
        if processing_error_occurred and not chunk_compile_success: 
            logger.error("Exiting main processing loop due to critical error.")
            break 

        # Clean up chunk PDF regardless of success/failure for this chunk
        if chunk_pdf_path and chunk_pdf_path.exists():
            logger.debug(f"Cleaning up chunk PDF: {chunk_pdf_path.name}")
            try: chunk_pdf_path.unlink()
            except OSError as e: logger.warning(f"Could not delete chunk PDF {chunk_pdf_path.name}: {e}")

        # Process result for the chunk
        if chunk_compile_success and extracted_preamble is not None and extracted_body is not None:
            logger.info(f"Successfully processed Chunk {chunk_num}.")
            successful_preambles.append(extracted_preamble)
            successful_bodies.append(extracted_body)
            last_completed_chunk_index = chunk_index + 1
            state["last_completed_chunk_index"] = last_completed_chunk_index
            write_state(state)
        else:
             # Log final failure for the chunk and add placeholder
             logger.error(f"Failed to process chunk {chunk_num} after all retries. Adding placeholder.")
             placeholder_reason = f"Chunk failed API/compilation after {MAX_CHUNK_RETRIES} attempts."
             placeholder_body = create_chunk_failure_placeholder(chunk_num, start_page, end_page, placeholder_reason)
             successful_bodies.append(placeholder_body)
             successful_preambles.append(f"% Preamble for failed chunk {chunk_num} omitted")
             # Ensure state is updated even on failure to avoid infinite loop
             last_completed_chunk_index = chunk_index + 1 
             state["last_completed_chunk_index"] = last_completed_chunk_index
             write_state(state)
             processing_error_occurred = True # Mark that at least one chunk failed
             # Save the *last* failed attempt's data if not already saved during compile fail
             if chunk_full_latex: # Only save if we got some output from API
                 save_failed_chunk_data(chunk_num, start_page, end_page, chunk_full_latex, compile_log) 

        # Clean up the last attempt .tex file for this chunk
        if chunk_tex_path.exists():
            try: chunk_tex_path.unlink()
            except OSError: logger.warning(f"Could not delete attempt file {chunk_tex_path.name}")
    
    # --- End Chunk Processing Loop ---

    # --- Step 4: Final Preamble Synthesis --- 
    final_preamble = None
    # Only attempt synthesis if all chunks were processed (even if some failed and have placeholders)
    if last_completed_chunk_index == total_chunks:
        logger.info("All chunks processed (or placeholders added). Attempting final preamble synthesis...")
        
        # Filter out placeholder preambles before synthesis
        valid_preambles_for_synthesis = [p for p in successful_preambles if not p.startswith("% Preamble for failed chunk")]
        
        if not valid_preambles_for_synthesis:
             logger.warning("No successful preambles extracted from any chunk. Using basic fallback preamble.")
             final_preamble = fallback_preamble # Use the full fallback defined at the start
        else:
            logger.info(f"Attempting to synthesize final preamble using AI from {len(valid_preambles_for_synthesis)} successful chunks...")
            final_preamble = generate_final_preamble_with_ai(
                valid_preambles_for_synthesis, # Use only valid preambles
                TARGET_PROVIDER,
                TARGET_MODEL_INFO,
                INPUT_PDF_PATH_ACTUAL if TARGET_PROVIDER == "gemini" else None
                )

            if not final_preamble:
                 logger.warning("AI preamble synthesis failed or returned invalid result. Using fallback concatenation (may be messy).")
                 all_preamble_lines = set()
                 doc_class_line = ""
                 # Try to find a documentclass line first
                 for p in valid_preambles_for_synthesis:
                     for line in p.splitlines():
                          stripped_line = line.strip()
                          if stripped_line.startswith("\\documentclass"):
                              if not doc_class_line: doc_class_line = stripped_line
                              # Optional: Check for conflicting document classes
                          elif stripped_line: # Add non-empty, non-documentclass lines
                              all_preamble_lines.add(stripped_line)
                 
                 # Assemble fallback preamble
                 if not doc_class_line:
                      logger.warning("No \\documentclass found in any extracted preamble. Using default from fallback.")
                      doc_class_line = fallback_preamble.splitlines()[0] # Get from global fallback
                 
                 # Combine unique lines, keeping documentclass first
                 other_lines = sorted(list(all_preamble_lines))
                 final_preamble = doc_class_line + "\n" + "\n".join(other_lines)
                 final_preamble = "% Preamble created using fallback concatenation (AI synthesis failed)\n" + final_preamble
            else:
                 logger.info("AI successfully synthesized the final preamble.")
                 # Add comment indicating AI synthesis
                 final_preamble = f"% Preamble synthesized by AI ({TARGET_PROVIDER} {TARGET_MODEL_INFO.get('nickname', '')})\n" + final_preamble.strip()


    # --- Step 5: Final Document Assembly --- 
    # Assemble only if preamble synthesis was attempted (i.e., all chunks processed)
    if final_preamble: 
        logger.info("Concatenating final document...")

        # Check if a bib file exists to add bibliography commands
        bib_commands = ""
        if bib_file_path.exists():
            bib_stem = bib_file_path.stem
            # Choose a common bibliography style (could be refined)
            bib_style = "amsplain" 
            bib_commands = f"\n\\bibliographystyle{{{bib_style}}}\n\\bibliography{{{bib_stem}}}\n"
            logger.info(f"Adding bibliography commands for {bib_stem}.bib with style {bib_style}")
        else:
            logger.info("No .bib file found or extracted, skipping bibliography commands.")

        # Combine preamble, bodies (including placeholders), and bib commands
        final_combined_latex = (
            final_preamble.strip() + "\n\n" + # Ensure preamble ends cleanly
            "\\begin{document}\n\n" +
            "\n\n% ========================================\n\n".join(successful_bodies) + # Join bodies/placeholders
            bib_commands + # Add bib commands (empty string if no bib file)
            "\n\n\\end{document}" # Use double backslash for literal LaTeX command
        )
        
        # Define final output paths
        final_output_filename_base = f"{INPUT_PDF_PATH_ACTUAL.stem}_translated"
        FINAL_TEX_FILE = WORKING_DIR / f"{final_output_filename_base}.tex"
        FINAL_PDF_FILE = WORKING_DIR / f"{final_output_filename_base}.pdf" # Path expected by compile_latex

        try:
            FINAL_TEX_FILE.write_text(final_combined_latex, encoding='utf-8')
            logger.info(f"Final concatenated LaTeX saved to {FINAL_TEX_FILE}")
            
            logger.info("Compiling the final document...")
            # Compile the final document (timeout increased)
            final_compile_ok, final_compile_log, final_pdf_path_from_compile = compile_latex(
                FINAL_TEX_FILE, 
                LATEX_AUX_DIR, 
                main_bib_file_path=bib_file_path, # Pass the main bib file path
                num_runs=3, # More runs often needed for final biblio/refs
                timeout=180 
            ) 
            
            if final_compile_ok and final_pdf_path_from_compile and final_pdf_path_from_compile.exists():
                 # Copy the compiled PDF from aux dir to the main working dir
                 shutil.copy2(final_pdf_path_from_compile, FINAL_PDF_FILE)
                 logger.info(f"Final PDF successfully compiled and saved to {FINAL_PDF_FILE}")
                 # Mark overall success ONLY if final compilation worked
                 # processing_error_occurred might be True if chunks failed, but final compile succeeded
                 # overall_success depends on final PDF existing.
            else: 
                 logger.error(f"Failed to compile the final concatenated document: {FINAL_TEX_FILE.name}")
                 # Don't set processing_error_occurred here, as chunk processing might have finished
                 # Final summary will indicate based on PDF existence

        except Exception as e: 
             logger.error(f"Error writing or compiling final LaTeX file {FINAL_TEX_FILE.name}: {e}", exc_info=True)
             # Mark error if writing/compiling final doc fails catastrophically
             processing_error_occurred = True 

    # --- Final Summary --- 
    end_time_main = time.time()
    total_time_main = end_time_main - start_time_main
    logger.info("="*60)
    
    # Determine overall success based *only* on final PDF existing after attempting assembly
    overall_success = FINAL_PDF_FILE.exists() and FINAL_PDF_FILE.stat().st_size > 100

    if overall_success:
        logger.info(f"--- PDF Translation Completed Successfully ---")
        logger.info(f"Final PDF: {FINAL_PDF_FILE.resolve()}")
        if processing_error_occurred: # Note if chunks failed but final PDF was generated
             logger.warning("Note: Some chunks failed processing and placeholders were used.")
    elif last_completed_chunk_index != total_chunks:
         logger.error(f"--- PDF Translation Failed Critically ---")
         logger.error("Processing stopped before all chunks were attempted.")
         logger.error(f"Last completed chunk index: {last_completed_chunk_index}/{total_chunks}")
    elif not final_preamble:
         logger.error(f"--- PDF Translation Failed ---")
         logger.error("All chunks processed, but final preamble synthesis failed and final document could not be assembled.")
    else: # Chunks finished, assembly attempted, but final PDF missing/empty
        logger.error(f"--- PDF Translation Finished, but Final Compilation Failed ---")
        logger.error(f"All chunks processed ({last_completed_chunk_index}/{total_chunks}), but the final document failed to compile.")
        logger.error(f"Check the final concatenated file: {FINAL_TEX_FILE.resolve()}")
        logger.error(f"Check the final compilation log in: {LATEX_AUX_DIR}")

    # Estimate token usage (very rough)
    # Need a better way to track actual tokens used per call
    total_tokens_estimate = 0 # Placeholder
    logger.info(f"Estimated total token usage (very approximate): ~{total_tokens_estimate}") 
    logger.info(f"Total script execution time: {total_time_main:.2f} seconds ({total_time_main/60:.2f} minutes).")
    logger.info(f"Logs and intermediate files are in: {WORKING_DIR.resolve()}")
    logger.info(f"Prompt/Response Log: {PROMPT_LOG_FILE.resolve()}")
    logger.info("="*60)

    # --- Cleanup ---
    # Clean up Gemini Cache (If logic was implemented and cache created)
    # if TARGET_PROVIDER == "gemini" and gemini_cache:
    #      logger.info(f"Gemini: Cleaning up cache: {gemini_cache.name}")
    #      try: genai.delete_cached_content(name=gemini_cache.name); logger.info("Gemini: Cache deleted successfully.")
    #      except Exception as e: logger.warning(f"Gemini: Failed to delete cache {gemini_cache.name}: {e}")
 
    # Clean up the original full PDF upload (Gemini/OpenAI) if it's stored in state
    provider_in_state = state.get("provider")
    file_ref_in_state = state.get("original_pdf_upload_ref")
    if provider_in_state in ["gemini", "openai"] and file_ref_in_state:
        logger.info(f"Cleaning up uploaded file reference ({provider_in_state}: {file_ref_in_state})...")
        file_ref_to_delete = file_ref_in_state # Use the ID/Name string directly for OpenAI/Gemini delete
        if provider_in_state == "gemini":
             # For Gemini, we need the File object potentially, but delete uses name string
             # If original_pdf_upload_ref holds the File object, get its name
             if isinstance(original_pdf_upload_ref, glm.File): file_ref_to_delete = original_pdf_upload_ref.name
             else: file_ref_to_delete = file_ref_in_state # Assume it's the name string already
        
        if file_ref_to_delete: # Ensure we have something to delete
             delete_provider_file(file_ref_to_delete, provider_in_state) # Pass the provider from state

# --- Command-Line Interface --- 
if __name__ == "__main__":
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
    STATE_FILE = WORKING_DIR / "state.json"
    FINAL_TEX_FILE = WORKING_DIR / "final_document.tex"
    FINAL_PDF_FILE = WORKING_DIR / "final_document.pdf"
    LATEX_AUX_DIR = WORKING_DIR / "latex_aux"
    FAILED_CHUNKS_DIR = WORKING_DIR / "failed_chunks"
    PROMPT_LOG_FILE = WORKING_DIR / "prompts_and_responses.log"
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
    FAILED_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    # Add File Handler to logger
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler): handler.close(); logger.removeHandler(handler)
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    logger.info(f"Logging to console and file: {log_file_path.resolve()}")

    # Call main function
    main(input_pdf_cli=args.input_pdf, requested_provider=args.provider, requested_model_key=args.model)
