import google.generativeai as genai
import os
import argparse
import logging
import subprocess
from dotenv import load_dotenv
from typing import List, Tuple, Optional, Dict, Union # Added Union
from pathlib import Path
from io import BytesIO
import base64
import re
import time
import anthropic
import openai
import pypdf
import json
import shutil
import random
import tempfile
from datetime import datetime
import sys # Added for sys.exit

# --- Image Handling Imports ---
try:
    from pdf2image import convert_from_path, pdfinfo_from_path
    from PIL import Image
except ImportError:
    logging.error("Error: pdf2image or Pillow not installed. Run: pip install pdf2image Pillow")
    exit(1)
except Exception as e:
    logging.error(f"Error importing pdf2image/Pillow: {e}")
    logging.error("Ensure poppler is installed (e.g., 'brew install poppler' or 'sudo apt-get install poppler-utils') and accessible in your system's PATH.")
    exit(1)

# --- Configuration ---
def get_model_configs():
    """Return all available model configurations"""
    # Note: Model names might change. Check provider documentation.
    return {
        # === Claude Models ===
         "claude-3.7-sonnet": {"provider": "claude", "name": "claude-3-7-sonnet-20250219", "nickname": "Claude 3.7 Sonnet", "thinking": True, "context_limit_tokens": 200000}, # Example future model

        # === Gemini Models === (Context limits are estimates/typical values)
         "gemini-2.5-pro-preview": {"provider": "gemini", "name": "gemini-2.5-pro-preview-03-25", "nickname": "Gemini 2.5 Pro Preview", "thinking": True, "context_limit_tokens": 1000000}, # Example future model
         "gemini-2.5-pro-exp": {"provider": "gemini", "name": "gemini-2.5-pro-exp-03-25", "nickname": "Gemini 2.5 Pro Preview", "thinking": True, "context_limit_tokens": 1000000}, # Example future model
        "gemini-2.0-flash": {"provider": "gemini", "name": "gemini-2.0-flash-latest", "nickname": "Gemini 1.5 Flash", "thinking": False, "context_limit_tokens": 1000000}, # Use 'latest' or specific version

        # === OpenAI Models ===
        "gpt-4o": {"provider": "gpt", "name": "gpt-4o", "nickname": "GPT-4o", "thinking": True, "context_limit_tokens": 128000}, # Using general name, API handles version
        "gpt-4o-mini": {"provider": "gpt", "name": "gpt-4o-mini", "nickname": "GPT-4o mini", "thinking": False, "context_limit_tokens": 128000},
         "o3-mini": {"provider": "gpt", "name": "o3-mini-2025-01-31", "nickname": "OpenAI o3 mini", "thinking": False, "context_limit_tokens": 128000 }, # Example future model
    }


# Single default model key if none is specified via --model
DEFAULT_MODEL_KEY = "claude-3.5-sonnet"

# Store the single chosen model config globally
TARGET_MODEL_INFO = None

# Context Truncation Threshold (conservative estimate: 1 char ~ 1 token)
# Reduce this if you hit context limit errors frequently
CONTEXT_TRUNCATION_THRESHOLD_TOKENS = 100000 # E.g., keep last ~100k chars/tokens

def setup_single_model(requested_model_key=None):
    """Set up the single model to be used for the entire run."""
    global TARGET_MODEL_INFO
    all_models = get_model_configs()
    target_key = requested_model_key if requested_model_key else DEFAULT_MODEL_KEY

    if target_key not in all_models:
        logging.error(f"Error: Requested model key '{target_key}' not found in configurations.")
        logging.error("Please use --list-models to see available keys or update get_model_configs().")
        logging.error(f"Falling back to default: '{DEFAULT_MODEL_KEY}'")
        target_key = DEFAULT_MODEL_KEY
        if target_key not in all_models:
             logging.critical(f"CRITICAL: Default model '{DEFAULT_MODEL_KEY}' also not found. Cannot proceed.")
             TARGET_MODEL_INFO = None
             return False

    TARGET_MODEL_INFO = all_models[target_key]
    # Adjust truncation threshold based on selected model's capability, if lower
    global CONTEXT_TRUNCATION_THRESHOLD_TOKENS
    model_limit = TARGET_MODEL_INFO.get('context_limit_tokens', CONTEXT_TRUNCATION_THRESHOLD_TOKENS)
    # Use a safety margin (e.g., 80-90% of model limit) for the actual truncation threshold
    safety_factor = 0.85
    effective_threshold = int(model_limit * safety_factor)
    if effective_threshold < CONTEXT_TRUNCATION_THRESHOLD_TOKENS:
        logging.warning(f"Adjusting context truncation threshold to {effective_threshold} tokens (based on {TARGET_MODEL_INFO['nickname']} limit {model_limit} with {1-safety_factor:.0%} buffer).")
        CONTEXT_TRUNCATION_THRESHOLD_TOKENS = effective_threshold
    else:
        logging.info(f"Using context truncation threshold: {CONTEXT_TRUNCATION_THRESHOLD_TOKENS} tokens (Model limit: {model_limit}).")

    logging.info(f"Using single model for translation: {TARGET_MODEL_INFO['nickname']} ({TARGET_MODEL_INFO['name']}) from provider '{TARGET_MODEL_INFO['provider']}'")
    return True

# Max retries per model for a single page (API or Compile failure)
MAX_RETRIES_PER_MODEL = 3
# Max cycles removed

# Initialize AI clients dictionary
api_clients = {}
# Initialize model stats tracking
model_stats = {"pages_translated": {}, "failures": {}, "context_truncations": {}} # Added context tracking

# Load environment variables
load_dotenv()

# --- API Key Handling and Client Initialization --- (No changes needed here)

# Claude (Anthropic)
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if CLAUDE_API_KEY:
    try:
        api_clients["claude"] = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
        logging.info("Anthropic (Claude) client initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize Anthropic client: {e}")
else:
    logging.warning("ANTHROPIC_API_KEY not set - Claude models cannot be used.")

# OpenAI (GPT)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    try:
        api_clients["gpt"] = openai.OpenAI(api_key=OPENAI_API_KEY)
        logging.info("OpenAI (GPT) client initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client: {e}")
else:
    logging.warning("OPENAI_API_KEY not set - GPT models cannot be used.")

# Google AI (Gemini)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        api_clients["gemini"] = genai # Store the module
        logging.info("Google AI (Gemini) client initialized.")
    except ImportError:
        logging.error("Google AI libraries not installed. Run: pip install google-generativeai")
        api_clients["gemini"] = None # Ensure it's None if import fails
    except Exception as e:
        logging.error(f"Failed to initialize Google AI client: {e}")
        api_clients["gemini"] = None # Ensure it's None on other errors
else:
    logging.warning("GOOGLE_API_KEY or GEMINI_API_KEY not set - Gemini models cannot be used.")


# --- Path and Processing Settings ---
INPUT_PDF_PATH = "input.pdf"
WORKING_DIR = Path("latex_processing_single_full_context") # New name to reflect strategy
STATE_FILE = WORKING_DIR / "state.json"
LAST_GOOD_TEX_FILE = WORKING_DIR / "cumulative_good.tex"
LAST_GOOD_PDF_FILE = WORKING_DIR / "cumulative_good.pdf"
CURRENT_ATTEMPT_TEX_FILE = WORKING_DIR / "current_attempt.tex"
LATEX_AUX_DIR = WORKING_DIR / "latex_aux"
MODEL_STATS_FILE = WORKING_DIR / "model_stats.json"
FAILED_PAGES_DIR = WORKING_DIR / "failed_pages"

# LaTeX Tag Constant
END_DOCUMENT_TAG = "\\end{document}"
BEGIN_DOCUMENT_TAG = "\\begin{document}" # Added for easier splitting

# pdf2image configuration
IMAGE_DPI = 200
IMAGE_FORMAT = 'JPEG'

# Fallback Preamble (No changes needed here)
fallback_preamble = """\\documentclass[11pt]{amsart}
% --- Essential Encoding and Font Packages ---
\\usepackage[utf8]{inputenc}
\\usepackage[T1]{fontenc}
\\usepackage{textcomp} % Required for certain symbols

% --- Math Packages ---
\\usepackage{amsmath, amssymb, amsthm}
\\usepackage{mathtools} % Extends amsmath
\\usepackage{mathrsfs} % For \\mathscr
\\usepackage[mathcal]{euscript} % For \\mathcal variants (optional)
\\usepackage{amsfonts} % For math fonts like blackboard bold
\\usepackage{bm} % For bold math symbols

% --- Graphics and Color ---
\\usepackage{graphicx}
\\usepackage{xcolor}

% --- Layout and Typography ---
\\usepackage[verbose,letterpaper,tmargin=3cm,bmargin=3cm,lmargin=2.5cm,rmargin=2.5cm]{geometry}
\\usepackage{microtype} % Improves typography
\\usepackage{csquotes} % For context-sensitive quotes (e.g., \\enquote{})

% --- Hyperlinks and References ---
\\usepackage{hyperref}
\\hypersetup{
    colorlinks=true,
    linkcolor={red!50!black},
    citecolor={blue!50!black},
    urlcolor={blue!80!black}
}
\\usepackage[capitalise]{cleveref} % Smarter referencing

% --- Diagrams ---
\\usepackage{tikz}
\\usepackage{tikz-cd}
\\usetikzlibrary{matrix, arrows, decorations.markings}

% --- Lists ---
\\usepackage[shortlabels]{enumitem}

% --- Tables ---
\\usepackage{booktabs} % For nicer tables

% --- Basic Theorem Environments (AMS style) ---
\\theoremstyle{plain}
\\newtheorem{theorem}{Theorem}[section] % Numbered within section
\\newtheorem{lemma}[theorem]{Lemma}
\\newtheorem{proposition}[theorem]{Proposition}
\\newtheorem{corollary}[theorem]{Corollary}
\\theoremstyle{definition}
\\newtheorem{definition}[theorem]{Definition}
\\newtheorem{example}[theorem]{Example}
\\newtheorem{remark}[theorem]{Remark}

% --- Basic Math Operators (Add more as needed) ---
\\DeclareMathOperator{\\Hom}{Hom}
\\DeclareMathOperator{\\End}{End}
\\DeclareMathOperator{\\Spec}{Spec}
\\DeclareMathOperator{\\Proj}{Proj}

% --- Title/Author/Date (Placeholders) ---
\\title{Translated Document (Fallback Preamble)}
\\author{AI Translator Script}
\\date{\\today}
"""

# Create directories if they don't exist
WORKING_DIR.mkdir(parents=True, exist_ok=True)
LATEX_AUX_DIR.mkdir(parents=True, exist_ok=True)
FAILED_PAGES_DIR.mkdir(parents=True, exist_ok=True)

# Logging setup
log_file_path = WORKING_DIR / "processing.log"
# Remove existing file handlers before adding new ones
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    if isinstance(handler, logging.FileHandler):
        handler.close()
        root_logger.removeHandler(handler)
    elif isinstance(handler, logging.StreamHandler): # Remove stream handler too if we add one below
        root_logger.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
        logging.StreamHandler() # Also print to console
    ]
)
logging.info(f"Logging to file: {log_file_path.resolve()}")

# --- Helper Functions ---

def extract_latex_parts_from_response(raw_latex: str, is_page_1: bool) -> Union[Tuple[Optional[str], Optional[str]], Optional[str]]:
    """
    Cleans LLM response and extracts LaTeX parts.
    - For Page 1: Returns (preamble, body_content_with_begin_doc) or (None, None).
    - For Pages > 1: Returns body_content_with_begin_doc or None.
    Removes markdown wrappers and extraneous text.
    """
    if not raw_latex:
        return (None, None) if is_page_1 else None

    cleaned_latex = raw_latex.strip()

    # 1. Remove ```latex ... ``` wrappers first
    wrapper_pattern = r'```(?:[lL]a[tT]e[xX])?\s*(.*?)\s*```'
    wrapper_match = re.search(wrapper_pattern, cleaned_latex, re.DOTALL | re.IGNORECASE)
    if wrapper_match:
        cleaned_latex = wrapper_match.group(1).strip()
        logging.debug("Removed markdown ```latex wrapper.")
    else:
        # If no wrapper, check for common refusal/apology phrases at the start
        refusal_patterns = [
            r"^\s*I cannot fulfill this request",
            r"^\s*I am unable to process",
            r"^\s*Sorry, I can't provide",
            r"^\s*As an AI,"
        ]
        for pattern in refusal_patterns:
            if re.match(pattern, cleaned_latex, re.IGNORECASE):
                logging.warning(f"Potential refusal detected in response: {cleaned_latex[:100]}...")
                return (None, None) if is_page_1 else None

    # 2. Remove \end{document} if present anywhere (shouldn't be, but belt-and-suspenders)
    cleaned_latex = re.sub(r'\\end\{document\}', '', cleaned_latex, flags=re.IGNORECASE).strip()

    # 3. Split or Isolate Content
    preamble = None
    body_content = None

    begin_doc_match = re.search(re.escape(BEGIN_DOCUMENT_TAG), cleaned_latex, re.IGNORECASE)

    if is_page_1:
        if begin_doc_match:
            preamble_end_index = begin_doc_match.start()
            preamble = cleaned_latex[:preamble_end_index].strip()
            # Body includes \begin{document} itself
            body_content = cleaned_latex[preamble_end_index:].strip()

            # Basic validation: Does preamble look like a preamble? Does body start correctly?
            if not preamble or not preamble.startswith("\\documentclass"):
                logging.warning("Page 1: Extracted preamble seems invalid.")
                # Try to recover if body looks okay? For now, treat as failure.
                # return None, None
                # Option: Maybe just return the body if it looks valid?
                if body_content and body_content.startswith(BEGIN_DOCUMENT_TAG):
                     logging.warning("Page 1: Preamble invalid, but body seems okay. Using fallback preamble.")
                     preamble = None # Signal to use fallback later
                     # Keep body_content
                else:
                     return None, None # Both seem invalid

            if not body_content or not body_content.startswith(BEGIN_DOCUMENT_TAG):
                 logging.warning("Page 1: Extracted body seems invalid (doesn't start with \\begin{document}).")
                 return None, None # Treat as failure

            logging.debug("Page 1: Successfully extracted preamble and body.")
            return preamble, body_content
        else:
            logging.warning("Page 1: Could not find '\\begin{document}' in the response.")
            # Maybe the AI *only* returned the body content? Check if it looks like body content.
            if cleaned_latex.startswith("\\maketitle") or cleaned_latex.startswith("\\section"): # Heuristic
                logging.warning("Page 1: Response might be body content only. Using fallback preamble.")
                return None, BEGIN_DOCUMENT_TAG + "\n" + cleaned_latex # Return None for preamble, construct body
            return None, None # Failed to parse

    else: # Subsequent page
        if begin_doc_match:
            # Body includes \begin{document}
            body_content = cleaned_latex[begin_doc_match.start():].strip()
            if not body_content.startswith(BEGIN_DOCUMENT_TAG):
                 logging.warning(f"Page > 1: Extracted body seems invalid (doesn't start with {BEGIN_DOCUMENT_TAG}).")
                 return None
            logging.debug("Page > 1: Successfully extracted body content.")
            return body_content
        else:
            # Maybe the AI *only* returned the content for the *new* page?
            # This deviates from the prompt, but try to handle it.
            logging.warning(f"Page > 1: Did not find '{BEGIN_DOCUMENT_TAG}'. Assuming response is *only* the new page content (requires modification in calling function).")
            # Returning just the cleaned latex - the caller will need to handle this case.
            # This is problematic for the current design. Let's treat failure to return full body as an error.
            logging.error(f"Page > 1: AI response did not contain '{BEGIN_DOCUMENT_TAG}' as expected. Treating as failure.")
            return None # Failed to parse

# --- get_total_pages, get_image_from_page, pil_image_to_bytes, save_image_for_failed_page ---
# (No changes needed in these functions)

def get_total_pages(pdf_path):
    """Get the total number of pages in the PDF."""
    try:
        reader = pypdf.PdfReader(str(pdf_path)) # Ensure path is string
        count = len(reader.pages)
        if count > 0:
            return count
        else:
            logging.error(f"pypdf reported 0 pages for {pdf_path}. Check PDF validity.")
            return 0
    except FileNotFoundError:
         logging.error(f"Input PDF not found: {pdf_path}")
         return -1 # Indicate file not found error
    except Exception as e_pypdf:
        logging.warning(f"pypdf failed to get page count ({e_pypdf}). Trying pdfinfo fallback...")
        try:
            # Fallback using pdfinfo_from_path (requires poppler)
            info = pdfinfo_from_path(str(pdf_path), userpw=None, poppler_path=None)
            pages = info.get('Pages')
            if pages and pages > 0:
                logging.info(f"Obtained page count ({pages}) via pdfinfo.")
                return pages
            else:
                 logging.error(f"pdfinfo could not determine page count or returned 0. Info: {info}")
                 return 0
        except Exception as e_pdfinfo:
             logging.error(f"Error getting page count with pdfinfo fallback: {e_pdfinfo}")
             logging.error("Ensure poppler is installed and in PATH.")
             return 0 # Indicate failure

def get_image_from_page(pdf_path, page_num, dpi=IMAGE_DPI, img_format=IMAGE_FORMAT):
    """Convert a PDF page to an image."""
    logging.debug(f"Converting page {page_num} to image (DPI: {dpi}, Format: {img_format})...")
    try:
        images = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            first_page=page_num,
            last_page=page_num,
            fmt=img_format.lower(),
            thread_count=1,
            timeout=120 # Increased timeout
        )
        if images:
            logging.debug(f"Successfully converted page {page_num} to image.")
            return images[0]
        else:
            logging.error(f"pdf2image returned no image for page {page_num}.")
            return None
    except Exception as e:
        logging.error(f"Error converting page {page_num} to image: {e}")
        if "poppler" in str(e).lower():
             logging.error("Check Poppler installation and PATH. Try specifying poppler_path if needed.")
        return None

def pil_image_to_bytes(image, img_format=IMAGE_FORMAT):
    """Convert PIL Image to bytes and get MIME type."""
    if image is None: return None, None
    byte_arr = BytesIO()
    # Ensure format consistency (e.g., handle JPG/JPEG)
    pillow_format = 'JPEG' if img_format.upper() == 'JPG' else img_format.upper()
    mime_type = f"image/{pillow_format.lower()}"
    try:
        image.save(byte_arr, format=pillow_format)
        img_bytes = byte_arr.getvalue()
        logging.debug(f"Image converted to {mime_type}, size: {len(img_bytes) / 1024:.1f} KB")
        return img_bytes, mime_type
    except Exception as e:
        logging.error(f"Error converting PIL Image to bytes (format: {pillow_format}): {e}")
        return None, None

def save_image_for_failed_page(image, page_num):
    """Save an image of a failed page."""
    if image is None:
        logging.warning(f"Cannot save image for failed page {page_num} - image data is missing.")
        return None
    failed_page_image_path = FAILED_PAGES_DIR / f"failed_page_{page_num}.{IMAGE_FORMAT.lower()}"
    try:
        pillow_format = 'JPEG' if IMAGE_FORMAT.upper() == 'JPG' else IMAGE_FORMAT.upper()
        image.save(failed_page_image_path, pillow_format)
        logging.info(f"Saved image of failed page {page_num} to {failed_page_image_path}")
        return str(failed_page_image_path)
    except Exception as e:
        logging.error(f"Error saving failed page {page_num} image: {e}")
        return None


def get_latex_from_image_with_model(
    page_image: Image.Image,
    page_num: int,
    full_preamble: Optional[str], # Full preamble if page > 1
    full_cumulative_body: Optional[str], # Full body up to previous page if page > 1
    model_info: Dict,
    retry_count: int = 0
) -> Union[Tuple[Optional[str], Optional[str]], Optional[str], Tuple[None, None]]:
    """
    Generate LaTeX using a specific model, providing full context.

    - For Page 1: Returns tuple (preamble, body_content_incl_begin_doc) or (None, None).
    - For Pages > 1: Returns string (updated_body_content_incl_begin_doc) or None.
    - Returns (None, None) or None on failure.
    """
    global model_stats # To record truncation

    provider = model_info["provider"]
    model_name = model_info["name"]
    nickname = model_info["nickname"]
    allow_thinking = model_info.get("thinking", False)
    is_first_page = (page_num == 1)

    # --- Pre-checks ---
    if provider not in api_clients or api_clients[provider] is None: # Check for None too
        logging.error(f"Skipping {nickname}: API client for '{provider}' not available.")
        return (None, None) if is_first_page else None
    if page_image is None:
        logging.error(f"Cannot process page {page_num} with {nickname}: page_image is None.")
        return (None, None) if is_first_page else None

    image_bytes, mime_type = pil_image_to_bytes(page_image, IMAGE_FORMAT)
    if not image_bytes or not mime_type:
        logging.error(f"Failed to convert image for page {page_num} to bytes.")
        return (None, None) if is_first_page else None

    # --- Prompt Engineering ---
    prompt_parts = [
        f"# Task: LaTeX Translation (Page {page_num})"
    ]
    context_description = ""
    truncated_context = False

    if is_first_page:
        prompt_parts.extend([
            "Analyze the provided image, which is the **first page** of a non-English academic document.",
            "Generate the **complete** English LaTeX code for this document, including:",
            "1.  A suitable LaTeX **Preamble** (documentclass, essential packages like inputenc, fontenc, amsmath, graphicx, hyperref, etc., theorem environments if needed).",
            "2.  Document **Metadata** (\\title, \\author, \\date based on image, placeholders if unclear).",
            f"3.  The document body starting with **`{BEGIN_DOCUMENT_TAG}`** and including `\\maketitle` (if applicable).",
            f"4.  The translated LaTeX for all content visible on this first page image.",
            f"**CRUCIAL:** Your response must contain *both* the preamble and the body (starting with `{BEGIN_DOCUMENT_TAG}`). Do **NOT** include the final `{END_DOCUMENT_TAG}` tag."
        ])
        context_description = "This is the first page. Generate full document preamble and body for this page."
    else: # Subsequent page
        prompt_parts.extend([
            f"Analyze the provided image (Page {page_num}) and the existing LaTeX document context.",
            f"The goal is to **append** the translated English LaTeX content of the current page image to the existing document body.",
            "\n# Provided Context:",
            "```latex",
            "%%% PREAMBLE START %%%",
            full_preamble.strip() if full_preamble else "% ERROR: Preamble was missing!",
            "%%% PREAMBLE END %%%",
            "",
            "%%% EXISTING BODY CONTENT START %%%",
            # Truncate body context if it's too long
            full_cumulative_body.strip() if full_cumulative_body else f"% No previous body content provided (Starting after {BEGIN_DOCUMENT_TAG})",
             "%%% EXISTING BODY CONTENT END %%%",
            "```",
        ])

        # --- Context Truncation Logic ---
        # Estimate token count (very rough: 1 char ~ 1 token, plus some overhead)
        estimated_context_len = len(full_preamble or "") + len(full_cumulative_body or "")
        if estimated_context_len > CONTEXT_TRUNCATION_THRESHOLD_TOKENS:
            truncated_context = True
            original_len = len(full_cumulative_body or "")
            # Keep the end of the body content
            truncated_body = (full_cumulative_body or "")[-CONTEXT_TRUNCATION_THRESHOLD_TOKENS:]
            # Try to find the last sensible break point (e.g., \section, \subsection, paragraph break)
            # This is complex; for now, just truncate by character count.
            # Rebuild the context prompt parts with truncated body
            prompt_parts[-3] = f"% Note: Previous body content truncated due to length (showing last ~{CONTEXT_TRUNCATION_THRESHOLD_TOKENS} chars)." + "\n" + truncated_body.strip()

            context_description = f"Existing preamble and TRUNCATED previous body provided (original body len {original_len}, truncated to ~{CONTEXT_TRUNCATION_THRESHOLD_TOKENS}). Append content for page {page_num}."
            logging.warning(f"Page {page_num}: Cumulative body content length ({original_len}) > threshold ({CONTEXT_TRUNCATION_THRESHOLD_TOKENS}). Truncating context sent to AI.")
            # Record truncation
            page_str = str(page_num)
            if page_str not in model_stats["context_truncations"]: model_stats["context_truncations"][page_str] = []
            model_stats["context_truncations"][page_str].append({"original_len": original_len, "truncated_len": CONTEXT_TRUNCATION_THRESHOLD_TOKENS, "timestamp": datetime.now().isoformat()})
            save_model_stats() # Save stats when truncation happens
        else:
            context_description = f"Existing preamble and full previous body provided (body len {len(full_cumulative_body or '')}). Append content for page {page_num}."


        prompt_parts.extend([
             "\n# Instructions for Page {page_num}:",
             "- Analyze the image for this page.",
             "- Translate its content to English LaTeX.",
             "- **Append** this new LaTeX content seamlessly to the end of the `%%% EXISTING BODY CONTENT END %%%` provided above.",
             "- Ensure mathematical accuracy, structural consistency (sections, theorems etc.), and correct LaTeX syntax.",
             f"- **CRUCIAL:** Your response must be the **complete, updated body content**, starting from `{BEGIN_DOCUMENT_TAG}` and including *all* previous content plus the newly added content for page {page_num}.",
             f"- Do **NOT** include the preamble or the final `{END_DOCUMENT_TAG}` tag in your response.",
             "- Represent figures/tables with standard environments and placeholders."
        ])

    # --- Thinking Prompt & Response Format ---
    if allow_thinking:
         prompt_parts.extend([
            "\n# Reasoning Step (Optional):",
            "1. Briefly analyze the new page's content and how it connects to the previous context.",
            "2. Mention any specific LaTeX environments needed for the new page.",
            "\n# Final LaTeX Output:",
            "After reasoning, provide the final LaTeX output in a single block:",
            "```latex",
            f"[Preamble + {BEGIN_DOCUMENT_TAG} + ... Page 1 Content ...]" if is_first_page else f"[{BEGIN_DOCUMENT_TAG} + ... Previous Content ... + ... Page {page_num} Content ...]",
            "```"
        ])
    else: # Direct format for non-thinking models
        prompt_parts.extend([
             "\n# Response Format:",
             f"Respond ONLY with the raw LaTeX code as specified in the instructions (preamble + body for page 1; full updated body for page > 1). Do not include explanations or markdown formatting like ```."
        ])

    prompt = "\n".join(prompt_parts)
    logging.debug(f"Prompt for Page {page_num}:\n{context_description}\n--- Start of Prompt Snippet ---\n{prompt[:500]}...\n--- End of Prompt Snippet ---")


    # --- API Call ---
    response_text = None
    success = False
    start_time = time.time()
    api_error = None

    try:
        logging.info(f"Sending Page {page_num} to {nickname} ({model_name}) [Retry {retry_count}/{MAX_RETRIES_PER_MODEL}]...")

        # === API Call Logic (Claude, GPT, Gemini) - No changes needed here ===
        # (Assuming the API call implementations themselves are correct)
        if provider == "claude":
            client = api_clients["claude"]
            message_list = [{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": base64.b64encode(image_bytes).decode("utf-8")}}, {"type": "text", "text": prompt}]}]
            response = client.messages.create(model=model_name, max_tokens=4096, temperature=0.15, messages=message_list, timeout=240.0) # Increased timeout
            if response.content and isinstance(response.content, list) and hasattr(response.content[0], 'text'):
                response_text = response.content[0].text
                success = True
            else: api_error = f"Unexpected response structure: {response}"

        elif provider == "gpt":
            client = api_clients["gpt"]
            response = client.chat.completions.create(model=model_name, messages=[{"role": "system", "content": "You are an expert LaTeX translator. Follow the instructions precisely regarding context and output format."}, {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('utf-8')}"}}]}], max_tokens=4096, temperature=0.15, timeout=240.0) # Increased timeout
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                response_text = response.choices[0].message.content
                success = True
            else: api_error = f"Unexpected response structure: {response}"

        elif provider == "gemini":
            genai_module = api_clients["gemini"]
            safety_settings = [{"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
            # Use the specific model name from model_info
            gemini_model_instance = genai_module.GenerativeModel(
                model_info['name'], # Use the specific name like 'gemini-1.5-pro-latest'
                safety_settings=safety_settings,
                generation_config={"temperature": 0.15, "max_output_tokens": 8192} # Gemini often supports larger outputs
            )
            content_parts = [prompt, {"mime_type": mime_type, "data": image_bytes}]
            try:
                response = gemini_model_instance.generate_content(content_parts, request_options={"timeout": 240}) # Increased timeout
                if response.parts:
                     response_text = "".join(part.text for part in response.parts if hasattr(part, "text"))
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                     api_error = f"Blocked by API. Reason: {response.prompt_feedback.block_reason}. Ratings: {response.prompt_feedback.safety_ratings}"
                     response_text = None
                elif response_text:
                     success = True
                else:
                     api_error = f"Empty response content. Prompt Feedback: {response.prompt_feedback}"

            except ValueError as e:
                 # This often indicates blocking or invalid response structure
                 api_error = f"Content blocked or invalid response format. Error: {e}. Feedback: {getattr(response, 'prompt_feedback', 'N/A')}"
                 success = False
                 response_text = None
                 # Specific check for context window exceeded errors (adjust pattern as needed)
                 if "token limit" in str(e).lower() or "size limit" in str(e).lower():
                      logging.error(f"Potential Context Window Exceeded error for page {page_num} with Gemini: {e}")
                      api_error = f"Context Window Exceeded? Error: {e}"
            except Exception as e:
                 api_error = f"Gemini API call failed: {e}"
                 success = False
                 response_text = None
                 if "400" in str(e) and "token" in str(e).lower(): # Catching generic 400 errors that might be token limits
                      logging.error(f"Potential Context Window Exceeded error (400 Bad Request) for page {page_num} with Gemini: {e}")
                      api_error = f"Context Window Exceeded (400)? Error: {e}"


        elapsed_time = time.time() - start_time
        if success and response_text:
            logging.info(f"Received response for Page {page_num} from {nickname} in {elapsed_time:.2f}s.")
            # --- Extract Parts ---
            extracted_parts = extract_latex_parts_from_response(response_text, is_first_page)

            if is_first_page:
                preamble, body = extracted_parts
                if body: # Check if body extraction was successful (preamble might be None -> fallback)
                    logging.info(f"Page 1: Extracted body successfully. Preamble: {'Found' if preamble else 'Fallback Needed'}.")
                    return preamble, body # Return tuple
                else:
                    logging.warning(f"Failed to extract valid LaTeX parts from Page 1 response from {nickname}.")
                    return None, None # Indicate failure
            else: # Subsequent page
                body = extracted_parts
                if body:
                    # Sanity check: Does the returned body *contain* some part of the input body?
                    if full_cumulative_body and len(full_cumulative_body) > 100:
                        snippet = full_cumulative_body[-100:] # Check if end of previous body is present
                        if snippet not in body[-150:]: # Allow for slight changes/additions at the very end
                           logging.warning(f"Page > 1: Returned body from {nickname} might not contain the end of the previous content. Check consistency.")
                    elif not full_cumulative_body and not body.strip().startswith(BEGIN_DOCUMENT_TAG):
                         logging.warning(f"Page > 1: No previous body and returned content doesn't start with {BEGIN_DOCUMENT_TAG}.")
                         # If no previous body, it should *at least* start with \begin{document}
                         return None # Treat as failure

                    logging.info(f"Page > 1: Extracted body successfully.")
                    return body # Return string
                else:
                    logging.warning(f"Failed to extract valid LaTeX body from Page > 1 response from {nickname}.")
                    return None # Indicate failure

        else:
            if api_error: logging.warning(f"API response issue for page {page_num} from {nickname} (Retry {retry_count}): {api_error}")
            else: logging.warning(f"Empty or invalid response content for page {page_num} from {nickname} (Retry {retry_count}).")
            return (None, None) if is_first_page else None # Return failure matching expected type

    except anthropic.APIError as e:
        elapsed_time = time.time() - start_time
        logging.error(f"Anthropic API Error (Page {page_num}, Retry {retry_count}, {elapsed_time:.2f}s): {e}", exc_info=False)
        if "context_length_exceeded" in str(e):
             logging.error(f"CONTEXT LENGTH EXCEEDED for page {page_num}.")
             # Consider stopping or adding specific handling
        return (None, None) if is_first_page else None
    except openai.APIError as e:
         elapsed_time = time.time() - start_time
         logging.error(f"OpenAI API Error (Page {page_num}, Retry {retry_count}, {elapsed_time:.2f}s): {e}", exc_info=False)
         if "context_length_exceeded" in str(e):
             logging.error(f"CONTEXT LENGTH EXCEEDED for page {page_num}.")
         return (None, None) if is_first_page else None
    # Add specific Gemini error catching if library provides distinct types
    except Exception as e: # Catch-all for other errors (network, timeouts, etc.)
        elapsed_time = time.time() - start_time
        # Check for common context length error messages in the generic exception text
        err_str = str(e).lower()
        if "context length" in err_str or "token limit" in err_str or "request payload size" in err_str:
             logging.error(f"CONTEXT LENGTH EXCEEDED (detected in generic exception) for page {page_num}.")
             logging.error(f"API Exception with {nickname} (Retry {retry_count}, {elapsed_time:.2f}s): {e}", exc_info=False)
        else:
             logging.error(f"API Exception with {nickname} for page {page_num} (Retry {retry_count}, {elapsed_time:.2f}s): {e}", exc_info=False)
        return (None, None) if is_first_page else None


def compile_latex(tex_filepath, aux_dir, num_runs=2, timeout=120):
    """Compile a LaTeX file using pdflatex and return success status and log."""
    tex_filepath = Path(tex_filepath)
    if not tex_filepath.is_file():
        return False, f"Error: LaTeX source file not found: {tex_filepath}"

    aux_dir = Path(aux_dir)
    aux_dir.mkdir(parents=True, exist_ok=True)
    base_name = tex_filepath.stem
    log_file = aux_dir / f"{base_name}.log"
    pdf_file = aux_dir / f"{base_name}.pdf"
    aux_output_dir = aux_dir / f"{base_name}_aux_files" # Subdir for aux files
    aux_output_dir.mkdir(parents=True, exist_ok=True)


    # Command configuration
    command = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-file-line-error",
        "-halt-on-error",
        f"-output-directory={str(aux_dir.resolve())}", # Main PDF output dir
        f"-aux-directory={str(aux_output_dir.resolve())}", # Specific aux dir
        str(tex_filepath.resolve())
    ]

    full_log = f"--- Compiling: {tex_filepath.name} in {aux_dir.name} ---\n"
    success = False
    last_log_content = "" # Store log content from the final pass

    for i in range(num_runs):
        pass_log = f"\n--- Pass {i+1}/{num_runs} ---\nRun: {' '.join(command)}\n"
        full_log += pass_log
        log_content = "" # Reset log content for the pass

        try:
            # Clear log file before the run? Maybe not, let it append.
            # If PDF exists from previous pass, keep it unless this pass fails badly.
            if pdf_file.exists() and i > 0:
                pdf_exists_before_pass = True
            else:
                pdf_exists_before_pass = False

            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=timeout,
                check=False
            )

            full_log += f"Return Code: {process.returncode}\n"
            stdout_snippet = process.stdout[-1000:].strip()
            stderr_snippet = process.stderr.strip()
            if stdout_snippet: full_log += f"Stdout (last 1k):\n{stdout_snippet}\n...\n"
            if stderr_snippet: full_log += f"Stderr:\n{stderr_snippet}\n"

            # Read log file content
            if log_file.exists():
                try:
                    log_content = log_file.read_text(encoding='utf-8', errors='replace')
                    last_log_content = log_content # Update last known log
                    # Check for fatal errors aggressively
                    # Added more patterns based on common LaTeX failures
                    fatal_error_patterns = [
                        r"Fatal error occurred", r"Emergency stop", r"! LaTeX Error:",
                        r"Undefined control sequence", r"Missing \S+ inserted", r"Misplaced alignment tab character",
                        r"Runaway argument", r"Paragraph ended before \S+ was complete", r"Extra \}", r"Too many \}"
                        r"\\begin\{document\} ended by \\end\{(.+?)\}", # Specific error from user log
                        r"\\begin\{(.+?)\} on input line \d+ ended by \\end\{(.+?)\}", # Mismatched envs
                        ]
                    # Check the *whole* log on the final pass, or the end on intermediate passes
                    check_area = log_content if i == num_runs - 1 else log_content[-5000:]
                    errors_found = [pat for pat in fatal_error_patterns if re.search(pat, check_area)]

                    if errors_found:
                        full_log += f"\n--- LaTeX Error detected in log file (Pass {i+1}): {errors_found} ---\n"
                        success = False
                        # Try to extract specific error line from log
                        match = re.search(r"^! LaTeX Error:.*?\n.*?l\.(\d+)", log_content, re.MULTILINE)
                        if not match:
                            match = re.search(r"^! Undefined control sequence.*?\n.*?l\.(\d+)", log_content, re.MULTILINE)
                        if not match: # Check for the specific mismatched environment error
                             match = re.search(r"\\begin\{(.+?)\} on input line (\d+) ended by \\end\{(.+?)\}", log_content)
                             if match:
                                 full_log += f"Mismatch Detail: Env '{match.group(1)}' on line {match.group(2)} ended by '{match.group(3)}'\n"
                        if match and len(match.groups()) >= 1:
                            line_num = match.group(1) if len(match.groups()) == 1 else match.group(2) # Adjust based on regex
                            try:
                                error_line = int(line_num)
                                context_lines = 5
                                source_lines = tex_filepath.read_text(encoding='utf-8').splitlines()
                                start = max(0, error_line - context_lines -1)
                                end = min(len(source_lines), error_line + context_lines)
                                full_log += f"--- Source Context Around Line {error_line} ---\n"
                                for k, line_text in enumerate(source_lines[start:end], start=start + 1):
                                     prefix = ">>" if k == error_line else "  "
                                     full_log += f"{prefix}{k:5d}: {line_text}\n"
                                full_log += "--------------------------------------\n"
                            except Exception as e_context:
                                 full_log += f"(Could not extract source context: {e_context})\n"

                        break # Stop on fatal error

                except Exception as log_read_e:
                    full_log += f"\n--- Error reading log file {log_file}: {log_read_e} ---\n"
                    success = False
                    # If log can't be read, assume failure, especially if pdflatex exited non-zero
                    if process.returncode != 0:
                        break

            else: # Log file doesn't exist
                 full_log += f"\n--- Log file {log_file.name} not found after pass {i+1}. Compilation likely failed early. ---\n"
                 success = False
                 if process.returncode != 0: # If command failed and no log, definitely stop
                      break

            # Intermediate pass check: If pdflatex exited non-zero, maybe stop?
            # Let's allow it to continue to the next pass for now, as sometimes errors resolve.
            # if process.returncode != 0 and i < num_runs - 1:
            #    full_log += f"--- Warning: pdflatex exited with code {process.returncode} on pass {i+1}, continuing... ---\n"

        except FileNotFoundError:
            full_log += "\nError: 'pdflatex' command not found. Is LaTeX installed and in PATH?\n"; success = False; break
        except subprocess.TimeoutExpired:
            full_log += f"\nError: pdflatex timed out after {timeout} seconds (Pass {i+1}).\n"; success = False; break
        except Exception as e:
            full_log += f"\nError running pdflatex (Pass {i+1}): {e}\n"; success = False; break

        # After the loop completes *or* breaks due to error
        if i == num_runs - 1 and not errors_found: # If last pass finished without detecting errors
             if pdf_file.is_file() and pdf_file.stat().st_size > 100: # Basic PDF validity check
                  success = True
                  full_log += f"\n--- Compilation successful after {num_runs} passes (PDF found). ---\n"
             else:
                  full_log += f"\n--- Compilation finished, but PDF file missing or empty after {num_runs} passes. ---\n"
                  success = False
                  # Add log snippet if available
                  if last_log_content:
                       full_log += f"--- Final Log Snippet (tail):\n{last_log_content[-1000:]}\n---\n"


    if not success:
         # Provide more info if compilation failed
         fail_log_path = WORKING_DIR / f"compile_fail_{tex_filepath.stem}.log"
         if not any(err in full_log for err in ["Error detected", "failed early", "timed out", "command not found"]):
              full_log += "\n--- Compilation failed (Reason unclear or occurred late in process. Check full log). ---"
         if last_log_content and "Error detected" not in full_log : # Add log snippet if not already added via error detection
              full_log += f"\n--- Final Log Snippet (tail):\n{last_log_content[-1000:]}\n---\n"
         try:
              fail_log_path.write_text(full_log, encoding='utf-8')
              logging.error(f"Compilation failed. Detailed log saved to: {fail_log_path.name}")
         except Exception as e_write:
              logging.error(f"Could not write detailed failure log to {fail_log_path}: {e_write}")
              logging.error("--- Failure Log Snippet ---\n" + full_log[-1500:] + "\n--- End Snippet ---")


    # Cleanup aux dir? Maybe not, could be useful for debugging.
    # shutil.rmtree(aux_output_dir, ignore_errors=True)

    return success, full_log


def record_model_success(page_num, model_info, retry):
    """Record successful translation stats."""
    page_str = str(page_num)
    if page_str not in model_stats["pages_translated"]:
        model_stats["pages_translated"][page_str] = []
    model_stats["pages_translated"][page_str].append({
        "model": model_info["nickname"],
        "model_name": model_info["name"],
        "provider": model_info["provider"],
        # "cycle": cycle, # Removed cycle
        "retry": retry,
        "timestamp": datetime.now().isoformat()
    })
    # save_model_stats() # Save less frequently

def record_failed_page(page_num, reason, saved_image_path):
    """Record a failed page in the model stats."""
    page_str = str(page_num)
    if "failures" not in model_stats: model_stats["failures"] = {}
    if page_str not in model_stats["failures"]: model_stats["failures"][page_str] = []
    # Avoid duplicate entries for the same reason if called multiple times
    existing_reasons = [f.get("reason") for f in model_stats["failures"][page_str]]
    if reason not in existing_reasons:
        model_stats["failures"][page_str].append({
            "reason": reason,
            "saved_image": saved_image_path,
            "timestamp": datetime.now().isoformat()
        })
        logging.info(f"Recorded failure for page {page_num}: {reason}")
        save_model_stats() # Save stats on failure

def save_model_stats():
    """Save the model stats to a file."""
    try:
        # Backup before writing
        if MODEL_STATS_FILE.exists():
            backup_file = MODEL_STATS_FILE.with_suffix(".json.bak")
            shutil.copy2(MODEL_STATS_FILE, backup_file)

        with open(MODEL_STATS_FILE, 'w', encoding='utf-8') as f:
            # Use default=str to handle potential non-serializable types like datetime
            json.dump(model_stats, f, indent=2, default=str)
        logging.debug(f"Model stats saved to {MODEL_STATS_FILE}")
    except Exception as e:
        logging.error(f"Error saving model stats to {MODEL_STATS_FILE}: {e}")

def record_model_compilation_failure(page_num, model_info, retry, reason):
    """Record a model's failure due to compilation error."""
    # Could add more detail to model_stats if needed
    logging.warning(f"Recorded compilation failure for page {page_num} by {model_info['nickname']} (Retry {retry}): {reason}")
    pass # Keep it simple for now


def create_failure_placeholder(page_num, failure_type, reason, image_path=None):
    """Generates a LaTeX placeholder for a failed page."""
    # Escape LaTeX special characters in the reason string
    reason_escaped = reason.replace('\\', r'\textbackslash{}') # Must be first
    reason_escaped = reason_escaped.replace('&', r'\&')
    reason_escaped = reason_escaped.replace('%', r'\%')
    reason_escaped = reason_escaped.replace('$', r'\$')
    reason_escaped = reason_escaped.replace('#', r'\#')
    reason_escaped = reason_escaped.replace('_', r'\_')
    reason_escaped = reason_escaped.replace('{', r'\{')
    reason_escaped = reason_escaped.replace('}', r'\}')
    reason_escaped = reason_escaped.replace('~', r'\textasciitilde{}')
    reason_escaped = reason_escaped.replace('^', r'\textasciicircum{}')

    image_info = ""
    if image_path:
        img_filename = Path(image_path).name
        img_filename_escaped = img_filename.replace('_', '\\_')
        image_info = f"Original page image saved for review:\\\\ \\texttt{{{img_filename_escaped}}}"
    else:
        image_info = "(Original page image could not be saved or was missing)."

    # Use texttt for reason for better rendering of escaped chars
    placeholder = f"""
% ========== PAGE {page_num}: {failure_type} ==========
\\newpage % More reliable than clearpage inside document sometimes
\\vspace*{{1cm}} % Add some space at the top
\\begin{{center}}
\\fbox{{
    \\begin{{minipage}}{{0.85\\textwidth}} % Use minipage for better text wrapping
    \\centering
    \\Large\\textbf{{Page {page_num}: Translation/Compilation Failed}}\\\\[1em]
    \\normalsize Reason: \\textcolor{{red}}{{\\texttt{{{reason_escaped}}}}}\\\\[1em]
    \\normalsize {image_info}
    \\end{{minipage}}
}}
\\vfill % Push to center vertically if page is empty otherwise
\\end{{center}}
\\newpage
% ========== END OF PAGE {page_num} PLACEHOLDER ==========
"""
    return placeholder


# --- Modified Processing Function ---
def get_latex_with_single_model(
    page_num: int,
    page_image: Optional[Image.Image],
    current_doc_preamble: str, # <<< CHANGE THIS PARAMETER NAME if it was 'current_preamble'
    cumulative_body_content_up_to_previous_page: str
) -> Optional[str]:
    """
    Tries to get LaTeX using the TARGET_MODEL_INFO, providing full context,
    with retries on API/compile failure.

    Returns:
        - The *full*, successfully compiled body content (including the new page) as a string.
        - A placeholder string if the page failed but the placeholder compiles.
        - None if a critical error occurs (e.g., placeholder doesn't compile, client missing).

    Side effect: May update the global `current_preamble` if page 1 is processed successfully.
    """
    global TARGET_MODEL_INFO, fallback_preamble, current_preamble 

    if page_image is None:
        logging.error(f"Process skipped for page {page_num}: page_image is None.")
        record_failed_page(page_num, "image_extraction_failed", None)
        placeholder_text = create_failure_placeholder(page_num, "Image Missing", "Image could not be extracted/provided.")
        # Compile check for placeholder needed here
        placeholder_full_body = cumulative_body_content_up_to_previous_page.strip() + "\n\n" + placeholder_text
        placeholder_tex_to_compile = current_preamble + "\n" + placeholder_full_body + "\n" + END_DOCUMENT_TAG + "\n"
        try: CURRENT_ATTEMPT_TEX_FILE.write_text(placeholder_tex_to_compile, encoding='utf-8')
        except Exception: pass # Ignore write error, compile check will fail
        placeholder_ok, _ = compile_latex(CURRENT_ATTEMPT_TEX_FILE, LATEX_AUX_DIR)
        return placeholder_full_body if placeholder_ok else None


    if not current_preamble and page_num > 1: # Preamble must exist after page 1
        logging.error(f"CRITICAL: Cannot process page {page_num}, preamble is missing.")
        return None
    if not TARGET_MODEL_INFO:
        logging.error(f"CRITICAL: TARGET_MODEL_INFO not set. Cannot proceed with page {page_num}.")
        return None

    # Check client availability
    provider = TARGET_MODEL_INFO["provider"]
    if provider not in api_clients or api_clients[provider] is None:
        logging.critical(f"CRITICAL: API Client for provider '{provider}' of target model {TARGET_MODEL_INFO['nickname']} not available. Stopping.")
        record_failed_page(page_num, f"api_client_missing_for_{provider}", None)
        return None

    saved_image_path = None
    is_first_page = (page_num == 1)

    for attempt_num in range(1, MAX_RETRIES_PER_MODEL + 1):
        logging.info(f"--- Attempt {attempt_num}/{MAX_RETRIES_PER_MODEL} for Page {page_num} using {TARGET_MODEL_INFO['nickname']} ---")
        api_result = None
        compile_error = False
        api_call_error = False
        extracted_preamble = None # Specific to page 1
        extracted_body = None

        try:
            # 1. Get LaTeX from API (provides full context internally now)
            api_result = get_latex_from_image_with_model(
                page_image,
                page_num,
                current_preamble, # Pass current preamble
                cumulative_body_content_up_to_previous_page, # Pass previous body
                TARGET_MODEL_INFO,
                retry_count=attempt_num
            )

            # --- Process API Result ---
            if api_result is not None:
                 # Determine Preamble and Body to Compile
                 preamble_to_compile = current_preamble # Default
                 body_to_compile = None

                 if is_first_page:
                      if isinstance(api_result, tuple) and len(api_result) == 2:
                           extracted_preamble, extracted_body = api_result
                           if extracted_body: # Body is mandatory
                               if extracted_preamble:
                                    preamble_to_compile = extracted_preamble # Use AI preamble
                               else:
                                    logging.warning("Page 1: AI did not provide preamble or it was invalid, using fallback.")
                                    preamble_to_compile = fallback_preamble.strip() # Use fallback
                               body_to_compile = extracted_body
                           else:
                               logging.warning(f"Page 1: API succeeded but body extraction failed (Attempt {attempt_num}).")
                               api_call_error = True # Treat as API failure for retry logic
                      else:
                           logging.error(f"Page 1: API returned unexpected type: {type(api_result)} (Attempt {attempt_num}).")
                           api_call_error = True
                 else: # Subsequent page
                      if isinstance(api_result, str):
                           # api_result is the full updated body from \begin{document} onwards
                           extracted_body = api_result
                           body_to_compile = extracted_body
                           preamble_to_compile = current_preamble # Should already be set
                      else:
                           logging.error(f"Page > 1: API returned unexpected type: {type(api_result)} (Attempt {attempt_num}).")
                           api_call_error = True

                 # --- Compile Check ---
                 if body_to_compile and preamble_to_compile: # Only proceed if we have parts to compile
                      logging.info(f"API Success! Content received for Page {page_num} from {TARGET_MODEL_INFO['nickname']} (Attempt {attempt_num}). Attempting compilation...")

                      tex_content_to_compile = (
                          preamble_to_compile.strip() + "\n" +
                          body_to_compile.strip() + "\n" + # This already includes \begin{document}
                          END_DOCUMENT_TAG + "\n"
                      )

                      # Write attempt to file
                      try:
                          CURRENT_ATTEMPT_TEX_FILE.write_text(tex_content_to_compile, encoding='utf-8')
                          logging.debug(f"Wrote {len(tex_content_to_compile)} chars to {CURRENT_ATTEMPT_TEX_FILE.name} for compilation check.")
                      except Exception as write_e:
                          logging.error(f"CRITICAL: Failed to write temp file for compilation check: {write_e}. Treating as model failure.")
                          record_model_compilation_failure(page_num, TARGET_MODEL_INFO, attempt_num, f"File write error: {write_e}")
                          compile_error = True
                          # Continue to sleep/retry logic below

                      if not compile_error:
                          # Compile the attempt
                          compile_ok, compile_log = compile_latex(CURRENT_ATTEMPT_TEX_FILE, LATEX_AUX_DIR)

                          if compile_ok:
                              logging.info(f"Compilation successful for page {page_num} using {TARGET_MODEL_INFO['nickname']} (Attempt {attempt_num}).")
                              record_model_success(page_num, TARGET_MODEL_INFO, attempt_num)

                              # Update global preamble if it's page 1 and AI provided one
                              if is_first_page and extracted_preamble:
                                  logging.info("Page 1: Updating global preamble with AI generated version.")
                                  global_preamble_ref = preamble_to_compile # Update the global reference
                              elif is_first_page and not extracted_preamble:
                                   logging.info("Page 1: Using fallback preamble.")
                                   global_preamble_ref = preamble_to_compile # Ensure global matches fallback

                              # Return the successfully compiled body content
                              return extracted_body # Return the full body string

                          else:
                              # Compilation failed
                              logging.warning(f"Compilation FAILED for page {page_num} using output from {TARGET_MODEL_INFO['nickname']} (Attempt {attempt_num}).")
                              record_model_compilation_failure(page_num, TARGET_MODEL_INFO, attempt_num, "LaTeX compilation error")
                              compile_error = True
                              extracted_body = None # Discard body if compilation failed
                              # Fall through to sleep/retry logic
                 else:
                      # This path is taken if api_result was valid but extraction failed above
                      api_call_error = True # Mark as failure for retry

            else: # API call failed (api_result is None or (None, None))
                 logging.warning(f"{TARGET_MODEL_INFO['nickname']} (Attempt {attempt_num}) failed API call or content extraction for page {page_num}.")
                 api_call_error = True
                 # Fall through to sleep/retry logic

        except Exception as e:
            # Catch unexpected errors during the model call or compilation check
            logging.error(f"Critical error during processing with {TARGET_MODEL_INFO['nickname']} for page {page_num} (Attempt {attempt_num}): {e}", exc_info=True)
            api_call_error = True # Flag it generally as non-compile failure for retry logic
            extracted_body = None # Discard any potentially extracted body

        # --- Retry Logic ---
        if extracted_body is None and attempt_num < MAX_RETRIES_PER_MODEL: # Check if body is None (indicates failure)
             sleep_duration = random.uniform(2, 5)
             failure_type = "API/Extraction" if api_call_error else "Compilation/Write"
             logging.info(f"Sleeping {sleep_duration:.1f}s before retrying {TARGET_MODEL_INFO['nickname']} after {failure_type} failure...")
             time.sleep(sleep_duration)
        elif extracted_body is None: # Failed on the last attempt
            logging.error(f"All {MAX_RETRIES_PER_MODEL} attempts failed for page {page_num} with {TARGET_MODEL_INFO['nickname']}.")
            break # Exit retry loop


    # --- All Retries Failed or Critical Error Occurred ---
    if extracted_body is None: # Check final status
        logging.error(f"Single model {TARGET_MODEL_INFO['nickname']} failed to produce compilable LaTeX for page {page_num} after {MAX_RETRIES_PER_MODEL} attempts.")
        if saved_image_path is None: # Save image only if not saved before
             saved_image_path = save_image_for_failed_page(page_image, page_num)
        record_failed_page(page_num, f"single_model_all_retries_failed ({TARGET_MODEL_INFO['nickname']})", saved_image_path)

        # --- Final Attempt: Compile with Placeholder ---
        logging.info(f"Attempting final compilation for page {page_num} with placeholder...")
        placeholder_latex = create_failure_placeholder(page_num, "Translation/Compilation Failed", f"{TARGET_MODEL_INFO['nickname']} failed after {MAX_RETRIES_PER_MODEL} attempts.", saved_image_path)
        # Construct the full body including the placeholder
        placeholder_full_body = cumulative_body_content_up_to_previous_page.strip() + "\n\n" + placeholder_latex

        placeholder_tex_to_compile = (
            current_preamble + "\n" + # Use the preamble we started with
            placeholder_full_body.strip() + "\n" +
            END_DOCUMENT_TAG + "\n"
        )
        try:
            CURRENT_ATTEMPT_TEX_FILE.write_text(placeholder_tex_to_compile, encoding='utf-8')
        except Exception as write_e:
            logging.error(f"CRITICAL: Failed to write placeholder attempt file: {write_e}. Stopping.")
            return None # Indicate critical failure

        placeholder_ok, ph_compile_log = compile_latex(CURRENT_ATTEMPT_TEX_FILE, LATEX_AUX_DIR)

        if placeholder_ok:
            logging.info(f"Page {page_num}: Successfully compiled WITH PLACEHOLDER.")
            # Return the full body content including the placeholder
            return placeholder_full_body
        else:
            logging.error(f"CRITICAL: Failed to compile even with placeholder for page {page_num}. Preamble ('{current_preamble[:50]}...') or previous content likely has unrecoverable LaTeX error.")
            logging.error("Stopping processing.")
            record_failed_page(page_num, "compilation_failed_with_placeholder_FATAL", saved_image_path)
            fatal_log_path = WORKING_DIR / f"compile_fatal_placeholder_page_{page_num}.log"
            try: fatal_log_path.write_text(ph_compile_log, encoding='utf-8')
            except Exception: pass
            return None # Indicate critical failure

    # Should not be reached if logic is correct, but as a safeguard:
    logging.error("Reached unexpected end of get_latex_with_single_model function.")
    return None


# --- Main Execution Logic ---
def main(input_pdf_cli=None, preferred_model_cli=None):
    # Need global reference to update preamble from within get_latex_with_single_model
    global TARGET_MODEL_INFO, fallback_preamble, CURRENT_ATTEMPT_TEX_FILE, LATEX_AUX_DIR
    global LAST_GOOD_TEX_FILE, LAST_GOOD_PDF_FILE, END_DOCUMENT_TAG, BEGIN_DOCUMENT_TAG
    global model_stats, current_preamble # Make current_preamble global

    # --- Setup ---
    start_time_main = time.time()
    logging.info("="*60)
    logging.info(f"--- Starting PDF to LaTeX Conversion (Single Model, Full Context) ---")
    logging.info(f"Timestamp: {datetime.now().isoformat()}")
    logging.info("="*60)

    INPUT_PDF_PATH_STR = input_pdf_cli if input_pdf_cli else INPUT_PDF_PATH
    pdf_path = Path(INPUT_PDF_PATH_STR).resolve()
    logging.info(f"Input PDF: {pdf_path}")
    logging.info(f"Working Directory: {WORKING_DIR.resolve()}")
    logging.warning("Using FULL CONTEXT mode. This is resource-intensive and may fail on long documents due to AI context limits.")

    if not setup_single_model(preferred_model_cli):
         return

    # Double check client availability after setup
    if not TARGET_MODEL_INFO or TARGET_MODEL_INFO['provider'] not in api_clients or api_clients[TARGET_MODEL_INFO['provider']] is None:
        logging.critical(f"CRITICAL ERROR: API Client for provider '{TARGET_MODEL_INFO['provider'] if TARGET_MODEL_INFO else 'N/A'}' for model '{TARGET_MODEL_INFO['nickname'] if TARGET_MODEL_INFO else 'N/A'}' is not available.")
        logging.critical("Ensure the corresponding API key is set and client initialized.")
        # Consider creating an error PDF here too
        return
    else:
        logging.info(f"API Client for {TARGET_MODEL_INFO['provider']} confirmed available.")

    # Load existing stats
    if MODEL_STATS_FILE.exists():
        try:
            with open(MODEL_STATS_FILE, 'r', encoding='utf-8') as f: model_stats = json.load(f)
            logging.info("Loaded existing model statistics.")
        except Exception as e: logging.error(f"Could not load model stats from {MODEL_STATS_FILE}: {e}.")
        # Ensure required keys exist
        model_stats.setdefault("pages_translated", {})
        model_stats.setdefault("failures", {})
        model_stats.setdefault("context_truncations", {})

    # Validate Input PDF
    if not pdf_path.is_file():
        logging.critical(f"CRITICAL ERROR: Input PDF not found at {pdf_path}")
        # Create error PDF if possible
        return
    total_pages = get_total_pages(pdf_path)
    if total_pages <= 0:
        logging.critical(f"CRITICAL ERROR: Could not get page count or PDF is invalid/empty: {pdf_path} (Result: {total_pages})")
        # Create error PDF if possible
        return
    logging.info(f"PDF '{pdf_path.name}' has {total_pages} pages.")

    # --- State Management and Resumption ---
    state = read_state()
    last_completed_page = state.get("last_completed_page", 0)
    # Preamble is now loaded directly from the TEX file if resuming
    start_page = last_completed_page + 1

    logging.info(f"Attempting to resume from page: {start_page} (Last successfully completed/placeholder: {last_completed_page})")

    # --- Initialize or Load Cumulative LaTeX Content ---
    cumulative_body_content = "" # Stores body content from \begin{document} onwards
    current_preamble = ""      # Stores preamble separately (global variable)

    if start_page > 1:
        if LAST_GOOD_TEX_FILE.exists():
            logging.info(f"Resuming: Loading existing content from {LAST_GOOD_TEX_FILE}...")
            try:
                loaded_tex = LAST_GOOD_TEX_FILE.read_text(encoding='utf-8').strip()
                # Find the first \begin{document} to split preamble and body
                begin_doc_match = re.search(re.escape(BEGIN_DOCUMENT_TAG), loaded_tex, re.IGNORECASE | re.DOTALL)
                end_doc_match = re.search(re.escape(END_DOCUMENT_TAG), loaded_tex, re.IGNORECASE | re.DOTALL)

                if begin_doc_match:
                    preamble_end_index = begin_doc_match.start()
                    current_preamble = loaded_tex[:preamble_end_index].strip()
                    logging.info(f"Loaded preamble (approx {len(current_preamble)} chars) from {LAST_GOOD_TEX_FILE.name}.")

                    body_start_index = begin_doc_match.start() # Include \begin{document}
                    body_end_index = end_doc_match.start() if end_doc_match else len(loaded_tex)
                    cumulative_body_content = loaded_tex[body_start_index:body_end_index].strip()
                    logging.info(f"Loaded body content (approx {len(cumulative_body_content)} chars) from {LAST_GOOD_TEX_FILE.name}.")

                    if not current_preamble or not cumulative_body_content:
                        raise ValueError("Extracted preamble or body is empty.")
                else:
                    raise ValueError(f"Could not find '{BEGIN_DOCUMENT_TAG}' in {LAST_GOOD_TEX_FILE.name}")

            except Exception as e:
                logging.error(f"Could not read/parse existing {LAST_GOOD_TEX_FILE}: {e}. Resetting content and state.")
                cumulative_body_content = ""
                current_preamble = ""
                last_completed_page = 0
                start_page = 1
                state = {"last_completed_page": 0} # Only store page number
                write_state(state)
        else:
             logging.error(f"Resuming from page {start_page} but {LAST_GOOD_TEX_FILE} not found. Resetting state and starting fresh.")
             cumulative_body_content = ""
             current_preamble = ""
             last_completed_page = 0
             start_page = 1
             state = {"last_completed_page": 0}
             write_state(state)
    else:
         # Starting fresh
         current_preamble = fallback_preamble.strip() # Start with fallback for page 1 processing
         cumulative_body_content = "" # Body is initially empty
         logging.info("Starting fresh. Initial preamble set to fallback.")


    # --- Process Pages ---
    processing_error_occurred = False
    last_successful_compile_page = last_completed_page

    for page_num in range(start_page, total_pages + 1):
        page_start_time = time.time()
        logging.info(f"--- Processing Page {page_num}/{total_pages} ---")

        # Preamble check (should always be set by now, either from file or fallback)
        if not current_preamble:
             logging.error(f"Critical error: Preamble is unexpectedly empty before processing page {page_num}. Stopping.")
             processing_error_occurred = True
             break

        # Get Page Image
        page_image = None
        try:
            page_image = get_image_from_page(pdf_path, page_num)
        except Exception as img_e:
            logging.error(f"Failed to get image for page {page_num}: {img_e}")
            # Try to continue with placeholder logic within the model function

        # --- Call Single Model Function ---
        # It receives the current preamble and body, performs API call & compilation,
        # updates global preamble if page 1, and returns the new *full* body or placeholder body or None
        page_result_body = get_latex_with_single_model(
            page_num,
            page_image,
            current_preamble, # Pass the current global preamble
            cumulative_body_content # Pass the body content up to the previous page
        )

        # --- Handle Result ---
        if page_result_body is None:
            # Critical error occurred inside get_latex_with_single_model (e.g., placeholder failed compile)
            logging.error(f"Processing failed critically for page {page_num}. Stopping.")
            processing_error_occurred = True
            # Record failure (already done inside the function if possible)
            record_failed_page(page_num, "critical_processing_failure", None) # Record again just in case
            break # Stop processing
        else:
            # Success! page_result_body contains the full, compiled body content
            # (either updated by AI or including a placeholder)
            logging.info(f"Successfully processed page {page_num}. Updating cumulative content.")
            cumulative_body_content = page_result_body # Replace the old body with the new full body

            # --- Save Good State ---
            full_tex_content_good = (
                current_preamble.strip() + "\n" +    # Use the potentially updated global preamble
                cumulative_body_content.strip() + "\n" + # This is the full body
                END_DOCUMENT_TAG + "\n"
            )
            try:
                LAST_GOOD_TEX_FILE.write_text(full_tex_content_good, encoding='utf-8')
                logging.info(f"Saved updated cumulative LaTeX ({len(full_tex_content_good)} chars) to {LAST_GOOD_TEX_FILE.name}")

                # Copy the successfully compiled PDF from the aux dir
                compiled_pdf_path = LATEX_AUX_DIR / CURRENT_ATTEMPT_TEX_FILE.with_suffix(".pdf").name
                if compiled_pdf_path.exists():
                    shutil.copy2(str(compiled_pdf_path), str(LAST_GOOD_PDF_FILE))
                    logging.info(f"Copied successfully compiled PDF to {LAST_GOOD_PDF_FILE.name}")
                else:
                    logging.warning(f"Compiled PDF ({compiled_pdf_path.name}) was expected but not found after successful page processing.")

            except Exception as e_write_good:
                 logging.error(f"Error writing/copying good state files after page {page_num}: {e_write_good}")
                 # Continue processing, but this is a bad sign

            # --- Update State ---
            last_successful_compile_page = page_num
            state["last_completed_page"] = page_num
            # No need to save preamble_generated flag anymore
            write_state(state)
            page_end_time = time.time()
            logging.info(f"Page {page_num} finished in {page_end_time - page_start_time:.2f} seconds.")

        # --- Cleanup ---
        if page_image:
            try: del page_image
            except Exception: pass
        # Consider periodic garbage collection if memory becomes an issue
        # import gc; gc.collect()


    # --- End of Processing Loop ---

    # --- Final Wrap-up ---
    save_model_stats() # Final save of stats
    if not processing_error_occurred and last_successful_compile_page == total_pages:
        logging.info(f"--- Successfully processed and compiled all {total_pages} pages! ---")
        logging.info(f"Final LaTeX file: {LAST_GOOD_TEX_FILE.resolve()}")
        if LAST_GOOD_PDF_FILE.exists():
            logging.info(f"Final PDF: {LAST_GOOD_PDF_FILE.resolve()}")
            # --- Append Reports/Extras ---
            logging.info("Appending report and failed images...")
            append_extras_to_final_pdf(LAST_GOOD_PDF_FILE)
        else:
            logging.warning(f"Final PDF ({LAST_GOOD_PDF_FILE}) not found, although processing finished.")
    elif processing_error_occurred:
         logging.error(f"--- Processing stopped due to critical error. Last successful compile was after page: {last_successful_compile_page} ---")
         logging.error(f"Check logs and {LAST_GOOD_TEX_FILE} (last good state) for details.")
         # Try appending reports to the last good PDF anyway
         if LAST_GOOD_PDF_FILE.exists():
              logging.info("Appending report and failed images to the last successfully generated PDF...")
              append_extras_to_final_pdf(LAST_GOOD_PDF_FILE)
         else:
              logging.warning("No good PDF found to append reports to.")
    else: # Finished but not all pages completed (e.g., user interrupt?)
         logging.warning(f"--- Processing finished. Last successful compile was after page: {last_successful_compile_page} / {total_pages} ---")
         logging.warning(f"See logs and {LAST_GOOD_TEX_FILE} for details.")
         if LAST_GOOD_PDF_FILE.exists():
              logging.info("Appending report and failed images to the partially completed PDF...")
              append_extras_to_final_pdf(LAST_GOOD_PDF_FILE)
         else:
              logging.warning("No good PDF found to append reports to.")


    # --- Final Summary Log ---
    end_time_main = time.time()
    total_time_main = end_time_main - start_time_main
    logging.info(f"Total script execution time: {total_time_main:.2f} seconds ({total_time_main/60:.2f} minutes).")
    logging.info("="*60)


# --- Utility Functions (read_state, write_state, report generation, PDF merging) ---
# (No changes needed in these functions, assuming they are correct)

def read_state():
    """Read the current state from the state file."""
    default_state = {"last_completed_page": 0} # Simplified state
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, 'r', encoding='utf-8') as f: state = json.load(f)
            if isinstance(state.get("last_completed_page"), int):
                 logging.info(f"Loaded state: Last completed page = {state['last_completed_page']}")
                 return state
            else:
                 logging.warning(f"State file {STATE_FILE} has invalid format. Resetting state.")
                 return default_state
        except (json.JSONDecodeError, TypeError, Exception) as e:
            logging.warning(f"Could not parse/read state file {STATE_FILE}: {e}. Resetting state.")
            # Attempt to delete corrupted state file? Or back it up?
            try:
                 corrupt_backup = STATE_FILE.with_suffix(".json.corrupt")
                 shutil.move(STATE_FILE, corrupt_backup)
                 logging.info(f"Moved corrupted state file to {corrupt_backup.name}")
            except Exception as move_e:
                 logging.error(f"Could not move corrupted state file: {move_e}")
            return default_state
    logging.info("No state file found or state reset. Starting fresh.")
    return default_state

def write_state(state_dict):
    """Write the current state to the state file."""
    try:
        # Backup previous state file
        if STATE_FILE.exists():
            backup_file = STATE_FILE.with_suffix(".json.bak")
            shutil.copy2(STATE_FILE, backup_file)

        # Only save last_completed_page
        state_to_save = {"last_completed_page": state_dict.get("last_completed_page", 0)}
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state_to_save, f, indent=2)
        logging.info(f"State updated and saved: {state_to_save}")
    except Exception as e:
        logging.error(f"Could not write state file {STATE_FILE}: {e}")

def generate_model_report():
    """Generate a LaTeX report summarizing model performance."""
    if not MODEL_STATS_FILE.exists():
        logging.warning("Model statistics file not found. Cannot generate report.")
        return "\\documentclass{article}\\usepackage{xcolor}\\definecolor{errorcolor}{rgb}{0.8, 0, 0}\\begin{document}\\section*{\\color{errorcolor}No Model Statistics Available}\\end{document}"

    try:
        with open(MODEL_STATS_FILE, 'r', encoding='utf-8') as f:
            stats = json.load(f)
    except Exception as e:
        logging.error(f"Error loading statistics file {MODEL_STATS_FILE}: {e}")
        return f"\\documentclass{{article}}\\usepackage{{xcolor}}\\definecolor{{errorcolor}}{{rgb}}{{0.8, 0, 0}}\\begin{{document}}\\section*{{Error Loading Statistics}}\\texttt{{{e}}}\\end{{document}}"

    report_parts = [
        "\\documentclass[10pt,landscape]{article}", # Landscape might be better for tables
        "\\usepackage[utf8]{inputenc}", "\\usepackage[T1]{fontenc}",
        "\\usepackage{geometry}", "\\geometry{a4paper, margin=0.7in, landscape}", # Adjust margins
        "\\usepackage{hyperref}", "\\hypersetup{colorlinks=true, linkcolor=blue, urlcolor=blue}",
        "\\usepackage{graphicx}", "\\usepackage{longtable}", "\\usepackage{xcolor}", "\\usepackage{booktabs}",
        "\\usepackage{fancyhdr}", "\\usepackage{datetime2}", # For ISO date formatting
        "\\pagestyle{fancy}", "\\fancyhf{}", "\\renewcommand{\\headrulewidth}{0.4pt}", "\\renewcommand{\\footrulewidth}{0.4pt}",
        "\\lhead{AI Translation Report (Full Context Mode)}", "\\rhead{\\today}", "\\cfoot{Page \\thepage}",
        "\\title{AI Translation Model Report\\\\ \\large (Single Model: " + TARGET_MODEL_INFO['nickname'] + ", Full Context Mode)}",
        "\\author{Automated Translation System}",
        "\\date{\\today}",
        "\\begin{document}", "\\maketitle", "\\thispagestyle{empty}", "\\tableofcontents", "\\clearpage"
    ]

    # --- Summary ---
    report_parts.append("\\section{Summary}")
    pages_translated_dict = stats.get("pages_translated", {})
    failures_dict = stats.get("failures", {})
    truncations_dict = stats.get("context_truncations", {})
    total_translated = len(pages_translated_dict)
    total_failed = len(failures_dict)
    total_processed = total_translated + total_failed

    # Collate model usage details
    model_attempts = {} # {page: [(model, retry)]}
    for page_str, attempts in pages_translated_dict.items():
        if page_str not in model_attempts: model_attempts[page_str] = []
        for attempt in attempts: # Should only be one successful attempt per page now
             model_attempts[page_str].append((attempt.get("model", "Unknown"), attempt.get("retry", 0)))

    # Calculate retry distribution if desired

    if total_processed > 0:
        success_rate = (total_translated / total_processed) * 100
        failure_rate = (total_failed / total_processed) * 100
        report_parts.append("\\begin{itemize}")
        report_parts.append(f"  \\item Target Model: \\textbf{{{TARGET_MODEL_INFO['nickname']}}}")
        report_parts.append(f"  \\item Total pages with recorded outcome: {total_processed}")
        report_parts.append(f"  \\item Successfully translated pages: {total_translated} ({success_rate:.1f}\\%)")
        report_parts.append(f"  \\item Failed pages (placeholder inserted): {total_failed} ({failure_rate:.1f}\\%)")
        report_parts.append(f"  \\item Pages where context was truncated: {len(truncations_dict)}")
        report_parts.append("\\end{itemize}")
    else:
        report_parts.append("No pages were processed or recorded.")

    # --- Successful Translation Details (Simplified for single model) ---
    report_parts.append("\\section{Translation Attempt Details}")
    if pages_translated_dict:
        report_parts.append("\\begin{longtable}{p{0.1\\textwidth} p{0.1\\textwidth}}") # Page | Retries
        report_parts.append("\\toprule \\textbf{Page} & \\textbf{Successful Retry} \\\\ \\midrule \\endfirsthead")
        report_parts.append("\\caption[]{Translation Attempt Details (Continued)} \\\\ \\toprule \\textbf{Page} & \\textbf{Successful Retry} \\\\ \\midrule \\endhead")
        report_parts.append("\\midrule \\multicolumn{2}{r}{{Continued on next page}} \\\\ \\bottomrule \\endfoot")
        report_parts.append("\\bottomrule \\endlastfoot")
        successful_pages = sorted([int(p) for p in pages_translated_dict.keys()])
        for page_num in successful_pages:
             page_str = str(page_num)
             # Get the retry count from the first (and only) success entry
             retry_count = pages_translated_dict[page_str][0].get("retry", "N/A") if pages_translated_dict[page_str] else "N/A"
             report_parts.append(f"  {page_str} & {retry_count} \\\\")
        report_parts.append("\\end{longtable}")
    else:
         report_parts.append("No pages were successfully translated.")

    # --- Failed Pages Details ---
    if failures_dict:
        report_parts.append("\\section{Failed Page Details}")
        report_parts.append("\\begin{longtable}{p{0.08\\textwidth} p{0.6\\textwidth} p{0.22\\textwidth}}") # Adjusted widths
        report_parts.append("\\toprule \\textbf{Page} & \\textbf{Last Recorded Reason} & \\textbf{Image Saved?} \\\\ \\midrule \\endfirsthead")
        report_parts.append("\\caption[]{Failed Page Details (Continued)} \\\\ \\toprule \\textbf{Page} & \\textbf{Last Recorded Reason} & \\textbf{Image Saved?} \\\\ \\midrule \\endhead")
        report_parts.append("\\midrule \\multicolumn{3}{r}{{Continued on next page}} \\\\ \\bottomrule \\endfoot")
        report_parts.append("\\bottomrule \\endlastfoot")

        for page_num_str in sorted(failures_dict.keys(), key=int):
             failure_list = failures_dict[page_num_str]
             if failure_list:
                  last_failure = failure_list[-1] # Get the last reason recorded
                  reason = last_failure.get("reason", "Unknown").replace("_", "\\_")
                  reason = reason.replace('&', r'\&').replace('%', r'\%').replace('#', r'\#') # Basic escaping
                  image_path = last_failure.get("saved_image")
                  image_saved_text = ""
                  if image_path:
                       img_fname = Path(image_path).name.replace("_", "\\_")
                       image_saved_text = f"Yes (\\texttt{{{img_fname}}})"
                  else: image_saved_text = "No"
                  report_parts.append(f"  {page_num_str} & \\texttt{{{reason}}} & {image_saved_text} \\\\") # Use tt for reason
        report_parts.append("\\end{longtable}")
    else:
         report_parts.append("\\section{Failed Pages}")
         report_parts.append("No page failures were recorded.")

    # --- Context Truncation Details ---
    if truncations_dict:
        report_parts.append("\\section{Context Truncation Details}")
        report_parts.append("\\begin{longtable}{p{0.1\\textwidth} p{0.2\\textwidth} p{0.2\\textwidth}}")
        report_parts.append("\\toprule \\textbf{Page} & \\textbf{Original Length} & \\textbf{Truncated Length} \\\\ \\midrule \\endfirsthead")
        report_parts.append("\\caption[]{Context Truncation Details (Continued)} \\\\ \\toprule \\textbf{Page} & \\textbf{Original Length} & \\textbf{Truncated Length} \\\\ \\midrule \\endhead")
        report_parts.append("\\midrule \\multicolumn{3}{r}{{Continued on next page}} \\\\ \\bottomrule \\endfoot")
        report_parts.append("\\bottomrule \\endlastfoot")

        for page_num_str in sorted(truncations_dict.keys(), key=int):
             trunc_list = truncations_dict[page_num_str]
             if trunc_list:
                 last_trunc = trunc_list[-1] # Show last truncation event for the page
                 orig_len = last_trunc.get("original_len", "N/A")
                 trunc_len = last_trunc.get("truncated_len", "N/A")
                 report_parts.append(f"  {page_num_str} & {orig_len} & {trunc_len} \\\\")
        report_parts.append("\\end{longtable}")
    else:
         report_parts.append("\\section{Context Truncations}")
         report_parts.append("No context truncations were recorded.")


    report_parts.append(f"\n{END_DOCUMENT_TAG}")
    return "\n".join(report_parts)

def insert_failed_page_images_into_final_pdf():
    """Creates a LaTeX appendix PDF containing images of failed pages."""
    failed_image_files = []
    if FAILED_PAGES_DIR.exists():
         try:
             failed_image_files = sorted(
                 FAILED_PAGES_DIR.glob(f"*.{IMAGE_FORMAT.lower()}"),
                 # Robust sorting: handle potential non-numeric filenames
                 key=lambda p: int(re.search(r'failed_page_(\d+)', p.name).group(1)) if re.search(r'failed_page_(\d+)', p.name) else 0
            )
         except Exception as e:
              logging.error(f"Error listing failed page images: {e}")


    if not failed_image_files:
        logging.info("No failed page images found to create appendix.")
        return None

    logging.info(f"Found {len(failed_image_files)} failed page images for appendix.")
    failed_pages_tex_content = [
        "\\documentclass[10pt]{article}", "\\usepackage[utf8]{inputenc}", "\\usepackage[T1]{fontenc}",
        "\\usepackage{graphicx}", "\\usepackage[a4paper, margin=0.5in]{geometry}", # Portrait, small margin
        "\\usepackage[colorlinks=true,linkcolor=blue]{hyperref}",
        "\\usepackage{fancyhdr}", "\\pagestyle{fancy}", "\\fancyhf{}",
        "\\renewcommand{\\headrulewidth}{0.4pt}", "\\renewcommand{\\footrulewidth}{0.4pt}",
        "\\lhead{Appendix: Untranslated Pages}", "\\rhead{\\today}", "\\cfoot{Page \\thepage}",
        "\\title{Appendix: Untranslated Original Page Images}", "\\author{Translation System}", "\\date{\\today}",
        "\\begin{document}", "\\maketitle", "\\thispagestyle{empty}",
        "\\section*{Introduction}",
        "The following pages could not be automatically translated or compiled.",
        "The original scanned images are included below for reference.",
        "\\clearpage"
    ]

    for img_path in failed_image_files:
        try:
            page_num_match = re.search(r'failed_page_(\d+)', img_path.name)
            if page_num_match:
                page_num = page_num_match.group(1)
                # Use relative path if possible, but absolute is safer across systems
                absolute_img_path = str(img_path.resolve()).replace("\\", "/") # Ensure forward slashes
                failed_pages_tex_content.extend([
                    f"\\subsection*{{Original Page {page_num}}}",
                    "\\begin{center}",
                    # Scale to fit page width/height while keeping aspect ratio
                    f"\\includegraphics[width=0.98\\textwidth, height=0.9\\textheight, keepaspectratio]{{{absolute_img_path}}}",
                    "\\end{center}",
                    "\\clearpage"
                ])
            else:
                 logging.warning(f"Could not extract page number from image filename: {img_path.name}")
        except Exception as e:
             logging.error(f"Error processing failed image {img_path} for appendix: {e}")

    failed_pages_tex_content.append(END_DOCUMENT_TAG)

    # Write and compile this LaTeX file
    failed_pages_tex_file = WORKING_DIR / "failed_pages_appendix.tex"
    compiled_pdf_path = LATEX_AUX_DIR / failed_pages_tex_file.with_suffix(".pdf").name
    try:
        failed_pages_tex_file.write_text("\n".join(failed_pages_tex_content), encoding='utf-8')
        logging.info(f"Compiling failed pages appendix: {failed_pages_tex_file.name}")
        # Use a dedicated aux dir for this compilation
        appendix_aux_dir = LATEX_AUX_DIR / "appendix_aux"
        appendix_aux_dir.mkdir(exist_ok=True)
        success, compile_log = compile_latex(failed_pages_tex_file, LATEX_AUX_DIR, num_runs=1) # Compile into main aux dir

        if success:
            if compiled_pdf_path.exists():
                 logging.info("Failed pages appendix compiled successfully.")
                 return compiled_pdf_path
            else:
                 logging.error(f"Failed pages appendix PDF not found at {compiled_pdf_path} after compilation success.")
                 return None
        else:
            logging.error("Failed to compile the failed pages appendix.")
            appendix_fail_log = WORKING_DIR / "appendix_compile.log" # Changed log name
            try: appendix_fail_log.write_text(compile_log, encoding='utf-8')
            except Exception: pass
            return None
    except Exception as e:
        logging.error(f"Error creating or compiling failed pages appendix: {e}")
        return None

def append_extras_to_final_pdf(main_pdf_path):
    """Generate report/appendix PDFs and merge them with the main PDF."""
    if not main_pdf_path or not main_pdf_path.exists():
        logging.warning(f"Cannot append reports: Main translated PDF not found or path is invalid ({main_pdf_path}).")
        return False

    pdfs_to_combine = [str(main_pdf_path.resolve())]

    # 1. Generate and compile the Model Report
    logging.info("Generating model report...")
    report_tex_content = generate_model_report()
    report_tex_file = WORKING_DIR / "model_report.tex"
    report_pdf_file = LATEX_AUX_DIR / report_tex_file.with_suffix(".pdf").name
    try:
        report_tex_file.write_text(report_tex_content, encoding='utf-8')
        logging.info(f"Compiling model report: {report_tex_file.name}")
        success, compile_log = compile_latex(report_tex_file, LATEX_AUX_DIR, num_runs=1) # Compile into main aux dir
        if success and report_pdf_file.exists():
             pdfs_to_combine.append(str(report_pdf_file.resolve()))
             logging.info("Model report compiled successfully.")
        else:
            logging.error(f"Failed to compile model report PDF. PDF exists: {report_pdf_file.exists()}")
            report_fail_log = WORKING_DIR / "report_compile.log"
            try: report_fail_log.write_text(compile_log, encoding='utf-8')
            except Exception: pass
    except Exception as e:
        logging.error(f"Error creating or compiling model report: {e}")

    # 2. Generate and compile the Failed Pages Appendix
    logging.info("Generating failed pages appendix...")
    failed_pages_appendix_pdf = insert_failed_page_images_into_final_pdf()
    if failed_pages_appendix_pdf and failed_pages_appendix_pdf.exists():
        pdfs_to_combine.append(str(failed_pages_appendix_pdf.resolve()))
        logging.info("Failed pages appendix added to merge list.")
    elif failed_pages_appendix_pdf:
         logging.warning("Failed pages appendix compilation succeeded but PDF not found.")
    else:
         logging.info("Failed pages appendix generation or compilation failed, or no images found.")


    # 3. Merge PDFs using pdfpages if more than one PDF exists
    if len(pdfs_to_combine) <= 1:
        logging.info("No additional reports/appendices to merge, or main PDF was missing.")
        final_output_pdf_path = WORKING_DIR / f"{main_pdf_path.stem}_final.pdf" # Output in working dir
        try:
            if main_pdf_path.exists():
                 shutil.copy2(str(main_pdf_path), str(final_output_pdf_path))
                 logging.info(f"Final PDF (no reports merged) saved as: {final_output_pdf_path.name}")
                 return True
            else:
                 # Maybe create a minimal PDF indicating the main one was missing?
                 logging.error("Main PDF was missing, cannot create final output.")
                 return False
        except Exception as e:
            logging.error(f"Could not copy/rename final PDF: {e}")
            return False

    # --- Perform the merge ---
    logging.info(f"Merging {len(pdfs_to_combine)} PDFs...")
    # Use a more descriptive name for the combined file
    base_name = main_pdf_path.stem.replace("_good", "") # Clean up base name
    combined_tex_file = WORKING_DIR / f"{base_name}_combined.tex"
    combined_pdf_output_file = LATEX_AUX_DIR / f"{base_name}_combined.pdf"
    final_output_pdf_path = WORKING_DIR / f"{base_name}_final_with_report.pdf" # Output in working dir

    combine_tex_content = [
        "\\documentclass{article}",
        "\\usepackage{pdfpages}",
        # Geometry settings might be needed if pages have different sizes,
        # but fitpaper=true usually handles this.
        # "\\usepackage[paperwidth=..., paperheight=...]{geometry}",
        "\\usepackage[hidelinks]{hyperref}", # Avoid boxes around links
        "\\begin{document}"
    ]
    for pdf_path_str in pdfs_to_combine:
        # Ensure path correctness for LaTeX
        safe_pdf_path = pdf_path_str.replace("\\", "/")
        combine_tex_content.append(f"\\includepdf[pages=-, fitpaper=true]{{{safe_pdf_path}}}")
    combine_tex_content.append(END_DOCUMENT_TAG)

    try:
        combined_tex_file.write_text("\n".join(combine_tex_content), encoding='utf-8')
        logging.info(f"Compiling combined document: {combined_tex_file.name}")
        # Use a dedicated aux dir for merging compilation
        merge_aux_dir = LATEX_AUX_DIR / "merge_aux"
        merge_aux_dir.mkdir(exist_ok=True)
        success, compile_log = compile_latex(combined_tex_file, LATEX_AUX_DIR, num_runs=1) # Compile into main aux dir

        if success and combined_pdf_output_file.exists():
             logging.info("Combined document compiled successfully.")
             shutil.copy2(str(combined_pdf_output_file), str(final_output_pdf_path))
             logging.info(f"Final merged PDF saved to: {final_output_pdf_path.name}")
             return True
        else:
             logging.error(f"Failed to compile the combined PDF document. PDF exists: {combined_pdf_output_file.exists()}")
             combined_fail_log = WORKING_DIR / "combined_compile.log"
             try: combined_fail_log.write_text(compile_log, encoding='utf-8')
             except Exception: pass
             # Fallback: Copy the original PDF without reports?
             logging.warning("Merging failed. Copying the main translated PDF without reports.")
             fallback_final_path = WORKING_DIR / f"{base_name}_final_NO_REPORT.pdf"
             try:
                  shutil.copy2(str(main_pdf_path), str(fallback_final_path))
                  logging.info(f"Fallback PDF saved as: {fallback_final_path.name}")
             except Exception as copy_e:
                   logging.error(f"Could not copy fallback PDF: {copy_e}")
             return False # Indicate merge failure
    except Exception as e:
        logging.error(f"Error creating or compiling combined PDF document: {e}")
        return False


# --- Command-Line Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Translate a PDF document page-by-page to LaTeX using a single AI model with full context.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'input_pdf',
        nargs='?',
        default=INPUT_PDF_PATH,
        help=f'Path to the input PDF file to translate.\n(Default: {INPUT_PDF_PATH})'
    )
    parser.add_argument(
        '--model', '-m',
        choices=list(get_model_configs().keys()),
        default=DEFAULT_MODEL_KEY, # Default is now set
        help=f'AI model key to use.\nSee --list-models for available keys.\n(Default: {DEFAULT_MODEL_KEY})'
    )
    parser.add_argument(
        '--list-models', '-l',
        action='store_true',
        help='List all configured model keys and their nicknames, then exit.'
    )
    parser.add_argument(
        '--working-dir', '-w',
        default=str(WORKING_DIR),
        help=f'Directory to store intermediate files, logs, and output.\n(Default: {WORKING_DIR})'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=IMAGE_DPI,
        help=f'Resolution (DPI) for converting PDF pages to images.\n(Default: {IMAGE_DPI})'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=MAX_RETRIES_PER_MODEL,
        help=f'Maximum number of retries per page if API or compilation fails.\n(Default: {MAX_RETRIES_PER_MODEL})'
    )

    args = parser.parse_args()

    # Handle --list-models action
    if args.list_models:
        print("\nAvailable model keys and nicknames:")
        print("-" * 60)
        # Sort by provider then nickname for readability
        sorted_models = sorted(get_model_configs().items(), key=lambda item: (item[1]['provider'], item[1]['nickname']))
        current_provider = ""
        for model_id, config in sorted_models:
             if config['provider'] != current_provider:
                  print(f"\n--- {config['provider'].upper()} Models ---")
                  current_provider = config['provider']
             limit = config.get('context_limit_tokens', 'N/A')
             print(f"  {model_id:<25} ({config['nickname']}) - Context: {limit}")
        print("-" * 60)
        print(f"Default model key: {DEFAULT_MODEL_KEY}")
        print("-" * 60)
        sys.exit(0)


    # --- Apply CLI Arguments ---
    WORKING_DIR = Path(args.working_dir)
    INPUT_PDF_PATH = args.input_pdf # Update global default used by main if not passed directly
    IMAGE_DPI = args.dpi
    MAX_RETRIES_PER_MODEL = args.max_retries

    # --- Reconfigure Paths and Logging based on WORKING_DIR ---
    STATE_FILE = WORKING_DIR / "state.json"
    LAST_GOOD_TEX_FILE = WORKING_DIR / "cumulative_good.tex"
    LAST_GOOD_PDF_FILE = WORKING_DIR / "cumulative_good.pdf"
    CURRENT_ATTEMPT_TEX_FILE = WORKING_DIR / "current_attempt.tex"
    LATEX_AUX_DIR = WORKING_DIR / "latex_aux"
    MODEL_STATS_FILE = WORKING_DIR / "model_stats.json"
    FAILED_PAGES_DIR = WORKING_DIR / "failed_pages"
    log_file_path = WORKING_DIR / "processing.log" # Update log file path

    # Re-create directories
    try:
        WORKING_DIR.mkdir(parents=True, exist_ok=True)
        LATEX_AUX_DIR.mkdir(parents=True, exist_ok=True)
        FAILED_PAGES_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as dir_e:
         # Use basic print as logging might not be fully set up
         print(f"CRITICAL ERROR: Could not create working directories in {WORKING_DIR}: {dir_e}", file=sys.stderr)
         sys.exit(1)

    # Reconfigure logging handler to use the potentially updated path
    root_logger = logging.getLogger()
    # Close existing file handlers first
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
             try: handler.close()
             except Exception: pass
             root_logger.removeHandler(handler)
        # Optionally remove and re-add stream handler to ensure it's present
        elif isinstance(handler, logging.StreamHandler):
            root_logger.removeHandler(handler)

    # Add new handlers
    try:
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

        root_logger.setLevel(logging.INFO) # Ensure level is set
    except Exception as log_e:
        print(f"ERROR: Could not configure logging to {log_file_path}: {log_e}", file=sys.stderr)
        # Continue without file logging if necessary? Or exit? For now, continue.
        pass


    # Call main function with parsed arguments
    main(input_pdf_cli=args.input_pdf, preferred_model_cli=args.model)
