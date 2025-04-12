import os
import sys
import anthropic
import openai
import pypdf
import subprocess
import time
import logging
from pathlib import Path
import re
from io import BytesIO
import json
import shutil
import base64
import random
import tempfile
from datetime import datetime
import argparse  # Add this at the top with other imports

# --- Image Handling Imports ---
try:
    from pdf2image import convert_from_path, pdfinfo_from_path
    from PIL import Image
except ImportError:
    logging.error("Error: pdf2image or Pillow not installed. Run: pip install pdf2image Pillow")
    exit(1)
except Exception as e:
    logging.error(f"Error importing pdf2image/Pillow: {e}")
    # Added suggestion for poppler path troubleshooting
    logging.error("Ensure poppler is installed (e.g., 'brew install poppler' or 'sudo apt-get install poppler-utils') and accessible in your system's PATH.")
    exit(1)

# --- Configuration ---
def get_model_configs():
    """Return all available model configurations"""
    return {
        # Claude models
        "claude-3-7": {"provider": "claude", "name": "claude-3-7-sonnet-20250219", "nickname": "Claude 3.7 Sonnet", "thinking": True},
        "claude-3-sonnet": {"provider": "claude", "name": "claude-3-sonnet-20240229", "nickname": "Claude 3 Sonnet", "thinking": True},
        "claude-3-haiku": {"provider": "claude", "name": "claude-3-haiku-20240307", "nickname": "Claude 3 Haiku", "thinking": False},
        
        # Gemini models
        "gemini-2-5-pro-exp": {"provider": "gemini", "name": "gemini-2.5-pro-exp-03-25", "nickname": "Gemini 2.5 Pro Exp", "thinking": True},
        "gemini-2-5-pro-preview": {"provider": "gemini", "name": "gemini-2.5-pro-preview-03-25", "nickname": "Gemini 2.5 Pro Preview", "thinking": True},
        "gemini-1-5-pro": {"provider": "gemini", "name": "gemini-1.5-pro-latest", "nickname": "Gemini 1.5 Pro", "thinking": True},
        "gemini-1-5-flash": {"provider": "gemini", "name": "gemini-1.5-flash-latest", "nickname": "Gemini 1.5 Flash", "thinking": False},
        
        # GPT models
        "gpt-4o": {"provider": "gpt", "name": "gpt-4o", "nickname": "GPT-4o", "thinking": True},
        "gpt-4o-mini": {"provider": "gpt", "name": "gpt-4o-mini", "nickname": "GPT-4o mini", "thinking": False},
    }

# Default model queue (your original preference)
DEFAULT_MODEL_QUEUE = [
    "claude-3-7",
    "gemini-2-5-pro-exp",
    "gpt-4o",
    "gemini-2-5-pro-preview"
]

def setup_model_queue(preferred_model=None):
    """Set up the model queue based on preferred model"""
    models = get_model_configs()
    
    if preferred_model:
        if preferred_model not in models:
            logging.warning(f"Requested model '{preferred_model}' not found. Using default queue.")
            queue = [models[model] for model in DEFAULT_MODEL_QUEUE]
        else:
            # Put preferred model first, then others in default order
            queue = [models[preferred_model]]
            queue.extend(models[model] for model in DEFAULT_MODEL_QUEUE if model != preferred_model)
    else:
        queue = [models[model] for model in DEFAULT_MODEL_QUEUE]
    
    return queue

# Max retries per model within a cycle
MAX_RETRIES_PER_MODEL = 2 # Reduced default retries
# Max cycles through all models for a single page
MAX_MODEL_CYCLES = 2 # Reduced default cycles

# Initialize AI clients
api_clients = {}
model_stats = {"pages_translated": {}, "failures": {}}  # Track which models translated which pages

# --- API Key Handling and Client Initialization ---
# Claude
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if CLAUDE_API_KEY:
    try:
        api_clients["claude"] = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
        logging.info("Claude API client initialized")
    except Exception as e:
        logging.error(f"Failed to initialize Claude client: {e}")
else:
    logging.warning("ANTHROPIC_API_KEY not set - Claude models will be skipped")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    try:
        # Use the new OpenAI client initialization
        api_clients["gpt"] = openai.OpenAI(api_key=OPENAI_API_KEY)
        logging.info("OpenAI client initialized")
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client: {e}")
else:
    logging.warning("OPENAI_API_KEY not set - GPT models will be skipped")

# Google AI (Gemini)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    try:
        import google.generativeai as genai
        # Removed glm import as it's less commonly needed directly with new API
        # import google.ai.generativelanguage as glm

        genai.configure(api_key=GOOGLE_API_KEY)
        # Simplified client storage
        api_clients["gemini"] = genai
        logging.info("Google AI client initialized")
    except ImportError:
        logging.error("Google AI libraries not installed. Run: pip install google-generativeai")
        # Explicitly remove gemini from clients if import fails
        if "gemini" in api_clients: del api_clients["gemini"]
    except Exception as e:
        logging.error(f"Failed to initialize Google AI client: {e}")
        if "gemini" in api_clients: del api_clients["gemini"]
else:
    logging.warning("GOOGLE_API_KEY not set - Gemini models will be skipped")


# --- Path and Processing Settings ---
INPUT_PDF_PATH = "input.pdf"  # <--- CHANGE THIS
WORKING_DIR = Path("latex_processing")
STATE_FILE = WORKING_DIR / "state.json"
LAST_GOOD_TEX_FILE = WORKING_DIR / "cumulative_good.tex"
LAST_GOOD_PDF_FILE = WORKING_DIR / "cumulative_good.pdf"
CURRENT_ATTEMPT_TEX_FILE = WORKING_DIR / "current_attempt.tex"
LATEX_AUX_DIR = WORKING_DIR / "latex_aux"
MODEL_STATS_FILE = WORKING_DIR / "model_stats.json"
FAILED_PAGES_DIR = WORKING_DIR / "failed_pages"

# LaTeX Tag Constant
END_DOCUMENT_TAG = "\\end{document}"

# pdf2image configuration
IMAGE_DPI = 200
IMAGE_FORMAT = 'JPEG' # JPEG is generally smaller than PNG for photos/scans

# Create directories
WORKING_DIR.mkdir(parents=True, exist_ok=True)
LATEX_AUX_DIR.mkdir(parents=True, exist_ok=True)
FAILED_PAGES_DIR.mkdir(parents=True, exist_ok=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(WORKING_DIR / "processing.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# --- Helper Functions ---

def clean_llm_latex_output(raw_latex):
    """Clean up LLM output to extract just the LaTeX code."""
    if not raw_latex:
        return ""

    # Check for markdown code blocks (common LLM formatting)
    # Handles ```latex ... ```, ``` ... ```, etc.
    box_pattern = r'```(?:[lL]a[tT]e[xX])?\s*(.*?)```'
    box_matches = re.findall(box_pattern, raw_latex, re.DOTALL | re.IGNORECASE)

    if box_matches:
        # If multiple blocks found, concatenate them assuming they are part of the same page
        # or use the longest one if concatenation seems wrong. Let's try longest first.
        box_matches.sort(key=len, reverse=True)
        cleaned_latex = box_matches[0].strip()
        logging.debug(f"Extracted LaTeX from markdown block.")
    else:
        # If no markdown block, assume the entire output is LaTeX, but clean it up.
        cleaned_latex = raw_latex.strip()
        # Remove potential introductory/explanatory text before \documentclass or content
        # This is heuristic - might need adjustment based on observed LLM behavior
        docclass_match = re.search(r'\\documentclass', cleaned_latex, re.IGNORECASE)
        if docclass_match:
            cleaned_latex = cleaned_latex[docclass_match.start():]
        else:
            # If no \documentclass, try to find common starting commands
            common_starts = ['\\section', '\\subsection', '\\begin', '%']
            first_match_pos = len(cleaned_latex)
            for start in common_starts:
                try:
                    pos = cleaned_latex.find(start)
                    if pos != -1 and pos < first_match_pos:
                        first_match_pos = pos
                except: # Should not happen with find, but safety first
                    pass
            if first_match_pos > 0 and first_match_pos < len(cleaned_latex):
                 # Check if the text before the match looks like explanation (e.g., contains "Here is the LaTeX")
                 preamble_text = cleaned_latex[:first_match_pos].lower()
                 if "latex" in preamble_text or "here is" in preamble_text or "page content" in preamble_text:
                      logging.debug(f"Removing potential preamble text before first LaTeX command.")
                      cleaned_latex = cleaned_latex[first_match_pos:]


    # Remove \end{document} tag specifically - it will be managed globally
    # Use regex for robustness against surrounding whitespace
    cleaned_latex = re.sub(r'\\end\{document\}', '', cleaned_latex, flags=re.IGNORECASE)

    # Optional: Remove common non-LaTeX explanations sometimes added at the end
    common_ends = ["that's the latex", "end of latex", "hope this helps"]
    for ending in common_ends:
        if cleaned_latex.lower().endswith(ending):
            # Find the last newline before the phrase to remove it cleanly
            last_newline = cleaned_latex.rfind('\n', 0, len(cleaned_latex) - len(ending))
            if last_newline != -1:
                cleaned_latex = cleaned_latex[:last_newline]
            else: # If no preceding newline, just remove the phrase
                cleaned_latex = cleaned_latex[:-len(ending)]
            cleaned_latex = cleaned_latex.strip()
            logging.debug(f"Removed potential ending phrase: '{ending}'")


    return cleaned_latex.strip()


def get_total_pages(pdf_path):
    """Get the total number of pages in the PDF."""
    try:
        # Try pypdf first
        reader = pypdf.PdfReader(str(pdf_path)) # Ensure path is string
        return len(reader.pages)
    except Exception as e_pypdf:
        logging.warning(f"pypdf failed to get page count ({e_pypdf}). Trying pdfinfo...")
        try:
            # Fallback to pdfinfo (requires poppler)
            info = pdfinfo_from_path(str(pdf_path), userpw=None, poppler_path=None)
            pages = info.get('Pages', 0)
            if pages > 0:
                return pages
            else:
                 logging.error(f"pdfinfo returned 0 pages or Pages key missing.")
                 return 0
        except Exception as e_pdfinfo:
             logging.error(f"Error getting page count with both pypdf and pdfinfo: {e_pdfinfo}")
             return 0

def get_image_from_page(pdf_path, page_num, dpi=IMAGE_DPI, img_format=IMAGE_FORMAT):
    """Convert a PDF page to an image."""
    logging.debug(f"Converting page {page_num} to image...")
    try:
        images = convert_from_path(
            str(pdf_path), # Ensure path is string
            dpi=dpi,
            first_page=page_num,
            last_page=page_num,
            fmt=img_format.lower(),
            thread_count=1, # Avoid potential multi-threading issues
            timeout=120 # Add a timeout (in seconds)
        )
        if images:
            logging.debug(f"Successfully converted page {page_num} to image.")
            return images[0]
        else:
            logging.error(f"convert_from_path returned empty list for page {page_num}.")
            return None
    except Exception as e:
        # Catch specific pdf2image errors if possible, otherwise generic exception
        logging.error(f"Error converting page {page_num} to image: {e}")
        # Provide more help if poppler path might be the issue
        if "poppler" in str(e).lower():
             logging.error("This might indicate a problem with Poppler installation or path.")
        return None

def pil_image_to_bytes(image, img_format=IMAGE_FORMAT):
    """Convert PIL Image to bytes."""
    if image is None:
        return None, None
    byte_arr = BytesIO()
    mime_type = f"image/{img_format.lower()}"
    try:
        image.save(byte_arr, format=img_format)
        return byte_arr.getvalue(), mime_type
    except Exception as e:
        logging.error(f"Error converting PIL Image to bytes: {e}")
        return None, None

def save_image_for_failed_page(image, page_num):
    """Save an image of a failed page."""
    if image is None:
        logging.warning(f"Cannot save image for failed page {page_num} - image data is missing.")
        return None

    failed_page_image_path = FAILED_PAGES_DIR / f"failed_page_{page_num}.{IMAGE_FORMAT.lower()}"
    try:
        image.save(failed_page_image_path, IMAGE_FORMAT)
        logging.info(f"Saved image of failed page {page_num} to {failed_page_image_path}")
        return str(failed_page_image_path) # Return path as string
    except Exception as e:
        logging.error(f"Error saving failed page {page_num} image: {e}")
        return None

def get_latex_from_image_with_model(page_image, page_num, context_info, model_info, retry_count=0):
    """Generate LaTeX from an image using a specific model."""
    provider = model_info["provider"]
    model_name = model_info["name"]
    nickname = model_info["nickname"]
    allow_thinking = model_info.get("thinking", False)

    # Pre-check: If provider client isn't initialized, fail fast.
    if provider not in api_clients:
        logging.error(f"API client for provider '{provider}' not available. Skipping {nickname}.")
        return f"% ERROR: {provider} API client not initialized for {model_name}.\n", False

    # Pre-check: Ensure page_image is valid
    if page_image is None:
        logging.error(f"Cannot process page {page_num} with {nickname}: page_image is None.")
        # Return success=False but allow tournament to continue potentially
        return f"% ERROR: Input image data missing for page {page_num}.\n", False

    image_bytes, mime_type = pil_image_to_bytes(page_image, IMAGE_FORMAT)
    if not image_bytes or not mime_type:
        logging.error(f"Failed to convert image for page {page_num} to bytes.")
        return f"% ERROR: Could not convert image bytes for page {page_num}.\n", False

    # --- Prompt Engineering ---
    # Base prompt structure
    prompt_parts = [
        "# Task",
        f"Analyze the image of Page {page_num} from a non-English academic document (likely mathematical or scientific) and generate the corresponding English LaTeX code for this page only.",
        "\n# Requirements",
        "- Translate all non-English text to English.",
        "- Accurately represent all mathematical formulas, equations, and symbols using standard LaTeX (amsmath, amssymb, etc.).",
        "- Preserve the original layout structure where semantically important (e.g., sections, subsections, theorems, definitions, lists, figures, tables). Use appropriate LaTeX environments.",
        "- Maintain logical flow and consistency with potential previous/following pages.",
    ]

    # Context Handling
    if context_info:
        if context_info.get("is_first_page", False):
            prompt_parts.extend([
                "\n# Context: First Page",
                "This is the **first page** of the document.",
                "- Infer the document type (e.g., article, report, thesis).",
                "- Generate a suitable LaTeX **preamble** (including \\documentclass, standard packages like inputenc, fontenc, amsmath, amssymb, amsthm, graphicx, hyperref). Set up common theorem-like environments if content suggests they are needed.",
                "- Include \\title{}, \\author{}, \\date{} commands based on the visible content.",
                "- Start the document body with \\begin{document} and include \\maketitle.",
                "- Add \\tableofcontents if appropriate.",
                "- Finally, include the translated LaTeX content for this first page.",
                f"- IMPORTANT: Do **NOT** include the final `{END_DOCUMENT_TAG}` tag in your response."
            ])
        elif context_info.get("previous_latex"):
            # Clean previous latex snippet for context prompt
            clean_prev_latex = re.sub(r'\\end\{document\}', '', context_info["previous_latex"], flags=re.IGNORECASE).strip()
            context_snippet = clean_prev_latex[-1500:] # Limit context size
            prompt_parts.extend([
                "\n# Context: Previous Content",
                "This is a subsequent page. Maintain consistency with the document structure established earlier.",
                "Here is a snippet of the LaTeX code generated for the *previous* page(s) for context (do NOT repeat this content):",
                "```latex",
                context_snippet,
                "```",
                 f"- IMPORTANT: Provide **ONLY** the LaTeX code for the current Page {page_num}. Do **NOT** include preamble, \\begin{{document}}, or \\maketitle.",
                 f"- IMPORTANT: Do **NOT** include the final `{END_DOCUMENT_TAG}` tag in your response."
            ])
        else:
             # Subsequent page, but no previous LaTeX context available (shouldn't normally happen after page 1)
             prompt_parts.extend([
                  "\n# Context: Subsequent Page (No Snippet)",
                  "This is a subsequent page. Assume a standard LaTeX preamble and document structure has already been established.",
                  f"- IMPORTANT: Provide **ONLY** the LaTeX code for the current Page {page_num}. Do **NOT** include preamble, \\begin{{document}}, or \\maketitle.",
                  f"- IMPORTANT: Do **NOT** include the final `{END_DOCUMENT_TAG}` tag in your response."
             ])

    # Thinking Prompt (Conditional)
    if allow_thinking:
        prompt_parts.extend([
            "\n# Reasoning (Optional)",
            "Before generating the LaTeX, briefly outline your understanding of the page content and structure (e.g., 'This page contains Theorem 1.2 and its proof...').",
            "Then, provide the final LaTeX code enclosed in triple backticks.",
            "```latex",
            "[Your LaTeX code for Page {page_num} here]",
            "```"
        ])
        response_format_instruction = "Enclose the final LaTeX code block within triple backticks (```latex ... ```)."
    else:
        response_format_instruction = "Respond ONLY with the raw LaTeX code for this page. Do not include explanations, apologies, or markdown formatting like backticks."

    prompt_parts.extend(["\n# Response Format", response_format_instruction])

    prompt = "\n".join(prompt_parts)

    # --- API Call ---
    response_text = None
    start_time = time.time()
    try:
        logging.info(f"Sending Page {page_num} to {nickname} ({model_name}) (Cycle {context_info.get('cycle', '?')}, Retry {retry_count})...")

        if provider == "claude":
            client = api_clients["claude"]
            message_list = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": base64.b64encode(image_bytes).decode("utf-8"),
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            response = client.messages.create(
                model=model_name,
                max_tokens=4096, # Max for Sonnet 3.5 / Haiku
                temperature=0.2, # Lower temp for more deterministic LaTeX
                messages=message_list,
            )
            if response.content and isinstance(response.content, list) and hasattr(response.content[0], 'text'):
                 response_text = response.content[0].text
            else:
                 logging.warning(f"Unexpected response structure from Claude {nickname}: {response}")


        elif provider == "gpt":
            client = api_clients["gpt"]
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    # System prompt can sometimes help guide the model
                    {"role": "system", "content": "You are an expert LaTeX translator specializing in academic documents. Translate French to English LaTeX accurately."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                # Using base64 data URI directly
                                "url": f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('utf-8')}"
                            }
                        }
                    ]}
                ],
                max_tokens=4096,
                temperature=0.2,
            )
            if response.choices and response.choices[0].message:
                 response_text = response.choices[0].message.content
            else:
                 logging.warning(f"Unexpected response structure from GPT {nickname}: {response}")


        elif provider == "gemini":
            genai_client = api_clients["gemini"]
            # Safety settings - adjust threshold if needed
            safety_settings = [
                {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} # Or BLOCK_LOW_AND_ABOVE / BLOCK_NONE
                for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
            ]
            gemini_model = genai_client.GenerativeModel(
                 model_name,
                 safety_settings=safety_settings,
                 generation_config={"temperature": 0.2, "max_output_tokens": 8192} # Gemini often supports larger output
            )
            # Prepare content parts for Gemini
            content_parts = [
                prompt, # Text prompt first
                {"mime_type": mime_type, "data": image_bytes} # Image data
            ]
            response = gemini_model.generate_content(content_parts, request_options={"timeout": 180}) # Add timeout
            # Accessing response text might differ slightly based on version/response type
            try:
                 response_text = response.text
            except ValueError as e:
                 logging.warning(f"Gemini response blocked or invalid for page {page_num}: {e}. Response parts: {response.parts}")
                 # Try accessing parts if available and text failed
                 try:
                      if response.parts:
                           response_text = "".join(part.text for part in response.parts if hasattr(part, "text"))
                           if not response_text: # If still empty after joining parts
                                return f"% ERROR: Gemini response blocked or empty for page {page_num}. Reason: {response.prompt_feedback}\n", False
                      else: # No parts either
                           return f"% ERROR: Gemini response blocked or empty for page {page_num}. Reason: {response.prompt_feedback}\n", False
                 except Exception as part_e:
                      logging.error(f"Could not extract text from Gemini response parts: {part_e}")
                      return f"% ERROR: Gemini response unreadable for page {page_num}.\n", False

            except Exception as e:
                 logging.error(f"Error accessing Gemini response text: {e}")
                 return f"% ERROR: Could not read Gemini response for page {page_num}.\n", False


        elapsed_time = time.time() - start_time
        # --- Response Handling ---
        if response_text:
            logging.info(f"Received response for Page {page_num} from {nickname} in {elapsed_time:.2f}s.")
            clean_latex = clean_llm_latex_output(response_text)
            if not clean_latex:
                 logging.warning(f"Cleaned LaTeX output is empty for page {page_num} from {nickname}.")
                 return f"% ERROR: Empty LaTeX output after cleaning from {nickname}.\n", False
            return clean_latex, True
        else:
            # Handle cases where API call succeeded but response was empty/invalid
            logging.warning(f"Empty or invalid response content for page {page_num} from {nickname} (Retry {retry_count}).")
            return f"% ERROR: Empty or invalid response from {nickname}.\n", False

    except Exception as e:
        elapsed_time = time.time() - start_time
        # Catch API errors, timeouts, connection issues etc.
        logging.error(f"API Error calling {nickname} for page {page_num} (Retry {retry_count}, {elapsed_time:.2f}s): {e}")
        # Add more specific error type logging if possible
        if isinstance(e, openai.APITimeoutError):
             logging.error("Timeout occurred during OpenAI API call.")
        elif isinstance(e, anthropic.APIConnectionError):
             logging.error("Connection error during Anthropic API call.")
        # Add similar checks for Google AI exceptions if available
        return f"% ERROR: API Exception with {nickname}: {e}\n", False


def get_latex_from_image_tournament(page_image, page_num, context_info):
    """Run a tournament of models to get LaTeX from an image, with multiple cycles if needed."""
    if page_image is None:
        logging.error(f"Tournament skipped for page {page_num}: page_image is None.")
        record_failed_page(page_num, "image_extraction_failed_before_tournament")
        # Return a placeholder directly
        return f"""
% ========== PAGE {page_num} IMAGE MISSING ==========
% Image could not be extracted/provided for processing.
\\begin{{center}}\\fbox{{\\textbf{{Page {page_num}: Image Missing}}}}\\end{{center}}
% ========== END OF PAGE {page_num} PLACEHOLDER ==========
"""

    # Store the image path early in case all models fail
    saved_image_path = None

    for cycle in range(MAX_MODEL_CYCLES):
        logging.info(f"--- Starting Model Cycle {cycle+1}/{MAX_MODEL_CYCLES} for Page {page_num} ---")
        context_info['cycle'] = cycle + 1 # Pass cycle info to model call for logging

        # Optional: Shuffle model queue for each cycle?
        # random.shuffle(MODEL_QUEUE)

        for model_info in MODEL_QUEUE:
            provider = model_info["provider"]
            nickname = model_info["nickname"]

            if provider not in api_clients:
                logging.warning(f"Skipping {nickname} (Cycle {cycle+1}) - API client not available.")
                continue

            for retry in range(MAX_RETRIES_PER_MODEL):
                attempt_num = retry + 1
                latex_content = None
                success = False
                try:
                    # Pass page_image directly
                    latex_content, success = get_latex_from_image_with_model(
                        page_image, page_num, context_info, model_info, retry_count=attempt_num
                    )

                    if success and latex_content and not latex_content.startswith("% ERROR"):
                        logging.info(f"Success! Page {page_num} translated by {nickname} (Cycle {cycle+1}, Retry {attempt_num}).")
                        # Record successful model stats
                        if str(page_num) not in model_stats["pages_translated"]:
                            model_stats["pages_translated"][str(page_num)] = []
                        model_stats["pages_translated"][str(page_num)].append({
                            "model": nickname,
                            "model_name": model_info["name"],
                            "cycle": cycle + 1,
                            "retry": attempt_num,
                            "timestamp": datetime.now().isoformat()
                        })
                        save_model_stats()

                        # Add attribution comment
                        latex_with_attribution = f"% Page {page_num} translated by {nickname} (Cycle {cycle+1}, Retry {attempt_num})\n{latex_content}"
                        return latex_with_attribution # Return successful result

                    elif latex_content and latex_content.startswith("% ERROR"):
                         logging.warning(f"{nickname} (Cycle {cycle+1}, Retry {attempt_num}) failed for page {page_num}. Reason: {latex_content.strip()}")
                    else: # success was False or latex_content was empty
                         logging.warning(f"{nickname} (Cycle {cycle+1}, Retry {attempt_num}) failed for page {page_num}. Reason: Call returned success=False or empty content.")

                except Exception as e:
                    # Catch unexpected errors within the model call itself
                    logging.error(f"Critical error during call to {nickname} for page {page_num} (Cycle {cycle+1}, Retry {attempt_num}): {e}", exc_info=True)
                    # Ensure we mark as failed if exception occurs
                    success = False

                # Small delay between retries of the *same* model
                if attempt_num < MAX_RETRIES_PER_MODEL:
                     time.sleep(random.uniform(1, 3)) # Randomized sleep

            # Delay between *different* models in the same cycle
            time.sleep(random.uniform(0.5, 1.5))

    # --- All models failed after all cycles ---
    logging.error(f"All models failed to translate page {page_num} after {MAX_MODEL_CYCLES} cycles.")

    # Save the image for manual review if not already saved
    if saved_image_path is None: # Check if already saved
         saved_image_path = save_image_for_failed_page(page_image, page_num)

    # Record the overall failure for this page
    record_failed_page(page_num, "all_models_failed_all_cycles")

    # Create a placeholder that allows compilation to continue
    placeholder = f"""
% ========== PAGE {page_num} TRANSLATION FAILED ==========
\\clearpage
\\begin{{center}}
\\vspace*{{2cm}}
\\fbox{{
    \\begin{{minipage}}{{0.8\\textwidth}}
    \\centering
    \\textbf{{Page {page_num} - Translation Failed}}\\\\[1em]
    \\textcolor{{red}}{{All translation attempts failed for this page.}}\\\\[1em]
    The original page image has been saved for manual review.
    \\end{{minipage}}
}}
\\vspace*{{2cm}}
\\end{{center}}
\\clearpage
% ========== END OF PAGE {page_num} PLACEHOLDER ==========
"""
    return placeholder


def record_failed_page(page_num, reason):
    """Record a failed page in the model stats."""
    page_str = str(page_num)
    if "failures" not in model_stats:
        model_stats["failures"] = {}

    if page_str not in model_stats["failures"]:
        model_stats["failures"][page_str] = []

    model_stats["failures"][page_str].append({
        "reason": reason,
        "timestamp": datetime.now().isoformat()
    })
    logging.info(f"Recorded failure for page {page_num}: {reason}")
    save_model_stats() # Save stats after recording failure

def save_model_stats():
    """Save the model stats to a file."""
    try:
        with open(MODEL_STATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(model_stats, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving model stats to {MODEL_STATS_FILE}: {e}")

def compile_latex(tex_filepath, aux_dir):
    """Compile a LaTeX file and return success status and detailed log."""
    if not tex_filepath.is_file():
        return False, "Error: LaTeX source file not found."

    aux_dir = Path(aux_dir)
    aux_dir.mkdir(parents=True, exist_ok=True)

    # Try pdflatex first (simpler, might be more reliable for basic docs)
    command = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-file-line-error",
        f"-output-directory={aux_dir.resolve()}",
        str(tex_filepath.resolve())
    ]

    log_output = f"--- Running: {' '.join(command)} ---\n"
    success = False

    try:
        # Run pdflatex twice to resolve references
        for attempt in range(2):
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                encoding='utf-8',
                errors='replace',
                timeout=60  # Shorter timeout per run
            )
            
            log_output += f"\nPass {attempt + 1} output:\n"
            log_output += f"Stdout:\n{process.stdout}\n"
            log_output += f"Stderr:\n{process.stderr}\n"
            
            # Check for critical errors in the output
            if "Fatal error occurred" in process.stdout or "Emergency stop" in process.stdout:
                success = False
                log_output += "\nFatal LaTeX error detected.\n"
                break
            
            # Check if PDF was generated
            pdf_file = aux_dir / tex_filepath.with_suffix(".pdf").name
            if pdf_file.exists() and pdf_file.stat().st_size > 0:
                success = True
            else:
                success = False
        
        if success:
            log_output += "\n--- Compilation completed successfully ---"
        else:
            log_output += "\n--- Compilation failed ---"
            
    except subprocess.TimeoutExpired:
        log_output += "\nCompilation timed out"
        success = False
    except Exception as e:
        log_output += f"\nUnexpected error during compilation: {e}"
        success = False

    return success, log_output


def read_state():
    """Read the current state from the state file."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                state = json.load(f)
            # Basic validation
            if isinstance(state.get("last_completed_page"), int) and isinstance(state.get("preamble_generated"), bool):
                 logging.info(f"Loaded state: {state}")
                 return state
            else:
                 logging.warning(f"State file {STATE_FILE} has invalid format. Resetting state.")
                 return {"last_completed_page": 0, "preamble_generated": False}
        except json.JSONDecodeError as e:
            logging.warning(f"Could not parse state file {STATE_FILE}: {e}. Starting from scratch.")
            return {"last_completed_page": 0, "preamble_generated": False}
        except Exception as e:
             logging.error(f"Error reading state file {STATE_FILE}: {e}. Starting from scratch.")
             return {"last_completed_page": 0, "preamble_generated": False}
    return {"last_completed_page": 0, "preamble_generated": False} # Default state


def write_state(state_dict):
    """Write the current state to the state file."""
    try:
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state_dict, f, indent=2)
        logging.info(f"State updated and saved: {state_dict}")
    except Exception as e:
        logging.error(f"Could not write state file {STATE_FILE}: {e}")


def generate_model_report():
    """Generate a report of which models translated which pages."""
    if not MODEL_STATS_FILE.exists():
        return "\\section*{No Model Statistics Available}\nNo model statistics file found."

    try:
        with open(MODEL_STATS_FILE, 'r', encoding='utf-8') as f:
            stats = json.load(f)
    except Exception as e:
        return f"\\section*{{Error Loading Statistics}}\nError loading model statistics file ({MODEL_STATS_FILE}): {e}"

    # Use LaTeX formatting directly
    report_parts = ["\\documentclass{article}",
                    "\\usepackage[utf8]{inputenc}",
                    "\\usepackage[T1]{fontenc}",
                    "\\usepackage{geometry}",
                    "\\geometry{a4paper, margin=1in}", # Adjust margins
                    "\\usepackage{hyperref}",
                    "\\usepackage{graphicx}", # Needed for failed page images potentially
                    "\\usepackage{array}", # For tables maybe
                    "\\usepackage{longtable}", # For long lists
                    "\\usepackage{xcolor}", # For highlighting perhaps
                    "\\title{AI Translation Model Tournament Report}",
                    "\\author{Translation System}",
                   f"\\date{{{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}}",
                    "\\begin{document}",
                    "\\maketitle"]

    # --- Pages Translated by Model ---
    report_parts.append("\\section{Pages Translated by Model}")
    model_pages = {}
    total_translated = 0
    for page_num_str, translations in stats.get("pages_translated", {}).items():
        if translations:
            try:
                page_num = int(page_num_str)
                total_translated += 1
                # Get info from the first successful translation record for this page
                model_nickname = translations[0].get("model", "Unknown Model")
                # model_name = translations[0].get("model_name", "N/A") # Optionally add full name
                if model_nickname not in model_pages:
                    model_pages[model_nickname] = []
                model_pages[model_nickname].append(page_num)
            except ValueError:
                logging.warning(f"Invalid page number format in stats: {page_num_str}")
                continue # Skip malformed entries

    if not model_pages:
        report_parts.append("No pages were successfully translated.")
    else:
        report_parts.append("\\begin{itemize}")
        # Sort models by number of pages translated (descending)
        for model, pages in sorted(model_pages.items(), key=lambda item: len(item[1]), reverse=True):
            pages.sort()
            # Format page numbers with ranges
            page_ranges = []
            if pages:
                start = end = pages[0]
                for i in range(1, len(pages)):
                    if pages[i] == end + 1:
                        end = pages[i]
                    else:
                        page_ranges.append(str(start) if start == end else f"{start}-{end}")
                        start = end = pages[i]
                page_ranges.append(str(start) if start == end else f"{start}-{end}") # Add the last range

            report_parts.append(f"  \\item \\textbf{{{model}}}: {len(pages)} pages ({', '.join(page_ranges)})")
        report_parts.append("\\end{itemize}")


    # --- Failed Pages ---
    failed_pages_dict = stats.get("failures", {})
    total_failed = len(failed_pages_dict)
    if failed_pages_dict:
        report_parts.append("\\section{Failed Pages}")
        report_parts.append("\\begin{longtable}{p{0.15\\textwidth} p{0.8\\textwidth}}")
        report_parts.append("\\hline \\textbf{Page} & \\textbf{Last Recorded Reason} \\\\ \\hline \\endfirsthead")
        report_parts.append("\\hline \\textbf{Page} & \\textbf{Last Recorded Reason} \\\\ \\hline \\endhead")
        report_parts.append("\\hline \\endfoot")
        report_parts.append("\\hline \\endlastfoot")

        # Sort failed pages numerically
        for page_num_str, failure_list in sorted(failed_pages_dict.items(), key=lambda item: int(item[0])):
             if failure_list:
                  last_failure = failure_list[-1] # Get the last reason recorded
                  reason = last_failure.get("reason", "Unknown reason").replace("_", "\\_") # Escape underscores for LaTeX
                  timestamp = last_failure.get("timestamp", "")
                  # Make reason text smaller if too long? Or allow wrapping via p{} column type.
                  report_parts.append(f"  Page {page_num_str} & {reason} \\\\") # ({timestamp}) # Optionally add timestamp
        report_parts.append("\\end{longtable}")
    else:
         report_parts.append("\\section{Failed Pages}")
         report_parts.append("No page failures were recorded.")


    # --- Performance Summary ---
    report_parts.append("\\section{Performance Summary}")
    total_processed = total_translated + total_failed
    if total_processed > 0:
        success_rate = (total_translated / total_processed) * 100
        failure_rate = (total_failed / total_processed) * 100
        report_parts.append("\\begin{itemize}")
        report_parts.append(f"  \\item Total pages attempted: {total_processed}")
        report_parts.append(f"  \\item Successfully translated: {total_translated} ({success_rate:.1f}\\%)")
        report_parts.append(f"  \\item Failed pages: {total_failed} ({failure_rate:.1f}\\%)")
        report_parts.append("\\end{itemize}")

        # Optionally add more stats like average time per page, cost estimation (if possible) etc.

    else:
        report_parts.append("No pages were processed.")

    report_parts.append(f"\n{END_DOCUMENT_TAG}")
    return "\n".join(report_parts)


def insert_failed_page_images_into_final_pdf():
    """Create a LaTeX document embedding images of pages that failed all translation attempts."""
    failed_image_files = sorted(FAILED_PAGES_DIR.glob("failed_page_*.jpg"),
                                key=lambda p: int(re.search(r'failed_page_(\d+)', p.name).group(1)))

    if not failed_image_files:
        logging.info("No failed page images found to insert.")
        return None # No failed pages to process

    failed_pages_tex_content = [
        "\\documentclass{article}",
        "\\usepackage[utf8]{inputenc}",
        "\\usepackage[T1]{fontenc}",
        "\\usepackage{graphicx}",
        "\\usepackage{geometry}",
        "\\geometry{a4paper, margin=1in}",
        "\\usepackage[colorlinks=true,linkcolor=blue]{hyperref}", # Added hyperref
        "\\title{Appendix: Untranslated Original Page Images}",
        "\\author{Translation System}",
        "\\date{\\today}",
        "\\begin{document}",
        "\\maketitle",
        "\\section*{Introduction}",
        "The following pages could not be automatically translated by any of the configured AI models after multiple attempts.",
        "The original scanned images are included here for reference and potential manual translation.",
        "\\clearpage"
    ]

    for img_path in failed_image_files:
        try:
            page_num_match = re.search(r'failed_page_(\d+)', img_path.name)
            if page_num_match:
                page_num = page_num_match.group(1)
                # Use hypertarget and hyperref for navigation if desired
                failed_pages_tex_content.append(f"\\hypertarget{{failedpage{page_num}}}{{}}")
                failed_pages_tex_content.append(f"\\subsection*{{Original Page {page_num}}}")
                # Need to use absolute path or ensure graphicspath is set
                # Using relative path from the aux directory perspective
                # relative_img_path = Path("..") / FAILED_PAGES_DIR.name / img_path.name
                # Or just use absolute path which is safer if aux dir changes
                absolute_img_path = str(img_path.resolve())
                # Use includegraphics with scaling
                failed_pages_tex_content.append("\\begin{center}")
                # Use width=\textwidth, height=\textheight, keepaspectratio to fit page
                failed_pages_tex_content.append(f"\\includegraphics[width=\\textwidth, height=0.9\\textheight, keepaspectratio]{{{absolute_img_path}}}")
                failed_pages_tex_content.append("\\end{center}")
                failed_pages_tex_content.append("\\clearpage") # Page break after each image
            else:
                logging.warning(f"Could not extract page number from filename: {img_path.name}")
        except Exception as e:
             logging.error(f"Error processing failed image {img_path}: {e}")


    failed_pages_tex_content.append(END_DOCUMENT_TAG)

    # Write and compile this LaTeX file
    failed_pages_tex_file = WORKING_DIR / "failed_pages_appendix.tex"
    try:
        failed_pages_tex_file.write_text("\n".join(failed_pages_tex_content), encoding='utf-8')
        logging.info(f"Compiling failed pages appendix: {failed_pages_tex_file}")
        success, compile_log = compile_latex(failed_pages_tex_file, LATEX_AUX_DIR)

        if success:
            compiled_pdf_path = LATEX_AUX_DIR / failed_pages_tex_file.with_suffix(".pdf").name
            logging.info("Failed pages appendix compiled successfully.")
            return compiled_pdf_path
        else:
            logging.error("Failed to compile the failed pages appendix.")
            # Log the compilation failure details
            appendix_fail_log = WORKING_DIR / "failed_appendix_compile.log"
            try:
                 appendix_fail_log.write_text(compile_log, encoding='utf-8')
                 logging.error(f"Failed appendix compilation log saved to: {appendix_fail_log}")
            except Exception as log_e:
                 logging.error(f"Could not write failed appendix compile log: {log_e}")
            return None
    except Exception as e:
        logging.error(f"Error creating or compiling failed pages appendix: {e}")
        return None


def append_report_to_final_pdf():
    """Generate report/appendix PDFs and merge them with the main PDF."""
    if not LAST_GOOD_PDF_FILE.exists():
        logging.warning("Cannot append reports: Main translated PDF file does not exist.")
        return False

    pdfs_to_combine = [str(LAST_GOOD_PDF_FILE.resolve())] # Start with the main doc

    # 1. Generate and compile the Model Report
    report_tex_content = generate_model_report()
    report_tex_file = WORKING_DIR / "model_report.tex"
    report_pdf_file = None
    try:
        report_tex_file.write_text(report_tex_content, encoding='utf-8')
        logging.info(f"Compiling model report: {report_tex_file}")
        success, compile_log = compile_latex(report_tex_file, LATEX_AUX_DIR)
        if success:
            report_pdf_file = LATEX_AUX_DIR / report_tex_file.with_suffix(".pdf").name
            pdfs_to_combine.append(str(report_pdf_file.resolve()))
            logging.info("Model report compiled successfully.")
        else:
            logging.error("Failed to compile model report PDF.")
            # Log the compilation failure details
            report_fail_log = WORKING_DIR / "report_compile.log"
            try:
                 report_fail_log.write_text(compile_log, encoding='utf-8')
                 logging.error(f"Report compilation log saved to: {report_fail_log}")
            except Exception as log_e:
                 logging.error(f"Could not write report compile log: {log_e}")

    except Exception as e:
        logging.error(f"Error creating or compiling model report: {e}")

    # 2. Generate and compile the Failed Pages Appendix (if any)
    failed_pages_appendix_pdf = insert_failed_page_images_into_final_pdf()
    if failed_pages_appendix_pdf and failed_pages_appendix_pdf.exists():
        pdfs_to_combine.append(str(failed_pages_appendix_pdf.resolve()))
        logging.info("Failed pages appendix will be included.")


    # 3. Merge PDFs using pdfpages if more than one PDF exists
    if len(pdfs_to_combine) > 1:
        logging.info(f"Merging {len(pdfs_to_combine)} PDFs...")
        combined_tex_file = WORKING_DIR / "combined_document.tex"
        combined_pdf_output_file = LATEX_AUX_DIR / "combined_document.pdf"
        final_output_pdf_path = LAST_GOOD_PDF_FILE.parent / f"{LAST_GOOD_PDF_FILE.stem}_final_with_report.pdf"


        combine_tex_content = [
            "\\documentclass{article}",
            "\\usepackage{pdfpages}",
             # Add hyperref for potentially preserving links (though merging can break them)
            "\\usepackage[colorlinks=false, pdfborder={0 0 0}]{hyperref}",
            "\\hypersetup{pdfauthor={AI Translation System}, pdftitle={Translated Document with Report}}",
            "\\begin{document}"
        ]
        for pdf_path in pdfs_to_combine:
            # Ensure paths are properly formatted for LaTeX (e.g., handle spaces, use forward slashes)
            # Using resolve() should give absolute path, generally safer.
            # pdfpages usually handles paths well, but be cautious with special chars.
             combine_tex_content.append(f"\\includepdf[pages=-, fitpaper=true]{{{pdf_path}}}") # fitpaper tries to match size

        combine_tex_content.append(END_DOCUMENT_TAG)

        try:
            combined_tex_file.write_text("\n".join(combine_tex_content), encoding='utf-8')
            logging.info(f"Compiling combined document: {combined_tex_file}")
            success, compile_log = compile_latex(combined_tex_file, LATEX_AUX_DIR)

            if success and combined_pdf_output_file.exists():
                 logging.info("Combined document compiled successfully.")
                 # Create a backup of the original PDF first? Or just save with new name.
                 # Let's save with a new name to avoid overwriting intermediate good state.

                 shutil.copy(str(combined_pdf_output_file), str(final_output_pdf_path))
                 logging.info(f"Final merged PDF saved to: {final_output_pdf_path}")
                 # Optionally replace LAST_GOOD_PDF_FILE with this final one if desired workflow
                 # shutil.copy(str(combined_pdf_output_file), str(LAST_GOOD_PDF_FILE))
                 return True
            else:
                 logging.error("Failed to compile the combined PDF document.")
                 # Log the compilation failure details
                 combined_fail_log = WORKING_DIR / "combined_compile.log"
                 try:
                      combined_fail_log.write_text(compile_log, encoding='utf-8')
                      logging.error(f"Combined compilation log saved to: {combined_fail_log}")
                 except Exception as log_e:
                      logging.error(f"Could not write combined compile log: {log_e}")

                 return False # Merging failed
        except Exception as e:
            logging.error(f"Error creating or compiling combined PDF document: {e}")
            return False
    else:
         logging.info("Only the main document exists. No merging needed.")
         # Optionally rename the single PDF to the final name standard
         final_output_pdf_path = LAST_GOOD_PDF_FILE.parent / f"{LAST_GOOD_PDF_FILE.stem}_final.pdf"
         try:
              shutil.copy(str(LAST_GOOD_PDF_FILE), str(final_output_pdf_path))
              logging.info(f"Final PDF (no reports appended) saved as: {final_output_pdf_path}")
         except Exception as e:
              logging.error(f"Could not copy final PDF: {e}")
         return True # Considered success as main doc exists


def create_minimal_valid_latex(target_file=CURRENT_ATTEMPT_TEX_FILE):
    """Create a minimal valid LaTeX document that can compile."""
    minimal_latex = """\\documentclass[12pt]{article}
\\usepackage[utf8]{inputenc}
\\usepackage[T1]{fontenc}
\\usepackage{amsmath,amsthm,amssymb}
\\usepackage{graphicx}
\\usepackage[margin=1in]{geometry}
\\usepackage{xcolor}
\\usepackage{tikz-cd}  % Added for commutative diagrams

% Additional math operators and environments
\\DeclareMathOperator{\\Spin}{Spin}
\\DeclareMathOperator{\\CSpin}{CSpin}
\\DeclareMathOperator{\\SO}{SO}

\\title{Minimal Fallback Document}
\\author{AI Translation System}
\\date{\\today}

\\begin{document}
\\maketitle

\\section*{Placeholder Content}
This is a minimal valid LaTeX document created as a fallback.

\\end{document}
"""
    try:
        target_path = Path(target_file)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(minimal_latex, encoding='utf-8')
        logging.info(f"Created minimal valid LaTeX file at: {target_path}")
        return True
    except Exception as e:
        logging.error(f"Error creating minimal LaTeX file {target_path}: {e}")
        return False


def main(input_pdf=None, preferred_model=None):
    global MODEL_QUEUE, INPUT_PDF_PATH
    
    if input_pdf:
        INPUT_PDF_PATH = input_pdf
    
    # Set up model queue based on preference
    MODEL_QUEUE = setup_model_queue(preferred_model)
    
    start_time = time.time()
    logging.info(f"--- Starting PDF to LaTeX Conversion ---")
    logging.info(f"Input PDF: {INPUT_PDF_PATH}")
    logging.info(f"Working Directory: {WORKING_DIR}")
    logging.info("Using Model Tournament approach with available models.")

    # Create a minimal valid LaTeX file early as a potential fallback compile target
    create_minimal_valid_latex(CURRENT_ATTEMPT_TEX_FILE)

    # Log available models
    available_models = [f"{m['nickname']} ({m['provider']})" for m in MODEL_QUEUE if m['provider'] in api_clients]
    if not available_models:
        logging.warning("!!!!!!!! No API clients available. Set API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY) as environment variables. !!!!!!!!")
        # Allow script to continue? Or exit? Let's allow to potentially generate error report.
    else:
        logging.info(f"Available models for tournament: {', '.join(available_models)}")

    # Load model stats if they exist
    global model_stats
    if MODEL_STATS_FILE.exists():
        try:
            with open(MODEL_STATS_FILE, 'r', encoding='utf-8') as f:
                model_stats = json.load(f)
            logging.info("Loaded existing model statistics.")
        except Exception as e:
            logging.error(f"Could not load model stats from {MODEL_STATS_FILE}: {e}. Starting fresh stats.")
            model_stats = {"pages_translated": {}, "failures": {}}

    # Check input PDF
    pdf_path = Path(INPUT_PDF_PATH)
    if not pdf_path.is_file():
        logging.critical(f"CRITICAL ERROR: Input PDF not found at {pdf_path}")
        # Create a dummy PDF documenting this specific error
        error_tex_content = f"""
\\documentclass{{article}} \\usepackage{{xcolor}} \\definecolor{{errorcolor}}{{rgb}}{{0.8, 0, 0}}
\\begin{{document}} \\centering \\vspace*{{\\fill}}
{{\\Huge \\bfseries \\color{{errorcolor}} ERROR}}\\\\ \\vspace{{1cm}}
{{\\Large Input PDF file not found:}}\\\\ \\texttt{{{str(pdf_path).replace('_', '\\_')}}}
\\vspace*{{\\fill}} {END_DOCUMENT_TAG}"""
        error_tex_file = WORKING_DIR / "error_pdf_not_found.tex"
        try:
            error_tex_file.write_text(error_tex_content, encoding='utf-8')
            compile_latex(error_tex_file, LATEX_AUX_DIR)
            error_pdf = LATEX_AUX_DIR / error_tex_file.with_suffix(".pdf").name
            if error_pdf.exists():
                 shutil.copy(str(error_pdf), LAST_GOOD_PDF_FILE.parent / "ERROR_PDF_NOT_FOUND.pdf")
                 logging.info(f"Error PDF created at: {LAST_GOOD_PDF_FILE.parent / 'ERROR_PDF_NOT_FOUND.pdf'}")
        except Exception as e_report:
             logging.error(f"Could not create error report PDF: {e_report}")
        return # Stop execution

    # Get total pages
    total_pages = get_total_pages(pdf_path)
    if total_pages <= 0:
        logging.critical(f"CRITICAL ERROR: Could not get page count or PDF is invalid/empty: {pdf_path}")
        # Create a dummy PDF documenting this
        error_tex_content = f"""
\\documentclass{{article}} \\usepackage{{xcolor}} \\definecolor{{errorcolor}}{{rgb}}{{0.8, 0, 0}}
\\begin{{document}} \\centering \\vspace*{{\\fill}}
{{\\Huge \\bfseries \\color{{errorcolor}} ERROR}}\\\\ \\vspace{{1cm}}
{{\\Large Could not read page count or PDF is invalid/empty:}}\\\\ \\texttt{{{str(pdf_path).replace('_', '\\_')}}}
\\vspace*{{\\fill}} {END_DOCUMENT_TAG}"""
        error_tex_file = WORKING_DIR / "error_invalid_pdf.tex"
        try:
            error_tex_file.write_text(error_tex_content, encoding='utf-8')
            compile_latex(error_tex_file, LATEX_AUX_DIR)
            error_pdf = LATEX_AUX_DIR / error_tex_file.with_suffix(".pdf").name
            if error_pdf.exists():
                 shutil.copy(str(error_pdf), LAST_GOOD_PDF_FILE.parent / "ERROR_INVALID_PDF.pdf")
                 logging.info(f"Error PDF created at: {LAST_GOOD_PDF_FILE.parent / 'ERROR_INVALID_PDF.pdf'}")
        except Exception as e_report:
             logging.error(f"Could not create error report PDF: {e_report}")
        return # Stop execution
    logging.info(f"PDF '{pdf_path.name}' has {total_pages} pages.")

    # --- State Management and Resumption ---
    state = read_state()
    last_completed_page = state.get("last_completed_page", 0)
    preamble_generated = state.get("preamble_generated", False)
    start_page = last_completed_page + 1

    logging.info(f"Resuming processing from page: {start_page} (Last successfully completed: {last_completed_page})")

    # --- Initialize or Load Cumulative LaTeX Content ---
    # This variable stores the body of the successfully compiled LaTeX, *without* the final \end{document}
    cumulative_tex_content = ""
    if start_page > 1 and LAST_GOOD_TEX_FILE.exists():
        try:
            # Load the last known good state
            loaded_tex = LAST_GOOD_TEX_FILE.read_text(encoding='utf-8')
            # Remove the \end{document} tag for internal processing
            cumulative_tex_content = re.sub(r'\\end\{document\}', '', loaded_tex, flags=re.IGNORECASE).strip()
            # Ensure preamble flag matches loaded content (basic check)
            if cumulative_tex_content and not preamble_generated:
                 logging.warning("Loaded existing LaTeX content, but state indicates preamble wasn't generated. Forcing preamble_generated=True.")
                 preamble_generated = True
            elif not cumulative_tex_content and preamble_generated:
                 logging.warning("State indicates preamble was generated, but loaded LaTeX is empty. Resetting preamble_generated=False.")
                 preamble_generated = False
            logging.info(f"Loaded and cleaned content from {LAST_GOOD_TEX_FILE.name}")
        except Exception as e:
            logging.error(f"Error reading {LAST_GOOD_TEX_FILE.name}: {e}. Starting from scratch for page {start_page}.")
            # Reset if loading failed, but respect start_page unless it's clearly wrong
            cumulative_tex_content = ""
            if start_page > 1 :
                 logging.warning("Could not load previous content, progress might be lost if state file was also corrupted.")
                 # Decide: Reset start_page to 1 or trust state? Let's trust state for now.
                 # start_page = 1
                 # preamble_generated = False


    # --- Fallback Preamble if starting fresh ---
    # Use this only if page 1 translation *fails* to produce a preamble
    fallback_preamble = """\\documentclass[12pt]{article}
\\usepackage[utf8]{inputenc}
\\usepackage[T1]{fontenc}
\\usepackage{amsmath,amsthm,amssymb}
\\usepackage{graphicx}
\\usepackage[margin=1in]{geometry}
\\usepackage{hyperref}
\\usepackage{fancyhdr}
\\usepackage{float}
\\usepackage{xcolor}

% Define theorem-like environments
\\theoremstyle{plain}
\\newtheorem{theorem}{Theorem}[section]
\\newtheorem{lemma}[theorem]{Lemma}
\\newtheorem{proposition}[theorem]{Proposition}
\\newtheorem{corollary}[theorem]{Corollary}

\\theoremstyle{definition}
\\newtheorem{definition}[theorem]{Definition}
\\newtheorem{example}[theorem]{Example}

\\theoremstyle{remark}
\\newtheorem{remark}[theorem]{Remark}

% Set up page style
\\pagestyle{fancy}
\\fancyhf{}
\\rhead{Page \\thepage}
\\lhead{Translated Document}
\\cfoot{\\thepage}

\\title{Translated Document}
\\author{AI Translation System}
\\date{\\today}

\\begin{document}
\\maketitle
\\thispagestyle{plain}

% Optional table of contents
% \\tableofcontents
% \\newpage

% --- Start of Page 1 Content ---
"""


    # --- Process Pages ---
    for page_num in range(start_page, total_pages + 1):
        page_start_time = time.time()
        logging.info(f"--- Processing Page {page_num}/{total_pages} ---")
        page_image = None # Ensure page_image is defined in this scope
        page_latex = ""   # Ensure page_latex is defined

        try:
            # 1. Get image for current page
            try:
                page_image = get_image_from_page(pdf_path, page_num)
            except Exception as img_e:
                logging.error(f"Failed to get image for page {page_num}: {img_e}")
                page_image = None # Ensure it's None on error

            if page_image is None:
                logging.error(f"Could not get image for page {page_num}. Creating placeholder.")
                placeholder = f"""
% ========== PAGE {page_num} IMAGE EXTRACTION FAILED ==========
\\begin{{center}} \\vspace{{1cm}}
\\fbox{{\\parbox{{0.8\\textwidth}}{{ \\centering
    \\textbf{{Page {page_num} - Image Extraction Failed}}\\\\[0.5cm]
    Could not extract page image from the PDF file. This page cannot be processed.
}}}} \\vspace{{1cm}} \\end{{center}} \\clearpage
% ========== END OF PAGE {page_num} PLACEHOLDER ==========
"""
                page_latex = placeholder
                record_failed_page(page_num, "image_extraction_failed")
            else:
                # 2. Run model tournament for this page image
                context_info = {
                    "is_first_page": (page_num == 1 and not preamble_generated),
                    # Pass the current cumulative *body* content for context
                    "previous_latex": cumulative_tex_content if cumulative_tex_content else None,
                    # Cycle info will be added inside tournament function
                }
                try:
                    # This call returns the translated LaTeX (or a failure placeholder)
                    page_latex = get_latex_from_image_tournament(page_image, page_num, context_info)
                except Exception as tournament_e:
                    logging.error(f"Unexpected error calling tournament function for page {page_num}: {tournament_e}", exc_info=True)
                    # Create a placeholder for this unexpected failure
                    saved_img_path = save_image_for_failed_page(page_image, page_num)
                    page_latex = f"""
% ========== PAGE {page_num} TOURNAMENT FUNCTION ERROR ==========
% Error occurred within get_latex_from_image_tournament itself. Check logs.
% Image saved to: {saved_img_path if saved_img_path else 'Could not save image.'}
\\begin{{center}} \\vspace{{1cm}}
\\fbox{{\\parbox{{0.8\\textwidth}}{{ \\centering
    \\textbf{{Page {page_num} - Processing Error}}\\\\[0.5cm]
    A critical error occurred during the AI translation process for this page. Check logs. \\\\ Error: {str(tournament_e)[:100]}...
}}}} \\vspace{{1cm}} \\end{{center}} \\clearpage
% ========== END OF PAGE {page_num} PLACEHOLDER ==========
"""
                    record_failed_page(page_num, f"tournament_function_error: {str(tournament_e)[:100]}")


            # --- Content Combination ---
            # Always clean the cumulative content and the new page content before combining
            clean_cumulative = re.sub(r'\\end\{document\}', '', cumulative_tex_content, flags=re.IGNORECASE).strip()
            clean_page_latex = re.sub(r'\\end\{document\}', '', page_latex, flags=re.IGNORECASE).strip()

            # If it's the first page AND no preamble exists yet, the LLM response should contain it.
            if page_num == 1 and not preamble_generated:
                # Basic check: Does the generated LaTeX look like it has a preamble?
                if "\\documentclass" in clean_page_latex[:200]: # Heuristic check near start
                     current_attempt_content = clean_page_latex # Use the full output
                     preamble_generated = True # Mark preamble as generated
                     logging.info(f"Page {page_num}: Using LLM-generated preamble and content.")
                else:
                     # LLM failed to generate preamble for page 1, use fallback
                     logging.warning(f"Page {page_num}: LLM did not generate preamble. Using fallback preamble.")
                     current_attempt_content = fallback_preamble + clean_page_latex
                     preamble_generated = True # Mark preamble as generated (using fallback)

            else:
                 # Subsequent pages: Append new page content to cleaned cumulative content
                 current_attempt_content = clean_cumulative + f"\n\n% --- Page {page_num} Content ---\n" + clean_page_latex + "\n\n"


            # --- Attempt Compilation ---
            # Prepare the content with the \end{document} tag for compilation
            compilation_content = current_attempt_content.strip() + f"\n{END_DOCUMENT_TAG}\n"

            try:
                logging.info(f"Writing attempt for page {page_num} to {CURRENT_ATTEMPT_TEX_FILE.name}")
                CURRENT_ATTEMPT_TEX_FILE.write_text(compilation_content, encoding='utf-8')
            except Exception as write_e:
                logging.error(f"Failed to write {CURRENT_ATTEMPT_TEX_FILE}: {write_e}. Skipping compilation for page {page_num}.")
                # Record failure, save image, and continue
                record_failed_page(page_num, f"file_write_error: {write_e}")
                if page_image: save_image_for_failed_page(page_image, page_num)
                if page_image: del page_image # Cleanup
                continue # Move to next page


            logging.info(f"Attempting to compile LaTeX up to page {page_num}...")
            compile_ok = False
            compile_log = ""
            try:
                compile_ok, compile_log = compile_latex(CURRENT_ATTEMPT_TEX_FILE, LATEX_AUX_DIR)
            except Exception as compile_e:
                logging.error(f"Exception during compile_latex call for page {page_num}: {compile_e}", exc_info=True)
                compile_ok = False
                compile_log = f"Exception during compilation: {compile_e}"


            # --- Handle Compilation Result ---
            if compile_ok:
                logging.info(f"Page {page_num}: Successfully compiled.")
                # Update the main cumulative content (without end tag)
                cumulative_tex_content = current_attempt_content.strip()

                # Save the successfully compiled state (with end tag)
                save_content = cumulative_tex_content + f"\n{END_DOCUMENT_TAG}\n"
                try:
                    LAST_GOOD_TEX_FILE.write_text(save_content, encoding='utf-8')
                except Exception as e_write:
                     logging.error(f"Error writing {LAST_GOOD_TEX_FILE}: {e_write}")
                     # Continue, but state might be inconsistent with file

                # Copy the compiled PDF to the final output location
                compiled_pdf = LATEX_AUX_DIR / CURRENT_ATTEMPT_TEX_FILE.with_suffix(".pdf").name
                if compiled_pdf.exists():
                    try:
                        shutil.copy(str(compiled_pdf), str(LAST_GOOD_PDF_FILE))
                        logging.info(f"Updated {LAST_GOOD_PDF_FILE.name}")
                    except Exception as e_copy:
                        logging.error(f"Could not copy compiled PDF {compiled_pdf} to {LAST_GOOD_PDF_FILE}: {e_copy}")
                else:
                     logging.error(f"Compiled PDF {compiled_pdf.name} not found in aux directory after successful compile.")


                # Update and save the state file
                state["last_completed_page"] = page_num
                state["preamble_generated"] = preamble_generated
                write_state(state)

            else: # Compilation FAILED for the current page's content
                logging.warning(f"Page {page_num}: Compilation FAILED. Attempting to use placeholder.")
                record_failed_page(page_num, "compilation_failed") # Record the initial failure

                # Save the image that caused the failure (if not already saved by tournament)
                if page_image:
                     save_image_for_failed_page(page_image, page_num)

                # Create placeholder text
                placeholder = f"""
% ========== PAGE {page_num} COMPILATION FAILED ==========
% LaTeX generated for this page caused a compilation error.
% Original image saved. Check compilation log for details.
\\begin{{center}} \\vspace{{1cm}}
\\fbox{{\\parbox{{0.8\\textwidth}}{{ \\centering
    \\textbf{{Page {page_num} - Compilation Failed}}\\\\[0.5cm]
    The LaTeX code generated for this page caused a compilation error. \\\\
    A placeholder has been inserted. \\\\ Check \\texttt{{compile\\_fail\\_page\\_{page_num}.log}} for details.
}}}} \\vspace{{1cm}} \\end{{center}} \\clearpage
% ========== END OF PAGE {page_num} PLACEHOLDER ==========
"""
                # Revert to the *previous* good cumulative content (already cleaned)
                clean_cumulative_previous = re.sub(r'\\end\{document\}', '', cumulative_tex_content, flags=re.IGNORECASE).strip()

                # Create content with placeholder for re-compilation check
                placeholder_attempt_content = clean_cumulative_previous + f"\n\n% --- Page {page_num} Placeholder ---\n" + placeholder + "\n\n"
                placeholder_compile_content = placeholder_attempt_content.strip() + f"\n{END_DOCUMENT_TAG}\n"

                try:
                    CURRENT_ATTEMPT_TEX_FILE.write_text(placeholder_compile_content, encoding='utf-8')
                    logging.info(f"Attempting to compile with placeholder for page {page_num}...")
                    placeholder_ok, _ = compile_latex(CURRENT_ATTEMPT_TEX_FILE, LATEX_AUX_DIR)
                except Exception as e_ph_compile:
                     logging.error(f"Error compiling placeholder for page {page_num}: {e_ph_compile}")
                     placeholder_ok = False # Treat as failure if exception occurs

                if placeholder_ok:
                    logging.info(f"Successfully compiled with placeholder for page {page_num}.")
                    # Update cumulative content to include the placeholder (without end tag)
                    cumulative_tex_content = placeholder_attempt_content.strip()

                    # Save this placeholder state as the new "last good" state
                    save_content = cumulative_tex_content + f"\n{END_DOCUMENT_TAG}\n"
                    try:
                        LAST_GOOD_TEX_FILE.write_text(save_content, encoding='utf-8')
                    except Exception as e_write:
                         logging.error(f"Error writing {LAST_GOOD_TEX_FILE} with placeholder: {e_write}")

                    # Copy the placeholder PDF
                    compiled_pdf = LATEX_AUX_DIR / CURRENT_ATTEMPT_TEX_FILE.with_suffix(".pdf").name
                    if compiled_pdf.exists():
                        try:
                            shutil.copy(str(compiled_pdf), str(LAST_GOOD_PDF_FILE))
                            logging.info(f"Updated {LAST_GOOD_PDF_FILE.name} with placeholder for page {page_num}")
                        except Exception as e_copy:
                            logging.error(f"Could not copy placeholder PDF: {e_copy}")
                    else:
                         logging.error(f"Placeholder PDF {compiled_pdf.name} not found after successful compile.")


                    # Update state to reflect page completion (with placeholder)
                    state["last_completed_page"] = page_num
                    state["preamble_generated"] = preamble_generated # Preamble state doesn't change here
                    write_state(state)
                else:
                    # Critical failure: Even placeholder didn't compile.
                    # This suggests a problem with the *previous* cumulative content or LaTeX setup.
                    logging.error(f"CRITICAL: Failed to compile even with placeholder for page {page_num}.")
                    logging.error("This might indicate an unrecoverable LaTeX error in the document structure before this page.")
                    logging.error("Processing cannot reliably continue. Stopping.")
                    # Record a more severe failure reason
                    record_failed_page(page_num, "compilation_failed_with_placeholder_FATAL")
                    # Consider exiting or trying drastic recovery (e.g., reverting to last *PDF* compile state)
                    # For now, we stop processing.
                    if page_image: del page_image # Cleanup before potentially exiting
                    raise RuntimeError(f"Unrecoverable compilation error on page {page_num}. Previous content likely broken.")


            # --- Page Cleanup ---
            page_end_time = time.time()
            logging.info(f"Page {page_num} processed in {page_end_time - page_start_time:.2f} seconds.")
            if page_image:
                try:
                    del page_image # Free up memory
                except Exception as del_e:
                    logging.warning(f"Could not delete page image object: {del_e}")

        except Exception as page_loop_e:
            # Catch unexpected errors within the main loop for a page
            logging.error(f"Unexpected error processing page {page_num}: {page_loop_e}", exc_info=True) # Log traceback
            record_failed_page(page_num, f"unexpected_page_error: {str(page_loop_e)[:100]}")

            # Attempt to save image if it exists in context
            if 'page_image' in locals() and page_image:
                try:
                    save_image_for_failed_page(page_image, page_num)
                except Exception as save_e:
                    logging.error(f"Error saving image for page {page_num} after unexpected error: {save_e}")
                try:
                     del page_image # Cleanup
                except: pass


            # Decide whether to continue or stop on unexpected errors
            logging.warning("Continuing to the next page despite unexpected error.")
            continue # Try to process the next page


    # --- End of Processing ---
    final_completed_page = state.get("last_completed_page", 0)
    logging.info(f"--- Processing Finished ---")
    logging.info(f"Last successfully processed page (or placeholder): {final_completed_page}")

    if LAST_GOOD_TEX_FILE.exists():
        logging.info(f"Final cumulative LaTeX saved to: {LAST_GOOD_TEX_FILE}")
    else:
        logging.warning(f"Final cumulative LaTeX file not found ({LAST_GOOD_TEX_FILE}).")

    if LAST_GOOD_PDF_FILE.exists():
         logging.info(f"Final cumulative PDF saved to: {LAST_GOOD_PDF_FILE}")
    else:
         logging.warning(f"Final cumulative PDF file not found ({LAST_GOOD_PDF_FILE}).")


    # --- Generate and Append Reports ---
    try:
        logging.info("Generating final report and appending (if applicable)...")
        append_report_to_final_pdf()
    except Exception as report_e:
        logging.error(f"Error generating or appending final report: {report_e}", exc_info=True)


    # --- Final Summary ---
    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes).")

    if final_completed_page < total_pages:
        logging.warning(f"Processing stopped early. Completed {final_completed_page} out of {total_pages} pages.")
    elif final_completed_page == total_pages:
        logging.info("Successfully processed all pages (or inserted placeholders).")
    else: # Should not happen
         logging.error(f"Inconsistent state: Final completed page ({final_completed_page}) > Total pages ({total_pages})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Translate PDF document using AI models')
    parser.add_argument('input_pdf', nargs='?', default="input.pdf",
                      help='Input PDF file to translate (default: input.pdf)')
    parser.add_argument('--model', '-m', choices=list(get_model_configs().keys()),
                      help='Preferred AI model to use first (will fall back to others if failed)')
    parser.add_argument('--list-models', '-l', action='store_true',
                      help='List all available models and exit')
    
    args = parser.parse_args()
    
    if args.list_models:
        print("\nAvailable models:")
        for model_id, config in get_model_configs().items():
            print(f"  {model_id:<20} ({config['nickname']})")
        print("\nDefault queue:")
        for model in DEFAULT_MODEL_QUEUE:
            print(f"  {model}")
        sys.exit(0)
    
    main(input_pdf=args.input_pdf, preferred_model=args.model)
