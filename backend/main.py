"""
Pharma LBL FAQ Generator & Q&A Tool - FastAPI Backend

To set up:
1. Create virtual environment: python -m venv venv
2. Activate venv:
   - Windows: venv\\Scripts\\activate
   - Linux/Mac: source venv/bin/activate
3. Install dependencies: pip install -r requirements.txt
4. Set environment variables:
   - OPENAI_API_KEY (or your LLM API key)
5. Run server: uvicorn backend.main:app --reload
"""

import os
import uuid
import subprocess
import tempfile
import time
import random
import json
import re
from typing import List, Dict, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pdfplumber
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
try:
    import pytesseract
    from PIL import Image
    import io
    OCR_LIBRARY_AVAILABLE = True
except ImportError:
    OCR_LIBRARY_AVAILABLE = False
from sentence_transformers import SentenceTransformer
import numpy as np
from openai import OpenAI
# Google Gemini removed - using Groq only
# Groq API (fast inference) - uses OpenAI-compatible client
groq_client = None
GROQ_AVAILABLE = False
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Pharma LBL FAQ Generator & Q&A Tool")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize sentence transformer model
print("Loading sentence transformer model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!")

# Check OCR availability
OCR_AVAILABLE = False
if OCR_LIBRARY_AVAILABLE:
    try:
        pytesseract.get_tesseract_version()
        OCR_AVAILABLE = True
        print("✓ Tesseract OCR is available for image-based PDF processing.")
    except Exception as e:
        print(f"⚠ WARNING: Tesseract OCR not found. Image-based PDFs will not be processed.")
        print(f"   Install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
        print(f"   Or see INSTALL_TESSERACT.md for detailed instructions.")
        print(f"   Error: {str(e)}")
        # Try to set Tesseract path for Windows common locations
        if os.name == 'nt':  # Windows
            common_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                os.path.expanduser(r'~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'),
            ]
            for path in common_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    try:
                        pytesseract.get_tesseract_version()
                        OCR_AVAILABLE = True
                        print(f"✓ Found Tesseract at: {path}")
                        break
                    except:
                        pass
else:
    print("⚠ OCR library not available. Install pytesseract for image-based PDF support.")

# In-memory RAG index
# Structure: {label_id: {"chunks": [...], "embeddings": np.array, "metadata": [...]}}
rag_index: Dict[str, Dict] = {}

# FAQ answer cache: label_id -> list of FAQ answers
faq_cache: Dict[str, List[Dict]] = {}

# Throttling for /ask endpoint
active_ask_requests: Dict[str, float] = {}  # label_id -> timestamp of last ask call
ASK_COOLDOWN_SECONDS = 15  # Cooldown between ask requests

# Budget tracking
BUDGET_LIMIT = 1.0  # $1.00 limit
budget_spent = 0.0  # Track total spend in dollars

# OpenAI pricing (as of 2024, gpt-3.5-turbo)
# Input: $0.50 per 1M tokens, Output: $1.50 per 1M tokens
INPUT_COST_PER_1M = 0.50
OUTPUT_COST_PER_1M = 1.50

def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in dollars for API call."""
    input_cost = (input_tokens / 1_000_000) * INPUT_COST_PER_1M
    output_cost = (output_tokens / 1_000_000) * OUTPUT_COST_PER_1M
    return input_cost + output_cost

def check_budget(estimated_cost: float) -> bool:
    """Check if we can afford this API call."""
    global budget_spent
    if budget_spent + estimated_cost > BUDGET_LIMIT:
        return False
    return True

# LLM client setup - Groq (fast inference) as primary LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()  # "groq" or "openai"
openai_api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

openai_client = None
groq_client = None
GROQ_AVAILABLE = False

if LLM_PROVIDER == "groq":
    if not groq_api_key:
        print("WARNING: GROQ_API_KEY not set. LLM features will not work.")
        print("   Get your API key from: https://console.groq.com/")
    else:
        try:
            # Groq uses OpenAI-compatible API
            groq_client = OpenAI(
                api_key=groq_api_key,
                base_url="https://api.groq.com/openai/v1"
            )
            GROQ_AVAILABLE = True
            print("✓ Groq API (fast inference) configured")
            print(f"💰 Budget limit set to ${BUDGET_LIMIT:.2f}")
            print(f"📊 Current spend: ${budget_spent:.4f} / ${BUDGET_LIMIT:.2f}")
        except Exception as e:
            print(f"ERROR configuring Groq API: {str(e)}")
            GROQ_AVAILABLE = False
elif LLM_PROVIDER == "openai":
    if not openai_api_key:
        print("WARNING: OPENAI_API_KEY not set. LLM features will not work.")
    else:
        openai_client = OpenAI(api_key=openai_api_key)
        print("✓ OpenAI API configured")
        print(f"💰 Budget limit set to ${BUDGET_LIMIT:.2f}")
        print(f"📊 Current spend: ${budget_spent:.4f} / ${BUDGET_LIMIT:.2f}")
else:
    print(f"WARNING: Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Use 'groq' or 'openai'")


class GenerateFAQsRequest(BaseModel):
    label_id: str


class AskRequest(BaseModel):
    label_id: str
    question: str


def extract_text_from_pdf(pdf_file) -> List[Dict[str, any]]:
    """Extract text from PDF with page numbers. Tries pdfplumber first, then PyMuPDF as fallback."""
    pages = []
    pdf_file.seek(0)
    
    # Try pdfplumber first
    try:
        with pdfplumber.open(pdf_file) as pdf:
            total_pages = len(pdf.pages)
            if total_pages == 0:
                raise HTTPException(
                    status_code=400, 
                    detail="PDF appears to be empty or corrupted. No pages found."
                )
            
            for page_num, page in enumerate(pdf.pages, start=1):
                # Try multiple extraction methods in order of preference
                text = None
                
                # Method 1: Standard text extraction
                try:
                    text = page.extract_text()
                except:
                    pass
                
                # Method 2: Extract with layout preservation
                if not text or not text.strip():
                    try:
                        text = page.extract_text(layout=True)
                    except:
                        pass
                
                # Method 3: Extract words and join them
                if not text or not text.strip():
                    try:
                        words = page.extract_words()
                        if words:
                            text = " ".join([word.get("text", "") for word in words if word.get("text")])
                    except:
                        pass
                
                # Method 4: Character-level extraction
                if not text or not text.strip():
                    try:
                        chars = page.chars
                        if chars:
                            # Sort by y coordinate (top to bottom) then x (left to right)
                            sorted_chars = sorted(chars, key=lambda c: (-c.get('top', 0), c.get('x0', 0)))
                            text = " ".join([char.get('text', '') for char in sorted_chars if char.get('text')])
                    except:
                        pass
                
                if text and text.strip():
                    pages.append({
                        "page_number": page_num,
                        "text": text.strip()
                    })
    except HTTPException:
        raise
    except Exception as e:
        # If pdfplumber fails, we'll try PyMuPDF below
        pass
    
    # If pdfplumber didn't extract anything, try PyMuPDF (fitz) as fallback
    if not pages and PYMUPDF_AVAILABLE:
        try:
            pdf_file.seek(0)
            pdf_bytes = pdf_file.read()
            pdf_file.seek(0)
            
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            total_pages = len(doc)
            
            if total_pages == 0:
                raise HTTPException(
                    status_code=400, 
                    detail="PDF appears to be empty or corrupted. No pages found."
                )
            
            for page_num in range(total_pages):
                page = doc[page_num]
                text = page.get_text()
                
                if text and text.strip():
                    pages.append({
                        "page_number": page_num + 1,
                        "text": text.strip()
                    })
            
            doc.close()
        except HTTPException:
            raise
        except Exception as e:
            # If both methods fail, continue to try pdftotext
            pass
    
    # Try pdftotext (poppler) as another fallback
    if not pages:
        try:
            pdf_file.seek(0)
            pdf_bytes = pdf_file.read()
            pdf_file.seek(0)
            
            # Save PDF to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
                tmp_pdf.write(pdf_bytes)
                tmp_pdf_path = tmp_pdf.name
            
            # Try to use pdftotext
            try:
                # Try pdftotext command
                result = subprocess.run(
                    ['pdftotext', '-layout', tmp_pdf_path, '-'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    # pdftotext extracted text successfully
                    text = result.stdout.strip()
                    # Count pages by trying to extract page by page
                    total_pages = 1
                    try:
                        # Try to get page count
                        page_result = subprocess.run(
                            ['pdfinfo', tmp_pdf_path],
                            capture_output=True,
                            text=True,
                            timeout=10
                        )
                        if page_result.returncode == 0:
                            for line in page_result.stdout.split('\n'):
                                if 'Pages:' in line:
                                    total_pages = int(line.split(':')[1].strip())
                                    break
                    except:
                        pass
                    
                    # Split text by pages if possible, otherwise treat as single page
                    if total_pages > 1:
                        # Try to extract per page
                        for page_num in range(1, total_pages + 1):
                            page_result = subprocess.run(
                                ['pdftotext', '-f', str(page_num), '-l', str(page_num), '-layout', tmp_pdf_path, '-'],
                                capture_output=True,
                                text=True,
                                timeout=10
                            )
                            if page_result.returncode == 0 and page_result.stdout.strip():
                                pages.append({
                                    "page_number": page_num,
                                    "text": page_result.stdout.strip()
                                })
                    else:
                        # Single page or couldn't determine
                        pages.append({
                            "page_number": 1,
                            "text": text
                        })
            except FileNotFoundError:
                # pdftotext not installed, skip
                pass
            except subprocess.TimeoutExpired:
                # Timeout, skip
                pass
            except Exception as e:
                # Other error, skip
                pass
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_pdf_path)
                except:
                    pass
        except Exception as e:
            # Continue to OCR if pdftotext fails
            pass
    
    # If still no text found, try OCR on image-based PDFs
    if not pages and OCR_AVAILABLE:
        try:
            pdf_file.seek(0)
            pdf_bytes = pdf_file.read()
            pdf_file.seek(0)
            
            # Use PyMuPDF to extract images from PDF pages
            if PYMUPDF_AVAILABLE:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                total_pages = len(doc)
                
                if total_pages == 0:
                    raise HTTPException(
                        status_code=400, 
                        detail="PDF appears to be empty or corrupted. No pages found."
                    )
                
                print(f"Attempting OCR on {total_pages} page(s)...")
                
                for page_num in range(total_pages):
                    page = doc[page_num]
                    
                    # Render page as image (PNG)
                    mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR quality
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    
                    # Convert to PIL Image
                    img = Image.open(io.BytesIO(img_data))
                    
                    # Perform OCR
                    try:
                        text = pytesseract.image_to_string(img, lang='eng')
                        if text and text.strip():
                            pages.append({
                                "page_number": page_num + 1,
                                "text": text.strip()
                            })
                            print(f"OCR extracted text from page {page_num + 1}")
                    except Exception as ocr_error:
                        print(f"OCR error on page {page_num + 1}: {str(ocr_error)}")
                        # Try with different OCR settings
                        try:
                            text = pytesseract.image_to_string(img, lang='eng', config='--psm 6')
                            if text and text.strip():
                                pages.append({
                                    "page_number": page_num + 1,
                                    "text": text.strip()
                                })
                        except:
                            pass
                
                doc.close()
            else:
                # Fallback: try with pdfplumber to get page images
                with pdfplumber.open(pdf_file) as pdf:
                    total_pages = len(pdf.pages)
                    for page_num, page in enumerate(pdf.pages, start=1):
                        try:
                            # Try to get page as image
                            # Note: pdfplumber doesn't directly support image extraction
                            # This is a fallback that might not work well
                            pass
                        except:
                            pass
        except HTTPException:
            raise
        except Exception as e:
            print(f"OCR processing error: {str(e)}")
            # Don't raise error yet, we'll check if we got any pages
    
    # Final check
    if not pages:
        error_msg = "PDF could not be processed. No extractable text found."
        if OCR_AVAILABLE:
            error_msg += " OCR was attempted but failed. Please ensure:"
            error_msg += " 1) Tesseract OCR is installed on your system,"
            error_msg += " 2) The PDF contains readable text/images."
        else:
            error_msg += " This might be an image-based PDF (scanned document)."
            error_msg += " Please install Tesseract OCR for image-to-text conversion."
        raise HTTPException(status_code=400, detail=error_msg)
    
    return pages


def chunk_text(pages: List[Dict], chunk_size: int = 1000, overlap: int = 150) -> List[Dict]:
    """Chunk text into overlapping segments with page numbers."""
    chunks = []
    full_text = ""
    page_boundaries = []  # Track which page each character belongs to
    
    # Build full text and track page boundaries
    for page in pages:
        start_idx = len(full_text)
        full_text += page["text"] + "\n\n"
        end_idx = len(full_text)
        page_boundaries.append((start_idx, end_idx, page["page_number"]))
    
    # Create overlapping chunks
    start = 0
    chunk_id = 0
    while start < len(full_text):
        end = start + chunk_size
        chunk_text = full_text[start:end]
        
        # Find which pages this chunk spans
        chunk_pages = set()
        for page_start, page_end, page_num in page_boundaries:
            if not (end < page_start or start > page_end):
                chunk_pages.add(page_num)
        
        if chunk_text.strip():
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text.strip(),
                "pages": sorted(list(chunk_pages)),
                "start": start,
                "end": end
            })
            chunk_id += 1
        
        start = end - overlap
        if start >= len(full_text):
            break
    
    return chunks


def create_embeddings(chunks: List[Dict]) -> np.ndarray:
    """Create embeddings for all chunks."""
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_model.encode(texts, show_progress_bar=False)
    return embeddings


def retrieve_similar_chunks(query: str, label_id: str, top_k: int = 3) -> List[Dict]:
    """Retrieve top-k most similar chunks for a query."""
    if label_id not in rag_index:
        raise HTTPException(status_code=404, detail="Label ID not found")
    
    index_data = rag_index[label_id]
    query_embedding = embedding_model.encode([query])[0]
    
    # Compute cosine similarity
    embeddings = index_data["embeddings"]
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Return chunks with similarity scores
    results = []
    for idx in top_indices:
        chunk = index_data["chunks"][idx]
        results.append({
            **chunk,
            "similarity": float(similarities[idx])
        })
    
    return results


def call_groq_llm_raw(prompt: str) -> str:
    """Call Groq API (fast inference) - OpenAI-compatible endpoint."""
    global budget_spent
    
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq API not configured.")
    
    # Split prompt into system and user if needed
    if "CRITICAL RULES:" in prompt:
        parts = prompt.split("\n\nQuestion:", 1)
        system_prompt = parts[0] if len(parts) > 1 else "You are a helpful assistant."
        user_prompt = parts[1] if len(parts) > 1 else prompt
    else:
        system_prompt = "You are a helpful assistant."
        user_prompt = prompt
    
    # Estimate tokens
    estimated_input_tokens = len(prompt) // 4
    estimated_output_tokens = 2000
    estimated_cost = estimate_cost(estimated_input_tokens, estimated_output_tokens)
    
    # Check budget
    if not check_budget(estimated_cost):
        remaining = BUDGET_LIMIT - budget_spent
        raise HTTPException(status_code=402, detail=f"Budget limit exceeded. Remaining: ${remaining:.4f}")
    
    try:
        resp = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",  # supported model
        messages=[
        {"role": "system", "content": "You are a pharma label assistant. Answer ONLY from the provided label text."},
        {"role": "user", "content": prompt},
        ],
            temperature=0.2,
            max_tokens=512,
        )
        response = resp.choices[0].message.content.strip()
        return response
    except Exception as e:
        error_str = str(e)
        error_msg = f"Groq API error: {error_str}"
        status_code = 500
        
        # Handle specific Groq API errors
        if "quota" in error_str.lower() or "429" in error_str or "rate limit" in error_str.lower():
            status_code = 429
            error_msg = "Groq API rate limit exceeded. Please wait a moment and try again."
        elif "api_key" in error_str.lower() or "401" in error_str or "403" in error_str:
            status_code = 401
            error_msg = "Invalid Groq API key. Please check your GROQ_API_KEY in .env file."
        
        raise HTTPException(status_code=status_code, detail=error_msg)


def call_llm_with_retry(prompt: str, max_retries: int = 5) -> str:
    """Call LLM with exponential backoff retry for rate limit errors. Uses Groq or OpenAI."""
    delay = 1.0  # Start with 1 second
    
    for attempt in range(max_retries):
        try:
            if LLM_PROVIDER == "groq":
                if not groq_client:
                    raise HTTPException(status_code=500, detail="Groq API not configured.")
                return call_groq_llm_raw(prompt)
            elif LLM_PROVIDER == "openai":
                if not openai_client:
                    raise HTTPException(status_code=500, detail="OpenAI API key not configured.")
                return call_openai_llm_raw(prompt)
            else:
                raise HTTPException(status_code=500, detail=f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")
        except HTTPException as e:
            # If it's a rate limit error, retry with backoff
            if e.status_code == 429 and attempt < max_retries - 1:
                sleep_time = delay + random.uniform(0, 0.5)
                print(f"⏳ Rate limit hit (attempt {attempt + 1}/{max_retries}). Waiting {sleep_time:.1f} seconds before retry...")
                time.sleep(sleep_time)
                delay *= 2  # Exponential backoff
                continue
            raise  # Re-raise if not rate limit or final attempt
        except Exception as e:
            error_str = str(e)
            # Check if it's a rate limit error even if not HTTPException
            if ("429" in error_str or "rate limit" in error_str.lower() or "quota" in error_str.lower()) and attempt < max_retries - 1:
                sleep_time = delay + random.uniform(0, 0.5)
                print(f"⏳ Rate limit detected (attempt {attempt + 1}/{max_retries}). Waiting {sleep_time:.1f} seconds before retry...")
                time.sleep(sleep_time)
                delay *= 2
                continue
            # Convert to HTTPException if needed
            raise HTTPException(status_code=500, detail=f"LLM error: {error_str}")
    
    # If all retries failed
    raise HTTPException(
        status_code=429,
        detail="Rate limit reached after multiple retries. Please wait 1 minute and try again."
    )


def call_llm(question: str, context_chunks: List[Dict], system_prompt: Optional[str] = None) -> str:
    """Call LLM with question and context chunks. Uses Groq or OpenAI."""
    # Build prompt for single question (aggressive truncation)
    context_parts = []
    for chunk in context_chunks:
        pages_str = ", ".join([f"Page {p}" for p in chunk["pages"]])
        chunk_text = chunk['text'][:700] if len(chunk['text']) > 700 else chunk['text']  # Truncate to 700 chars
        context_parts.append(f"[{pages_str}]\n{chunk_text}")
    context = "\n\n---\n\n".join(context_parts)
    
    all_pages = set()
    for chunk in context_chunks:
        all_pages.update(chunk["pages"])
    sources_str = ", ".join([f"Page {p}" for p in sorted(all_pages)])
    
    if system_prompt is None:
        system_prompt = """You are a medical information assistant. Answer questions about pharmaceutical drug labels.
        
CRITICAL RULES:
1. Answer ONLY using the provided label text. Do not use any external knowledge.
2. If the label does not clearly contain the answer, explicitly state: "The label does not clearly provide this information."
3. Provide 2-4 short bullet points when possible.
4. Always end your answer with "Sources: [page numbers]" based on the chunks provided.
5. Be accurate and concise."""
    
    prompt = f"""{system_prompt}

Question: {question}

Label Text:
{context}

Please answer the question using ONLY the label text above. If the information is not clearly available, say so explicitly."""
    
    answer = call_llm_with_retry(prompt)
    
    # Ensure sources are included
    if "Sources:" not in answer:
        answer += f"\n\nSources: {sources_str}"
    
    return answer


# ------------------ Low-level helpers ------------------ #

def _build_batched_prompt(question_contexts: List[Dict]) -> str:
    """
    Build a single prompt for all FAQ questions + their contexts.
    question_contexts: [{ "question": str, "chunks": [ { "page_number": int, "text": str }, ... ] }, ...]
    """
    header = (
        "You are a pharma drug label assistant.\n"
        "Answer ONLY using the provided label text.\n"
        "If the label does not clearly provide the information, say exactly:\n"
        "'The label does not clearly provide this information.'\n\n"
        "There are multiple questions. For each, answer in this format:\n"
        "Q#: <question>\n"
        "A#: <2-4 short bullet points>\n"
        "Sources: Page X, Page Y\n\n"
    )

    parts = [header]

    for idx, qc in enumerate(question_contexts, start=1):
        q = qc["question"]
        chunks = qc["chunks"]
        context_snippets = []
        for i, c in enumerate(chunks):
            text = c.get("text", "")
            # Truncate each chunk to keep tokens low
            text = text[:800]
            # Handle both "page_number" and "pages" formats
            page_num = c.get("page_number") or (list(c.get("pages", []))[0] if c.get("pages") else "?")
            context_snippets.append(
                f"[Chunk {i+1}, Page {page_num}]\n{text}"
            )
        joined_context = "\n\n".join(context_snippets) or "(No context available)"

        parts.append(
            f"Question {idx}: {q}\n"
            f"Context {idx}:\n{joined_context}\n"
        )

    # Ask model to return answers as a JSON list for easier parsing
    parts.append(
        "Now answer ALL questions at once as a JSON array.\n"
        "Each item must be an object: {\"question_index\": number, \"answer\": string}.\n"
        "Do not include any extra text outside the JSON."
    )

    return "\n\n".join(parts)


def _parse_batched_answers(raw_text: str, num_questions: int) -> List[str]:
    """
    Parse JSON list produced by the model.
    Fallback: if parsing fails, return a generic fallback answer for each question.
    """
    fallback_answer = (
        "The label does not clearly provide this information.\n\n"
        "Sources: Please review the label document directly."
    )

    try:
        # Try to extract JSON from the response (might have extra text)
        # Look for JSON array pattern
        json_match = re.search(r'\[.*\]', raw_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = raw_text
        
        data = json.loads(json_str)
        if not isinstance(data, list):
            raise ValueError("Expected a JSON array")

        # Map by index
        answers = [""] * num_questions
        for item in data:
            if not isinstance(item, dict):
                continue
            idx = int(item.get("question_index", -1)) - 1
            ans = item.get("answer")
            if 0 <= idx < num_questions and isinstance(ans, str):
                answers[idx] = ans.strip()

        # Replace any missing with fallback
        answers = [a if a else fallback_answer for a in answers]
        return answers
    except Exception:
        # If anything goes wrong, just repeat the fallback for all
        return [fallback_answer for _ in range(num_questions)]


# Removed _call_gemini_with_backoff - using Groq only


# Removed _call_groq_once - using call_groq_llm_raw directly now


# ------------------ High-level batched FAQ call ------------------ #

def call_llm_for_faq_batch(question_contexts: List[Dict]) -> List[str]:
    """
    Main function used by /generate_faqs.

    Strategy:
    - Build one prompt with all questions.
    - Call Groq (or OpenAI) with retry logic.
    - Parse JSON response into individual answers.
    """
    num_q = len(question_contexts)
    if num_q == 0:
        return []

    prompt = _build_batched_prompt(question_contexts)

    # Call LLM with retry (uses Groq or OpenAI based on LLM_PROVIDER)
    try:
        response_text = call_llm_with_retry(prompt)
        return _parse_batched_answers(response_text, num_q)
    except HTTPException as e:
        # Re-raise to trigger heuristic fallback in /generate_faqs if needed
        raise


def call_openai_llm_raw(prompt: str) -> str:
    """Call OpenAI API (raw, no retry logic)."""
    global budget_spent
    
    # Split prompt into system and user if needed
    if "CRITICAL RULES:" in prompt:
        parts = prompt.split("\n\nQuestion:", 1)
        system_prompt = parts[0] if len(parts) > 1 else "You are a helpful assistant."
        user_prompt = parts[1] if len(parts) > 1 else prompt
    else:
        system_prompt = "You are a helpful assistant."
        user_prompt = prompt
    
    # Estimate tokens
    estimated_input_tokens = len(prompt) // 4
    estimated_output_tokens = 2000
    estimated_cost = estimate_cost(estimated_input_tokens, estimated_output_tokens)
    
    # Check budget
    if not check_budget(estimated_cost):
        remaining = BUDGET_LIMIT - budget_spent
        raise HTTPException(status_code=402, detail=f"Budget limit exceeded. Remaining: ${remaining:.4f}")
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        # Track usage
        usage = response.usage
        actual_cost = estimate_cost(usage.prompt_tokens, usage.completion_tokens)
        budget_spent += actual_cost
        print(f"💰 API call cost: ${actual_cost:.6f} | Total: ${budget_spent:.4f} / ${BUDGET_LIMIT:.2f}")
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        error_str = str(e)
        error_detail = ""
        error_code = ""
        error_type = ""
        
        # Try to extract error details from OpenAI API response
        try:
            if hasattr(e, 'response') and hasattr(e.response, 'json'):
                error_data = e.response.json()
                if 'error' in error_data:
                    error_detail = error_data['error'].get('message', '')
                    error_code = error_data['error'].get('code', '')
                    error_type = error_data['error'].get('type', '')
        except:
            pass
        
        # Handle specific OpenAI API errors
        if "insufficient_quota" in error_str or "insufficient_quota" in error_detail or error_code == "insufficient_quota":
            error_msg = "OpenAI API quota exceeded. Please check your billing and usage limits at https://platform.openai.com/account/billing"
            if error_detail:
                error_msg += f"\n\nDetails: {error_detail}"
            raise HTTPException(status_code=402, detail=error_msg)
        elif "invalid_api_key" in error_str.lower() or "invalid_api_key" in error_detail or error_code == "invalid_api_key":
            error_msg = "Invalid OpenAI API key. Please check your .env file and ensure OPENAI_API_KEY is set correctly."
            raise HTTPException(status_code=401, detail=error_msg)
        elif "rate_limit" in error_str.lower() or error_type == "rate_limit_exceeded" or "429" in error_str:
            error_msg = "OpenAI API rate limit exceeded. Please wait a moment and try again."
            raise HTTPException(status_code=429, detail=error_msg)
        else:
            # Generic error with more context
            error_msg = f"LLM error: {error_str}"
            if error_detail:
                error_msg += f"\n\nDetails: {error_detail}"
            raise HTTPException(status_code=500, detail=error_msg)


# Removed call_google_llm_raw - using Groq only


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF label and create RAG index."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Generate label ID
    label_id = str(uuid.uuid4())
    
    # Read file content into bytes for better compatibility with pdfplumber
    try:
        import io
        file_content = await file.read()
        pdf_file_obj = io.BytesIO(file_content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading uploaded file: {str(e)}")
    
    # Extract text from PDF
    pages = extract_text_from_pdf(pdf_file_obj)
    if not pages:
        raise HTTPException(status_code=400, detail="No text found in PDF")
    
    # Chunk text
    chunks = chunk_text(pages)
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks created from PDF")
    
    # Create embeddings
    embeddings = create_embeddings(chunks)
    
    # Store in RAG index
    rag_index[label_id] = {
        "chunks": chunks,
        "embeddings": embeddings,
        "metadata": {
            "filename": file.filename,
            "num_pages": len(pages),
            "num_chunks": len(chunks)
        }
    }
    
    return {"label_id": label_id, "num_pages": len(pages), "num_chunks": len(chunks)}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF label and create RAG index."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Generate label ID
    label_id = str(uuid.uuid4())
    
    # Read file content into bytes for better compatibility with pdfplumber
    try:
        import io
        file_content = await file.read()
        pdf_file_obj = io.BytesIO(file_content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading uploaded file: {str(e)}")
    
    # Extract text from PDF
    pages = extract_text_from_pdf(pdf_file_obj)
    if not pages:
        raise HTTPException(status_code=400, detail="No text found in PDF")
    
    # Chunk text
    chunks = chunk_text(pages)
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks created from PDF")
    
    # Create embeddings
    embeddings = create_embeddings(chunks)
    
    # Store in RAG index
    rag_index[label_id] = {
        "chunks": chunks,
        "embeddings": embeddings,
        "metadata": {
            "filename": file.filename,
            "num_pages": len(pages),
            "num_chunks": len(chunks)
        }
    }
    
    return {"label_id": label_id, "num_pages": len(pages), "num_chunks": len(chunks)}


@app.post("/generate_faqs")
async def generate_faqs(request: GenerateFAQsRequest):
    """Generate FAQs from the label using a single batched API call. Returns cached if available."""
    label_id = request.label_id
    
    if label_id not in rag_index:
        raise HTTPException(status_code=404, detail="Label ID not found")
    
    # Check cache first - return immediately if FAQs already generated
    if label_id in faq_cache:
        print(f"✓ Returning cached FAQs for label {label_id} (no API call needed)")
        return {"faqs": faq_cache[label_id], "cached": True}
    
    # Hard-coded FAQ questions
    faq_questions = [
        "What is this pdf about?",
        "What is this drug used for?",
        "Who should not take this drug?",
        "What are the common side effects?",
        "How should this drug be taken?",
        "What are the important warnings and precautions?"
    ]
    
    try:
        # 1) Build context per question
        question_contexts = []
        for question in faq_questions:
            chunks = retrieve_similar_chunks(question, label_id, top_k=3)
            question_contexts.append({
                "question": question,
                "chunks": chunks,
            })
        
        # 2) Single batched LLM call (with retry wrapper)
        print(f"🚀 Generating all {len(faq_questions)} FAQs in a single API call...")
        answers = call_llm_for_faq_batch(question_contexts)
        
        # 3) Map back to results
        results = [
            {"question": qc["question"], "answer": answer}
            for qc, answer in zip(question_contexts, answers)
        ]
        
        # Cache FAQ answers for reuse
        faq_cache[label_id] = results
        
        print(f"✓ Successfully generated {len(results)} FAQs")
        return {"faqs": results, "cached": False}
        
    except HTTPException as e:
        # Re-raise HTTP exceptions (including 429 from retry wrapper)
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during FAQ generation: {str(e)}")


def find_similar_faq(question: str, cached_faqs: List[Dict]) -> Optional[str]:
    """Check if question matches a cached FAQ (simple string similarity)."""
    question_lower = question.lower().strip()
    
    for faq in cached_faqs:
        faq_question = faq["question"].lower().strip()
        # Simple matching: check if question contains key words from FAQ or vice versa
        if question_lower == faq_question:
            return faq["answer"]
        
        # Check for similar questions (fuzzy match)
        question_words = set(question_lower.split())
        faq_words = set(faq_question.split())
        common_words = question_words & faq_words
        
        # If >50% word overlap, consider it a match
        if len(common_words) > 0 and len(common_words) / max(len(question_words), len(faq_words)) > 0.5:
            return faq["answer"]
    
    return None


@app.post("/ask")
async def ask_question(request: AskRequest):
    """Answer a custom question about the label. Checks FAQ cache first. Throttled to prevent spam."""
    global active_ask_requests
    
    label_id = request.label_id
    question = request.question.strip()
    
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    if label_id not in rag_index:
        raise HTTPException(status_code=404, detail="Label ID not found")
    
    # Throttling: Check if there's an active request for this label
    current_time = time.time()
    if label_id in active_ask_requests:
        last_call_time = active_ask_requests[label_id]
        time_since_last = current_time - last_call_time
        if time_since_last < ASK_COOLDOWN_SECONDS:
            wait_time = ASK_COOLDOWN_SECONDS - time_since_last
            raise HTTPException(
                status_code=429,
                detail=f"Please wait {int(wait_time)} seconds before asking another question. This helps prevent API rate limits."
            )
    
    # Update last call time
    active_ask_requests[label_id] = current_time
    
    # Check FAQ cache first (reuse answers to avoid API calls)
    if label_id in faq_cache:
        cached_answer = find_similar_faq(question, faq_cache[label_id])
        if cached_answer:
            print(f"✓ Reusing cached FAQ answer (no API call needed)")
            return {"answer": cached_answer, "cached": True}
    
    # Retrieve similar chunks
    chunks = retrieve_similar_chunks(question, label_id, top_k=3)
    
    # Truncate chunks aggressively
    for chunk in chunks:
        if len(chunk['text']) > 700:
            chunk['text'] = chunk['text'][:700] + "..."
    
    # Call LLM
    answer = call_llm(question, chunks)
    
    return {"answer": answer, "cached": False}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok", 
        "indexed_labels": len(rag_index),
        "budget_spent": round(budget_spent, 4),
        "budget_limit": BUDGET_LIMIT,
        "budget_remaining": round(BUDGET_LIMIT - budget_spent, 4)
    }

@app.get("/budget")
async def get_budget():
    """Get current budget status."""
    return {
        "budget_spent": round(budget_spent, 4),
        "budget_limit": BUDGET_LIMIT,
        "budget_remaining": round(BUDGET_LIMIT - budget_spent, 4),
        "usage_percentage": round((budget_spent / BUDGET_LIMIT) * 100, 2),
        "cached_labels": len(faq_cache),
        "groq_available": GROQ_AVAILABLE,
        "llm_provider": LLM_PROVIDER
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
