"""
Multi-Document Vectorless RAG Pipeline
LLM: Groq (FREE) - llama-3.3-70b
OCR: EasyOCR with positional text extraction (reads table columns correctly)
Supports: Marksheets, Certificates, Resumes, Research Papers, Reports
"""

import os, sys, json, re, argparse, io
import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from groq import Groq

GROQ_MODEL = "llama-3.3-70b-versatile"
INDEX_FILE  = "vectorless_index.json"


# ─────────────────────────────────────────────
# PDF TEXT EXTRACTION
# ─────────────────────────────────────────────
def extract_text_pdfplumber(pdf_path: str) -> str:
    all_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text       = page.extract_text() or ""
                tables     = page.extract_tables()
                table_text = ""
                for table in tables:
                    for row in table:
                        if row:
                            clean = [str(c).strip() if c else "" for c in row]
                            table_text += " | ".join(clean) + "\n"
                all_text.append((text + "\n" + table_text).strip())
    except Exception as e:
        print(f"     [!] pdfplumber error: {e}")
    return "\n\n--- PAGE BREAK ---\n\n".join(all_text).strip()


def extract_text_easyocr(pdf_path: str) -> str:
    """
    Use EasyOCR with bounding boxes.
    Groups text into rows by Y position, sorts each row by X position.
    This preserves table column order: subject | grade_point | grade | credits
    """
    try:
        import easyocr
        import numpy as np
        from PIL import Image

        print("     Using EasyOCR (reads table columns by position)...")
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        doc    = fitz.open(pdf_path)
        pages_text = []

        for page_num, page in enumerate(doc):
            print(f"     EasyOCR page {page_num+1}/{len(doc)}...")
            pix    = page.get_pixmap(dpi=200)
            img    = Image.open(io.BytesIO(pix.tobytes("png")))
            img_np = np.array(img)

            results = reader.readtext(img_np, detail=1)

            # Group text by row (similar Y coordinate = same row)
            # Each result: (bbox, text, confidence)
            # bbox = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
            rows = {}
            for bbox, text, conf in results:
                if conf < 0.3:
                    continue
                y_center = int((bbox[0][1] + bbox[2][1]) / 2)
                x_left   = int(bbox[0][0])
                # Group into rows with 10px tolerance
                row_key  = (y_center // 10) * 10
                if row_key not in rows:
                    rows[row_key] = []
                rows[row_key].append((x_left, text.strip()))

            # Sort rows by Y, sort items within each row by X
            page_lines = []
            for y_key in sorted(rows.keys()):
                items     = sorted(rows[y_key], key=lambda x: x[0])
                line_text = "  |  ".join(item[1] for item in items)
                page_lines.append(line_text)

            pages_text.append("\n".join(page_lines))

        doc.close()
        return "\n\n--- PAGE BREAK ---\n\n".join(pages_text)

    except ImportError:
        print("     [!] EasyOCR not installed. Run: pip install easyocr")
        return ""
    except Exception as e:
        print(f"     [!] EasyOCR error: {e}")
        return ""


def extract_text_tesseract(pdf_path: str) -> str:
    try:
        import pytesseract
        from PIL import Image
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        print("     Using Tesseract OCR fallback...")
        doc   = fitz.open(pdf_path)
        texts = []
        for i, page in enumerate(doc):
            pix  = page.get_pixmap(dpi=300)
            img  = Image.open(io.BytesIO(pix.tobytes("png")))
            texts.append(pytesseract.image_to_string(img))
        doc.close()
        return "\n".join(texts).strip()
    except Exception as e:
        print(f"     [!] Tesseract error: {e}")
        return ""


def extract_text(pdf_path: str) -> str:
    """Try pdfplumber → EasyOCR → Tesseract."""
    # 1. pdfplumber (works for text-based PDFs)
    text = extract_text_pdfplumber(pdf_path)
    if len(text) > 100:
        print("     Extracted with pdfplumber")
        return text

    # 2. EasyOCR (best for scanned tables)
    text = extract_text_easyocr(pdf_path)
    if len(text) > 100:
        print("     Extracted with EasyOCR")
        return text

    # 3. Tesseract fallback
    text = extract_text_tesseract(pdf_path)
    if text:
        print("     Extracted with Tesseract")
    return text


# ─────────────────────────────────────────────
# DOCUMENT TYPE DETECTOR
# ─────────────────────────────────────────────
def detect_doc_type(text: str, filename: str) -> str:
    t = text.lower()
    f = filename.lower()
    if any(k in t for k in ["consolidated memo", "grade point", "hall ticket", "cgpa", "semester", "b.tech", "credits"]):
        return "marksheet"
    if any(k in t for k in ["provisional certificate", "this is to certify", "degree examination"]):
        return "certificate"
    if any(k in t for k in ["experience", "skills", "objective", "curriculum vitae", "resume"]):
        return "resume"
    if any(k in t for k in ["abstract", "introduction", "references", "conclusion", "doi", "journal"]):
        return "research_paper"
    if any(k in f for k in ["report", "memo", "minutes", "summary", "brief"]):
        return "report"
    return "general"


# ─────────────────────────────────────────────
# PARSE PROMPTS
# ─────────────────────────────────────────────
PARSE_PROMPTS = {
    "marksheet": """You are given OCR text extracted from an academic marksheet.
The text uses | to separate columns: subject title | grade_point | grade | credits.
Rows are grouped by semester (I YEAR I SEMESTER, I YEAR II SEMESTER, etc.)

Extract ALL data and return this exact JSON:
{
  "document_type": "marksheet",
  "student_info": {
    "name": "",
    "father_name": "",
    "hall_ticket_no": "",
    "college": "",
    "degree": "",
    "branch": "",
    "year_of_admission": "",
    "serial_no": "",
    "class_awarded": "",
    "exam_month_year": ""
  },
  "semesters": [
    {
      "year": "I YEAR",
      "semester": "I SEMESTER",
      "subjects": [
        {
          "sno": 1,
          "subject_title": "MATHEMATICS - I",
          "grade_point": 5,
          "grade": "C",
          "credits": 4.0
        }
      ]
    }
  ],
  "summary": {
    "total_credits_registered": 160,
    "total_credits_secured": 160,
    "cgpa": 7.14,
    "date_of_issue": ""
  }
}

RULES:
- grade_point is a number (5, 6, 7, 8, 9, 10) — never empty
- grade is a letter (C, B, B+, A, A+, O) — never empty
- credits is a decimal number (1.0, 1.5, 2.0, 3.0, 4.0) — never empty
- If a subject has ^ it means mandatory pass, set grade_point to null
- Extract ALL semesters from I YEAR to IV YEAR

Return ONLY valid JSON. No markdown.""",

    "certificate": """Extract ALL data from this certificate OCR text into JSON:
{
  "document_type": "certificate",
  "student_info": {
    "name": "", "father_name": "", "hall_ticket_no": "", "college": "",
    "degree": "", "branch": "", "class_awarded": "", "exam_month_year": ""
  },
  "certificate_details": {
    "certificate_type": "", "issued_date": "", "issuing_authority": "", "serial_no": ""
  }
}
Return ONLY valid JSON. No markdown.""",

    "resume": """Extract ALL data from this resume OCR text into JSON:
{
  "document_type": "resume",
  "personal_info": {"name": "", "email": "", "phone": "", "location": "", "linkedin": "", "github": ""},
  "education": [{"degree": "", "institution": "", "year": "", "cgpa_or_percentage": ""}],
  "skills": [],
  "experience": [{"role": "", "company": "", "duration": "", "description": ""}],
  "projects": [{"title": "", "description": "", "technologies": []}],
  "certifications": [],
  "achievements": []
}
Return ONLY valid JSON. No markdown.""",

    "research_paper": """Extract ALL data from this research paper OCR text into JSON:
{
  "document_type": "research_paper",
  "title": "", "authors": [], "abstract": "", "keywords": [],
  "sections": [{"heading": "", "summary": ""}],
  "conclusion": "", "references_count": null,
  "publication_info": {"journal": "", "year": "", "doi": ""}
}
Return ONLY valid JSON. No markdown.""",

    "report": """Extract ALL data from this report OCR text into JSON:
{
  "document_type": "report",
  "title": "", "author_or_org": "", "date": "", "summary": "",
  "sections": [{"heading": "", "key_points": []}],
  "conclusions": []
}
Return ONLY valid JSON. No markdown.""",

    "general": """Extract all key information from this OCR text into structured JSON.
Include: document_type, title, author, date, main_content (sections), key_facts.
Return ONLY valid JSON. No markdown."""
}


# ─────────────────────────────────────────────
# PARSE PDF
# ─────────────────────────────────────────────
def parse_pdf(pdf_path: str, client: Groq) -> dict:
    filename = Path(pdf_path).name
    print(f"     Extracting text...")
    text     = extract_text(pdf_path)
    doc_type = detect_doc_type(text, filename)
    print(f"     Detected type: {doc_type}")

    if not text:
        print("     [!] No text extracted from PDF")
        return {"_source_file": filename, "_doc_type": "general", "_raw_text": ""}

    # Mixed PDF (certificate + marksheet) → use marksheet parser
    if "this is to certify" in text.lower() and any(k in text.lower() for k in ["cgpa", "semester", "grade"]):
        doc_type = "marksheet"
        print("     Mixed PDF detected, using marksheet parser")

    prompt = f"""{PARSE_PROMPTS[doc_type]}

OCR Text from document:
{text[:8000]}

Return ONLY valid JSON."""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are a document parser. Return only valid JSON. No markdown, no explanation."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(raw)
        parsed["_source_file"] = filename
        parsed["_doc_type"]    = doc_type
        return parsed
    except json.JSONDecodeError as e:
        print(f"     [!] JSON parse error: {e}")
    print("     Retrying once...")

    # 🔁 Retry LLM one more time
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "Return ONLY valid JSON. Fix any errors."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(raw)
        parsed["_source_file"] = filename
        parsed["_doc_type"]    = doc_type
        return parsed
    except:
        print("     [!] Retry also failed, saving as raw")
        return {"_source_file": filename, "_doc_type": doc_type, "_raw_text": text[:2000]}
    # except json.JSONDecodeError as e:
    #     print(f"     [!] JSON parse error: {e}")
    #     print(f"     Raw: {raw[:300]}")
    #     return {"_source_file": filename, "_doc_type": doc_type, "_raw_text": text[:2000]}


# ─────────────────────────────────────────────
# INDEX MANAGEMENT
# ─────────────────────────────────────────────
def load_index() -> dict:
    if not os.path.exists(INDEX_FILE):
        return {"students": {}, "documents": []}
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_index(index: dict):
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    print(f"Index saved -> {INDEX_FILE}")

def get_student_name(parsed: dict):
    for key in ["student_info", "personal_info"]:
        if key in parsed:
            name = parsed[key].get("name", "").strip()
            if name:
                return name.upper()
    return None

def add_to_index(parsed: dict, index: dict):
    doc_type = parsed.get("_doc_type", "general")
    name     = get_student_name(parsed)
    if name and doc_type in ("marksheet", "certificate", "resume"):
        if name not in index["students"]:
            index["students"][name] = {}
        index["students"][name][doc_type] = parsed
        print(f"     Indexed under student: {name} -> {doc_type}")
    else:
        index["documents"].append(parsed)
        print(f"     Indexed as: {doc_type}")


# ─────────────────────────────────────────────
# QUERY ROUTER
# ─────────────────────────────────────────────
def route_query(query: str, index: dict):
    q     = query.lower()
    names = list(index.get("students", {}).keys())
    matched = [n for n in names if n.split()[0].lower() in q or n.lower() in q]
    comparative = any(k in q for k in [
        "highest", "lowest", "best", "worst", "top", "rank", "compare",
        "who has", "who got", "all students", "everyone", "list all"
    ])
    if comparative or len(matched) > 1: return "compare", names
    elif len(matched) == 1:             return "single_student", matched
    elif any(k in q for k in ["paper", "report", "research", "document"]): return "general_docs", []
    else:                               return "all", names

def build_context(query: str, index: dict):
    route, students = route_query(query, index)
    print(f"  Route: [{route}]")
    if route == "single_student":
        ctx = {"student": index["students"].get(students[0], {})}
    elif route == "compare":
        summary = {}
        for name in students:
            s    = index["students"].get(name, {})
            ms   = s.get("marksheet", {})
            cert = s.get("certificate", {})
            summary[name] = {
                "cgpa":          ms.get("summary", {}).get("cgpa"),
                "class":         ms.get("student_info", {}).get("class_awarded") or cert.get("student_info", {}).get("class_awarded"),
                "branch":        ms.get("student_info", {}).get("branch"),
                "college":       ms.get("student_info", {}).get("college"),
                "total_credits": ms.get("summary", {}).get("total_credits_secured"),
                "exam_year":     ms.get("student_info", {}).get("exam_month_year"),
            }
        ctx = {"comparison_summary": summary, "full_data": {n: index["students"][n] for n in students}}
    elif route == "general_docs":
        ctx = {"documents": index.get("documents", [])}
    else:
        ctx = index
    return json.dumps(ctx, indent=2), route


# ─────────────────────────────────────────────
# ANSWER GENERATION
# ─────────────────────────────────────────────
def ask(query: str, client: Groq) -> str:
    index = load_index()
    if not index["students"] and not index["documents"]:
        return "No documents indexed yet. Run: python rag_groq.py ingest <pdf>"
    context, _ = build_context(query, index)
    system = """You are a helpful academic assistant with access to structured student documents.
For comparisons: be specific, use names, and format clearly.
For single student: be detailed and precise.
Always mention which document your answer comes from.
If info is missing, say so clearly."""
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": f"--- DATA ---\n{context}\n--- END ---\n\nQuestion: {query}\nAnswer:"}
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# INGEST
# ─────────────────────────────────────────────
def ingest(pdf_paths: list, client: Groq):
    index = load_index()
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            print(f"  [!] File not found: {pdf_path}"); continue
        print(f"\n  Parsing: {pdf_path}")
        parsed = parse_pdf(pdf_path, client)
        add_to_index(parsed, index)
    save_index(index)
    print(f"\nIndex Summary:")
    print(f"   Students indexed : {len(index['students'])}")
    for name, docs in index["students"].items():
        print(f"      * {name}: {', '.join(docs.keys())}")
    print(f"   Other documents  : {len(index['documents'])}")


# ─────────────────────────────────────────────
# LIST / REMOVE
# ─────────────────────────────────────────────
def list_index():
    index = load_index()
    print(f"\nStudents ({len(index['students'])}):")
    for name, docs in index["students"].items():
        print(f"  * {name}")
        for dtype, d in docs.items():
            print(f"      [{dtype}] {d.get('_source_file','')}")
    print(f"\nOther Documents ({len(index['documents'])}):")
    for d in index["documents"]:
        print(f"  * [{d.get('_doc_type','?')}] {d.get('_source_file','')} - {d.get('title','')}")
def remove(name_or_file: str):
    index = load_index()
    key   = name_or_file.upper()
    if key in index["students"]:
        del index["students"][key]; save_index(index); print(f"Removed: {key}")
    else:
        before = len(index["documents"])
        index["documents"] = [d for d in index["documents"] if d.get("_source_file","") != name_or_file]
        if len(index["documents"]) < before: save_index(index); print(f"Removed: {name_or_file}")
        else: print(f"[!] Not found: {name_or_file}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Vectorless RAG - Groq + EasyOCR (FREE)")
    sub    = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("ingest"); p.add_argument("pdfs", nargs="+")
    p = sub.add_parser("ask");    p.add_argument("question")
    sub.add_parser("chat"); sub.add_parser("list"); sub.add_parser("show-index")
    p = sub.add_parser("remove"); p.add_argument("name_or_file")
    args = parser.parse_args()

    if args.command == "list":       list_index(); return
    if args.command == "show-index": print(json.dumps(load_index(), indent=2)); return
    if args.command == "remove":     remove(args.name_or_file); return

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        sys.exit("Set GROQ_API_KEY first:\n  $env:GROQ_API_KEY='gsk_...'")

    client = Groq(api_key=api_key)

    if args.command == "ingest":
        ingest(args.pdfs, client)
    elif args.command == "ask":
        print(ask(args.question, client))
    elif args.command == "chat":
        index    = load_index()
        students = list(index["students"].keys())
        print(f"Vectorless RAG (Groq + EasyOCR) | {len(students)} student(s) indexed")
        if students: print(f"   Students: {', '.join(students)}")
        print("Type 'quit' to exit.\n")
        while True:
            try: q = input("You: ").strip()
            except (EOFError, KeyboardInterrupt): break
            if not q or q.lower() in {"quit", "exit"}: break
            print(f"\nAssistant: {ask(q, client)}\n")

if __name__ == "__main__":main()

# python rag_vectorless.py ingest "memos.pdf"
#python rag_vectorless.py chat
# 2. Add friends' PDFs anytime
#python rag_vectorless.py ingest "friend1.pdf" "friend2.pdf"



# 4. See who is indexed
#python rag_vectorless.py list

# 5. Remove someone
#python rag_vectorless.py remove "STUDENT NAME"