from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pymupdf
from scraper import HolyGrailScraper
import glob 
from collections import Counter
import re

app = FastAPI()

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ScrapedData(BaseModel):
    text: str

class AIResult(BaseModel):
    result: str

# function to clean pdf text
def extract_clean_body_text(
    pdf_path, top_margin_ratio=0.1, bottom_margin_ratio=0.1, repetition_threshold=0.6,font_size_tolerance=1.5):

    doc = pymupdf.open(pdf_path)
    num_pages = len(doc)

    # ------------------------------------------------------------
    # Pass 1: Collect global statistics
    # ------------------------------------------------------------
    line_counter = Counter()
    font_sizes = Counter()

    for page in doc:
        page_dict = page.get_text("dict")
        for block in page_dict["blocks"]:
            for line in block.get("lines", []):
                line_text = "".join(span["text"] for span in line["spans"]).strip()
                if line_text:
                    line_counter[line_text.lower()] += 1
                for span in line["spans"]:
                    font_sizes[span["size"]] += 1

    # Most common font size = body text
    body_font_size = font_sizes.most_common(1)[0][0]

    # ------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------
    def is_repeated(text):
        return line_counter[text.lower()] / num_pages >= repetition_threshold

    def is_page_number(text):
        text = text.strip()

        # must be digits only
        if not re.fullmatch(r"\d{1,4}", text):
            return False

        # explicitly exclude bracketed content
        if "[" in text or "]" in text or "(" in text or ")" in text:
            return False

        return True


    def is_body_font(size):
        return abs(size - body_font_size) <= font_size_tolerance

    # ------------------------------------------------------------
    # Pass 2: Extract filtered body text
    # ------------------------------------------------------------
    output_pages = []

    for page in doc:
        page_height = page.rect.height
        page_dict = page.get_text("dict")
        page_lines = []

        for block in page_dict["blocks"]:  
            if block["type"] != 0:  # not text
                continue

            y0, y1 = block["bbox"][1], block["bbox"][3]

            # Positional filtering (header/footer)
            if y0 < page_height * top_margin_ratio:
                continue
            if y1 > page_height * (1 - bottom_margin_ratio):
                continue

            for line in block.get("lines", []):
                spans = line["spans"]
                text = "".join(span["text"] for span in spans).strip()

                if not text:
                    continue
                #if is_page_number(text):
                #    continue
                
                if is_repeated(text):
                    continue
                if not any(is_body_font(span["size"]) for span in spans):
                    continue

                page_lines.append(text)

        output_pages.append("\n".join(page_lines))

    full_text = "\n\n".join(output_pages)

    # Fix hyphenation and whitespace
    full_text = re.sub(r"-\n", "", full_text)
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)

    return full_text.strip()


@app.get("/test")
def test_connection():
    return {"status": "connected", "message": "FastAPI backend is running!"}

@app.get("/data")
def get_data():
    files = glob.glob('/home/ernie/grail_scraper/documents/*.pdf')
    #for file in files:
    model_prompt = ""
    
    clean_text = extract_clean_body_text(files[2])
    doc = pymupdf.open(files[2])
    for page in doc:
        print(page.get_text())
    #print(clean_text)
        model_prompt += str(page.get_text()).replace(' ', '')

    return {"text": model_prompt}
@app.post("/data")
def receive_data(data: ScrapedData):
    print("Received data:", data.text)
    return {"status": "ok"}

@app.post("/ai-result")
def receive_ai_result(result: AIResult):
    print("AI OUTPUT:", result.result)
    # trigger next step 
    return {"status": "received"}
