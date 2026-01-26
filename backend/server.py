from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pymupdf
from scraper import HolyGrailScraper
import glob 

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

@app.get("/test")
def test_connection():
    return {"status": "connected", "message": "FastAPI backend is running!"}

@app.get("/data")
def get_data():
    files = glob.glob('/home/ernie/grail_scraper/documents/*.pdf')
    #for file in files:
    doc = pymupdf.open(files[2])
    model_prompt = ""
    for page in doc: # iterate the document pages
        text = page.get_text() # get plain text encoded as UTF-8
        model_prompt += str(text)
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
