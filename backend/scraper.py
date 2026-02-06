from playwright.async_api import async_playwright
import asyncio
import subprocess 
import os 
import pymupdf
import shutil 
from pathlib import Path

class HolyGrailScraper:
    def __init__(self, category, subject, year=None, documentType="Exam Papers", pages=1, headless=True):
        self.category = category
        self.subject = subject
        self.documentType = documentType
        self.year = year
        self.pages = pages
        self.headless = headless
        self.documents = []
        self.current_document_name = None
        self.current_source_link = None

    def set_current_document(self, source_link, document_name):
        self.current_source_link = source_link
        self.current_document_name = document_name

    def _ensure_documents_cached(self):
        if self.documents:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            new_loop = asyncio.new_event_loop()
            try:
                new_loop.run_until_complete(self.get_documents())
            finally:
                new_loop.close()
        else:
            asyncio.run(self.get_documents())
    async def get_documents(self):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)
            page = await browser.new_page()
            params = [
                ("category", self.category),
                ("subject", self.subject),
                ("year", self.year),
                ("doc_type", self.documentType)
            ]
            query = "&".join(f"{k}={v.replace(' ', '+')}" for k, v in params if v)
            link = f"https://grail.moe/library?{query}"            
            await page.goto(link, timeout=60000, wait_until="domcontentloaded")
            try:
                await page.wait_for_selector(
                    'a[href^="https://document.grail.moe/"][href$=".pdf"]',
                    timeout=30000
                )
            except Exception as exc:
                print(f"Scraper warning: {type(exc).__name__}: {exc}")

            """
            for i, option_name in enumerate(filters):
                if option_name:  
                    input_box = page.locator('input[role="combobox"]').nth(i)
                    await input_box.fill(option_name)
                    await asyncio.sleep(0.5)

                    await page.get_by_role("option", name=option_name).click()
                    await input_box.wait_for(state="visible") # it somehow only selects second field correctly with this line but if it works it works
                    await asyncio.sleep(0.5)
                    # should probably add additional checks that all filters have been applied correctly -- doesnt work sometimes 
            """
            documents = {}
            for i in range(1,self.pages+1):  # get links for each page 
            
                pdf_links = await page.locator('a[href^="https://document.grail.moe/"][href$=".pdf"]').all()
               
                for link in pdf_links:
                    href = await link.get_attribute("href")
                    document_name = await link.inner_text()
                    documents[href] = document_name
                    self.documents.append({
                        "source_link": href,
                        "document_name": document_name,
                    })
        
                if i == self.pages:
                    break

                next_button = page.get_by_role("button", name="Next")
                if await next_button.count() == 0 or not await next_button.is_enabled():
                    break
                await next_button.click(no_wait_after=True)
                try:
                    await page.wait_for_url(
                        lambda url: f"page={i+1}" in url,
                        timeout=15000
                    )
                except Exception:
                    break

            await browser.close()
            return documents 
        
    def download_documents(
        self,
        documents,
        download_root=None,
        subject_label=None,
    ):
        if not download_root:
            project_root = Path(__file__).resolve().parents[1]
            download_root = os.environ.get("DOCUMENTS_ROOT", str(project_root / "documents"))
        subject_name = subject_label if subject_label else self.subject
        safe_subject = "".join(
            ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in subject_name.strip()
        ) or "unknown"
        download_dir = os.path.join(download_root, safe_subject)
        os.makedirs(download_dir, exist_ok=True)
        ans_dir = f"{download_dir}/answer_key"
        question_dir = f"{download_dir}/question_paper"
        os.makedirs(ans_dir, exist_ok=True)
        os.makedirs(question_dir, exist_ok=True)
        for link, document_name in documents.items():
            self.set_current_document(link, document_name)
            document_name = document_name + ".pdf"
            if not os.path.exists(f"{ans_dir}/{document_name}") and not os.path.exists(f"{question_dir}/{document_name}"):
                subprocess.run(
                    ["wget", link, "-O", document_name],
                    cwd=download_dir,
                    check=False
                )
                pdf = pymupdf.open(f"{download_dir}/{document_name}")
                ans = ['markscheme', 'answerkey', 'answersheet', "suggestedanswers", "examinersreportf"]
                if any(word in pdf[0].get_text().strip().replace(' ', '').lower() for word in ans) or any(word in document_name.strip().replace(' ', '').lower() for word in ans):
                    # check if file is answer key using the file name or first page of pdf 
                    # maybe can remove first page check? but it barely adds any latency
                    os.makedirs(ans_dir, exist_ok=True)
                    target_path = f"{ans_dir}/{document_name}"
                    shutil.move(f"{download_dir}/{document_name}", target_path)
                else:
                    print('question paper')
                    os.makedirs(question_dir, exist_ok=True)
                    target_path = f"{question_dir}/{document_name}"
                    shutil.move(f"{download_dir}/{document_name}", target_path)
            else:
                print(f"Document {document_name} already downloaded")
        
    def get_scraper_context(self):
        source_link = self.current_source_link
        document_name = self.current_document_name
        if not source_link or not document_name:
            self._ensure_documents_cached()
            if self.documents:
                source_link = self.documents[0].get("source_link")
                document_name = self.documents[0].get("document_name")
        return {
            'year': self.year,
            'subject': self.subject,
            'category': self.category,
            'question_type': 'exam',
            'source_link': source_link,
            'document_name': document_name
        }

if __name__ == "__main__":
    scraper = HolyGrailScraper("GCE 'A' Levels", "H2 Economics", None, "Exam Papers", pages=3)
    docs = asyncio.run(scraper.get_documents())
    #print(docs)
    scraper.download_documents(docs)
