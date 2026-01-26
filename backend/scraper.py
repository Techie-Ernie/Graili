from playwright.async_api import async_playwright
import asyncio
import subprocess 
import os 

class HolyGrailScraper:
    def __init__(self, category, subject, year=None, documentType="Exam Papers", pages=2):
        self.category = category
        self.subject = subject
        self.documentType = documentType
        self.year = year
        self.pages = pages
    async def get_documents(self):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            page = await browser.new_page()
            await page.goto(f"https://grail.moe/library", timeout=60000)
            
            # apply filters 
            filters = [self.category, self.subject, self.year, self.documentType]
            
            for i, option_name in enumerate(filters):
                if option_name:  
                    input_box = page.locator('input[role="combobox"]').nth(i)
                    await input_box.fill(option_name)
                    await asyncio.sleep(0.5)

                    await page.get_by_role("option", name=option_name).click()
                    await input_box.wait_for(state="visible") # it somehow only selects second field correctly with this line but if it works it works
                    await asyncio.sleep(0.5)
                    # should probably add additional checks that all filters have been applied correctly -- doesnt work sometimes 

            documents = {}
            for i in range(1,self.pages+1):  # get links for each page 
            
                pdf_links = await page.locator('a[href^="https://document.grail.moe/"][href$=".pdf"]').all()
               
                for link in pdf_links:
                    href = await link.get_attribute("href")
                    document_name = await link.inner_text()
                    documents[href] = document_name
        
                if i == self.pages:
                    break

                await page.get_by_role("button", name="Next").click()

                await page.wait_for_url(
                    lambda url: f"page={i+1}" in url,
                    timeout=10000
                )

            await browser.close()
            return documents 
        
    def download_documents(self, documents, download_dir='/home/ernie/grail_scraper/documents'):
        print(documents)
        for link, document_name in documents.items():
            if os.name == 'nt':
                subprocess.run(
                    ["curl", "-LO", f["url"]],
                    cwd=download_dir,
                    check=False,
                )
            else:
                subprocess.run(
                    ["wget", link],
                    cwd=download_dir,
                    check=False
                )
        
            

if __name__ == "__main__":
    scraper = HolyGrailScraper("GCE 'A' Levels", "H2 Economics", None, "Exam Papers")
    docs = asyncio.run(scraper.get_documents())
    #print(docs)
    scraper.download_documents(docs)