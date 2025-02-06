from enum import Enum
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

from danswer.configs.app_configs import INDEX_BATCH_SIZE
from danswer.configs.constants import DocumentSource
from danswer.connectors.interfaces import GenerateDocumentsOutput, LoadConnector
from danswer.connectors.models import Document, Section
from danswer.file_processing.html_utils import web_html_cleanup
from danswer.utils.logger import setup_logger

logger = setup_logger()

# Enum for different GEORISQUES connector settings
class GEORISQUES_CONNECTOR_VALID_SETTINGS(str, Enum):
    SINGLE = "single"  # Index only the given page

# Base URL for Georisques
BASE_URL = "https://www.georisques.gouv.fr"
MAX_PAGES_TO_VISIT = 1000

def start_playwright():
    """Initialize playwright browser"""
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context(
        viewport={"width": 1920, "height": 1080},
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36"
    )
    return playwright, context

class GeorisquesConnector(LoadConnector):
    def __init__(
        self,
        base_url: str = BASE_URL,
        georisques_connector_type: str = GEORISQUES_CONNECTOR_VALID_SETTINGS.SINGLE.value,
        batch_size: int = INDEX_BATCH_SIZE,
    ) -> None:
        self.batch_size = batch_size
        self.loaded_count = 0
        self.base_url = base_url
        self.to_visit_list = [base_url]

        if georisques_connector_type != GEORISQUES_CONNECTOR_VALID_SETTINGS.SINGLE.value:
            raise ValueError("Only 'single' type is supported for this connector.")

    def load_credentials(self, credentials: dict[str, Any]) -> dict[str, Any] | None:
        if credentials:
            logger.warning("Unexpected credentials provided for Georisques Connector")
        return None

    def load_from_state(self) -> GenerateDocumentsOutput:
        visited_links: set[str] = set()
        to_visit: list[str] = self.to_visit_list.copy()

        if not to_visit:
            raise ValueError("No URLs to visit")

        loaded_count: int = self.loaded_count
        doc_batch: list[Document] = []
        restart_playwright = False
        at_least_one_doc = False
        last_error = None

        playwright, context = start_playwright()

        while to_visit:
            loaded_count += 1
            if loaded_count > MAX_PAGES_TO_VISIT:
                logger.warning("Stopping after visiting 1000 URLs to avoid infinite loops")
                break

            current_url = to_visit.pop()
            if current_url in visited_links:
                continue
            visited_links.add(current_url)

            try:
                page = context.new_page()
                page_response = page.goto(current_url)
                content = page.content()
                soup = BeautifulSoup(content, "html.parser")

                # Extract text content
                parsed_html = web_html_cleanup(soup)
                
                # Create document
                doc_batch.append(
                    Document(
                        id=current_url,
                        sections=[Section(link=current_url, text=parsed_html.cleaned_text)],
                        source=DocumentSource.GEORISQUES,
                        semantic_identifier=parsed_html.title or current_url,
                        metadata={},
                    )
                )

                # Look for PDF links
                for link in soup.find_all("a", href=True):
                    href = link.get("href")
                    if href and href.endswith('.pdf'):
                        pdf_url = urljoin(current_url, href)
                        if pdf_url not in visited_links:
                            to_visit.append(pdf_url)

                page.close()
                at_least_one_doc = True

            except Exception as e:
                last_error = f"Failed to fetch '{current_url}': {e}"
                logger.error(last_error)
                continue

            if len(doc_batch) >= self.batch_size:
                yield doc_batch
                doc_batch = []

        if doc_batch:
            yield doc_batch

        playwright.stop()

        if not at_least_one_doc:
            if last_error:
                raise RuntimeError(last_error)
            raise RuntimeError("No valid pages found.")