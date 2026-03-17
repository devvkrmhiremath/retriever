import os
from openai import AsyncAzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from dotenv import load_dotenv

load_dotenv()

class AzureClientFactory:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AzureClientFactory, cls).__new__(cls)
            cls._instance._openai_client = None
            cls._instance._search_index_client = None
            cls._instance._search_clients = {}
        return cls._instance

    @property
    def openai_client(self) -> AsyncAzureOpenAI:
        if self._openai_client is None:
            self._openai_client = AsyncAzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
            )
        return self._openai_client

    @property
    def search_index_client(self) -> SearchIndexClient:
        if self._search_index_client is None:
            self._search_index_client = SearchIndexClient(
                endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
                credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY"))
            )
        return self._search_index_client

    def get_search_client(self, index_name: str) -> SearchClient:
        if index_name not in self._search_clients:
            self._search_clients[index_name] = SearchClient(
                endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
                index_name=index_name,
                credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY"))
            )
        return self._search_clients[index_name]

# Singleton instance
clients = AzureClientFactory()
