from dataclasses import dataclass, field
from rojak.retrievers import RetrieverActivities, Retriever
from temporalio import activity
from qdrant_client import QdrantClient
from qdrant_client.qdrant_fastembed import QueryResponse


@dataclass
class QdrantRetrieverOptions:
    url: str
    collection_name: str


@dataclass
class QdrantRetriever(Retriever):
    type: str = field(init=False, default="qdrant")


class QdrantRetrieverActivities(RetrieverActivities):
    def __init__(self, options: QdrantRetrieverOptions):
        super().__init__(options)
        self.client = QdrantClient(url=options.url)
        self.collection_name = options.collection_name

    async def retrieve(self, text: str) -> list[QueryResponse]:
        retrieval_results = self.client.query(
            collection_name=self.collection_name, query_text=text
        )
        return retrieval_results

    @activity.defn(name="qdrant_retrieve_and_combine_results")
    async def retrieve_and_combine_results(self, text: str) -> str:
        retrieval_results = await self.retrieve(text)
        return self.combine_retreival_results(retrieval_results)

    @staticmethod
    def combine_retreival_results(results: list[QueryResponse]) -> str:
        return "\n".join([result.document for result in results])
