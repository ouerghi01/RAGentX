import faiss
from sentence_transformers import SentenceTransformer
import time
import json
from services.agent_service import AgentInterface,User
def init_cache():
    index = faiss.IndexFlatL2(768)
    if index.is_trained:
        print("Index trained")    
    encoder = SentenceTransformer("all-mpnet-base-v2")

    return index,encoder
def retrieve_cache(json_file):
    try:
        with open(json_file, "r") as file:
            cache = json.load(file)
    except FileNotFoundError:
        cache = {"questions": [], "embeddings": [], "answers": [], "response_text": []}

    return cache
def store_cache(json_file, cache):
    with open(json_file, "w") as file:
        json.dump(cache, file)
class semantic_cache:
    def __init__(self, json_file="cache_file.json", thresold=0.35, max_response=100, eviction_policy="FIFO"):
        """Initializes the semantic cache.

        Args:
        json_file (str): The name of the JSON file where the cache is stored.
        thresold (float): The threshold for the Euclidean distance to determine if a question is similar.
        max_response (int): The maximum number of responses the cache can store.
        eviction_policy (str): The policy for evicting items from the cache.
        """
        # Initialize Faiss index with Euclidean distance
        self.index, self.encoder = init_cache()
        self.euclidean_threshold = thresold
        self.json_file = json_file
        self.cache = retrieve_cache(self.json_file)
        self.max_response = max_response
        self.eviction_policy = eviction_policy

    def evict(self):
        """Evicts an item from the cache based on the eviction policy."""
        if len(self.cache["questions"]) > self.max_response:
            if self.eviction_policy == "FIFO":
                while len(self.cache["questions"]) > self.max_response:
                    self.cache["questions"].pop(0)
                    self.cache["embeddings"].pop(0)
                    self.cache["answers"].pop(0)
                    self.cache["response_text"].pop(0)

    def ask(self, question: str, chain, compression_retriever) -> str:
        start_time = time.time()

        try:
            embedding = self.encoder.encode([question])
            self.index.nprobe = 8
            D, I = self.index.search(embedding, 1)
            if D[0] >= 0:
                if I[0][0] >= 0 and D[0][0] <= self.euclidean_threshold:
                    row_id = int(I[0][0])
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"Time taken: {elapsed_time:.3f} seconds")
                    return self.cache["response_text"][row_id]

            # If not found in cache, use compression_retriever and chain
            docs = compression_retriever.invoke(question)
            context_memory = "\n".join([msg.page_content for msg in docs])
            final_answer = chain.invoke(question)

            self.cache["questions"].append(question)
            self.cache["embeddings"].append(embedding[0].tolist())
            self.cache["answers"].append(context_memory)
            self.cache["response_text"].append(final_answer)

            print("Answer recovered from external source.")
            print(f"response_text: {final_answer}")

            self.index.add(embedding)
            self.evict()

            store_cache(self.json_file, self.cache)

            end_time = time.time()
            print(f"Time taken: {end_time - start_time:.3f} seconds")
            return final_answer

        except Exception as e:
            raise RuntimeError(f"Error during 'ask' method: {e}")



import asyncio
def rrrr():
    agent=AgentInterface()
    agent.compression_retriever=asyncio.run( agent.setup_ensemble_retrievers())
    agent.chain=agent.retrieval_chain(
    User(
        username="aziz",
        password="assistant",
        his_job="finance pro",
        hashed_password="assistant"
    )
    )
    semanticcache=semantic_cache()
    for r in range(2):
        reponse=semanticcache.ask("Help me learn about finance ",agent.chain,agent.compression_retriever)
        with open("index.html","w") as f :
                f.write(reponse)


rrrr() #https://huggingface.co/learn/cookbook/en/information_extraction_haystack_nuextract