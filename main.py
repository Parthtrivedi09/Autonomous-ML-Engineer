from models.llm import llm
from models.embeddings import embeddings


def main():

    print("Testing LLM...")

    response = llm.invoke("Who is Virat Kohli?")

    print(response.content)

    print()

    print("Testing Embeddings...")

    embedding = embeddings.embed_query("Who is Virat Kohli?")

    print("Embedding Dimension:", len(embedding))

    print(embedding[:10])


if __name__ == "__main__":
    main()