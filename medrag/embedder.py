import openai
from medrag.config import OPENAI_API_KEY, EMBED_MODEL

_client = openai.OpenAI(api_key=OPENAI_API_KEY)


def embed(text: str) -> list[float]:
    resp = _client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


def embed_batch(texts: list[str]) -> list[list[float]]:
    resp = _client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]
