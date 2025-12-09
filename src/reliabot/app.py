import gradio as gr
from dotenv import load_dotenv
from openai import AsyncOpenAI

from src.utils import (
    AsyncWeaviateKnowledgeBase,
    Configs,
    get_weaviate_async_client,
    pretty_print,
)

configs = Configs.from_env_var()
async_weaviate_client = get_weaviate_async_client(
    http_host=configs.weaviate_http_host,
    http_port=configs.weaviate_http_port,
    http_secure=configs.weaviate_http_secure,
    grpc_host=configs.weaviate_grpc_host,
    grpc_port=configs.weaviate_grpc_port,
    grpc_secure=configs.weaviate_grpc_secure,
    api_key=configs.weaviate_api_key,
)
async_openai_client = AsyncOpenAI()
async_knowledgebase = AsyncWeaviateKnowledgeBase(
    async_weaviate_client,
    collection_name="enwiki_20250520",
)

DESCRIPTION = """\
In the example below, we check our knowledge \

"""

async def search_and_pretty_format(keyword: str) -> str:
    """Search knowledgebase and pretty-format output."""
    output = await async_knowledgebase.search_knowledgebase(keyword)
    return pretty_print(output)

json_codeblock = gr.Code(language="json", wrap_lines=True)


demo = gr.Interface(
    fn=search_and_pretty_format,
    inputs=["text"],
    outputs=[json_codeblock],
    title="1.0: Knowledge Base ADR demo",
    description=DESCRIPTION,
    examples=[
        "Nananana",
        "ButtCheek",
        "Chris P. Bacon"
    ],
)

demo.launch(share=True)