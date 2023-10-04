from llama_index import VectorStoreIndex, GithubRepositoryReader, ServiceContext
import logging
import sys
from llama_index import StorageContext, load_index_from_storage, SimpleDirectoryReader
from llama_index.llms import OpenAI
import os

def enable_llama_index_logging():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def enable_openai_call_logging():
    import llama_index.llms.openai
    llama_index.llms.openai.log = "debug"


def get_documents_from_dir():
    return SimpleDirectoryReader(input_dir='../llama_index', recursive=True, required_exts=[".txt", ".yaml", ".rst", ".py", ".md", ".ipynb", ".html"]).load_data()


def get_documents_from_github():
    github_token = os.environ.get("GITHUB_TOKEN")
    return GithubRepositoryReader(
        owner="run-llama",
        repo="llama_index",
        use_parser=False,
        verbose=False,
        ignore_file_extensions=[".pdf", ".png", ".pack", ".csv", ".idx", ".html", ".json"],
        github_token=github_token
    ).load_data(branch="main")


def get_index_from_disk():
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    # load index
    return load_index_from_storage(storage_context)


def save_index_to_disk(index):
    index.storage_context.persist()


def build_index_from_source_and_persist():
    documents = get_documents_from_dir()
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    save_index_to_disk(index)
    return index


def load_or_build_index():
    if os.path.exists("./storage"):
        return get_index_from_disk()
    else:
        return build_index_from_source_and_persist()


def get_query_engine(index):
    service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo"))
    return index.as_query_engine(service_context=service_context)


def main():
    enable_llama_index_logging()

    index = load_or_build_index()

    query_engine = get_query_engine(index)

    print("please ask a question of the query engine:")
    while True:
        human_input = input()
        llm_response = query_engine.query(human_input)
        print(llm_response)


main()
