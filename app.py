import streamlit as st

# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import OllamaEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.callbacks.manager import CallbackManager
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain.chat_models import ChatOllama
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from functools import wraps
# -------
import time
from IPython.display import Image
from pprint import pprint
import torch
import rich
import random
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from haystack.dataclasses import Document

from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import ComponentDevice
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.builders import PromptBuilder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

# Decorator for measuring execution time
def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"\nFunction {func.__name__} Took {total_time:.4f} seconds")
        return result

    return timeit_wrapper


@timeit
def load_chunk_data():
    # oad data from websites
    urls= ['https://csrc.nist.gov/projects/olir/informative-reference-catalog/details?referenceId=99#/', 
           'https://attack.mitre.org/',
            'https://cloudsecurityalliance.org/',
            'https://www.ftc.gov/business-guidance/small-businesses/cybersecurity/basics',
            'https://www.pcisecuritystandards.org/',
            'https://www.google.com/url?q=https://gdpr.eu/&sa=U&sqi=2&ved=2ahUKEwjJ8Ib2_6WFAxUxhYkEHQcPDYkQFnoECBoQAQ&usg=AOvVaw0wq2V0DbVTnZS1IzbdX0Os']
    docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        data = loader.load()

        # Split the loaded data
        text_splitter = CharacterTextSplitter(separator='\n',
                                        chunk_size=1000,
                                        chunk_overlap=40)

        doc = text_splitter.split_documents(data)
        docs.extend(doc)
    # load data from pdf
    loader = PyPDFLoader("23NYCRR500_0.pdf")
    pages = loader.load_and_split()

    doc = text_splitter.split_documents(pages)
    docs.extend(doc)

    raw_docs=[]

    for doc in docs:
        doc = Document(content=doc.page_content, meta=doc.metadata)
        raw_docs.append(doc)
    return raw_docs
@timeit
def  indexing_pipeline(raw_docs):
    document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
    indexing = Pipeline()
    indexing.add_component("cleaner", DocumentCleaner())
    indexing.add_component("splitter", DocumentSplitter(split_by='sentence', split_length=2))
    indexing.add_component("doc_embedder", SentenceTransformersDocumentEmbedder(model="thenlper/gte-large",
                                                                                device=ComponentDevice.from_str("cpu"),
                                                                                meta_fields_to_embed=["title"]))
    indexing.add_component("writer", DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE))

    indexing.connect("cleaner", "splitter")
    indexing.connect("splitter", "doc_embedder")
    indexing.connect("doc_embedder", "writer")
    #raw_docs = load_chunk_data()
    indexing.run({"cleaner":{"documents":raw_docs}})
    return document_store
@timeit
def rag_pipeline(document_store):
    generator = HuggingFaceLocalGenerator("HuggingFaceH4/zephyr-7b-beta",
                                 
                                 generation_kwargs={"max_new_tokens": 1000})
    generator.warm_up()
    prompt_template = """<|system|>Using the information contained in the context, give a comprehensive answer to the question.
    If the answer is contained in the context, also report the source URL.
    If the answer cannot be deduced from the context, do not give an answer.</s>
    <|user|>
    Context:
    {% for doc in documents %}
    {{ doc.content }} URL:{{ doc.meta['url'] }}
    {% endfor %};
    Question: {{query}}
    </s>
    <|assistant|>
    """
    prompt_builder = PromptBuilder(template=prompt_template)
    rag = Pipeline()
    rag.add_component("text_embedder", SentenceTransformersTextEmbedder(model="thenlper/gte-large",
                                                                        device=ComponentDevice.from_str("cpu")))
    rag.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store, top_k=5))
    rag.add_component("prompt_builder", prompt_builder)
    rag.add_component("llm", generator)

    rag.connect("text_embedder", "retriever")
    rag.connect("retriever.documents", "prompt_builder.documents")
    rag.connect("prompt_builder.prompt", "llm.prompt")
    return rag
@timeit
def get_generative_answer(query,rag):

  results = rag.run({
      "text_embedder": {"text": query},
      "prompt_builder": {"query": query}
    }
  )

  answer = results["llm"]["replies"][0]
  return answer

# Function to handle user input and generate responses
@timeit
def handle_userinput(user_question, rag):
    answer = get_generative_answer(user_question, rag)
    st.write(bot_template.replace("{{MSG}}", answer), unsafe_allow_html=True)
# Function to create a conversation chain
# @timeit
# def get_conversation_chain(vectorstore):
#     llm = ChatOllama(
#         model="llama2:70b-chat",
#         callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
#         # num_gpu=2
#     )
#     # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm, retriever=vectorstore.as_retriever(), memory=memory
#     )
#     return conversation_chain


# Function to handle user input and generate responses



# Main function
def main():
    
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Streamlit app layout
    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
    # Load and index data only once
        if "document_store" not in st.session_state:
            raw_docs = load_chunk_data()
            document_store = indexing_pipeline(raw_docs)
            st.session_state.document_store = document_store
            st.session_state.rag = rag_pipeline(document_store)
            print(user_question)
        handle_userinput(user_question, st.session_state.rag)


if __name__ == "__main__":
    main()
