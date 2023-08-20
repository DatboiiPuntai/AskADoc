import os
from typing import Optional, Tuple
import gradio as gr
from threading import Lock

from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter

# os.envrion["OPENAI_API_KEY"] = "sk-..."  # Replace with your key

embeddings = OpenAIEmbeddings()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)


def predict(message, history):
    chat_history = []
    for human, ai in history:
        chat_history.append(HumanMessage(content=human))
        chat_history.append(AIMessage(content=ai))
    if chain:
        response = chain({"question": message, "chat_history": chat_history}).content
    else:
        response = "Please input a document"
    chat_history.append(HumanMessage(content=message))
    return response


def input_url(url_string: str):
    urls = list(map(str.strip, url_string.split(",")))
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    documents = text_splitter.split_documents(data)
    vectorstore = Chroma.from_documents(documents, embeddings)
    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(), vectorstore.as_retriever()
    )
    return chain

class ChatWrapper:
    def __init__(self):
        self.lock = Lock()
    def __call__(
        self, api_key: str, inp: str, history, chain
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            history = history or []
            # If chain is None, that is because no API key was provided.
            if chain is None:
                history.append((inp, "Please paste your OpenAI key to use"))
                return history, history
            # Set OpenAI key
            import openai
            openai.api_key = api_key
            # Run chain and append input.
            output = chain.run(input=inp)
            history.append((inp, output))
        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history




with gr.Blocks() as app:
    chain = gr.State(None)
    with gr.Row():
        with gr.Tab("Upload file"):
            gr.Markdown("Upload your document here. Must be a PDF.")
            file_input = gr.File()
        with gr.Tab("Input URL"):
            gr.Markdown("Input your URL here. For multiple pages, seperate by comma.")
            with gr.Row():
                url_input_textbox = gr.Textbox(
                    placeholder="https://en.wikipedia.org/wiki/Doge_(meme)",
                    show_label=False,
                )
                url_submit_button = gr.Button("Submit", scale=0)
    gr.ChatInterface(predict)

    url_submit_button.click(input_url, inputs=[url_input_textbox], outputs=[chain])

app.launch()
