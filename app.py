from typing import Optional, Tuple

from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter

import os
from typing import Optional, Tuple

import requests
import gradio as gr

from threading import Lock

embeddings = OpenAIEmbeddings()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

def load_chain(vectorstore: Chroma):
    """Logic for loading the chain you want to use should go here."""
    llm = OpenAI(temperature=0)
    chain = ConversationalRetrievalChain.from_llm(llm, retriever=vectorstore.as_retriever())
    return chain


def is_valid_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for non-200 status codes
        return True
    except (requests.exceptions.RequestException, requests.exceptions.HTTPError):
        return False


def input_url(url_string: str):
    urls = list(map(str.strip, url_string.split(",")))
    for url in urls:
        if not is_valid_url(url):
            raise ValueError(f"Invalid URL: {url}")
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    documents = text_splitter.split_documents(data)
    vectorstore = Chroma.from_documents(documents, embeddings)
    chain = load_chain(vectorstore)
    return chain


class ChatWrapper:
    def __init__(self):
        self.lock = Lock()

    def __call__(
        self,
        inp: str,
        history: Optional[Tuple[str, str]],
        chain,
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            history = history or []
            # If chain is None, that is because no API key was provided.
            if chain is None:
                history.append((inp, "Please input a document"))
                return history, history
            # Run chain and append input.
            output = chain({"question": inp, "chat_history": history})['answer']
            history.append((inp, output))
        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history


chat = ChatWrapper()

block = gr.Blocks()

with block:
    with gr.Row():
        gr.Markdown("<h3><center>LangChain Demo</center></h3>")

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
    chatbot = gr.Chatbot()
    with gr.Row():
        message = gr.Textbox(
            label="What's your question?",
            placeholder="What's the answer to life, the universe, and everything?",
            lines=1,
        )
        submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

    gr.Examples(
        examples=[
            "Hi! How's it going?",
            "What should I do tonight?",
            "Whats 2 + 2?",
        ],
        inputs=message,
    )

    state = gr.State()
    agent_state = gr.State()

    submit.click(
        chat,
        inputs=[message, state, agent_state],
        outputs=[chatbot, state],
    )
    message.submit(
        chat,
        inputs=[message, state, agent_state],
        outputs=[chatbot, state],
    )
    url_submit_button.click(
        input_url, 
        inputs=[url_input_textbox], 
        outputs=[agent_state]
    )

block.launch(debug=True)
