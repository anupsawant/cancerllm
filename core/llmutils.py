import os
import openai
import sys
from enum import Enum
from langchain.document_loaders import PyPDFLoader,PyMuPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch, Chroma, FAISS
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

from django.conf import settings
from core.models import LLMTransaction

os.environ['OPENAI_API_KEY'] = settings.OPENAI_API_KEY

class Stores(Enum):
    FAISIS = 'faisis'
    DOCMEMORY = 'docmemory'
    CHROMA = 'chroma'

class Loaders(Enum):
    PYPDF = 'pypdf'
    PYMUPDF = 'pymupdf'

class Splitters(Enum):
    CHAR = 'char'
    RECURSIVE = 'recursive'

class LLMTYPE(Enum):
    CHATGPT = 'gpt-3.5-turbo'
    CHATGPT_EXTRATOKEN = 'gpt-3.5-turbo-16k'
    GPT4 = 'gpt4'
    LLAMA = 'llama'
    LLAMA2 = 'llama2'

class LLMRecipe:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LLMRecipe, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, store=Stores.DOCMEMORY, loader=Loaders.PYPDF, splitter=Splitters.CHAR, llmtype=LLMTYPE.CHATGPT_EXTRATOKEN, context_len=10):
        if self._initialized == False:
            self.store = store.value
            self.loader = loader.value
            self.splitter = splitter.value
            self.documents = []
            self.db = None
            openai.api_key = os.environ["OPENAI_API_KEY"]
            self.setIngredients()
            self.llm = ConversationalRetrievalChain.from_llm(
                            ChatOpenAI(temperature=0.0, model_name=llmtype.value),
                            self.db.as_retriever(search_kwargs={'k':context_len}),
                            return_source_documents=True,
                            verbose=False
                        )
            self._initialized == True

    def loadPDFs(self):
        print('Loading pdfs.....')
        for file in os.listdir(settings.MEDIA_ROOT):
            if file.endswith(".pdf"):
                pdf_path = settings.MEDIA_ROOT + '/' + file
                loader = None
                if self.loader == 'pypdf':
                    loader = PyPDFLoader(pdf_path)
                elif self.loader == 'pymupdf':
                    loader = PyMuPDFLoader(pdf_path)
                else:
                    raise RuntimeError('Need to provide valid loader')
                self.documents.extend(loader.load())

    def splitText(self):
        print('Splitting text.....')
        text_splitter = None
        if self.splitter == 'char':
            text_splitter = CharacterTextSplitter(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
        elif self.splitter == 'recursive':
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
        else:
            raise RuntimeError('Need to provide valid splitter')
        self.documents = text_splitter.split_documents(self.documents)

    def createVectorDB(self):
        print('Creating Vector DB.....')
        if self.store == 'faisis':
            self.db = FAISS.from_documents(self.documents, embedding=OpenAIEmbeddings())
        elif self.store == 'docmemory':
            self.db = DocArrayInMemorySearch.from_documents(self.documents, embedding=OpenAIEmbeddings())
        else:
            raise RuntimeError('Need to provide valid store option')

    def setIngredients(self):
        self.loadPDFs()
        self.splitText()
        self.createVectorDB()

recipe = LLMRecipe()

class LLM:
    def __init__(self,ip):
        self.recipe = recipe
        self.ip = ip
        self.chat_history = [(item.query,item.response) for item in LLMTransaction.objects.filter(ip=self.ip).order_by('-created_at')[:10]]

    def run(self,query):
        result = self.recipe.llm({"question": query, "chat_history": self.chat_history})
        LLMTransaction(query=query,response=result["answer"], ip=self.ip).save()
        return result["answer"]
