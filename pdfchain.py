# import
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter

# load the document and split it into chunks
loader = TextLoader("temp/123.txt")
documents = loader.load()

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
docs = text_splitter.split_documents(documents)

from langchain_openai import OpenAIEmbeddings
__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(docs, embeddings)

# query it
query = "chromadb"
docs = db.similarity_search(query)

# print results
print(docs[0].page_content)

from langchain_community.document_loaders import UnstructuredFileLoader
def load_pdf(pdf_path):
    loader = UnstructuredFileLoader(pdf_path)
    docs = loader.load()
    return docs
docs= load_pdf("temp/sample-pdf.pdf")
print (f'You have {len(docs)} document(s) in your data')
print (f'There are {len(docs[0].page_content)}characters in your document')


from langchain_community.document_loaders import PyPDFLoader
#loader = PyPDFLoader("sample-pdf.pdf")
#pages = loader.load_and_split()
#print(pages)
loader = PyPDFLoader("temp/sample-pdf.pdf", extract_images=True)
pages = loader.load()

print(pages[0].page_content)


import PyPDF2
def extract_pages(pdf_path, start_page, end_page):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        pages_to_extract = list(range(start_page - 1, end_page))
        extracted_text = ""
        for page_num in pages_to_extract:
            page = reader.pages[page_num]
            extracted_text += page.extract_text()
        return extracted_text

pdf_path = "temp/xxx.pdf"
start_page = 2
end_page = 3
text = extract_pages(pdf_path, start_page, end_page)
print(text)

# pip install rapidocr-onnxruntime
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("temp/xxx.pdf", extract_images=True)
pages = loader.load()
pages[4].page_content

from langchain_community.document_loaders import MathpixPDFLoader
loader = MathpixPDFLoader("temp/xxx.pdf")
data = loader.load()

from langchain_community.document_loaders import UnstructuredPDFLoader
loader = UnstructuredPDFLoader("temp/xxx.pdf")
data = loader.load()


from langchain_community.document_loaders import OnlinePDFLoader
loader = OnlinePDFLoader("https://arxiv.org/pdf/2302.03803.pdf")
data = loader.load()

print(data)


from langchain_community.document_loaders import PyPDFium2Loader
loader = PyPDFium2Loader("temp/xxx.pdf")
pages = loader.load()
print(pages[1])


from langchain_community.document_loaders import PDFPlumberLoader
loader = PDFPlumberLoader("temp/xxx.pdf",extract_images=True)
data = loader.load()
data[1]


from sentence_transformers import SentenceTransformer
model = SentenceTransformer('moka-ai/m3e-base')

#Our sentences we like to encode
sentences =['xxxxx?' ,
]
#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)
len(embeddings)
len(embeddings[0])


from layoutparser import TesseractAgent
ocr_agent = TesseractAgent()
text = ocr_agent.detect('temp/test.jpg')
print(text)


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig


import torch
from modelscope import snapshot_download, Model
model_dir = snapshot_download("baichuan-inc/Baichuan2-7B-Chat", revision='master')
model_dir

model = Model.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, torch_dtype=torch.float32)


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Load pdf
loader = PyPDFLoader("temp/baichuan.pdf")
data = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(data[:6])
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=splits, embedding=embeddings)

import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chat_models import ChatOpenAI

question = "what is baichuan2?"
llm = ChatOpenAI(temperature=0)
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(), llm=llm
)

docs = retriever_from_llm.get_relevant_documents(query=question)
len(docs)

print("docs:",docs)

