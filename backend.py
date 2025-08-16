from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os, zipfile, shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings

load_dotenv()
cohere_api_key = os.getenv("cohere")
os.environ["COHERE_API_KEY"] = cohere_api_key

app = FastAPI()

@app.post("/analyze")
async def analyze(job_description: str = Form(...), file: UploadFile = File(...)):
    extract_dir = "pdfs"
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)


    zip_path = "uploaded.zip"
    with open(zip_path, "wb") as f:
        f.write(await file.read())
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)


    docs = []
    for filename in os.listdir(extract_dir):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(extract_dir, filename))
            pages = loader.load()
            text = "\n".join([p.page_content for p in pages])
            docs.append(Document(page_content=text, metadata={"source": filename}))


    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=cohere_api_key)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    retriever = vectorstore.as_retriever()

    llm = Cohere(model="command", temperature=0.3, max_tokens=500)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="Analyze candidates for the job: {question}\nCandidates: {context}\nGive top 3 with scores."
    )

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type_kwargs={"prompt": prompt})
    result = qa_chain.invoke({"query": job_description})

    return JSONResponse(content={"analysis": result["result"]})
