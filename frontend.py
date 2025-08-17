import streamlit as st
import os
import zipfile
import tempfile
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

st.title("Resume Screener")

# Get API key
cohere_api_key = os.getenv("COHERE_API_KEY") or os.getenv("cohere")
if not cohere_api_key:
    st.error("Please set COHERE_API_KEY in your environment")
    st.stop()

zip_file = st.file_uploader("Upload ZIP file with PDF resumes", type="zip")
job_description = st.text_area("Job Description", height=150)

if st.button("Analyze"):
    if not zip_file or not job_description:
        st.error("Please upload resumes and enter job description")
        st.stop()
    
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_file, "r") as z:
        z.extractall(temp_dir)
    
    documents = []
    for filename in os.listdir(temp_dir):
        if filename.endswith('.pdf'):
            try:
                file_path = os.path.join(temp_dir, filename)
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                content = "\n".join([page.page_content for page in pages])
                documents.append(Document(page_content=content, metadata={"source": filename}))
            except:
                st.warning(f"Could not read {filename}")
    
    if not documents:
        st.error("No valid PDF files found")
        st.stop()
    
    st.success(f"Loaded {len(documents)} resumes")
    
    with st.spinner("Processing..."):
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        
        embeddings = CohereEmbeddings(
            model="embed-english-v3.0",
            cohere_api_key=cohere_api_key
        )
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        llm = Cohere(
            model="command", 
            temperature=0.3,
            cohere_api_key=cohere_api_key
        )
        
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are an HR recruiter. Compare these candidates to the job requirements.
            
            Job: {question}
            Candidates: {context}
            
            For each candidate, give:
            - Name: [filename]
            - Score: [1-10]
            - Why: [brief reason]
            
            Then rank the top 3 candidates.
            """
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt}
        )
        
        result = qa_chain.invoke({"query": job_description})
    
    st.subheader("Results")
    st.write(result["result"])