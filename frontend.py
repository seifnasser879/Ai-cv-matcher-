import streamlit as st
<<<<<<< HEAD
import os
import zipfile
import tempfile
=======
import os, zipfile
>>>>>>> a34be679f524b38afb65f28ef48725a7386ce51c
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
<<<<<<< HEAD

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
=======
cohere_api_key = os.getenv("cohere")
os.environ["COHERE_API_KEY"] = cohere_api_key
st.title("AI-Powered Resume Screening")


zip_file = st.file_uploader("Upload candidate resumes (zip of PDFs)", type="zip")


job_description = st.text_area("Paste the Job Description", height=200)

if st.button("Analyze Candidates"):
    if not zip_file or not job_description.strip():
        st.error("Please upload resumes and enter a job description.")
    else:
        extract_dir = "pdfs"
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(zip_file, "r") as z:
            z.extractall(extract_dir)

        st.write(" Extracted Files:", os.listdir(extract_dir))

        def load_pdf_cvs(folder_path):
            cvs = []
            for filename in os.listdir(folder_path):
                if filename.endswith(".pdf"):
                    try:
                        file_path = os.path.join(folder_path, filename)
                        loader = PyPDFLoader(file_path)
                        pages = loader.load()
                        combined_content = "\n".join([page.page_content for page in pages])
                        metadata = {"source": filename, "page_count": len(pages)}
                        cvs.append(Document(page_content=combined_content, metadata=metadata))
                    except Exception as e:
                        st.warning(f"Error loading {filename}: {str(e)}")
            return cvs

        cvs = load_pdf_cvs(extract_dir)
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            length_function=len
        )
        split_docs = splitter.split_documents(cvs)

>>>>>>> a34be679f524b38afb65f28ef48725a7386ce51c
        
        embeddings = CohereEmbeddings(
            model="embed-english-v3.0",
            cohere_api_key=cohere_api_key
        )
<<<<<<< HEAD
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
=======
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        retriever = vectorstore.as_retriever()

        llm = Cohere(
            model="command",
            temperature=0.3,
            max_tokens=500
        )

        
        hr_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a senior technical recruiter analyzing candidates for a machine learning position.
Evaluate how well each candidate matches these job requirements:

JOB DESCRIPTION:
{question}

CANDIDATE PROFILES:
{context}

ANALYSIS INSTRUCTIONS:
1. For each candidate, evaluate:
   - Technical skills match (40% weight)
   - Relevant experience (30% weight)
   - Project relevance (20% weight)
   - Education (10% weight)
2. Score each candidate from 1-10 (10=perfect fit)
3. Provide 1-2 sentence justification per score
4. Rank only the top 3 candidates

OUTPUT FORMAT:
- Candidate: [Name]
  Score: [X]/10
  Match: [Brief explanation]
  Strengths: [Key strengths]

TOP 3 CANDIDATES:
1. [Name] (Score: [X])
2. [Name] (Score: [X])
3. [Name] (Score: [X])
"""
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            verbose=True,
            chain_type_kwargs={"prompt": hr_prompt}
        )

        with st.spinner("Analyzing candidates..."):
            result = qa_chain.invoke({"query": job_description})

        st.subheader("Top Candidates Analysis")
        st.write(result["result"])
>>>>>>> a34be679f524b38afb65f28ef48725a7386ce51c
