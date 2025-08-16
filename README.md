# Resume Screening System

This project is a complete **AI-powered resume screening system** with a **FastAPI backend** and a **Streamlit frontend**.

It allows recruiters to upload candidate resumes (PDFs in a ZIP file), provide a job description, and receive an automated analysis of the top candidates using **LangChain** + **Cohere** embeddings/LLMs.

---

## 🚀 Features
- **Frontend (Streamlit):**
  - Upload resumes via web UI
  - Paste job description
  - View ranked candidates with analysis

- **Backend (FastAPI):**
  - Accepts resumes and job description via API
  - Processes resumes into vector embeddings
  - Uses Cohere LLM to evaluate and rank candidates
  - Returns JSON with top candidates and analysis

---

## 📦 Requirements
- Python 3.9+
- [Cohere API key](https://dashboard.cohere.ai/)

Install dependencies:
```bash
pip install fastapi uvicorn streamlit python-dotenv langchain langchain-community langchain-cohere
```

---

## ⚙️ Setup
1. Clone this repo.
2. Create a `.env` file with your Cohere API key:
   ```env
   cohere=YOUR_COHERE_API_KEY
   ```

### Run Backend (FastAPI)
```bash
uvicorn main:app --reload
```
- API runs at: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Swagger UI docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Run Frontend (Streamlit)
```bash
streamlit run streamlit_app.py
```
- App runs at: [http://localhost:8501](http://localhost:8501)

---

## 📡 API Usage (FastAPI)

### Endpoint: `/analyze`
**Method:** `POST`

**Parameters:**
- `job_description` (form field, string)
- `file` (form file, ZIP containing PDF resumes)

**Example cURL:**
```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -F "job_description=We are hiring a Machine Learning Engineer..." \
  -F "file=@cvs.zip"
```

**Response JSON:**
```json
{
  "analysis": "...Top 3 candidate ranking and scores..."
}
```

---

## 🏆 Example Output
```
- Candidate: John Doe
  Score: 9/10
  Match: Strong ML + LangChain experience
  Strengths: Deep learning, FAISS, LLMs

TOP 3 CANDIDATES:
1. John Doe (Score: 9)
2. Jane Smith (Score: 8)
3. Alice Johnson (Score: 7)
```

---

## 📌 Notes
- The backend (`main.py`) handles resume analysis.
- The frontend (`streamlit_app.py`) provides a recruiter-friendly interface.
- Extend the `PromptTemplate` to customize scoring criteria.
- Ensure resumes are in **PDF format** inside a ZIP file before uploading.
