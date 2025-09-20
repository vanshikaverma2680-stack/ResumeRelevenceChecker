import streamlit as st
import os
import fitz  # PyMuPDF
import docx2txt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- Utility functions ----
def extract_text_from_file(uploaded_file):
    text = ""
    if uploaded_file.name.endswith(".pdf"):
        pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in pdf:
            text += page.get_text()
    elif uploaded_file.name.endswith(".docx"):
        with open("temp.docx", "wb") as f:
            f.write(uploaded_file.read())
        text = docx2txt.process("temp.docx")
        os.remove("temp.docx")
    else:
        st.warning(f"Unsupported file type: {uploaded_file.name}")
    return text.strip()

def calculate_similarity(jd_text, resumes):
    documents = [jd_text] + resumes
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    return similarity_scores[0]

# ---- Streamlit UI ----
st.title("📄 Resume–JD Matching App")
st.write("Upload multiple **Resumes (PDF/DOCX)** and **Job Descriptions (PDF/DOCX)** to get matching scores.")

uploaded_resumes = st.file_uploader("Upload Resumes", type=["pdf", "docx"], accept_multiple_files=True)
uploaded_jds = st.file_uploader("Upload Job Descriptions", type=["pdf", "docx"], accept_multiple_files=True)

if uploaded_resumes and uploaded_jds:
    st.success(f"✅ {len(uploaded_resumes)} resumes and {len(uploaded_jds)} JDs uploaded.")
    
    resumes_texts = [extract_text_from_file(r) for r in uploaded_resumes]
    jds_texts = [extract_text_from_file(j) for j in uploaded_jds]

    for jd_index, jd_text in enumerate(jds_texts):
        st.subheader(f"📌 Evaluating JD {jd_index+1} ({uploaded_jds[jd_index].name})")
        
        scores = calculate_similarity(jd_text, resumes_texts)
        
        for i, score in enumerate(scores):
            relevance = round(score*100, 2)
            if relevance >= 70:
                verdict = "High ✅"
            elif relevance >= 40:
                verdict = "Medium ⚠️"
            else:
                verdict = "Low ❌"
            
            st.write(f"**Resume {i+1} ({uploaded_resumes[i].name})** → Relevance: {relevance}% | Verdict: {verdict}")
