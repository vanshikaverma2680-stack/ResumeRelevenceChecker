# app.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Resume Relevance Checker API")

# Input data model
class InputData(BaseModel):
    job_descriptions: List[str]
    resumes: List[str]

# API endpoint
@app.post("/evaluate/")
def evaluate_resumes(data: InputData):
    results = []

    for jd_index, jd_text in enumerate(data.job_descriptions):
        documents = [jd_text] + data.resumes
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(documents)
        similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

        jd_keywords = set(word.lower().replace('-', ' ') for word in jd_text.replace(',', '').split())

        jd_results = []
        for i, score in enumerate(similarity_scores[0]):
            relevance = round(score * 100, 2)
            verdict = "High ✅" if relevance >= 70 else "Medium ⚠️" if relevance >= 40 else "Low ❌"
            resume_words = set(word.lower() for word in data.resumes[i].replace(',', '').split())
            missing_keywords = jd_keywords - resume_words

            jd_results.append({
                "resume_index": i + 1,
                "relevance_score": relevance,
                "verdict": verdict,
                "top_missing_keywords": list(missing_keywords)[:5]
            })

        results.append({
            "job_description_index": jd_index + 1,
            "results": jd_results
        })

    return {"evaluation": results}
