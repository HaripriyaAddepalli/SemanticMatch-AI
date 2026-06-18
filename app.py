import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import os

# Page config
st.set_page_config(page_title="SemanticMatch AI", layout="wide")
st.title("🚀 SemanticMatch AI")
st.subheader("Intelligent Candidate Ranking System")
st.markdown("**Team:** SemanticMatch AI | **Leader:** Addepali Haripriya")

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Sample Data
candidates = [
    {
        "name": "Rahul Sharma",
        "title": "Senior Python Developer",
        "experience": "6 years",
        "skills": "Python, Django, FastAPI, PostgreSQL, AWS, Docker, Machine Learning",
        "summary": "Experienced backend developer with strong expertise in building scalable web applications and ML pipelines."
    },
    {
        "name": "Priya Patel",
        "title": "Software Engineer",
        "experience": "4 years",
        "skills": "Python, Flask, React, MySQL, AWS, Kubernetes",
        "summary": "Full-stack developer passionate about clean code and DevOps practices."
    },
    {
        "name": "Amit Kumar",
        "title": "Data Scientist",
        "experience": "5 years",
        "skills": "Python, TensorFlow, Pandas, Scikit-learn, SQL, AWS",
        "summary": "Data scientist with experience in building recommendation systems and predictive models."
    },
    {
        "name": "Sneha Reddy",
        "title": "Backend Developer",
        "experience": "3 years",
        "skills": "Python, Django, REST APIs, MongoDB, Azure",
        "summary": "Backend engineer focused on API development and database optimization."
    }
]

# Default JD
default_jd = """We are looking for a Senior Python Developer with 5+ years of experience.
Strong expertise in Django or FastAPI, PostgreSQL, and cloud platforms (AWS preferred).
Experience with Docker, CI/CD, and building scalable web applications is required.
Knowledge of Machine Learning is a plus."""

# Sidebar
with st.sidebar:
    st.header("Job Description")
    jd = st.text_area("Enter Job Description", default_jd, height=300)
    
    if st.button("🔄 Refresh Rankings"):
        st.session_state.rerun = True

# Main Area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Ranked Candidates")
    
    # Compute embeddings
    jd_embedding = model.encode([jd])[0]
    
    candidate_texts = [f"{c['name']} {c['title']} {c['experience']} {c['skills']} {c['summary']}" for c in candidates]
    candidate_embeddings = model.encode(candidate_texts)
    
    # Semantic Similarity
    similarities = cosine_similarity([jd_embedding], candidate_embeddings)[0]
    
    # Create DataFrame
    df = pd.DataFrame({
        "Rank": range(1, len(candidates)+1),
        "Candidate": [c["name"] for c in candidates],
        "Title": [c["title"] for c in candidates],
        "Experience": [c["experience"] for c in candidates],
        "Semantic Score": [round(s*100, 1) for s in similarities],
        "Skills": [c["skills"] for c in candidates]
    })
    
    # Sort by score
    df = df.sort_values("Semantic Score", ascending=False).reset_index(drop=True)
    df["Rank"] = range(1, len(df)+1)
    
    st.dataframe(df, use_container_width=True, hide_index=True)

with col2:
    st.subheader("Top Candidate Details")
    if not df.empty:
        top_candidate = df.iloc[0]
        st.metric("Top Match", top_candidate["Candidate"], f"{top_candidate['Semantic Score']}%")
        
        # Find original candidate
        orig = next(c for c in candidates if c["name"] == top_candidate["Candidate"])
        st.write("**Summary:**", orig["summary"])
        st.write("**Key Skills:**", orig["skills"])
        
        st.markdown("**Why this match?**")
        st.write("Strong semantic alignment in Python expertise, frameworks, and cloud experience.")

# Footer
st.markdown("---")
st.caption("SemanticMatch AI • Powered by Sentence Transformers + FAISS-ready architecture")