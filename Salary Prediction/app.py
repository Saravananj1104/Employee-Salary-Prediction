import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import docx2txt
import fitz 
import re
import textstat
import language_tool_python
from wordcloud import WordCloud
from io import BytesIO
import base64

model = joblib.load(r"C:\Users\NEO\Desktop\salary\random_forest_model (2).pkl")
scaler = joblib.load(r"C:\Users\NEO\Desktop\salary\scaler (1).pkl")

education_dict = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
job_title_dict = {
    "Data Scientist": 0,
    "Software Engineer": 1,
    "Product Manager": 2,
    "HR": 3,
    "Accountant": 4
}
reverse_job_dict = {v: k for k, v in job_title_dict.items()}

domain_skills = {
    "Data Science": ["pandas", "numpy", "machine learning", "scikit-learn", "tensorflow"],
    "Web Development": ["html", "css", "javascript", "react", "node.js"],
    "DevOps": ["docker", "kubernetes", "jenkins", "aws", "azure"],
    "Finance": ["excel", "accounting", "budgeting"],
    "HR": ["recruitment", "employee engagement"]
}

def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def match_keywords(text):
    found = set()
    for skills in domain_skills.values():
        for skill in skills:
            if re.search(rf"\\b{re.escape(skill)}\\b", text.lower()):
                found.add(skill)
    return found


def detect_domain(text):
    text = text.lower()
    scores = {domain: sum(1 for skill in skills if skill in text)
              for domain, skills in domain_skills.items()}
    return max(scores, key=scores.get) if scores else "General"

def extract_contacts(text):
    email = re.findall(r"[\w\.-]+@[\w\.-]+", text)
    phone = re.findall(r"\+?\d[\d\s\-()]{8,}\d", text)
    linkedin = re.findall(r"linkedin\.com\/in\/\S+", text)
    github = re.findall(r"github\.com\/\S+", text)
    return email, phone, linkedin, github

def grammar_check(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    return len(matches)

def generate_wordcloud(text):
    wc = WordCloud(width=600, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)


st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🌍 Global Employee Salary Predictor + Resume Analyzer</h1>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: justify; font-size: 16px;">
Upload your resume or manually enter your information to get a predicted salary. Our analyzer also checks your resume for keywords, readability, grammar, completeness, and gives domain-specific suggestions.
</div><br>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📈 Salary Prediction", "📄 Resume Upload & Analyzer"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("👤 Gender", ["Male", "Female"])
        age = st.slider("🎂 Age", 18, 65, 30)
        experience = st.slider("💼 Experience (Years)", 0, 40, 5)
    with col2:
        education = st.selectbox("🎓 Education Level", list(education_dict.keys()))
        job_title = st.selectbox("🧑‍💻 Job Title", list(job_title_dict.keys()))

    gender_encoded = 1 if gender == "Male" else 0

    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender_encoded,
        "Education Level": education_dict[education],
        "Job Title": job_title_dict[job_title],
        "Years of Experience": experience
    }])

    scaled_df = scaler.transform(input_df)

    if st.button("🔮 Predict Salary"):
        prediction = model.predict(scaled_df)[0]
        st.success(f"💰 Predicted Salary: ₹{int(prediction):,}")
        st.balloons()

with tab2:
    resume_file = st.file_uploader("Upload Resume (.pdf or .docx)", type=["pdf", "docx"])

    if resume_file:
        if resume_file.name.endswith(".pdf"):
            text = extract_text_from_pdf(resume_file)
        else:
            text = docx2txt.process(resume_file)

        st.text_area("📄 Extracted Resume Text", text, height=300)

        predicted_experience = int(re.findall(r"(\d+)\s+years", text.lower())[0]) if re.findall(r"(\d+)\s+years", text.lower()) else 2
        predicted_job = detect_domain(text)
        matched_skills = match_keywords(text)
        readability_score = textstat.flesch_reading_ease(text)
        grammar_issues = grammar_check(text)
        email, phone, linkedin, github = extract_contacts(text)

        input_resume_df = pd.DataFrame([{
            "Age": 30,
            "Gender": 1,
            "Education Level": 2,
            "Job Title": job_title_dict.get(predicted_job, 1),
            "Years of Experience": predicted_experience
        }])
        input_resume_df = scaler.transform(input_resume_df)
        predicted_salary = model.predict(input_resume_df)[0]

        st.success(f"💼 Predicted Domain: {predicted_job}")
        st.success(f"💰 Estimated Salary: ₹{int(predicted_salary):,}")

        st.markdown("### 🧠 Resume Insights")
        st.markdown(f"- ✅ **Matched Skills**: {', '.join(matched_skills) if matched_skills else 'None Found'}")
        st.markdown(f"- 📖 **Readability Score**: {readability_score:.2f} (Higher is easier)")
        st.markdown(f"- 📝 **Grammar Issues Found**: {grammar_issues}")
        st.markdown(f"- 📇 **Contact Info Found**: Email: {bool(email)}, Phone: {bool(phone)}, LinkedIn: {bool(linkedin)}, GitHub: {bool(github)}")

        st.markdown("### ☁️ Word Cloud")
        generate_wordcloud(text)

        st.markdown("### 🔍 Completeness Check")
        for section in ["education", "experience", "skills", "projects"]:
            found = section in text.lower()
            st.markdown(f"- {'✅' if found else '❌'} {section.capitalize()} section")

        st.markdown("### 💡 Suggestions")
        st.markdown("""
        - Use more **action verbs** like led, managed, developed.
        - Add quantifiable results (e.g., "improved sales by 20%").
        - Highlight leadership, teamwork, and adaptability.
        - Ensure your resume is not too lengthy.
        - Include **LinkedIn** or **GitHub** links for credibility.
        """)
