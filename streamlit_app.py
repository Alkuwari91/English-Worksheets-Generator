import os
import pandas as pd
import streamlit as st
from openai import OpenAI

# ==============================
# Helper functions
# ==============================

def get_api_key() -> str:
    """Get OpenAI API key from environment or Streamlit secrets."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        try:
            key = st.secrets.get("OPENAI_API_KEY", None)
        except Exception:
            key = None
    return key


def classify_level(score: float, low_thr: float = 50, high_thr: float = 75) -> str:
    """Low / Medium / High classification."""
    if score < low_thr:
        return "Low"
    elif score < high_thr:
        return "Medium"
    return "High"


def transform_thesis_format(df: pd.DataFrame) -> pd.DataFrame:
    """Convert your dataset into long format."""
    thesis_cols = {
        "StudentNumber", "StudentName",
        "LanguageFunction", "ReadingComprehension",
        "Grammar", "Writing"
    }

    if thesis_cols.issubset(df.columns):
        df_long = df.melt(
            id_vars=["StudentNumber", "StudentName"],
            value_vars=[
                "LanguageFunction", "ReadingComprehension",
                "Grammar", "Writing"
            ],
            var_name="skill",
            value_name="score",
        )

        df_long = df_long.rename(columns={
            "StudentNumber": "student_id",
            "StudentName": "student_name",
        })
        return df_long

    return df


def generate_worksheet(
    client: OpenAI,
    student_name: str,
    student_grade: int,
    curriculum_grade: int,
    skill: str,
    level: str,
    num_questions: int = 5
):
    """Generate worksheet using GPT based on mapped curriculum grade."""
    
    system_prompt = (
        "You are an educational content generator for primary school English "
        "within the Qatari National Curriculum. Adjust difficulty and language "
        "based on the TARGET curriculum grade. Keep content clear and suitable for students."
    )

    user_prompt = f"""
Student name: {student_name}
Actual school grade: {student_grade}
Target curriculum grade: {curriculum_grade}
Skill: {skill}
Performance level: {level}

Task:
1. Write a short reading passage (80–120 words) appropriate for the target grade.
2. Focus on the given skill.
3. Create {num_questions} MCQs (A–D).
4. Provide an answer key clearly.

Format required:
PASSAGE:
...

QUESTIONS:
1) ...
A) ...
B) ...
C) ...
D) ...

ANSWER KEY:
1) ...
"""

    response = client.chat.comp
