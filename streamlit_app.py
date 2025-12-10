import os
import pandas as pd
import streamlit as st
from openai import OpenAI

# ==============================
# Helpers
# ==============================

def get_api_key() -> str:
    """Get OpenAI API key from environment or Streamlit secrets (no UI input)."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        try:
            key = st.secrets.get("OPENAI_API_KEY", None)
        except Exception:
            key = None
    return key


def classify_level(score: float, low_thr: float = 50, high_thr: float = 75) -> str:
    """Low / Medium / High based on thresholds."""
    if score < low_thr:
        return "Low"
    elif score < high_thr:
        return "Medium"
    return "High"


def transform_thesis_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supports thesis format:
    StudentNumber, StudentName, LanguageFunction, ReadingComprehension, Grammar, Writing, Total
    and converts it into:
    student_id, student_name, skill, score
    """
    thesis_cols = {
        "StudentNumber",
        "StudentName",
        "LanguageFunction",
        "ReadingComprehension",
        "Grammar",
        "Writing",
    }

    if thesis_cols.issubset(df.columns):
        df_long = df.melt(
            id_vars=["StudentNumber", "StudentName"],
            value_vars=[
                "LanguageFunction",
                "ReadingComprehension",
                "Grammar",
                "Writing",
            ],
            var_name="skill",
            value_name="score",
        )

        df_long = df_long.rename(
            columns={
                "StudentNumber": "student_id",
                "StudentName": "student_name",
            }
        )
        return df_long

    # إذا كان جاهز أصلاً بالصيغة المطلوبة نرجعه كما هو
    return df


def generate_worksheet(
    client: OpenAI,
    student_name: str,
    student_grade: int,
    curriculum_grade: int,
    skill: str,
    level: str,
    num_questions: int = 5,
) -> str:
    """Call GPT to generate
