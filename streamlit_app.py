import os
import pandas as pd
import streamlit as st
from openai import OpenAI

# ==============================
# Helper: Load API Key
# ==============================
def get_api_key():
    key = None
    try:
        key = st.secrets.get("OPENAI_API_KEY", None)
    except:
        key = None

    if not key:
        # Allow manual input during testing
        key = st.sidebar.text_input("🔑 أدخلي OpenAI API Key", type="password")

    return key


# ==============================
# Level Classification
# ==============================
def classify_level(score, low_thr=50, high_thr=75):
    if score < low_thr:
        return "Low"
    elif score < high_thr:
        return "Medium"
    return "High"


# ==============================
# Convert Your Excel Format → Long Format
# ==============================
def transform_thesis_format(df):
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

    return df


# ==============================
# GPT Worksheet Generator
# ==============================
def generate_worksheet(
    client,
    student_name,
    student_grade,
    curriculum_grade,
    skill,
    level,
    num_questions=5,
):
    system_prompt = (
        "You are an educational content generator for primary school English "
        "within the Qatari National Curriculum. Adjust difficulty based on the "
        "TARGET curriculum grade, not the student's actual grade."
    )

    user_prompt = f"""
Student name: {student_name}
Actual school grade: {student_grade}
TARGET curriculum grade: {curriculum_grade}
Skill: {skill}
Performance level: {level}

Task:
1. Write a short reading passage (80–120 words) matching the TARGET grade difficulty.
2. Focus on the specified skill.
3. Create {num_questions} MCQs (A–D) appropriate for the target grade.
4. Provide the correct answer clearly.

Return clean text.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.6,
    )

    return response.choices[0].message.content


# ==============================
# STREAMLIT APP UI
# ==============================
def main():
    st.set_page_config(page_title="English Worksheets Generator", layout="wide")

    st.title("📚 English Worksheets Generator")
    st.write("Generate adaptive worksheets using Pandas + GPT API + dynamic curriculum mapping.")

    # ==========================
    # Sidebar – Settings
    # ==========================
    st.sidebar.header("Settings")

    # Grade of your class
    class_grade = st.sidebar.selectbox(
        "Grade you are teaching (Actual student grade)",
        [1, 2, 3, 4, 5, 6],
        index=4,  # Default = Grade 5
    )

    st.sidebar.subheader("Curriculum Mapping Based on Performance Level")

    grade_for_low = st.sidebar.selectbox(
        "Curriculum grade for LOW students",
        [1, 2, 3, 4, 5, 6],
        index=max(class_grade - 2, 1) - 1,
    )

    grade_for_medium = st.sidebar.selectbox(
        "Curriculum grade for MEDIUM students",
        [1, 2, 3, 4, 5, 6],
        index=class_grade - 1,
    )

    grade_for_high = st.sidebar.selectbox(
        "Curriculum grade for HIGH students",
        [1, 2, 3, 4, 5, 6],
        index=min(class_grade, 6) - 1,
    )

    def map_to_curriculum(level):
        if level == "Low":
            return grade_for_low
        elif level == "Medium":
            return grade_for_medium
        return grade_for_high

    # Number of questions
    num_questions = st.sidebar.slider("Number of questions", 3, 10, 5)

    # API Key
    api_key = get_api_key()
    if not api_key:
        st.warning(" OpenAI API Key")
        st.stop()

    client = OpenAI(api_key=api_key)

    # ==========================
    # Upload File
    # ==========================
    st.subheader("1️⃣ Upload student performance CSV")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is None:
        st.info("Upload CSV")
        return

    df_raw = pd.read_csv(uploaded)
    st.write("### Raw Data Preview")
    st.dataframe(df_raw.head())

    # Convert to long format
    df = transform_thesis_format(df_raw)

    # Add grade column (all students same grade)
    df["grade"] = class_grade

    # Classify performance
    df["level"] = df["score"].apply(classify_level)

    # Curriculum mapping
    df["target_curriculum_grade"] = df["level"].apply(map_to_curriculum)

    st.write("### Processed Data")
    st.dataframe(df.head())

    # ==========================
    # Skill selection
    # ==========================
    skills = sorted(df["skill"].unique())
    selected_skill = st.selectbox("Choose a skill", skills)

    levels = ["Low", "Medium", "High"]
    selected_level = st.selectbox("Performance Level", levels)

    target_df = df[(df["skill"] == selected_skill) & (df["level"] == selected_level)]

    st.write(f"Students matching selection: {len(target_df)}")

    # ==========================
    # Generate Worksheets
    # ==========================
    if st.button("Generate Worksheets"):
        if target_df.empty:
            st.error("No students found for this skill + level.")
            return

        st.success("Generating worksheets... please wait ⏳")

        for _, row in target_df.iterrows():
            text = generate_worksheet(
                client,
                student_name=row["student_name"],
                student_grade=row["grade"],
                curriculum_grade=row["target_curriculum_grade"],
                skill=row["skill"],
                level=row["level"],
                num_questions=num_questions,
            )

            st.write(f"### Worksheet for {row['student_name']}")
            st.text(text)


if __name__ == "__main__":
    main()
