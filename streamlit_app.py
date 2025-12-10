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
# Custom CSS (header style)
# ==============================

CUSTOM_CSS = """
<style>

header, footer {visibility: hidden;}

body, .stApp {
    background-color: #f6f7fb;
    font-family: "Cairo", sans-serif;
}

/* Header */
.app-header {
    width: 100%;
    padding: 1.2rem 2rem;
    background: linear-gradient(135deg, #8A1538, #600d26);
    border-radius: 0 0 18px 18px;
    color: white;
    margin-bottom: 1.5rem;
}

.header-title {
    font-size: 1.9rem;
    font-weight: 800;
    margin-bottom: -4px;
}

.header-sub {
    font-size: 0.95rem;
    opacity: 0.9;
}

/* Auth box (UI only) */
.auth-box {
    background: white;
    padding: 1rem 1.3rem;
    border-radius: 12px;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
    margin-top: 1rem;
}

.auth-tabs {
    display: flex;
    gap: 0.3rem;
    margin-bottom: 0.8rem;
}

.auth-tab {
    flex: 1;
    text-align: center;
    background: #eee;
    padding: 0.35rem;
    border-radius: 8px;
    font-size: 0.85rem;
    cursor: default;
    color: #444;
}

.auth-tab.active {
    background: #8A1538;
    color: white;
}

/* Cards */
.card {
    background: white;
    padding: 1.4rem 1.6rem;
    border-radius: 16px;
    margin-bottom: 1.2rem;
    box-shadow: 0 6px 22px rgba(0, 0, 0, 0.06);
}

.step-title {
    color: #8A1538;
    font-size: 1.2rem;
    font-weight: bold;
}

.step-help {
    color: #555;
    font-size: 0.9rem;
}

</style>
"""


# ==============================
# Streamlit App
# ==============================

def main():

    st.set_page_config(
        page_title="English Worksheets Generator",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ----------------- HEADER -----------------
    st.markdown(
        """
        <div class="app-header">
            <div class="header-title">English Worksheets Generator</div>
            <div class="header-sub">
                Adaptive worksheets based on student performance and mapped curriculum levels
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----------------- API Key -----------------
    api_key = get_api_key()
    if not api_key:
        st.error("OPENAI_API_KEY is missing. Add it in Settings → Secrets.")
        return

    client = OpenAI(api_key=api_key)

    # ----------------- Auth Box (UI only) -----------------
    st.markdown('<div class="auth-box">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="auth-tabs">
            <div class="auth-tab active">Sign in</div>
            <div class="auth-tab">Sign up</div>
            <div class="auth-tab">Forgot password</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.text_input("Email address", placeholder="name@education.qa")
    st.text_input("Password", type="password", placeholder="••••••••")
    st.caption("UI only — no real login is performed.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ----------------- Sidebar settings -----------------
    st.sidebar.title("Settings")

    class_grade = st.sidebar.selectbox(
        "Student grade",
        [1, 2, 3, 4, 5, 6],
        index=4,
    )

    grade_for_low = st.sidebar.selectbox(
        "Curriculum for LOW",
        [1, 2, 3, 4, 5, 6],
        index=max(class_grade - 2, 1) - 1,
    )

    grade_for_medium = st.sidebar.selectbox(
        "Curriculum for MEDIUM",
        [1, 2, 3, 4, 5, 6],
        index=class_grade - 1,
    )

    grade_for_high = st.sidebar.selectbox(
        "Curriculum for HIGH",
        [1, 2, 3, 4, 5, 6],
        index=min(class_grade, 6) - 1,
    )

    low_thr = st.sidebar.slider("Low threshold", 0, 100, 50)
    high_thr = st.sidebar.slider("High threshold", 0, 100, 75)
    num_questions = st.sidebar.slider("Questions per worksheet", 3, 10, 5)

    def map_to_curriculum(level: str):
        if level == "Low":
            return grade_for_low
        elif level == "Medium":
            return grade_for_medium
        return grade_for_high

    # ----------------- STEP 1 -----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="step-title">Step 1 — Upload student file</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="step-help">Upload the CSV file containing your students’ skill scores.</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Upload Students.csv", type=["csv"])

    if uploaded is None:
        st.info("Please upload a CSV to continue.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    df_raw = pd.read_csv(uploaded)
    st.write("Raw data preview:")
    st.dataframe(df_raw.head(), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ----------------- STEP 2 -----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="step-title">Step 2 — Process data</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="step-help">The system reshapes the data, classifies students, and maps levels.</div>',
        unsafe_allow_html=True,
    )

    df = transform_thesis_format(df_raw)
    df["grade"] = class_grade
    df["level"] = df["score"].apply(lambda x: classify_level(x, low_thr, high_thr))
    df["target_curriculum_grade"] = df["level"].apply(map_to_curriculum)

    st.write("Processed data:")
    st.dataframe(df.head(), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ----------------- STEP 3 -----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="step-title">Step 3 — Generate worksheets</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="step-help">Choose a skill and level, then generate one worksheet per student in this group.</div>',
        unsafe_allow_html=True,
    )

    skills = sorted(df["skill"].unique())
    selected_skill = st.selectbox("Choose skill", skills)

    levels = ["Low", "Medium", "High"]
    selected_level = st.selectbox("Choose performance level", levels)

    target_df = df[(df["skill"] == selected_skill) & (df["level"] == selected_level)]

    st.markdown(f"Students in this group: **{len(target_df)}**")

    if st.button("Generate worksheets"):
        if target_df.empty:
            st.error("No students match this skill + level.")
        else:
            with st.spinner("Generating worksheets…"):
                try:
                    for _, row in target_df.iterrows():
                        ws_text = generate_worksheet(
                            client=client,
                            student_name=row["student_name"],
                            student_grade=row["grade"],
                            curriculum_grade=row["target_curriculum_grade"],
                            skill=row["skill"],
                            level=row["level"],
                            num_questions=num_questions,
                        )

                        st.markdown(f"### Worksheet for {row['student_name']}")
                        st.text(ws_text)

                    st.success("All worksheets generated successfully ✅")

                except Exception as e:
                    st.error(f"Error while calling OpenAI API: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
