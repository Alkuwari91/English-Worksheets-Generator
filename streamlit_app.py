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
    """Call GPT to generate one worksheet text."""
    system_prompt = (
        "You are an educational content generator for primary school English "
        "within the Qatari National Curriculum. Adjust difficulty and language "
        "based on the TARGET curriculum grade, not the student's current grade. "
        "Keep tasks clear, age-appropriate, and focused on the requested skill."
    )

    user_prompt = f"""
Student name: {student_name}
Actual school grade: {student_grade}
TARGET curriculum grade for this worksheet: {curriculum_grade}
Skill to focus on: {skill}
Performance level in this skill: {level} (Low/Medium/High)

Task:
1. Write a short reading passage (80–120 words) matching the TARGET curriculum grade.
2. Focus on the specified skill.
3. Create {num_questions} multiple-choice questions (A–D) based on the passage.
4. Indicate the correct option clearly for each question.

Return the result with headings:
PASSAGE
QUESTIONS
ANSWER KEY
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
# Simple styling (MoE-inspired)
# ==============================

CUSTOM_CSS = """
<style>
header {visibility: hidden;}
footer {visibility: hidden;}

body, .stApp {
    background-color: #f5f6fa;
    font-family: "Helvetica Neue", Arial, sans-serif;
}

/* Generic cards */
.card {
    background-color: #ffffff;
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    box-shadow: 0 10px 26px rgba(0, 0, 0, 0.06);
    margin-bottom: 1.2rem;
}

/* Hero */
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    color: #8A1538; /* Qatar maroon */
    margin-bottom: 0.3rem;
}
.hero-subtitle {
    font-size: 0.98rem;
    color: #555;
}

/* Step titles */
.step-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #8A1538;
    margin-bottom: 0.2rem;
}
.step-help {
    font-size: 0.9rem;
    color: #666;
    margin-bottom: 0.6rem;
}

/* Fake auth card */
.auth-card-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #333;
    margin-bottom: 0.4rem;
}
.auth-tabs {
    display: flex;
    gap: 0.4rem;
    margin-bottom: 0.6rem;
}
.auth-tab {
    flex: 1;
    padding: 0.2rem 0.4rem;
    text-align: center;
    border-radius: 999px;
    font-size: 0.85rem;
    border: 1px solid #ddd;
    cursor: default;
}
.auth-tab.active {
    background-color: #8A1538;
    color: #fff;
    border-color: #8A1538;
}
.auth-link {
    font-size: 0.8rem;
    color: #8A1538;
    text-decoration: none;
}
.auth-link:hover {
    text-decoration: underline;
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
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ---- API key (no UI) ----
    api_key = get_api_key()
    if not api_key:
        st.error(
            "No OpenAI API key found. Please add it in Settings → Secrets as OPENAI_API_KEY."
        )
        return
    client = OpenAI(api_key=api_key)

    # ---- Sidebar settings (minimal) ----
    st.sidebar.markdown("### ⚙️ Teacher settings")

    class_grade = st.sidebar.selectbox(
        "Grade you are teaching",
        [1, 2, 3, 4, 5, 6],
        index=4,  # default grade 5
    )

    st.sidebar.markdown("#### Curriculum by level")

    grades_list = [1, 2, 3, 4, 5, 6]

    grade_for_low = st.sidebar.selectbox(
        "Target for LOW",
        grades_list,
        index=max(class_grade - 2, 1) - 1,
    )

    grade_for_medium = st.sidebar.selectbox(
        "Target for MEDIUM",
        grades_list,
        index=class_grade - 1,
    )

    grade_for_high = st.sidebar.selectbox(
        "Target for HIGH",
        grades_list,
        index=min(class_grade, 6) - 1,
    )

    low_thr = st.sidebar.slider("Low threshold", 0, 100, 50)
    high_thr = st.sidebar.slider("High threshold", 0, 100, 75)
    num_questions = st.sidebar.slider("Questions per worksheet", 3, 10, 5)

    def map_to_curriculum(level: str) -> int:
        if level == "Low":
            return grade_for_low
        elif level == "Medium":
            return grade_for_medium
        return grade_for_high

    # ==========================
    # HERO + Fake Sign-in shape
    # ==========================
    hero_col1, hero_col2 = st.columns([2, 1])

    with hero_col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="hero-title">English Worksheets Generator</div>
            <div class="hero-subtitle">
                A prototype for adaptive remedial worksheets based on student performance,
                mapped to the Qatari English curriculum.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with hero_col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="auth-card-title">Teacher portal</div>
            <div class="auth-tabs">
                <div class="auth-tab active">Sign in</div>
                <div class="auth-tab">Sign up</div>
                <div class="auth-tab">Forgot password</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # عناصر شكلية فقط – لا تستخدم في المنطق
        st.text_input("Email address", placeholder="name@education.qa")
        st.text_input("Password", type="password", placeholder="••••••••")
        st.caption("This is a prototype UI. No real account is created or stored.")
        st.markdown(
            '<a class="auth-link">Need a new account? Sign up</a><br>'
            '<a class="auth-link">Forgot your password?</a>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ==========================
    # STEP 1 – Upload CSV
    # ==========================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="step-title">Step 1 – Upload student performance file</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="step-help">'
        'Upload the CSV file with student scores. The app reshapes it and prepares it for grouping.'
        '</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Upload CSV (thesis format)", type=["csv"])

    if uploaded is None:
        st.info("No file uploaded yet. Please upload your Students.csv file to continue.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    df_raw = pd.read_csv(uploaded)
    st.markdown("**Raw data preview (first rows):**")
    st.dataframe(df_raw.head(), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ==========================
    # STEP 2 – Process & inspect
    # ==========================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="step-title">Step 2 – Process data & review groups</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="step-help">'
        "The app creates one row per student and skill, classifies performance using the thresholds, "
        "and maps each level to a curriculum grade."
        "</div>",
        unsafe_allow_html=True,
    )

    df = transform_thesis_format(df_raw)
    df["grade"] = class_grade
    df["level"] = df["score"].apply(lambda s: classify_level(s, low_thr, high_thr))
    df["target_curriculum_grade"] = df["level"].apply(map_to_curriculum)

    st.markdown("**Processed data (first rows):**")
    st.dataframe(df.head(), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ==========================
    # STEP 3 – Generate worksheets
    # ==========================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="step-title">Step 3 – Generate worksheets</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="step-help">'
        "Choose a skill and performance level. One worksheet will be generated for each student "
        "in that group, using the mapped curriculum grade."
        "</div>",
        unsafe_allow_html=True,
    )

    skills = sorted(df["skill"].unique())
    selected_skill = st.selectbox("Skill", skills)

    levels = ["Low", "Medium", "High"]
    selected_level = st.selectbox("Performance level", levels)

    target_df = df[(df["skill"] == selected_skill) & (df["level"] == selected_level)]
    st.markdown(f"**Students in this group:** {len(target_df)}")

    if st.button("Generate worksheets"):
        if target_df.empty:
            st.error("No students found for this combination of skill and level.")
        else:
            st.success("Generating worksheets… please wait ⏳")
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
                st.markdown(f"#### Worksheet for {row['student_name']}")
                st.text(ws_text)

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
