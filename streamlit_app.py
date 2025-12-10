import os
import pandas as pd
import streamlit as st
from openai import OpenAI

# ==============================
# Configuration
# ==============================
ALLOWED_DOMAIN = "education.qa"  # فقط إيميلات الوزارة


# ==============================
# Helpers: Auth & API Key
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


def check_auth() -> bool:
    """Simple sign-in: email must end with @education.qa + correct password."""
    if "authenticated" in st.session_state and st.session_state["authenticated"]:
        return True

    st.markdown(
        """
        <div class="login-card">
            <h2>🔐 English Worksheets Generator</h2>
            <p>Please sign in using your Ministry of Education email account.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("login_form"):
        email = st.text_input("Ministry email ( @education.qa )")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Sign in")

    if submit:
        if not email or not password:
            st.error("Please enter both email and password.")
            return False

        if not email.lower().endswith("@" + ALLOWED_DOMAIN):
            st.error(f"Only {ALLOWED_DOMAIN} accounts are allowed.")
            return False

        app_password = None
        try:
            app_password = st.secrets.get("APP_PASSWORD", None)
        except Exception:
            app_password = None

        # في البروتوتايب: لو ما تم ضبط APP_PASSWORD، نستخدم قيمة افتراضية
        if app_password is None:
            app_password = "education123"

        if password != app_password:
            st.error("Incorrect password.")
            return False

        # نجح تسجيل الدخول
        st.session_state["authenticated"] = True
        st.session_state["user_email"] = email
        st.success("Signed in successfully.")
        st.experimental_rerun()

    return False


# ==============================
# Level Classification
# ==============================
def classify_level(score: float, low_thr: float = 50, high_thr: float = 75) -> str:
    if score < low_thr:
        return "Low"
    elif score < high_thr:
        return "Medium"
    return "High"


# ==============================
# Convert Your Excel Format → Long Format
# ==============================
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

    # لو الملف أصلاً جاهز بالصيغة المطلوبة نرجعه كما هو
    return df


# ==============================
# GPT Worksheet Generator
# ==============================
def generate_worksheet(
    client: OpenAI,
    student_name: str,
    student_grade: int,
    curriculum_grade: int,
    skill: str,
    level: str,
    num_questions: int = 5,
) -> str:
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

Return the result as clean text with headings:
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
# Custom CSS for clean UI (MoE style)
# ==============================
CUSTOM_CSS = """
<style>
/* Hide Streamlit default header & footer */
header {visibility: hidden;}
footer {visibility: hidden;}

body, .stApp {
    background-color: #f7f8fb;
    font-family: "Helvetica Neue", Arial, sans-serif;
}

/* Main container */
.main-card {
    background-color: #ffffff;
    border-radius: 18px;
    padding: 1.8rem 2rem;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.06);
    margin-bottom: 1.5rem;
}

/* Step titles */
.step-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #8A1538;  /* Qatar maroon */
    margin-bottom: 0.3rem;
}

/* Small instructional text */
.step-help {
    font-size: 0.9rem;
    color: #555;
    margin-bottom: 0.6rem;
}

/* Login card */
.login-card {
    max-width: 480px;
    margin: 4rem auto 1rem auto;
    background: #ffffff;
    padding: 2rem 2.3rem;
    border-radius: 18px;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.10);
    text-align: center;
}
.login-card h2 {
    color: #8A1538;
    margin-bottom: 0.6rem;
}
.login-card p {
    color: #555;
}

/* Main title */
.app-title {
    font-size: 2rem;
    font-weight: 800;
    color: #8A1538;
    margin-bottom: 0.25rem;
}
.app-subtitle {
    font-size: 0.98rem;
    color: #666;
    margin-bottom: 1rem;
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

    # ---------- AUTH ----------
    if not check_auth():
        return

    # ---------- API KEY ----------
    api_key = get_api_key()
    if not api_key:
        st.error(
            "No OpenAI API key found. Please add it in Settings → Secrets as OPENAI_API_KEY."
        )
        return

    client = OpenAI(api_key=api_key)

    # ---------- SIDEBAR: SETTINGS ----------
    st.sidebar.markdown("### ⚙️ Teacher Settings")

    class_grade = st.sidebar.selectbox(
        "Grade you are teaching (actual student grade)",
        [1, 2, 3, 4, 5, 6],
        index=4,  # default grade 5
    )

    st.sidebar.markdown("#### Curriculum mapping by level")

    grades_list = [1, 2, 3, 4, 5, 6]

    grade_for_low = st.sidebar.selectbox(
        "Target curriculum for LOW",
        grades_list,
        index=max(class_grade - 2, 1) - 1,
    )

    grade_for_medium = st.sidebar.selectbox(
        "Target curriculum for MEDIUM",
        grades_list,
        index=class_grade - 1,
    )

    grade_for_high = st.sidebar.selectbox(
        "Target curriculum for HIGH",
        grades_list,
        index=min(class_grade, 6) - 1,
    )

    low_thr = st.sidebar.slider("Low performance threshold", 0, 100, 50)
    high_thr = st.sidebar.slider("High performance threshold", 0, 100, 75)
    num_questions = st.sidebar.slider("Number of questions per worksheet", 3, 10, 5)

    def map_to_curriculum(level: str) -> int:
        if level == "Low":
            return grade_for_low
        elif level == "Medium":
            return grade_for_medium
        return grade_for_high

    # ---------- MAIN LAYOUT ----------
    st.markdown(
        """
        <div class="main-card">
            <div class="app-title">English Worksheets Generator</div>
            <div class="app-subtitle">
                Adaptive remedial worksheets based on student performance, curriculum mapping,
                and GPT-powered content generation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ===== STEP 1: Upload Data =====
    with st.container():
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="step-title">Step 1 – Upload student performance file</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="step-help">'
            'Upload the CSV file containing your students’ scores in each skill. '
            'The app will automatically reshape it and classify students into Low, Medium, and High.'
            "</div>",
            unsafe_allow_html=True,
        )

        uploaded = st.file_uploader("Upload CSV (thesis format or prepared file)", type=["csv"])

        if uploaded is None:
            st.info("No file uploaded yet. Please upload your Students.csv file to continue.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        df_raw = pd.read_csv(uploaded)
        st.markdown("**Raw data preview (first rows):**")
        st.dataframe(df_raw.head(), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ===== STEP 2: Process & Inspect =====
    with st.container():
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="step-title">Step 2 – Process data & review groups</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="step-help">'
            "The app converts the table into one row per student and skill, "
            "then classifies performance using the thresholds you set in the sidebar. "
            "Use this view to check that levels and curriculum mapping look reasonable "
            "before generating worksheets."
            "</div>",
            unsafe_allow_html=True,
        )

        df = transform_thesis_format(df_raw)
        df["grade"] = class_grade

        # classify + map
        df["level"] = df["score"].apply(lambda s: classify_level(s, low_thr, high_thr))
        df["target_curriculum_grade"] = df["level"].apply(map_to_curriculum)

        st.markdown("**Processed data (first rows):**")
        st.dataframe(df.head(), use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ===== STEP 3: Select skill & level, generate worksheets =====
    with st.container():
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="step-title">Step 3 – Generate worksheets</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="step-help">'
            "Choose a skill and performance level. The app will generate a tailored worksheet "
            "for each student in that group, using the mapped curriculum grade."
            "</div>",
            unsafe_allow_html=True,
        )

        skills = sorted(df["skill"].unique())
        selected_skill = st.selectbox("Choose a skill", skills)

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
                    worksheet_text = generate_worksheet(
                        client=client,
                        student_name=row["student_name"],
                        student_grade=row["grade"],
                        curriculum_grade=row["target_curriculum_grade"],
                        skill=row["skill"],
                        level=row["level"],
                        num_questions=num_questions,
                    )

                    st.markdown(f"#### Worksheet for {row['student_name']}")
                    st.text(worksheet_text)

        st.markdown("</div>", unsafe_allow_html=True)

    # ===== HELP / USER GUIDE =====
    with st.expander(" Help & User Guide"):
        st.markdown(
            """
            **Step-by-step usage**

            1. **Sign in** using your `@education.qa` account and teacher password.
            2. **Set thresholds and mapping** from the sidebar:
               - Choose the grade you are teaching (e.g. Grade 5).
               - Choose which curriculum grade should be used for Low / Medium / High.
               - Adjust Low / High score thresholds if needed.
            3. **Upload the CSV file** with the students’ scores.
            4. **Review processed data** in Step 2 to ensure levels and target curriculum grades look correct.
            5. **Select a skill and level**, then click **Generate worksheets**.

            If a group is empty, the app will warn you. If the OpenAI API key is missing,
            the app will show an error so that data and content are not silently lost.
            """
        )


if __name__ == "__main__":
    main()
