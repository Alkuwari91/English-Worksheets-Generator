import os
import io
import pandas as pd
import streamlit as st
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

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
    """
    Convert thesis dataset into long format (one row per student & skill).
    Expected columns:
    StudentNumber, StudentName, LanguageFunction, ReadingComprehension, Grammar, Writing
    """
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

    # If already in long format, just return as is
    return df


def build_skill_instruction(skill: str) -> str:
    """Return extra instructions depending on the skill name."""
    s = skill.lower()
    if "grammar" in s:
        return (
            "Focus the questions on grammar usage, sentence structure, "
            "verb tenses, and correct/incorrect forms. Include fill-the-gap "
            "or error-correction style MCQs where appropriate."
        )
    if "reading" in s:
        return (
            "Focus the questions on reading comprehension: main idea, details, "
            "inference, and vocabulary in context related to the passage."
        )
    if "writing" in s:
        return (
            "Focus the questions on writing skills: organizing ideas, "
            "choosing correct connectors, and building clear sentences. "
            "Include questions that ask students to choose the best sentence or connector."
        )
    if "languagefunction" in s or "language function" in s:
        return (
            "Focus the questions on language functions such as making requests, "
            "giving advice, asking for information, agreeing/disagreeing, etc. "
            "Use everyday school situations in the passage and questions."
        )
    # default
    return (
        "Make sure the questions clearly practice the given skill in an age-appropriate way."
    )


def generate_worksheet(
    client: OpenAI,
    student_name: str,
    student_grade: int,
    curriculum_grade: int,
    skill: str,
    level: str,
    num_questions: int = 5
) -> str:
    """Generate worksheet using GPT based on mapped curriculum grade and skill."""

    skill_instruction = build_skill_instruction(skill)

    system_prompt = (
        "You are an educational content generator for primary school English "
        "within the Qatari National Curriculum. Adjust difficulty and language "
        "based on the TARGET curriculum grade. Keep content clear, culturally neutral, "
        "and suitable for students."
    )

    user_prompt = f"""
Student name: {student_name}
Actual school grade: {student_grade}
Target curriculum grade: {curriculum_grade}
Skill: {skill}
Performance level: {level} (Low/Medium/High)

Additional instructions about the skill:
{skill_instruction}

Task:
1. Write a short reading passage (80–120 words) appropriate for the target grade.
2. The passage and questions must clearly practise the given skill.
3. Create {num_questions} multiple-choice questions (A–D).
4. Provide an answer key clearly.

Required format (use exactly these headings):

PASSAGE:
<your passage>

QUESTIONS:
1) ...
A) ...
B) ...
C) ...
D) ...
2) ...
...

ANSWER KEY:
1) C
2) A
...

Only return the worksheet text in this format.
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


def worksheet_to_pdf(student_name: str, worksheet_text: str) -> bytes:
    """
    Convert worksheet text to a simple A4 PDF (in memory).
    Returns the PDF as bytes so it can be downloaded in Streamlit.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Simple margins
    x = 40
    y = height - 60

    # Title
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, f"Worksheet for {student_name}")
    y -= 30

    c.setFont("Helvetica", 11)

    # Draw text with simple line wrapping
    for line in worksheet_text.split("\n"):
        # wrap manually if line too long
        while len(line) > 110:
            part = line[:110]
            c.drawString(x, y, part)
            line = line[110:]
            y -= 14
            if y < 40:
                c.showPage()
                y = height - 60
                c.setFont("Helvetica", 11)
        c.drawString(x, y, line)
        y -= 14
        if y < 40:
            c.showPage()
            y = height - 60
            c.setFont("Helvetica", 11)

    c.showPage()
    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


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

/* Worksheet card */
.worksheet-box {
    background: #ffffff;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.05);
    white-space: pre-wrap;
    font-family: "Cairo", sans-serif;
    font-size: 0.95rem;
    line-height: 1.5;
}

.worksheet-box h4 {
    margin-top: 0;
    margin-bottom: 0.6rem;
    color: #8A1538;
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
        '<div class="step-help">Choose a skill and level, then generate one worksheet per student in this group. Each worksheet is tailored to the selected skill and can be downloaded as a PDF.</div>',
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

                        # Show worksheet nicely
                        st.markdown(
                            f"""
                            <div class="worksheet-box">
                                <h4>Worksheet for {row['student_name']}</h4>
                                {ws_text}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # Create PDF and download button
                        pdf_bytes = worksheet_to_pdf(row["student_name"], ws_text)
                        st.download_button(
                            label=f"Download PDF for {row['student_name']}",
                            data=pdf_bytes,
                            file_name=f"worksheet_{row['student_name']}.pdf",
                            mime="application/pdf",
                        )

                    st.success("All worksheets generated successfully ✅")

                except Exception as e:
                    st.error(f"Error while calling OpenAI API: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
