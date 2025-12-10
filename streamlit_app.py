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
    s = str(skill).lower()
    if "grammar" in s:
        return (
            "Focus the questions on grammar usage, sentence structure, verb tenses, "
            "and error-correction style MCQs, appropriate for the target grade."
        )
    if "reading" in s:
        return (
            "Focus the questions on reading comprehension: main idea, details, "
            "inference, and vocabulary in context related to the passage."
        )
    if "writing" in s:
        return (
            "Focus the questions on writing skills: organising ideas, choosing "
            "correct connectors, and building clear sentences."
        )
    if "languagefunction" in s or "language function" in s:
        return (
            "Focus the questions on language functions such as making requests, "
            "giving advice, asking for information, agreeing and disagreeing, etc."
        )
    return (
        "Make sure the questions clearly practise the given skill in an "
        "age-appropriate way."
    )


def build_rag_context(curriculum_df: pd.DataFrame, skill: str, curriculum_grade: int) -> str:
    """
    Very simple RAG: filter curriculum bank by grade & skill and
    convert rows into short bullet points.
    Expected columns: grade, skill, plus any other descriptive columns.
    """
    if curriculum_df is None:
        return ""

    required_cols = {"grade", "skill"}
    if not required_cols.issubset(curriculum_df.columns):
        return ""

    try:
        temp = curriculum_df.copy()
        temp["grade_str"] = temp["grade"].astype(str)
        mask = (
            (temp["grade_str"] == str(curriculum_grade)) &
            (temp["skill"].astype(str).str.lower() == str(skill).lower())
        )
        subset = temp[mask]
        if subset.empty:
            return ""

        bullets = []
        for _, row in subset.iterrows():
            row_dict = row.to_dict()
            row_dict.pop("grade_str", None)
            g = row_dict.pop("grade", None)
            sk = row_dict.pop("skill", None)
            rest = " | ".join(f"{k}: {v}" for k, v in row_dict.items() if pd.notna(v))
            bullets.append(f"- Grade {g}, Skill {sk}: {rest}")

        return "\n".join(bullets[:8])  # limit context
    except Exception:
        return ""


def generate_worksheet(
    client: OpenAI,
    student_name: str,
    student_grade: int,
    curriculum_grade: int,
    skill: str,
    level: str,
    num_questions: int = 5,
    rag_context: str = ""
) -> str:
    """Generate worksheet using GPT based on mapped curriculum grade, skill, and RAG context."""

    skill_instruction = build_skill_instruction(skill)

    system_prompt = (
        "You are an educational content generator for primary school English "
        "within the Qatari National Curriculum. Adjust difficulty and language "
        "based on the TARGET curriculum grade. Keep content clear, culturally appropriate, "
        "and suitable for students."
    )

    rag_section = ""
    if rag_context:
        rag_section = f"""
Curriculum RAG context (reference material from the official curriculum bank):
{rag_context}

Use this information to align the passage topic, vocabulary, and question focus
with the curriculum expectations for this grade and skill.
"""

    user_prompt = f"""
Student name: {student_name}
Actual school grade: {student_grade}
Target curriculum grade: {curriculum_grade}
Skill: {skill}
Performance level: {level} (Low / Medium / High)

Additional instructions about the skill:
{skill_instruction}

{rag_section}

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


def split_worksheet_and_answer(text: str):
    """Split GPT output into worksheet body (no answers) and answer key."""
    marker = "ANSWER KEY:"
    idx = text.upper().find(marker)
    if idx == -1:
        return text.strip(), "ANSWER KEY:\n(Not clearly provided by the model.)"
    body = text[:idx].strip()
    answer = text[idx:].strip()
    return body, answer


def text_to_pdf(title: str, content: str) -> bytes:
    """
    Convert text to a simple A4 PDF (in memory).
    Returns the PDF as bytes so it can be downloaded in Streamlit.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    x = 40
    y = height - 60

    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, title)
    y -= 30

    c.setFont("Helvetica", 11)

    for line in content.split("\n"):
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
# Custom CSS (ألوان مرتّبة + Tabs احترافية)
# ==============================

CUSTOM_CSS = """
<style>

/* نخفي هيدر ستريملت الافتراضي */
header, footer {visibility: hidden;}

body, .stApp {
    background: #f3f5f9;           /* رمادي فاتح مريح للعين */
    font-family: "Cairo", sans-serif;
    color: #1f2933;                /* نص أساسي غامق وواضح */
}

/* الهيدر العلوي */
.app-header {
    width: 100%;
    padding: 1.4rem 2rem;
    background: linear-gradient(135deg, #8A1538, #5b0c25); /* درجات ماروني */
    border-radius: 0 0 18px 18px;
    color: #ffffff;
    margin-bottom: 1.2rem;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.18);
}

.header-title {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: 0.03em;
    margin-bottom: 0.15rem;
}

.header-sub {
    font-size: 0.95rem;
    opacity: 0.95;
}

/* الكروت الأساسية */
.card {
    background: #ffffff;
    padding: 1.4rem 1.6rem;
    border-radius: 18px;
    margin-bottom: 1.1rem;
    box-shadow: 0 6px 20px rgba(15, 23, 42, 0.06);
    border: 1px solid #e5e7f0;
}

.step-title {
    color: #7b1035;                 /* ماروني أغمق للعنوانين */
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 0.35rem;
}

.step-help {
    color: #4b5563;                 /* رمادي واضح للنصوص */
    font-size: 0.92rem;
}

/* Badges للأدوات */
.tool-tag {
    display: inline-block;
    background: #fdf2f7;
    color: #9d174d;
    border-radius: 999px;
    padding: 3px 12px;
    font-size: 0.78rem;
    margin-right: 5px;
    margin-top: 4px;
}

/* تنسيق التابات (Tabs) */
.stTabs {
    margin-top: 0.3rem;
    margin-bottom: 0.8rem;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem;
}

.stTabs [data-baseweb="tab"] {
    background-color: #e5e7f0;      /* خلفية التاب غير المختار */
    color: #4b5563;
    border-radius: 999px;
    padding: 0.4rem 1.1rem;
    font-size: 0.92rem;
    border: none;
    box-shadow: none;
}

.stTabs [data-baseweb="tab"]:hover {
    background-color: #d4d7e5;
    color: #111827;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #8A1538, #b91c4b);
    color: #ffffff;
    font-weight: 700;
    box-shadow: 0 4px 12px rgba(148, 27, 66, 0.35);
}

/* أزرار ستريملت */
.stButton > button {
    background: linear-gradient(135deg, #8A1538, #b91c4b);
    color: #ffffff;
    border-radius: 999px;
    border: none;
    padding: 0.45rem 1.3rem;
    font-weight: 600;
    font-size: 0.9rem;
    box-shadow: 0 4px 12px rgba(148, 27, 66, 0.35);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #7a0f31, #a01a44);
}

/* download buttons */
.stDownloadButton > button {
    border-radius: 999px;
    border: 1px solid #e5e7f0;
    background: #ffffff;
    color: #374151;
    padding: 0.35rem 1rem;
    font-size: 0.86rem;
}

.stDownloadButton > button:hover {
    background: #f3f4ff;
    border-color: #c7d2fe;
}

/* سلايدر وعدد الأسئلة */
.stSlider > div > div > div {
    color: #7b1035;
}

/* رسائل التنبيه */
.stAlert {
    border-radius: 12px;
}

/* نخلي النص داخل التطبيق غامق وواضح دائمًا */
.stMarkdown, .stText, .stDataFrame {
    color: #1f2933;
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

    # HEADER
    st.markdown(
        """
        <div class="app-header">
            <div class="header-title">English Worksheets Generator</div>
            <div class="header-sub">
                Prototype for adaptive remedial worksheets using Pandas + RAG + GPT API
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # API KEY
    api_key = get_api_key()
    if not api_key:
        st.error("OPENAI_API_KEY is missing. Add it in Settings → Secrets.")
        return

    client = OpenAI(api_key=api_key)

    # Session state
    if "processed_df" not in st.session_state:
        st.session_state["processed_df"] = None
    if "curriculum_df" not in st.session_state:
        st.session_state["curriculum_df"] = None

    # Tabs
    tab_overview, tab_data, tab_generate, tab_help = st.tabs(
        ["Overview", "Data & RAG", "Generate Worksheets", "Help & Tools"]
    )

    # -------- Overview --------
    with tab_overview:
        st.markdown(
            """
            <div class="card">
                <div class="step-title">How the prototype works</div>
                <p class="step-help">
                This prototype follows three main steps:
                </p>
                <ol class="step-help">
                  <li><b>Upload & process student performance data</b> (Pandas) to classify students into Low / Medium / High for each skill.</li>
                  <li><b>Attach curriculum knowledge</b> via a small curriculum bank CSV. This is used as a simple <b>RAG</b> layer to ground GPT in real topics.</li>
                  <li><b>Generate personalised worksheets</b> for each student using the GPT API, aligned with the selected skill and curriculum grade.</li>
                </ol>
                <p class="step-help">
                  Use the tabs above to move between steps. The prototype does not store any personal data; all processing happens in memory
                  for demonstration purposes.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------- Data & RAG --------
    with tab_data:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="step-title">Step 1 — Upload student performance CSV</div>', unsafe_allow_html=True)
        st.markdown(
            '<span class="tool-tag">Pandas</span><span class="tool-tag">Data validation</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="step-help">Expected format (from thesis dataset): '
            '<code>StudentNumber, StudentName, LanguageFunction, ReadingComprehension, Grammar, Writing</code>.</p>',
            unsafe_allow_html=True,
        )

        uploaded = st.file_uploader("Upload Students.csv", type=["csv"], key="students_csv")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="step-title">Step 2 — Optional curriculum bank for RAG</div>', unsafe_allow_html=True)
        st.markdown(
            '<span class="tool-tag">RAG</span><span class="tool-tag">Curriculum bank</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="step-help">Upload a small CSV with at least columns '
            '<code>grade</code> and <code>skill</code>, plus any descriptive fields '
            '(objective, topic, example, etc.). The system will retrieve rows matching '
            'the target grade and skill to guide the GPT prompts.</p>',
            unsafe_allow_html=True,
        )

        curriculum_file = st.file_uploader(
            "Upload curriculum bank CSV (optional, used for RAG)",
            type=["csv"],
            key="curriculum_csv"
        )

        if curriculum_file is not None:
            try:
                cur_df = pd.read_csv(curriculum_file)
                st.session_state["curriculum_df"] = cur_df
                st.write("Curriculum bank preview:")
                st.dataframe(cur_df.head(), use_container_width=True)
            except Exception as e:
                st.error(f"Could not read curriculum bank: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="step-title">Step 3 — Process data & classify levels</div>', unsafe_allow_html=True)
        st.markdown(
            '<span class="tool-tag">Pandas</span><span class="tool-tag">Rule-based classifier</span>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            class_grade = st.selectbox(
                "Actual student grade (the class you are teaching)", [1, 2, 3, 4, 5, 6], index=4
            )
            low_thr = st.slider("Low threshold", 0, 100, 50)
            high_thr = st.slider("High threshold", 0, 100, 75)

        with col2:
            grade_for_low = st.selectbox("Curriculum grade for LOW", [1, 2, 3, 4, 5, 6], index=0)
            grade_for_medium = st.selectbox("Curriculum grade for MEDIUM", [1, 2, 3, 4, 5, 6], index=class_grade-1)
            grade_for_high = st.selectbox("Curriculum grade for HIGH", [1, 2, 3, 4, 5, 6], index=min(class_grade, 6)-1)

        def map_to_curriculum(level: str):
            if level == "Low":
                return grade_for_low
            elif level == "Medium":
                return grade_for_medium
            return grade_for_high

        if st.button("Process student data"):
            if uploaded is None:
                st.error("Please upload the student performance CSV first.")
            else:
                try:
                    df_raw = pd.read_csv(uploaded)
                    df = transform_thesis_format(df_raw)
                    df["grade"] = class_grade
                    df["level"] = df["score"].apply(lambda x: classify_level(x, low_thr, high_thr))
                    df["target_curriculum_grade"] = df["level"].apply(map_to_curriculum)

                    st.session_state["processed_df"] = df

                    st.success("Student data processed successfully.")
                    st.write("Processed data preview:")
                    st.dataframe(df.head(), use_container_width=True)
                except Exception as e:
                    st.error(f"Error while processing data: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # -------- Generate Worksheets --------
    with tab_generate:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="step-title">Step 4 — Generate worksheets (PDF only)</div>', unsafe_allow_html=True)
        st.markdown(
            '<span class="tool-tag">GPT API</span><span class="tool-tag">RAG</span><span class="tool-tag">PDF export</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="step-help">For each student in the selected skill and level, '
            'the system generates a personalised worksheet and a separate answer key. '
            'Only download buttons are shown (no raw text on screen).</p>',
            unsafe_allow_html=True,
        )

        df = st.session_state.get("processed_df", None)
        curriculum_df = st.session_state.get("curriculum_df", None)

        if df is None:
            st.info("Please go to the 'Data & RAG' tab and process the student data first.")
        else:
            skills = sorted(df["skill"].unique())
            selected_skill = st.selectbox("Choose skill", skills)

            levels = ["Low", "Medium", "High"]
            selected_level = st.selectbox("Choose performance level", levels)

            num_q = st.slider("Number of questions per worksheet", 3, 10, 5)

            target_df = df[(df["skill"] == selected_skill) & (df["level"] == selected_level)]

            st.markdown(f"Students in this group: **{len(target_df)}**")

            if st.button("Generate PDFs for this group"):
                if target_df.empty:
                    st.error("No students match this skill + level.")
                else:
                    with st.spinner("Generating worksheets and answer keys…"):
                        try:
                            for _, row in target_df.iterrows():
                                rag_context = build_rag_context(
                                    curriculum_df,
                                    skill=row["skill"],
                                    curriculum_grade=row["target_curriculum_grade"],
                                )

                                full_text = generate_worksheet(
                                    client=client,
                                    student_name=row["student_name"],
                                    student_grade=row["grade"],
                                    curriculum_grade=row["target_curriculum_grade"],
                                    skill=row["skill"],
                                    level=row["level"],
                                    num_questions=num_q,
                                    rag_context=rag_context,
                                )

                                worksheet_body, answer_key = split_worksheet_and_answer(full_text)

                                ws_pdf = text_to_pdf(
                                    title=f"Worksheet for {row['student_name']}",
                                    content=worksheet_body,
                                )
                                ak_pdf = text_to_pdf(
                                    title=f"Answer Key for {row['student_name']}",
                                    content=answer_key,
                                )

                                st.markdown(f"#### {row['student_name']}")
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.download_button(
                                        label="Download worksheet PDF",
                                        data=ws_pdf,
                                        file_name=f"worksheet_{row['student_name']}.pdf",
                                        mime="application/pdf",
                                    )
                                with c2:
                                    st.download_button(
                                        label="Download answer key PDF",
                                        data=ak_pdf,
                                        file_name=f"answer_key_{row['student_name']}.pdf",
                                        mime="application/pdf",
                                    )

                            st.success("All PDFs generated successfully ✅")
                        except Exception as e:
                            st.error(f"Error while generating worksheets: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # -------- Help & Tools --------
    with tab_help:
        st.markdown(
            """
            <div class="card">
                <div class="step-title">Tools & implementation summary</div>
                <p class="step-help">
                    This tab documents the main tools used in the prototype for your dissertation:
                </p>
                <ul class="step-help">
                    <li><b>Pandas</b> — used to read the CSV files, reshape the thesis dataset into long format, and perform rule-based classification of students into performance levels.</li>
                    <li><b>Rule-based classifier</b> — thresholds (Low / Medium / High) based on total scores, then mapped to curriculum grades chosen by the teacher.</li>
                    <li><b>RAG (Retrieval-Augmented Generation)</b> — a simple curriculum bank CSV with columns such as <code>grade</code>, <code>skill</code>, and objectives/examples is used as a retrieval layer. The app filters this bank by grade and skill and injects short bullet points into the GPT prompt.</li>
                    <li><b>GPT API (gpt-4o-mini)</b> — generates a reading passage, skill-focused questions, and an answer key per student, guided by both performance data and RAG context.</li>
                    <li><b>PDF generation (reportlab)</b> — converts the generated worksheet text and answer key into separate A4 PDFs which the teacher can download.</li>
                </ul>
                <p class="step-help">
                    These points can be used directly in the Methodology and Implementation chapters to explain how the app operationalises data analysis, RAG, and AI-based content generation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
