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

/* سلايدر وعدد الأسئلة */
.stSlider > div > div > div {
    color: #7b1035;
}

/* تنسيق رسائل النجاح/الخطأ */
.stAlert {
    border-radius: 12px;
}

/* نحسّن شكل download button */
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

/* نخلي النص داخل التطبيق غامق وواضح دائمًا */
.stMarkdown, .stText, .stDataFrame {
    color: #1f2933;
}

</style>
"""
