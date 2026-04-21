import math

import warnings

from pathlib import Path

import joblib

import pandas as pd

import streamlit as st

from sklearn.exceptions import InconsistentVersionWarning


st.set_page_config(page_title="Gender Classification", layout="wide")

BASE_DIR = Path(__file__).resolve().parents[1]

# Load the same saved classifier artifact used by the Flask version of this demo.
MODEL_PATH = BASE_DIR / "joblib files" / "gender_classifier_best_model.joblib"

# The users dataset is reused for metrics, sample names, and overview charts.
USERS_DATA_PATH = BASE_DIR / "dataset" / "travel_capstone" / "users.csv"


@st.cache_resource
def load_model():
    # This older joblib artifact can still load cleanly even when sklearn versions differ.
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_users_data():
    # Cache the source data because the app reads it several times during one rerun.
    return pd.read_csv(USERS_DATA_PATH)


def get_selected_theme():
    # Carry the theme choice across reruns so the page does not jump back to default styling.
    if "gender_theme_mode" not in st.session_state:
        st.session_state.gender_theme_mode = "System"

    if st.session_state.gender_theme_mode == "White":
        st.session_state.gender_theme_mode = "Light"

    return st.session_state.gender_theme_mode


def format_percent(value):
    # Keep the strength label formatting consistent across the result card and metrics.
    return f"{value:.1f}%"


def calculate_decision_strength(raw_score):
    # Convert the raw margin into a friendlier confidence-style percentage for the UI.
    strength = 50 + (1 - math.exp(-abs(float(raw_score)))) * 49

    return round(strength, 1)


def explain_strength(strength):
    # Short labels make the result easier to read than exposing only the raw score.
    if strength >= 85:
        return "Strong model signal"

    if strength >= 70:
        return "Moderate model signal"

    return "Close decision boundary"


def apply_theme_css(selected_theme):
    # The palette swaps here without changing the actual layout structure.
    if selected_theme == "Light":
        theme_variables = """
        :root {
            --bg: #fff7ed;
            --bg-2: #eef8f4;
            --surface: #ffffff;
            --surface-soft: #f8fbf8;
            --text: #1f2933;
            --muted: #637083;
            --border: #eadfd2;
            --primary: #0f766e;
            --primary-2: #2563eb;
            --accent: #f97316;
            --accent-soft: rgba(249, 115, 22, 0.14);
            --button-text: #ffffff;
            --shadow: 0 22px 60px rgba(99, 72, 38, 0.14);
        }
        """

    elif selected_theme == "Dark":
        theme_variables = """
        :root {
            --bg: #071423;
            --bg-2: #102b36;
            --surface: #13243a;
            --surface-soft: #1e3652;
            --text: #f8fafc;
            --muted: #c4d3e6;
            --border: #3b587a;
            --primary: #5eead4;
            --primary-2: #93c5fd;
            --accent: #fb923c;
            --accent-soft: rgba(251, 146, 60, 0.20);
            --button-text: #071423;
            --shadow: 0 24px 70px rgba(0, 0, 0, 0.38);
        }
        """

    else:
        theme_variables = """
        @media (prefers-color-scheme: light) {
            :root {
                --bg: #fff7ed;
                --bg-2: #eef8f4;
                --surface: #ffffff;
                --surface-soft: #f8fbf8;
                --text: #1f2933;
                --muted: #637083;
                --border: #eadfd2;
                --primary: #0f766e;
                --primary-2: #2563eb;
                --accent: #f97316;
                --accent-soft: rgba(249, 115, 22, 0.14);
                --button-text: #ffffff;
                --shadow: 0 22px 60px rgba(99, 72, 38, 0.14);
            }
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --bg: #071423;
                --bg-2: #102b36;
                --surface: #13243a;
                --surface-soft: #1e3652;
                --text: #f8fafc;
                --muted: #c4d3e6;
                --border: #3b587a;
                --primary: #5eead4;
                --primary-2: #93c5fd;
                --accent: #fb923c;
                --accent-soft: rgba(251, 146, 60, 0.20);
                --button-text: #071423;
                --shadow: 0 24px 70px rgba(0, 0, 0, 0.38);
            }
        }
        """

    st.markdown(
        f"""
        <style>
        {theme_variables}
        #MainMenu, footer, [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {{
            visibility: hidden;
            height: 0;
        }}
        .stApp {{
            background:
                radial-gradient(circle at 8% 8%, var(--accent-soft), transparent 24rem),
                radial-gradient(circle at 88% 6%, rgba(15, 118, 110, 0.20), transparent 25rem),
                linear-gradient(135deg, var(--bg), var(--bg-2));
            color: var(--text);
            font-family: "Aptos", "Candara", "Trebuchet MS", sans-serif;
        }}
        [data-testid="stAppViewContainer"] {{
            background: transparent;
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, var(--surface), var(--surface-soft));
            border-right: 1px solid var(--border);
        }}
        [data-testid="stSidebar"] * {{
            color: var(--text) !important;
        }}
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {{
            color: var(--text) !important;
            font-weight: 800 !important;
        }}
        .block-container {{
            max-width: 1180px;
            padding-top: 1.5rem;
            padding-bottom: 2.4rem;
        }}
        h1, h2, h3, h4, h5, h6, p, label, span, div {{
            color: inherit;
        }}
        [data-testid="stMarkdownContainer"] p {{
            color: var(--muted);
        }}
        .hero-card {{
            background:
                radial-gradient(circle at 88% 18%, rgba(255, 255, 255, 0.20), transparent 15rem),
                linear-gradient(135deg, #0f766e 0%, #164e63 54%, #f97316 132%);
            border: 1px solid rgba(255, 255, 255, 0.24);
            border-radius: 32px;
            padding: 2.1rem;
            box-shadow: var(--shadow);
            margin-bottom: 1.1rem;
            overflow: hidden;
        }}
        .hero-badge {{
            display: inline-block;
            background: rgba(255, 255, 255, 0.16);
            color: #fff7ed;
            padding: 0.38rem 0.82rem;
            border-radius: 999px;
            font-size: 0.86rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.9rem;
        }}
        .hero-title {{
            max-width: 760px;
            color: #fffaf0;
            font-family: "Georgia", "Cambria", serif;
            font-size: clamp(2.5rem, 5.6vw, 5rem);
            font-weight: 700;
            line-height: 0.95;
            letter-spacing: -0.06em;
            margin-bottom: 0.9rem;
        }}
        .hero-copy {{
            max-width: 780px;
            color: rgba(255, 250, 240, 0.88);
            font-size: 1.05rem;
            line-height: 1.7;
        }}
        .panel-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 26px;
            padding: 1.25rem;
            box-shadow: var(--shadow);
            margin-bottom: 1rem;
        }}
        .section-kicker {{
            color: var(--primary);
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin-bottom: 0.3rem;
        }}
        .section-title {{
            color: var(--text);
            font-family: "Georgia", "Cambria", serif;
            font-size: 1.55rem;
            font-weight: 700;
            line-height: 1.12;
            margin-bottom: 0.45rem;
        }}
        .section-copy {{
            color: var(--muted);
            line-height: 1.65;
        }}
        .result-card {{
            background: linear-gradient(145deg, var(--surface), var(--surface-soft));
            border: 1px solid var(--border);
            border-radius: 28px;
            padding: 1.45rem;
            box-shadow: var(--shadow);
        }}
        .result-label {{
            color: var(--muted);
            font-size: 0.84rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }}
        .result-value {{
            color: var(--text);
            font-family: "Georgia", "Cambria", serif;
            font-size: clamp(2.5rem, 4vw, 4rem);
            font-weight: 700;
            line-height: 1;
            letter-spacing: -0.05em;
            margin: 0.75rem 0 0.35rem;
            text-transform: capitalize;
        }}
        .result-note {{
            color: var(--muted);
            line-height: 1.55;
        }}
        .name-chip {{
            display: inline-block;
            background: var(--accent-soft);
            color: var(--accent);
            border-radius: 999px;
            padding: 0.45rem 0.72rem;
            font-weight: 800;
            margin-top: 1rem;
        }}
        div[data-testid="stForm"] {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 26px;
            padding: 1.2rem;
            box-shadow: var(--shadow);
        }}
        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div {{
            background: var(--surface-soft) !important;
            border: 1px solid var(--border) !important;
            border-radius: 18px !important;
            color: var(--text) !important;
            box-shadow: none !important;
            min-height: 3rem !important;
        }}
        div[data-baseweb="input"] input {{
            color: var(--text) !important;
            background: transparent !important;
            caret-color: var(--primary) !important;
        }}
        div[data-baseweb="input"] input::placeholder {{
            color: var(--muted) !important;
            opacity: 1 !important;
        }}
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div {{
            color: var(--text) !important;
        }}
        div[data-baseweb="popover"],
        ul[role="listbox"] {{
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            color: var(--text) !important;
        }}
        li[role="option"],
        li[role="option"] div,
        li[role="option"] span {{
            background: var(--surface) !important;
            color: var(--text) !important;
        }}
        li[role="option"]:hover,
        li[role="option"][aria-selected="true"] {{
            background: var(--surface-soft) !important;
            color: var(--text) !important;
        }}
        .stRadio label,
        .stTextInput label,
        .stSelectbox label,
        .stSlider label {{
            color: var(--text) !important;
        }}
        .stButton > button,
        div[data-testid="stFormSubmitButton"] button,
        button[data-testid="stBaseButton-secondaryFormSubmit"],
        button[data-testid="stBaseButton-primaryFormSubmit"] {{
            min-height: 3rem;
            border: 0 !important;
            border-radius: 999px;
            background: linear-gradient(135deg, var(--primary), var(--primary-2)) !important;
            color: var(--button-text) !important;
            font-weight: 800 !important;
            box-shadow: 0 16px 34px rgba(15, 118, 110, 0.26) !important;
        }}
        .stButton > button:hover,
        div[data-testid="stFormSubmitButton"] button:hover,
        button[data-testid="stBaseButton-secondaryFormSubmit"]:hover,
        button[data-testid="stBaseButton-primaryFormSubmit"]:hover {{
            color: var(--button-text) !important;
            transform: translateY(-1px);
            box-shadow: 0 20px 42px rgba(15, 118, 110, 0.34) !important;
        }}
        .stButton > button p,
        div[data-testid="stFormSubmitButton"] button p,
        button[data-testid="stBaseButton-secondaryFormSubmit"] p,
        button[data-testid="stBaseButton-primaryFormSubmit"] p {{
            color: var(--button-text) !important;
            font-weight: 800 !important;
        }}
        [data-testid="stMetric"] {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 0.9rem 1rem;
            box-shadow: var(--shadow);
        }}
        [data-testid="stMetric"] label {{
            color: var(--muted) !important;
        }}
        [data-testid="stMetricValue"] {{
            color: var(--text) !important;
            font-family: "Georgia", "Cambria", serif;
        }}
        [data-baseweb="tab-list"] {{
            gap: 0.5rem;
        }}
        button[data-baseweb="tab"] {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: 0.46rem 1rem;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            background: var(--primary);
            border-color: var(--primary);
        }}
        button[data-baseweb="tab"][aria-selected="true"] p {{
            color: #ffffff;
            font-weight: 800;
        }}
        [data-testid="stDataFrame"] {{
            border-radius: 20px;
            overflow: hidden;
        }}
        .stAlert {{
            border-radius: 20px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# Load the model and the backing dataset before rendering the dashboard sections.
model = load_model()

users_df = load_users_data()

# Separate labeled and unlabeled records so the summary cards reflect what the model can learn from.
labeled_users_df = users_df[users_df["gender"].isin(["female", "male"])].copy()

total_records = len(users_df)

labeled_records = len(labeled_users_df)

unlabeled_records = int((users_df["gender"] == "none").sum())

company_count = users_df["company"].nunique()

gender_distribution_df = users_df["gender"].value_counts().rename_axis("gender").reset_index(name="count")

company_distribution_df = users_df["company"].value_counts().rename_axis("company").reset_index(name="count")

# Apply the last selected theme immediately, then let the radio update it if needed.
selected_theme = get_selected_theme()

apply_theme_css(selected_theme)

selected_theme = st.radio(
    "Theme",
    options=["System", "Light", "Dark"],
    index=["System", "Light", "Dark"].index(selected_theme),
    horizontal=True,
)

st.session_state.gender_theme_mode = selected_theme

apply_theme_css(selected_theme)

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-badge">Classification Model</div>
        <div class="hero-title">Gender Classification Dashboard</div>
        <div class="hero-copy">
            This app presents a trained gender classification model in a clear and interactive format.
            Enter a name, run the prediction, and review the dataset summary used for project understanding.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4, gap="medium")

metric_col_1.metric("User Records", f"{total_records:,}")

metric_col_2.metric("Labeled Names", f"{labeled_records:,}")

metric_col_3.metric("None Labels", f"{unlabeled_records:,}")

metric_col_4.metric("Companies", f"{company_count:,}")

st.write("")

input_col, result_col = st.columns([1.02, 0.98], gap="large")

with input_col:
    # The form keeps the typed input and dataset sample selector together in one prediction flow.
    st.markdown(
        """
        <div class="panel-card">
            <div class="section-kicker">Prediction Workspace</div>
            <div class="section-title">Enter a name</div>
            <div class="section-copy">
                The model reads the name as text and predicts one of the trained classes: female or male.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("gender_prediction_form", border=False):
        person_name = st.text_input(
            "Person name",
            value="Anita Sharma",
            placeholder="Example: Anita Sharma",
        )

        sample_name = st.selectbox(
            "Try a sample from the dataset",
            options=["Use typed name"] + users_df["name"].head(30).tolist(),
        )

        submitted = st.form_submit_button("Classify Gender", use_container_width=True)

selected_name = person_name if sample_name == "Use typed name" else sample_name

selected_name = selected_name.strip()

if submitted and selected_name:
    # Run the saved classifier and keep both the label and its raw decision score for explanation.
    predicted_gender = model.predict([selected_name])[0]

    raw_score = float(model.decision_function([selected_name])[0])

    decision_strength = calculate_decision_strength(raw_score)

    st.session_state.gender_prediction_result = {
        "name": selected_name,
        "gender": predicted_gender,
        "raw_score": raw_score,
        "strength": decision_strength,
        "strength_label": explain_strength(decision_strength),
    }

elif submitted and not selected_name:
    # Clear the previous result when the form is submitted without a usable name.
    st.session_state.gender_prediction_result = None

    st.warning("Please enter a name before running the prediction.")

with result_col:
    # Show the current prediction when present, otherwise keep the result panel ready for the next run.
    st.markdown(
        """
        <div class="panel-card">
            <div class="section-kicker">Model Output</div>
            <div class="section-title">Prediction result</div>
            <div class="section-copy">
                The result card shows the predicted class and a decision-strength indicator from the model margin.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    prediction_result = st.session_state.get("gender_prediction_result")

    if prediction_result:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Predicted gender</div>
                <div class="result-value">{prediction_result["gender"]}</div>
                <div class="result-note">
                    Decision strength: {format_percent(prediction_result["strength"])}
                    ({prediction_result["strength_label"]}).
                </div>
                <div class="name-chip">Name checked: {prediction_result["name"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        result_metric_col_1, result_metric_col_2 = st.columns(2)

        result_metric_col_1.metric("Raw Decision Score", f"{prediction_result['raw_score']:.4f}")

        result_metric_col_2.metric("Selected Class", prediction_result["gender"].title())

    else:
        st.markdown(
            """
            <div class="result-card">
                <div class="result-label">Predicted gender</div>
                <div class="result-value">Ready</div>
                <div class="result-note">
                    Enter a name and click Classify Gender to generate a prediction.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")

overview_tab, data_tab = st.tabs(["Dataset Overview", "Dataset Preview"])

with overview_tab:
    # These views help explain the training data distribution behind the classifier demo.
    overview_col_1, overview_col_2 = st.columns([1, 1], gap="large")

    with overview_col_1:
        st.markdown(
            """
            <div class="panel-card">
                <div class="section-kicker">Class Balance</div>
                <div class="section-title">Gender label distribution</div>
                <div class="section-copy">
                    This chart shows how the available labels are distributed in the users dataset.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.bar_chart(gender_distribution_df.set_index("gender")["count"], height=320)

    with overview_col_2:
        st.markdown(
            """
            <div class="panel-card">
                <div class="section-kicker">Dataset Source</div>
                <div class="section-title">Company distribution</div>
                <div class="section-copy">
                    This table summarizes how many user records are present for each company.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.dataframe(company_distribution_df, width="stretch", hide_index=True)

with data_tab:
    # Keep a raw preview handy so the dataset columns can be checked during demos.
    st.markdown(
        """
        <div class="panel-card">
            <div class="section-kicker">Source Data</div>
            <div class="section-title">Users dataset preview</div>
            <div class="section-copy">
                The preview helps verify the name, company, gender, and age fields used in the project.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    preview_rows = st.slider("Rows to preview", min_value=5, max_value=40, value=12)

    st.dataframe(users_df.head(preview_rows), width="stretch", hide_index=True)
