from pathlib import Path

import joblib

import pandas as pd

import streamlit as st


st.set_page_config(page_title="Hotel Recommender", layout="wide")

BASE_DIR = Path(__file__).resolve().parents[1]

# Reuse the saved recommendation bundle instead of rebuilding similarities at app startup.
MODEL_PATH = BASE_DIR / "joblib files" / "hotel_recommender_simple.joblib"

# The hotel CSV provides the descriptive fields used to explain recommendations.
HOTELS_DATA_PATH = BASE_DIR / "dataset" / "travel_capstone" / "hotels.csv"


@st.cache_data
def load_assets():
    # Load the saved recommender and the hotel metadata once for the current session.
    model_bundle = joblib.load(MODEL_PATH)

    hotels_df = pd.read_csv(HOTELS_DATA_PATH)

    return model_bundle, hotels_df


def build_hotel_summary(hotels_df):
    # Collapse repeated booking rows into one hotel-level summary for the dashboard tables.
    summary_df = (
        hotels_df.groupby("name", as_index=False)
        .agg(
            place=("place", "first"),
            avg_price=("price", "mean"),
            avg_total=("total", "mean"),
            avg_days=("days", "mean"),
            booking_count=("travelCode", "count"),
        )
    )

    summary_df["avg_price"] = summary_df["avg_price"].round(2)

    summary_df["avg_total"] = summary_df["avg_total"].round(2)

    summary_df["avg_days"] = summary_df["avg_days"].round(1)

    return summary_df


def get_popular_hotels(model_bundle, hotel_summary_df):
    # Enrich the saved popularity ranking with hotel metadata from the CSV.
    popular_df = model_bundle["popular_hotels"].copy()

    popular_df = popular_df.merge(hotel_summary_df, on="name", how="left")

    popular_df["avg_rating"] = popular_df["avg_rating"].round(2)

    return popular_df


def get_similar_hotels(model_bundle, hotel_summary_df, selected_hotel, top_n):
    # This view combines content similarity and interaction similarity into one simple score.
    hotel_index = model_bundle["hotel_index"]

    if selected_hotel not in hotel_index.index:
        return pd.DataFrame()

    selected_position = hotel_index[selected_hotel]

    content_similarity = model_bundle["content_similarity"]

    item_similarity_df = model_bundle["item_similarity_df"]

    hotel_names = list(item_similarity_df.columns)

    recommendation_rows = []

    for hotel_name in hotel_names:
        if hotel_name == selected_hotel:
            continue

        candidate_position = hotel_index[hotel_name]

        content_score = float(content_similarity[selected_position][candidate_position])

        collaborative_score = float(item_similarity_df.loc[selected_hotel, hotel_name])

        combined_score = round((content_score + collaborative_score) / 2, 4)

        recommendation_rows.append(
            {
                "name": hotel_name,
                "content_score": round(content_score, 4),
                "collaborative_score": round(collaborative_score, 4),
                "combined_score": combined_score,
            }
        )

    recommendations_df = pd.DataFrame(recommendation_rows)

    recommendations_df = recommendations_df.sort_values(
        by="combined_score", ascending=False
    ).head(top_n)

    recommendations_df = recommendations_df.merge(
        hotel_summary_df, on="name", how="left"
    )

    return recommendations_df


def get_user_recommendations(model_bundle, hotel_summary_df, selected_user, top_n):
    # Score unseen hotels for one user by weighting item similarity against their existing history.
    user_item_matrix = model_bundle["user_item_matrix"]

    item_similarity_df = model_bundle["item_similarity_df"]

    if selected_user not in user_item_matrix.index:
        return pd.DataFrame(), pd.DataFrame()

    user_ratings = user_item_matrix.loc[selected_user]

    visited_hotels = user_ratings[user_ratings > 0].sort_values(ascending=False)

    score_rows = []

    for candidate_hotel in user_item_matrix.columns:
        if user_ratings[candidate_hotel] > 0:
            continue

        similarity_vector = item_similarity_df[candidate_hotel]

        weighted_scores = similarity_vector * user_ratings

        numerator = weighted_scores.sum()

        denominator = similarity_vector.abs().sum()

        predicted_score = 0.0 if denominator == 0 else float(numerator / denominator)

        score_rows.append(
            {
                "name": candidate_hotel,
                "predicted_score": round(predicted_score, 4),
            }
        )

    recommendations_df = pd.DataFrame(score_rows)

    if recommendations_df.empty:
        recommendations_df = pd.DataFrame(
            columns=[
                "name",
                "predicted_score",
                "place",
                "avg_price",
                "avg_total",
                "avg_days",
                "booking_count",
            ]
        )

        history_df = visited_hotels.reset_index()

        history_df.columns = ["name", "interaction_score"]

        history_df = history_df.merge(hotel_summary_df, on="name", how="left")

        return recommendations_df, history_df

    recommendations_df = recommendations_df.sort_values(
        by="predicted_score", ascending=False
    ).head(top_n)

    recommendations_df = recommendations_df.merge(
        hotel_summary_df, on="name", how="left"
    )

    history_df = visited_hotels.reset_index()

    history_df.columns = ["name", "interaction_score"]

    history_df = history_df.merge(hotel_summary_df, on="name", how="left")

    return recommendations_df, history_df


# Load the saved artifact and derive the dashboard tables before the layout starts.
model_bundle, hotels_df = load_assets()

hotel_summary_df = build_hotel_summary(hotels_df)

popular_hotels_df = get_popular_hotels(model_bundle, hotel_summary_df)

evaluation_results_df = model_bundle["evaluation_results"].copy()

user_item_matrix_df = model_bundle["user_item_matrix"]

user_seen_counts = (user_item_matrix_df > 0).sum(axis=1)

eligible_users = user_seen_counts[user_seen_counts < user_item_matrix_df.shape[1]].index.tolist()

top_places_df = (
    hotels_df.groupby("place", as_index=False)
    .agg(booking_count=("travelCode", "count"))
    .sort_values(by="booking_count", ascending=False)
)

best_model_row = evaluation_results_df.sort_values(
    by="HitRate@5", ascending=False
).iloc[0]

# Keep the page styling local to this app so the demo feels distinct from the other dashboards.
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f4f7fb 0%, #eef3f8 100%);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
        padding: 2rem;
        border-radius: 22px;
        color: #ffffff;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.18);
        margin-bottom: 1.25rem;
    }
    .hero-tag {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.14);
        font-size: 0.9rem;
        margin-bottom: 0.8rem;
    }
    .hero-title {
        font-size: 2.7rem;
        font-weight: 800;
        line-height: 1.15;
        margin-bottom: 0.55rem;
    }
    .hero-text {
        font-size: 1.05rem;
        color: #dbe7f4;
        max-width: 820px;
    }
    .section-card {
        background: #ffffff;
        border: 1px solid #d9e4f2;
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06);
    }
    .small-note {
        color: #4b5563;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    # The sidebar explains the recommender modes before the user starts exploring the tabs.
    st.header("Project Guide")

    st.write(
        "This dashboard demonstrates a hotel recommendation system built from booking data and a saved machine learning artifact."
    )

    st.write("Methods used:")
    st.write("- Popularity-based ranking")
    st.write("- Content-based similarity")
    st.write("- Collaborative filtering")

    st.write("Files used:")
    st.code(
        "joblib files/hotel_recommender_simple.joblib\n"
        "dataset/travel_capstone/hotels.csv"
    )

    st.info(
        "Start with the Overview tab, then test Similar Hotels and User Recommendations to see how the recommender behaves."
    )


st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-tag">MSc CS Project Demo</div>
        <div class="hero-title">Voyage Analytics Hotel Recommendation Dashboard</div>
        <div class="hero-text">
            This dashboard presents a hybrid hotel recommendation system using popularity-based ranking,
            content similarity, and collaborative filtering. It loads the trained artifact from
            <b>hotel_recommender_simple.joblib</b> and turns it into an interactive project demo.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

hero_col_1, hero_col_2 = st.columns([2, 1])

with hero_col_1:
    st.markdown(
        """
        <div class="section-card">
            <h4 style="margin-top: 0;">Project Objective</h4>
            <p class="small-note">
                The goal of this project is to recommend relevant hotels to users by combining behavioral patterns
                with hotel similarity information from the saved model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with hero_col_2:
    st.metric("Best Saved Model", best_model_row["Model"], f"HitRate@5 = {best_model_row['HitRate@5']:.4f}")

with st.expander("Technical Notes", expanded=False):
    st.write(
        "The app loads the recommender from `joblib files/hotel_recommender_simple.joblib` and reads hotel details from `dataset/travel_capstone/hotels.csv`."
    )

    st.write(
        "The saved file was created with an older scikit-learn version, so a version warning may appear during loading. This is a compatibility note, not a crash."
    )


overview_tab, popular_tab, similar_tab, user_tab, metrics_tab = st.tabs(
    [
        "Overview",
        "Popular Picks",
        "Similar Hotels",
        "Personalized Picks",
        "Model Evaluation",
    ]
)


with overview_tab:
    # The overview tab explains what was saved inside the recommender artifact.
    st.subheader("Project Overview")

    st.write(
        "This project artifact already stores popularity scores, hotel-to-hotel similarity, user-item interactions, and saved evaluation metrics."
    )

    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)

    metric_col_1.metric("Hotels", len(hotel_summary_df))

    metric_col_2.metric("Users", int(model_bundle["user_item_matrix"].shape[0]))

    metric_col_3.metric("Bookings", int(hotels_df.shape[0]))

    metric_col_4.metric(
        "Best HitRate@5",
        f"{evaluation_results_df['HitRate@5'].max():.4f}",
    )

    overview_col_1, overview_col_2 = st.columns([1.2, 1])

    with overview_col_1:
        st.markdown("#### Recommendation Logic")

        st.write(
            "This dashboard demonstrates three recommendation views. Popular Picks shows globally frequent hotels, Similar Hotels compares one hotel with others using saved similarity scores, and Personalized Picks estimates hotels a selected user may prefer based on interaction history."
        )

        st.markdown("#### Hotel Summary Table")

        st.dataframe(
            hotel_summary_df.sort_values(by="booking_count", ascending=False),
            width="stretch",
            hide_index=True,
        )

    with overview_col_2:
        st.markdown("#### Most Booked Destinations")

        st.bar_chart(
            top_places_df.head(8).set_index("place"),
            height=360,
        )

        st.caption(
            "This chart shows which destinations appear most often in the hotel booking data."
        )


with popular_tab:
    # Popular picks are the quickest way to show a result even without user context.
    st.subheader("Top Popular Hotels")

    st.write(
        "This section shows hotels that perform best overall based on the saved popularity table inside the model."
    )

    popular_count = st.slider(
        "How many popular hotels do you want to see?",
        min_value=3,
        max_value=len(popular_hotels_df),
        value=min(5, len(popular_hotels_df)),
    )

    st.dataframe(
        popular_hotels_df.head(popular_count),
        width="stretch",
        hide_index=True,
    )


with similar_tab:
    # Similar-hotel lookup is useful for showing the item-to-item side of the recommender.
    st.subheader("Find Hotels Similar To A Selected Hotel")

    similar_control_col, similar_result_col = st.columns([1, 1.6])

    with similar_control_col:
        selected_hotel = st.selectbox(
            "Choose a hotel",
            options=sorted(model_bundle["hotel_index"].index.tolist()),
        )

        similar_count = st.slider(
            "How many similar hotels do you want?",
            min_value=3,
            max_value=max(3, len(model_bundle["hotel_index"]) - 1),
            value=min(5, max(3, len(model_bundle["hotel_index"]) - 1)),
        )

        selected_hotel_details = hotel_summary_df[
            hotel_summary_df["name"] == selected_hotel
        ].iloc[0]

        st.write(f"Place: {selected_hotel_details['place']}")

        st.write(f"Average price: {selected_hotel_details['avg_price']}")

        st.write(f"Booking count: {int(selected_hotel_details['booking_count'])}")

    similar_hotels_df = get_similar_hotels(
        model_bundle=model_bundle,
        hotel_summary_df=hotel_summary_df,
        selected_hotel=selected_hotel,
        top_n=similar_count,
    )

    with similar_result_col:
        st.write(
            "The table below combines content similarity and collaborative similarity into one simple ranking score."
        )

        st.dataframe(similar_hotels_df, width="stretch", hide_index=True)


with user_tab:
    # Personalized recommendations depend on a user having at least some existing interactions.
    st.subheader("Recommend Hotels For A Selected User")

    st.write(
        "This section estimates which hotels a selected user may like next based on their existing interaction pattern."
    )

    user_options = sorted(user_item_matrix_df.index.tolist())

    default_user = eligible_users[0] if eligible_users else user_options[0]

    default_user_index = user_options.index(default_user)

    user_filter_col_1, user_filter_col_2 = st.columns([1.2, 1])

    with user_filter_col_1:
        selected_user = st.selectbox(
            "Choose a user code",
            options=user_options,
            index=default_user_index,
        )

    with user_filter_col_2:
        user_recommendation_count = st.slider(
            "How many recommendations do you want for this user?",
            min_value=3,
            max_value=len(model_bundle["user_item_matrix"].columns),
            value=min(5, len(model_bundle["user_item_matrix"].columns)),
        )

    st.write(
        f"This user has already seen {int(user_seen_counts[selected_user])} out of {user_item_matrix_df.shape[1]} hotels."
    )

    user_recommendations_df, user_history_df = get_user_recommendations(
        model_bundle=model_bundle,
        hotel_summary_df=hotel_summary_df,
        selected_user=selected_user,
        top_n=user_recommendation_count,
    )

    if user_recommendations_df.empty:
        st.info(
            "This user has already interacted with all available hotels, so there are no new hotels left to recommend."
        )

    recommendation_col, history_col = st.columns(2)

    with recommendation_col:
        st.write("Recommended hotels:")

        st.dataframe(user_recommendations_df, width="stretch", hide_index=True)

    with history_col:
        st.write("Hotels already seen by this user:")

        st.dataframe(user_history_df, width="stretch", hide_index=True)


with metrics_tab:
    # Saved evaluation metrics help compare the recommendation strategies used during training.
    st.subheader("Saved Model Evaluation Scores")

    st.write(
        "These scores were already saved inside the joblib file, so this tab helps you compare how the recommendation methods performed during evaluation."
    )

    st.metric(
        "Top Model Based On HitRate@5",
        best_model_row["Model"],
        f"{best_model_row['HitRate@5']:.4f}",
    )

    st.dataframe(evaluation_results_df, width="stretch", hide_index=True)
