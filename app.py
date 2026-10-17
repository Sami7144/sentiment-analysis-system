
import streamlit as st
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------------
# 1. Page configuration
# -------------------------------
st.set_page_config(
    page_title="Sentiment Analysis System",
    page_icon="💬",
    layout="wide"
)

# -------------------------------
# 2. Sample training dataset
# -------------------------------
reviews = [
    "This product is excellent",
    "I love this product",
    "Amazing quality and great service",
    "Very happy with my purchase",
    "The product works perfectly",
    "Excellent experience",
    "I am very satisfied",
    "The quality is wonderful",
    "Fast delivery and great product",
    "Highly recommended",
    "This is fantastic",
    "Really good value for money",
    "The product is useful",
    "Beautiful design and good quality",
    "I would buy this again",
    "Customer service was excellent",
    "Absolutely wonderful product",
    "The experience was great",
    "Good quality and affordable",
    "I am pleased with this purchase",
    "This product is terrible",
    "I hate this product",
    "Very poor quality",
    "I am disappointed",
    "The product does not work",
    "Terrible experience",
    "I am very unhappy",
    "The quality is awful",
    "Delivery was late and bad",
    "Not recommended",
    "This is horrible",
    "A complete waste of money",
    "The product is useless",
    "Poor design and bad quality",
    "I would not buy this again",
    "Customer service was terrible",
    "Absolutely disappointing product",
    "The experience was bad",
    "Low quality and expensive",
    "I regret this purchase"
]

sentiments = (
    ["Positive"] * 20
    + ["Negative"] * 20
)

# -------------------------------
# 3. Split the dataset
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    reviews,
    sentiments,
    test_size=0.25,
    random_state=42,
    stratify=sentiments
)

# -------------------------------
# 4. Create the ML pipeline
# -------------------------------
@st.cache_resource
def train_model():
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2)
            )
        ),
        (
            "classifier",
            LogisticRegression(max_iter=1000)
        )
    ])

    model.fit(X_train, y_train)
    return model

model = train_model()

# Evaluate on held-out sample reviews
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# -------------------------------
# 5. Application heading
# -------------------------------
st.title("💬 Sentiment Analysis System")
st.write(
    "Analyze customer reviews using Machine Learning "
    "and identify positive or negative sentiment."
)

st.divider()

# -------------------------------
# 6. Dashboard metrics
# -------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Training Reviews", len(X_train))
col2.metric("Test Reviews", len(X_test))
col3.metric("Test Accuracy", f"{accuracy * 100:.1f}%")

st.caption(
    "Note: This is a small demonstration dataset. "
    "Test accuracy can vary and does not represent "
    "real-world performance."
)

st.divider()

# -------------------------------
# 7. Review input
# -------------------------------
st.subheader("Analyze a Customer Review")

review = st.text_area(
    "Enter your review below:",
    placeholder=(
        "Example: The product quality is excellent "
        "and I am very happy!"
    ),
    height=140
)

# Example reviews
st.write("**Try an example:**")

example_col1, example_col2 = st.columns(2)

with example_col1:
    if st.button("😊 Positive Example", use_container_width=True):
        st.session_state["review"] = (
            "The product is excellent and I love it"
        )

with example_col2:
    if st.button("😞 Negative Example", use_container_width=True):
        st.session_state["review"] = (
            "The product is terrible and very disappointing"
        )

# Keep the selected example in the input box
if "review" in st.session_state:
    review = st.session_state["review"]

# -------------------------------
# 8. Predict sentiment
# -------------------------------
if st.button("Analyze Sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter a review first.")

    elif not any(char.isalpha() for char in review):
        st.warning("Please enter a review containing words.")

    else:
        prediction = model.predict([review])[0]
        probabilities = model.predict_proba([review])[0]

        classes = model.named_steps[
            "classifier"
        ].classes_

        confidence = probabilities[
            list(classes).index(prediction)
        ] * 100

        st.divider()
        st.subheader("Analysis Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            if prediction == "Positive":
                st.success("😊 Positive Sentiment")
            else:
                st.error("😞 Negative Sentiment")

        with result_col2:
            st.metric("Model Confidence", f"{confidence:.1f}%")

        st.progress(float(confidence / 100))

        st.info(
            "Confidence is the model's estimated probability, "
            "not a guarantee that the prediction is correct."
        )

# -------------------------------
# 9. Model information
# -------------------------------
with st.expander("How does this project work?"):
    st.write("""
    1. Training reviews are labelled Positive or Negative.
    2. TF-IDF converts text into numerical features.
    3. Logistic Regression learns patterns from the training data.
    4. The trained model predicts sentiment for a new review.
    5. The held-out test set provides a basic accuracy estimate.
    """)

st.divider()

st.caption("Built with Python, Scikit-learn and Streamlit")
