import streamlit as st
import pandas as pd
from components.select_target import get_numerical_columns
from components.bar_charts import plot_avg_by_category, plot_correlation
from components.train import train_model
from components.predict import predict

# Custom CSS for gray headers, centering, and square radio buttons
st.markdown("""
    <style>
    .gray-header {
        background-color: #e0e0e0;
        padding: 10px;
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
        border-radius: 4px;
    }
    .centered {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .stRadio > div {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    .stRadio > div > label {
        display: flex;
        align-items: center;
        justify-content: center;
        width: auto;
        height: auto;
        padding: 5px 10px;
        border: 1px solid #ccc;
        border-radius: 4px;
        cursor: pointer;
    }
    .stRadio > div > label:hover {
        background-color: #f0f0f0;
    }
    .stRadio > div > label input[type="radio"] {
        display: none;
    }
    .stRadio > div > label input[type="radio"]:checked + span {
        background-color: #007bff;
        color: white;
        border-color: #007bff;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Group 16 Regression App")

# --- Upload File ---
st.markdown('<div class="gray-header">Upload File</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["csv"])

if uploaded_file is not None:
    # Smart index handling
    sample = pd.read_csv(uploaded_file, nrows=0)
    uploaded_file.seek(0)
    first_col = sample.columns[0]
    if (first_col.startswith("Unnamed") or first_col == ""):
        df = pd.read_csv(uploaded_file, index_col=0)
    else:
        df = pd.read_csv(uploaded_file)
    st.write("Preview of dataset(First 10 rows):", df.head(10)) #Preview will only show first 10 rows.

    # --- Select Target ---
    st.markdown('<div class="gray-header">Select Target:</div>', unsafe_allow_html=True)
    numerical_columns = get_numerical_columns(df)
    if not numerical_columns:
        st.error("No numerical columns found in the dataset.")
    else:
        target_row = st.columns([1, 3])
        with target_row[0]:
            st.markdown("Select target:")
        with target_row[1]:
            target = st.selectbox(
                label="Select target variable",
                options=numerical_columns,
                key="target_select",
                label_visibility="collapsed"
            )

        # --- Categorical Variable Selection (Radio Buttons) ---
        # Only include categorical columns with ≤ 15 unique values
        categorical_columns = [
            col for col in df.select_dtypes(include=['object']).columns
            if df[col].nunique() <= 15
        ]
        if categorical_columns:
            st.markdown('<div class="gray-header" style="margin-top:20px;">Select Categorical Variable</div>', unsafe_allow_html=True)
            selected_category = st.radio(
                label="Select a category:",
                options=categorical_columns,
                key="cat_radio",
                horizontal=True  # Makes the radio buttons horizontal
            )
            st.markdown(f"<div class='centered'><b>Selected Category: {selected_category}</b></div>", unsafe_allow_html=True)
        else:
            selected_category = None
            st.info("No suitable categorical columns (with ≤ 15 unique values) available for selection.")

        # --- Bar Charts Side by Side ---
        st.markdown('<div class="gray-header" style="margin-top:20px;">Data Analysis</div>', unsafe_allow_html=True)
        chart_cols = st.columns(2)

        with chart_cols[0]:
            st.markdown(f"<div class='centered'><b>Average {target} by {selected_category if selected_category else 'Category'}</b></div>", unsafe_allow_html=True)
            if selected_category:
                plot_avg_by_category(df, target, selected_category)
            else:
                st.info("No categorical columns available for bar chart.")

        with chart_cols[1]:
            st.markdown(f"<div class='centered'><b>Correlation Strength of Numerical Variables with {target}</b></div>", unsafe_allow_html=True)
            plot_correlation(df, target)

        # --- Feature Selection ---
        st.markdown('<div class="gray-header" style="margin-top:20px;">Feature Selection & Training</div>', unsafe_allow_html=True)
        features = st.multiselect(
            label="Select features for training",
            options=df.columns.tolist(),
            default=numerical_columns,
            key="feature_multiselect",
            label_visibility="collapsed"
        )

        # --- Train Button ---
        train_btn_col = st.columns([3, 1, 3])
        with train_btn_col[1]:
            train_clicked = st.button("Train")

        # --- Training Result and Prediction ---
        if train_clicked:
            if not features:
                st.error("Please select at least one feature for training.")
            else:
                model, r2 = train_model(df, target, features)
                st.session_state["model"] = model  # Persist the model
                st.session_state["features"] = features  # Persist the selected features
                st.markdown(f"<div class='centered'>The R² score is: <b>{r2:.2f}</b></div>", unsafe_allow_html=True)

        # Check if the model is already trained
        if "model" in st.session_state and "features" in st.session_state:
            # --- Prediction ---
            input_placeholder = ", ".join([f"{f}" for f in st.session_state["features"]])
            pred_row = st.columns([2, 4, 2])  # Adjusted column widths for better layout
            with pred_row[1]:
                input_text = st.text_input(
                    label="Enter feature values (comma-separated)",
                    value="",
                    placeholder=input_placeholder,
                    key="predict_input",
                    label_visibility="collapsed"
                )
            with pred_row[2]:
                predict_clicked = st.button("Predict")  # Button now fits in one line

            if predict_clicked:
                try:
                    input_features = [float(x.strip()) if x.strip().replace('.', '', 1).isdigit() else x.strip() for x in input_text.split(",")]
                    if len(input_features) != len(st.session_state["features"]):
                        st.error(f"Expected {len(st.session_state['features'])} values, got {len(input_features)}.")
                    else:
                        # Build a DataFrame for prediction with correct column names
                        input_df = pd.DataFrame([input_features], columns=st.session_state["features"])
                        prediction = predict(st.session_state["model"], input_df)
                        st.markdown(f"<div class='centered'>Predicted {target} is: <b>{prediction:.2f}</b></div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Invalid input: {e}")
else:
    st.info("Please upload a CSV file to begin.")