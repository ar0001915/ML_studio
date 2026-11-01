import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error, confusion_matrix
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

st.set_page_config(page_title="ML Studio | Smart Data Analyzer", page_icon="🧠", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.7rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background: rgba(120,120,120,.08); padding: 14px; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

st.title("🧠 ML Studio")
st.caption("Intelligent Data Analysis & Automated Machine Learning Platform")
st.write("Upload a real-world CSV dataset, explore it, clean it, train machine-learning models, and compare results.")

with st.sidebar:
    st.header("⚙️ Settings")
    uploaded = st.file_uploader("Upload dataset (CSV)", type=["csv"])
    st.divider()
    st.info("Tip: choose a dataset with a column you want to predict (Target).")

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.session_state["sample_df"] = df
        st.session_state.pop("sample_choice", None)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()
else:
    st.markdown("### Get started")
    st.write("Upload a CSV file from the sidebar, or try a sample dataset below.")
    sample_choice = st.selectbox("Sample dataset", ["None", "Student Performance (synthetic)", "House Prices (synthetic)"])
    if sample_choice == "Student Performance (synthetic)":
        rng = np.random.default_rng(7)
        n = 250
        hours = rng.uniform(0, 10, n).round(1)
        attendance = rng.uniform(50, 100, n).round(1)
        sleep = rng.uniform(4, 10, n).round(1)
        score = np.clip(25 + hours * 5 + attendance * .25 + sleep * 1.5 + rng.normal(0, 8, n), 0, 100).round(1)
        df = pd.DataFrame({
            "Study_Hours": hours, "Attendance": attendance, "Sleep_Hours": sleep,
            "Final_Score": score, "Pass": np.where(score >= 40, "Yes", "No")
        })
        st.session_state["sample_df"] = df
    elif sample_choice == "House Prices (synthetic)":
        rng = np.random.default_rng(11)
        n = 250
        area = rng.uniform(500, 3500, n).round(0)
        bedrooms = rng.integers(1, 7, n)
        age = rng.integers(0, 50, n)
        price = (area * 4500 + bedrooms * 180000 - age * 12000 + rng.normal(0, 250000, n)).clip(200000).round(0)
        df = pd.DataFrame({"Area_sqft": area, "Bedrooms": bedrooms, "Age_years": age, "Price": price})
        st.session_state["sample_df"] = df
    else:
        st.session_state.pop("sample_df", None)
        st.stop()

if "sample_df" not in st.session_state:
    st.stop()

df = st.session_state["sample_df"].copy()
if df.empty or len(df.columns) < 2:
    st.error("Dataset must contain at least 2 columns and 1 row.")
    st.stop()

tabs = st.tabs(["📁 Dataset Overview", "🧹 Data Cleaning", "📊 EDA", "🤖 Train Models", "🔮 Predict"])

with tabs[0]:
    st.subheader("Dataset Preview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    st.dataframe(df.head(100), use_container_width=True)
    st.subheader("Column information")
    info = pd.DataFrame({
        "Column": df.columns, "Data type": df.dtypes.astype(str).values,
        "Missing values": df.isna().sum().values,
        "Unique values": df.nunique(dropna=True).values
    })
    st.dataframe(info, use_container_width=True, hide_index=True)
    st.subheader("Statistical summary")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

with tabs[1]:
    st.subheader("Clean your dataset")
    st.write("Choose how to handle missing values and duplicate rows. Preview changes before downloading.")
    st.metric("Duplicate rows detected", int(df.duplicated().sum()))
    remove_dups = st.checkbox("Remove duplicate rows", value=True)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    missing_strategy_num = st.selectbox(
        "Numeric missing values", ["Median", "Mean", "Drop rows with missing values", "Fill with 0"]
    )
    missing_strategy_cat = st.selectbox(
        "Categorical missing values", ["Most frequent", "Fill with 'Unknown'", "Drop rows with missing values"]
    )
    cleaned = df.copy()
    if remove_dups:
        cleaned = cleaned.drop_duplicates()

    if (missing_strategy_num == "Drop rows with missing values"
            or missing_strategy_cat == "Drop rows with missing values"):
        cleaned = cleaned.dropna()
    else:
        for col in numeric_cols:
            if cleaned[col].isna().any():
                if missing_strategy_num == "Median":
                    cleaned[col] = cleaned[col].fillna(cleaned[col].median())
                elif missing_strategy_num == "Mean":
                    cleaned[col] = cleaned[col].fillna(cleaned[col].mean())
                elif missing_strategy_num == "Fill with 0":
                    cleaned[col] = cleaned[col].fillna(0)
        for col in categorical_cols:
            if cleaned[col].isna().any():
                if missing_strategy_cat == "Most frequent":
                    mode = cleaned[col].mode(dropna=True)
                    cleaned[col] = cleaned[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")
                elif missing_strategy_cat == "Fill with 'Unknown'":
                    cleaned[col] = cleaned[col].fillna("Unknown")

    st.success(f"Preview: {len(df)} → {len(cleaned)} rows")
    st.dataframe(cleaned.head(100), use_container_width=True)
    st.download_button(
        "⬇️ Download cleaned CSV", cleaned.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_dataset.csv", mime="text/csv"
    )
    st.session_state["cleaned_df"] = cleaned

with tabs[2]:
    st.subheader("Exploratory Data Analysis")
    working = st.session_state.get("cleaned_df", df)
    num_cols = working.select_dtypes(include=np.number).columns.tolist()
    cat_cols = working.select_dtypes(exclude=np.number).columns.tolist()
    if num_cols:
        col = st.selectbox("Numeric column distribution", num_cols)
        st.plotly_chart(px.histogram(working, x=col, marginal="box", title=f"Distribution of {col}"), use_container_width=True)
        if len(num_cols) >= 2:
            xcol = st.selectbox("X-axis", num_cols, index=0)
            ycol = st.selectbox("Y-axis", num_cols, index=1)
            st.plotly_chart(px.scatter(working, x=xcol, y=ycol, trendline="ols", title=f"{ycol} vs {xcol}"), use_container_width=True)
            corr = working[num_cols].corr(numeric_only=True)
            st.plotly_chart(px.imshow(corr, text_auto=True, aspect="auto", title="Correlation heatmap"), use_container_width=True)
    if cat_cols:
        cat = st.selectbox("Categorical column frequency", cat_cols)
        counts = working[cat].astype(str).value_counts().head(20).reset_index()
        counts.columns = [cat, "Count"]
        st.plotly_chart(px.bar(counts, x=cat, y="Count", title=f"Top categories in {cat}"), use_container_width=True)

with tabs[3]:
    st.subheader("Train and compare models")
    work = st.session_state.get("cleaned_df", df).copy()
    target = st.selectbox("Select target column (what do you want to predict?)", work.columns.tolist())
    task_choice = st.radio("Problem type", ["Auto-detect", "Regression", "Classification"], horizontal=True)
    y = work[target]
    X = work.drop(columns=[target])

    if task_choice == "Auto-detect":
        if y.dtype == "object" or y.dtype.name == "category" or y.nunique() <= 10:
            task = "classification"
        else:
            task = "regression"
    else:
        task = task_choice.lower()

    st.info(f"Detected task: **{task.title()}**")
    if y.isna().any():
        st.warning("Target has missing values. Rows with missing target values will be excluded during training.")
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]

    if task == "classification" and y.nunique() < 2:
        st.error("Classification needs at least 2 target classes.")
        st.stop()

    test_size = st.slider("Test set size (%)", 10, 40, 20, 5) / 100
    random_state = st.number_input("Random seed", min_value=0, max_value=99999, value=42)

    if task == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=2000),
            "Naive Bayes (Gaussian)": GaussianNB(),
            "Decision Tree": DecisionTreeClassifier(random_state=int(random_state)),
            "Random Forest": RandomForestClassifier(n_estimators=150, random_state=int(random_state)),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
            "Support Vector Machine": SVC()
        }
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor(random_state=int(random_state)),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=150, random_state=int(random_state)),
            "K-Nearest Neighbors Regressor": KNeighborsRegressor(n_neighbors=5),
            "Support Vector Regressor": SVR(),
            "Gradient Boosting": GradientBoostingRegressor(random_state=int(random_state))
        }

    selected_models = st.multiselect("Models to train", list(models.keys()), default=list(models.keys()))
    if st.button("🚀 Train selected models", type="primary"):
        if X.shape[1] == 0:
            st.error("Please keep at least one feature column besides the target.")
        elif len(X) < 5:
            st.error("Dataset is too small. Please use at least 5 valid rows.")
        elif not selected_models:
            st.error("Select at least one model.")
        else:
            num_features = X.select_dtypes(include=np.number).columns.tolist()
            cat_features = X.select_dtypes(exclude=np.number).columns.tolist()
            numeric_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            categorical_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ])
            preprocessor = ColumnTransformer([
                ("num", numeric_pipe, num_features),
                ("cat", categorical_pipe, cat_features)
            ])
            stratify = y if task == "classification" and y.value_counts().min() >= 2 else None

            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=int(random_state), stratify=stratify
                )
            except Exception as e:
                st.error(f"Train/test split failed: {e}")
                st.stop()

            results, fitted = [], {}
            progress = st.progress(0)
            for i, name in enumerate(selected_models):
                pipe = Pipeline([("preprocessing", preprocessor), ("model", models[name])])
                try:
                    pipe.fit(X_train, y_train)
                    pred = pipe.predict(X_test)
                    if task == "classification":
                        row = {
                            "Model": name,
                            "Accuracy": accuracy_score(y_test, pred),
                            "Precision (weighted)": precision_score(y_test, pred, average="weighted", zero_division=0),
                            "Recall (weighted)": recall_score(y_test, pred, average="weighted", zero_division=0),
                            "F1 (weighted)": f1_score(y_test, pred, average="weighted", zero_division=0)
                        }
                    else:
                        row = {
                            "Model": name, "R²": r2_score(y_test, pred),
                            "MAE": mean_absolute_error(y_test, pred),
                            "RMSE": float(np.sqrt(mean_squared_error(y_test, pred)))
                        }
                    results.append(row)
                    fitted[name] = {
                        "pipeline": pipe, "X_test": X_test, "y_test": y_test, "predictions": pred
                    }
                except Exception as e:
                    st.warning(f"{name} could not be trained: {e}")
                progress.progress((i + 1) / len(selected_models))

            if results:
                result_df = pd.DataFrame(results)
                sort_col = "F1 (weighted)" if task == "classification" else "R²"
                result_df = result_df.sort_values(sort_col, ascending=False)
                st.session_state["model_results"] = result_df
                st.session_state["fitted_models"] = fitted
                st.session_state["task"] = task
                st.session_state["target"] = target
                st.session_state["feature_columns"] = X.columns.tolist()
                st.session_state["training_X"] = X
                st.success("Training completed!")

    if "model_results" in st.session_state:
        st.subheader("🏆 Model leaderboard")
        st.dataframe(st.session_state["model_results"], use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download model comparison",
            st.session_state["model_results"].to_csv(index=False).encode("utf-8"),
            "model_comparison.csv", "text/csv"
        )
        fitted = st.session_state.get("fitted_models", {})
        if fitted:
            chosen = st.selectbox("Inspect model", list(fitted.keys()))
            item = fitted[chosen]
            if st.session_state.get("task") == "regression":
                plot_df = pd.DataFrame({"Actual": item["y_test"], "Predicted": item["predictions"]})
                st.plotly_chart(px.scatter(plot_df, x="Actual", y="Predicted", title="Actual vs Predicted"), use_container_width=True)
            else:
                labels = np.unique(np.concatenate([item["y_test"].astype(str), item["predictions"].astype(str)]))
                cm = confusion_matrix(item["y_test"].astype(str), item["predictions"].astype(str), labels=labels)
                st.plotly_chart(px.imshow(
                    cm, x=labels, y=labels, text_auto=True,
                    labels={"x": "Predicted", "y": "Actual", "color": "Count"},
                    title="Confusion matrix"
                ), use_container_width=True)

with tabs[4]:
    st.subheader("Make a prediction with a trained model")
    fitted = st.session_state.get("fitted_models", {})
    if not fitted:
        st.info("Go to **Train Models** tab and train at least one model first.")
    else:
        model_name = st.selectbox("Choose trained model", list(fitted.keys()))
        pipeline = fitted[model_name]["pipeline"]
        features = st.session_state["feature_columns"]
        source = st.session_state.get("training_X", df.drop(columns=[st.session_state.get("target", df.columns[-1])]))
        st.write("Enter values for one new record:")
        values = {}
        cols = st.columns(2)
        for idx, feature in enumerate(features):
            series = source[feature]
            with cols[idx % 2]:
                if pd.api.types.is_numeric_dtype(series):
                    median = float(series.median()) if series.notna().any() else 0.0
                    values[feature] = st.number_input(feature, value=median, key=f"predict_{feature}")
                else:
                    choices = series.dropna().astype(str).unique().tolist()
                    values[feature] = st.selectbox(feature, choices if choices else ["Unknown"], key=f"predict_{feature}")
        if st.button("🔮 Predict", type="primary"):
            new_row = pd.DataFrame([values], columns=features)
            prediction = pipeline.predict(new_row)[0]
            st.success(f"Predicted **{st.session_state.get('target', 'target')}**: {prediction}")

st.divider()
st.caption("ML Studio • Educational project prototype • Built with Streamlit, Pandas, Plotly & Scikit-learn")
