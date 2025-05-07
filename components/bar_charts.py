import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def plot_avg_by_category(df, target, category):
    avg_values = df.groupby(category)[target].mean().reset_index()
    fig, ax = plt.subplots()
    sns.barplot(x=category, y=target, data=avg_values, ax=ax)
    ax.set_title(f"Average {target} by {category}")
    ax.set_xlabel(category)
    ax.set_ylabel(f"Average {target}")
    st.pyplot(fig)

def plot_correlation(df, target):
    numeric_df = df.select_dtypes(include='number')
    if target not in numeric_df.columns:
        st.info("Target variable must be numeric for correlation plot.")
        return
    correlations = numeric_df.corr()[target].abs().sort_values(ascending=False)
    correlations = correlations.drop(target)
    if correlations.empty:
        st.info("No numeric features to correlate with the target.")
        return
    fig, ax = plt.subplots()
    sns.barplot(x=correlations.index, y=correlations.values, ax=ax)  # Vertical bars
    ax.set_title(f"Correlation with {target}")
    ax.set_xlabel("Features")
    ax.set_ylabel("Correlation Strength")
    st.pyplot(fig)