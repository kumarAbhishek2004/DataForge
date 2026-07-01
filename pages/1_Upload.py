import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Upload Dataset", page_icon="", layout="wide")

from theme import apply_premium_sidebar
apply_premium_sidebar()


st.title("Upload Dataset")
st.markdown("Import your business data to begin analysis.")

uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.success("Dataset loaded successfully!")
        
        # Save to session state
        st.session_state['raw_data'] = df
        
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        
        st.subheader("Dataset Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", df.isna().sum().sum())
        
        st.markdown("---")
        st.subheader("Target Variable Selection")
        target_col = st.selectbox("Select the column you want to predict:", df.columns)
        
        if st.button("Confirm Target & Proceed"):
            st.session_state['target_column'] = target_col
            # Save raw data to a standard location for backend access
            os.makedirs("data", exist_ok=True)
            df.to_csv("data/raw_dataset.csv", index=False)
            
            with open("data/target.txt", "w") as f:
                f.write(target_col)
                
            st.success(f"Target variable '{target_col}' confirmed! You can now proceed to the EDA or AutoML modules.")
            
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
else:
    st.info("Awaiting file upload...")
