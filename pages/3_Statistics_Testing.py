import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px

st.set_page_config(page_title="Statistics & Hypothesis Testing", page_icon="", layout="wide")

from theme import apply_premium_sidebar
apply_premium_sidebar()

st.title("Statistical Analysis & Hypothesis Testing")

if 'raw_data' not in st.session_state:
    try:
        df = pd.read_csv("data/raw_dataset.csv")
        st.session_state['raw_data'] = df
    except:
        st.warning("No dataset found. Please upload a dataset in the Upload module first.")
        st.stop()
        
df = st.session_state['raw_data']
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

tab1, tab2 = st.tabs(["Descriptive & Distribution Stats", "Hypothesis Testing Engine"])

with tab1:
    st.markdown("### Descriptive Statistics")
    if num_cols:
        desc_df = df[num_cols].describe().T
        desc_df['variance'] = df[num_cols].var()
        desc_df['iqr'] = desc_df['75%'] - desc_df['25%']
        desc_df['skewness'] = df[num_cols].skew()
        desc_df['kurtosis'] = df[num_cols].kurtosis()
        
        st.dataframe(desc_df[['mean', '50%', 'std', 'variance', 'min', 'max', 'iqr', 'skewness', 'kurtosis']], use_container_width=True)
        
        st.markdown("---")
        st.markdown("### Distribution Testing (Normality)")
        st.info("Algorithms like Linear Regression assume normally distributed features. We test for this using the Shapiro-Wilk test.")
        
        col_to_test = st.selectbox("Select Feature to Test Normality", num_cols)
        
        if col_to_test:
            # Drop NAs for stats tests
            data = df[col_to_test].dropna()
            
            # Shapiro-Wilk test (limit to 5000 samples for performance/accuracy)
            stat, p_val = stats.shapiro(data.sample(min(5000, len(data)), random_state=42))
            
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Shapiro-Wilk p-value", f"{p_val:.5f}")
                if p_val > 0.05:
                    st.success(f"**Fail to reject H₀**: The feature `{col_to_test}` looks Normally Distributed (Gaussian).")
                else:
                    st.error(f"**Reject H₀**: The feature `{col_to_test}` is NOT Normally Distributed (Skewed). Non-parametric models (Tree-based) may perform better.")
            
            with c2:
                fig = px.histogram(df, x=col_to_test, marginal="box", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### Automated Hypothesis Testing Engine")
    st.write("Select two variables. The engine will automatically determine and execute the correct statistical test (e.g. ANOVA, Chi-Square, t-Test).")
    
    col1, col2 = st.columns(2)
    var1 = col1.selectbox("Variable 1", df.columns)
    var2 = col2.selectbox("Variable 2", df.columns, index=1 if len(df.columns)>1 else 0)
    
    if var1 and var2 and var1 != var2:
        v1_type = "Numeric" if var1 in num_cols else "Categorical"
        v2_type = "Numeric" if var2 in num_cols else "Categorical"
        
        st.markdown(f"Detected Relationship: **{v1_type} vs {v2_type}**")
        
        if v1_type == "Numeric" and v2_type == "Numeric":
            st.info("**Recommended Test:** Pearson Correlation (if normal) or Spearman Rank Correlation (if skewed)")
            
            test_type = st.radio("Select Test", ["Pearson", "Spearman"])
            
            data1 = df[var1].dropna()
            data2 = df[var2].dropna()
            
            # Align indices
            common_idx = data1.index.intersection(data2.index)
            d1, d2 = data1[common_idx], data2[common_idx]
            
            if test_type == "Pearson":
                stat, p_val = stats.pearsonr(d1, d2)
            else:
                stat, p_val = stats.spearmanr(d1, d2)
                
            st.markdown(f"**H₀:** There is no monotonic relationship between {var1} and {var2}.")
            st.metric("p-value", f"{p_val:.5e}")
            
            if p_val < 0.05:
                st.success(f"**Decision:** Reject H₀. Business Insight: There is a statistically significant correlation between {var1} and {var2} (Correlation = {stat:.3f}).")
            else:
                st.warning(f"**Decision:** Fail to Reject H₀. Business Insight: There is no significant correlation.")
                
        elif v1_type == "Categorical" and v2_type == "Categorical":
            st.info("**Recommended Test:** Chi-Square Test of Independence")
            
            contingency = pd.crosstab(df[var1], df[var2])
            stat, p_val, dof, expected = stats.chi2_contingency(contingency)
            
            st.markdown(f"**H₀:** {var1} and {var2} are completely independent (no relationship).")
            st.metric("p-value", f"{p_val:.5e}")
            
            if p_val < 0.05:
                st.success(f"**Decision:** Reject H₀. Business Insight: {var1} and {var2} are significantly dependent. Targeting one will likely affect the other.")
            else:
                st.warning(f"**Decision:** Fail to Reject H₀. Business Insight: There is no statistical relationship between {var1} and {var2}.")
                
        else:
            # Mixed (Num vs Cat)
            st.info("**Recommended Test:** One-Way ANOVA (if >2 categories) or Independent t-Test (if 2 categories)")
            
            cat_var = var1 if v1_type == "Categorical" else var2
            num_var = var1 if v1_type == "Numeric" else var2
            
            groups = [group[num_var].dropna().values for name, group in df.groupby(cat_var)]
            
            if len(df[cat_var].dropna().unique()) == 2:
                stat, p_val = stats.ttest_ind(groups[0], groups[1], equal_var=False) # Welch's t-test
                st.markdown(f"**H₀:** The average {num_var} is exactly the same across both {cat_var} groups.")
            else:
                stat, p_val = stats.f_oneway(*groups)
                st.markdown(f"**H₀:** The average {num_var} is exactly the same across all {cat_var} groups.")
                
            st.metric("p-value", f"{p_val:.5e}")
            
            if p_val < 0.05:
                st.success(f"**Decision:** Reject H₀. Business Insight: The {cat_var} significantly impacts the {num_var}.")
            else:
                st.warning(f"**Decision:** Fail to Reject H₀. Business Insight: The {cat_var} does not significantly impact the {num_var}.")
    else:
        st.warning("Please select two distinct variables.")
