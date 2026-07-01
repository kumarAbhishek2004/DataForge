import streamlit as st

def apply_premium_sidebar():
    st.markdown("""
    <style>
    /* Dark sidebar styling to match React */
    [data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }
    
    /* React-style Header: "Activity Icon + DataForge" */
    [data-testid="stSidebar"]::before {
        content: "DataForge";
        display: block;
        padding: 1.5rem 1.5rem 0.5rem 1.5rem;
        font-size: 1.2rem;
        font-weight: 600;
        color: #F8FAFC;
        border-bottom: 1px solid transparent;
        margin-bottom: 0.5rem;
    }
    
    /* "Main Menu" Label Injection */
    [data-testid="stSidebarNav"]::before {
        content: "MAIN MENU";
        display: block;
        padding: 0 1.5rem 0.5rem 1.5rem;
        font-size: 0.75rem;
        font-weight: 700;
        color: #64748B;
        letter-spacing: 0.05em;
    }
    
    /* Remove padding from the nav container to reduce gap */
    [data-testid="stSidebarNav"] {
        padding-top: 0 !important;
    }
    
    /* Force remove Streamlit's default flex gaps between list items */
    [data-testid="stSidebarNav"] ul {
        gap: 0.2rem !important;
    }
    
    [data-testid="stSidebarNav"] li {
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Sidebar Links - Reduced Gaps & React styling */
    [data-testid="stSidebarNav"] a {
        padding: 0.4rem 1.5rem !important;
        margin: 0 1rem !important;
        border-radius: 6px;
        transition: all 0.2s ease-in-out;
        color: #94A3B8 !important;
        font-size: 0.95rem;
        font-weight: 500;
        text-decoration: none !important;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Hover state */
    [data-testid="stSidebarNav"] a:hover {
        background-color: #1E293B;
        color: #F8FAFC !important;
    }
    
    /* Active Page state */
    [data-testid="stSidebarNav"] a[aria-current="page"] {
        background-color: #1E293B;
        color: #3B82F6 !important;
        border-left: 3px solid #3B82F6;
        border-radius: 0 6px 6px 0;
    }
    
    [data-testid="stSidebarNav"] a[aria-current="page"] * {
        color: #3B82F6 !important;
        font-weight: 600;
    }
    
    /* Adjust default 'Pages' header */
    [data-testid="stSidebarNav"] > div:first-child {
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
