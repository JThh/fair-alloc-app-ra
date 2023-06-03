import streamlit as st

st.set_page_config(
    page_title="Welcome",
    page_icon="👋",
)

st.write("# Welcome to Fair Allocation App! 👋")

st.sidebar.success("Select an implementation above.")

st.write("This app allows you to perform fair allocation of indivisible items.")
st.markdown(
    """
    We have provided a range of algorithmic implementations to allocating indivisible goods or chores, or a mix of them.
"""
)
st.header("App Layout")
st.write("The app consists of the following components:")
st.markdown("- **Sidebar**: The sidebar on the left provides navigation options.")
st.markdown("- **Main Section**: The main section displays the content of the selected page.")
st.markdown("- **User Guide**: The default page provides a user guide to help you understand how to use the app.")
st.markdown("- **Allocation App**: The second page is dedicated to the allocation app, where you can input preferences and weights to calculate item allocation.")

