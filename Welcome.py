import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Welcome to the Fair Allocation App!",
    page_icon="🌟",
    layout="wide",
)

# Disable the scrollbar for the Streamlit sidebar
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        overflow-y: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Page 1: Introduction
st.title(" Welcome to the Fair Allocation App! 👋")

st.write("This app is designed to help you achieve fair allocation of indivisible items!")
st.markdown(
    """
    We offer a variety of algorithmic implementations to ensure a fair distribution of these items, taking into account the concept of weighted envy-freeness (WEF).
    
    Weighted envy-freeness is a fairness criterion that aims to minimize the envy experienced by participants in an allocation process. It assigns a weight to each participant's preferences, allowing for personalized fairness. Our app will help you find an allocation that balances these preferences as closely as possible, maximizing overall satisfaction.
"""
)

# Page 2: App Layout and Components
st.header("App Layout")
st.write("Let's take a quick tour of the app's components:")

# Load and display the image
tab1, tab2, tab3 = st.tabs(["App 1","App 2","App 3"])
with tab1:
    image = "./resource/layout1.png"
    st.image(image, caption="App Layout", use_column_width=True)
    
with tab2:
    image = "./resource/layout2.png"
    st.image(image, caption="App Layout", use_column_width=True)
    
with tab3:
    image = "./resource/layout3.png"
    st.image(image, caption="App Layout", use_column_width=True)

st.markdown("- **Sidebar**: On the left, you'll find a handy sidebar for easy navigation.")
st.markdown("- **Main Section**: The main section in the center will display the content of the selected page.")
st.markdown("- **User Guide**: The default page provides a user guide to help you understand how to use the app.")

# Page 3: Allocation App Introduction
st.header("Allocation App: Achieving Weighted Envy-Freeness")
st.write(
    "The allocation app is designed to help you achieve weighted envy-freeness by fairly allocating indivisible items. Here's how it works:"
)
st.markdown(
    """
    1. Start by providing the number of participants and the available items.
    2. For each participant, specify their preferences for the items. You can assign weights to these preferences to reflect their importance.
    3. Once you've entered all the preferences, click on the 'Run Allocation Algorithm' button.
    4. Our algorithms will process the inputs and generate an allocation that minimizes envy while considering the weights assigned to preferences.
    5. The app will display the resulting allocation along with any relevant statistics or insights.
"""
)
st.write(
    "With our fair allocation app, you can explore various scenarios and experiment with different preferences and weightings to find an allocation that best satisfies the weighted envy-freeness objective."
)


hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.markdown(
    """
    <div class="footer" style="padding-top: 200px; margin-top: auto; text-align: left; font-size: 10px; color: #777777;">
    <p>Developed by <a href="https://www.linkedin.com/in/jiatong-han-06636419b/" target="_blank">Jiatong Han</a>, 
    kindly advised by Prof. <a href="https://www.comp.nus.edu.sg/~warut/" target="_blank">Warut Suksompong</a></p>
    <p>&copy; 2023. All rights reserved.</p>
    </div>
    """,
    unsafe_allow_html=True
)