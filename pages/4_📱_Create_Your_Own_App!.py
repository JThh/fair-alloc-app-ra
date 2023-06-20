# Required Libraries
import re
import webbrowser
from xml.etree.ElementTree import TreeBuilder

# Streamlit API
import streamlit as st


def main():
    # Set page title
    st.set_page_config(page_title="Code Generator", page_icon="📱", layout="wide")
    
    # Custom CSS styles
    st.markdown(
        """
        <style>
        .header {
            color: #28517f;
            font-size: 40px;
            padding: 20px 0 20px 0;
            text-align: center;
            font-ranking: bold;
        }
        .subheader {
            color: #28517f;
            font-size: 20px;
            margin-bottom: 12px;
            text-align: center;
            font-style: italic;
        }
        .sidebar {
            padding: 20px;
            background-color: var(--sidebar-background-color);
        }
        .guide {
            font-size: 16px;
            line-height: 1.6;
            background-color: var(--guide-background-color);
            color: var(--guide-color);
            padding: 20px;
            border-radius: 8px;
        }
        .guide-title {
            color: #28517f;
            font-size: 24px;
            margin-bottom: 10px;
        }
        .guide-step {
            margin-bottom: 10px;
        }
        .disclaimer {
            font-size: 12px;
            color: #777777;
            margin-top: 20px;
        }
        .information-card-content {
            font-family: Arial, sans-serif;
            font-size: 16px;
            line-height: 1.6;
        }
        .information-card-text {
            font-ranking: bold;
            color: #28517f;
            margin-bottom: 10px;
        }
        .information-card-list {
            list-style-type: decimal;
            margin-left: 20px;
            margin-bottom: 10px;
        }
        .information-card-disclaimer {
            font-size: 12px;
            color: #777777;
            margin-top: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Page header
    st.title("Create your Own App! 📱")

    # Introduction
    st.markdown('<p style="font-size: 18px;">This tool generates code templates based on your inputs.</p>', unsafe_allow_html=True)

    st.sidebar.title("User Guide")

    # Define theme colors based on light and dark mode
    light_mode = {
        "sidebar-background-color": "#f7f7f7",
        "guide-background-color": "#eef4ff",
        "guide-color": "#333333",
    }

    dark_mode = {
        "sidebar-background-color": "#1a1a1a",
        "guide-background-color": "#192841",
        "guide-color": "#ffffff",
    }

    # Determine the current theme mode
    theme_mode = st.sidebar.radio("Theme Mode", ("Light", "Dark"))

    # Select the appropriate colors based on the theme mode
    theme_colors = light_mode if theme_mode == "Light" else dark_mode

    # Add user guide content to sidebar
    st.sidebar.markdown(
        f"""
        <div class="guide" style="background-color: {theme_colors['guide-background-color']}; color: {theme_colors['guide-color']}">
        <p>This app allows you to generate Python code based on your algorithm requirements. Follow these steps to create your own app:</p>

        <ol>
            <li>Name your algorithm by providing a descriptive title.</li>
            <li>Configure the input widgets to elicit the necessary inputs for your algorithm.</li>
            <li>Enter or paste the algorithm logic in pseudo-code format.</li>
            <li>Follow the remaining instructions to launch your custom app!</li>
        </ol>

        <p><em><strong>Note:</strong> Ensure that your algorithm is logically correct and efficient before generating the code.</em></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # User inputs
    col1, _ = st.columns([0.8,0.2])
    algorithm_name = col1.text_input("Enter algorithm name:", value="Weighted Fair Allocation")

    # Input widgets configuration
    st.header("Input Widgets Configuration")
    input_widget_config = generate_widget_config()
    col1, _ = st.columns([0.8,0.2])
    col1.code(input_widget_config, language="python")

    # Pseudo-algorithm
    st.header("Pseudo-Algorithm")
    pseudo_algorithm = st.text_area(f"Enter pseudo-algorithm for *{algorithm_name}*:", value=f"Write your algorithm's body here.", help="Please do not include your algorithm signature. Refer to this guide for more instructions: https://github.com/JThh/fair-alloc-app-ra/blob/new_main/contribution/CONTRIBUTION.md")
    st.code(pseudo_algorithm, language="plaintext")
    
    algorithm_name = algorithm_name.replace(' ', "_")

    # Generate code button
    if st.button("Generate Code"):
        # Generate the code
        code = generate_code(algorithm_name, input_widget_config, pseudo_algorithm)

        # Display the generated code
        st.header("Generated Code")
        st.code(code, language="python")
        
        st.markdown("## Next step")
        
        st.markdown("Follow the <a href='https://github.com/JThh/fair-alloc-app-ra/blob/new_main/contribution/CONTRIBUTION.md'>contribution guide</a> to complete the remaining steps and launch your app!", unsafe_allow_html=True)

        st.markdown("Alternatively, if you are an experienced programmer, you can refer to this <a href='https://github.com/JThh/fair-alloc-app-ra/blob/new_main/contribution/CONTRIBUTION.md'>guide</a> for implementing more complex functionalities.<br><br> If you wish to publish your app on this site (https://fair-alloc.streamlit.app), please email us. Make sure to include any **paper references** that are relevant, and optionally, you can include **your code** in the email body.<br><br>", unsafe_allow_html=True)
        # Email button
        # st.markdown("### Email your codes to the developer team")
        subject = f"Generated Code - {algorithm_name}"
        mailto_link = generate_mailto_link(subject, code)

        # Display the email link
        st.markdown(mailto_link, unsafe_allow_html=True)


# Function to generate input widget configuration
def generate_widget_config():
    widget_config = {}

    col1, col2 = st.columns([0.4,0.6])
    num_inputs = col1.number_input("Number of input widgets:", min_value=0, value=1, step=1)

    for i in range(num_inputs):
        col1, col2, _ = st.columns([0.4,0.4,0.2])
        with col1:
            widget_name = st.text_input(f"Widget {i+1} name:", value=f"Widget {i+1}")
        with col2:
            widget_type = st.selectbox(
                f"Widget {i+1} type:",
                options=["Text Input", "Number Input", "Slider", "Checkbox"],
            )
        src = "https://doc-text-input.streamlit.app"
        if widget_type == "Number Input":
            src = "https://doc-number-input.streamlit.app"
        elif widget_type == "Slider":
            src = "https://doc-slider.streamlit.app/"
        elif widget_type == "Checkbox":
            src = "https://doc-checkbox.streamlit.app/"
        st.write(f"Embedded Example for {widget_type}:")
        st.markdown(f"""               
            <iframe
            src="{src}/?embed=true&embed_options=light_theme"
            height="170"
            style="width:80%;border:none;"
            ></iframe>
        """
        , unsafe_allow_html=True)
        st.divider()
            
        widget_config[widget_name] = widget_type

    return widget_config


# Function to generate code
def generate_code(algorithm_name, input_widget_config, pseudo_algorithm):
    # Remove leading/trailing whitespaces
    algorithm_code = pseudo_algorithm.strip()

    # Replace common keywords/phrases with their corresponding Python syntax
    algorithm_code = re.sub(r'\bloop\b', 'for i in range(n):', algorithm_code)
    algorithm_code = re.sub(r'\bif\b', 'if condition:', algorithm_code)
    algorithm_code = re.sub(r'\belse\b', 'else:', algorithm_code)
    algorithm_code = re.sub(r'\bprint\b', 'print()', algorithm_code)

    # Code template
    code = f"""
import Streamlit as st

# Input Widgets
input_data = dict()
"""

    for widget_name, widget_type in input_widget_config.items():
        if widget_type == "Text Input":
            code += f"""
input_data['{widget_name}'] = st.text_input("{widget_name}:")
"""
        elif widget_type == "Number Input":
            code += f"""
input_data['{widget_name}'] = st.number_input("{widget_name}:")
"""
        elif widget_type == "Slider":
            code += f"""
input_data['{widget_name}'] = st.slider("{widget_name}:")
"""
        elif widget_type == "Checkbox":
            code += f"""
input_data['{widget_name}'] = st.checkbox("{widget_name}")
"""

    code += f"""
# Pseudo-Algorithm
'''
{pseudo_algorithm}
'''

# Algorithm Function
def {algorithm_name}(input_data):
    # Algorithm code goes here (based on your pseudo-codes)
    {algorithm_code}
    pass

# Execute the algorithm function
result = {algorithm_name}(input_data)

# Display the outputs
st.write(result)
"""

    return code


def generate_mailto_link(subject, body):
    # Prepare email details
    receiver_email = "julius.han@outlook.com"
    cc_email = "warut@comp.nus.edu.sg"

    # Create mailto link
    email_link = f"mailto:{receiver_email}?cc={cc_email}&subject={subject}"
    return f'<a href="{email_link}" style="font-size: 16px; color: #fff; background-color: #009688; padding: 10px 20px; border-radius: 5px; text-decoration: none;">Send email</a>'


# Execute the main function
if __name__ == "__main__":
    main()
