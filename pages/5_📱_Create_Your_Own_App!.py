from functools import partial

import numpy as np
import pandas as pd
import streamlit as st


def load_table(m, n, i):
    if hasattr(st.session_state, f"table_{i}"):
        table = getattr(st.session_state, f"table_{i}")
        old_n = table.shape[0]
        old_m = table.shape[1]
        if n <= old_n and m <= old_m:
            setattr(st.session_state, f"table_{i}", table.iloc[:n, :m])
        elif n > old_n:
            setattr(st.session_state, f"table_{i}", pd.concat([table, 
                                                               pd.DataFrame(np.random.randint(1, 100, (n - old_n, m)),
                                                                            columns=[
                                                                                f"Column Entity {i+1}" for i in range(m)],
                                                                            index=[f"Row Entity {i+1}" for i in range(old_n, n)])],
                                                              axis=0))
        elif m > old_m:
            setattr(st.session_state, f"table_{i}", pd.concat([table, 
                                                               pd.DataFrame(np.random.randint(1, 100, (n, m - old_m)),
                                                                            columns=[
                                                                                f"Column Entity {i+1}" for i in range(old_m, m)],
                                                                            index=[f"Row Entity {i+1}" for i in range(n)])],
                                                              axis=1))
        else:
            raise ValueError("Cannot alter both dimensions of table at the same time.")
    else:
        setattr(st.session_state, f"table_{i}", pd.DataFrame(np.random.randint(1, 100, (n, m)), 
                                                             columns=[f"Column Entity {i+1}" for i in range(m)],
                                                             index=[f"Row Entity {i+1}" for i in range(n)]))
    return getattr(st.session_state, f"table_{i}")


def tchange_callback(table, i):
    for col in table.columns:
        table[col] = table[col].apply(
            lambda x: int(float(x)))
    setattr(st.session_state, f"table_{i}", table)


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

    # Select the appropriate colors based on the theme mode
    theme_colors = light_mode

    # Add user guide content to sidebar
    st.sidebar.markdown(
        f"""
        <div class="guide" style="background-color: {theme_colors['guide-background-color']}; color: {theme_colors['guide-color']}">
        <p>This app allows you to generate Python code based on your algorithm requirements. Follow these steps to create your own app:</p>

        <ol>
            <li>Name your algorithm by providing a descriptive title.</li>
            <li>Configure the input widgets to elicit the necessary inputs for your algorithm.</li>
            <li>Enter or paste the algorithm code into the text box.</li>
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
    
    algorithm_name = algorithm_name.replace(' ', "_")
    
    # Algorithm
    st.header("Algorithm Code")
    col1, _ = st.columns([0.8,0.2])
    with col1:
        algorithm = st.text_area(f"Enter executable Python code for *{algorithm_name}*:", 
                                value=f"""def {algorithm_name}(input_data): 
    # Write or paste your algorithm's body below
                                            """, 
                                help="If unsure about the input data format, click 'Generate Code' first. Refer to this guide for more instructions: https://github.com/JThh/fair-alloc-app-ra/blob/new_main/contribution/CONTRIBUTION.md")
        st.code(algorithm, language="python")
    

    # Generate code button
    if st.button("Generate Code"):
        # Generate the code
        code = generate_code(algorithm, algorithm_name, input_widget_config)

        # Display the generated code
        st.header("Generated Code")
        st.code(code, language="python")
        
        st.markdown("## Next Step")
        
        st.markdown("Follow the <a href='https://github.com/JThh/fair-alloc-app-ra/blob/new_main/contribution/CONTRIBUTION.md'>contribution guide</a> to complete the remaining steps and launch your app!", unsafe_allow_html=True)

        st.markdown("Alternatively, if you are an experienced programmer, you can refer to this <a href='https://github.com/JThh/fair-alloc-app-ra/blob/new_main/contribution/ADVANCED_CONTRIBUTION.md'>advanced guide</a> for implementing more complex functionalities.", unsafe_allow_html=True)
        
        st.markdown("**If you wish to publish your app on this site (https://fair-alloc.streamlit.app), please make a pull request at our repository (at the upper right corner).** Make sure to include any **paper references** that are relevant in the PR description.<br><br>", unsafe_allow_html=True)
        # Email button
        # st.markdown("### Email your codes to the developer team")
        subject = f"Generated Code - {algorithm_name}"
        mailto_link = generate_mailto_link(subject, code)

        # Display the email link
        st.markdown(mailto_link, unsafe_allow_html=True)
        

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


# Function to generate input widget configuration
def generate_widget_config():
    widget_config = {}

    col1, col2 = st.columns([0.4,0.6])
    num_inputs = col1.number_input("Number of input widgets:", min_value=1, max_value=10, value=1, step=1)

    for i in range(num_inputs):
        col1, col2, _ = st.columns([0.4,0.4,0.2])
        with col1:
            widget_name = st.text_input(f"Widget {i+1} name:", value=f"Widget {i+1}")
        with col2:
            widget_type = st.selectbox(
                f"Widget {i+1} type:",
                options=["Text Input", "Table Input", "Number Input", "Slider", "Checkbox"],
            )
            
        col1, _ = st.columns([0.6,0.4])
        if widget_type == "Number Input":
            col1.number_input("Example number input widget", value=5, step=1, 
                              help="You may use this for eliciting number of items or agents.", key=f"{i}_number")
            col1.write("💡 You may use this for entering the number of items or agents.")
        elif widget_type == "Slider":
            col1.slider("Example slider", min_value=-100, max_value=100, value=(-10, 10), step=1, 
                        help="You may use this to restrict the range of random preference values.", key=f"{i}_slider")
            col1.write("💡 You may use this to restrict the range of random preference values.")
        elif widget_type == "Text Input":
            output_ = col1.text_input("Example test input box", value="Type some text here", max_chars=100, 
                            help="You may use this to specify textual inputs for your algorithm (e.g. agent names).", key=f"{i}_text")
            col1.code("You typed: "+output_, language="plaintext")
            col1.write("💡 You may use this to specify string arguments for your algorithm.")
        elif widget_type == "Table Input":
            # Add input components
            subcol1, subcol2, _ = st.columns([0.3,0.3,0.4])
            # min_col, max_col = subcol1.slider("Allowed number of column entities (e.g. items)", min_value=0, max_value=1000, value=(2, 100), step=1, 
            #              key=f"{i}_item_slider")
            # min_row, max_row = subcol2.slider("Allowed number of row entities (e.g. agents)", min_value=0, max_value=1000, value=(2, 100), step=1, 
            #              key=f"{i}_agent_slider")
            min_col, max_col = 2, 100
            min_row, max_row = 2, 100
            m = subcol1.number_input("Number of Column Entities (n)",
                                min_value=min_col, max_value=max_col, step=1, key=f"{i}_nbr_col")
            n = subcol2.number_input("Number of Row Entities (m)", min_value=min_row,
                                max_value=max_row, value=3, step=1, key=f"{i}_nbr_row")
            table = load_table(m, n, i)
            for col in table.columns:
                table[col] = table[col].map(str)
            edited_table = st.data_editor(table,
                                        key=f"table_editor_{i}",
                                        column_config={
                                            f"Column Entity {j}": st.column_config.TextColumn(
                                                f"Column Entity {j}",
                                                max_chars=4,
                                                validate=r"^(?:1000|[1-9]\d{0,2}|0)$",
                                                required=True,
                                            )
                                            for j in range(1, m+1)
                                        }
                                        |
                                        {
                                            "_index": st.column_config.Column(
                                                "💡 Hint",
                                                disabled=True,
                                            ),
                                        },
                                        on_change=partial(
                                            tchange_callback, table, i),
                                        )
            with st.spinner('Updating...'):
                tchange_callback(edited_table, i)
            col1.write("💡 You may use this to collect tabular inputs (e.g. preference table).")
        else:
            col1.checkbox("Example check box", value=True, 
                          help="You may use this to alter algorithm settings, such as 'weighted' or 'unweighted' for Row Entitys.",
                          key=f"{i}_checkbox")
            col1.write("💡 You may use this to alter algorithm settings, such as 'weighted' or 'unweighted' for Row Entitys.")
        # src = "https://doc-text-input.streamlit.app"
        # if widget_type == "Number Input":
        #     src = "https://doc-number-input.streamlit.app"
        # elif widget_type == "Slider":
        #     src = "https://doc-slider.streamlit.app/"
        # elif widget_type == "Checkbox":
        #     src = "https://doc-checkbox.streamlit.app/"
        # st.write(f"Embedded Example for {widget_type}:")
        # st.markdown(f"""               
        #     <iframe
        #     src="{src}/?embed=true&embed_options=light_theme"
        #     height="170"
        #     style="width:80%;border:none;"
        #     ></iframe>
        # """
        # , unsafe_allow_html=True)
        st.divider()
            
        widget_config[widget_name] = widget_type

    return widget_config


# Function to generate code
def generate_code(algorithm, algorithm_name, input_widget_config):
    # Code template
    code = """
import streamlit as st
    """
    
    if "Table Input" in input_widget_config.values():
        code = """
import numpy as np
import pandas as pd    
import streamlit as st    

from functools import partial

# NOTE: auxiliary functions (necessary if table inputs are used)
def load_table(m, n, i): # i-th table
    if hasattr(st.session_state, f"table_{i}"):
        table = getattr(st.session_state, f"table_{i}")
        old_n = table.shape[0]
        old_m = table.shape[1]
        if n <= old_n and m <= old_m:
            setattr(st.session_state, f"table_{i}", table.iloc[:n, :m])
        elif n > old_n:
            setattr(st.session_state, f"table_{i}", pd.concat([table, 
                                                               pd.DataFrame(np.random.randint(1, 100, (n - old_n, m)),
                                                                            columns=[
                                                                                f"Column Entity {i+1}" for i in range(m)],
                                                                            index=[f"Row Entity {i+1}" for i in range(old_n, n)])],
                                                              axis=0))
        elif m > old_m:
            setattr(st.session_state, f"table_{i}", pd.concat([table, 
                                                               pd.DataFrame(np.random.randint(1, 100, (n, m - old_m)),
                                                                            columns=[
                                                                                f"Column Entity {i+1}" for i in range(old_m, m)],
                                                                            index=[f"Row Entity {i+1}" for i in range(n)])],
                                                              axis=1))
        else:
            raise ValueError("Cannot alter both dimensions of table at the same time.")
    else:
        setattr(st.session_state, f"table_{i}", pd.DataFrame(np.random.randint(1, 100, (n, m)), 
                                                             columns=[f"Column Entity {i+1}" for i in range(m)],
                                                             index=[f"Row Entity {i+1}" for i in range(n)]))
    return getattr(st.session_state, f"table_{i}")


def tchange_callback(table, i):
    for col in table.columns:
        table[col] = table[col].apply(
            lambda x: int(float(x)))
    setattr(st.session_state, f"table_{i}", table)

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
        elif widget_type == "Table Input":
            code += """
i = ... # TODO: replace i with the index of table (ith table; avoid collision of session states).

# adjust limit of table sizes (and row/col names) if necessary.
min_col, max_col = 2, 100
min_row, max_row = 2, 100

m = subcol1.number_input("Number of Column Entities (n)",
                    min_value=min_col, max_value=max_col, step=1, key=f"{i}_nbr_col")
n = subcol2.number_input("Number of Row Entities (m)", min_value=min_row,
                    max_value=max_row, value=3, step=1, key=f"{i}_nbr_row")

table = load_table(m, n, i)
for col in table.columns:
    table[col] = table[col].map(str)  # map to string for regex verification

edited_table = st.data_editor(table,
                            key=f"table_editor_{i}",
                            column_config={
                                f"Column Entity {j}": st.column_config.TextColumn(
                                    f"Column Entity {j}",
                                    max_chars=4,
                                    validate=r"^(?:1000|[1-9]\d{0,2}|0)$",
                                    required=True,
                                )
                                for j in range(1, m+1)
                            }
                            |
                            {
                                "_index": st.column_config.Column(
                                    "💡 Hint",
                                    disabled=True,
                                ),
                            },
                            on_change=partial(
                                tchange_callback, table, i),
                            )            
"""
            code += f"""

input_data['{widget_name}'] = edited_table.values # convert pd.dataframe to python lists
            """
            
            
    code += f"""
{algorithm}

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
    return f'<a href="{email_link}" style="font-size: 16px; color: #fff; background-color: #009688; padding: 10px 20px; border-radius: 5px; text-decoration: none;">Send email to us</a>'


# Execute the main function
if __name__ == "__main__":
    main()
