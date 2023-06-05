from collections import defaultdict
import base64
import json
import time

import numpy as np
import pandas as pd
import streamlit as st


# Set page configuration
st.set_page_config(
    page_title="Weighted Fairness App",
    page_icon="icon.png",
    layout="wide",
)

# Custom CSS styles
st.markdown(
    """
    <style>
    .header {
        color: #28517f;
        font-size: 40px;
        padding: 20px 0 20px 0;
        text-align: center;
        font-weight: bold;
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
        font-weight: bold;
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

@st.cache_data
def load_preferences(m, n, upload_preferences):
    if upload_preferences:
        preferences_default = None
        # Load the user-uploaded preferences file
        try:
            preferences_default = pd.read_csv(upload_preferences)
            if preferences_default.shape != (n, m):
                st.error(f"The uploaded preferences file should have a shape of ({n}, {m}).")
                st.stop()
        except Exception as e:
            st.error("An error occurred while loading the preferences file.")
            st.stop()
    else:
        preferences_default = pd.DataFrame(np.random.randint(1, 100, (n, m)), columns=[f"Item {i+1}" for i in range(m)], 
                                            index=[f"Agent {i+1}" for i in range(n)])
    return preferences_default

@st.cache_data
def load_weights(n, unweighted=False):
    if unweighted:
        weights = np.ones(n)
    else:
        weights = np.arange(1, n+1)
    return pd.DataFrame(weights, index=[f'Agent {i+1}' for i in range(n)], columns=['Weights'])

# def wef1_po_algorithm(x, m, n, weights, preferences):
#     # Implementation of WEF1+PO algorithm
#     # Add your code here
#     objects = sorted(list(range(m)), key=lambda x:preferences[0][x] / preferences[1][x], reverse=True)
#     d = 1
#     while sum([preferences[0][objects[i]] for i in range(d+1)]) / weights[0] \
#             < sum([preferences[0][objects[j]] for j in range(d+2, m)]) / weights[1]:
#         d += 1
#     return {0: objects[:d+1], 1:  objects[d+1:]}

def wef1x_algorithm(x, m, n, weights, preferences):
    # Implementation of WEF1 algorithm
    # Add your code here
    bundles = defaultdict(list)
    times = np.zeros(n)
    remaining_items = list(range(m))
    # print(bundles, times, remaining_items)
    while remaining_items:
        i = np.argmin((times + (1 - x)) / weights)
        o = remaining_items[np.argmax(preferences[i][remaining_items])]
        # Add o to bundle A_i
        bundles[i].append(o)
        # Remove o from items
        remaining_items.remove(o)
        times[i] += 1
    return bundles

def wef1x_checker(outcomes, x, m, n, weights, preferences):
    # Implementation of WEF1 checker
    # Add your code here
    y = 1 - x
    def is_single_wef1x(i, j, bundle_i, bundle_j):
        left = sum(preferences[i][bundle_j]) / weights[j] - sum(preferences[i][bundle_i]) / weights[i]
        right = (y / weights[i] + x / weights[j]) * max(preferences[i][bundle_j])
        return left <= right
    for i in range(n):
        for j in range(i, n):
            bi, bj = outcomes[i][1], outcomes[j][1]
            if not is_single_wef1x(i, j, bi, bj):
                st.write(f"Not fulfiling WEF({x}, {1-x}).")
                return
    st.write(f"Fulfiled WEF({x}, {1-x}).")
    return

# Set the title and layout of the web application
st.markdown('<h1 class="header">Fast and Fair Goods Allocation</h1>', unsafe_allow_html=True)

# Subheader
# st.markdown('<h2 class="subheader">Developed by Jiatong Han @ NUS</h2>', unsafe_allow_html=True)

# Insert header image
st.sidebar.image("./head_image.png", use_column_width=True, caption='Image Credit: Fulfillment.com')

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
    <p>This app calculates outcomes using the Weighted Picking Sequence algorithm.</p>

    <h3>Follow these steps to use the app:</h3>

    <ol>
        <li>Specify the number of items (m) and agents (n) using the number input boxes.</li>
        <li>Choose to either upload a preferences file or edit the  preferences.</li>
        <li>Click the 'Run Algorithm' button to start the algorithm.</li>
        <li>You can download the outcomes as a JSON file or the preferences as a CSV file using the provided links.</li>
    </ol>

    <p><em><strong>Disclaimer:</strong> The generated outcomes are for demonstration purposes only and may not reflect real-world scenarios.</em></p>

    <p><em>Image Credit: <a href="https://www.thefulfillmentlab.com/blog/product-allocation">Image Source</a></em>.
    <em>Icon Credit: <a href="https://www.flaticon.com/free-icon/orange_135620">Icon Source</a></em>.</p>
    </div>
    """,
    unsafe_allow_html=True
)

def uncheck_callback():
    st.session_state.weight_checkbox = False

# Add input components
col1, col2, col3 = st.columns(3)
n = col1.number_input("Number of agents (n)", min_value=2, step=1)
m = col2.number_input("Number of goods (m)", min_value=2, value=10, step=1)
x = col3.slider("Choose a value for x in WEF(x, 1-x)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

upload_preferences = None
unweighted = False

col1, col2 = st.columns([0.5, 0.5])
with col1:
    if st.checkbox("⭐ Symmetric Agents (Unweighted Settings)", 
                   key='weight_checkbox', 
                   value=False):
        unweighted = True

with col2:
    if st.checkbox("⭐ Upload Local Preferences CSV"):
        upload_preferences = st.file_uploader(f"Upload Preferences of shape ({n}, {m})", type=['csv'])

if m < n:
    st.warning("The number of goods (m) must be equal to or greater than the number of agents (n).")
else:
    # Agent Weights
    st.write("🌟 Agent Weights (1-1000):")
    weights = load_weights(n, unweighted)
    edited_ws = st.data_editor(weights.T, 
                                key="weight_editor",
                                column_config={
                                    f"Agent {i}": st.column_config.NumberColumn(
                                        f"Agent {i}",
                                        help=f"Agent {i}'s Weight",
                                        min_value=1,
                                        max_value=1000,
                                        step=1,
                                        format="%d",
                                    )
                                for i in range(1, n+1)},
                                # on_change=uncheck_callback,
                               )
    weights = edited_ws.values[0]
    # invalid_weights = any((w < 1 or w > 1000) for w in weights)
    # if invalid_weights:
    #     st.error("Invalid weight values. Please enter positive integers less than or equal to 1000.")
    #     st.stop()

    # Agent Preferences
    st.write("📊 Agent Preferences (0-1000, copyable from local sheets):")
    preferences = load_preferences(m, n, upload_preferences)
    edited_prefs = st.data_editor(preferences, 
                                key="pref_editor",
                                column_config={
                                    f"Item {j}": st.column_config.NumberColumn(
                                        f"Item {j}",
                                        help=f"Agents' Preferences towards Item {j}",
                                        min_value=0,
                                        max_value=1000,
                                        step=1,
                                        format="%d",
                                    )
                                for j in range(1, m+1)},
                                )
    preferences = edited_prefs.values
    # invalid_prefs = any((p < 1 or p > 1000) for p in preferences.flatten())
    # if invalid_prefs:
    #     st.error("Invalid preference values. Please enter positive integers less than or equal to 1000.")
    #     st.stop()
    
    # Download preferences as CSV
    preferences_csv = edited_prefs.to_csv(index=False)
    b64 = base64.b64encode(preferences_csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="preferences.csv">Download Preferences CSV</a>'
    st.markdown(href, unsafe_allow_html=True)
    
    # Add expandable information card
    with st.expander("ℹ️ Information", expanded=False):
        st.markdown(
            """
            <style>
            .information-card-content {
                margin-top: 20px;
            }
            .information-card-text {
                font-size: 14px;
                line-height: 1.5;
                color: var(--text-color);
            }
            .information-card-citation {
                font-size: 12px;
                font-style: italic;
                color: #777777;
            }
            .information-card-formula {
                font-size: 14px;
                line-height: 1.5;
                color: #555555;
                font-weight: bold;
            }
            </style>
            <div class="information-card-content">
                <h2 class="information-card-header">Information</h2>
                <p class="information-card-text">
                    The  Weighted Picking Sequence algorithm is used for goods allocation in situations where the items are indivisible.
                    It provides a method to allocate goods to agents in a way that balances fairness and efficiency.
                </p>
                <h3 class="information-card-header">Algorithm Overview</h3>
                <p class="information-card-text">
                    The algorithm starts with each agent's initial weights. These weights represent the relative importance or priority of the agents in the allocation process.
                    The algorithm then iteratively selects an item to allocate and assigns it to the agent who values it the most based on their preferences.
                    The weights of the agents are adjusted after each allocation to reflect the items already allocated to them.
                    This adjustment ensures that agents with fewer allocated items are given higher weights to maintain fairness in subsequent allocations.
                    The process continues until all items are allocated or no further allocations can be made while satisfying certain fairness criteria.
                </p>
                <h3 class="information-card-header">Fairness Considerations</h3>
                <p class="information-card-text">
                    The Weighted Picking Sequence algorithm incorporates fairness notions by dynamically adjusting the weights of agents during the allocation process.
                    By giving higher weights to agents with fewer allocated items, the algorithm aims to balance the distribution of goods among agents.
                    This helps prevent situations where some agents receive a disproportionate number of items, leading to unfair outcomes.
                </p>
                <h3 class="information-card-header">Efficiency Trade-offs</h3>
                <p class="information-card-text">
                    The  Weighted Picking Sequence algorithm also considers efficiency by allocating items to agents based on their preferences.
                    By allocating items to agents who value them the most, the algorithm aims to maximize overall utility and satisfaction.
                    However, achieving perfect efficiency may not always be possible while ensuring fairness.
                    Trade-offs between efficiency and fairness are inherent in the allocation process, and the algorithm seeks to strike a balance between these objectives.
                </p>
                <h3 class="information-card-header">Mathematical Formulation</h3>
                <p class="information-card-text">
                    The Weighted Picking Sequence algorithm can be represented using the following formula:
                </p>
                <p class="information-card-formula">
                    next_pick = argmin<sub>i ∈ N</sub> {(t<sub>i</sub> + (1 - x)) / w<sub>i</sub>}
                </p>
                <p class="information-card-text">
                    Where:
                </p>
                <ul class="information-card-text">
                    <li>w<sub>i</sub> is the weight of agent i</li>
                    <li>t<sub>i</sub> is the number of times agent i has picked so far</li>
                    <li>N is the set of agents</li>
                </ul>
                <p class="information-card-text">
                    For a detailed explanation of the Weighted Picking Sequence algorithm for WEF(x, 1-x) and its theoretical foundations, please refer to the following paper:
                </p>
                <p class="information-card-citation">
                   Mithun Chakraborty, Erel Segal-Halevi, and Warut Suksompong. 2022. Weighted Fairness Notions for Indivisible Items Revisited. Proceedings of the 36th AAAI Conference on Artificial Intelligence (AAAI)(2022), 4949–4956.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    start_algo = st.button("⏳ Run Weighted Picking Sequence Algorithm ")    
    if start_algo:
        with st.spinner('Executing...'):
            if n * m * 0.01 > 3:
                time.sleep(3)
            else:
                time.sleep(n * m * 0.01)

        start_time = time.time()
        outcomes = wef1x_algorithm(x, m, n, weights, preferences)
        end_time = time.time()
        elapsed_time = end_time - start_time

        st.write("🎉 Outcomes:")
        outcomes = [[key, sorted(value)] for key, value in outcomes.items()]
        outcomes_df = pd.DataFrame(outcomes, columns=['Agents', 'Items'])
        outcomes_df['Agents'] += 1
        outcomes_df['Agents'] = outcomes_df['Agents'].apply(str)
        outcomes_df['Items'] = outcomes_df['Items'].apply(lambda x : [_x + 1 for _x in x])
        outcomes_df['Items'] = outcomes_df['Items'].apply(lambda x : ', '.join(map(str, x)))
        
        # Sort the table
        outcomes_df = outcomes_df.sort_values(['Agents','Items'])
        
        # # CSS to inject contained in a string
        # hide_table_row_index = """
        #             <style>
        #             thead tr th:first-child {display:none}
        #             tbody th {display:none}
        #             </style>
        #             """

        # # Inject CSS with Markdown
        # st.markdown(hide_table_row_index, unsafe_allow_html=True)

        # Display a static table
        # st.table(outcomes_df)
        st.data_editor(outcomes_df, 
                    column_config={
                        "Items": st.column_config.ListColumn(
                            "Items",
                            help="The list of items allocated to agents",
                        ),
                    },
                    hide_index=True,
                )
        
        # Print timing results
        st.write(f"⏱️ Timing Results:")
        st.write(f"Elapsed Time: {elapsed_time:.4f} seconds")
        
        print({otc[0]: otc[1] for otc in outcomes_df.to_numpy()})
            
        # Download outcomes in JSON format
        outcomes_json = json.dumps({otc[0]: otc[1] for otc in outcomes_df.to_numpy()}, indent=4)
        st.markdown("### Download Outcomes as JSON")
        b64 = base64.b64encode(outcomes_json.encode()).decode()
        href = f'<a href="data:application/json;base64,{b64}" download="outcomes.json">Download Outcomes JSON</a>'
        st.markdown(href, unsafe_allow_html=True)
        st.json(outcomes_json)

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