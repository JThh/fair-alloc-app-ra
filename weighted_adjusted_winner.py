from collections import defaultdict
import base64
from functools import partial
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


def load_preferences(m, n, upload_preferences):
    if hasattr(st.session_state, "preferences"):
        if upload_preferences:
            preferences_default = None
            # Load the user-uploaded preferences file
            try:
                preferences_default = pd.read_csv(upload_preferences, index_col=0)
                if preferences_default.shape != (n, m):
                    x, y = preferences_default.shape
                    st.session_state.preferences.iloc[:x,
                                                      :y] = preferences_default
                else:
                    st.session_state.preferences = pd.DataFrame(preferences_default,
                                                                columns=st.session_state.preferences.columns,
                                                                index=st.session_state.preferences.index)
                return st.session_state.preferences
            except Exception as e:
                st.error(f"An error occurred while loading the preferences file.")
                st.stop()
        old_n = st.session_state.preferences.shape[0]
        old_m = st.session_state.preferences.shape[1]
        if n <= old_n and m <= old_m:
            st.session_state.preferences = st.session_state.preferences.iloc[:n, :m]
            return st.session_state.preferences
        elif n > old_n:
            st.session_state.preferences = pd.concat([st.session_state.preferences,
                                                      pd.DataFrame(np.random.randint(1, 100, (n - old_n, m)),
                                                                   columns=[
                                                          f"Item {i+1}" for i in range(m)],
                                                          index=[f"Agent {i+1}" for i in range(old_n, n)])],
                                                     axis=0)
            return st.session_state.preferences
        elif m > old_m:
            st.session_state.preferences = pd.concat([st.session_state.preferences,
                                                      pd.DataFrame(np.random.randint(1, 100, (n, m - old_m)),
                                                                   columns=[
                                                          f"Item {i+1}" for i in range(old_m, m)],
                                                          index=[f"Agent {i+1}" for i in range(n)])],
                                                     axis=1)
            return st.session_state.preferences
        else:
            st.session_state.preferences = pd.DataFrame(np.random.randint(1, 100, (n, m)), columns=[f"Item {i+1}" for i in range(m)],
                                                        index=[f"Agent {i+1}" for i in range(n)])
            return st.session_state.preferences

    if upload_preferences:
        preferences_default = None
        # Load the user-uploaded preferences file
        try:
            preferences_default = pd.read_csv(upload_preferences)
            if preferences_default.shape != (n, m):
                st.error(
                    f"The uploaded preferences file should have a shape of ({n}, {m}).")
                st.stop()
        except Exception as e:
            st.error("An error occurred while loading the preferences file.")
            st.stop()
    else:
        preferences_default = pd.DataFrame(np.random.randint(1, 100, (n, m)), columns=[f"Item {i+1}" for i in range(m)],
                                           index=[f"Agent {i+1}" for i in range(n)])
    st.session_state.preferences = preferences_default
    return st.session_state.preferences


def load_weights(n, unweighted=False):
    if hasattr(st.session_state, "weights"):
        if unweighted:
            weights = np.ones(n)
            weights = pd.DataFrame(weights, index=[
                                   f'Agent {i+1}' for i in range(n)], columns=['Weights'], dtype=int)
            return weights
        if n < st.session_state.weights.shape[0]:
            weights = st.session_state.weights.iloc[:n, :]
            return weights
        else:
            old_n = st.session_state.weights.shape[0]
            weights = pd.concat([st.session_state.weights,
                                 pd.DataFrame(
                                     np.arange(
                                         old_n+1, n+1),
                                     index=[f"Agent {i}" for i in range(
                                         old_n+1, n+1)],
                                     columns=['Weights'], dtype=int)], axis=0)
            return weights
    if unweighted:
        weights = np.ones(n)
    else:
        weights = np.arange(1, n+1)
    weights = pd.DataFrame(weights, index=[
                           f'Agent {i+1}' for i in range(n)], columns=['Weights'], dtype=int)
    return weights


def wchange_callback(weights):
    st.session_state.weight_checkbox = False
    for col in weights.columns:
        weights[col] = weights[col].map(lambda x: int(float(x)))
    st.session_state.weights = weights


def pchange_callback(preferences):
    for col in preferences.columns:
        preferences[col] = preferences[col].apply(
            lambda x: int(float(x)))
    st.session_state.preferences = preferences


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
        left = sum(preferences[i][bundle_j]) / weights[j] - \
            sum(preferences[i][bundle_i]) / weights[i]
        right = (y / weights[i] + x / weights[j]) * \
            max(preferences[i][bundle_j])
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
st.markdown('<h1 class="header">Fast and Fair Goods Allocation</h1>',
            unsafe_allow_html=True)

# Insert header image
st.sidebar.image("./head_image.png", use_column_width=True,
                 caption='Image Credit: Fulfillment.com')

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

    <h3 style="color: {theme_colors['guide-color']};">Follow these steps to use the app:</h3>

    <ol>
        <li>Specify the number of agents (n) and items (m) using the number input boxes.</li>
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

# Add input components
col1, col2, col3 = st.columns(3)
n = col1.number_input("Number of agents (n)",
                      min_value=2, max_value=100, step=1)
m = col2.number_input("Number of goods (m)", min_value=2,
                      max_value=1000, value=6, step=1)
x = col3.slider("Choose a value for x in WEF(x, 1-x)",
                min_value=0.0, max_value=1.0, value=0.5, step=0.01, help="💡 Large x favors low-weight agents")

upload_preferences = None
unweighted = False

col1, col2 = st.columns([0.5, 0.5])
with col1:
    unweighted = st.checkbox("⭐ Symmetric Agents (Unweighted Settings)",
                             key='weight_checkbox',
                             value=st.session_state.weight_checkbox
                             if hasattr(st.session_state, "weight_checkbox") else False)
with col2:
    if st.checkbox("⭐ Upload Local Preferences CSV"):
        upload_preferences = st.file_uploader(
            f"Upload Preferences of shape ({n}, {m})", type=['csv'])

st.write("🌟 Agent Weights (1-1000):")

with st.spinner("Loading..."):
    weights = load_weights(n, unweighted)
    st.session_state.weights = weights
    for col in weights.columns:
        weights[col] = weights[col].map(str)

edited_ws = st.data_editor(weights.T,
                            key="weight_editor",
                            column_config={
                                f"Agent {i}": st.column_config.TextColumn(
                                    f"Agent {i}",
                                    help=f"Agent {i}'s Weight",
                                    # min_value=1,
                                    # max_value=1000,
                                    # width='medium',  # Set the desired width here
                                    # step=1,
                                    # format="%d",
                                    required=True,
                                    max_chars=4,
                                    validate=r"^(?:[1-9]\d{0,2}|1000)$",
                                )
                                for i in range(1, n+1)
                            }
                            |
                            {
                                "_index": st.column_config.Column(
                                    "💡 Hint",
                                    help="Support copy-paste from Excel sheets and bulk edits",
                                    disabled=True,
                                ),
                            },
                            on_change=partial(wchange_callback, weights),
                            )
with st.spinner("Updating..."):
    for col in edited_ws.columns:
        edited_ws[col] = edited_ws[col].map(lambda x: int(round(float(x))))
    st.session_state.weights = edited_ws.T


weights = edited_ws.values[0]

# with col3:
#     edited_ws['Variations'] = edited_ws.values.tolist()
#     st.line_chart(edited_ws['Variations'])
    # print(edited_ws['Variations'])
    # st.dataframe(edited_ws['Variations'],
    #              column_config={
    #                 "Variations": st.column_config.LineChartColumn(
    #                     "Agent Weights Variations", y_min=0, y_max=1000
    #                 ),
    #              },
    #              hide_index=True)

# Download weights as CSV
weights_csv = edited_ws.to_csv()
b64 = base64.b64encode(weights_csv.encode()).decode()
href = f'<a href="data:file/csv;base64,{b64}" download="weights.csv">Download Weights CSV</a>'
st.markdown(href, unsafe_allow_html=True)

# Agent Preferences
st.write("📊 Agent Preferences (0-1000, copyable from local sheets):")

preferences = load_preferences(m, n, upload_preferences)
for col in preferences.columns:
    preferences[col] = preferences[col].map(str)

edited_prefs = st.data_editor(preferences,
                                key="pref_editor",
                                column_config={
                                    f"Item {j}": st.column_config.TextColumn(
                                        f"Item {j}",
                                        help=f"Agents' Preferences towards Item {j}",
                                        max_chars=4,
                                        validate=r"^(?:1000|[1-9]\d{0,2}|0)$",
                                        # width='small',  # Set the desired width here
                                        # min_value=0,
                                        # max_value=1000,
                                        # step=1,
                                        # format="%d",
                                        required=True,
                                    )
                                    for j in range(1, m+1)
                                }
                                |
                                {
                                    "_index": st.column_config.Column(
                                        "💡 Hint",
                                        help="Support copy-paste from Excel sheets and bulk edits",
                                        disabled=True,
                                    ),
                                },
                                on_change=partial(
                                    pchange_callback, preferences),
                                )
with st.spinner('Updating...'):
    for col in edited_prefs.columns:
        edited_prefs[col] = edited_prefs[col].apply(
            lambda x: int(float(x)))
    st.session_state.preferences = edited_prefs

preferences = edited_prefs.values

# Download preferences as CSV
preferences_csv = edited_prefs.to_csv()
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
                Mithun Chakraborty, Erel Segal-Halevi, and Warut Suksompong. 2022. <a href="https://arxiv.org/pdf/2112.04166.pdf" target="_blank">Weighted Fairness Notions for Indivisible Items Revisited.</a> Proceedings of the 36th AAAI Conference on Artificial Intelligence (AAAI)(2022), 4949–4956.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

start_algo = st.button("⏳ Run Weighted Picking Sequence Algorithm ")
if start_algo:
    with st.spinner('Executing...'):
        if n * m * 0.01 > 3:
            time.sleep(2)
        else:
            time.sleep(n * m * 0.01)

    start_time = time.time()
    outcomes = wef1x_algorithm(x, m, n, weights, preferences)
    end_time = time.time()
    elapsed_time = end_time - start_time

    st.write("🎉 Outcomes:")
    outcomes_list = [[key, sorted(value)] for key, value in outcomes.items()]
    outcomes_df = pd.DataFrame(outcomes_list, columns=['Agent', 'Items'])
    outcomes_df['Agent'] += 1
    outcomes_df['Agent'] = outcomes_df['Agent'].apply(str)
    outcomes_df['Items'] = outcomes_df['Items'].apply(
        lambda x: [_x + 1 for _x in x])
    outcomes_df['Items'] = outcomes_df['Items'].apply(
        lambda x: ', '.join(map(str, x)))

    # Sort the table
    outcomes_df = outcomes_df.sort_values(['Agent'],
                                          key=lambda col: col.astype(int))

    st.data_editor(outcomes_df,
                   column_config={
                       "Agents": st.column_config.NumberColumn(
                           "Agent",
                           help="The list of agents that get allocated",
                           step=1,
                       ),
                       "Items": st.column_config.ListColumn(
                           "Items",
                           help="The list of items allocated to agents",
                       ),
                   },
                   hide_index=True,
                   disabled=True,
                   )

    # Print timing results
    st.write(f"⏱️ Timing Results:")
    st.write(f"Elapsed Time: {elapsed_time:.4f} seconds")

    # Add expandable information card
    with st.expander("Explanation of the outcomes", expanded=False):
        print(outcomes)
        # outcomes is already a dictionary.
        output_str = ""
        has_lead_str = False
        for i in range(n):
            if not has_lead_str:
                b = outcomes[i]
                output_str += f"**Agent {i+1}** has weight {weights[i]} and \
                    receives value {sum(preferences[i][b])}.\n\n"
                has_lead_str = True
            for j in range(n):
                if i == j:
                    continue
                else:
                    bi, bj = outcomes[i], outcomes[j]
                    if sum(preferences[i][bj]) == 0:
                        output_str += f"Agent {i+1} has value 0 for the bundle of Agent {j+1}, so Agent {i+1} does not envy Agent {j+1}.\n"
                    else:
                        output_str += f"Agent {i+1} has value {sum(preferences[i][bj])} for the bundle of Agent {j+1}, \
                            who has weight {weights[j]}. Agent {i+1}'s maximum value for an item in Agent {j+1}'s \
                                bundle is {max(preferences[i][bj])}. Agent {i+1} does not envy Agent {j+1} according to WEF({x:.2f}, {1-x:.2f}) \
                                    because ({sum(preferences[i][bi])} + {1-x:.2f} * {max(preferences[i][bj])}) / {weights[i]} \
                                        = {(sum(preferences[i][bi]) + (1-x)*max(preferences[i][bj])) / weights[i]:.2f} \
                                        > {(sum(preferences[i][bj]) - x*max(preferences[i][bj])) / weights[j]:.2f} \
                                            = ({sum(preferences[i][bj])} - {x:.2f} * {max(preferences[i][bj])}) / {weights[j]}.\n\n"
            has_lead_str = False
        st.markdown(output_str)

    print({otc[0]: otc[1] for otc in outcomes_df.to_numpy()})

    # Download outcomes in JSON format
    outcomes_json = json.dumps({otc[0]: otc[1]
                               for otc in outcomes_df.to_numpy()}, indent=4)
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
