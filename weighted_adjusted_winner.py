from collections import defaultdict
import base64
import json
import time

import numpy as np
import pandas as pd
import streamlit as st

# Custom CSS styles
st.markdown(
    """
    <style>
    .header {
        color: #28517f;
        font-size: 40px;
        padding: 30px 0;
        text-align: center;
        font-weight: bold;
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
    <p>This app calculates outcomes using the Weighted Adjusted Winner algorithm.</p>

    <h3>Follow these steps to use the app:</h3>

    <ol>
        <li>Specify the number of items (m) and agents (n) using the number input boxes.</li>
        <li>Choose to either upload a preferences file or edit the random preferences.</li>
        <li>You can download the outcomes as a JSON file or the preferences as a CSV file using the provided links.</li>
    </ol>

    <p><em><strong>Disclaimer:</strong> The generated outcomes are for demonstration purposes only and may not reflect real-world scenarios.</em></p>

    <p><em>Image Credit: <a href="https://www.thefulfillmentlab.com/blog/product-allocation">Image Source</a></em></p>
    </div>
    """,
    unsafe_allow_html=True
)


# Add input components
col1, col2, col3 = st.columns(3)
m = col1.number_input("Number of goods (m)", min_value=2, value=10, step=1)
n = col2.number_input("Number of agents (n)", min_value=2, step=1)
x = col3.slider("Choose a value for x in WEF(x,1-x)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

upload_preferences = st.file_uploader(f"Upload Preferences of shape ({n}, {m})", type=['csv'])

if m < n:
    st.warning("The number of goods (m) must be equal to or greater than the number of agents (n).")
else:
    weights = np.arange(1, n+1)
    
    st.write("Agent Preferences (copyable from sheets):")
    preferences = load_preferences(m, n, upload_preferences)
    edited_prefs = st.experimental_data_editor(preferences, key="data_editor")
    # Convert to numpy arrays
    preferences = edited_prefs.values
    
    # Download preferences as CSV
    preferences_csv = edited_prefs.to_csv(index=False)
    b64 = base64.b64encode(preferences_csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="preferences.csv">Download Preferences CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

    st.write("⏳ Run the Weight-Adjusted Picker (WEF(x,1-x))...")
    
    with st.spinner('Executing...'):
        time.sleep(n * m * 0.01)    

    start_time = time.time()
    outcomes = wef1x_algorithm(x, m, n, weights, preferences)
    end_time = time.time()
    elapsed_time = end_time - start_time

    st.write("🎉 Outcomes:")
    outcomes = [[key, sorted(value)] for key, value in outcomes.items()]
    outcomes_df = pd.DataFrame(outcomes, columns=['Agents', 'Items'])
    outcomes_df['Agents'] += 1
    outcomes_df['Items'] = outcomes_df['Items'].apply(lambda x : [_x + 1 for _x in x])
    outcomes_df['Items'] = outcomes_df['Items'].apply(lambda x : ', '.join(map(str, x)))
    st.table(outcomes_df)
    
    # Print timing results
    st.write("⏱️ Timing Results:")
    st.write(f"Elapsed Time: {elapsed_time:.4f} seconds")

    # st.warning("⚠️ Running the WEF Checker takes exponential time!")

    # if st.button(f"Run WEF({x}, {1-x}) Checker"):
    #     st.write(f"Running WEF Checker...")
    #     wef1x_checker(outcomes, x, m, n, weights, preferences)
        
    # Download outcomes in JSON format
    outcomes_json = json.dumps([oc[1] for oc in outcomes], indent=4)
    st.markdown("### Download Outcomes as JSON")
    b64 = base64.b64encode(outcomes_json.encode()).decode()
    href = f'<a href="data:application/json;base64,{b64}" download="outcomes.json">Download Outcomes JSON</a>'
    st.markdown(href, unsafe_allow_html=True)
    st.json(outcomes_json)
    # st.download_button(
    #     label="Download Allocations",
    #     file_name="outcomes.json",
    #     mime="application/json",
    #     data=outcomes_json,
    # )