from collections import defaultdict
from functools import lru_cache
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

@lru_cache
def generate_preferences(n, m):
    return np.random.rand(n, m)

def wef1_po_algorithm(m, n, weights, preferences):
    # Implementation of WEF1+PO algorithm
    # Add your code here
    objects = sorted(list(range(m)), key=lambda x:preferences[0][x] / preferences[1][x], reverse=True)
    d = 1
    while sum([preferences[0][objects[i]] for i in range(d+1)]) / weights[0] \
            < sum([preferences[0][objects[j]] for j in range(d+2, m)]) / weights[1]:
        d += 1
    return {0: objects[:d+1], 1:  objects[d+1:]}

def wef1_algorithm(m, n, weights, preferences):
    # Implementation of WEF1 algorithm
    # Add your code here
    bundles = defaultdict(list)
    times = np.zeros(n)
    remaining_items = list(range(m))
    # print(bundles, times, remaining_items)
    while remaining_items:
        i = np.argmin(times / weights)
        o = remaining_items[np.argmax(preferences[i][remaining_items])]
        # Add o to bundle A_i
        bundles[i].append(o)
        # Remove o from items
        remaining_items.remove(o)
        times[i] += 1
    return bundles

def wef1_checker(outcomes, m, n, weights, preferences):
    # Implementation of WEF1 checker
    # Add your code here
    print(outcomes)
    def is_single_wef1(i, j, bundle_i, bundle_j):
        if sum(preferences[i][bundle_i]) / weights[i] < \
                sum(preferences[i][bundle_j]) / weights[j]:
            most_valuable_item_value = max(preferences[i][bundle_j])
            return sum(preferences[i][bundle_i]) / weights[i] >= \
                (sum(preferences[i][bundle_j]) - most_valuable_item_value) / weights[j]
        return True

    for i in range(n):
        for j in range(i, n):
            bi, bj = outcomes[i][1], outcomes[j][1]
            if not is_single_wef1(i, j, bi, bj) or not is_single_wef1(j, i, bj, bi):
                st.write("Not fulfiling WEF1.")
                return
    st.write("Fulfiled WEF1.")
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
        <li>Enter the number of items (m) and agents (n) in the input fields on the left.</li>
        <li>The app will generate agent preferences and run the algorithm.</li>
        <li>The outcomes will be displayed in a table.</li>
        <li>You can click the <strong>"Run WEF1 Checker"</strong> button to perform a WEF1 check.</li>
        <li>To download the outcomes as a JSON file, click the link at the bottom.</li>
    </ol>

    <p><em><strong>Disclaimer:</strong> The generated outcomes are for demonstration purposes only and may not reflect real-world scenarios.</em></p>

    <p><em>Image Credit: <a href="https://www.thefulfillmentlab.com/blog/product-allocation">Image Source</a></em></p>
    </div>
    """,
    unsafe_allow_html=True
)


# Add input components
col1, col2 = st.columns(2)
m = col1.number_input("Number of indivisible goods (m)", min_value=2, max_value=1000, value=10, step=1)
n = col2.number_input("Number of agents (n)", min_value=2, max_value=1000, step=1)

if m < n:
    st.warning("The number of goods (m) must be equal to or greater than the number of agents (n).")
else:
    weights = np.arange(1, n+1)
    preferences = generate_preferences(n, m)

    st.write("Agent Preferences:")
    st.dataframe(preferences)

    st.write("🔍 Running the Algorithm... ⏳")

    if n == 2:
        st.write("Running Weighted Adjust Winner (WEF1+PO) algorithm...")
        # with st.beta_expander("The Algorithm"):
        #     st.write("")
        start_time = time.time()
        outcomes = wef1_po_algorithm(m, n, weights, preferences)
        end_time = time.time()
        elapsed_time = end_time - start_time
    else:
        st.write("Pick the Least Weight-Adjusted Frequent Picker (WEF1)...")
        start_time = time.time()
        outcomes = wef1_algorithm(m, n, weights, preferences)
        end_time = time.time()
        elapsed_time = end_time - start_time

    st.write("🎉 Outcomes:")
    outcomes = [[key, sorted(value)] for key, value in outcomes.items()]
    outcomes_df = pd.DataFrame(outcomes, columns=['Agent', 'Items'])
    outcomes_df['Items'] = outcomes_df['Items'].apply(lambda x: ', '.join(map(str, x)))
    st.table(outcomes_df)
    
    # Print timing results
    st.write("⏱️ Timing Results:")
    st.write(f"Elapsed Time: {elapsed_time:.4f} seconds")

    # for agent_id, item_list in outcomes.items():
    #     st.write(f"{agent_id}: {sorted(item_list)}")

    st.warning("⚠️ Running the WEF1 Checker takes exponential time!")

    if st.button("Run WEF1 Checker"):
        st.write("Running WEF1 Checker...")
        wef1_checker(outcomes, m, n, weights, preferences)
        st.write("WEF1 Checker completed.")
        
    # Download outcomes in JSON format
    outcomes_json = json.dumps([oc[1] for oc in outcomes], indent=4)
    st.markdown("### Download Outcomes as JSON")
    st.download_button(
        label="Download JSON",
        file_name="outcomes.json",
        mime="application/json",
        data=outcomes_json,
    )
    st.json(outcomes_json)
