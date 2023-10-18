# Required Libraries
from collections import defaultdict
import base64
from functools import partial
import json
import time

import numpy as np
import pandas as pd
import streamlit as st
import networkz as nx

def generate_random_preferences(n, m):
    # Generate a DataFrame with random values between 1 and m
    random_data = np.random.randint(1, m + 1, size=(n, m))
    preferences = pd.DataFrame(random_data, columns=[f"Item {i+1}" for i in range(m)],
                               index=[f"Agent {i+1}" for i in range(n)])
    
    return preferences

# Load Preferences
def load_preferences(m, n, upload_preferences):
    if hasattr(st.session_state, "preferences"):
        if upload_preferences:
            preferences_default = None
            # Load the user-uploaded preferences file
            try:
                preferences_default = pd.read_csv(
                    upload_preferences, index_col=0)
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
        elif n > old_n:
            new_rows = n - old_n
            new_data = generate_random_preferences(new_rows, old_m)
            st.session_state.preferences = pd.concat([st.session_state.preferences, new_data], axis=0)
        elif m > old_m:
            new_cols = m - old_m
            new_data = generate_random_preferences(old_n, new_cols)
            st.session_state.preferences = pd.concat([st.session_state.preferences, new_data], axis=1)
        else:
            st.session_state.preferences = generate_random_preferences(n, m)

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
        preferences_default = pd.DataFrame(np.random.randint(1, 10, (n, m)), columns=[f"Item {i+1}" for i in range(m)],
                                           index=[f"Agent {i+1}" for i in range(n)])
    st.session_state.preferences = preferences_default
    return st.session_state.preferences


# Preference Change Callback: used in Streamlit widget on_click / on_change
def preference_change_callback(preferences):
    for col in preferences.columns:
        preferences[col] = preferences[col].apply(
            lambda x: int(float(x)))
    st.session_state.preferences = preferences

# Algorithm Implementation
def algorithm(m, n, preferences):
    ranker_list =[f"Agent {i+1}" for i in range(n)]
    print('ranker list:', ranker_list)
    G = nx.Graph()
    for i in range(n):
        for j in range(m):
            print(preferences[i,j])            
            rank = preferences[i,j]
            #if rank!= '' and type(rank) == int and rank>= 0:
            G.add_edge(f"Agent {i+1}", f"Item {j+1}", rank=rank)
    M = nx.rank_maximal_matching(G=G, top_nodes=ranker_list, rank='rank')
    half_length = len(M) // 2
    half_M = {k: M[k] for k in list(M)[:half_length]}
    return half_M
    
   

# Checker Function for Algorithm
def algorithm_checker(outcomes, x, m, n, preferences):
    # Function to check the outcomes of the algorithm
    # ...
    pass

# Set page configuration
st.set_page_config(
    page_title="Rank Maximal Matching App",
    page_icon="⚖️",
    layout="wide",
)

# Custom CSS styles
css = """
    /* Insert your custom CSS styles here */
    body {
        font-family: Arial, sans-serif;
        margin: 0;
        padding: 0;
        background-color: #f1f1f1;
    }
    
    .header {
        padding: 20px;
        background-color: #fff;
        text-align: center;
    }
    
    .title {
        font-size: 28px;
        color: #333;
        margin-bottom: 20px;
    }
    
    .content {
        display: flex;
        flex-direction: row;
        align-items: flex-start;
    }
    
    .sidebar {
        flex: 0 0 20%;
        padding: 20px;
        background-color: #fff;
        margin-right: 20px;
    }
    
    .main {
        flex: 1;
        padding: 20px;
        background-color: #fff;
    }
    
    .section {
        margin-bottom: 20px;
    }
    
    .section-title {
        font-size: 20px;
        color: #333;
        margin-bottom: 10px;
    }
    
    .section-content {
        font-size: 16px;
        color: #666;
    }
    
    .button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 16px;
    }
    
    .button:hover {
        background-color: #45a049;
    }
"""

# Set the title and layout of the web application
st.title("Rank Maximal Matching App")

# Add custom CSS style
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Header Message
header_html = """
    <div class="header">
        <h1 class="title">Rank Maximal Matching App</h1>
        <p>This app is based on a Rank Maximal Matching algorithm for fair resource allocation.</p>
        <p>Reference: [Insert Reference Here]</p>
        <p>Credit: [Insert Credit Here]</p>
    </div>
"""
st.markdown(header_html, unsafe_allow_html=True)
MIN_AGENTS = 2
MAX_AGENTS = 500
MIN_ITEMS = 1
MAX_ITEMS = 1000
# Add input components
col1, col2, col3 = st.columns(3)
n = col1.number_input("Number of Agents (n)",
                      min_value=MIN_AGENTS, max_value=MAX_AGENTS, step=1)
m = col2.number_input("Number of Items (m)", min_value=MIN_ITEMS,
                      max_value=MAX_ITEMS, value=MIN_ITEMS, step=1)

upload_preferences = None

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
                                      validate=r"^(?:10|[1-9]\d{0,2}|0)$",
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
                                  preference_change_callback, preferences),
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


start_algo = st.button("⏳ Run Rank Maximal Matching Algorithm ")
if start_algo:
    with st.spinner('Executing...'):
        if n * m * 0.01 > 3:
            time.sleep(2)
        else:
            time.sleep(n * m * 0.01)

    start_time = time.time()
    outcomes = algorithm(m, n, preferences)
    end_time = time.time()
    elapsed_time = end_time - start_time

    st.write("🎉 Outcomes:")
    outcomes_list = [[key, value] for key, value in outcomes.items()]
    print('out list:' ,outcomes_list)
    outcomes_df = pd.DataFrame(outcomes_list, columns=['Agent', 'Items'])
    # Sort the table
    outcomes_df = outcomes_df.sort_values(['Agent'],
                                         )

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
# Sidebar
st.sidebar.title("User Preferences")
# ...

# Main Content
st.header("Algorithm")
# ...

# Run Algorithm Button
if st.button("Run Algorithm"):
    # Implementation of run algorithm button
    # ...
    pass

# Download Outcomes as JSON
# ...

# Community Contribution Guidelines
# ...

# Main function (optional)
# if __name__ == "__main__":
#     main()