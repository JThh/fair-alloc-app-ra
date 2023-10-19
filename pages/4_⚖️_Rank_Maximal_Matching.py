# Required Libraries
from collections import defaultdict
import base64
from functools import partial
import json
import time
import random
import numpy as np
import pandas as pd
import streamlit as st
import networkz as nx


MIN_AGENTS = 2
MAX_AGENTS = 500
MIN_ITEMS = 1
MAX_ITEMS = 1000

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
        max_rank = m+1
        if n <= old_n and m <= old_m:
            st.session_state.preferences = st.session_state.preferences.iloc[:n, :m]
            return st.session_state.preferences
        elif n > old_n:
            st.session_state.preferences = pd.concat([st.session_state.preferences,
                                                      pd.DataFrame(np.random.randint(1, max_rank,(n - old_n,m) ),
                                                                   columns=[
                                                          f"Item {i+1}" for i in range(m)],
                                                          index=[f"Agent {i+1}" for i in range(old_n, n)])],
                                                     axis=0)
            return st.session_state.preferences
        elif m > old_m:
            new_items = m - old_m
            population = list(range(1, max_rank))
            random_ranks =[random.sample(population,m) for _ in range(n)]
            st.session_state.preferences = pd.DataFrame(random_ranks,
                                                                   columns=[
                                                          f"Item {i+1}" for i in range(0, m)],
                                                          index=[f"Agent {i+1}" for i in range(n)])
            return st.session_state.preferences
        else:
            st.session_state.preferences = pd.DataFrame(np.random.randint(1, max_rank, (n, m)), columns=[f"Item {i+1}" for i in range(m)],
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
        preferences_default = pd.DataFrame(np.random.randint(1, m+1, (n, m)), 
                                           columns=[f"Item {i+1}" for i in range(m)],
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
        # font-weight: bold;
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
"""

# Add custom CSS style
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Header Message
header_html = """
   <h1 class="header">Fast and Fair Posts Allocation</h1>
"""
st.markdown(header_html, unsafe_allow_html=True)

# Add input components
col1, col2, col3 = st.columns(3)
n = col1.number_input("Number of Agents (n)",
                      min_value=MIN_AGENTS, max_value=MAX_AGENTS, step=1)
m = col2.number_input("Number of Items (m)", min_value=MIN_ITEMS,
                      max_value=MAX_ITEMS, value=MIN_ITEMS, step=1)

upload_preferences = None

# Agent Preferences
st.write("📊 Agent Preferences (0-m, copyable from local sheets):")

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

    def cindex(name: str) -> int:
        return int(name.split()[1])-1

    st.write("🎉 Outcomes:")
    outcomes_list = [[agent, item, preferences[cindex(agent),cindex(item)]]
                      for agent, item in outcomes.items()]
    outcomes_df = pd.DataFrame(outcomes_list, columns=['Agent', 'Item','Rank'])
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
# ..

# Download Outcomes as JSON
# ...

# Community Contribution Guidelines
# ...

# Main function (optional)
# if __name__ == "__main__":
#     main()