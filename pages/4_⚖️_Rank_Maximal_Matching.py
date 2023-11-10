import logging

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



# Transform agent/item names into their corresponding index positions.
def pindex(name: str) -> int:
    return int(name.split()[1])-1

def get_rank(agent,item):
    return preferences[pindex(agent),pindex(item)]



# Load Preferences
def load_preferences(m, n, upload_preferences = False, shuffle = False):
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
                logging.debug("file uploding error: ", e)
                st.stop()
                
        old_n = st.session_state.preferences.shape[0]
        old_m = st.session_state.preferences.shape[1]     
        max_rank = m+1
   
        if shuffle:
            population = list(range(1, max_rank))
            random_ranks =[random.sample(population,m) for _ in range(n)]
            st.session_state.preferences = pd.DataFrame(random_ranks,
                                                                   columns=[
                                                          f"Item {i+1}" for i in range(0, m)],
                                                          index=[f"Agent {i+1}" for i in range(n)])
            print(st.session_state.preferences)
            return st.session_state.preferences
        
        
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
    logging.debug('ranker list:', ranker_list)
    G = nx.Graph()
    for i in range(n):
        for j in range(m):
            rank = preferences[i,j]
            G.add_edge(f"Agent {i+1}", f"Item {j+1}", rank=rank)
    M = nx.rank_maximal_matching(G=G, top_nodes=ranker_list, rank='rank')
    logging.debug("RMM Matching: ", M)
    half_length = len(M) // 2
    half_M = {k: M[k] for k in list(M)[:half_length]}
    return half_M
    
   
# Checker Function for Algorithm - 
def algorithm_checker(outcomes,preferences):
    from collections import Counter
    result_vector = dict(Counter([get_rank(agent,item) for agent, item in outcomes.items()]))
    logging.debug('Result Vector:', result_vector)
    return result_vector

# Set page configuration
st.set_page_config(
    page_title="Rank Maximal Matching App",
    page_icon="⚖️",
    layout="wide",
)

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
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="header">Fast and Efficient Matching</h1>',
            unsafe_allow_html=True)

# Insert header image
st.sidebar.image("./resource/applicants.jpg", use_column_width=True,)

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
    <p>This app calculates outcomes using the Rank Maximal Matching algorithm.</p>

    <h3 style="color: {theme_colors['guide-color']};">Follow these steps to use the app:</h3>

    <ol>
        <li>Specify the number of agents (n) and items (m) using the number input boxes.</li>
        <li>Choose to either upload a preferences file or edit the  preferences.</li>
        <li>Click the 'Run Algorithm' button to start the algorithm.</li>
        <li>You can download the outcomes as a CSV file using the provided links.</li>
    </ol>

    <p><em><strong>Disclaimer:</strong> The generated outcomes are for demonstration purposes only and may not reflect real-world scenarios.</em></p>

    <p><em>Image Credit: <a href="https://www.freepik.com/free-vector/creative-illustration-recruitment-concept_9453228.htm#page=2&position=26&from_view=search&track=ais">Image Source</a></em>.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Add input components
col1, col2, col3 = st.columns(3)
n = col1.number_input("Number of Agents (n)",
                      min_value=MIN_AGENTS, max_value=MAX_AGENTS, step=1)
m = col2.number_input("Number of Items (m)", min_value=MIN_ITEMS,
                      max_value=MAX_ITEMS, value=3, step=1)

upload_preferences = None
with col1:
    if st.checkbox("⭐ Upload Local Preferences CSV"):
        upload_preferences = st.file_uploader(
            f"Upload Preferences of shape ({n}, {m})", type=['csv'])
        
# Agent Preferences
st.write("📊 Agent Preferences (0-m, copyable from local sheets):")

shuffle = st.button('Shuffle Rankings')

with st.spinner("Loading..."):
    preferences = load_preferences(m, n, shuffle=shuffle)
    for col in preferences.columns:
        preferences[col] = preferences[col].map(str)

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
        .compact-expression {
            font-size: 0.8em;
            vertical-align: middle;
        }
        </style>
        <div class="information-card-content">
            <h2 class="information-card-header">Rank Maximal Matching</h2>
            <p class="information-card-text">
                The  Rank Maximal Matching algorithm is used for applicants and posts, where each applicant ranks posts based on their preferences.
                The goal is to allocate the posts to the applicants in a way that maximizes the number of applicants matched to their first choice post,
                and if possible, maximizes the number of applicants matched to their second choice post, and so on.
            </p>
            <h3 class="information-card-header">Algorithm Overview</h3>
            <p class="information-card-text">
                <ul>
                <li>
                    <p class="information-card-text">
                        The algorithm starts with an initial matching, which can be any maximum matching in the graph.
                    </p>
                </li>
                <li>
                    <p class="information-card-text">
                        At each iteration, the algorithm follow the following steps:
                        <div class="information-card-text"> a. Partition the nodes into sets based on the current matching. These sets include even nodes (matched nodes), odd nodes (unmatched nodes), and unreachable nodes. </div>
                        <div class="information-card-text"> b. Delete certain edges from the graph based on the partitioning. Remove edges of rank higher than the current iteration that are incident to odd and unreachable nodes. </div>
                        <div class="information-card-text"> c. Augment the current matching by finding augmenting paths in a suitable subgraph. This step aims to increase the cardinality of the matching while maintaining the rank-maximality property. </div>
                    </p>
                <li> 
                    <p class="information-card-text">
                        Throughout the iterations, certain invariants are maintained: every rank-maximal matching in the current graph has all its edges in the modified graph, and the current matching is a rank-maximal matching.
                    </p>
                </li>
                <li> 
                    <p class="information-card-text">
                        The iterations continue until a stopping condition is met. This can be when the maximum rank is reached, or when the current matching is already a maximum matching in the modified graph. 
                    </p>
                </li>
                </ul>
            </p>
            <!--
            <h3 class="information-card-header">Notation</h3>
            <p class="information-card-text">
                <p>Let A be the set of applicants and P be the set of posts.</p>
                <p>Each edge (a, p) has a rank i, indicating that post p is the i-th choice for applicant a.</p>
                <p>A matching is a set of (applicant, post) pairs where each applicant and post appear in at most one pair.</p>
                <p>A rank-maximal matching aims to maximize the number of applicants matched to their first choice post, followed by their subsequent choices.</p>
            </p>
            <h3 class="information-card-header">Algorithm Description</h3>
            <div class="information-card-text" style="background-color: #F7F7F7; padding: 10px;">
                <p class="information-card-text">Algorithm 1: Algorithm for Computing a Rank-Maximal Matching</p>
                <p class="information-card-text">
                    <p>Given a bipartite graph G = (A, P, E) with preference lists and ranks assigned to each edge.</p>
                    <p>Initialize an empty matching M.</p>
                    <p>Repeat the following steps until there are no more augmenting paths in G:</p>
                    <p>&nbsp;&nbspa. Find an augmenting path in G using a suitable algorithm (e.g., breadth-first search).</p>
                    <p>&nbsp;&nbspb. Augment the matching M along the augmenting path.</p>
                    <p>Return the rank-maximal matching M.</p>
                </p>
            </div>
            -->
            <p class="information-card-text">
                For a detailed explanation of the Rank Maximal Matching algorithm and its theoretical foundations, please refer to the following paper:
            </p>
            <p class="information-card-citation">
                Robert W. Irving, Telikepalli Kavitha, Kurt Mehlhorn, Dimitrios Michail, and Katarzyna Paluch. 2006. 
                <a href="https://d-michail.github.io/assets/papers/RankMaximalMatchings-journal.pdf" target="_blank">Rank-maximal matchings.</a> 
                ACM Transactions on Algorithms (TALG), 2(4), 602-610.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


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

    outcomes_list = [[agent, item, get_rank(agent,item)] 
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

    st.write("🗒️ Outcomes Summary:")

    vector = algorithm_checker(outcomes, preferences)
    vector_list = [[rank, count] for rank,count in vector.items()]
    vector_df = pd.DataFrame(vector_list, columns=['Rank', 'Count'])
    st.data_editor(vector_df,
                   column_config={
                       "Ranks": st.column_config.NumberColumn(
                           "Rank",
                           help="the agent's preference for the item",
                           step=1,
                       ),
                       "Counts": st.column_config.ListColumn(
                           "Count",
                           help="the occurrences count of each preference value",
                       ),
                   },
                   hide_index=True,
                   disabled=True,
                   )

    # Print timing results
    st.write(f"⏱️ Timing Results:")
    st.write(f"Elapsed Time: {elapsed_time:.4f} seconds")
    
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
    <p>Contributed by <a href="https://github.com/oriyalperin" target="_blank">Oriya Alperin</a> and 
    Prof. <a href="http://erelsgl.github.io/" target="_blank">Erel Segal-Halevi</a></p>
    <p>&copy; 2023. All rights reserved.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar
# # ..

# Download Outcomes as JSON
# ...

# Community Contribution Guidelines
# ...

# Main function (optional)
# if __name__ == "__main__":
#     main()
