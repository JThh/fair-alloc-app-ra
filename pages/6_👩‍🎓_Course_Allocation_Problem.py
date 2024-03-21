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
import fairpyx

#--- Settings ---#
MIN_AGENTS = 2
MAX_AGENTS = 500
MIN_ITEMS = 3
MAX_ITEMS = 100
MAX_POINTS = 1000


#--- Page elements ---#

# Set page configuration
st.set_page_config(
    page_title="Course Allocation Problem App",
    page_icon="👩‍🎓",
    layout="wide",
)

# Set page style
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

# Page title
st.markdown('<h1 class="header">Fast and Efficient Matching</h1>',
            unsafe_allow_html=True)

# Page sidebar - User guide
# Insert header image
st.sidebar.image("./resource/students.jpg", use_column_width=True,)

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
        <li>Choose to either upload a items_capacities/agents_capacities/preferences file or edit the items_capacities/agents_capacities/preferences.</li>
        <li>Specify whether the algorithm uses compensation. </li>
        <li>Click the 'Run Algorithm' button to start the algorithm.</li>
        <li>You can download the outcomes as a CSV file using the provided links.</li>
    </ol>

    <p><em><strong>Disclaimer:</strong> The generated outcomes are for demonstration purposes only and may not reflect real-world scenarios.</em></p>

    <p><em>Image Credit: <a href="https://www.freepik.com/free-vector/intelligent-pupils-studying-classroom_9649994.htm#fromView=search&page=1&position=41&uuid=820ff6c0-3cb7-413c-ae4d-859e748356dc">Image Source</a></em>.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Divide the page to 3 columns.
col1, col2, col3 = st.columns(3)

#--- Input components ---#
# n agents and m items
n = col1.number_input("Number of Agents (n)",
                      min_value=MIN_AGENTS, max_value=MAX_AGENTS, step=1)
m = col2.number_input("Number of Items (m)", min_value=MIN_ITEMS,
                      max_value=MAX_ITEMS, value=MIN_ITEMS, step=1)


# Upload input as csv file buttons
upload_preferences = None
upload_items_capacities = None
upload_agents_capacities = None

# Locate the upload buttons
with col1:
    if st.checkbox("⭐ Upload Local Items Capacities CSV"):
        upload_items_capacities = st.file_uploader(
            f"Upload Items Capacities of shape ({m}, {1})", type=['csv']) 
    if st.checkbox("⭐ Upload Local Agents Capacities CSV"):
        upload_agents_capacities = st.file_uploader(
            f"Upload Agents Capacities of shape ({m}, {1})", type=['csv'])       
with col2:
    if st.checkbox("⭐ Upload Local Preferences CSV"):
        upload_preferences = st.file_uploader(
            f"Upload Preferences of shape ({m}, {1})", type=['csv'])

# Shuffle data button
shuffle = st.button('Shuffle All Data')

# Table Change Callback: used in Streamlit widget on_click / on_change
def change_callback(table):
    for col in table.columns:
        table[col] = table[col].apply(
            lambda x: int(float(x)))
    return table
        
#--- Items items_capacities ---#
st.write("📊 Items Capacities (10-100, copyable from local sheets):")

# Load Items Capacities - handle table initialization and changes
def load_items_capacities(m, upload_items_capacities = False, shuffle = False):
    MAX_CAPACITY = 100
    MIN_CAPACITY = 10

    if hasattr(st.session_state, "items_capacities"): # if items_capacities table is exist
        if upload_items_capacities:                   # if user clicked on upload button
            items_capacities_default = None
            # Load the user-uploaded items_capacities file
            try:
                items_capacities_default = pd.read_csv(
                    upload_items_capacities, index_col=0)
                
                if items_capacities_default.shape != (m,1): # if size doesn't match the input
                    x, y = items_capacities_default.shape
                    st.session_state.items_capacities.iloc[:x,
                                                      :y] = items_capacities_default
                else:
                    st.session_state.items_capacities = pd.DataFrame(items_capacities_default,
                                                                columns=st.session_state.items_capacities.columns,
                                                                index=st.session_state.items_capacities.index)
                return st.session_state.items_capacities
            
            except Exception as e:
                st.error(f"An error occurred while loading the items capacities file.")
                logging.debug("file uploading error: ", e)
                st.stop()
                
        old_m = st.session_state.items_capacities.shape[0]     # the previous number of items (before changes)
        
        if shuffle:
            # Create m random values in range (min-max)
            random_ranks = np.random.randint(MIN_CAPACITY, MAX_CAPACITY,(m))
            # Apply the random ranks to the items_capacities table
            st.session_state.items_capacities = pd.DataFrame(random_ranks,
                                                                   columns=[
                                                          "Item Capacity"],
                                                          index=[f"Item {i+1}" for i in range(m)])
            return st.session_state.items_capacities
        
        # if user decrease the number of items
        if m <= old_m:
            st.session_state.items_capacities = st.session_state.items_capacities.iloc[:m, :1]
            return st.session_state.items_capacities
        # if user increase the number of items
        elif m > old_m:
            st.session_state.items_capacities = pd.concat([st.session_state.items_capacities,
                                                      pd.DataFrame(np.random.randint(MIN_CAPACITY, MAX_CAPACITY, (m - old_m,1)),
                                                                   columns=["Item Capacity"],
                                                          index=[f"Item {i+1}" for i in range(old_m, m)])],
                                                     )
            return st.session_state.items_capacities
    # if the table isn't exist and the user wants to upload a csv file
    if upload_items_capacities:
            items_capacities_default = None
            # Load the user-uploaded items_capacities file
            try:
                items_capacities_default = pd.read_csv(upload_items_capacities)
                if items_capacities_default.shape != (m, 1):
                    st.error(
                        f"The uploaded items capacities file should have a shape of ({m}, {1}).")
                    st.stop()
            except Exception as e:
                st.error("An error occurred while loading the items capacities file.")
                st.stop()
    else:
        # Create m random values in range (min-max) and insert them to a data frame
        items_capacities_default = pd.DataFrame(np.random.randint(MIN_CAPACITY,MAX_CAPACITY, (m)), 
                                        columns=["Item Capacity"],
                                        index=[f"Item {i+1}" for i in range(m)])

    st.session_state.items_capacities = items_capacities_default
    return st.session_state.items_capacities

# Loading the items_capacities table (initial/after changes)
with st.spinner("Loading..."):
    items_capacities=  load_items_capacities(m,upload_items_capacities,shuffle)
    for col in items_capacities.columns:
        items_capacities[col] = items_capacities[col].map(str)

# Items Capacities Change Callback: used in Streamlit widget on_click / on_change
def item_capacity_change_callback(items_capacities):
    st.session_state.items_capacities = change_callback(items_capacities)

# Items Capacities table as editor 
edited_item_capa = st.data_editor(items_capacities,
                              key="item_capa_editor",
                              column_config={
                                  f"Item Capacity": st.column_config.TextColumn(
                                      f"Item Capacity",
                                      help=f"Item Capacity",
                                      max_chars=4,
                                      validate=r"^(?:10|[1-9]\d{0,2}|0)$",
                                      # width='small',  # Set the desired width here
                                      # min_value=0,
                                      # max_value=1000,
                                      # step=1,
                                      # format="%d",
                                      required=True,
                                  )
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
                                  item_capacity_change_callback, items_capacities),
                              )

# Convert the editor changes from str to float
with st.spinner('Updating...'):
    for col in edited_item_capa.columns:
        edited_item_capa[col] = edited_item_capa[col].apply(
            lambda x: int(float(x)))
    st.session_state.items_capacities = edited_item_capa

# Apply the changes
items_capacities = edited_item_capa.values

# Download items_capacities as CSV
items_capacities_csv = edited_item_capa.to_csv()
b64 = base64.b64encode(items_capacities_csv.encode()).decode()
href = f'<a href="data:file/csv;base64,{b64}" download="items_capacities.csv">Download Items Capacities CSV</a>'
st.markdown(href, unsafe_allow_html=True)

#--- Agents Capacities (same as thr items_capacities except the size [n instead of m]) ---#
st.write("📊 Agents Capacities (0-10, copyable from local sheets):")

# Load Agents Capacities 
def load_agents_capacities(n, upload_agents_capacities = False, shuffle = False):
    MAX_CAPACITY = 10
    MIN_CAPACITY = 1
    if hasattr(st.session_state, "agents_capacities"):
        if upload_agents_capacities:
            agents_capacities_default = None
            # Load the user-uploaded agents_capacities file
            try:
                agents_capacities_default = pd.read_csv(
                    upload_agents_capacities, index_col=0)
                if agents_capacities_default.shape != (n,1):
                    x, y = agents_capacities_default.shape
                    st.session_state.agents_capacities.iloc[:x,
                                                      :y] = agents_capacities_default
                else:
                    st.session_state.agents_capacities = pd.DataFrame(agents_capacities_default,
                                                                columns=st.session_state.agents_capacities.columns,
                                                                index=st.session_state.agents_capacities.index)
                return st.session_state.agents_capacities
            except Exception as e:
                st.error(f"An error occurred while loading the agents_capacities file.")
                logging.debug("file uploading error: ", e)
                st.stop()
                
        old_n = st.session_state.agents_capacities.shape[0]     
   
        if shuffle:
            random_ranks = np.random.randint(MIN_CAPACITY, MAX_CAPACITY,(n))
            st.session_state.agents_capacities = pd.DataFrame(random_ranks,
                                                                   columns=[
                                                          "Agent Capacity"],
                                                          index=[f"Agent {i+1}" for i in range(n)])
            return st.session_state.agents_capacities
        
        
        if n <= old_n:
            st.session_state.agents_capacities = st.session_state.agents_capacities.iloc[:n, :1]
            return st.session_state.agents_capacities
        elif n > old_n:
            st.session_state.agents_capacities = pd.concat([st.session_state.agents_capacities,
                                                      pd.DataFrame(np.random.randint(MIN_CAPACITY, MAX_CAPACITY, (n - old_n,1)),
                                                                   columns=[
                                                          "Agent Capacity"],
                                                          index=[f"Agent {i+1}" for i in range(old_n, n)])],
                                                     )
            return st.session_state.agents_capacities
        
    if upload_agents_capacities:
            agents_capacities_default = None
            # Load the user-uploaded agents_capacities file
            try:
                agents_capacities_default = pd.read_csv(upload_agents_capacities)
                if agents_capacities_default.shape != (n, 1):
                    st.error(
                        f"The uploaded agents_capacities file should have a shape of ({n}, {1}).")
                    st.stop()
            except Exception as e:
                st.error("An error occurred while loading the agents capacities file.")
                st.stop()
    else:
        agents_capacities_default = pd.DataFrame(np.random.randint(MIN_CAPACITY,MAX_CAPACITY, (n)), 
                                        columns=["Agent Capacity"],
                                        index=[f"Agent {i+1}" for i in range(n)])
    st.session_state.agents_capacities = agents_capacities_default
    return st.session_state.agents_capacities


with st.spinner("Loading..."):
    agents_capacities=  load_agents_capacities(n,upload_agents_capacities,shuffle)
    for col in agents_capacities.columns:
        agents_capacities[col] = agents_capacities[col].map(str)

def agent_capacity_change_callback(agents_capacities):
    st.session_state.agents_capacities = change_callback(agents_capacities)

edited_agent_capa = st.data_editor(agents_capacities,
                              key="agent_capa_editor",
                              column_config={
                                  f"Capacity": st.column_config.TextColumn(
                                      f"Agent Capacity",
                                      help=f"' ",
                                      max_chars=4,
                                      validate=r"^(?:10|[1-9]\d{0,2}|0)$",
                                      # width='small',  # Set the desired width here
                                      # min_value=0,
                                      # max_value=1000,
                                      # step=1,
                                      # format="%d",
                                      required=True,
                                  )
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
                                  agent_capacity_change_callback, agents_capacities),
                              )

with st.spinner('Updating...'):
    for col in edited_agent_capa.columns:
        edited_agent_capa[col] = edited_agent_capa[col].apply(
            lambda x: int(float(x)))
    st.session_state.agents_capacities = edited_agent_capa

agents_capacities = edited_agent_capa.values

# Download agents_capacities as CSV
agents_capacities_csv = edited_agent_capa.to_csv()
b64 = base64.b64encode(agents_capacities_csv.encode()).decode()
href = f'<a href="data:file/csv;base64,{b64}" download="agents_capacities.csv">Download Agents Capacities CSV</a>'
st.markdown(href, unsafe_allow_html=True)



#--- Preferences ---#

st.write("📊 Agent Preferences (0-100, copyable from local sheets):")

# Helper - generate random values for the preferences table; each row sum is equal to MAXPOINTS.
def generate_random_integers_array(m,n):
    preferences = []
    for i in range(n):
        random_array = np.random.randint(0, 100, m)
        scaled_array = random_array / random_array.sum() * MAX_POINTS
        rounded_array = np.round(scaled_array).astype(int)
        rounded_array[-1] += MAX_POINTS - rounded_array.sum()
        preferences.append(random_array)
    return preferences

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
                logging.debug("file uploading error: ", e)
                st.stop()
                
        old_n = st.session_state.preferences.shape[0] # the previous number of agents
        old_m = st.session_state.preferences.shape[1] # the previous number of items
   
        if shuffle: # shuffle button clicked
            random_ranks = generate_random_integers_array(m,n)
            st.session_state.preferences = pd.DataFrame(random_ranks,
                                                                   columns=[
                                                          f"Item {i+1}" for i in range(m)],
                                                          index=[f"Agent {i+1}" for i in range(n)])
            return st.session_state.preferences
        
        # if n or m are decreased
        if n <= old_n and m <= old_m:
            st.session_state.preferences = st.session_state.preferences.iloc[:n, :m]
            return st.session_state.preferences
        # if user increase n
        elif n > old_n:
            # add one more row to preferences table
            st.session_state.preferences = pd.concat([st.session_state.preferences,
                                                      pd.DataFrame(generate_random_integers_array(m,n - old_n),
                                                                   columns=[
                                                          f"Item {i+1}" for i in range(m)],
                                                          index=[f"Agent {i+1}" for i in range(old_n, n)])],
                                                     axis=0)
            return st.session_state.preferences
        # if user increase m
        elif m > old_m:
            # add one more column to preferences table
            st.session_state.preferences =  pd.concat([st.session_state.preferences,
                                                      pd.DataFrame(np.random.randint(1,MAX_POINTS,(n, m - old_m)),
                                                                   columns=[
                                                          f"Item {i+1}" for i in range(old_m,m)],
                                                          index=[f"Agent {i+1}" for i in range(n)])],
                                                     axis=1)
            return st.session_state.preferences
        else:
            random_ranks = generate_random_integers_array(m,n) # generate new random values
            st.session_state.preferences = pd.DataFrame(random_ranks, columns=[f"Item {i+1}" for i in range(m)],
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
        random_ranks = generate_random_integers_array(m,n) # generate new random values
        # apply the random ranks to the table
        preferences_default = pd.DataFrame(random_ranks,
                                                                   columns=[
                                                          f"Item {i+1}" for i in range(m)],
                                                          index=[f"Agent {i+1}" for i in range(n)])
    st.session_state.preferences = preferences_default
    return st.session_state.preferences

with st.spinner("Loading..."):
    preferences = load_preferences(m, n, shuffle=shuffle)
    for col in preferences.columns:
        preferences[col] = preferences[col].map(str)

preferences = load_preferences(m, n, upload_preferences)
for col in preferences.columns:
    preferences[col] = preferences[col].map(str)

def preference_change_callback(preferences):
    st.session_state.preferences = change_callback(preferences)

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

# compensation button - optional for this algorithm
compensation = st.checkbox("Use compensation")

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
            <h2 class="information-card-header">Iterated Maximum Matching with Compensation</h2>
            <h3 class="information-card-header">Algorithm Overview</h3>
            <p class="information-card-text">
                <div> The algorithm proceeds in rounds. </div>
                <div> In each round, the algorithm finds a maximum-weight matching between the agents with remaining items_capacities, and the items with remaining items_capacities, and allocates each matched items to its matched agent. </div>
                <div> If an agent does not win their maximum match in the present round, the difference between the maximum and the actual match are moved to the next-best option as a compensation, to increase their chances to win it in the next rounds. </div>
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
                The Iterated Maximum Matching idea (without the compensation) is Algorithm 1 in the following paper:
            </p>
            <p class="information-card-citation">
                - Johannes Brustle, Jack Dippel, Vishnu V. Narayan, Mashbat Suzuki, Adrian Vetta (2020) <br/>
                - "One Dollar Each Eliminates Envy" <br/>
                - Proceedings of the 21st ACM Conference on Economics and Computation. 2020 <br/>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# Running Algorithm

# Algorithm Implementation
def algorithm(m, n, items_capacities, agents_capacities, preferences, compensation=False):
    pref_dict = {}
    capa_dict = {}
    req_dict = {}
    agents_conflicts = {}
    items_conflicts = {}

    for i in range(n):
        pref_dict[f"Agent {i+1}"] = {}
        req_dict[f"Agent {i+1}"] = agents_capacities[i,0]
        agents_conflicts[f"Agent {i+1}"] = {}
        for j in range(m):
            pref_dict[f"Agent {i+1}"][f"Item {j+1}"] = preferences[i,j]
    for i in range(m):
        capa_dict[f"Item {i+1}"] = items_capacities[i,0]
        items_conflicts[f"Item {i+1}"] = {}
        
    instance = fairpyx.Instance(
        agent_capacities=req_dict, 
        valuations=pref_dict,
        item_capacities=capa_dict,
        item_conflicts=items_conflicts,
        agent_conflicts=agents_conflicts,
        )

    algorithm = fairpyx.algorithms.iterated_maximum_matching
    # string_explanation_logger = fairpyx.StringsExplanationLogger(instance.agents)
    string_explanation_logger = fairpyx.StringsExplanationLogger({
        agent for agent in instance.agents
    },language='en', mode='w', encoding="utf-8")
    
    allocation = fairpyx.divide(algorithm=algorithm, instance=instance, explanation_logger=string_explanation_logger, adjust_utilities=compensation)
    return allocation,string_explanation_logger, instance

# Checker Function for Algorithm
def algorithm_checker(instance,allocation):
    matrix:fairpyx.AgentBundleValueMatrix = fairpyx.AgentBundleValueMatrix(instance, allocation)
    result_vector = [["Utilitarian value", matrix.utilitarian_value()],
                      ["Egalitarian value", matrix.egalitarian_value()],
                       ["Max envy", matrix.max_envy()],
                         ["Mean envy", matrix.mean_envy()]]
    logging.debug('Result Vector:', result_vector)
    return result_vector


start_algo = st.button("⏳ Run Iterated Maximum Matching Algorithm")
if start_algo:
    with st.spinner('Executing...'):
        if n * m * 0.01 > 3:
            time.sleep(2)
        else:
            time.sleep(n * m * 0.01)

    start_time = time.time()
    outcomes,explanations, instance = algorithm(m, n, items_capacities,agents_capacities,preferences, compensation)
    end_time = time.time()
    elapsed_time = end_time - start_time
    st.write("🎉 Outcomes:")
    outcomes_list = []
    for i in range(n):
        outcomes_items = outcomes[f"Agent {i+1}"]
        outcomes_str = ", ".join([items for items in outcomes_items])
        outcomes_list.append([f"Agent {i+1}"]+[outcomes_str]+[explanations.agent_string(f'Agent {i+1}')])
    items_head = ['Items', 'Explanation']
    outcomes_df = pd.DataFrame(outcomes_list, columns=['Agent']+items_head)


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

    vector = algorithm_checker(instance,outcomes)
    vector_df = pd.DataFrame(vector, columns=['Parameter', 'Value'])
    st.data_editor(vector_df,
                   column_config={
                       "Parameters": st.column_config.NumberColumn(
                           "Parameter",
                           help="The desired parameter to calculate",
                           step=1,
                       ),
                       "Values": st.column_config.ListColumn(
                           "value",
                           help="The parameter value",
                       ),
                   },
                   hide_index=True,
                   disabled=True,
                   )

    # Print timing results
    st.write(f"⏱️ Timing Results:")
    st.write(f"Elapsed Time: {elapsed_time:.4f} seconds")
