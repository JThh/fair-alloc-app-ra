from collections import defaultdict
import base64
from functools import partial
import json
import random
import time

import numpy as np
import pandas as pd
from pandas import Index
import streamlit as st


# Set page configuration
st.set_page_config(
    page_title="House Assignment App",
    page_icon="🏠",
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


def compute_envyfree_assignment(n, m, orderings):
    M = list(range(m))
    while n <= len(M):
        edges_nm = defaultdict(list)
        edges_mn = defaultdict(list)
        for m_ in M:
            for i in range(n):
                if orderings[i][m_] == min(orderings[i][M]):
                    edges_nm[i].append(m_)
                    edges_mn[m_].append(i)
        
        # Check if n-saturating matching exists
        matching = dict()
        houses_left = set(range(m))
        for house, agents in edges_mn.items():
            if len(agents) == 1 and agents[0] not in matching:
                matching[agents[0]] = house
                houses_left.remove(house)
            else:
                for agent in agents:
                    if agent not in matching and len(edges_nm[agent]) == 1:
                        matching[agent] = house
                        houses_left.remove(house)
                        break

        for agent, houses in edges_nm.items():
            if agent in matching:
                continue
            for house in houses:
                if house in houses_left:
                    matching[agent] = house
                    houses_left.remove(house)
                    break
                
        if len(matching.keys()) == n:
            return matching, True
        
        # No n-saturating match exists
        X_u = set(range(n)) - set(matching.keys())
        assert len(X_u) > 0, "Unmatched vertices in X should not be empty"
        
        edges = defaultdict(list)
        for agent, houses in edges_nm.items():
            for house in houses:
                edges[agent].append(house + n)
            
        for agent, house in matching.items():
            edges[house + n].append(agent)
            
        Z = set()
        rand_v = list(X_u)[0]
        
        def DFS(vert, Z):
            Z.add(vert)
            for nbr in edges[vert]:
                if nbr not in Z:
                    DFS(nbr, Z)
        DFS(rand_v, Z)
        for z in Z:
            if z < n and z in edges_nm:
                for house in edges_nm[z]:
                    if house in M:
                        M.remove(house)

    return matching, False


def restore_orderings(orderings):
    orderings = orderings.T
    def apply_list(arr: list):
        indices = sorted(range(len(arr)), key=lambda i: (arr[i], i))
        new_ranks = arr.copy()
        cur_rank = 1
        for ind in indices:
            new_ranks[ind] = cur_rank
            cur_rank += 1
        for i in range(1, len(arr)):
            if arr[indices[i]] == arr[indices[i - 1]]:
                new_ranks[indices[i]] = new_ranks[indices[i - 1]]
        return new_ranks
    for col in orderings.columns:
        orderings[col] = apply_list(orderings[col].tolist())
    return orderings.T


def load_orderings(n, m, shuffle=False):
    def generate_random_orderings():
        # n_fav = np.random.randint(0, m//2.5)
        # return pd.DataFrame(np.asarray([sorted(np.concatenate([np.ones(n_fav),np.arange(n_fav+1, m+1, dtype=int)], axis=0), 
        #                                        key=lambda _:random.random()) for _ in range(n)]),
        rankings = pd.DataFrame(np.asarray([sorted(np.random.randint(1, m+1, size=(m,)), 
                                               key=lambda _:random.random()) for _ in range(n)]),
                                columns=[
                                    f"House {i+1}" for i in range(m)],
                                index=[f"Agent {i+1}" for i in range(n)],
                                dtype=int)
        return restore_orderings(rankings)
    if shuffle:
        return generate_random_orderings()
    if hasattr(st.session_state, "orderings"):
        old_n = st.session_state.orderings.shape[0]
        old_m = st.session_state.orderings.shape[1]
        if n <= old_n and m <= old_m:
            orderings = st.session_state.orderings.iloc[:n, :m]
            return restore_orderings(orderings)
        elif n > old_n:
            orderings = pd.concat([st.session_state.orderings,
                                  pd.DataFrame(
                                      np.asarray([sorted(np.arange(1, m+1, dtype=int), key=lambda _:random.random()) for _ in range(n - old_n)]),
                                      columns=[
                                      f"House {i+1}" for i in range(m)],
                                      index=[
                                          f"Agent {i+1}" for i in range(old_n, n)],
                                      dtype=int)],
                                 axis=0)
            return restore_orderings(orderings)
        elif m > old_m:
            orderings = pd.concat([st.session_state.orderings,
                                  pd.DataFrame(np.tile(
                                      np.arange(old_m+1, m+1, dtype=int), (n, 1)),
                                      columns=[
                                      f"House {i+1}" for i in range(old_m, m)],
                                      index=[
                                          f"Agent {i+1}" for i in range(n)],
                                      dtype=int)],
                                 axis=1)
            return restore_orderings(orderings)
        else:
            return generate_random_orderings()
    return generate_random_orderings()


def ochange_callback(orderings):
    for col in orderings.columns:
        orderings[col] = orderings[col].map(lambda x: int(float(x)))
    st.session_state.orderings = restore_orderings(orderings)


# Set the title and layout of the web application
st.markdown('<h1 class="header">Fast and Fair House Assignment</h1>',
            unsafe_allow_html=True)

# Insert header image
st.sidebar.image("houses.png", use_column_width=True,
                 caption='Image Credit: Freepik.com')

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
    <p>This app calculates outcomes using the N-Saturating Matching algorithm.</p>

    <h3 style="color: {theme_colors['guide-color']};">Follow these steps to use the app:</h3>

    <ol>
        <li>Specify the number of Agents (n) and Houses (m) using the number input boxes.</li>
        <li>Enter the Agent preferences towards houses (as rankings) in the table.</li>
        <li>Click the "Run Algorithm" button to get the assignment outcome.</li>
    </ol>

    <p><em><strong>Disclaimer:</strong> The generated outcomes are for demonstration purposes only and may not reflect real-world scenarios.</em></p>

    <p><em>Image Credit: <a href="https://www.freepik.com/free-vector/real-estate-agent-offering-house-young-family-couple-wife-husband-choosing-new-suburb-home-living_24023256.htm">Image Source</a></em>.
    </div>
    """,
    unsafe_allow_html=True
)

# Add input components
col1, col2, col3, col4 = st.columns([0.35,0.01,0.35,0.35])
n = col1.number_input("Number of Agents (n)",
                      min_value=2, max_value=100, step=1)
m = col3.number_input("Number of Houses (m)", min_value=2,
                      max_value=1000, value=2, step=1)
if m < n:
    st.error("Number of Houses (m) must be greater than or equal to Number of Agents (n). Please adjust the values.")

ordinal = lambda n: "%s" % ("tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])    

st.markdown(
    f"🌟 Agent Preferences towards Houses (ranks from {1}<sup>st</sup> to {m}<sup>{ordinal(m)}</sup> with ties permitted):", unsafe_allow_html=True)

shuffle = st.button('Shuffle Rankings')

with st.spinner("Loading..."):
    orderings = load_orderings(n, m, shuffle)
    st.session_state.orderings = orderings
    for col in orderings.columns:
        orderings[col] = orderings[col].map(str)

edited_ws = st.data_editor(orderings,
                        key="ranking_editor",
                        column_config={
                            f"House {i}": st.column_config.TextColumn(
                                f"House {i}",
                                help=f"Agent's orderings for House {i}",
                                max_chars=4,
                                validate=r'^(?:100|[1-9]\d?|0)$',
                                required=True,
                            )
                            for i in range(1, m+1)
                        }
                        |
                        {
                            "_index": st.column_config.Column(
                                "💡 Hint",
                                help="You may set arbitrary values. We will reconcile the ranks upon algorithmic runs.",
                                disabled=True,
                            ),
                        },
                        on_change=partial(ochange_callback, orderings),
                        )
with st.spinner("Updating..."):
    for col in edited_ws.columns:
        edited_ws[col] = edited_ws[col].map(lambda x: int(float(x)))
    st.session_state.orderings = restore_orderings(edited_ws)

st.markdown(
        f"Colored Ranking Table (Preview):", unsafe_allow_html=True)
    
orderings = st.session_state.orderings
# Define formatter function
def format_cell_color(val):
    max_val = orderings.values.astype(np.int32).max()
    min_val = orderings.values.astype(np.int32).min()
    span = max_val - min_val + 1
    cell_val = (max_val - int(float(val))) / span  # Normalize value between 0 and 1
    thickness = int(10 * cell_val)  # Adjust thickness as per preference
    color = f'rgba(67, 147, 195, {cell_val})'  # Blue color with alpha value based on normalized value
    style = f'background-color: {color}; border-bottom: {thickness}px solid {color}'
    return style

with st.spinner("Loading Table..."):
    st.dataframe(orderings.style.applymap(format_cell_color))

orderings = orderings.to_numpy()

# Download orderings as CSV
orderings_csv = edited_ws.to_csv()
b64 = base64.b64encode(orderings_csv.encode()).decode()
href = f'<a href="data:file/csv;base64,{b64}" download="orderings.csv">Download Rankings CSV</a>'
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
            font-size: 0.9em;
            vertical-align: middle;
        }
        </style>
        <div class="information-card-content">
            <h2 class="information-card-header">Envy-Freeness in House Allocation Problems</h2>
            <h3 class="information-card-header">Introduction</h3>
            <p class="information-card-text">
                The house allocation problem involves assigning a set of houses to a set of agents based on their preferences, while maintaining certain fairness conditions. Envy-freeness is a well-established fairness notion, where each agent prefers their assigned house at least as much as any other assigned house. This problem becomes more complex when the number of houses exceeds the number of agents. The goal is to determine whether an envy-free assignment is possible and, if so, compute one.
            </p>
            <h3 class="information-card-header">Preliminaries</h3>
            <p class="information-card-text">
                In the house allocation problem, we consider a bipartite graph G = (X, Y, E) with bipartite vertex sets X and Y, representing agents and houses respectively. An X-saturating matching is a matching that covers every vertex in X. A set Z ⊆ X is called a Hall violator if |Z| > |S(Z)|, where S(V) denotes the set of vertices adjacent to at least one vertex in V. 
            </p>
            <h3 class="information-card-header">Algorithm Description</h3>
            <p class="information-card-text">
                The algorithm for computing an envy-free assignment in the house allocation problem is as follows:
            </p>
            <div class="information-card-text" style="background-color: #F7F7F7; padding: 10px;">
                <p class="information-card-text">Algorithm 1: Algorithm for Computing an Envy-Free Assignment</p>
                <p class="information-card-text">
                    &nbsp;&nbsp;M' ← M<br>
                    &nbsp;&nbsp;while |N| ≤ |M'|<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Construct a bipartite graph G = (N, M', E) where there is an edge from an agent to a house if and only if the house is among the most preferred houses in M' for the agent<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if there exists an N-saturating matching then<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return the corresponding assignment<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;else<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Find a minimal Hall violator Z ⊆ N<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Remove all houses adjacent to Z in G from M'<br>
                    &nbsp;&nbsp;return null
                </p>
            </div>
            <p class="information-card-text">
                The algorithm first initializes a set M' with all houses, and then iteratively checks if there exists an N-saturating matching in a bipartite graph constructed based on the preferences of the agents. If such a matching exists, it returns the corresponding assignment, which is an envy-free allocation. Otherwise, it finds a minimal Hall violator, removes all houses adjacent to the violator, and repeats the process until either an envy-free assignment is found or no more houses can be assigned. The algorithm terminates in polynomial time since each iteration reduces the size of M' or finds an envy-free assignment.
            </p>
            <p class="information-card-text">
                For a detailed characterization of the algorithm for envy-free house allocation, please refer to the following paper:
            </p>
            <p class="information-card-citation">
                Jiarui Gan, Warut Suksompong and Alexandros A. Voudouris. <a href="https://arxiv.org/pdf/1905.00468.pdf" target="_blank">Envy-Freeness in House Allocation Problems.</a> arXiv preprint arXiv:1905.00468, 2019
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


start_algo = st.button("⏳ Run Fair Assignment Algorithm ")
if start_algo:
    with st.spinner('Executing...'):
        if n * m * 0.01 > 3:
            time.sleep(2)
        else:
            time.sleep(n * m * 0.01)

    start_time = time.time()
    outcomes, flag = compute_envyfree_assignment(n, m, orderings)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    outcomes_list = [[key, value] for key, value in outcomes.items()]
    outcomes_df = pd.DataFrame(outcomes_list, columns=['Agent', 'House'])
    outcomes_df += 1
    outcomes_df['Agent'] = outcomes_df['Agent'].apply(str)
    outcomes_df['House'] = outcomes_df['House'].apply(str)

    # Sort the table
    outcomes_df = outcomes_df.sort_values(['Agent'],
                                        key=lambda col: col.astype(int))
    
    if not flag:
        st.warning("No envy-free allocation found!", icon="⚠️")
        if not outcomes_df.empty:
            st.write("Maximum Matching Outcomes:")
            # Define formatter function
            def format_cell_color(val):
                color = f'rgba(211, 211, 211, 0.3)'  # Blue color with alpha value based on normalized value
                style = f'background-color: {color}; border-bottom: {1}px solid {color}'
                return style
            
            st.data_editor(outcomes_df.style.applymap(format_cell_color),
                            column_config={
                                "Agent": st.column_config.NumberColumn(
                                    "Agent",
                                    help="The list of Agent that get matched",
                                    step=1,
                                ),
                                "House": st.column_config.ListColumn(
                                    "House",
                                    help="The House allocated to an Agent",
                                ),
                            },
                            hide_index=True,
                            disabled=True,
                            )
        else:
            st.write("No houses get allocated in the end.")
        
        output_str = '<h3 class="information-card-header">Not Fulfilling Envy-Freeness</h3>\n\n'
        
        u_agents = sorted(list(set(range(n)) - set(outcomes.keys())))
        u_houses = sorted(list(set(range(m)) - set(outcomes.values())))
        
        for ua in u_agents:
            output_str += f"Agent {ua+1} gets unallocated.\n\n"
            for uh in u_houses:
                output_str += f"**If it gets allocated House {uh+1} ranked at {orderings[ua][uh]}<sup>{ordinal(orderings[ua][uh])}</sup>**, "
                for (a, h) in outcomes.items():
                    if orderings[a][h] > orderings[a][uh]:
                        output_str += f"Agent {a+1} will envy it as Agent {a+1} ranks House {uh+1} \
                            at {orderings[a][uh]}<sup>{ordinal(orderings[a][uh])}</sup> and its current house \
                                at {orderings[a][h]}<sup>{ordinal(orderings[a][h])}</sup>, "
                                
                    if orderings[ua][h] < orderings[ua][uh]:
                        output_str += f"it will envy Agent {a+1} as it ranks House {h+1} \
                            at {orderings[ua][h]}<sup>{ordinal(orderings[ua][h])}</sup>, "
                                
                output_str += "and hence, it does not constitute an envy-free allocation.\n\n"
        
        with st.expander(f"Reasons for Failures", expanded=True):
            st.markdown(output_str, unsafe_allow_html=True)

    else:
        st.write("🎉 Outcomes:")

        st.data_editor(outcomes_df,
                    column_config={
                        "Agent": st.column_config.NumberColumn(
                            "Agent",
                            help="The list of Agent that get matched",
                            step=1,
                        ),
                        "House": st.column_config.ListColumn(
                            "House",
                            help="The House allocated to an Agent",
                        ),
                    },
                    hide_index=True,
                    disabled=True,
                    )

        # Print timing results
        st.write(f"⏱️ Timing Results:")
        st.write(f"Elapsed Time: {elapsed_time:.4f} seconds")
    
        output_str = '<h3 class="information-card-header">Envy-Freeness</h3>\n\n'
        has_lead_str = False
        
        for i in range(n):
            if not has_lead_str:
                b = outcomes[i]
                output_str += f"**Agent {i+1}** has received House {outcomes[i]+1} ranked at {orderings[i][outcomes[i]]}<sup>{ordinal(orderings[i][outcomes[i]])}</sup>.\n\n"
                has_lead_str = True
            for j in range(n):
                if i == j:
                    continue
                else:
                    bi, bj = outcomes[i], outcomes[j]
                    output_str += f"Agent {i+1} ranks Agent {j+1}'s House {bj+1} at {orderings[i][bj]}<sup>{ordinal(orderings[i][bj])}</sup>, so it does not envy Agent {j+1} as rank {orderings[i][bj]}<sup>{ordinal(orderings[i][bj])}</sup> is lower than or equal to rank {orderings[i][bi]}<sup>{ordinal(orderings[i][bi])}</sup>.\n\n"
                    
            has_lead_str = False
            
        with st.expander(f"Explanations of Outcomes (**about {n**2} lines**)", expanded=False):
            st.download_button('Download Full Explanations', output_str,
                               file_name=f"{n}_Agents_{m}_Houses_assign_expl.txt")
            st.markdown(output_str, unsafe_allow_html=True)

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
