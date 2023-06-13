from collections import defaultdict
import base64
from functools import partial
import json
import random
import time

import numpy as np
import pandas as pd
import streamlit as st


# Set page configuration
st.set_page_config(
    page_title="Fair Matching App",
    page_icon="🔄",
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


def compute_EF11_ssba(n, m, preferences, ranks):
    Q = list(range(n)) * (m // n) + list(range(n))[:(m % n)]
    assert len(Q) == m
    P = list(range(m))
    match = defaultdict(list)
    for team in range(m):
        players = np.asarray(P)[np.argwhere(preferences[Q[team]][P]
                                            == np.amax(preferences[Q[team]][P])).flatten().tolist()]
        match[team].extend(players.tolist())
        P.remove(players[0])

    rev_match = defaultdict(list)
    for (team, players) in match.items():
        for player in players:
            rev_match[player].append(team)

    final_match = defaultdict(list)
    T = list(range(m))
    for player in rev_match.keys():
        if len(rev_match[player]) == 1:
            team = rev_match[player][0]
            real_team = Q[team]
            final_match[real_team].append(player)
            T.remove(team)
        else:
            continue

    for player in rev_match.keys():
        if len(rev_match[player]) == 1:
            continue
        else:
            teams = rev_match[player]
            real_teams = list(map(lambda x: Q[x], teams))
            real_team = real_teams[np.argmin(ranks[player][real_teams])]
            final_match[real_team].append(player)
            team = teams[np.argmin(ranks[player][real_teams])]
            try:
                T.remove(team)
            except:
                continue

    return final_match


def load_preferences(m, n):
    low = -100
    high = 100

    if hasattr(st.session_state, "prefs"):         
        old_n = st.session_state.prefs.shape[0]
        old_m = st.session_state.prefs.shape[1]
        if n <= old_n and m <= old_m:
            st.session_state.prefs = st.session_state.prefs.iloc[:n, :m]
            return st.session_state.prefs
        elif n > old_n:
            st.session_state.prefs = pd.concat([st.session_state.prefs,
                                                      pd.DataFrame(np.random.randint(low, high, (n - old_n, m)),
                                                                   columns=[
                                                          f"Player {i+1}" for i in range(m)],
                                                          index=[f"Team {i+1}" for i in range(old_n, n)],
                                                          dtype=int)],
                                                     axis=0)
            return st.session_state.prefs
        else:
            st.session_state.prefs = pd.concat([st.session_state.prefs,
                                                      pd.DataFrame(np.random.randint(low, high,(n, m - old_m)),
                                                                   columns=[
                                                          f"Player {i+1}" for i in range(old_m, m)],
                                                          index=[f"Team {i+1}" for i in range(n)],
                                                          dtype=int)],
                                                     axis=1)
            return st.session_state.prefs

    preferences_default = pd.DataFrame(np.random.randint(low, high, (n, m)), columns=[f"Player {i+1}" for i in range(m)],
                                       index=[f"Team {i+1}" for i in range(n)],
                                       dtype=int)
    st.session_state.prefs = preferences_default
    return st.session_state.prefs


def restore_rankings(rankings):
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
    for col in rankings.columns:
        rankings[col] = apply_list(rankings[col].tolist())
    return rankings


def load_rankings(n, m, shuffle=False):
    if shuffle:
        rankings = pd.DataFrame(np.asarray([sorted(np.arange(1, n+1, dtype=int), key=lambda _:random.random()) for _ in range(m)]),
                                index=[
                                    f"Player {i+1}" for i in range(m)],
                                columns=[f"Team {i+1}" for i in range(n)],
                                dtype=int).T
        return rankings
    if hasattr(st.session_state, "rankings"):
        old_n = st.session_state.rankings.shape[0]
        old_m = st.session_state.rankings.shape[1]
        if n <= old_n and m <= old_m:
            rankings = st.session_state.rankings.iloc[:n, :m]
            return restore_rankings(rankings)
        elif n > old_n:
            rankings = pd.concat([st.session_state.rankings,
                                  pd.DataFrame(np.tile(
                                      np.arange(old_n+1, n+1, dtype=int), (m, 1)),
                                      index=[
                                      f"Player {i+1}" for i in range(m)],
                                      columns=[
                                          f"Team {i+1}" for i in range(old_n, n)],
                                      dtype=int).T],
                                 axis=0)
            return restore_rankings(rankings)
        elif m > old_m:
            rankings = pd.concat([st.session_state.rankings,
                                  pd.DataFrame(
                                      np.asarray([sorted(np.arange(1, n+1, dtype=int), key=lambda _:random.random()) for _ in range(m - old_m)]),
                                      index=[
                                      f"Player {i+1}" for i in range(old_m, m)],
                                      columns=[
                                          f"Team {i+1}" for i in range(n)],
                                      dtype=int).T],
                                 axis=1)
            return restore_rankings(rankings)
        else:
            rankings = pd.DataFrame(np.asarray([sorted(np.arange(1, n+1, dtype=int), key=lambda _:random.random()) for _ in range(m)]),
                                    index=[
                                        f"Player {i+1}" for i in range(m)],
                                    columns=[f"Team {i+1}" for i in range(n)],
                                    dtype=int).T
            return rankings
    rankings = pd.DataFrame(np.asarray([sorted(np.arange(1, n+1, dtype=int), key=lambda _:random.random()) for _ in range(m)]),
                            index=[
                                f"Player {i+1}" for i in range(m)],
                            columns=[f"Team {i+1}" for i in range(n)],
                            dtype=int).T
    return rankings


def wchange_callback(rankings):
    for col in rankings.columns:
        rankings[col] = rankings[col].map(lambda x: int(float(x)))
    st.session_state.rankings = restore_rankings(rankings)


def pchange_callback2(preferences):
    for col in preferences.columns:
        preferences[col] = preferences[col].apply(
            lambda x: int(float(x)))
    st.session_state.prefs = preferences


# Set the title and layout of the web application
st.markdown('<h1 class="header">Efficient and Fair Team Formation</h1>',
            unsafe_allow_html=True)

# Insert header image
st.sidebar.image("players.png", use_column_width=True,
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
    <p>This app calculates outcomes using the Polynomial Matching algorithm.</p>

    <h3 style="color: {theme_colors['guide-color']};">Follow these steps to use the app:</h3>

    <ol>
        <li>Specify the number of teams (n) and players (m) using the number input boxes.</li>
        <li>Enter the team and player preferences in the table.</li>
        <li>Click the "Run Algorithm" button to get the matching outcome.</li>
    </ol>

    <p><em><strong>Disclaimer:</strong> The generated outcomes are for demonstration purposes only and may not reflect real-world scenarios.</em></p>

    <p><em>Image Credit: <a href="https://www.freepik.com/premium-vector/soccer-stadium-players-game-with-ball-football-field-struggle-different-teams-sport-match-athletes-competing-world-cup-sportsmen-play-together-vector-championship-concept_21049134.htm">Image Source</a></em>.
    </div>
    """,
    unsafe_allow_html=True
)

# Add input components
col1, col2, col3, col4 = st.columns([0.35,0.01,0.35,0.33])
n = col1.number_input("Number of Teams (n)",
                      min_value=2, max_value=100, step=1)
m = col3.number_input("Number of Players (m)", min_value=2,
                      max_value=1000, value=6, step=1)

tab1, tab2 = st.tabs(["Team Preferences", "Player Preferences"])

with tab1:
    # Agent Preferences
    st.markdown("📊 Team Preferences towards Players (-1000 to 1000):",
                unsafe_allow_html=True)

    preferences = load_preferences(m, n)
    for col in preferences.columns:
        preferences[col] = preferences[col].map(str)

    edited_prefs = st.data_editor(preferences,
                                key="pref_editor2",
                                column_config={
                                    f"Player {j}": st.column_config.TextColumn(
                                        f"Player {j}",
                                        help=f"Teams' Preferences towards Player {j}",
                                        max_chars=5,
                                        validate=r'^-?[1-9][0-9]{0,2}$|^-?1000$|^0$',
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
                                    pchange_callback2, preferences),
                                )
    with st.spinner('Updating...'):
        for col in edited_prefs.columns:
            edited_prefs[col] = edited_prefs[col].apply(
                lambda x: int(float(x)))
        st.session_state.prefs = edited_prefs

    preferences = edited_prefs.values

    # Download preferences as CSV
    preferences_csv = edited_prefs.to_csv()
    b64 = base64.b64encode(preferences_csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="preferences.csv">Download Preferences CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

with tab2:
    st.markdown(
        f"🌟 Player Rankings of Teams ({1} - {n}, Permitting Ties, {1} means the highest rank):", unsafe_allow_html=True)

    shuffle = st.button('Shuffle Rankings')

    with st.spinner("Loading..."):
        rankings = load_rankings(n, m, shuffle)
        st.session_state.rankings = rankings
        for col in rankings.columns:
            rankings[col] = rankings[col].map(str)

    edited_ws = st.data_editor(rankings.T,
                            key="ranking_editor",
                            column_config={
                                f"Team {i}": st.column_config.TextColumn(
                                    f"Team {i}",
                                    help=f"Player's Rankings for Team {i}",
                                    max_chars=4,
                                    validate=r'^(?:100|[1-9]\d?|0)$',
                                    required=True,
                                )
                                for i in range(1, n+1)
                            }
                            |
                            {
                                "_index": st.column_config.Column(
                                    "💡 Hint",
                                    help="You may set arbitrary values. We will reconcile the ranks upon algorithmic runs.",
                                    disabled=True,
                                ),
                            },
                            on_change=partial(wchange_callback, rankings),
                            )
    with st.spinner("Updating..."):
        for col in edited_ws.columns:
            edited_ws[col] = edited_ws[col].map(lambda x: int(float(x)))
        st.session_state.rankings = restore_rankings(edited_ws.T)
    
    st.markdown(
            f"Colored Ranking Table (Preview):", unsafe_allow_html=True)
        
    rankings = st.session_state.rankings
    # Define formatter function
    def format_cell_color(val):
        max_val = rankings.values.astype(np.int32).max()
        min_val = rankings.values.astype(np.int32).min()
        span = max_val - min_val + 1
        cell_val = (max_val - int(float(val))) / span  # Normalize value between 0 and 1
        thickness = int(10 * cell_val)  # Adjust thickness as per preference
        color = f'rgba(67, 147, 195, {cell_val})'  # Blue color with alpha value based on normalized value
        style = f'background-color: {color}; border-bottom: {thickness}px solid {color}'
        return style
    
    with st.spinner("Loading Table..."):
        st.dataframe(rankings.T.style.applymap(format_cell_color))
    
    rankings = rankings.T.to_numpy()

    # Download rankings as CSV
    rankings_csv = edited_ws.to_csv()
    b64 = base64.b64encode(rankings_csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="rankings.csv">Download Rankings CSV</a>'
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
            <h2 class="information-card-header">Fair Division with Two-Sided Preferences</h2>
            <p class="information-card-text">
                This matching algorithm is used for a fair division setting in which a number of players are to be fairly distributed among a
                set of teams. In this setting, not only do the teams have preferences over the players as in the canonical fair division setting, but the players also have
                preferences over the teams. The algorithm can generate an allocation satisfying EF[1,1], balancedness, swap stability, and can compute it in polynomial time, even
                when teams may have positive or negative values for players.
            </p>
            <h3 class="information-card-header">Overview of Notions</h4>
            <h4 class="information-card-header">EF[1,1]</h4>
            <p class="information-card-text">
                For an allocation to satisfy EF[1,1], it must meet the condition that for any two distinct teams, 
                denoted as i and j, there exist subsets X ⊆ A<sub>i</sub> and Y ⊆ A<sub>j</sub>, where |X| and |Y| are at most 1, such that 
                the value v<sub>i</sub>(A<sub>i</sub> \ X) is greater than or equal to v<sub>i</sub>(A<sub>j</sub> \ Y). In other words, each team should have 
                an advantage over the other in terms of the values assigned to their players.
            </p>
            <h4 class="information-card-header">Balancedness</h4>
            <p class="information-card-text">
                An allocation is considered balanced if the absolute difference between the sizes of any two teams, denoted as
                |A<sub>i</sub>| and |A<sub>j</sub>|, is at most 1 for all i and j. This ensures that the teams are relatively equal in terms of the 
                number of players assigned to them.
            </p>
            <h4 class="information-card-header">Swap Stability</h4>
            <p class="information-card-text">
                A swap between players p ∈ A<sub>i</sub> and q ∈ A<sub>j</sub> (where i and j are distinct teams) is deemed beneficial if it improves the 
                situation for at least one of the involved parties without making any of them worse off. An allocation is considered swap stable if it 
                does not allow for any beneficial swaps, ensuring that no team or player can improve their situation through such exchanges.
            </p>
            <h3 class="information-card-header">Algorithm Description</h3>
            <p class="information-card-text">For computing an EF[1,1], swap stable, and balanced allocation</p>
            <ol>
                <li>
                    <p class="information-card-text">
                        Construct a complete bipartite graph G = (Q, P; E) with weight function w: E → R where Q = [m] and
                        w(q, p) = v<sub>f(q)</sub>(p) for each (q, p) ∈ Q &times; P; <span style="color: grey; font-style: italic">// The set Q mimics 
                        the available positions within the teams</span>
                    </p>
                </li>
                <li>
                    <p class="information-card-text">
                        Compute a perfect matching µ ⊆ Q &times; P such that the weight of the edge adjacent to vertex 1 ∈ Q is as large as
                        possible, and subject to this condition, the weight of the edge adjacent to vertex 2 ∈ Q is as large as possible, and so on
                        until vertex m ∈ Q; <span style="color: grey; font-style: italic">// Simulate round-robin with only the players'
                        values being assigned to teams</span>
                    </p>
                </li>
                <li>
                    <p class="information-card-text">
                        Let E<sup>*</sup>  = {(q, p) ∈ Q &times; P | w(q, p) = w(q, µ<sub>q</sub>)}; <span style="color: grey; 
                        font-style: italic">// Keep edges that are as good for the teams as those in µ</span>
                    </p>
                </li>
                <li>
                    <p class="information-card-text">
                        Compute a perfect matching µ<sup>*</sup> in G<sup>*</sup> = (Q, P; E<sup>*</sup>) such that the sum over all players p ∈ P of the rank of team 
                        f(µ<sub>p</sub><sup>*</sup>) for player p is minimized
                    </p>
                </li>
                <li>
                    <p class="information-card-text">
                        Return the allocation A such that p is allocated to team f(q) for each (q, p) ∈ µ<sup>*</sup>
                    </p>
                </li>
            </ol>
            <br>
            <p class="information-card-text">
                For a detailed characterization of the algorithm for EF[1,1], swap stability, and balancedness, please refer to the following paper:
            </p>
            <p class="information-card-citation">
                Ayumi Igarashi, Yasushi Kawase, Warut Suksompong, and Hanna Sumita. <a href="https://arxiv.org/pdf/2206.05879.pdf" target="_blank">Fair division with two-sided preferences.</a> arXiv preprint arXiv:2206.05879, 2022
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    

start_algo = st.button("⏳ Run Matching Algorithm ")
if start_algo:
    with st.spinner('Executing...'):
        if n * m * 0.01 > 3:
            time.sleep(2)
        else:
            time.sleep(n * m * 0.01)

    start_time = time.time()
    outcomes = compute_EF11_ssba(n, m, preferences, rankings)
    end_time = time.time()
    elapsed_time = end_time - start_time

    st.write("🎉 Outcomes:")
    outcomes_list = [[key, sorted(value)] for key, value in outcomes.items()]
    outcomes_df = pd.DataFrame(outcomes_list, columns=['Team', 'Players'])
    outcomes_df['Team'] += 1
    outcomes_df['Team'] = outcomes_df['Team'].apply(str)
    outcomes_df['Players'] = outcomes_df['Players'].apply(
        lambda x: [_x + 1 for _x in x])
    outcomes_df['Players'] = outcomes_df['Players'].apply(
        lambda x: ', '.join(map(str, x)))

    # Sort the table
    outcomes_df = outcomes_df.sort_values(['Team'],
                                          key=lambda col: col.astype(int))

    st.data_editor(outcomes_df,
                   column_config={
                       "Teams": st.column_config.NumberColumn(
                           "Team",
                           help="The list of team that get matched",
                           step=1,
                       ),
                       "Players": st.column_config.ListColumn(
                           "Players",
                           help="The list of players allocated to teams",
                       ),
                   },
                   hide_index=True,
                   disabled=True,
                   )

    # Print timing results
    st.write(f"⏱️ Timing Results:")
    st.write(f"Elapsed Time: {elapsed_time:.4f} seconds")
    
    # EF[1,1] for every pair of teams.
    # Swap-stable for every pair of players.
    balancedness = max(map(lambda x:len(x), outcomes.values())) - min(map(lambda x:len(x), outcomes.values()))
    ordinal = lambda n: "%s" % ("tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])    
    
    output_str = f"The teams have a **balanced** number of players (with a maximum difference of **{int(balancedness)}**). \n\n"

    output_str += '<h3 class="information-card-header">Fulfilling EF[1,1]</h3>\n\n'
    has_lead_str = False

    for i in range(n):
        if not has_lead_str:
            b = outcomes[i]
            output_str += f"**Team {i+1}** is allocated with players valued at {sum(preferences[i][b])} in total.\n\n"
            has_lead_str = True
        for j in range(n):
            if i == j:
                continue
            else:
                bi, bj = outcomes[i], outcomes[j]
                if sum(preferences[i][bj]) <= sum(preferences[i][bi]):
                    output_str += f"Team {i+1} values Team {j+1}'s allocation at {sum(preferences[i][bj])}, and it does not envy Team {j+1} because {sum(preferences[i][bi])} ≥ {sum(preferences[i][bj])}.\n\n"
                elif (preferences[i][bi].size > 0 and min(preferences[i][bi]) < 0) and (preferences[i][bj].size > 0 and max(preferences[i][bj]) >= 0):
                    output_str += f"Team {i+1} values Team {j+1}'s allocation at {sum(preferences[i][bj])}, but its own player has a minimum value of {min(preferences[i][bi])}. Team {i+1}'s maximum value for a player in Team {j+1} is {max(preferences[i][bj])}. Team {i+1} does not envy Team {j+1} under EF[1,1] because the difference between {sum(preferences[i][bi])} and {min(preferences[i][bi])} equals {sum(preferences[i][bi]) - min(preferences[i][bi])}, which is ≥ {sum(preferences[i][bj]) - max(preferences[i][bj])} = {sum(preferences[i][bj])} - {max(preferences[i][bj])}.\n\n"
                elif (preferences[i][bi].size == 0 or min(preferences[i][bi]) >= 0) and (preferences[i][bj].size > 0 and max(preferences[i][bj]) >= 0):
                    output_str += f"Team {i+1} values Team {j+1}'s allocation at {sum(preferences[i][bj])}. Team {i+1}'s maximum value for a player in Team {j+1}'s allocation is {max(preferences[i][bj])}. Despite this, Team {i+1} does not envy Team {j+1} under EF[1,1] because {sum(preferences[i][bi])} ≥ {sum(preferences[i][bj]) - max(preferences[i][bj])} = {sum(preferences[i][bj])} - {max(preferences[i][bj])}.\n\n"
                elif (preferences[i][bi].size > 0 and min(preferences[i][bi]) < 0) and (preferences[i][bj].size == 0 or max(preferences[i][bj]) < 0):
                    output_str += f"Team {i+1} values Team {j+1}'s allocation at {sum(preferences[i][bj])}, but its own player has a minimum value of {min(preferences[i][bi])}. Despite this, Team {i+1} does not envy Team {j+1} under EF[1,1] because the difference between {sum(preferences[i][bi])} and {min(preferences[i][bi])} is {sum(preferences[i][bi]) - min(preferences[i][bi])}, which is ≥ {sum(preferences[i][bj])}.\n\n"
                else:
                    pass
                
# Team 2 values Team 1's allocation at -34, but its own player has a minimum value of -54. Despite this, Team 2 does not envy Team 1 under EF[1,1] because the difference (-54 - -54) equals 0, which is ≥ -34.

        has_lead_str = False
        
    def reverse_dict(outcome: dict):
        pl2tm = {}
        for tm, pls in outcome.items():
            for pl in pls:
                pl2tm[pl] = tm
        return pl2tm    

    output_str2 = '<h3 class="information-card-header">Fulfilling Swap Stability</h3>\n\n'
    pl2tm = reverse_dict(outcomes)
    for i in range(m):
        for j in range(i+1, m):
            ti = pl2tm[i]
            tj = pl2tm[j]
            if ti == tj:
                continue
            output_str2 += f"**If we swap Player {i+1} (Team {ti+1}) with Player {j+1} (Team {tj+1})**, "
            if preferences[ti][i] >= preferences[ti][j]:
                output_str2 += f"the values for Team {ti+1} will decrease from {preferences[ti][i]} to {preferences[ti][j]}, "
            if preferences[tj][j] >= preferences[tj][i]:
                output_str2 += f"the values for Team {tj+1} will decrease from {preferences[tj][j]} to {preferences[tj][i]}, "
            if rankings[i][ti] < rankings[i][tj]:
                output_str2 += f"Player {i+1}'s rank will drop from {rankings[i][ti]}<sup>{ordinal(rankings[i][ti])}</sup> to {rankings[i][tj]}<sup>{ordinal(rankings[i][tj])}</sup>, "
            if rankings[j][tj] < rankings[j][ti]:
                output_str2 += f"Player {j+1}'s rank will drop from {rankings[j][tj]}<sup>{ordinal(rankings[j][tj])}</sup> to {rankings[j][ti]}<sup>{ordinal(rankings[j][ti])}</sup>, "
    
            output_str2 += f"and hence swapping Player {i+1} with Player {j+1} is **not beneficial.**\n\n"

    with st.expander(f"Explanations of Outcomes (**about {n**2 + int(m*(m-1)/2) * 2} lines**)", expanded=False):
        st.download_button('Download Full Explanations', output_str + output_str2,
                           file_name=f"{n}_teams_{m}_players_match_expl.txt")
        st.markdown(output_str, unsafe_allow_html=True)
        st.markdown(output_str2, unsafe_allow_html=True)

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
