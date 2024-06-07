import logging

# Required Libraries
import base64
from functools import partial
import time
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
    page_title="Efficient and Fair Course Allocation App",
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
st.markdown('<h1 class="header">Efficient and Fair Course Allocation</h1>',
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
        <li>Specify the number of students (n) and courses (m) using the number input boxes.</li>
        <li>Choose to either upload a courses_capacities / students_capacities / preferences file or edit the courses_capacities / students_capacities / preferences.</li>
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

# Divide the page to 2 columns.
coln, colm = st.columns(2)

#--- Input components ---#
# n students and m courses
n = coln.number_input("Number of Students (n)",
                      min_value=MIN_AGENTS, max_value=MAX_AGENTS, step=1)
m = colm.number_input("Number of Courses (m)", min_value=MIN_ITEMS,
                      max_value=MAX_ITEMS, value=MIN_ITEMS, step=1)


# Upload input as csv file buttons
upload_preferences = None
upload_courses_capacities = None
upload_students_capacities = None

# Divide the page to 3 columns.
col1, col2, col3 = st.columns(3)

# Locate the upload buttons
with col1:
    if st.checkbox("⭐ Upload Local Courses Capacities CSV"):
        upload_courses_capacities = st.file_uploader(
            f"Upload Courses Capacities of shape ({m}, {1})", type=['csv'])   
with col2:
    if st.checkbox("⭐ Upload Local Students Capacities CSV"):
        upload_students_capacities = st.file_uploader(
            f"Upload Students Capacities of shape ({m}, {1})", type=['csv'])   
with col3:
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
        
#--- Courses courses_capacities ---#
st.write("📊 Courses Capacities (10-100, copyable from local sheets):")

# Load Courses Capacities - handle table initialization and changes
def load_courses_capacities(m, upload_courses_capacities = False, shuffle = False):
    MAX_CAPACITY = 100
    MIN_CAPACITY = 10

    if hasattr(st.session_state, "courses_capacities"): # if courses_capacities table is exist
        if upload_courses_capacities:                   # if user clicked on upload button
            courses_capacities_default = None
            # Load the user-uploaded courses_capacities file
            try:
                courses_capacities_default = pd.read_csv(
                    upload_courses_capacities, index_col=0)
                
                if courses_capacities_default.shape != (m,1): # if size doesn't match the input
                    x, y = courses_capacities_default.shape
                    st.session_state.courses_capacities.iloc[:x,
                                                      :y] = courses_capacities_default
                else:
                    st.session_state.courses_capacities = pd.DataFrame(courses_capacities_default,
                                                                columns=st.session_state.courses_capacities.columns,
                                                                index=st.session_state.courses_capacities.index)
                return st.session_state.courses_capacities
            
            except Exception as e:
                st.error(f"An error occurred while loading the courses capacities file.")
                logging.debug("file uploading error: ", e)
                st.stop()
                
        old_m = st.session_state.courses_capacities.shape[0]     # the previous number of courses (before changes)
        
        if shuffle:
            # Create m random values in range (min-max)
            random_ranks = np.random.randint(MIN_CAPACITY, MAX_CAPACITY,(m))
            # Apply the random ranks to the courses_capacities table
            st.session_state.courses_capacities = pd.DataFrame(random_ranks,
                                                                   columns=[
                                                          "Capacity"],
                                                          index=[f"Course {i+1}" for i in range(m)])
            return st.session_state.courses_capacities
        
        # if user decrease the number of courses
        if m <= old_m:
            st.session_state.courses_capacities = st.session_state.courses_capacities.iloc[:m, :1]
            return st.session_state.courses_capacities
        # if user increase the number of courses
        elif m > old_m:
            st.session_state.courses_capacities = pd.concat([st.session_state.courses_capacities,
                                                      pd.DataFrame(np.random.randint(MIN_CAPACITY, MAX_CAPACITY, (m - old_m,1)),
                                                                   columns=["Capacity"],
                                                          index=[f"Course {i+1}" for i in range(old_m, m)])],
                                                     )
            return st.session_state.courses_capacities
    # if the table isn't exist and the user wants to upload a csv file
    if upload_courses_capacities:
            courses_capacities_default = None
            # Load the user-uploaded courses_capacities file
            try:
                courses_capacities_default = pd.read_csv(upload_courses_capacities)
                if courses_capacities_default.shape != (m, 1):
                    st.error(
                        f"The uploaded courses capacities file should have a shape of ({m}, {1}).")
                    st.stop()
            except Exception as e:
                st.error("An error occurred while loading the courses capacities file.")
                st.stop()
    else:
        # Create m random values in range (min-max) and insert them to a data frame
        courses_capacities_default = pd.DataFrame(np.random.randint(MIN_CAPACITY,MAX_CAPACITY, (m)), 
                                        columns=["Capacity"],
                                        index=[f"Course {i+1}" for i in range(m)])

    st.session_state.courses_capacities = courses_capacities_default
    return st.session_state.courses_capacities

# Loading the courses_capacities table (initial/after changes)
with st.spinner("Loading..."):
    courses_capacities=  load_courses_capacities(m,upload_courses_capacities,shuffle)
    for col in courses_capacities.columns:
        courses_capacities[col] = courses_capacities[col].map(str)

# Courses Capacities Change Callback: used in Streamlit widget on_click / on_change
def course_capacity_change_callback(courses_capacities):
    st.session_state.courses_capacities = change_callback(courses_capacities)

# Courses Capacities table as editor 
edited_course_capa = st.data_editor(courses_capacities,
                              key="course_capa_editor",
                              column_config={
                                  f"Capacity": st.column_config.NumberColumn(
                                        f"Course Capacity",
                                        help=f"Course Capacity",
                                        min_value=10,
                                        max_value=100,
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
                                  course_capacity_change_callback, courses_capacities),
                              )

# Convert the editor changes from str to float
with st.spinner('Updating...'):
    for col in edited_course_capa.columns:
        edited_course_capa[col] = edited_course_capa[col].apply(
            lambda x: int(float(x)))
    st.session_state.courses_capacities = edited_course_capa

# Apply the changes
courses_capacities = edited_course_capa.values

# Download courses_capacities as CSV
courses_capacities_csv = edited_course_capa.to_csv()
b64 = base64.b64encode(courses_capacities_csv.encode()).decode()
href = f'<a href="data:file/csv;base64,{b64}" download="courses_capacities.csv">Download Courses Capacities CSV</a>'
st.markdown(href, unsafe_allow_html=True)

#--- Students Capacities (same as thr courses_capacities except the size [n instead of m]) ---#
st.write("📊 Students Capacities (1-10, copyable from local sheets):")

# Load s Capacities 
def load_students_capacities(n, upload_students_capacities = False, shuffle = False):
    MAX_CAPACITY = 10
    MIN_CAPACITY = 1
    if hasattr(st.session_state, "students_capacities"):
        if upload_students_capacities:
            students_capacities_default = None
            # Load the user-uploaded students_capacities file
            try:
                students_capacities_default = pd.read_csv(
                    upload_students_capacities, index_col=0)
                if students_capacities_default.shape != (n,1):
                    x, y = students_capacities_default.shape
                    st.session_state.students_capacities.iloc[:x,
                                                      :y] = students_capacities_default
                else:
                    st.session_state.students_capacities = pd.DataFrame(students_capacities_default,
                                                                columns=st.session_state.students_capacities.columns,
                                                                index=st.session_state.students_capacities.index)
                return st.session_state.students_capacities
            except Exception as e:
                st.error(f"An error occurred while loading the students_capacities file.")
                logging.debug("file uploading error: ", e)
                st.stop()
                
        old_n = st.session_state.students_capacities.shape[0]     
   
        if shuffle:
            random_ranks = np.random.randint(MIN_CAPACITY, MAX_CAPACITY,(n))
            st.session_state.students_capacities = pd.DataFrame(random_ranks,
                                                                   columns=[
                                                          "Capacity"],
                                                          index=[f"Student {i+1}" for i in range(n)])
            return st.session_state.students_capacities
        
        
        if n <= old_n:
            st.session_state.students_capacities = st.session_state.students_capacities.iloc[:n, :1]
            return st.session_state.students_capacities
        elif n > old_n:
            st.session_state.students_capacities = pd.concat([st.session_state.students_capacities,
                                                      pd.DataFrame(np.random.randint(MIN_CAPACITY, MAX_CAPACITY, (n - old_n,1)),
                                                                   columns=[
                                                          "Capacity"],
                                                          index=[f"Student {i+1}" for i in range(old_n, n)])],
                                                     )
            return st.session_state.students_capacities
        
    if upload_students_capacities:
            students_capacities_default = None
            # Load the user-uploaded students_capacities file
            try:
                students_capacities_default = pd.read_csv(upload_students_capacities)
                if students_capacities_default.shape != (n, 1):
                    st.error(
                        f"The uploaded students_capacities file should have a shape of ({n}, {1}).")
                    st.stop()
            except Exception as e:
                st.error("An error occurred while loading the students capacities file.")
                st.stop()
    else:
        students_capacities_default = pd.DataFrame(np.random.randint(MIN_CAPACITY,MAX_CAPACITY, (n)), 
                                        columns=["Capacity"],
                                        index=[f"Student {i+1}" for i in range(n)])
    st.session_state.students_capacities = students_capacities_default
    return st.session_state.students_capacities


with st.spinner("Loading..."):
    students_capacities=  load_students_capacities(n,upload_students_capacities,shuffle)
    for col in students_capacities.columns:
        students_capacities[col] = students_capacities[col].map(str)

def student_capacity_change_callback(students_capacities):
    st.session_state.students_capacities = change_callback(students_capacities)

edited_student_capa = st.data_editor(students_capacities,
                              key="student_capa_editor",
                              column_config={
                                  f"Capacity": st.column_config.NumberColumn(
                                      f"Student Capacity",
                                        help=f" ",
                                        min_value=1,
                                        max_value=10,
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
                                  student_capacity_change_callback, students_capacities),
                              )

with st.spinner('Updating...'):
    for col in edited_student_capa.columns:
        edited_student_capa[col] = edited_student_capa[col].apply(
            lambda x: int(float(x)))
    st.session_state.students_capacities = edited_student_capa

students_capacities = edited_student_capa.values

# Download students_capacities as CSV
students_capacities_csv = edited_student_capa.to_csv()
b64 = base64.b64encode(students_capacities_csv.encode()).decode()
href = f'<a href="data:file/csv;base64,{b64}" download="students_capacities.csv">Download Students Capacities CSV</a>'
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
                
        old_n = st.session_state.preferences.shape[0] # the previous number of students
        old_m = st.session_state.preferences.shape[1] # the previous number of courses
   
        if shuffle: # shuffle button clicked
            random_ranks = generate_random_integers_array(m,n)
            st.session_state.preferences = pd.DataFrame(random_ranks,
                                                                   columns=[
                                                          f"Course {i+1}" for i in range(m)],
                                                          index=[f"Student {i+1}" for i in range(n)])
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
                                                          f"Course {i+1}" for i in range(m)],
                                                          index=[f"Student {i+1}" for i in range(old_n, n)])],
                                                     axis=0)
            return st.session_state.preferences
        # if user increase m
        elif m > old_m:
            # add one more column to preferences table
            st.session_state.preferences =  pd.concat([st.session_state.preferences,
                                                      pd.DataFrame(np.random.randint(1,MAX_POINTS,(n, m - old_m)),
                                                                   columns=[
                                                          f"Course {i+1}" for i in range(old_m,m)],
                                                          index=[f"Student {i+1}" for i in range(n)])],
                                                     axis=1)
            return st.session_state.preferences
        else:
            random_ranks = generate_random_integers_array(m,n) # generate new random values
            st.session_state.preferences = pd.DataFrame(random_ranks, columns=[f"Course {i+1}" for i in range(m)],
                                                        index=[f"Student {i+1}" for i in range(n)])
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
                                                          f"Course {i+1}" for i in range(m)],
                                                          index=[f"Student {i+1}" for i in range(n)])
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
                                  f"Course {j}": st.column_config.NumberColumn(
                                        f"Course {j}",
                                        help=f"Students' Preferences towards Course {j}",
                                        min_value=0,
                                        max_value=100,
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
            <h2 class="information-card-header">Course Allocation Problem</h2>
            <h3 class="information-card-header">Problem Overview</h3>
            <p class="information-card-text">
                <div>
                    In the course allocation problem, a university administrator seeks to
                    efficiently and fairly <br/>
                    allocate seats (items) in over-demanded courses
                    among students (agents) with heterogeneous preferences.
                </div>
            </p>
            <h3 class="information-card-header">Algorithms</h3>
            <ul class="information-card-text">
                <li>Iterated Maximum Matching Unadjusted: this is Algorithm 1 from the following paper:
                    <p class="information-card-citation">
                        - Johannes Brustle, Jack Dippel, Vishnu V. Narayan, Mashbat Suzuki, Adrian Vetta (2020) <br/>
                        - "One Dollar Each Eliminates Envy" <br/>
                        - Proceedings of the 21st ACM Conference on Economics and Computation. 2020 <br/>
                    </p>
                    It iteratively runs a maximum-weight matching between the set of students with remaining capacity and the set of courses with remaining capacity.
                </li>
                <li>Iterated Maximum Matching Adjusted: similar to Iterated Maximum Matching, with an additional 'compensation' mechanism: 
                   at each round, for every student who did not get the maximum possible utility for that round, we add the difference in utilities to the next-best course,
                   to increase the chances of getting this course in the next round.
                </li>
                <li>Utilitarian Matching: 
                this algorithm selects the allocation that maximizes the sum of utilities of all students (the "utilitarian welfare"). It is efficient, but may be unfair.
                </li>
                <li>Round Robin: the students are arranged in an arbitrary order; each student in turn picks a course; then another round begins, until all students take all the courses they need.
                </li>
                <li>Bidirectional Round Robin  (also called Draft):
                the students are arranged in an arbitrary order; each student in turn picks a course; then the order of the students is reversed, and another round begins; until all students take all the courses they need.
                </li>
                <li>Serial Dictatorship: this simulates the current situation in course allocation, in which the agents who come first take their optimal bundle, 
                whereas the agents who come later can only choose from the remaining courses. The outcome is usually very unfair.
                </li>
            </ul>
            <p class="information-card-citation">
                    Credit: The algorithms are implemented in <a href="https://github.com/ariel-research/fairpyx">fairpyx</a>.
                    </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# Running Algorithm
    
algorithms_options = {
    "Iterated maximum matching unadjusted": fairpyx.algorithms.iterated_maximum_matching_unadjusted, 
    "Iterated maximum matching adjusted": fairpyx.algorithms.iterated_maximum_matching_adjusted,
    "Serial dictatorship": fairpyx.algorithms.serial_dictatorship,
    "Round robin": fairpyx.algorithms.round_robin, 
    "Bidirectional round robin": fairpyx.algorithms.bidirectional_round_robin, 
    "Utilitarian matching": fairpyx.algorithms.utilitarian_matching
}

# Algorithm Implementation
def algorithm(m, n, courses_capacities, students_capacities, preferences, algo_names: list):
    pref_dict = {}
    capa_dict = {}
    req_dict = {}
    students_conflicts = {}
    courses_conflicts = {}

    for i in range(n):
        pref_dict[f"Student {i+1}"] = {}
        req_dict[f"Student {i+1}"] = students_capacities[i,0]
        students_conflicts[f"Student {i+1}"] = {}
        for j in range(m):
            pref_dict[f"Student {i+1}"][f"Course {j+1}"] = preferences[i,j]
    for i in range(m):
        capa_dict[f"Course {i+1}"] = courses_capacities[i,0]
        courses_conflicts[f"Course {i+1}"] = {}

    instance = fairpyx.Instance(
        agent_capacities=req_dict, 
        valuations=pref_dict,
        item_capacities=capa_dict,
        item_conflicts=courses_conflicts,
        agent_conflicts=students_conflicts,
        )

    allocations = {} 

    for algo_name in algo_names:
        algorithm = algorithms_options[algo_name]
        if algo_name.startswith("Iterated maximum matching"):
        # string_explanation_logger = fairpyx.StringsExplanationLogger(instance.agents)
            string_explanation_logger = fairpyx.StringsExplanationLogger({
                agent for agent in instance.agents
            },language='en', mode='w', encoding="utf-8")
            allocation = fairpyx.divide(algorithm=algorithm, instance=instance, explanation_logger=string_explanation_logger)
            allocations[algo_name] = (allocation,string_explanation_logger)
        else:
            allocation = fairpyx.divide(algorithm=algorithm, instance=instance)
            allocations[algo_name] = (allocation, None)
            string_explanation_logger = None
    return allocations,instance

# Checker Function for Algorithm
def algorithm_checker(instance,allocations):
    result_vector = []
    for algo_name, allocation in allocations.items():
        allocation = allocation[0]
        matrix:fairpyx.AgentBundleValueMatrix = fairpyx.AgentBundleValueMatrix(instance, allocation)
        values = np.round([matrix.utilitarian_value(),
                            matrix.egalitarian_value(),
                            matrix.max_envy(),
                            matrix.mean_envy()])
        result_vector.append([algo_name]+list(values))
        logging.debug('Result Vector:', result_vector)
    return result_vector


algo_names = st.multiselect(
   "Which algorithm do you want to use?",
   tuple(algorithms_options.keys()),
   ["Iterated maximum matching unadjusted"],
   placeholder="Select Algorithm...",
)

start_algo = st.button(f"⏳ Run Algorithm")
if start_algo:
    with st.spinner('Executing...'):
        if n * m * 0.01 > 3:
            time.sleep(2)
        else:
            time.sleep(n * m * 0.01)

    start_time = time.time()
    outcomes, instance = algorithm(m, n, courses_capacities,students_capacities,preferences, algo_names)
    end_time = time.time()
    elapsed_time = end_time - start_time
    st.write("🎉 Outcomes:")
    for algo_name, values in outcomes.items():
        outcomes_list = ['']*n
        column_config = {}        
        courses_head = [algo_name + ' Results']
        column_config[algo_name + ' Results'] = st.column_config.ListColumn(
                        algo_name + ' Results',
                        help="The list of courses allocated to students",
                    )
        (allocation, explanation) = values            
        if explanation:
            courses_head+= ['Explanation']
            for i in range(n):
                outcomes_list[i] = [f"Student {i+1}"]
                outcomes_courses = allocation[f"Student {i+1}"]
                outcomes_str = ", ".join([courses for courses in outcomes_courses])
                outcomes_list[i]+=[outcomes_str]+[explanation.agent_string(f'Student {i+1}')]
        else:
            for i in range(n):
                outcomes_list[i] = [f"Student {i+1}"]
                outcomes_courses = allocation[f"Student {i+1}"]
                outcomes_str = ", ".join([courses for courses in outcomes_courses])
                outcomes_list[i]+=[outcomes_str]
            
            
        outcomes_df = pd.DataFrame(outcomes_list, columns=['Student']+courses_head)

        st.data_editor(outcomes_df,
                    column_config=column_config,
                    hide_index=True,
                    disabled=True,
                    )

    st.write("🗒️ Outcomes Summary:")

    vector = algorithm_checker(instance,outcomes)
    parameters = ["Algorithm","Utilitarian value","Egalitarian value","Max envy", "Mean envy"]
    vector_df = pd.DataFrame(vector, columns=parameters)
    st.data_editor(vector_df,
                   column_config={
                       "Algorithm": st.column_config.TextColumn(
                           "Algorithm",
                       ),
                       parameters[1]: st.column_config.NumberColumn(
                           parameters[1],
                           help="sum of students' values",
                       ),
                       parameters[2]: st.column_config.NumberColumn(
                           parameters[2],
                           help="smallest value of a student",
                       ),
                       parameters[3]: st.column_config.NumberColumn(
                           parameters[3],
                           help="largest envy among all pairs of students",
                       ),
                       parameters[4]: st.column_config.NumberColumn(
                           parameters[4],
                           help="average over all students, of the maximum envy felt towards another student",
                       ),
                   },
                   hide_index=True,
                   disabled=True,
                   )

    # Print timing results
    st.write(f"⏱️ Timing Results:")
    st.write(f"Elapsed Time: {elapsed_time:.4f} seconds")
