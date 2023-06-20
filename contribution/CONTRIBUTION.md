## Contribution Guideline for the Community

Please check out [the app](https://fair-alloc.streamlit.app/Create_Your_Own_App!) and generate code templates before proceeding with the rest steps!

The code templates can be as below. You should refer to the [Streamlit guide](https://docs.streamlit.io/library/api-reference/widgets) to adjust the actual function arguments (our template just provides the bare minimum version). For example, for `st.slider`, you can set the `min_value`, `max_value`, `step`, as well as `value`. 

Also, for the actual algorithm codes, you should implement the Python codes for yourself based on your entered pseudocodes. You may take the bottom code snippet as a good reference.

If you face any difficulty in implementing the app, feel free to [email us](mailto:julius.han@outlook.com?cc=warut@comp.nus.edu.sg&subject=Generated_Weighted_Fair_Allocation) for help.

### Steps
More details can be found in the [`Maintenance Guide`](../maintenance/MAINTENANCE.md).

1. [Fork this repository](https://github.com/JThh/fair-alloc-app-ra/fork) into your own account. [Clone your forked repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) into your local environment.

2. Save the above code snippet into a Python code file. For example, you may name the file as `4_📊_Weight_Fair_Allocation.py` where `4` is the index of your app. Add this code file into [`pages`](../pages).

3. Refer to the maintenance guide section [`Run Locally`](../maintenance/MAINTENANCE.md#run-locally) for how to make this app live on cloud and public to the world!

4. After adjusting the app to your favorite state, you may [deploy the app on Streamlit cloud](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app). 


**If you wish to publish your app on our site (https://fair-alloc.streamlit.app), please email us with link to your repository holding this app.**

    """
    from collections import defaultdict

    import streamlit as st
    import numpy as np

    # Input Widgets
    input_data = dict()

    input_data['Number of Agents'] = st.number_input("Number of Agents:")
    input_data['Number of Items'] = st.number_input("Number of Items:")
    input_data['x for WEF(x, 1-x)'] = st.slider("Choose x for WEF(x, 1-x):", 
                min_value=0.0, max_value=1.0, step=0.1, value=0.5)
    input_data['The Range of Agent Preferences'] = st.slider("Select the range of agent preferences:", 
                min_value=-100, max_value=100, value=[-10,10])
    input_data['The Range of Agent Weights'] = st.slider("Select the range of agent weights:", 
                min_value=-100, max_value=100, value=[0,10])

    # Algorithm Function
    def Weighted_Envy_Freeness_up_to_1_Item(input_data):
        # Algorithm code goes here
        m = input_data['Number of Items']
        n = input_data['Number of Agents']
        x = input_data['x for WEF(x, 1-x)']
        min_pref, max_pref = input_data['The Range of Agent Preferences']
        preferences = np.random.randint(min_pref, max_pref, (n, m))
        min_w, max_w = input_data['The Range of Agent Weights']
        weights = np.random.randint(min_w, max_w, (n,))

        bundles = defaultdict(list)
        times = np.zeros(n)
        remaining_items = list(range(m))

        while remaining_items:
            i = np.argmin((times + (1 - x)) / weights)
            o = remaining_items[np.argmax(preferences[i][remaining_items])]
            # Add o to bundle A_i
            bundles[i].append(o)
            # Remove o from items
            remaining_items.remove(o)
            times[i] += 1
        return bundles

    # Execute the algorithm function
    result = Weighted_Fair_Allocation(input_data)

    # Display the outputs
    st.write(result)
    """
