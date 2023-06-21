## Contribution Guideline for Experienced Coders

This serves as an advanced guide for experienced coders. You may skip the decomposition section and work directly on the [`Code Templates`](./ADVANCED_CONTRIBUTION.md#code-templates). **You are advised to submit your app file via a [Pull Request](https://github.com/JThh/fair-alloc-app-ra/compare) upon finishing all steps in this guide**. Our developers will review your PR (after it passes the CI) and incorporate your contribution at our site if possible!

### Code Section Decomposition

We take reference from the codes of the [weight picking sequence app](../pages/1_%F0%9F%8D%8A_Weighted_Picking_Sequence.py) to decompose the code components.

1. *Importing libraries*: The code imports various libraries such as `defaultdict`, `base64`, `partial`, `json`, `time`, `numpy`, `pandas`, and `streamlit` for different functionalities.

```
from collections import defaultdict
import base64
from functools import partial
...
```

2. *Setting page configuration*: The st.set_page_config function is used to configure the title, icon, and layout of the web application.

```
# Set page configuration
st.set_page_config(
    page_title="Fair Allocation App",
    page_icon="🍊",
    layout="wide",
)
```

3. *Custom CSS styles*: The st.markdown function is used to define custom CSS styles for different elements of the web application. It adds CSS code within the `<style>` tags to customize the appearance of headers, subheaders, sidebar, guide, information cards, and more.

```
# Custom CSS styles
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
    ...
)
```

4. *Function definitions*: The code defines several functions for loading preferences, weights, and implementing algorithms for weighted fairness. These functions are used to handle user inputs, perform calculations, and generate outcomes.

```
def load_preferences(m, n, upload_preferences):
    if hasattr(st.session_state, "preferences"):
        pass
    ...

def load_weights(n, unweighted=False):
    if hasattr(st.session_state, "weights"):
        pass
    ...
...
```

5. *User interface elements*: The code defines the user interface components using Streamlit functions like `st.markdown`, `st.sidebar`, `st.columns`, `st.number_input`, `st.slider`, `st.checkbox`, `st.file_uploader`, `st.data_editor`, and more. These elements allow users to interact with the web application by inputting values, uploading files, and modifying preferences and weights.

```
# Add input components
col1, col2, col3 = st.columns(3)
n = col1.number_input("Number of Agents (n)",
                      min_value=2, max_value=100, step=1)
m = col2.number_input("Number of Goods (m)", min_value=2,
                      max_value=1000, value=6, step=1)
x = col3.slider("Choose a value for x in WEF(x, 1-x)",
                min_value=0.0, max_value=1.0, value=0.5, step=0.01, help="💡 Large x favors low-weight agents")
...
```

6. *Callback functions*: The code defines callback functions that are executed when there are changes in the weights or preferences. These functions update the corresponding session states and trigger actions based on the changes.

```
def pchange_callback(preferences):
    ...
    st.session_state.preferences = preferences

xxx = st.data_editor(preferences,
                              key="preference_editor",
                              column_config={
                                  f"Item {j}": st.column_config.TextColumn(
                                      f"Item {j}",
                                      ...
                                  )
                                  for j in range(1, m+1)
                              },
                              on_change=partial(
                                  pchange_callback, preferences),
                              )
...
```

7. *Algorithm implementation*: The code includes an implementation of the WEF1 (Weighted Picking Sequence) algorithm. It contains functions for allocating goods based on weights and preferences, and for checking if the allocations satisfy certain fairness criteria.

```
def wef1x_algorithm(x, m, n, weights, preferences):
    # Implementation of WEF1 algorithm
    ...
```

8. *Web application layout*: The code defines the layout of the web application using Streamlit functions like st.markdown and st.sidebar. It includes the application title, a sidebar with a user guide, input components for agents and goods, options for uploading preferences and selecting weights, and buttons for running the algorithm and downloading outcomes.

```
# Set the title and layout of the web application
st.markdown('<h1 class="header">Your Title</h1>',
            unsafe_allow_html=True)

# Insert header image
st.sidebar.image("./resource/xxx.png")

st.sidebar.title("User Guide")
...
```

9. *File downloads*: The code provides links to download the weights and preferences as `csv` files.

```
# Download weights as CSV
weights_csv = xxx.to_csv()
b64 = base64.b64encode(weights_csv.encode()).decode()
href = f'<a href="data:file/csv;base64,{b64}" download="weights.csv">Download Weights CSV</a>'
st.markdown(href, unsafe_allow_html=True)
...
```

### Code Templates

We have provided a [code template](./template.py) that accommodates the sections above. You may fill up the template before adding the file to [`Pages`](../pages/) and making a [pull request](https://github.com/JThh/fair-alloc-app-ra/compare) to this repository. 

In your local development settings, you may refer to the guide as detailed in the [`Maintenance Guide`](../maintenance/MAINTENANCE.md#run-locally) to ensure your app runs perfectly with the existing apps.

### Submit your PR

Follow the rest steps to contribute your codes to our repository.

1. Run your apps locally and ensure it runs smoothly and as flawlessly as possible.
2. Create a `requirements.txt` file, save it at the root directory, and dd any additional package requirements (such as `networkx`) to the  file.
3. Submit your PR to the repository - the PR will trigger our CI/CD workflow and run some basic eligibility tests. Make sure all tests are passed.
4. Notify the developer team by tagging (`@`) us in your PR. We will get to your contribution as soon as possible.
5. Once your PR is reviewed, commented, and refined (or rebutted), we will merge your PR to our `dev` branch, and launch the app at a separate link for a short while to test its stability. 
6. After the test period, we will announce the official launch of this new app at our social media homepages and direct all traffic to our [original site link](https://fair-alloc.streamlit.app/). 