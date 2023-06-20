## Contribution Guideline for Experienced Coders

### Code decomposition

We take reference from the [first app](../pages/1_%F0%9F%8D%8A_Weighted_Picking_Sequence.py) to decompose the code components.

1. Importing libraries: *The code imports various libraries such as `defaultdict`, `base64`, `partial`, `json`, `time`, `numpy`, `pandas`, and `streamlit` for different functionalities.*

2. Setting page configuration: *The st.set_page_config function is used to configure the title, icon, and layout of the web application.*

3. Custom CSS styles: *The st.markdown function is used to define custom CSS styles for different elements of the web application. It adds CSS code within the `<style>` tags to customize the appearance of headers, subheaders, sidebar, guide, information cards, and more.*

4. Function definitions: *The code defines several functions for loading preferences, weights, and implementing algorithms for weighted fairness. These functions are used to handle user inputs, perform calculations, and generate outcomes.*

5. User interface elements: *The code defines the user interface components using Streamlit functions like `st.markdown`, `st.sidebar`, `st.columns`, `st.number_input`, `st.slider`, `st.checkbox`, `st.file_uploader`, `st.data_editor`, and more. These elements allow users to interact with the web application by inputting values, uploading files, and modifying preferences and weights.*

6. Callback functions: *The code defines callback functions that are executed when there are changes in the weights or preferences. These functions update the corresponding session states and trigger actions based on the changes.*

7. Algorithm implementation: *The code includes an implementation of the WEF1 (Weighted Picking Sequence) algorithm. It contains functions for allocating goods based on weights and preferences, and for checking if the allocations satisfy certain fairness criteria.*

8. Web application layout: *The code defines the layout of the web application using Streamlit functions like st.markdown and st.sidebar. It includes the application title, a sidebar with a user guide, input components for agents and goods, options for uploading preferences and selecting weights, and buttons for running the algorithm and downloading outcomes.*

9. File downloads: *The code provides links to download the weights and preferences as `CSV` files.*


### Code Templates

We have provided a [code template](./template.py) that accommodates the sections above. 
