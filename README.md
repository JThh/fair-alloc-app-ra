# Fast & Fair Web Application

![Welcome Page](./resource/welcome-nrw.png)

_**[News] We feel glad to share that our paper "Fast & Fair: A Collaborative Platform for Fair Division Applications" has been accepted to AAAI-24 for publication!**_ Camera-ready version is now available [here](https://www.comp.nus.edu.sg/~warut/fast-and-fair.pdf).

Welcome to Fast & Fair. We have implemented three algorithms coming from papers ([1](https://arxiv.org/pdf/2112.04166.pdf),[2](https://arxiv.org/pdf/2206.05879.pdf),[3](https://www.sciencedirect.com/science/article/abs/pii/S0165489619300599)). The app is available at https://fair-alloc.streamlit.app/. 

## Introduction

The allocation app is designed to help you achieve some envy-freeness notion (e.g. `WEF(x, 1-x)`, `EF[1,1]`, and `EF`) by fairly allocating indivisible items. Here's how it works:

1. Start by providing the number of participants and the available items.
2. For each participant, specify their preferences for the items. You can assign weights to these preferences to reflect their importance.
3. Once you've entered all the preferences, click on the 'Run Allocation Algorithm' button.
4. Our algorithms will process the inputs and generate an allocation that minimizes envy while considering participant preferences.
5. The app will display the resulting allocation along with any relevant statistics or insights.

With our fair allocation app, you can explore various scenarios and experiment with different preferences and weightings to find an allocation that best satisfies the corresponding fairness and efficiency objective.

## Cummunity and Contribution

Join our [Slack channel](https://join.slack.com/t/fastfaircommunity/shared_invite/zt-1xyl1akls-ukrAsy3Kmm5lilCuB1uOmQ) for latest updates and to ask questions.

We welcome community contribution. Please refer to the [Contribution Guide](./contribution/ADVANCED_CONTRIBUTION.md) for details. 
