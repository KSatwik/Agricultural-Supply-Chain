# Agricultural Supply Chain Lab

> **Stochastic MILP + Machine Learning for Perishable Produce Routing & Allocation**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://satwik-agri-scm.streamlit.app/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PuLP](https://img.shields.io/badge/PuLP-2.8.0-green.svg)](https://coin-or.github.io/pulp/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An interactive site that optimizes the distribution of perishable
agricultural produce from **M supply depots (farms)** to **N demand hubs** across
**T discrete time periods**. The pipeline combines a **scikit-learn** regression
model to predict produce deterioration with a **PuLP**-based Mixed-Integer Linear
Program (MILP) that finds the cost-minimal routing plan.

The app also features a full **Stochastic Programming** analysis (Sample Average
Approximation) that quantifies the value of planning for uncertainty — computing
**VSS (Value of the Stochastic Solution)** and **EVPI (Expected Value of Perfect
Information)**.

---

##  Features

- **Network & Data** — Generates a realistic synthetic supply chain network
  with geographically distributed farms and hubs, supply capacities, and demand
  profiles across multiple time periods.
- **ML Model Lab** — Trains and compares **5 regression models** (Linear,
  Ridge, Random Forest, Gradient Boosting, Hist Gradient Boosting) to predict a
  **Produce Deterioration Factor** ∈ [0,1] from transit time, temperature,
  humidity, and cargo type.
- **MILP Optimizer** — Solves a multi-period, multi-depot **Mixed-Integer
  Linear Program** with continuous flow variables, binary route activations,
  supply caps, demand fulfillment constraints, and link capacities.
- **Stochastic Lab** — Runs a full **two-stage stochastic program** across
  K weather scenarios, computing:
  - **EV** (Expected Value / deterministic mean problem)
  - **EEV** (Expected result of using the EV solution)
  - **RP** (Recourse Problem / stochastic optimum)
  - **WS** (Wait-and-See / perfect information)
  - **VSS = EEV − RP** and **EVPI = RP − WS**
  - **SAA Convergence Sweep** — shows how cost and deployed capacity converge
    as the number of scenarios K increases.
- ** Mathematics** — Fully documented mathematical formulation with LaTeX
  equations rendered in the app.
- ** Deploy** — Step-by-step deployment guide for GitHub + Streamlit Cloud.

---

## Screenshots

### Main Dashboard
<img width="1439" height="852" alt="image" src="https://github.com/user-attachments/assets/6ca95f14-f5c0-4d81-b7cf-eea8b2844eaa" />

<img width="1440" height="810" alt="image" src="https://github.com/user-attachments/assets/43e08592-3084-4c9a-a21a-c2668b819db9" />



### ML Model Comparison

<img width="1438" height="704" alt="image" src="https://github.com/user-attachments/assets/e3d5a85f-6869-4ec5-9154-499159ba021c" />

<img width="1440" height="753" alt="image" src="https://github.com/user-attachments/assets/b36ea5ae-7d80-4063-8766-585af9dca8c3" />

<img width="1440" height="800" alt="image" src="https://github.com/user-attachments/assets/06217bc5-945c-451d-9391-ef6d841d4c2a" />




### MILP Optimizer Results

<img width="1440" height="852" alt="image" src="https://github.com/user-attachments/assets/bce6551c-f27c-4e29-8d66-59bc916de8ca" />

<img width="1440" height="641" alt="image" src="https://github.com/user-attachments/assets/e0c49b6b-5469-4015-a609-71731a3fbf5c" />



### Stochastic Lab KPIs
<img width="1440" height="770" alt="image" src="https://github.com/user-attachments/assets/2c999ed3-0fed-4365-a7c6-04c9f181ab25" />

<img width="1440" height="515" alt="image" src="https://github.com/user-attachments/assets/3cdcc7ab-605b-44fd-88ef-62d3007c9ab6" />

<img width="1190" height="773" alt="image" src="https://github.com/user-attachments/assets/d6302a07-f1ff-4af8-abf3-cc757ffd0510" />

<img width="1440" height="717" alt="image" src="https://github.com/user-attachments/assets/79679b2c-e470-4701-b234-f98073c1bd57" />


---

##  Mathematical Formulation

### 1. Predictive Deterioration Model

<img width="913" height="168" alt="image" src="https://github.com/user-attachments/assets/34337915-bca9-4ff4-96b6-04626c1cac03" />


### 2. Deterministic MILP (EV Problem)

<img width="956" height="524" alt="image" src="https://github.com/user-attachments/assets/e8b699e2-ec8a-4e0d-b65b-150a54a392b1" />


### 3. Two-Stage Stochastic Program (RP)

<img width="772" height="180" alt="image" src="https://github.com/user-attachments/assets/49a93e98-00dc-46da-aff8-ffa4ad631e84" />

### 4. Value-of-Information Metrics

<img width="906" height="461" alt="image" src="https://github.com/user-attachments/assets/d0644425-f2eb-44d4-ae45-7cbc9a5cd94b" />

### 5. Why does capacity change with randomness?


<img width="1099" height="159" alt="image" src="https://github.com/user-attachments/assets/e4ca504a-4732-4fd1-abf6-6a8e7d711606" />


