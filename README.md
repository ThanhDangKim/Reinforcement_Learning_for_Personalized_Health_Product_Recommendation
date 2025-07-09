# 🧠 Reinforcement Learning for Personalized Health Product Recommendation on Amazon

## 📌 Overview

This project applies **Reinforcement Learning (RL)** to build a personalized recommendation system for health-related products on Amazon. A **custom environment** was constructed using the `gym` library to simulate the recommendation setting.

In this environment:
- **State** represents the current product a user is interacting with.
- **Actions** are a list of recommended products chosen by the agent.
- **Reward** is based on whether the user interacts with (e.g., purchases) the recommended item in the future.
- **Iteration limit** defines the maximum number of steps per training episode.

The custom environment is implemented in the `ProductRecommendationEnv` class, which encapsulates:
- A structured `state_dict` storing mappings of user IDs and product IDs based on review time to simulate time-dependent sequences.
- `product_states`, which include encoded metadata (e.g., product description, rating, and features) for use as inputs to neural networks.
- A reward mechanism where correctly predicted products yield a positive reward (1), while irrelevant recommendations return zero or negative reward. The agent improves its behavior over time based on feedback received from previous actions.

In recommendation systems, each product suggestion can be considered an **action**, and the **user's response** (e.g., clicks, purchases) acts as a **reward**. RL enables the system to not only optimize a single recommendation but also an entire sequence of suggestions, leading to a more dynamic and personalized user experience.

---

## 🧠 Reinforcement Learning Models Applied

### 🔷 Deep Reinforcement Learning (Deep RL)

#### 1. Deep Q-Network (DQN)
- A **model-free** algorithm based on Q-Learning.
- Uses a **deep neural network** to approximate the Q-value function.
- Neural input includes both:
  - **Textual metadata** (processed with TextVectorization, Embedding, and Bi-LSTM).
  - **Numerical ratings**.
- Utilizes:
  - **Experience Replay** to break data correlation.
  - **Epsilon-Greedy Policy** to balance exploration and exploitation.
  - **Target Network** to stabilize training.
- Loss calculated using Mean Squared Error (MSE) between predicted and target Q-values.
- Trained with Adam optimizer.

#### 2. Deep REINFORCE
- A **policy-based** method producing probabilistic action distributions via softmax output.
- Policy network architecture:
  - Textual metadata passed through Embedding + BiLSTM layers.
  - Concatenated with numerical ratings.
  - Final output: softmax probability for each product.
- Trained using **policy gradients**, where the agent:
  - Collects rewards through full episodes.
  - Computes discounted cumulative reward.
  - Updates the policy using cross-entropy loss.

### 🔷 Classic Reinforcement Learning Methods

#### 1. Q-Learning
- Maintains a **Q-table** for state-action values.
- Chooses actions via **epsilon-greedy strategy**.
- Updates Q-values using the Bellman equation:
Q[state, action] = Q[state, action] + α * (reward + γ * max(Q[next_state]) - Q[state, action])
- Epsilon decay is applied over time to favor exploitation.

#### 2. Monte Carlo Control
- Collects **full episodes** before updating values.
- Stores state-action-reward sequences in an episode list.
- After an episode ends:
- Calculates **discounted returns**.
- Updates value estimates and policy based on first-visit occurrences.
- Policy is improved by averaging returns across visits.

---

## 🏥 Application: Health Product Recommendation on Amazon

### 📊 Dataset Source

This project utilizes the **Health and Personal Care** category from the **Amazon Health Review dataset**, compiled by Prof. Julian McAuley and his research group at the **University of California, San Diego (UCSD)**.

The dataset includes detailed user reviews of health-related products, with rich metadata for personalized recommendation modeling.

### 📁 Dataset Features

| Field | Description | Data Type |
|-------|-------------|------------|
| `overall` | Overall product rating | Float |
| `vote` | Number of helpful votes | String |
| `verified` | Verified purchase indicator | Boolean |
| `reviewTime` | Date of review (string format) | String |
| `reviewerID` | Unique user identifier | String |
| `asin` | Amazon Standard Identification Number | String |
| `style` | Product style/variant (nested JSON) | JSON Object |
| `reviewerName` | Name of the reviewer | String |
| `reviewText` | Full review content | String |
| `summary` | Short summary of the review | String |
| `unixReviewTime` | Review timestamp (Unix format) | Integer |

> 📌 *Table 3.1: Summary of Dataset Structure*

---

## 🎯 Project Goal

The primary objective is to develop a **reinforcement learning-based recommendation system** that:
- Continuously learns from user behavior,
- Optimizes sequences of health product suggestions,
- Enhances personalization beyond static recommendation approaches.

---

## ⚙️ Technologies Used

- Python
- TensorFlow / PyTorch (for Deep RL)
- OpenAI Gym (or custom environment)
- Pandas / NumPy
- Amazon Review Dataset

---

## 🗂️ Project Structure

```plaintext
📦 RL-Recommendation-System
┣ 📁 agents/
┃ ┣ 📁 Health_and_Personal_Care_DQN/ # Deep Q-Network model
┃ ┣ 📁 Health_and_Personal_Care_DR/ # Deep Reinforce model
┃ ┣ 📜 q_table.npy # Q-learning value table
┃ ┣ 📜 mc_table.npy # Monte Carlo Control value table
┃
┣ 📁 main/ # Main training scripts
┃ ┗ 📜 main.py
┃
┣ 📁 OriginalDatasets/
┃ ┣ 📜 Health_and_Personal_Care.json # Original review dataset
┃ ┗ 📜 meta_Health_and_Personal_Care.jsonl # Metadata for products
┃
┣ 📁 ProcessedDatasets/
┃ ┣ 📜 *.json # Processed JSON files after:
┃ ┃ - metadata + review merging
┃ ┃ - column filtering
┃ ┃ - splitting per product/user
┃ ┃ - re-aggregation for training
┃
┣ 📁 UI/
┃ ┣ 📜 *.js   # Zeppelin notebooks and Angular frontend/backend
┃
┣ 📜 requirements.txt # Required packages
┗ 📜 README.md # Project documentation
```

## ✍️ Author

**Đặng Kim Thành**
**Bùi Quốc Khang**  
4th-Year Student, HCMC University of Technology and Education  
Specialization: Artificial Intelligence

---

## 📄 License

This project is for educational and research purposes. Dataset is publicly available and credited to the original authors at UCSD.
