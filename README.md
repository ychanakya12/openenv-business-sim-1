---
title: Business Simulation Agent
emoji: 🏢
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 🏢 AI CEO: Strategic Business Simulation Agent
### Meta PyTorch Hackathon — OpenEnv Round 1 Submission

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Certified-success?style=for-the-badge)](https://github.com/OpenEnv)

---

## 🚀 Overview
This repository contains a state-of-the-art **AI CEO Agent** developed for the Meta PyTorch Hackathon. The agent is designed to manage a software company through a highly complex, adversarial, and stochastic business simulation environment.

The agent leverages **GPT-OSS 20B** to make quarterly decisions across six crucial business factors:
1.  **Profit Potential**: Strategic project selection.
2.  **Risk Management**: Identifying and mitigating "Hidden Risk."
3.  **Team Capability**: Hiring, firing, and training management.
4.  **Resource Optimization**: Budgeting and tool choice (Tech Stack).
5.  **Market Alignment**: Adapting to Boom/Recession cycles.
6.  **Reputation**: Building long-term brand equity vs. short-term gains.

---

## 🧠 Problem Statement
The **BusinessSimEnv** is a multi-turn, delayed-reward environment where every action has cascading consequences. A CEO must survive up to **8 quarters** (2 years) while maintaining:
*   Positive cash flow in a recession.
*   Low employee burnout to avoid skill degradation.
*   High reputation to attract premium projects.

The **AdversarialAgent** in the environment injects realistic "shocks" (budget audits, key departures, and client disputes) to test the agent's resilience.

---

## 🛠️ Strategic Architecture
Our agent implements a **Reasoning-Driven Control Loop** with the following advanced heuristics:

### 📊 1. Proactive ROI Analysis
The agent doesn't just look at base profit. It performs a real-time **Return on Investment (ROI)** analysis that factors in:
*   Expected Profit * (1 - Total Risk).
*   Quarterly Salary Burn (Salary per developer * count).
*   Opportunity Cost of the team's resource pool.

### 📉 2. Salary Optimization Hack
To maximize scores on the **Easy (Survival)** and **Medium (Growth)** tasks, the agent implements a "ruthless efficiency" model. It dynamically **fires developers** if the current project portfolio can be managed with a smaller team, instantly saving ~$24,000 per quarter in overhead.

### 🛡️ 3. Bankruptcy Protection
The agent follows a strict **Budget-to-Penalty Ratio**. If a project failure would deplete >50% of the remaining budget, the agent will **"Wait"** and skip the quarter to ensure survival, rather than taking a "suicidal" gamble.

### 💎 4. Premium Stack Multiplier
By defaulting to a **Premium Tech Stack**, the agent reduces project risk and increases profit margins by 20%, which is the key to hitting the $150,000 budget target required for a perfect score in the Survival task.

---

## ⚙️ Installation & Usage

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/business-sim-agent.git
    cd business-sim-agent
    ```

2.  **Configure Environment Variables**:
    ```bash
    $env:HF_TOKEN="your_huggingface_token"
    ```

3.  **Run Inference**:
    ```bash
    python inference.py
    ```

4.  **Validate Submission**:
    ```bash
    python validate.py
    ```

---

## 🎨 Visualization
The agent provides a clean terminal output with `[START]`, `[STEP]`, and `[END]` markers, as required by the OpenEnv validator, making it fully compatible with automated leaderboard evaluation.

> *"Building a resilient business is not about never failing; it's about failing small and winning big."* — **AI CEO Agent**

---

## 🛠️ Tech Stack & Dependencies
The Agent and Environment are built with a modern, high-performance Python stack designed for low-latency inference and robust state management:
*   **Core Logic**: Python 3.12+ with strict Pydantic typing for action/observation schemas.
*   **Inference**: Leveraging **OpenAI-compatible endpoints** (Hugging Face Router) with **GPT-OSS 20B** for logical reasoning.
*   **Environment API**: **FastAPI** for the OpenEnv simulation server, ensuring standard-compliant communication.
*   **Communication**: **Httpx** for asynchronous networking between the agent and the simulation.

---

## 📊 Evaluation Methodology
Our agent is validated against three distinct "Gates" to ensure it meets Meta's resilience standards:
1.  **Unit Tests**: Verifying deterministic logic for budget calculations and salary burns (`test_grade.py`).
2.  **Adversarial Stress Test**: Running the Hard task (`adversarial_resilience`) which injects 10% chance shocks each quarter.
3.  **Cross-Scenario Generalization**: Ensuring the prompt heuristics work across Easy, Medium, and Hard tasks without manual parameter tuning.

---

## 🔮 Future Roadmap (v2.0)
The next evolution of the MetaStableMinds CEO Agent will focus on:
*   **Multi-Agent Negotiation**: Allowing the CEO to negotiate project costs with the Adversarial Agent.
*   **Dynamic Tech Stacks**: Implementing custom tech stacks where the agent can mix-and-match technologies.
*   **Reinforcement Learning (PPO)**: Transitioning from heuristic prompts to a fine-tuned model specifically trained on BusinessSim trajectories.

---

## 🤝 Team MetaStableMinds
*   **Y. Chanakya** - Lead Strategic Design & Infrastructure.
*   **Collaborators** - Strategic Heuristics & Analysis.

*Produced for the Meta PyTorch Hackathon 2024.*
