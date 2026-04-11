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

## 🧠 Problem Statement
The **BusinessSimEnv** is a multi-turn, delayed-reward environment where every action has cascading consequences. A CEO must survive up to **8 quarters** (2 years) while maintaining:
*   Positive cash flow in a recession.
*   Low employee burnout to avoid skill degradation.
*   High reputation to attract premium projects.

The **AdversarialAgent** in the environment injects realistic "shocks" (budget audits, key departures, and client disputes) to test the agent's resilience.

---

## 🌍 The Simulation Ecosystem: A Tri-Agent Interaction
Our simulation is not a static set of rules, but a dynamic, multi-agent world where the **AI CEO** must compete and survive against two other environmental agents:

### 👤 1. The AI CEO (The Decision Engine)
*   **Role**: The primary protagonist of the simulation.
*   **Actions**: Project acceptance, hiring/firing (Salary Optimization), tech stack selection, and skill training.
*   **Objective**: Maximize long-term corporate health (Budget, Reputation, and Low Burnout) through high-reasoning ROI analysis.

### 📉 2. The Market Agent (The Economic Engine)
*   **Role**: Simulates the macro-economic environment and consumer demand.
*   **Behavior**: Driven by a Markov Chain, this agent triggers **Boom, Stable, and Recession** cycles.
*   **Impact**: It alters the Domain Demand multipliers, forcing the CEO to adapt their strategy quarterly.

### 🛡️ 3. The Adversarial Agent (The Friction Engine)
*   **Role**: Simulates the "Chaos" of real-world business operations.
*   **Impact**: Triggers events like Budget Audits, Key Employee Departures, and Client Disputes.

---

## 🛠️ Strategic Architecture
Our agent implements a **Reasoning-Driven Control Loop** leveraging **GPT-OSS 20B** with the following advanced heuristics:

The agent leverages **GPT-OSS 20B** to make quarterly decisions across six crucial business factors:
1.  **Profit Potential**: Strategic project selection.
2.  **Risk Management**: Identifying and mitigating "Hidden Risk."
3.  **Team Capability**: Hiring, firing, and training management.
4.  **Resource Optimization**: Budgeting and tool choice (Tech Stack).
5.  **Market Alignment**: Adapting to Boom/Recession cycles.
6.  **Reputation**: Building long-term brand equity vs. short-term gains.

### 📊 1. Proactive ROI Analysis
The agent performs real-time **Return on Investment (ROI)** analysis factoring in Expected Profit, Total Risk, and Salary Burn.

### 📉 2. Salary Optimization Hack
The agent dynamically **fires developers** to instantly save ~$24,000 per quarter in overhead.

### 🛡️ 3. Bankruptcy Protection
Follows a strict **Budget-to-Penalty Ratio**. Skip the quarter rather than taking a "suicidal" gamble.

---

## 📊 Evaluation Methodology
Our agent is validated against three distinct "Gates":
1.  **Unit Tests**: Verifying deterministic logic for budget calculations (`test_grade.py`).
2.  **Adversarial Stress Test**: Running the Hard task (`adversarial_resilience`).
3.  **Generalization**: Heuristics that work across all task difficulties.

---

## 🛠️ Tech Stack & Dependencies
*   **Core Logic**: Python 3.12+ with Pydantic schemas.
*   **Inference**: **GPT-OSS 20B** on Hugging Face Router.
*   **Environment API**: **FastAPI** simulation server.
*   **Communication**: Asynchronous **Httpx**.

---

## 🔮 Future Roadmap (v2.0)
*   **Multi-Agent Negotiation**: Direct negotiation with the Adversarial Agent.
*   **Dynamic Tech Stacks**: Modular technology selection.
*   **Reinforcement Learning (PPO)**: Transitioning from heuristic prompts to fine-tuned models.

---

## ⚙️ Installation & Usage
1.  **Configure Token**: `$env:HF_TOKEN="your_token"`
2.  **Run Inference**: `python inference.py`
3.  **Validate**: `python validate.py`

---

## 🤝 Team MetaStableMinds
*   **Y. Chanakya** - Lead Strategic Design & Infrastructure.
*   **Collaborators** - Strategic Heuristics & Analysis.

---

> *"Complexity is the enemy of execution. Our agent brings strategic clarity to the chaotic frontier of the open market."*

