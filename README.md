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
*   **Objective**: Maximize long-term corporate health.

### 📉 2. The Market Agent (The Economic Engine)
*   **Role**: Simulates the macro-economic environment and consumer demand.
*   **Impact**: Driven by a Markov Chain, this agent triggers **Boom, Stable, and Recession** cycles.

### 🛡️ 3. The Adversarial Agent (The Friction Engine)
*   **Role**: Simulates the "Chaos" of real-world business operations.
*   **Impact**: Triggers events like Budget Audits, Key Employee Departures, and Client Disputes.

---

## 🛠️ Strategic Architecture
Our agent implements a **Reasoning-Driven Control Loop** leveraging **GPT-OSS 20B**:

### 📊 1. Proactive ROI Analysis
The agent performs real-time **ROI** analysis factoring in Expected Profit, Total Risk, and Salary Burn.

### 📉 2. Salary Optimization Hack
The agent dynamically **fires developers** to save ~$24,000 per quarter in overhead.

### 🛡️ 3. Bankruptcy Protection
Skip the quarter rather than taking a "suicidal" gamble.

---

## 📊 Evaluation Methodology
Our agent is validated against:
1.  **Unit Tests**: Deterministic logic verification.
2.  **Adversarial Stress**: survival under Hard task conditions.
3.  **Generalization**: Performance across all difficulties.

---

## 🤝 Team MetaStableMinds
*   **Y. Chanakya** - Lead Strategic Design & Infrastructure.
*   **Collaborators** - Strategic Heuristics & Analysis.

---

> *"Complexity is the enemy of execution. Our agent brings strategic clarity to the chaotic frontier of the open market."*

*Produced for the Meta PyTorch Hackathon 2024.*
