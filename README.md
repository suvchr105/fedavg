# 🧠 Federated Averaging (FedAvg) 

This repository contains a **from-scratch implementation of Federated Learning using the FedAvg algorithm**, demonstrating how decentralized clients collaboratively train a global model **without sharing raw data**.

The project simulates a **server–client architecture** where multiple clients train local models and a central server aggregates them using **Federated Averaging**.

---

## 📌 What is Federated Learning?

Federated Learning (FL) is a **privacy-preserving machine learning paradigm** where:

- Data stays **on client devices**
- Only **model updates (weights)** are shared
- A central server aggregates updates to improve a **global model**

This repository focuses on the **FedAvg algorithm**, the foundational method introduced by Google.

---

## ⚙️ FedAvg Algorithm (High-Level)

1. Server initializes a global model  
2. Server sends the model to selected clients  
3. Each client:
   - Trains locally on private data
   - Sends updated weights to server  
4. Server aggregates weights using **weighted averaging**  
5. Steps repeat for multiple communication rounds  

---

## 🗂️ Project Structure

```text
fedavg/
│
├── client.py      # Client-side local training logic
├── server.py      # Server-side aggregation (FedAvg)
├── models.py      # Neural network / ML model definitions
├── data.py        # Dataset loading and client data partitioning
├── utils.py       # Helper utilities (aggregation, metrics, etc.)
├── config.py      # Hyperparameters and configuration settings
├── main.py        # Entry point to run federated training
├── output.log     # Training logs
└── README.md      # Project documentation
