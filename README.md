# **Pusher-Mujoco**

## **Project Overview**
This project focuses on solving the multi-jointed robotic arm task in MuJoCo's Pusher environment by experimenting with various reinforcement learning algorithms.

---

## **Group Members**
- **Syed Arsal Rahman**
- **Syeda Eman Fatima**
- **Hunaina Ehsan**

---

## **Files and Descriptions**

### **1. Training Files**
- **`ppo.ipynb`**: Contains code for training the PPO model and saving its weights.  
- **`td3.ipynb`**: Contains code for training the TD3 model and saving its weights.  
- **`sac.ipynb`**: Contains code for training the SAC model and saving its weights.

### **2. Evaluation**
- **`evaluate_model.py`**: Used to run and render the environment and save a video of the agent's performance.  
  - Ensure to specify the weight file path and the exact model being used for evaluation.

### **3. Custom Wrapper**
- **HERCompatiblePusherWrapper**: A custom Gymnasium wrapper for the Pusher environment, enabling Hindsight Experience Replay (HER) with goal-conditioned observations and a binary reward function. Currently integrated with the SAC model.

### **4. Dependencies**
- **`requirements.txt`**: Contains all the required Python libraries for running the code.

---

## **Usage**
1. Install the dependencies from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
2. Train the desired model (PPO, TD3, or SAC) using the respective notebook.
3. Use evaluate_model.py to test the trained model, render the environment, and save a video.

---
   
## **Key Features**
1. Experimentation with PPO, TD3, and SAC for reinforcement learning in the Pusher environment.
2. Custom wrapper for Hindsight Experience Replay (HER) to handle sparse rewards efficiently.
