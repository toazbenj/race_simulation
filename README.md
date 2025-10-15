# Race Simulation 
 
![image](https://github.com/user-attachments/assets/20c42835-d007-4a02-a065-2173970bb711)

`race_simulation` is a Python-based simulation using Pygame to implement trajectory planning algorithms for racing scenarios.

## Features

- Trajectory Planning Algorithms: Implements a simple model predictive controller for trajectory selection
- Game Theory: Vector and Scalar cost bimatrix games played between two opponent cars, one attacker and one defender
- Reinforcement Learning Branch: Additional project for lane keeping with PID, image processing, and throttle learning by a DQN model

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/toazbenj/race_simulation.git
   ```

2. **Install Required Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the simulation:
1. **Edit the configuration file constants.py in the bike_race folder.**
   
2. **Execute the Main Script within the bike_race folder**:

   ```bash
   cd ~/race_simulation
   cd bike_race
   python main.py
   ```

   This will launch the simulation window.

3. **Interact with the Simulation**:

   You can skip to the next race using the big red skip button.
   
4. **Gather data and change parameters**:

   The constants.py file in the bike_race folder allows you to change the race settings. Data from the race will accumulate in this file:
   ```bash
   WRITE_FILE = "../data/test.csv"
   ```
   
