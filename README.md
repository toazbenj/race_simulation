# Race Simulation 
 
![image](https://github.com/user-attachments/assets/20c42835-d007-4a02-a065-2173970bb711)

`race_simulation` is a Python-based simulation using Pygame to implement trajectory planning algorithms for bike racing scenarios.

## Features

- Trajectory Planning Algorithms: Implements a simple model predictive controller for trajectory selection
- Game Theory: Vector and Scalar cost bimatrix games played between two opponent cars, one attacker and one defender
- Reinforcement Learning Branch: Additional project for lane keeping with PID, image processing, and throttle learning by a DQN model

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/toazbenj/race_simulation.git
   ```

2. **Navigate to the Project Directory**:

   ```bash
   cd race_simulation/bike_race
   ```

3. **Install Required Dependencies**:

   Ensure that Python and Pygame are installed on your system. You can install Pygame using pip:

   ```bash
   pip install pygame
   ```

## Usage

To run the simulation:
1. **Edit the configuration file constants.py in the bike_race folder.**
   
2. **Execute the Main Script within the bike_race folder**:

   ```bash
   python main.py
   ```

   This will launch the simulation window.

3. **Interact with the Simulation**:

   You can skip to the next race using the big red skip button
