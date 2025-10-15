# Race Simulation 
 
![image](https://github.com/user-attachments/assets/20c42835-d007-4a02-a065-2173970bb711)

This branch contains the implementation for the simulated autonomous vehicle races featured in the paper: Vector Cost Bimatrix Games with Applications to Autonomous Racing. The code includes an object-oriented python GUI and data generator, which can be used in headless mode for improved performance. Note that this repository has been updated since the publication and some branches are under active construction. Main is the branch that will always be kept stable with new features as they come in. The legacy_mecc2025 branch is the frozen state of the code as of the publication date.

Citation:
```bash
@inproceedings{VectorCostBimatrix,
  title={Vector Cost Bimatrix Games with Applications to Autonomous Racing}, 
  author={Benjamin R. Toaz and Shaunak D. Bopardikar},
  year={2025},
  booktitle={Proceedings of the Modeling, Estimation and Control Conference},
  month = {October},
  year={2025},
  address={Pittsburgh, PA, USA},
  publisher={AACC},
  url={https://arxiv.org/abs/2507.05171},
}
```

## Abstract

We formulate a vector cost alternative to the scalarization method for weighting and
combining multi-objective costs. The algorithm produces solutions to bimatrix games that are
simultaneously pure, unique Nash equilibria and Pareto optimal with guarantees for avoiding
worst case outcomes. We achieve this by enforcing exact potential game constraints to guide
cost adjustments towards equilibrium, while minimizing the deviation from the original cost
structure. The magnitude of this adjustment serves as a metric for differentiating between
Pareto optimal solutions. We implement this approach in a racing competition between agents
with heterogeneous cost structures, resulting in fewer collision incidents with a minimal decrease
in performance. 


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
   
