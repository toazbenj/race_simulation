# Race Simulation - SEMBAS

![image](https://github.com/toazbenj/race_simulation/blob/sembas/videos/overlay_grid.png)

This branch contains the bridge API to connect the repo to the State space Exploration of Multidimemsional Boundaries using Adherence Strategies (SEMBAS) software tool. We use this in the paper for parameter selection that drives the players in the simulation up to the point of failure and beyond, saving each of the useful parameter configurations for viewing and analysis. Boundary data from our experiments is in the data folder. You can view it in an interactive streamlit browser using the viewer.py script. In the video folder we also include several example edge cases taken from the boundary as well as scripts for parsing the data and creating the visualizations.

## Installation

1. **Get Rust**:

   Find instructions here: [Rust](https://rust-lang.org/tools/install/).
 
2. **Clone SEMBAS**:

   You can find it at [SEMBAS](https://github.com/toazbenj/racing_sembas/tree/main) or just use the command below. Note that this fork is very similar to the original SEMBAS repo except for a few minor adjustments to the parameters to account for 3 dimensional boundaries.
   
   ```bash
   git clone https://github.com/toazbenj/racing_sembas.git
   ```

3. **Checkout the race_simulation branch**:
   
   ```bash
   git checkout example-v0.4.x-race_simulation
   ```

## Basic Usage (assuming setup is building off the steps in main)
   
1. **Launch the race_simulation API**:

   In your first terminal:
   
   ```bash
   cd ~/race_simulation
   ```
  
   ```bash
   python3 bike_race/parameterization.py
   ```

3. **Launch SEMBAS (will install many packages on the first time)**:

   In your second terminal:

   ```bash
   cd ~/SEMBAS
   ```
  
   ```bash
   cargo run --example race_simulation --features all
   ```

4. **Data Gathering and viewing**:

   The data will be stored in the file data/test.json by default in the race_simulation directory. Edit this in the bike_race/constants.py file.

   You can look at the premade data on the [Race Simulation](https://racesimulation.streamlit.app/) web app. You can also plot your own results by editing the data/viewer.py file and launching it yourself.
   
   ```bash
   cd ~/race_simulation
   ```

   ```bash
   streamlit run data/viewer.py
   ```
   
