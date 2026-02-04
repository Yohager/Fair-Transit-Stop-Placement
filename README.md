Experimental source code for the Fair Transit Stop Placement project.

#### Requirements
Python >= 3.9
gurobipy >= 13.0.0
matplotlib >= 3.10.8
numpy >= 2.4.1
pandas >= 3.0.0
scipy >= 1.17.0

#### Reproduction
Go to the root folder and run `python3 main.py` with the following commands.

```
python3 main.py rounds n m k candidate_mode check_type transit_scaling output_dir
```
for example:
```
python3 main.py 50 400 800 20 0 "JR" 0 ./data/results/
```
Parameters:
- rounds: the number of rounds for sampling
- n: number of agents
- m: number of candidate stops
- k: number of stop selection
- candidate_mode: 0 represents candidate stops from agents' locations; 1 represents candidate stops from the global sampling pool
- check_type: JR or core
- transit_scaling: value for scaling the transit cost
- output_dir: folder for storing the results

