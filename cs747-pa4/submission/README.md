# CS 747: Programming Assignment 4

## Windy Gridworld task: SARSA (0)

### Code Files

The submission contains the following files:

- agent.py (SARSA Agent Implementation)
- environment.py (Baseline, King Moves and Stochastic variants implementation of Windy Gridwork)
- check_env.py (To manually check the implementation)
- main.py 

- run.sh (Runs the code and generates the plot)

### Instructions to Run:

To generate baseline plot:

`$ ./run.sh --baseline`

To generate kingmoves plot:

`$ ./run.sh --kingmoves`

To generate stochastic environment plot:

`$ ./run.sh --stochastic`

Multiple plots can be generated as follows:

`$ ./run.sh --baseline --kingmoves --stochastic`   

By default, the code runs for 10 seed values. For a different number of seed runs (say 20) and running baseline and kingmoves variant, execute:

`$ ./run.sh --baseline --kingmoves --num_seed_runs 20`  

The bash script run.sh uses python3 to call the main script and has been tested on IITB CSE lab SL2 servers.
