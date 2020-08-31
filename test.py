######## Runs test ########
# from utils.runs import Runs

# run = Runs()

# run.update_runs(1)

######### Generator test #########
from mazelab.generators import random_maze

maze = random_maze(height=12, width=12, density=0.1, complexity=0.10)

print(maze)