import numpy as np
from environment import Maze
from agent import Agent
import matplotlib.pyplot as plt

if __name__ == '__main__':
    maze = Maze()
    other_maze = Maze()
    robot = Agent(maze.maze, alpha=0.1, random_factor=0.25)
    moveHistory = []

    for i in range(3000):
        if i % 100 == 0:
            print()
            print("Round: ", i)

        while not maze.is_game_over():
            state, _ = maze.get_state_and_reward() # get the current state
            action = robot.choose_action(state, maze.allowed_states[state]) # choose an action (explore or exploit)
            maze.update_maze(action) # update the maze according to the action

            y, x = maze.robot_position
            other_maze.update_other_maze(y, x)

            state, reward = maze.get_state_and_reward() # get the new state and reward
            robot.update_state_history(state, reward) # update the robot memory with state and reward
            if maze.steps > 1000:
                # end the robot if it takes too long to find the goal
                maze.robot_position = (5, 5)
        
        robot.learn() # robot should learn after every episode
        moveHistory.append(maze.steps) # get a history of number of steps taken to plot later

        if i % 100 == 0:
            other_maze.print_maze()
            print("Steps robot needed: ", maze.steps)

        maze = Maze() # reinitialize the maze
        other_maze = Maze()

plt.semilogy(moveHistory, "b-")
plt.show()
