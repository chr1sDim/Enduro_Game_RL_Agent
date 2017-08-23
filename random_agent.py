import cv2
from enduro.agent import Agent
from enduro.action import Action
from enduro.state import EnvironmentState
import numpy as np
import matplotlib.pyplot as plt


plt.style.use('ggplot')
seed = 24102016
rng = np.random.RandomState(seed)


class RandomAgent(Agent):
    def __init__(self):
        super(RandomAgent, self).__init__()
        # Add member variables to your class here
        self.total_reward = 0
        self.max_reward = 0


    def initialise(self, grid):
        """ Called at the beginning of an episode. Use it to construct
        the initial state.
        """
        # Reset the total reward for the episode
        self.total_reward = 0
        self.max_reward = 0
        cv2.imshow("Enduro", self._image)
        cv2.imshow("Environment Grid", EnvironmentState.draw(grid))
        print grid

    def act(self):
        """ Implements the decision making process for selecting
        an action. Remember to store the obtained reward.
        """
        action_id = np.random.randint(4)
        #action = Action.NOOP
        if action_id == 0:
            action = Action.LEFT
        elif action_id == 1:
            action = Action.RIGHT
        elif action_id == 2:
            action = Action.ACCELERATE
        else:
            action = Action.BREAK


        # You can get the set of possible actions and print it with:
        # print [Action.toString(a) for a in self.getActionsSet()]

        # Execute the action and get the received reward signal
        # IMPORTANT NOTE:
        # 'action' must be one of the values in the actions set,
        # i.e. Action.LEFT, Action.RIGHT, Action.ACCELERATE or Action.BREAK
        # Do not use plain integers between 0 - 3 as it will not work
        # self.total_reward += self.move(action)
        self.total_reward += self.move(action)
    def sense(self, grid):
        """ Constructs the next state from sensory signals.

        gird -- 2-dimensional numpy array containing the latest grid
                representation of the environment
        """
        # Visualise the environment grid
        cv2.imshow("Environment Grid", EnvironmentState.draw(grid))
        print grid

    def learn(self):
        """ Performs the learning procudre. It is called after act() and
        sense() so you have access to the latest tuple (s, s', a, r).
        """
        pass

    def callback(self, learn, episode, iteration):
        """ Called at the end of each timestep for reporting/debugging purposes.
        """
        print "{0}/{1}: {2}".format(episode, iteration, self.total_reward)
        global ep
        global prev_reward
        if ep!= episode:
            rewards[episode-1] = prev_reward
            ep = episode
        if self.total_reward>self.max_reward:
            max_rewards[episode-1] = self.total_reward
            self.max_reward = self.total_reward
        prev_reward = self.total_reward
        # Show the game frame only if not learning
        if not learn:
            cv2.imshow("Enduro", self._image)
            cv2.waitKey(40)



if __name__ == "__main__":
    rewards = np.zeros(100)
    max_rewards = np.zeros(100)
    ep = 0
    prev_reward = 0
    a = RandomAgent()
    a.run(True, episodes=100, draw=True)
    np.savetxt('random_total.out', rewards)
    np.savetxt('random_max.out',max_rewards)
    print 'Total reward: ' + str(a.total_reward)
