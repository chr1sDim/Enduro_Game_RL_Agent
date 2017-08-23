import cv2
import numpy as np

from enduro.agent import Agent
from enduro.action import Action
import itertools


class FunctionApproximationAgent(Agent):
    def __init__(self):
        super(FunctionApproximationAgent, self).__init__()
        # Add member variables to your class here
        self.total_reward = 0
        self.features = np.zeros((4,11))
        self.prev_features = np.zeros((4,11))
        self.prev_speed = 0
        self.weights = np.random.uniform(0.0001,1.0,11)
        self.Q = np.zeros(4)
        """
        permutations = ["".join(seq) for seq in itertools.product("01", repeat=self.features.shape[0])]
        for state in permutations:
            self.Q[state] = np.array([0.1,0.1,1.0,0])
        """
        self.current_reward = 0
        self.a = 2
        self.a_prime = 2
        self.state = "0"*11
        self.state_prime = "0"*11

        self.learning_rate = 0.001
        self.gamma = 0.9
        self.epsilon = 0.01
        self.delta = 0

    def initialise(self, road, cars, speed, grid):
        """ Called at the beginning of an episode. Use it to construct
        the initial state.

        Args:
            road  -- 2-dimensional array containing [x, y] points
                     in pixel coordinates of the road grid
            cars  -- dictionary which contains the location and the size
                     of the agent and the opponents in pixel coordinates
            speed -- the relative speed of the agent with respect the others
            gird  -- 2-dimensional numpy array containing the latest grid
                     representation of the environment

        For more information on the arguments have a look at the README.md
        """

        # Reset the total reward for the episode
        self.delta = 0
        self.total_reward = 0
        self.max_reward = 0
        pos_me, pos_enemy, depth_enemy = self.find_positions(grid)

        # Create feature space
        self.features = np.zeros((4,11))
        self.update_features(grid, speed, pos_me, pos_enemy, depth_enemy)
        self.prev_features = self.features
        self.prev_speed = speed


    def act(self):
        """ Implements the decision making process for selecting
        an action. Remember to store the obtained reward.
        """
        # You can get the set of possible actions and print them with:
        # print [Action.toString(a) for a in self.getActionsSet()]

        # Execute the action and get the received reward signal
        self.Q = np.dot(self.weights.T,self.features.T)
        best_move = np.argmax(self.Q)
        if np.random.uniform(0., 1.) < self.epsilon:
            Q_s = self.Q
            probs = np.exp(Q_s) / np.sum(np.exp(Q_s))
            idx = np.random.choice(4, p=probs)
            if best_move == 0:
                action = Action.LEFT
            elif best_move == 1:
                action = Action.RIGHT
            elif best_move == 2:
                action = Action.ACCELERATE
            else:
                action = Action.BRAKE
            self.a = idx
        else:
            action_id = best_move
            #action = Action.NOOP
            if action_id == 0:
                action = Action.LEFT
            elif action_id == 1:
                action = Action.RIGHT
            elif action_id == 2:
                action = Action.ACCELERATE
            else:
                action = Action.BRAKE
            self.a = best_move

        self.current_reward = self.move(action)
        self.total_reward += self.current_reward
        # IMPORTANT NOTE:
        # 'action' must be one of the values in the actions set,
        # i.e. Action.LEFT, Action.RIGHT, Action.ACCELERATE or Action.BRAKE
        # Do not use plain integers between 0 - 3 as it will not work


    def sense(self, road, cars, speed, grid):
        """ Constructs the next state from sensory signals.

            cars  -- dictionary which contains the location and the size
                     of the agent and the opponents in pixel coordinates
            speed -- the relative speed of the agent with respect the others
            gird  -- 2-dimensional numpy array containing the latest grid
                     representation of the environment

        For more information on the arguments have a look at the README.md
        """
        pos_me, pos_enemy, depth_enemy = self.find_positions(grid)

        # Create feature space
        self.features = np.zeros((4,11))
        self.update_features(grid, speed, pos_me, pos_enemy, depth_enemy)
        self.prev_speed = speed


    def learn(self):
        """ Performs the learning procedure. It is called after act() and
        sense() so you have access to the latest tuple (s, s', a, r).
        """

        Q_s = np.dot(self.weights.T, self.prev_features[self.a].T)
        Q_s_prime = np.max(np.dot(self.weights.T, self.features.T))
        self.delta = (self.current_reward + self.gamma*Q_s_prime-Q_s)
        self.weights = self.weights + self.learning_rate * (self.current_reward + self.gamma*Q_s_prime-Q_s) * self.prev_features[self.a]
        self.prev_features = self.features




    def callback(self, learn, episode, iteration):
        #print "{0}/{1}: {2}".format(episode, iteration, self.total_reward)
        global ep
        global prev_reward
        if ep!= episode:
            rewards[episode-1] = prev_reward
            deltas[episode-1] = self.delta
            ep = episode
            print('Episode: {0}, Reward: {1}'.format(episode,prev_reward))
        if self.total_reward>self.max_reward:
            max_rewards[episode-1] = self.total_reward
            self.max_reward = self.total_reward
        prev_reward = self.total_reward

        # You could comment this out in order to speed up iterations
        #cv2.imshow("Enduro", self._image)
        #cv2.waitKey(40)

    def find_positions(self, grid):
        [[pos_agent]] = np.argwhere(grid[0, :] == 2)
        pos_enemy = 10
        depth_enemy = 11
        flag  = True
        for j in range(1,11):
            for i in range(0,10):
                if grid[j,i]==1 and flag:
                    pos_enemy = i
                    depth_enemy = j
                    flag = False

        return pos_agent, pos_enemy, depth_enemy

    def update_features(self,grid, speed, pos_me, pos_enemy, depth_enemy):
        if pos_me>pos_enemy+1:
            self.features[0,1] = 0
            self.features[1,1] = 0
            self.features[2,1] = 1
            self.features[3,1] = 0
        if pos_enemy == 10:
            self.features[0,0] = 0
            self.features[1,0] = 0
            self.features[2,0] = 1
            self.features[3,0] = 0
        if pos_me<pos_enemy-1:
            self.features[0,2] = 0
            self.features[1,2] = 0
            self.features[2,2] = 1
            self.features[3,2] = 0
        if (pos_me==pos_enemy) or (pos_me-1==pos_enemy) or (pos_me+1==pos_enemy):
            self.features[0,3] = 1
            self.features[1,3] = 1
            self.features[2,3] = 0
            self.features[3,3] = 1
        if depth_enemy > 6:
            self.features[0,4] = 0
            self.features[1,4] = 0
            self.features[2,4] = 1
            self.features[3,4] = 0
        if depth_enemy <= 6 and depth_enemy > 3:
            self.features[0,5] = 1
            self.features[1,5] = 1
            self.features[2,5] = 1
            self.features[3,5] = 0
        if depth_enemy <= 3:
            self.features[0,6] = 1
            self.features[1,6] = 1
            self.features[2,6] = 1
            self.features[3,6] = 0
        if grid[0,1]==2:
            self.features[0,7] = 0
            self.features[1,7] = 1
            self.features[2,7] = 0
            self.features[3,7] = 0
        if grid[0,8]==2:
            self.features[0,8] = 1
            self.features[1,8] = 0
            self.features[2,8] = 0
            self.features[3,8] = 0
        if speed == -50:
            self.features[0,9] = 0
            self.features[1,9] = 0
            self.features[2,9] = 1
            self.features[3,9] = 0
        if self.prev_speed < speed:
            self.features[0,10] = 1
            self.features[1,10] = 1
            self.features[2,10] = 1
            self.features[3,10] = 0

if __name__ == "__main__":
    rewards = np.zeros(500)
    max_rewards = np.zeros(500)
    deltas = np.zeros(500)
    ep = 0
    prev_reward = 0
    a = FunctionApproximationAgent()
    a.run(True, episodes=500, draw=True)
    np.savetxt('total_rand.out', rewards)
    np.savetxt('max_rand.out', max_rewards)
    np.savetxt('weights_rand.out', a.weights)
    np.savetxt('deltas_rand.out', deltas)
