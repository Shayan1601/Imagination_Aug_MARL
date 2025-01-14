#Defining the Treasure Finder environment class
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
import gymnasium
from gymnasium import spaces
import numpy as np

class TreasureFinderEnv(gymnasium.Env):
    def __init__(self, map_size):
        super(TreasureFinderEnv, self).__init__()
        self.map_size = map_size
        if map_size < 7:
            self.map_size = 7
        n_agents = 2
        self.action_space = spaces.MultiDiscrete([4] * 2)  # Four discrete actions: 0, 1, 2, 3 for # for two agents
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.map_size, self.map_size, 3), dtype=np.float32)
        
        self.half_pos = int((self.map_size - 1)/2)

        self.occupancy = np.zeros((self.map_size, self.map_size))
        for i in range(self.map_size):
            self.occupancy[0, i] = 1
            self.occupancy[i, 0] = 1
            self.occupancy[i, self.map_size - 1] = 1
            self.occupancy[self.map_size - 1, i] = 1
            self.occupancy[self.half_pos, i] = 1

        self.lever_pos = [self.map_size - 2, self.map_size - 2]

        # initialize agent 1
        self.agt1_pos = [self.half_pos+1, 1]
        self.occupancy[self.agt1_pos[0]][self.agt1_pos[1]] = 1

        # initialize agent 2
        self.agt2_pos = [self.map_size-2, 1]
        self.occupancy[self.agt2_pos[0]][self.agt2_pos[1]] = 1

        # initialize treasure
        self.treasure_pos = [1, self.map_size - 2]
        # self.treasure_pos = [self.half_pos - 1, self.half_pos]

        # sub pos = [self.map_size - 2, self.map_size - 2]
        self.sub_pos = [self.map_size - 3, self.map_size - 2]


    def reset(self,seed=None):
        self.occupancy = np.zeros((self.map_size, self.map_size))
        for i in range(self.map_size):
            self.occupancy[0, i] = 1
            self.occupancy[i, 0] = 1
            self.occupancy[i, self.map_size - 1] = 1
            self.occupancy[self.map_size - 1, i] = 1
            self.occupancy[self.half_pos, i] = 1

        self.lever_pos = [self.map_size - 2, self.map_size - 2]

        # initialize agent positions
        self.agt1_pos = [self.half_pos + 1, 1]
        self.agt2_pos = [self.map_size - 2, 1]

        # initialize treasure position
        self.treasure_pos = [1, self.map_size - 2]

        # sub position
        self.sub_pos = [self.map_size - 3, self.map_size - 2]

        # Reset the occupancy grid with agent positions
        self.occupancy[self.agt1_pos[0], self.agt1_pos[1]] = 1
        self.occupancy[self.agt2_pos[0], self.agt2_pos[1]] = 1

        # Create the initial state for each agent
        state = np.zeros((7, 7, 3))
        state[self.agt1_pos[0], self.agt1_pos[1], 0] = 1
        state[self.agt2_pos[0], self.agt2_pos[1], 1] = 1
        state[self.treasure_pos[0], self.treasure_pos[1], 2] = 1

        return state,[]

    def step(self, action_list):
        reward1 = 0
        reward = 0
        done = False
        # agent1 move
        if action_list[0] == 0:  # move up
            if self.occupancy[self.agt1_pos[0]-1][self.agt1_pos[1]] != 1:  # if can move
                self.agt1_pos[0] = self.agt1_pos[0] - 1
                self.occupancy[self.agt1_pos[0]+1][self.agt1_pos[1]] = 0
                self.occupancy[self.agt1_pos[0]][self.agt1_pos[1]] = 1
            else:
                reward = reward - 0.1
        elif action_list[0] == 1:  # move down
            if self.occupancy[self.agt1_pos[0]+1][self.agt1_pos[1]] != 1:  # if can move
                self.agt1_pos[0] = self.agt1_pos[0]+1
                self.occupancy[self.agt1_pos[0]-1][self.agt1_pos[1]] = 0
                self.occupancy[self.agt1_pos[0]][self.agt1_pos[1]] = 1
            else:
                reward = reward - 0.1
        elif action_list[0] == 2:  # move left
            if self.occupancy[self.agt1_pos[0]][self.agt1_pos[1]-1] != 1:  # if can move
                self.agt1_pos[1] = self.agt1_pos[1] - 1
                self.occupancy[self.agt1_pos[0]][self.agt1_pos[1]+1] = 0
                self.occupancy[self.agt1_pos[0]][self.agt1_pos[1]] = 1
            else:
                reward = reward - 0.1
        elif action_list[0] == 3:  # move right
            if self.occupancy[self.agt1_pos[0]][self.agt1_pos[1]+1] != 1:  # if can move
                self.agt1_pos[1] = self.agt1_pos[1] + 1
                self.occupancy[self.agt1_pos[0]][self.agt1_pos[1]-1] = 0
                self.occupancy[self.agt1_pos[0]][self.agt1_pos[1]] = 1
            else:
                reward = reward - 0.1

        # agent2 move
        if action_list[1] == 0:  # move up
            if self.occupancy[self.agt2_pos[0]-1][self.agt2_pos[1]] != 1:  # if can move
                self.agt2_pos[0] = self.agt2_pos[0] - 1
                self.occupancy[self.agt2_pos[0]+1][self.agt2_pos[1]] = 0
                self.occupancy[self.agt2_pos[0]][self.agt2_pos[1]] = 1
            else:
                reward = reward - 0.1
        elif action_list[1] == 1:  # move down
            if self.occupancy[self.agt2_pos[0]+1][self.agt2_pos[1]] != 1:  # if can move
                self.agt2_pos[0] = self.agt2_pos[0] + 1
                self.occupancy[self.agt2_pos[0]-1][self.agt2_pos[1]] = 0
                self.occupancy[self.agt2_pos[0]][self.agt2_pos[1]] = 1
            else:
                reward = reward - 0.1
        elif action_list[1] == 2:  # move left
            if self.occupancy[self.agt2_pos[0]][self.agt2_pos[1]-1] != 1:  # if can move
                self.agt2_pos[1] = self.agt2_pos[1] - 1
                self.occupancy[self.agt2_pos[0]][self.agt2_pos[1]+1] = 0
                self.occupancy[self.agt2_pos[0]][self.agt2_pos[1]] = 1
            else:
                reward = reward - 0.1
        elif action_list[1] == 3:  # move right
            if self.occupancy[self.agt2_pos[0]][self.agt2_pos[1]+1] != 1:  # if can move
                self.agt2_pos[1] = self.agt2_pos[1] + 1
                self.occupancy[self.agt2_pos[0]][self.agt2_pos[1]-1] = 0
                self.occupancy[self.agt2_pos[0]][self.agt2_pos[1]] = 1
            else:
                reward = reward - 0.1

        # check lever
        if self.agt1_pos == self.lever_pos or self.agt2_pos == self.lever_pos:
            self.occupancy[self.half_pos][self.half_pos] = 0  # open secret door
            self.occupancy[self.half_pos][self.half_pos-1] = 0  # open secret door
            self.occupancy[self.half_pos][self.half_pos+1] = 0  # open secret door
        else:
            self.occupancy[self.half_pos][self.half_pos] = 1  # don't open secret door/close secret door
            self.occupancy[self.half_pos][self.half_pos - 1] = 1  # don't open secret door/close secret door
            self.occupancy[self.half_pos][self.half_pos + 1] = 1  # don't open secret door/close secret door

        # check treasure
        if self.agt1_pos == self.treasure_pos or self.agt2_pos == self.treasure_pos:
            reward = reward + 100

        if (self.agt1_pos == self.sub_pos and self.agt2_pos == self.lever_pos) or (self.agt1_pos == self.lever_pos and self.agt2_pos == self.sub_pos):
            reward = reward + 3

            done = False
        if reward > 0:
            done = True
        #calculating global obs
        obs = np.zeros((self.map_size, self.map_size, 3))
        for i in range(self.map_size):
            for j in range(self.map_size):
                if self.occupancy[i][j] == 0:
                    obs[i, j, 0] = 1.0
                    obs[i, j, 1] = 1.0
                    obs[i, j, 2] = 1.0
        obs[self.lever_pos[0], self.lever_pos[1], 0] = 1.0
        obs[self.lever_pos[0], self.lever_pos[1], 1] = 1.0
        obs[self.lever_pos[0], self.lever_pos[1], 2] = 0.0
        obs[self.treasure_pos[0], self.treasure_pos[1], 0] = 0.0
        obs[self.treasure_pos[0], self.treasure_pos[1], 1] = 1.0
        obs[self.treasure_pos[0], self.treasure_pos[1], 2] = 0.0
        obs[self.agt1_pos[0], self.agt1_pos[1], 0] = 1.0
        obs[self.agt1_pos[0], self.agt1_pos[1], 1] = 0.0
        obs[self.agt1_pos[0], self.agt1_pos[1], 2] = 0.0
        obs[self.agt2_pos[0], self.agt2_pos[1], 0] = 0.0
        obs[self.agt2_pos[0], self.agt2_pos[1], 1] = 0.0
        obs[self.agt2_pos[0], self.agt2_pos[1], 2] = 1.0
        obs[self.sub_pos[0], self.sub_pos[1], 0] = 1.0
        obs[self.sub_pos[0], self.sub_pos[1], 1] = 0.0
        obs[self.sub_pos[0], self.sub_pos[1], 2] = 1.0
        state = obs  
        truncated = False  # You can adjust this based on your specific use case
        return state, reward, done, truncated,{}
    
    def render(self, mode='human'):
        obs = self.get_global_obs()
        enlarge = 30
        new_obs = np.ones((self.map_size*enlarge, self.map_size*enlarge, 3))
        for i in range(self.map_size):
            for j in range(self.map_size):

                if obs[i][j][0] == 0.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 0.0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (0, 0, 0), -1)
                if obs[i][j][0] == 1.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 0.0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (0, 0, 255), -1)
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 1.0 and obs[i][j][2] == 0.0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (0, 255, 0), -1)
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 1.0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (255, 0, 0), -1)
                if obs[i][j][0] == 1.0 and obs[i][j][1] == 1.0 and obs[i][j][2] == 0.0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (0, 255, 255), -1)
                if obs[i][j][0] == 1.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 1.0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (255, 0, 255), -1)
        cv2.imshow('image', new_obs)
        cv2.waitKey(100)

    # ... (rest of the methods)
    
    def plot_scene(self):
        fig = plt.figure(figsize=(5, 5))
        gs = GridSpec(3, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        plt.xticks([])
        plt.yticks([])
        ax2 = fig.add_subplot(gs[2, 0:1])
        plt.xticks([])
        plt.yticks([])
        ax3 = fig.add_subplot(gs[2, 1:2])
        plt.xticks([])
        plt.yticks([])

        ax1.imshow(self.get_global_obs())
        ax2.imshow(self.get_agt1_obs())
        ax3.imshow(self.get_agt2_obs())

        plt.show()
        
    def get_state(self):

        state = np.zeros((1, 4))
        state[0, 0] = self.agt1_pos[0] / self.map_size
        state[0, 1] = self.agt1_pos[1] / self.map_size
        state[0, 2] = self.agt2_pos[0] / self.map_size
        state[0, 3] = self.agt2_pos[1] / self.map_size
        return state   

    def get_agt1_obs(self):
        obs = np.zeros((3, 3, 3))
        for i in range(3):
            for j in range(3):
                if self.occupancy[self.agt1_pos[0]-1+i][self.agt1_pos[1]-1+j] == 0:
                    obs[i, j, 0] = 1.0
                    obs[i, j, 1] = 1.0
                    obs[i, j, 2] = 1.0
        d_x = self.lever_pos[0] - self.agt1_pos[0]
        d_y = self.lever_pos[1] - self.agt1_pos[1]
        if d_x>=-1 and d_x<=1 and d_y>=-1 and d_y<=1:
            obs[1+d_x, 1+d_y, 0] = 1.0
            obs[1+d_x, 1+d_y, 1] = 1.0
            obs[1+d_x, 1+d_y, 2] = 0.0
        d_x = self.treasure_pos[0] - self.agt1_pos[0]
        d_y = self.treasure_pos[1] - self.agt1_pos[1]
        if d_x >= -1 and d_x <= 1 and d_y >= -1 and d_y <= 1:
            obs[1+d_x, 1+d_y, 0] = 0.0
            obs[1+d_x, 1+d_y, 1] = 1.0
            obs[1+d_x, 1+d_y, 2] = 0.0
        d_x = self.agt2_pos[0] - self.agt1_pos[0]
        d_y = self.agt2_pos[1] - self.agt1_pos[1]
        if d_x >= -1 and d_x <= 1 and d_y >= -1 and d_y <= 1:
            obs[1 + d_x, 1 + d_y, 0] = 0.0
            obs[1 + d_x, 1 + d_y, 1] = 0.0
            obs[1 + d_x, 1 + d_y, 2] = 1.0
        obs[1, 1, 0] = 1.0
        obs[1, 1, 1] = 0.0
        obs[1, 1, 2] = 0.0
        return obs

    def get_agt2_obs(self):
        obs = np.zeros((3, 3, 3))
        for i in range(3):
            for j in range(3):
                if self.occupancy[self.agt2_pos[0]-1+i][self.agt2_pos[1]-1+j] == 0:
                    obs[i, j, 0] = 1.0
                    obs[i, j, 1] = 1.0
                    obs[i, j, 2] = 1.0
        d_x = self.lever_pos[0] - self.agt2_pos[0]
        d_y = self.lever_pos[1] - self.agt2_pos[1]
        if d_x>=-1 and d_x<=1 and d_y>=-1 and d_y<=1:
            obs[1+d_x, 1+d_y, 0] = 1.0
            obs[1+d_x, 1+d_y, 1] = 1.0
            obs[1+d_x, 1+d_y, 2] = 0.0
        d_x = self.treasure_pos[0] - self.agt2_pos[0]
        d_y = self.treasure_pos[1] - self.agt2_pos[1]
        if d_x >= -1 and d_x <= 1 and d_y >= -1 and d_y <= 1:
            obs[1+d_x, 1+d_y, 0] = 0.0
            obs[1+d_x, 1+d_y, 1] = 1.0
            obs[1+d_x, 1+d_y, 2] = 0.0
        d_x = self.agt1_pos[0] - self.agt2_pos[0]
        d_y = self.agt1_pos[1] - self.agt2_pos[1]
        if d_x >= -1 and d_x <= 1 and d_y >= -1 and d_y <= 1:
            obs[1 + d_x, 1 + d_y, 0] = 1.0
            obs[1 + d_x, 1 + d_y, 1] = 0.0
            obs[1 + d_x, 1 + d_y, 2] = 0.0
        obs[1, 1, 0] = 0.0
        obs[1, 1, 1] = 0.0
        obs[1, 1, 2] = 1.0
        return obs    
         
    def get_obs(self):
        return [self.get_agt1_obs(), self.get_agt2_obs()]


