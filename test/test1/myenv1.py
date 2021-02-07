import random
import gym
import gym.spaces
import numpy as np

class GridEnv1(gym.Env):
    '''
    长度为10的线段，每次只能左右移动，节点标为0..9,
    起点为0，终点为9，超过100步则死亡-100
    到达9则胜利+10
    '''
    def __init__(self):
        self.action_space=gym.spaces.Discrete(2)
        self.observation_space=gym.spaces.Box(np.array([0]),np.array([9]))
        self.reset()



    def reset(self):
        '''

        :return: state
        '''
        self.observation = [0]
        #self.reward = 10
        self.done=False
        self.step_num=0
        return [0]

    def step(self, action)->tuple:
        '''

        :param action:
        :return: tuple ->[observation，reward，done，info]
        '''

        if action==0:
            action=-1
        self.observation[0]+=action
        self.step_num+=1
        reward=-1
        if self.step_num>100 or self.observation[0]<0:
            reward=-100
            self.done=True
            return self.observation,reward,self.done, {}
        if self.observation[0]==9:
            reward=100
            self.done=True

        return self.observation,reward,self.done,{}



    def render(self, mode='human'):
        pass

