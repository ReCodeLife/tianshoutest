import gym


class Question:
    '''
    this class was used for describe the question
    '''
    def __init__(self):
        pass

class GymQuestion(Question):
    def __init__(self,gamename:str):
        '''

        :param gamename:gym
        '''
        super(GymQuestion, self).__init__()
        self.env=gym.make(gamename)
        self.state_shape = self.env.observation_space.shape or self.env.observation_space.n
        self.action_shape = self.env.action_space.shape or self.env.action_space.n


class DiyQuestion(Question):
    def __init__(self, gameclass:type ):
        '''
        :param gameclass: a class
        '''
        super().__init__()
        self.gameclass = gameclass
        self.env = gameclass()
        # 状态的纬度
        self.state_shape = self.env.observation_space.shape or self.env.observation_space.n
        # 行动数量
        self.action_shape = self.env.action_space.shape or self.env.action_space.n

