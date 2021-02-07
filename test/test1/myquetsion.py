from diylib.question.question import DiyQuestion

class DQNQuestion(DiyQuestion):
    def __init__(self,gameclass:type,reward_threshold:float):
        super(DQNQuestion, self).__init__(gameclass)
        self.reward_threshold=reward_threshold