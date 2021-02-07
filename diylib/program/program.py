import abc



class Program:
    '''
    this class was used for describe the program
    '''

    def stop_fn(self, reward_threshold:float):
        pass



    def train_fn(self, epoch, env_step:int):
        pass

    def test_fn(self, epoch, env_step):
        pass

    @abc.abstractmethod
    def sovle(self):
        pass



