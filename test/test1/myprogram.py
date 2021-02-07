from diylib.program.program import Program
from tianshou.trainer import offpolicy_trainer
from tianshou.data import ReplayBuffer
from typing import List


class DQNProgram(Program):
    def __init__(self, eps_train:float, eps_test:float, epoch:int, hidden_sizes: List[int],
                 buffer_size:int, gamma=0.9, n_step=3,
                 device='cpu', lr=1e-3, target_update_freq=320,
                 training_num=1, test_num=1, batch_size=64,
                 step_per_epoch=1, collect_per_step=10, policy_dir=None):
        super(DQNProgram, self).__init__()
        self.train_envs=None
        self.test_envs = None


        self.policy = None  # 策略
        self.net = None  # 网络
        self.optim = None  # 优化器
        self.gamma = gamma
        self.n_step = n_step
        self.target_update_freq = target_update_freq
        self.policy_dir=policy_dir



        self.eps_train = eps_train
        self.eps_test = eps_test
        self.buffer_size=buffer_size
        self.buf = ReplayBuffer(buffer_size)  # 缓存大小
        self.train_collector = None
        self.test_collector = None
        self.reward_threshold = None
        self.epoch = epoch
        self.hidden_sizes = hidden_sizes
        self.device = device
        self.lr = lr
        self.step_per_epoch = step_per_epoch
        self.collect_per_step = collect_per_step
        self.training_num = training_num
        self.test_num = test_num
        self.batch_size = batch_size
        self.ready = False


        # 停止条件 平均回报大于阈值

    def stop_fn(self, reward_threshold:float):
        return lambda mean_rewards: mean_rewards >= reward_threshold

    # 学习前调用的函数
    # 在每个epoch训练之前被调用的函数，输入的是当前第几轮epoch和当前用于训练的env一共step了多少次。
    # 此处为了实现根据一个世代中的迭代次数改变eps(贪婪策略的比例)
    def train_fn(self, epoch, env_step:int):
        # eps annnealing, just a demo
        if env_step <= 1000:
            self.policy.set_eps(self.eps_train)
        elif env_step <= 3000:
            eps = self.eps_train - (env_step - 1000) / \
                  2000 * (0.9 * self.eps_train)
            self.policy.set_eps(eps)
        else:
            self.policy.set_eps(0.1 * self.eps_train)

    def test_fn(self, epoch, env_step):
        self.policy.set_eps(self.eps_test)

    def sovle(self):
        if self.ready:
            return offpolicy_trainer(
                self.policy, self.train_collector, self.test_collector, self.epoch,
                self.step_per_epoch, self.collect_per_step, self.test_num,
                self.batch_size, train_fn=self.train_fn, test_fn=self.test_fn,
                stop_fn=self.stop_fn(self.reward_threshold))

        else:
            raise Exception('unkown error ,maybe you should use init() in class resolve')


