import pickle
import time
from diylib.resolve.resolve import Resolve
import torch
from tianshou.policy import DQNPolicy, BasePolicy
from tianshou.data import Collector, ReplayBuffer
from myquetsion import DQNQuestion
from myprogram import DQNProgram
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
import numpy as np
import os


class DQNResolve(Resolve):
    def __init__(self, que: DQNQuestion, prg: DQNProgram,seed=None):
        super(DQNResolve, self).__init__(que,prg)
        self.prg.train_envs = DummyVectorEnv(
            [lambda: self.que.gameclass() for _ in range(self.prg.training_num)])
        # test_envs = gym.make(args.task)
        self.prg.test_envs = DummyVectorEnv(
            [lambda: self.que.gameclass() for _ in range(self.prg.test_num)])
        self.set_seed(seed)
        # 为了复现，种子设置必须放在环境构建之后，net等其他参数之前
        # 因为net在初始化的时候会调用np的随机数，放在之前才能保证复现
        self.prg.net = Net(self.que.state_shape, self.que.action_shape,
                           hidden_sizes=self.prg.hidden_sizes, device=self.prg.device,
                           # dueling=(Q_param, V_param),
                           ).to(self.prg.device)  # 网络
        self.prg.optim = torch.optim.Adam(self.prg.net.parameters(), lr=self.prg.lr)  # 优化器
        self.prg.policy = DQNPolicy(
            self.prg.net, self.prg.optim, self.prg.gamma, self.prg.n_step,
            target_update_freq=self.prg.target_update_freq)  # 策略
        if not self.prg.policy_dir is None:
            t=torch.load(self.prg.policy_dir)
            self.prg.policy.load_state_dict(t)



        self.prg.train_collector = Collector(self.prg.policy, self.prg.train_envs, self.prg.buf)
        self.prg.test_collector = Collector(self.prg.policy, self.prg.test_envs)

        self.prg.reward_threshold = self.que.reward_threshold
        self.prg.ready = True

    def set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.prg.train_envs.seed(seed)
        self.prg.test_envs.seed(seed)

    def solve(self,modelsave=False,bufsave=False):
        ans=self.prg.sovle()
        if modelsave:
            self.saveModel()
        if bufsave:
            self.evalAndSaveBuf()
        return  ans



    def evalAndSaveBuf(self,savedir:str=None):
        # 测试过程存储
        # 路径
        if savedir is None:
            tdir=os.path.join('../..', 'buflog', type(self.que).__name__)
            if not os.path.exists(tdir):
                os.makedirs(tdir)
            savedir=os.path.join(tdir,time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime())+'.pkl')
        #
        buf = ReplayBuffer(self.prg.buffer_size)
        collector = Collector(self.prg.policy, self.prg.test_envs, buf)
        # 与环境进行交互，具体为每走一步就判断是否有环境结束，如果环境结束，则将环境所走的步数加入总步数
        # n_step为其最少的步数，即大于则收集结束
        collector.collect(n_step=self.prg.buffer_size)
        print(buf)
        # 测试过程存储
        pickle.dump(buf, open(savedir, "wb"))

    def saveModel(self,savedir=None):
        if savedir is None:
            tdir = os.path.join('../..', 'model', type(self.que).__name__)
            if not os.path.exists(tdir):
                os.makedirs(tdir)
            savedir=os.path.join(tdir,time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime())+'.pth')
        torch.save(self.prg.policy.state_dict(), savedir)
        # policy.load_state_dict(torch.load('dqn.pth'))





