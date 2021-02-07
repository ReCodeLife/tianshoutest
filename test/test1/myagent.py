import pprint
from myresolve import DQNResolve
from myprogram import DQNProgram
from myquetsion import DQNQuestion
from myenv1 import GridEnv1

def zero_train():
    reslv = DQNResolve(DQNQuestion(GridEnv1,80.0),
                    DQNProgram(0.1, 0.05, 1 ,[64,128,128,64], 10000,
                                        training_num=8, test_num=1,
                                        step_per_epoch=100),
                    seed=1626
                    )
    #reslv.evalAndSave()
    result=reslv.solve(modelsave=False,bufsave=False)
    pprint.pprint(result)

def read_eval():
    reslv = DQNResolve(DQNQuestion(GridEnv1,80.0),
                       DQNProgram(0.1, 0.05, 1, [64, 128, 128, 64], 10000,
                                  training_num=8, test_num=100,
                                  step_per_epoch=100, policy_dir='../../model/DiyQuestion/2021_02_06_14_44_47.pth'),
                       seed=1626
                       )
    # reslv.evalAndSave()
    reslv.evalAndSaveBuf()
    #pprint.pprint(result)


if __name__=='__main__':
    zero_train()
    #read_eval()