import abc

from diylib.program.program import Program
from diylib.question.question import Question


class Resolve:

    def __init__(self, que: Question, prg: Program):
        self.que = que
        self.prg = prg

    @abc.abstractmethod
    def solve(self):
        pass
