#

# a session of one task running

__all__ = [
    "AgentSession",
]

from .utils import get_unique_id

class AgentSession:
    def __init__(self, id=None, task="", **kwargs):
        self.id = id if id is not None else get_unique_id("S")
        self.info = {}
        self.info.update(kwargs)
        self.task = task  # target task
        self.steps = []  # a list of dicts to indicate each step's running, simply use dict to max flexibility

    def to_dict(self):
        return self.__dict__.copy()

    def from_dict(self, data: dict):
        for k, v in data.items():
            assert k in self.__dict__
            self.__dict__[k] = v

    @classmethod
    def init_from_dict(cls, data: dict):
        ret = cls()
        ret.from_dict(data)
        return ret

    @classmethod
    def init_from_data(cls, task, steps=(), **kwargs):
        ret = cls(**kwargs)
        ret.task = task
        ret.steps.extend(steps)
        return ret

    def num_of_steps(self):
        return len(self.steps)

    def get_current_step(self):
        return self.get_specific_step(idx=-1)

    def get_specific_step(self, idx: int):
        return self.steps[idx]

    def get_latest_steps(self, count=0, include_last=False):
        if count <= 0:
            ret = self.steps if include_last else self.steps[:-1]
        else:
            ret = self.steps[-count:] if include_last else self.steps[-count-1:-1]
        return ret

    def add_step(self, step_info):
        self.steps.append(step_info)
