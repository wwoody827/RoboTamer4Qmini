from env.legged_robot import LeggedRobotEnv

TASKS = {}


def register(cls):
    cls.name = cls.__name__
    assert cls.name not in TASKS, cls.name
    TASKS[cls.name] = cls
    return cls


def check(cls):
    assert any([cls == v for v in TASKS.values()]), \
        f"Please register the task:'{cls.name}'"


# Aliases: task names that map to another registered task class.
# Used to keep backward compat when task classes are merged.
_ALIASES = {
    'MIRLTask': 'BIRLTask',
}


def load_task_cls(name):
    name += "Task"
    name = _ALIASES.get(name, name)
    if name in TASKS:
        return TASKS[name]
    else:
        raise KeyError(f'Not exist task named {name}.')


class NullTask:

    def __init__(self, env: LeggedRobotEnv):
        check(self.__class__)
        self.env = env
        self.cfg = env.cfg
        self.debug = None
        self.device = env.device
        self.extra_info = {}
        self.num_observations = self.cfg.policy.num_observations
        self.num_actions = self.cfg.policy.num_actions

    def step(self):
        raise NotImplementedError

    def reset(self, env_ids):
        raise NotImplementedError

    def observation(self):
        raise NotImplementedError

    def action(self, net_out):
        raise NotImplementedError

    def reward(self):
        raise NotImplementedError

    def terminate(self):
        raise NotImplementedError

    def info(self):
        return self.extra_info
