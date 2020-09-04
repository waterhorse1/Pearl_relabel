
import math
import numpy as np
from gym import spaces
from gym.utils import seeding
from gym import utils
from . import register_env
import time

#from metaworld.benchmarks.base import Benchmark
from metaworld.core.serializable import Serializable
from metaworld.envs.mujoco.multitask_env import MultiClassMultiTaskEnv
from metaworld.envs.mujoco.env_dict import HARD_MODE_ARGS_KWARGS, HARD_MODE_CLS_DICT

@register_env('ML6_test')
class ML6(MultiClassMultiTaskEnv, Benchmark, Serializable):

    def __init__(self, env_type='train', n_tasks = 2, randomize_tasks=True, sample_all=False):
        self._serializable_initialized = True
        assert env_type == 'train' or env_type == 'test'
        Serializable.quick_init(self, locals())
        hard_cls_dict = dict(train=dict((k, HARD_MODE_CLS_DICT['train'][k]) for k in ('push-v1', 'button-press-v1', 'sweep-into-v1', 'plate-slide-v1')),
             test=dict((m, HARD_MODE_CLS_DICT['train'][m]) for m in ('coffee-button-v1', 'drawer-close-v1')))
        hard_args_dict = dict(train=dict((k, HARD_MODE_ARGS_KWARGS['train'][k]) for k in
                                        ('push-v1', 'button-press-v1', 'sweep-into-v1', 'plate-slide-v1')),
                             test=dict((m, HARD_MODE_ARGS_KWARGS['train'][m]) for m in
                                       ('coffee-button-v1', 'drawer-close-v1')))
        cls_dict = hard_cls_dict[env_type]
        args_kwargs = hard_args_dict[env_type]

        super().__init__(
            task_env_cls_dict=cls_dict,
            task_args_kwargs=args_kwargs,
            sample_goals=True,
            obs_type='plain',
            sample_all=sample_all)


        ##self._max_plain_dim = 9
        #ML1.__init__(self, task_name=task_name, env_type=env_type, n_goals=50)
        #def initsample(self, n_tasks,randomize_tasks=True):
        self.tasks = self.sample_tasks(n_tasks)
        self.reset_task(0)
    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self.set_task(self.tasks[idx])
        self._goal = self.active_env.goal
        # assume parameterization of task by single vector
    def get_all_task_idx(self):
        return range(len(self.tasks))
