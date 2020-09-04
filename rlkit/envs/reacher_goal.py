import numpy as np

from rlkit.envs.ant_multitask_base import MultitaskAntEnv
from . import register_env
from gym.envs.mujoco.reacher import ReacherEnv as ReacherEnv_

# Copy task structure from https://github.com/jonasrothfuss/ProMP/blob/master/meta_policy_search/envs/mujoco_envs/ant_rand_goal.py
@register_env('reacher-goal-sparse')
class ReacherGoalEnv_sparse(ReacherEnv_):
    def __init__(self, task={}, n_tasks=2, randomize_tasks=True, **kwargs):
        self.goals = self.sample_tasks(n_tasks)
        self.goal_radius = 0.09
        self._goal = [0,0,0.01]
        super(ReacherGoalEnv_sparse, self).__init__()
        self.reset_task(0)

    def get_all_task_idx(self):
        print(len(self.goals))
        return range(len(self.goals))

    def step(self, action):
        tmp_finger = self.get_body_com("fingertip")
        vec = self.get_body_com("fingertip") - self._goal

        reward_dist = - np.linalg.norm(vec)
        #print(vec,reward_dist)
        reward_ctrl = - np.square(action).sum()
        sparse_reward = self.sparsify_rewards(reward_dist)
        reward = reward_dist + reward_ctrl
        sparse_reward = sparse_reward + reward_ctrl
        reward = sparse_reward
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        done = False
        env_infos = dict(finger=tmp_finger.tolist(),reward_dist=reward_dist, reward_ctrl=reward_ctrl,sparse_reward=sparse_reward,goal=self._goal)
        return ob, reward, done, env_infos

    def sparsify_rewards(self, r):
        ''' zero out rewards when outside the goal radius '''
        mask = (r >= -self.goal_radius).astype(np.float32)
        r = r * mask
        if not r == 0:
            r += 0.2
        return r

    def sample_tasks(self, num_tasks):
        np.random.seed(1337)
        radius = np.random.uniform(0.2,0.25)
        angles = np.linspace(0, np.pi, num=num_tasks)
        xs = radius * np.cos(angles)
        ys = radius * np.sin(angles)
        heights = np.ones((num_tasks,), dtype=np.float32) * 0.01
        #print(xs.shape,heights.shape)
        goals = np.stack([xs, ys,heights], axis=1)

        #goals = np.stack([goals, heights], axis=1)
        np.random.shuffle(goals)
        goals = goals.tolist()
        return goals

    def reset_task(self, idx):
        ''' reset goal AND reset the agent '''
        self._goal = self.goals[idx]
        self.reset()