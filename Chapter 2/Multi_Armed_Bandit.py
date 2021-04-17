import numpy as np
from joblib import Parallel, delayed
from numpy.random import RandomState
import warnings


class Bandit:
    def __init__(self, n_arms: int = 10, eps: float = 0.1, dtype=None, seed: bool = None):
        """Initializes k-armed Bandit."""
        self.n_arms = n_arms
        self.eps = eps
        self.dtype = np.float16 if dtype is None else dtype
        self.seed = seed
        self.N = np.zeros(n_arms)
        self.Q = np.zeros(n_arms)
        self.action_means = np.random.normal(size=(self.n_arms,), scale=1.0)

    def get_reward(self, action, pull):
        if self.seed: np.random.seed(pull)
        reward = np.random.normal(loc=self.action_means, size=self.n_arms)
        # print('reward: ', reward)

        if self.seed: np.random.seed(pull)
        noise = np.random.randn(self.n_arms)

        reward = reward + noise
        return reward[action]

    def get_action(self, pull=None):
        if self.seed: np.random.seed(pull)
        if np.random.rand() > self.eps and np.any(self.Q):
            action = np.argmax(self.Q)
        else:
            action = np.random.randint(0, self.n_arms)
        return action

    def update_Q(self, action, reward):
        self.N[action] += 1
        self.Q[action] += 1 / self.N[action] * (reward - self.Q[action])
        return self.Q[action]

    def experiment(self, n_pulls: int):
        """Run experiment n_pulls times. Updates return Q on each iteration.

        Args:
            n_pulls (int): number of pulls

        Returns: numpy array of shape (n_pulls, n_arms)
        """
        self.N = np.zeros(self.n_arms)
        self.Q = np.zeros(self.n_arms)
        Q_history = np.array([])

        for pull in range(n_pulls):
            action = self.get_action(pull)
            reward = self.get_reward(action, pull)
            Q = self.update_Q(action, reward).copy().astype(self.dtype)
            Q_history = np.append(Q_history, Q)

        print(Q_history)
        return Q_history.reshape((n_pulls,))

    def experiments(self, n_exp, n_pulls, threads=None):
        if self.seed:
            warnings.warn(
                f'seed should be set to False in order to run independent experiments')

        def iterate(n_exp):
            experiments = np.zeros((n_pulls,))
            for i in range(n_exp):
                experiments += self.experiment(n_pulls)
            return experiments

        if threads:
            n_exp_list = [n_exp // threads] * threads
            n_exp_list[0] += n_exp % threads
            result = Parallel(n_jobs=threads)(
                delayed(iterate)(n_exp) for n_exp in n_exp_list
            )
            experiments = sum(result)
        else:
            experiments = iterate(n_exp)

        return experiments / np.float(n_exp)


bandit = Bandit(n_arms=4, seed=0)
experiment = bandit.experiment(4)
print(experiment)

experiments = bandit.experiments(5, 5)
print(experiments)