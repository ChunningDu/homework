import numpy as np
from cost_functions import trajectory_cost_fn
import time


class Controller():
    def __init__(self):
        pass

    # Get the appropriate action(s) for this state(s)
    def get_action(self, state):
        pass


class RandomController(Controller):
    def __init__(self, env):
        self._env = env

    def get_action(self, state):
        """ YOUR CODE HERE """
        return self._env.action_space.sample()


class MPCcontroller(Controller):
    """
    Controller built using the MPC method outlined in
    https://arxiv.org/abs/1708.02596
    """

    def __init__(self, env, dyn_model, horizon=5,
                 cost_fn=None, num_simulated_paths=10):
        self.env = env
        self.dyn_model = dyn_model
        self.horizon = horizon
        self.cost_fn = cost_fn
        self.num_simulated_paths = num_simulated_paths

    def get_action(self, state):
        """ YOUR CODE HERE """
        # Note: batch your simulations through the model for speed

        # sample K sequences of actions
        actions = []
        for _ in range(self.horizon):
            actions.append(
                [self.env.action_space.sample()
                 for _ in range(self.num_simulated_paths)])

        # use dynamics model to generate simulated rollouts
        states = [[state] * self.num_simulated_paths]
        next_states = []
        for t in range(self.horizon):
            ns = self.dyn_model.predict(states[t], actions[t])
            states.append(ns)
            next_states.append(ns)
        states = states[:-1]

        states = np.swapaxes(np.asarray(states), 0, 1)
        actions = np.swapaxes(np.asarray(actions), 0, 1)
        next_states = np.swapaxes(np.asarray(next_states), 0, 1)

        costs = [trajectory_cost_fn(self.cost_fn, states[j],
                                    actions[j], next_states[j])
                 for j in range(self.num_simulated_paths)]
        idx = np.argmin(costs)
        return actions[idx][0]
