import numpy as np


class HMM:
    def __init__(self, num_states, num_obs, pi=None, A=None, B=None):
        self.num_states = num_states
        self.num_obs = num_obs

        if pi is None:
            self.pi = np.ones(num_states) / num_states
        else:
            self.pi = pi

        if A is None:
            self.A = np.ones((num_states, num_states)) / num_states
        else:
            self.A = A

        if B is None:
            self.B = np.ones((num_states, num_obs)) / num_obs
        else:
            self.B = B

    def forward(self, obs):
        T = len(obs)
        alpha = np.zeros((T, self.num_states))
        alpha[0, :] = self.pi * self.B[:, obs[0]]
        for t in range(1, T):
            alpha[t, :] = np.dot(alpha[t - 1, :], self.A) * self.B[:, obs[t]]
        return alpha

    def backward(self, obs):
        T = len(obs)
        beta = np.zeros((T, self.num_states))
        beta[T - 1, :] = 1
        for t in range(T - 2, -1, -1):
            beta[t, :] = np.dot(self.A, self.B[:, obs[t + 1]] * beta[t + 1, :])
        return beta

    def baum_welch_train(self, obs, num_iters=100):
        T = len(obs)
        for n in range(num_iters):
            alpha = self.forward(obs)
            beta = self.backward(obs)
            xi = np.zeros((T - 1, self.num_states, self.num_states))
            for t in range(T - 1):
                xi[t, :, :] = self.A * np.outer(alpha[t, :], beta[t + 1, :] * self.B[:, obs[t + 1]])
                xi[t, :, :] /= np.sum(xi[t, :, :])
            self.A = np.sum(xi, axis=0) / np.sum(xi, axis=(0, 1)).reshape(-1, 1)
            gamma = alpha * beta / np.sum(alpha * beta, axis=1).reshape(-1, 1)
            self.pi = gamma[0, :]
            self.B = np.zeros((self.num_states, self.num_obs))
            for i in range(self.num_states):
                for k in range(self.num_obs):
                    mask = (obs == k)
                    self.B[i, k] = np.sum(gamma[mask, i]) / np.sum(gamma[:, i])

    def viterbi_predict(self, obs):
        T = len(obs)
        delta = np.zeros((T, self.num_states))
        psi = np.zeros((T, self.num_states), dtype=int)
        delta[0, :] = self.pi * self.B[:, obs[0]]
        for t in range(1, T):
            for j in range(self.num_states):
                aux = delta[t - 1, :] * self.A[:, j]
                delta[t, j] = np.max(aux) * self.B[j, obs[t]]
                psi[t, j] = np.argmax(aux)
        path = np.zeros(T, dtype=int)
        path[T - 1] = np.argmax(delta[T - 1, :])
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path
