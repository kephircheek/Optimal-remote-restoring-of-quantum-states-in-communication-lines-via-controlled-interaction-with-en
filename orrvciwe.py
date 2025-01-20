"""
Optimal remote restoring via controlled interaction with environment.

"""
import functools
from dataclasses import dataclass, replace

import numpy as np
import sympy as sp
import timeti
from scipy import optimize
from scipy import linalg
from quanty import matrix
from quanty.base import sz
from quanty.problem.transfer import TransferZQCAlongChain
from quanty.state.coherence import coherence_matrix
from quanty.task.transfer_ import TransferZQCPerfectlyTask, as_real_imag


class TransferProblem(TransferZQCAlongChain):
    @classmethod
    def init_classic(cls, hamiltonian, length: int, n_sender: int, **kwargs):
        n_ancillas = length - n_sender  # in quanty ancillas means not a reciever
        return super().init_classic(hamiltonian, length, n_sender, n_ancillas, **kwargs)

    def coherence_matrix(self):
        symbols, m = coherence_matrix(order=1, basis=self.sender_basis)
        r01 = sp.S("r01")
        r02 = sp.S("r02")
        symbols = {r01: (0, 1), r02: (0, 2)}
        m[0, 1] = r01
        m[0, 2] = r02
        m[1, 0] = r01.conjugate()
        m[2, 0] = r02.conjugate()

        sender_basis_reversed = self.sender_basis.reversed()
        m = self.sender_basis.reorder(m, sender_basis_reversed)
        return symbols, m

    @staticmethod
    def free_corners(mat: matrix.Matrix) -> matrix.Matrix:
        return mat

    def _is_extra_element(self, i, j):
        return False


def dephasing_values(gammas, ex):
    values = 0
    n = len(gammas)
    for i, gamma in enumerate(gammas):
        if gamma == 0:
            continue
        sz_ = matrix.crop(sz(n, i).toarray(), ex=ex).astype(float) / 2
        _values = np.diag(sz_).reshape(-1, 1)
        _values = _values @ _values.reshape(1, -1)
        _values -= 0.25
        _values = _values * gamma
        values += _values
    return values


def indentity_like(mat):
    return np.eye(mat.shape[0])


def left_superoperator(mat):
    indentity = indentity_like(mat)
    return np.kron(indentity, mat)


def right_superoperator(mat):
    indentity = indentity_like(mat)
    return np.kron(mat.transpose(), indentity)


def dephasing_superoperator(gammas, ex):
    values = 0
    n = len(gammas)
    for i, gamma in enumerate(gammas):
        if gamma == 0:
            continue
        sz_ = matrix.crop(sz(n, i).toarray(), ex=ex).astype(float) / 2
        _values = np.kron(sz_, sz_)  # left_superoperator(sz_) @ right_superoperator(sz_)
        _values -= indentity_like(_values) * 0.25
        _values = _values * gamma
        values += _values
    return values


@functools.cache
def dephasing_operator(t, gammas, ex):
    values = dephasing_values(gammas, ex=ex)
    exp = np.exp
    if any(isinstance(g, sp.Symbol) for g in gammas):
        exp = np.vectorize(sp.exp)
    return exp(values * t)


def dephasing(rho, t, gammas, ex=None):
    operator = dephasing_operator(t, gammas, ex=ex)
    return rho * operator


@dataclass(frozen=True)
class TransferTask(TransferZQCPerfectlyTask):
    tuning_time: int = 0
    nt_per_step: int = 3

    def tune(self, state):
        if self.nt_per_step is None:
            return self.tune_in_liouville_space(state)
        return self.tune_by_trotterization(state)

    def tune_by_trotterization(self, state):
        nt = self.nt_per_step * len(self.features)
        tt = self.tuning_time / nt
        u = self.problem.hamiltonian.U(self.problem.length, tt, ex=self.problem.ex)
        gammas = iter(self.features)
        ul = dephasing_operator(tt, next(gammas), ex=self.problem.ex)
        t = 0
        for it in range(nt):
            if it != 0 and it % self.nt_per_step == 0:
                ul = dephasing_operator(tt, next(gammas), ex=self.problem.ex)
            state = state * ul
            state = u @ state @ u.conjugate().transpose()
            t += tt
        return state

    def tune_in_liouville_space(self, state):
        n_state_rows = state.shape[0]
        state = state.transpose().flatten().reshape(-1, 1)
        ham = self.problem.hamiltonian(self.problem.length, ex=self.problem.ex)
        super_ham = left_superoperator(ham) - right_superoperator(ham)
        dt = self.tuning_time / len(self.features)
        for gammas in self.features:
            super_do = dephasing_superoperator(gammas, ex=self.problem.ex)
            superoperator = super_ham + 1j * super_do
            u = linalg.expm(-1j * superoperator * dt)
            state = u @ state
        state = state.reshape(n_state_rows, -1).T
        return state

    def _residuals(self, var, impact):
        k, m = self.problem.sender_params[var]
        return sum(
            (
                as_real_imag(impact[i, j])
                for i, j in self.problem.sender_params.values()
                if (i, j) != (k, m)
            ),
            tuple(),
        )

    def perfect_transferred_state_residuals(self, use_cache=True) -> np.ndarray:
        return np.array(
            sum(
                (
                    self._residuals(v, impact)
                    for v, impact in self.receiver_state_impacts(use_cache=use_cache).items()
                ),
                tuple(),
            )
        )

    def perfect_transferred_state_quality(self, use_cache=True) -> np.ndarray:
        return np.array(
            sum(
                (
                    as_real_imag(impact[self.problem.sender_params[v]])
                    for v, impact in self.receiver_state_impacts(use_cache=use_cache).items()
                ),
                tuple(),
            )
        )


@dataclass(frozen=True)
class FitTransferTask:
    task: TransferTask
    optimization_kwargs: dict = None
    regularization_importance: float = 1e-6
    polish: bool = True
    polish_kwargs: dict = None

    @property
    def n_controlled_nodes(self):
        return len(self.task.features[0])

    def reshape_features(self, x):
        features = x.reshape(-1, self.n_controlled_nodes)
        n_steps = features.shape[0]
        if features.shape[1] != self.task.problem.length:
            features = np.hstack(
                (np.zeros((n_steps, self.task.problem.length - self.n_controlled_nodes)), features)
            )
        features = tuple([tuple(row) for row in features])
        return features

    def residuals(self, x):
        features = self.reshape_features(x)
        task = replace(self.task, features=features)
        r = task.perfect_transferred_state_residuals(use_cache=False)
        return r

    def residuals_and_quality(self, x):
        features = self.reshape_features(x)
        task = replace(self.task, features=features)
        q = self.regularization_importance * np.abs(task.perfect_transferred_state_quality())
        r = task.perfect_transferred_state_residuals(use_cache=False)
        return np.hstack((r, 1 - q))

    def run(self):
        sw = timeti.Stopwatch()
        optimization_kwargs = (
            dict() if self.optimization_kwargs is None else self.optimization_kwargs
        )
        optimization_kwargs.setdefault("bounds", (0, 1))
        optimization_kwargs.setdefault("xtol", 1e-10)
        optimization_kwargs.setdefault("ftol", 1e-15)
        optimization_kwargs.setdefault("gtol", 1e-12)
        if self.regularization_importance is None:
            fun = self.residuals
        else:
            fun = self.residuals_and_quality
        x0 = np.array(self.task.features).flatten()
        optimization_res = optimize.least_squares(fun, x0, **optimization_kwargs)

        features = self.reshape_features(optimization_res.x)

        if self.polish and self.regularization_importance is None:
            import warnings

            warnings.warn("Warning: 'regularization_importance' is None but 'polish' is True")

        polish_res = None
        if self.polish:
            x0 = optimization_res.x
            polish_kwargs = dict() if self.polish_kwargs is None else self.polish_kwargs
            polish_kwargs.setdefault("bounds", (0, 1))
            polish_kwargs.setdefault("xtol", 1e-10)
            polish_kwargs.setdefault("ftol", 1e-15)
            polish_kwargs.setdefault("gtol", 1e-12)
            polish_res = optimize.least_squares(self.residuals, x0, **polish_kwargs)
            features = self.reshape_features(polish_res.x)

        execution_time = sw.timestamp
        return FitTransferTaskResult(
            task=self,
            features=features,
            optimization_result=optimization_res,
            polish_result=polish_res,
            execution_time=execution_time,
        )


@dataclass(frozen=True)
class FitTransferTaskResult:
    task: FitTransferTask
    features: tuple[tuple[float, ...], ...]
    optimization_result: optimize.OptimizeResult
    polish_result: optimize.OptimizeResult = None
    execution_time: float = None
