"""
Optimal remote restoring via controlled interaction with environment.

"""
import functools
from dataclasses import dataclass

import numpy as np
import sympy as sp
from quanty import matrix
from quanty.base import sz
from quanty.problem.transfer import TransferZQCAlongChain
from quanty.state.coherence import coherence_matrix
from quanty.task.transfer_ import TransferZQCPerfectlyTask, as_real_imag


class TransferProblem(TransferZQCAlongChain):
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
        # return state
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
