import unittest

import numpy as np
import qutip as qp
import scipy.linalg
from quanty import matrix
from quanty.base import sz
from quanty.geometry import UniformChain
from quanty.hamiltonian import XX, XXZ
from quanty.model.homo import Homogeneous
from quanty.task.transfer_ import FitTransmissionTimeTask

from orrvciwe import (
    TransferProblem,
    TransferTask,
    dephasing,
    dephasing_operator,
    dephasing_superoperator,
    indentity_like,
    left_superoperator,
    right_superoperator,
)


def tune_by_mesolve(task, state):
    n = task.problem.length
    ham = task.problem.hamiltonian(task.problem.length, ex=task.problem.ex)
    options = {
        "atol": 1e-13,
        "rtol": 1e-12,
    }
    ham = qp.Qobj(ham)
    state = qp.Qobj(state)
    dt = task.tuning_time / len(task.features)
    for gammas in task.features:
        c_ops = [
            [
                np.sqrt(gamma)
                * qp.Qobj(matrix.crop(sz(n, i).toarray(), ex=task.problem.ex).astype(float) / 2)
            ]
            for i, gamma in enumerate(gammas)
            if gamma != 0
        ]
        result = qp.mesolve(ham, state, [0, dt], c_ops=c_ops, options=options)
        state = result.states[-1]
    state = state.full()
    return state


def tune_by_trotterization_in_liouville_space(task, state):
    n_state_rows = state.shape[0]
    state = state.transpose().flatten().reshape(-1, 1)
    nt = task.nt_per_step * len(task.features)
    tt = task.tuning_time / nt
    ham = task.problem.hamiltonian(task.problem.length, ex=task.problem.ex)
    super_ham = left_superoperator(ham) - right_superoperator(ham)
    super_ham_u = scipy.linalg.expm(-1j * super_ham * tt)
    gammas = iter(task.features)
    super_do = dephasing_superoperator(next(gammas), ex=task.problem.ex)
    super_do_u = scipy.linalg.expm(super_do * tt)
    t = 0
    for it in range(nt):
        if it != 0 and it % task.nt_per_step == 0:
            super_do = dephasing_superoperator(next(gammas), ex=task.problem.ex)
            super_do_u = scipy.linalg.expm(super_do * tt)
        if super_do_u.shape == (1, 1):
            super_do_u = indentity_like(super_ham_u) * super_do_u[0, 0]
        state = super_do_u @ state
        state = super_ham_u @ state
        t += tt
    state = state.reshape(n_state_rows, -1).T
    return state


class TestTransferN6S2E1Task(unittest.TestCase):
    def setUp(self):
        excitations = 1
        length = 6
        n_sender = 2
        geometry = UniformChain()
        model = Homogeneous(geometry)  # All with all !
        hamiltonian = XXZ(model)
        self.problem = TransferProblem.init_classic(
            hamiltonian,
            length=length,
            n_sender=n_sender,
            excitations=n_sender if excitations is None else excitations,
        )
        self.transmission_time = 20
        self.tuning_time = 10

    def test_coeffs(self):
        features_raw = """
            {Gam[6][1] -> 0.7467966572771598, Gam[6][2] -> -0.05452340990634395,
             Gam[6][3] -> -0.0016970523999852466, Gam[6][4] -> 1.6616712163587286,
             Gam[5][1] -> 0.7841842353957827, Gam[5][2] -> 0.4675460310633446,
             Gam[5][3] -> 0.7936298905913943, Gam[5][4] -> 1.8475007474774925,
             Gam[4][1] -> 1.519445304120249, Gam[4][2] -> 1.409607025097288,
             Gam[4][3] -> 1.1637673536099047, Gam[4][4] -> 0.2726030636535845,
             Gam[3][1] -> 0.1111924085973801, Gam[3][2] -> 0.16046209900820732,
             Gam[3][3] -> 0.7948706012383544, Gam[3][4] -> 1.0471904022349234}
        """

        import re

        features_processed = [
            re.search(r"\w+\[(\d+)\]\[(\d+)\] -> ([-\.\d]+)", v.strip()).groups()
            for v in features_raw.split(",")
        ]

        n_steps = max(int(j) for _, j, _ in features_processed)

        features = [
            [
                0,
            ]
            * self.problem.length
            for _ in range(n_steps)
        ]
        for i_spin, i_time, value in features_processed:
            features[int(i_time) - 1][int(i_spin) - 1] = abs(float(value))

        features = tuple(tuple(row) for row in features)

        task = TransferTask(
            problem=self.problem,
            transmission_time=self.transmission_time - self.tuning_time,
            tuning_time=self.tuning_time,
            features=features,
            nt_per_step=4,
        )
        impacts = task.receiver_state_impacts(use_cache=False)
        desired_coefs = (
            (0.000625758, 0.00560245, 0, 0),
            (0, 0, 0.0205902, 0.0141093),
        )
        for (v, impact), desired in zip(impacts.items(), desired_coefs):
            with self.subTest(v):
                left, right = impact[0, 1:]
                np.testing.assert_allclose(
                    [left.real, left.imag, right.real, right.imag],
                    desired,
                    atol=1e-7,
                )


class TestTransferTaskN5S2E1TrooterVsLiouvilleSpace(unittest.TestCase):
    def setUp(self):
        length = 5
        n_sender = 2
        excitations = 1

        geometry = UniformChain()
        model = Homogeneous(geometry)  # All with all !
        hamiltonian = XXZ(model)
        self.problem = TransferProblem.init_classic(
            hamiltonian,
            length=length,
            n_sender=n_sender,
            excitations=n_sender if excitations is None else excitations,
        )
        self.transmission_time = 4.56617

    def assert_r01_r02_coefficients_trotter_vs_lspace(self, features, tuning_time, nt_per_step):
        task, task_ = (
            TransferTask(
                problem=self.problem,
                transmission_time=self.transmission_time - tuning_time,
                tuning_time=tuning_time,
                features=features,
                nt_per_step=nt_per_step,
            )
            for nt_per_step in (nt_per_step, None)
        )
        self.assert_r01_r02_coefficients(task, task_)

    def assert_r01_r02_coefficients_trotter_vs_mesolve(self, features, tuning_time, nt_per_step):
        class TransferTask_(TransferTask):
            def tune_in_liouville_space(self, state):
                return tune_by_mesolve(self, state)

        task, task_ = (
            TransferTask_(
                problem=self.problem,
                transmission_time=self.transmission_time - tuning_time,
                tuning_time=tuning_time,
                features=features,
                nt_per_step=nt_per_step,
            )
            for nt_per_step in (nt_per_step, None)
        )
        self.assert_r01_r02_coefficients(task, task_)

    def assert_r01_r02_coefficients_trotter_vs_lspace_trotter(
        self, features, tuning_time, nt_per_step
    ):
        class TransferTask_(TransferTask):
            def tune_by_trotterization(self, state):
                return tune_by_trotterization_in_liouville_space(self, state)

        task, task_ = (
            cls(
                problem=self.problem,
                transmission_time=self.transmission_time - tuning_time,
                tuning_time=tuning_time,
                features=features,
                nt_per_step=nt_per_step,
            )
            for cls in (TransferTask, TransferTask_)
        )
        self.assert_r01_r02_coefficients(task_, task)

    def assert_r01_r02_coefficients(self, task, task_):
        (r01, _), (r02, _) = sorted(
            [(k, v) for k, v in task.problem.sender_params.items()], key=lambda x: x[-1]
        )
        impacts = task.receiver_state_impacts(use_cache=False)
        r01_coeff = impacts[r01][0, 1:]
        r02_coeff = impacts[r02][0, 1:]
        impacts_ = task_.receiver_state_impacts(use_cache=False)
        r01_coeff_ = impacts_[r01][0, 1:]
        r02_coeff_ = impacts_[r02][0, 1:]
        np.testing.assert_allclose(r01_coeff_, r01_coeff, atol=1e-4, rtol=1e-3)
        np.testing.assert_allclose(r02_coeff_, r02_coeff, atol=1e-4, rtol=1e-3)

    def test_without_gammas(self):
        nt_per_step = 1
        features = ((0, 0, 0, 0, 0),)
        tuning_time = 3
        decimals = 14
        self.assert_r01_r02_coefficients_trotter_vs_lspace(features, tuning_time, nt_per_step)
        self.assert_r01_r02_coefficients_trotter_vs_mesolve(features, tuning_time, nt_per_step)
        self.assert_r01_r02_coefficients_trotter_vs_lspace_trotter(features, tuning_time, 1)

    def test_single_peak(self):
        nt_per_step = int(1e2)
        tuning_time = 5
        features = ((0.2, 0.1, 0.2, 0.1, 0.2),)
        self.assert_r01_r02_coefficients_trotter_vs_lspace(features, tuning_time, nt_per_step)
        self.assert_r01_r02_coefficients_trotter_vs_mesolve(features, tuning_time, nt_per_step)
        self.assert_r01_r02_coefficients_trotter_vs_lspace_trotter(features, tuning_time, 10)

    def test_multiple_peaks(self):
        nt_per_step = int(1e3)
        tuning_time = 5
        features = (
            (0, 0, 0.5, 1.5, 1.9),
            (0, 0, 1.4, 0.4, 2.2),
            (0, 0, 0.2, 1.2, 0.7),
        )
        self.assert_r01_r02_coefficients_trotter_vs_lspace(features, tuning_time, nt_per_step)
        self.assert_r01_r02_coefficients_trotter_vs_mesolve(features, tuning_time, nt_per_step)
        self.assert_r01_r02_coefficients_trotter_vs_lspace_trotter(features, tuning_time, 10)


class TestTransferN5S2E1Task(unittest.TestCase):
    def setUp(self):
        length = 5
        n_sender = 2
        excitations = 1

        geometry = UniformChain()
        model = Homogeneous(geometry)  # All with all !
        hamiltonian = XXZ(model)
        self.problem = TransferProblem.init_classic(
            hamiltonian,
            length=length,
            n_sender=n_sender,
            excitations=n_sender if excitations is None else excitations,
        )
        self.transmission_time = 4.56617

    def assert_r01_r02_coefficients(self, features, tuning_time, r01_coeff, r02_coeff):
        task = TransferTask(
            problem=self.problem,
            transmission_time=self.transmission_time - tuning_time,
            tuning_time=tuning_time,
            features=features,
        )
        impacts = task.receiver_state_impacts(use_cache=False)
        (r01, _), (r02, _) = sorted(
            [(k, v) for k, v in task.problem.sender_params.items()], key=lambda x: x[-1]
        )
        r01_coeff_ = impacts[r01][0, 1:]
        r02_coeff_ = impacts[r02][0, 1:]
        np.testing.assert_array_almost_equal(r01_coeff, r01_coeff_, decimal=12)
        np.testing.assert_array_almost_equal(r02_coeff, r02_coeff_, decimal=12)

    def test_transmission_time(self):
        task = FitTransmissionTimeTask(self.problem)
        result = task.run()
        self.assertEqual(result.transmission_time, self.transmission_time)

    def test_without_gammas(self):
        features = ((0, 0, 0, 0, 0),)
        tuning_time = 0
        r01 = (
            0.0226319621661733 + 0.244791124096331 * 1j,
            0.152932276556253 + 0.35985410196903 * 1j,
        )
        r02 = (
            0.152932276556253 + 0.359854101969031 * 1j,
            0.257935007412467 + 0.745556446958056 * 1j,
        )
        self.assert_r01_r02_coefficients(features, tuning_time, r01, r02)

    def test_single_peak(self):
        tuning_time = 3
        features = ((0, 0, 0.1, 0.2, 0.3),)
        r01 = (
            0.0137550371567518 + 0.203238181875406 * 1j,
            0.150693559305259 + 0.301630765688891 * 1j,
        )
        r02 = (
            0.0997458212786445 + 0.3047247557502 * 1j,
            0.219880444451306 + 0.620358605346225 * 1j,
        )
        self.assert_r01_r02_coefficients(features, tuning_time, r01, r02)

    def test_multiple_peaks(self):
        tuning_time = 3
        features = (
            (0, 0, 0.5, 1.5, 1.9),
            (0, 0, 1.4, 0.4, 2.2),
            (0, 0, 0.2, 1.2, 0.7),
        )
        r01 = (
            0.0102535037251317 + 0.0917824441431267 * 1j,
            0.0979116742995736 + 0.100079653619466 * 1j,
        )
        r02 = (
            0.0255478395304316 + 0.155740682188356 * 1j,
            0.0877138595782704 + 0.246525748551345 * 1j,
        )
        self.assert_r01_r02_coefficients(features, tuning_time, r01, r02)


class TestDephasing(unittest.TestCase):
    def test_pure_dephasing(self):
        cat = (qp.fock(2, 1) + qp.fock(2, 0)) / np.sqrt(2)
        rho = qp.tensor(cat, cat)
        rho = rho * rho.dag()

        gammas = (0.1, 0.2)
        t = 100

        dephased_rho = dephasing(rho.full(), t, gammas)

        c_ops = [
            np.sqrt(gammas[0]) * qp.tensor(qp.sigmaz(), qp.identity(2)) / 2,
            np.sqrt(gammas[1]) * qp.tensor(qp.identity(2), qp.sigmaz()) / 2,
        ]
        ham = qp.tensor(qp.identity(2), qp.identity(2))
        rho_dephased_ = qp.mesolve(
            ham,
            rho,
            [0, t],
            c_ops=c_ops,
        ).states[-1]

        np.testing.assert_array_almost_equal(rho_dephased_.full(), dephased_rho)

    def test_trotted_dephasing(self):
        cat = (qp.fock(2, 1) + qp.fock(2, 0)) / np.sqrt(2)
        ket = qp.tensor(cat, cat)
        rho = ket.proj()

        gammas = (0.1, 0.2)
        t = 10

        c_ops = [
            np.sqrt(gammas[0]) * qp.tensor(qp.sigmaz(), qp.identity(2)) / 2,
            np.sqrt(gammas[1]) * qp.tensor(qp.identity(2), qp.sigmaz()) / 2,
        ]
        ham = 0.25 * (
            qp.tensor(qp.sigmax(), qp.sigmax())
            + qp.tensor(qp.sigmay(), qp.sigmay())
            - 2 * qp.tensor(qp.sigmaz(), qp.sigmaz())
        )
        options = {
            "atol": 1e-13,
            "rtol": 1e-12,
        }
        rho_dephased = qp.mesolve(
            ham, rho, np.linspace(0, t, 2), c_ops=c_ops, options=options
        ).states[-1]

        geometry = UniformChain()
        model = Homogeneous(geometry)  # All with all !
        ham_ = XXZ(model)
        rho_ = rho.full()
        nt = int(1e5)
        tt = t / nt
        u = ham_.U(2, tt)
        ul = dephasing_operator(tt, gammas, ex=None)
        for it in range(nt):
            rho_ = u @ rho_ @ u.conjugate().transpose()
            rho_ = rho_ * ul
        rho_dephased_ = rho_

        np.testing.assert_array_almost_equal(ham.full(), ham_(2), decimal=12)
        np.testing.assert_allclose(rho_dephased.full(), rho_dephased_, atol=1e-6, rtol=1e-5)
