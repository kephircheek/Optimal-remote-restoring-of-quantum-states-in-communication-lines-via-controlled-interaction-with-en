import unittest

import numpy as np
import qutip as qp
from quanty.geometry import UniformChain
from quanty.hamiltonian import XX, XXZ
from quanty.model.homo import Homogeneous
from quanty.task.transfer_ import FitTransmissionTimeTask

from orrvciwe import TransferProblem, TransferTask, dephasing, dephasing_operator


class TestTransferTask(unittest.TestCase):
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
            n_ancillas=length - n_sender,  # in quanty ancillas means not a reciever
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
            [(k, v) for k, v in self.problem.sender_params.items()], key=lambda x: x[-1]
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
        np.testing.assert_array_almost_equal(rho_dephased.full(), rho_dephased_, decimal=6)
