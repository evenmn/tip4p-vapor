import numpy as np
from lammps_logfile import File
from lammps_simulator import Simulator
from molecular_builder import pack_water


class GibbsDuhem:
    """Gibbs Duhem integration to find boiling point and
    vaporization enthalpy of a certain pressure

    :param N1: number of molecules in box 1
    :type N1: int
    :param N2: number of molecules in box 2
    :type N2: int
    :param rho1: mass density of box 1, given in g/cm^3
    :type rho1: float
    :param rho2: mass density of box 2, given in g/cm^3
    :type rho2: float
    :param beta_init: initial beta-value, beta = 1/kT
    :type beta_init: float
    :param p_init: initial pressure value given in bar
    :type p_init: float
    :param p_final: final pressure value
    :type p_final: float
    """
    def __init__(self, rho1, rho2, N1, N2, beta_init, p_init, p_final=0.987):
        self.rho1 = rho1
        self.rho2 = rho2
        self.N1 = N1
        self.N2 = N2
        self.beta = beta_init
        self.p = p_init
        self.p_final = p_final

    @staticmethod
    def _run_npt(rho, N, p, T):
        """Run NPT simulation in LAMMPS to find equilibration enthalpy
        and value

        :param rho: initial density
        :type rho: float
        :param N: number of molecules in simulation
        :type N: int
        :param p: pressure in simulation
        :type p: float
        :param T: temperature in system
        :type T: float
        """
        # initialize system
        water = pack_water(nummol=N, density=rho)
        water.write("water.data", format="lammps-data")

        # run system
        sim = Simulator(directory="", overwrite=True)
        sim.set_input_script(self.lmp_file, p=p, T=T)
        sim.run(self.computer)

        # analyze simulations
        logger = File("log.lammps")
        volume = logger.find("Volume")
        enthalpy = logger.find("Enthalpy")
        return volume, enthalpy

    def _compute_f(self, p, dh, dv):
        """Compute the fugacity f using the Gibbs-Duhem equation

        :param p: pressure
        :type p: float
        :param dh: enthalpy difference
        :type dh: float
        :param dv: volume difference
        :type dv: float
        :returns: fugacity
        :rtype: float
        """
        return dh / (self.beta * p * dv)

    def _predict_p(self, f):
        """Predict pressure based on one fugacity value f

        :param f: fugacity
        :type f: float
        :returns: predicted pressure
        :rtype: float
        """
        return self.p * np.exp(self.dbeta * f)

    def _correct_p(self, f0, f1):
        """Compute pressure based on two fugacity values f1 and f2

        :param f0: fugacity at previous timestep
        :type f0: float
        :param f1: fugacity at current timestep
        :type f1: float
        :returns: corrected pressure
        :rtype: float
        """
        return self.p * np.exp(self.dbeta * (f0 + f1) / 2)

    def run(self, maxiter=100, dbeta=0.001):
        """Run Gibbs-Duham integration.

        :param maxiter: max number of allowed interations
        :type maxiter: int
        :param dbeta: beta-step for each iteration
        :type dbeta: float
        """
        self.dbeta = dbeta

        p_list = []
        beta_list = []
        V_vap = []
        H_vap = []

        converged = False
        iter = 0
        while iter < maxiter and not converged:
            v1, h1 = self._run_npt(self.rho1, self.N1, self.p, 1/self.beta)
            v2, h2 = self._run_npt(self.rho2, self.N2, self.p, 1/self.beta)
            Dv = v1 - v2
            Dh = h1 - h2

            p_list.append(self.p)
            beta_list.append(self.beta)
            V_vap.append(Dv)
            H_vap.append(Dh)

            f0 = self._compute_f(self.p, Dv, Dh)
            p_pred = self._predict_p(f0)
            self.beta += self.dbeta
            f1 = self._compute_f(p_pred, Dv, Dh)
            if self.p > self.p_final:
                converged = True
            self.p = self._correct_p(f0, f1)
            iter += 1
        return 1/self.beta


if __name__ == "__main__":
    rho1 = 0.997
    rho2 = 0.0005
    N1 = 2048
    N2 = 2048
    beta_init = 1/300
    p_init = 0.5
    p_final = 0.987

    gibbsduhem = GibbsDuhem(rho1, rho2, N1, N2, beta_init, p_init, p_final)
    gibbsduhem.run()
