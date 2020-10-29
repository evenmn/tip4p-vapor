import numpy as np


class GibbsDuhem:
    def __init__(self, beta0, p0, filename1, filename2):
        self.beta = beta0
        self.p = p0
        self.filename1 = filename1
        self.filename2 = filename2

    def _run_npt(self, filename):
        # run NPT in either MD or MC at self.beta, self.p
        # Measure enthalpy and volume
        v = 0
        h = 0
        return v, h

    def _compute_f(self, p, dh, dv):
        """Compute the fugacity f using the Gibbs-Duhem equation
        """
        return dh / (self.beta * p * dv)

    def _predict_p(self, f):
        """Predict pressure based on one fugacity value f
        """
        return self.p * np.exp(self.dbeta * f)

    def _correct_p(self, f0, f1):
        """Compute pressure based on two fugacity values f1 and f2
        """
        return self.p * np.exp(self.dbeta * (f0 + f1) / 2)

    def run(self, maxiter=100, dbeta=0.001):
        self.dbeta = dbeta
        iter = 0
        while iter < maxiter:
            v1, h1 = self._run_npt(self.filename1)
            v2, h2 = self._run_npt(self.filename2)
            f0 = self._compute_f(self.p, v1-v2, h1-h2)
            p_pred = self._predict_p(f0)
            self.beta += self.dbeta
            f1 = self._compute_f(p_pred, v1-v2, h1-h2)
            self.p = self._correct_p(f0, f1)
            iter += 1


if __name__ == "__main__":
    obj = GibbsDuhem()
