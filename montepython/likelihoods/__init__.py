import os
from montepython.likelihood_class import Likelihood_prior


class S8_DESY1(Likelihood_prior):

    # initialisation of the class is done within the parent Likelihood_prior. For
    # this case, it does not differ, actually, from the __init__ method in
    # Likelihood class.
    def loglkl(self, cosmo, data):

        sigma8 = cosmo.sigma8()
        Omega_m = cosmo.Omega_m()
        S8 = sigma8*(Omega_m/0.3)**(0.5)
        if S8 > self.S8:
            loglkl = -0.5 * (S8 - self.S8) ** 2 / (self.sigma_up ** 2)
        else:
            loglkl = -0.5 * (S8 - self.S8) ** 2 / (self.sigma_low ** 2)         
        return loglkl
