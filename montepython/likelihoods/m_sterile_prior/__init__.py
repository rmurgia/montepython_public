import os
from montepython.likelihood_class import Likelihood_prior


class m_sterile_prior(Likelihood_prior):

    # initialisation of the class is done within the parent Likelihood_prior. For
    # this case, it does not differ, actually, from the __init__ method in
    # Likelihood class.
    def loglkl(self, cosmo, data):

        m_sterile = cosmo.m_sterile()
        loglkl = -0.5 * (m_sterile - self.mean) ** 2 / (self.sigma ** 2)
        return loglkl
