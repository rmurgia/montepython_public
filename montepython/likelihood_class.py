"""
.. module:: likelihood_class
   :synopsis: Definition of the major likelihoods
.. moduleauthor:: Julien Lesgourgues <lesgourg@cern.ch>
.. moduleauthor:: Benjamin Audren <benjamin.audren@epfl.ch>

Contains the definition of the base likelihood class :class:`Likelihood`, with
basic functions, as well as more specific likelihood classes that may be reused
to implement new ones.

"""
from __future__ import print_function
import os
import numpy as np
import math
import warnings
import re
import scipy.constants as const
import scipy.integrate
import scipy.interpolate
import scipy.misc

import io_mp
from io_mp import dictitems,dictvalues,dictkeys


class Likelihood(object):
    """
    General class that all likelihoods will inherit from.

    """

    def __init__(self, path, data, command_line):
        """
        It copies the content of self.path from the initialization routine of
        the :class:`Data <data.Data>` class, and defines a handful of useful
        methods, that every likelihood might need.

        If the nuisance parameters required to compute this likelihood are not
        defined (either fixed or varying), the code will stop.

        Parameters
        ----------
        data : class
            Initialized instance of :class:`Data <data.Data>`
        command_line : NameSpace
            NameSpace containing the command line arguments

        """

        self.name = self.__class__.__name__
        self.folder = os.path.abspath(os.path.join(
            data.path['MontePython'], 'likelihoods', self.name))
        if not data.log_flag:
            path = os.path.join(command_line.folder, 'log.param')

        # Define some default fields
        self.data_directory = ''

        # Store all the default fields stored, for the method read_file.
        self.default_values = ['data_directory']

        # Recover the values potentially read in the input.param file.
        if hasattr(data, self.name):
            attributes = []
            exec("attributes = [e for e in dir(data.%s) if e.find('__') == -1]" % self.name)
            for elem in attributes:
                exec("setattr(self, elem, getattr(data.%s, elem))" % self.name)

        # Read values from the data file
        self.read_from_file(path, data, command_line)

        # Default state
        self.need_update = True

        # Check if the nuisance parameters are defined
        error_flag = False
        try:
            for nuisance in self.use_nuisance:
                if nuisance not in data.get_mcmc_parameters(['nuisance']):
                    error_flag = True
                    warnings.warn(
                        nuisance + " must be defined, either fixed or " +
                        "varying, for %s likelihood" % self.name)
            self.nuisance = self.use_nuisance
        except AttributeError:
            self.use_nuisance = []
            self.nuisance = []

        # If at least one is missing, raise an exception.
        if error_flag:
            raise io_mp.LikelihoodError(
                "Check your nuisance parameter list for your set of experiments")

        # Append to the log.param the value used (WARNING: so far no comparison
        # is done to ensure that the experiments share the same parameters)
        if data.log_flag:
            io_mp.log_likelihood_parameters(self, command_line)

    def loglkl(self, cosmo, data):
        """
        Placeholder to remind that this function needs to be defined for a
        new likelihood.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError(
            'Must implement method loglkl() in your likelihood')

    def read_from_file(self, path, data, command_line):
        """
        Extract the information from the log.param concerning this likelihood.

        If the log.param is used, check that at least one item for each
        likelihood is recovered. Otherwise, it means the log.param does not
        contain information on the likelihood. This happens when the first run
        fails early, before calling the likelihoods, and the program did not
        log the information. This check might not be completely secure, but it
        is better than nothing.

        .. warning::

            This checks relies on the fact that a likelihood should always have
            at least **one** line of code written in the likelihood.data file.
            This should be always true, but in case a run fails with the error
            message described below, think about it.

        .. warning::

            As of version 2.0.2, you can specify likelihood options in the
            parameter file. They have complete priority over the ones specified
            in the `likelihood.data` file, and it will be reflected in the
            `log.param` file.

        """

        # Counting how many lines are read.
        counter = 0

        self.path = path
        self.dictionary = {}
        if os.path.isfile(path):
            data_file = open(path, 'r')
            for line in data_file:
                if line.find('#') == -1:
                    if line.find(self.name+'.') != -1:
                        # Recover the name and value from the .data file
                        regexp = re.match(
                            "%s.(.*)\s*=\s*(.*)" % self.name, line)
                        name, value = (
                            elem.strip() for elem in regexp.groups())
                        # If this name was already defined in the parameter
                        # file, be sure to take this value instead. Beware,
                        # there are a few parameters which are always
                        # predefined, such as data_directory, which should be
                        # ignored in this check.
                        is_ignored = False
                        if name not in self.default_values:
                            try:
                                value = getattr(self, name)
                                is_ignored = True
                            except AttributeError:
                                pass
                        if not is_ignored:
                            exec('self.'+name+' = '+value)
                        value = getattr(self, name)
                        counter += 1
                        self.dictionary[name] = value
            data_file.seek(0)
            data_file.close()

        # Checking that at least one line was read, exiting otherwise
        if counter == 0:
            raise io_mp.ConfigurationError(
                "No information on %s likelihood " % self.name +
                "was found in the %s file.\n" % path +
                "This can result from a failed initialization of a previous " +
                "run. To solve this, you can do a \n " +
                "]$ rm -rf %s \n " % command_line.folder +
                "Be sure there is noting in it before doing this !")

    def get_cl(self, cosmo, l_max=-1):
        """
        Return the :math:`C_{\ell}` from the cosmological code in
        :math:`\mu {\\rm K}^2`

        """
        # get C_l^XX from the cosmological code
        cl = cosmo.lensed_cl(int(l_max))

        # convert dimensionless C_l's to C_l in muK**2
        T = cosmo.T_cmb()
        for key in dictkeys(cl):
            # All quantities need to be multiplied by this factor, except the
            # phi-phi term, that is already dimensionless
            # phi cross-terms should only be multiplied with this factor once
            if key not in ['pp', 'ell', 'tp', 'ep']:
                cl[key] *= (T*1.e6)**2
            elif key in ['tp', 'ep']:
                cl[key] *= (T*1.e6)

        return cl

    def get_unlensed_cl(self, cosmo, l_max=-1):
        """
        Return the :math:`C_{\ell}` from the cosmological code in
        :math:`\mu {\\rm K}^2`

        """
        # get C_l^XX from the cosmological code
        cl = cosmo.raw_cl(l_max)

        # convert dimensionless C_l's to C_l in muK**2
        T = cosmo.T_cmb()
        for key in dictkeys(cl):
            # All quantities need to be multiplied by this factor, except the
            # phi-phi term, that is already dimensionless
            # phi cross-terms should only be multiplied with this factor once
            if key not in ['pp', 'ell', 'tp', 'ep']:
                cl[key] *= (T*1.e6)**2
            elif key in ['tp', 'ep']:
                cl[key] *= (T*1.e6)

        return cl

    def need_cosmo_arguments(self, data, dictionary):
        """
        Ensure that the arguments of dictionary are defined to the correct
        value in the cosmological code

        .. warning::

            So far there is no way to enforce a parameter where `smaller is
            better`. A bigger value will always overried any smaller one
            (`cl_max`, etc...)

        Parameters
        ----------
        data : dict
            Initialized instance of :class:`data`
        dictionary : dict
            Desired precision for some cosmological parameters

        """
        array_flag = False
        for key, value in dictitems(dictionary):
            try:
                data.cosmo_arguments[key]
                try:
                    float(data.cosmo_arguments[key])
                    num_flag = True
                except ValueError:
                    num_flag = False
                except TypeError:
                    num_flag = True
                    array_flag = True

            except KeyError:
                try:
                    float(value)
                    num_flag = True
                    data.cosmo_arguments[key] = 0
                except ValueError:
                    num_flag = False
                    data.cosmo_arguments[key] = ''
                except TypeError:
                    num_flag = True
                    array_flag = True
            if num_flag is False:
                if data.cosmo_arguments[key].find(value) == -1:
                    data.cosmo_arguments[key] += ' '+value+' '
            else:
                if array_flag is False:
                    if float(data.cosmo_arguments[key]) < value:
                        data.cosmo_arguments[key] = value
                else:
                    data.cosmo_arguments[key] = '%.2g' % value[0]
                    for i in range(1, len(value)):
                        data.cosmo_arguments[key] += ',%.2g' % (value[i])

    def read_contamination_spectra(self, data):

        for nuisance in self.use_nuisance:
            # read spectrum contamination (so far, assumes only temperature
            # contamination; will be trivial to generalize to polarization when
            # such templates will become relevant)
            setattr(self, "%s_contamination" % nuisance,
                    np.zeros(self.l_max+1, 'float64'))
            try:
                File = open(os.path.join(
                    self.data_directory, getattr(self, "%s_file" % nuisance)),
                    'r')
                for line in File:
                    l = int(float(line.split()[0]))
                    if ((l >= 2) and (l <= self.l_max)):
                        exec("self.%s_contamination[l]=float(line.split()[1])/(l*(l+1.)/2./math.pi)" % nuisance)
            except:
                print('Warning: you did not pass a file name containing ')
                print('a contamination spectrum regulated by the nuisance ')
                print('parameter '+nuisance)

            # read renormalization factor
            # if it is not there, assume it is one, i.e. do not renormalize
            try:
                # do the following operation:
                # self.nuisance_contamination *= float(self.nuisance_scale)
                setattr(self, "%s_contamination" % nuisance,
                        getattr(self, "%s_contamination" % nuisance) *
                        float(getattr(self, "%s_scale" % nuisance)))
            except AttributeError:
                pass

            # read central value of nuisance parameter
            # if it is not there, assume one by default
            try:
                getattr(self, "%s_prior_center" % nuisance)
            except AttributeError:
                setattr(self, "%s_prior_center" % nuisance, 1.)

            # read variance of nuisance parameter
            # if it is not there, assume flat prior (encoded through
            # variance=0)
            try:
                getattr(self, "%s_prior_variance" % nuisance)
            except:
                setattr(self, "%s_prior_variance" % nuisance, 0.)

    def add_contamination_spectra(self, cl, data):

        # Recover the current value of the nuisance parameter.
        for nuisance in self.use_nuisance:
            nuisance_value = float(
                data.mcmc_parameters[nuisance]['current'] *
                data.mcmc_parameters[nuisance]['scale'])

            # add contamination spectra multiplied by nuisance parameters
            for l in range(2, self.l_max):
                exec("cl['tt'][l] += nuisance_value*self.%s_contamination[l]" % nuisance)

        return cl

    def add_nuisance_prior(self, lkl, data):

        # Recover the current value of the nuisance parameter.
        for nuisance in self.use_nuisance:
            nuisance_value = float(
                data.mcmc_parameters[nuisance]['current'] *
                data.mcmc_parameters[nuisance]['scale'])

            # add prior on nuisance parameters
            if getattr(self, "%s_prior_variance" % nuisance) > 0:
                # convenience variables
                prior_center = getattr(self, "%s_prior_center" % nuisance)
                prior_variance = getattr(self, "%s_prior_variance" % nuisance)
                lkl += -0.5*((nuisance_value-prior_center)/prior_variance)**2

        return lkl

    def computeLikelihood(self, ctx):
        """
        Interface with CosmoHammer

        Parameters
        ----------
        ctx : Context
                Contains several dictionaries storing data and cosmological
                information

        """
        # Recover both instances from the context
        cosmo = ctx.get("cosmo")
        data = ctx.get("data")

        loglkl = self.loglkl(cosmo, data)

        return loglkl


###################################
#
# END OF GENERIC LIKELIHOOD CLASS
#
###################################



###################################
# PRIOR TYPE LIKELIHOOD
# --> H0,...
###################################
class Likelihood_prior(Likelihood):

    def loglkl(self):
        raise NotImplementedError('Must implement method loglkl() in your likelihood')


###################################
# NEWDAT TYPE LIKELIHOOD
# --> spt,boomerang,etc.
###################################
class Likelihood_newdat(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        self.need_cosmo_arguments(
            data, {'lensing': 'yes', 'output': 'tCl lCl pCl'})

        # open .newdat file
        newdatfile = open(
            os.path.join(self.data_directory, self.file), 'r')

        # find beginning of window functions file names
        window_name = newdatfile.readline().strip('\n').replace(' ', '')

        # initialize list of fist and last band for each type
        band_num = np.zeros(6, 'int')
        band_min = np.zeros(6, 'int')
        band_max = np.zeros(6, 'int')

        # read number of bands for each of the six types TT, EE, BB, EB, TE, TB
        line = newdatfile.readline()
        for i in range(6):
            band_num[i] = int(line.split()[i])

        # read string equal to 'BAND_SELECTION' or not
        line = str(newdatfile.readline()).strip('\n').replace(' ', '')

        # if yes, read 6 lines containing 'min, max'
        if (line == 'BAND_SELECTION'):
            for i in range(6):
                line = newdatfile.readline()
                band_min[i] = int(line.split()[0])
                band_max[i] = int(line.split()[1])

        # if no, set min to 1 and max to band_num (=use all bands)
        else:
            band_min = [1 for i in range(6)]
            band_max = band_num

        # read line defining calibration uncertainty
        # contains: flag (=0 or 1), calib, calib_uncertainty
        line = newdatfile.readline()
        calib = float(line.split()[1])
        if (int(line.split()[0]) == 0):
            self.calib_uncertainty = 0
        else:
            self.calib_uncertainty = float(line.split()[2])

        # read line defining beam uncertainty
        # contains: flag (=0, 1 or 2), beam_width, beam_sigma
        line = newdatfile.readline()
        beam_type = int(line.split()[0])
        if (beam_type > 0):
            self.has_beam_uncertainty = True
        else:
            self.has_beam_uncertainty = False
        beam_width = float(line.split()[1])
        beam_sigma = float(line.split()[2])

        # read flag (= 0, 1 or 2) for lognormal distributions and xfactors
        line = newdatfile.readline()
        likelihood_type = int(line.split()[0])
        if (likelihood_type > 0):
            self.has_xfactors = True
        else:
            self.has_xfactors = False

        # declare array of quantitites describing each point of measurement
        # size yet unknown, it will be found later and stored as
        # self.num_points
        self.obs = np.array([], 'float64')
        self.var = np.array([], 'float64')
        self.beam_error = np.array([], 'float64')
        self.has_xfactor = np.array([], 'bool')
        self.xfactor = np.array([], 'float64')

        # temporary array to know which bands are actually used
        used_index = np.array([], 'int')

        index = -1

        # scan the lines describing each point of measurement
        for cltype in range(6):
            if (int(band_num[cltype]) != 0):
                # read name (but do not use it)
                newdatfile.readline()
                for band in range(int(band_num[cltype])):
                    # read one line corresponding to one measurement
                    line = newdatfile.readline()
                    index += 1

                    # if we wish to actually use this measurement
                    if ((band >= band_min[cltype]-1) and
                            (band <= band_max[cltype]-1)):

                        used_index = np.append(used_index, index)

                        self.obs = np.append(
                            self.obs, float(line.split()[1])*calib**2)

                        self.var = np.append(
                            self.var,
                            (0.5*(float(line.split()[2]) +
                                  float(line.split()[3]))*calib**2)**2)

                        self.xfactor = np.append(
                            self.xfactor, float(line.split()[4])*calib**2)

                        if ((likelihood_type == 0) or
                                ((likelihood_type == 2) and
                                (int(line.split()[7]) == 0))):
                            self.has_xfactor = np.append(
                                self.has_xfactor, [False])
                        if ((likelihood_type == 1) or
                                ((likelihood_type == 2) and
                                (int(line.split()[7]) == 1))):
                            self.has_xfactor = np.append(
                                self.has_xfactor, [True])

                        if (beam_type == 0):
                            self.beam_error = np.append(self.beam_error, 0.)
                        if (beam_type == 1):
                            l_mid = float(line.split()[5]) +\
                                0.5*(float(line.split()[5]) +
                                     float(line.split()[6]))
                            self.beam_error = np.append(
                                self.beam_error,
                                abs(math.exp(
                                    -l_mid*(l_mid+1)*1.526e-8*2.*beam_sigma *
                                    beam_width)-1.))
                        if (beam_type == 2):
                            if (likelihood_type == 2):
                                self.beam_error = np.append(
                                    self.beam_error, float(line.split()[8]))
                            else:
                                self.beam_error = np.append(
                                    self.beam_error, float(line.split()[7]))

                # now, skip and unused part of the file (with sub-correlation
                # matrices)
                for band in range(int(band_num[cltype])):
                    newdatfile.readline()

        # number of points that we will actually use
        self.num_points = np.shape(self.obs)[0]

        # total number of points, including unused ones
        full_num_points = index+1

        # read full correlation matrix
        full_covmat = np.zeros((full_num_points, full_num_points), 'float64')
        for point in range(full_num_points):
            full_covmat[point] = newdatfile.readline().split()

        # extract smaller correlation matrix for points actually used
        covmat = np.zeros((self.num_points, self.num_points), 'float64')
        for point in range(self.num_points):
            covmat[point] = full_covmat[used_index[point], used_index]

        # recalibrate this correlation matrix
        covmat *= calib**4

        # redefine the correlation matrix, the observed points and their
        # variance in case of lognormal likelihood
        if (self.has_xfactors):

            for i in range(self.num_points):

                for j in range(self.num_points):
                    if (self.has_xfactor[i]):
                        covmat[i, j] /= (self.obs[i]+self.xfactor[i])
                    if (self.has_xfactor[j]):
                        covmat[i, j] /= (self.obs[j]+self.xfactor[j])

            for i in range(self.num_points):
                if (self.has_xfactor[i]):
                    self.var[i] /= (self.obs[i]+self.xfactor[i])**2
                    self.obs[i] = math.log(self.obs[i]+self.xfactor[i])

        # invert correlation matrix
        self.inv_covmat = np.linalg.inv(covmat)

        # read window function files a first time, only for finding the
        # smallest and largest l's for each point
        self.win_min = np.zeros(self.num_points, 'int')
        self.win_max = np.zeros(self.num_points, 'int')
        for point in range(self.num_points):
            for line in open(os.path.join(
                    self.data_directory, 'windows', window_name) +
                    str(used_index[point]+1), 'r'):
                if any([float(line.split()[i]) != 0.
                        for i in range(1, len(line.split()))]):
                    if (self.win_min[point] == 0):
                        self.win_min[point] = int(line.split()[0])
                    self.win_max[point] = int(line.split()[0])

        # infer from format of window function files whether we will use
        # polarisation spectra or not
        num_col = len(line.split())
        if (num_col == 2):
            self.has_pol = False
        else:
            if (num_col == 5):
                self.has_pol = True
            else:
                raise io_mp.LikelihoodError(
                    "In likelihood %s. " % self.name +
                    "Window function files are understood if they contain " +
                    "2 columns (l TT), or 5 columns (l TT TE EE BB)." +
                    "In this case the number of columns is %d" % num_col)

        # define array of window functions
        self.window = np.zeros(
            (self.num_points, max(self.win_max)+1, num_col-1), 'float64')

        # go again through window function file, this time reading window
        # functions; that are distributed as: l TT (TE EE BB) where the last
        # columns contaim W_l/l, not W_l we mutiply by l in order to store the
        # actual W_l
        for point in range(self.num_points):
            for line in open(os.path.join(
                    self.data_directory, 'windows', window_name) +
                    str(used_index[point]+1), 'r'):
                l = int(line.split()[0])
                if (((self.has_pol is False) and (len(line.split()) != 2))
                        or ((self.has_pol is True) and
                            (len(line.split()) != 5))):
                    raise io_mp.LikelihoodError(
                        "In likelihood %s. " % self.name +
                        "for a given experiment, all window functions should" +
                        " have the same number of columns, 2 or 5. " +
                        "This is not the case here.")
                if ((l >= self.win_min[point]) and (l <= self.win_max[point])):
                    self.window[point, l, :] = [
                        float(line.split()[i])
                        for i in range(1, len(line.split()))]
                    self.window[point, l, :] *= l

        # eventually, initialise quantitites used in the marginalization over
        # nuisance parameters
        if ((self.has_xfactors) and
                ((self.calib_uncertainty > 1.e-4) or
                 (self.has_beam_uncertainty))):
            self.halfsteps = 5
            self.margeweights = np.zeros(2*self.halfsteps+1, 'float64')
            for i in range(-self.halfsteps, self.halfsteps+1):
                self.margeweights[i+self.halfsteps] = np.exp(
                    -(float(i)*3./float(self.halfsteps))**2/2)
            self.margenorm = sum(self.margeweights)

        # store maximum value of l needed by window functions
        self.l_max = max(self.win_max)

        # impose that the cosmological code computes Cl's up to maximum l
        # needed by the window function
        self.need_cosmo_arguments(data, {'l_max_scalars': self.l_max})

        # deal with nuisance parameters
        try:
            self.use_nuisance
            self.nuisance = self.use_nuisance
        except:
            self.use_nuisance = []
            self.nuisance = []
        self.read_contamination_spectra(data)

        # end of initialisation

    def loglkl(self, cosmo, data):
        # get Cl's from the cosmological code
        cl = self.get_cl(cosmo)

        # add contamination spectra multiplied by nuisance parameters
        cl = self.add_contamination_spectra(cl, data)

        # get likelihood
        lkl = self.compute_lkl(cl, cosmo, data)

        # add prior on nuisance parameters
        lkl = self.add_nuisance_prior(lkl, data)

        return lkl

    def compute_lkl(self, cl, cosmo, data):
        # checks that Cl's have been computed up to high enough l given window
        # function range. Normally this has been imposed before, so this test
        # could even be supressed.
        if (np.shape(cl['tt'])[0]-1 < self.l_max):
            raise io_mp.LikelihoodError(
                "%s computed Cls till l=" % data.cosmological_module_name +
                "%d " % (np.shape(cl['tt'])[0]-1) +
                "while window functions need %d." % self.l_max)

        # compute theoretical bandpowers, store them in theo[points]
        theo = np.zeros(self.num_points, 'float64')

        for point in range(self.num_points):

            # find bandpowers B_l by convolving C_l's with [(l+1/2)/2pi W_l]
            for l in range(self.win_min[point], self.win_max[point]):

                theo[point] += cl['tt'][l]*self.window[point, l, 0] *\
                    (l+0.5)/2./math.pi

                if (self.has_pol):
                    theo[point] += (
                        cl['te'][l]*self.window[point, l, 1] +
                        cl['ee'][l]*self.window[point, l, 2] +
                        cl['bb'][l]*self.window[point, l, 3]) *\
                        (l+0.5)/2./math.pi

        # allocate array for differencve between observed and theoretical
        # bandpowers
        difference = np.zeros(self.num_points, 'float64')

        # depending on the presence of lognormal likelihood, calibration
        # uncertainty and beam uncertainity, use several methods for
        # marginalising over nuisance parameters:

        # first method: numerical integration over calibration uncertainty:
        if (self.has_xfactors and
                ((self.calib_uncertainty > 1.e-4) or
                 self.has_beam_uncertainty)):

            chisq_tmp = np.zeros(2*self.halfsteps+1, 'float64')
            chisqcalib = np.zeros(2*self.halfsteps+1, 'float64')
            beam_error = np.zeros(self.num_points, 'float64')

            # loop over various beam errors
            for ibeam in range(2*self.halfsteps+1):

                # beam error
                for point in range(self.num_points):
                    if (self.has_beam_uncertainty):
                        beam_error[point] = 1.+self.beam_error[point] *\
                            (ibeam-self.halfsteps)*3/float(self.halfsteps)
                    else:
                        beam_error[point] = 1.

                # loop over various calibraion errors
                for icalib in range(2*self.halfsteps+1):

                    # calibration error
                    calib_error = 1+self.calib_uncertainty*(
                        icalib-self.halfsteps)*3/float(self.halfsteps)

                    # compute difference between observed and theoretical
                    # points, after correcting the later for errors
                    for point in range(self.num_points):

                        # for lognormal likelihood, use log(B_l+X_l)
                        if (self.has_xfactor[point]):
                            difference[point] = self.obs[point] -\
                                math.log(
                                    theo[point]*beam_error[point] *
                                    calib_error+self.xfactor[point])
                        # otherwise use B_l
                        else:
                            difference[point] = self.obs[point] -\
                                theo[point]*beam_error[point]*calib_error

                    # find chisq with those corrections
                    # chisq_tmp[icalib] = np.dot(np.transpose(difference),
                    # np.dot(self.inv_covmat, difference))
                    chisq_tmp[icalib] = np.dot(
                        difference, np.dot(self.inv_covmat, difference))

                minchisq = min(chisq_tmp)

            # find chisq marginalized over calibration uncertainty (if any)
                tot = 0
                for icalib in range(2*self.halfsteps+1):
                    tot += self.margeweights[icalib]*math.exp(
                        max(-30., -(chisq_tmp[icalib]-minchisq)/2.))

                chisqcalib[ibeam] = -2*math.log(tot/self.margenorm)+minchisq

            # find chisq marginalized over beam uncertainty (if any)
            if (self.has_beam_uncertainty):

                minchisq = min(chisqcalib)

                tot = 0
                for ibeam in range(2*self.halfsteps+1):
                    tot += self.margeweights[ibeam]*math.exp(
                        max(-30., -(chisqcalib[ibeam]-minchisq)/2.))

                chisq = -2*math.log(tot/self.margenorm)+minchisq

            else:
                chisq = chisqcalib[0]

        # second method: marginalize over nuisance parameters (if any)
        # analytically
        else:

            # for lognormal likelihood, theo[point] should contain log(B_l+X_l)
            if (self.has_xfactors):
                for point in range(self.num_points):
                    if (self.has_xfactor[point]):
                        theo[point] = math.log(theo[point]+self.xfactor[point])

            # find vector of difference between observed and theoretical
            # bandpowers
            difference = self.obs-theo

            # find chisq
            chisq = np.dot(
                np.transpose(difference), np.dot(self.inv_covmat, difference))

            # correct eventually for effect of analytic marginalization over
            # nuisance parameters
            if ((self.calib_uncertainty > 1.e-4) or self.has_beam_uncertainty):

                denom = 1.
                tmpi = np.dot(self.inv_covmat, theo)
                chi2op = np.dot(np.transpose(difference), tmp)
                chi2pp = np.dot(np.transpose(theo), tmp)

                # TODO beam is not defined here !
                if (self.has_beam_uncertainty):
                    for points in range(self.num_points):
                        beam[point] = self.beam_error[point]*theo[point]
                    tmp = np.dot(self.inv_covmat, beam)
                    chi2dd = np.dot(np.transpose(beam), tmp)
                    chi2pd = np.dot(np.transpose(theo), tmp)
                    chi2od = np.dot(np.transpose(difference), tmp)

                if (self.calib_uncertainty > 1.e-4):
                    wpp = 1/(chi2pp+1/self.calib_uncertainty**2)
                    chisq = chisq-wpp*chi2op**2
                    denom = denom/wpp*self.calib_uncertainty**2
                else:
                    wpp = 0

                if (self.has_beam_uncertainty):
                    wdd = 1/(chi2dd-wpp*chi2pd**2+1)
                    chisq = chisq-wdd*(chi2od-wpp*chi2op*chi2pd)**2
                    denom = denom/wdd

                chisq += math.log(denom)

        # finally, return ln(L)=-chi2/2

        self.lkl = -0.5 * chisq
        return self.lkl


###################################
# CLIK TYPE LIKELIHOOD
# --> clik_fake_planck,clik_wmap,etc.
###################################
class Likelihood_clik(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)
        self.need_cosmo_arguments(
            data, {'lensing': 'yes', 'output': 'tCl lCl pCl'})

        try:
            import clik
        except ImportError:
            raise io_mp.MissingLibraryError(
                "You must first activate the binaries from the Clik " +
                "distribution. Please run : \n " +
                "]$ source /path/to/clik/bin/clik_profile.sh \n " +
                "and try again.")
        # for lensing, some routines change. Intializing a flag for easier
        # testing of this condition
        #if self.name == 'Planck_lensing':
        if 'lensing' in self.name and 'Planck' in self.name:
            self.lensing = True
        else:
            self.lensing = False

        try:
            if self.lensing:
                self.clik = clik.clik_lensing(self.path_clik)
                try:
                    self.l_max = max(self.clik.get_lmax())
                # following 2 lines for compatibility with lensing likelihoods of 2013 and before
                # (then, clik.get_lmax() just returns an integer for lensing likelihoods;
                # this behavior was for clik versions < 10)
                except:
                    self.l_max = self.clik.get_lmax()
            else:
                self.clik = clik.clik(self.path_clik)
                self.l_max = max(self.clik.get_lmax())
        except clik.lkl.CError:
            raise io_mp.LikelihoodError(
                "The path to the .clik file for the likelihood "
                "%s was not found where indicated:\n%s\n"
                % (self.name,self.path_clik) +
                " Note that the default path to search for it is"
                " one directory above the path['clik'] field. You"
                " can change this behaviour in all the "
                "Planck_something.data, to reflect your local configuration, "
                "or alternatively, move your .clik files to this place.")
        except KeyError:
            raise io_mp.LikelihoodError(
                "In the %s.data file, the field 'clik' of the " % self.name +
                "path dictionary is expected to be defined. Please make sure"
                " it is the case in you configuration file")

        self.need_cosmo_arguments(
            data, {'l_max_scalars': self.l_max})

        self.nuisance = list(self.clik.extra_parameter_names)

        # line added to deal with a bug in planck likelihood release: A_planck called A_Planck in plik_lite
        if (self.name == 'Planck15_highl_lite') or (self.name == 'Planck15_highl_TTTEEE_lite'):
            for i in range(len(self.nuisance)):
                if (self.nuisance[i] == 'A_Planck'):
                    self.nuisance[i] = 'A_planck'
            print("In %s, MontePython corrected nuisance parameter name A_Planck to A_planck" % self.name)

        # testing if the nuisance parameters are defined. If there is at least
        # one non defined, raise an exception.
        exit_flag = False
        nuisance_parameter_names = data.get_mcmc_parameters(['nuisance'])
        for nuisance in self.nuisance:
            if nuisance not in nuisance_parameter_names:
                exit_flag = True
                print('%20s\tmust be a fixed or varying nuisance parameter' % nuisance)

        if exit_flag:
            raise io_mp.LikelihoodError(
                "The likelihood %s " % self.name +
                "expected some nuisance parameters that were not provided")

        # deal with nuisance parameters
        try:
            self.use_nuisance
        except:
            self.use_nuisance = []

        # Add in use_nuisance all the parameters that have non-flat prior
        for nuisance in self.nuisance:
            if hasattr(self, '%s_prior_center' % nuisance):
                self.use_nuisance.append(nuisance)

    def loglkl(self, cosmo, data):

        nuisance_parameter_names = data.get_mcmc_parameters(['nuisance'])

        # get Cl's from the cosmological code
        cl = self.get_cl(cosmo)

        # testing for lensing
        if self.lensing:
            try:
                length = len(self.clik.get_lmax())
                tot = np.zeros(
                    np.sum(self.clik.get_lmax()) + length +
                    len(self.clik.get_extra_parameter_names()))
            # following 3 lines for compatibility with lensing likelihoods of 2013 and before
            # (then, clik.get_lmax() just returns an integer for lensing likelihoods,
            # and the length is always 2 for cl['pp'], cl['tt'])
            except:
                length = 2
                tot = np.zeros(2*self.l_max+length + len(self.clik.get_extra_parameter_names()))
        else:
            length = len(self.clik.get_has_cl())
            tot = np.zeros(
                np.sum(self.clik.get_lmax()) + length +
                len(self.clik.get_extra_parameter_names()))

        # fill with Cl's
        index = 0
        if not self.lensing:
            for i in range(length):
                if (self.clik.get_lmax()[i] > -1):
                    for j in range(self.clik.get_lmax()[i]+1):
                        if (i == 0):
                            tot[index+j] = cl['tt'][j]
                        if (i == 1):
                            tot[index+j] = cl['ee'][j]
                        if (i == 2):
                            tot[index+j] = cl['bb'][j]
                        if (i == 3):
                            tot[index+j] = cl['te'][j]
                        if (i == 4):
                            tot[index+j] = 0 #cl['tb'][j] class does not compute tb
                        if (i == 5):
                            tot[index+j] = 0 #cl['eb'][j] class does not compute eb

                    index += self.clik.get_lmax()[i]+1

        else:
            try:
                for i in range(length):
                    if (self.clik.get_lmax()[i] > -1):
                        for j in range(self.clik.get_lmax()[i]+1):
                            if (i == 0):
                                tot[index+j] = cl['pp'][j]
                            if (i == 1):
                                tot[index+j] = cl['tt'][j]
                            if (i == 2):
                                tot[index+j] = cl['ee'][j]
                            if (i == 3):
                                tot[index+j] = cl['bb'][j]
                            if (i == 4):
                                tot[index+j] = cl['te'][j]
                            if (i == 5):
                                tot[index+j] = 0 #cl['tb'][j] class does not compute tb
                            if (i == 6):
                                tot[index+j] = 0 #cl['eb'][j] class does not compute eb

                        index += self.clik.get_lmax()[i]+1

            # following 8 lines for compatibility with lensing likelihoods of 2013 and before
            # (then, clik.get_lmax() just returns an integer for lensing likelihoods,
            # and the length is always 2 for cl['pp'], cl['tt'])
            except:
                for i in range(length):
                    for j in range(self.l_max):
                        if (i == 0):
                            tot[index+j] = cl['pp'][j]
                        if (i == 1):
                            tot[index+j] = cl['tt'][j]
                    index += self.l_max+1

        # fill with nuisance parameters
        for nuisance in self.clik.get_extra_parameter_names():

            # line added to deal with a bug in planck likelihood release: A_planck called A_Planck in plik_lite
            if (self.name == 'Planck15_highl_lite') or (self.name == 'Planck15_highl_TTTEEE_lite'):
                if nuisance == 'A_Planck':
                    nuisance = 'A_planck'

            if nuisance in nuisance_parameter_names:
                nuisance_value = data.mcmc_parameters[nuisance]['current'] *\
                    data.mcmc_parameters[nuisance]['scale']
            else:
                raise io_mp.LikelihoodError(
                    "the likelihood needs a parameter %s. " % nuisance +
                    "You must pass it through the input file " +
                    "(as a free nuisance parameter or a fixed parameter)")
            #print("found one nuisance with name",nuisance)
            tot[index] = nuisance_value
            index += 1

        # compute likelihood
        #print("lkl:",self.clik(tot))
        lkl = self.clik(tot)[0]

        # add prior on nuisance parameters
        lkl = self.add_nuisance_prior(lkl, data)

        # Option added by D.C. Hooper to deal with the joint prior on ksz_norm (A_ksz in Planck notation)
        # and A_sz (A_tsz in Planck notation), of the form ksz_norm + 1.6 * A_sz (according to eq. 23 of 1907.12875).
        # Behaviour (True/False), centre, and variance set in the .data files (default = True).

        # Check if the joint prior has been requested
        if getattr(self, 'joint_sz_prior', False):

            # Check that the joint_sz prior is only requested when A_sz and ksz_norm are present
            if not ('A_sz' in self.clik.get_extra_parameter_names() and 'ksz_norm' in self.clik.get_extra_parameter_names()):
                 raise io_mp.LikelihoodError(
                    "You requested a gaussian prior on ksz_norm + 1.6 * A_sz," +
                    "however A_sz or ksz_norm are not present in your param file.")

            # Recover the current values of the two sz nuisance parameters
            A_sz =  data.mcmc_parameters['A_sz']['current'] * data.mcmc_parameters['A_sz']['scale']
            ksz_norm = data.mcmc_parameters['ksz_norm']['current'] * data.mcmc_parameters['ksz_norm']['scale']

            # Combine the two into one new nuisance-like variable
            joint_sz = ksz_norm + 1.6 * A_sz

            # Check if the user has passed the prior center and variance on sz, otherwise abort
            if not (hasattr(self, 'joint_sz_prior_center') and hasattr(self, 'joint_sz_prior_variance')):
                raise io_mp.LikelihoodError(
                    " You requested a gaussian prior on ksz_norm + 1.6 * A_sz," +
                    " however you did not pass the center and variance." +
                    " You can pass this in the .data file.")

            # add prior on joint_sz parameter
            if not self.joint_sz_prior_variance == 0:
                lkl += -0.5*((joint_sz-self.joint_sz_prior_center)/self.joint_sz_prior_variance)**2

            # End of block for joint sz prior.

        return lkl


###################################
# MOCK CMB TYPE LIKELIHOOD
# --> mock planck, cmbpol, etc.
###################################
class Likelihood_mock_cmb(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        self.need_cosmo_arguments(
            data, {'lensing': 'yes', 'output': 'tCl lCl pCl'})

        ################
        # Noise spectrum
        ################

        try:
            self.noise_from_file
        except:
            self.noise_from_file = False

        if self.noise_from_file:

            try:
                self.noise_file
            except:
                raise io_mp.LikelihoodError("For reading noise from file, you must provide noise_file")

            self.noise_T = np.zeros(self.l_max+1, 'float64')
            self.noise_P = np.zeros(self.l_max+1, 'float64')
            if self.LensingExtraction:
                self.Nldd = np.zeros(self.l_max+1, 'float64')

            if os.path.exists(os.path.join(self.data_directory, self.noise_file)):
                noise = open(os.path.join(
                    self.data_directory, self.noise_file), 'r')
                line = noise.readline()
                while line.find('#') != -1:
                    line = noise.readline()

                for l in range(self.l_min, self.l_max+1):
                    ll = int(float(line.split()[0]))
                    if l != ll:
                        # if l_min is larger than the first l in the noise file we can skip lines
                        # until we are at the correct l. Otherwise raise error
                        while l > ll:
                            try:
                                line = noise.readline()
                                ll = int(float(line.split()[0]))
                            except:
                                raise io_mp.LikelihoodError("Mismatch between required values of l in the code and in the noise file")
                        if l < ll:
                            raise io_mp.LikelihoodError("Mismatch between required values of l in the code and in the noise file")
                    # read noise for C_l in muK**2
                    self.noise_T[l] = float(line.split()[1])
                    self.noise_P[l] = float(line.split()[2])
                    if self.LensingExtraction:
                        try:
                            # read noise for C_l^dd = l(l+1) C_l^pp
                            self.Nldd[l] = float(line.split()[3])/(l*(l+1)/2./math.pi)
                        except:
                            raise io_mp.LikelihoodError("For reading lensing noise from file, you must provide one more column")
                    line = noise.readline()
            else:
                raise io_mp.LikelihoodError("Could not find file ",self.noise_file)


        else:
            # convert arcmin to radians
            self.theta_fwhm *= np.array([math.pi/60/180])
            self.sigma_T *= np.array([math.pi/60/180])
            self.sigma_P *= np.array([math.pi/60/180])

            # compute noise in muK**2
            self.noise_T = np.zeros(self.l_max+1, 'float64')
            self.noise_P = np.zeros(self.l_max+1, 'float64')

            for l in range(self.l_min, self.l_max+1):
                self.noise_T[l] = 0
                self.noise_P[l] = 0
                for channel in range(self.num_channels):
                    self.noise_T[l] += self.sigma_T[channel]**-2 *\
                                       math.exp(
                                           -l*(l+1)*self.theta_fwhm[channel]**2/8/math.log(2))
                    self.noise_P[l] += self.sigma_P[channel]**-2 *\
                                       math.exp(
                                           -l*(l+1)*self.theta_fwhm[channel]**2/8/math.log(2))
                self.noise_T[l] = 1/self.noise_T[l]
                self.noise_P[l] = 1/self.noise_P[l]


        # trick to remove any information from polarisation for l<30
        try:
            self.no_small_l_pol
        except:
            self.no_small_l_pol = False

        if self.no_small_l_pol:
            for l in range(self.l_min,30):
                # plug a noise level of 100 muK**2, equivalent to no detection at all of polarisation
                self.noise_P[l] = 100.

        # trick to remove any information from temperature above l_max_TT
        try:
            self.l_max_TT
        except:
            self.l_max_TT = False

        if self.l_max_TT:
            for l in range(self.l_max_TT+1,l_max+1):
                # plug a noise level of 100 muK**2, equivalent to no detection at all of temperature
                self.noise_T[l] = 100.

        # impose that the cosmological code computes Cl's up to maximum l
        # needed by the window function
        self.need_cosmo_arguments(data, {'l_max_scalars': self.l_max})

        # if you want to print the noise spectra:
        #test = open('noise_T_P','w')
        #for l in range(self.l_min, self.l_max+1):
        #    test.write('%d  %e  %e\n'%(l,self.noise_T[l],self.noise_P[l]))

        ###########################################################################
        # implementation of default settings for flags describing the likelihood: #
        ###########################################################################

        # - ignore B modes by default:
        try:
            self.Bmodes
        except:
            self.Bmodes = False
        # - do not use delensing by default:
        try:
            self.delensing
        except:
            self.delensing = False
        # - do not include lensing extraction by default:
        try:
            self.LensingExtraction
        except:
            self.LensingExtraction = False
        # - neglect TD correlation by default:
        try:
            self.neglect_TD
        except:
            self.neglect_TD = True
        # - use lthe lensed TT, TE, EE by default:
        try:
            self.unlensed_clTTTEEE
        except:
            self.unlensed_clTTTEEE = False
        # - do not exclude TTEE by default:
        try:
            self.ExcludeTTTEEE
            if self.ExcludeTTTEEE and not self.LensingExtraction:
                raise io_mp.LikelihoodError("Mock CMB likelihoods where TTTEEE is not used have only been "
                                            "implemented for the deflection spectrum (i.e. not for B-modes), "
                                            "but you do not seem to have lensing extraction enabled")
        except:
            self.ExcludeTTTEEE = False

	#added by Siavash Yasini
        try:
            self.OnlyTT
            if self.OnlyTT and self.ExcludeTTTEEE:
                raise io_mp.LikelihoodError("OnlyTT and ExcludeTTTEEE cannot be used simultaneously.")
        except:
            self.OnlyTT = False

        ##############################################
        # Delensing noise: implemented by  S. Clesse #
        ##############################################

        if self.delensing:

            try:
                self.delensing_file
            except:
                raise io_mp.LikelihoodError("For delensing, you must provide delensing_file")

            self.noise_delensing = np.zeros(self.l_max+1)
            if os.path.exists(os.path.join(self.data_directory, self.delensing_file)):
                delensing_file = open(os.path.join(
                    self.data_directory, self.delensing_file), 'r')
                line = delensing_file.readline()
                while line.find('#') != -1:
                    line = delensing_file.readline()

                for l in range(self.l_min, self.l_max+1):
                    ll = int(float(line.split()[0]))
                    if l != ll:
                        # if l_min is larger than the first l in the delensing file we can skip lines
                        # until we are at the correct l. Otherwise raise error
                        while l > ll:
                            try:
                                line = delensing_file.readline()
                                ll = int(float(line.split()[0]))
                            except:
                                raise io_mp.LikelihoodError("Mismatch between required values of l in the code and in the delensing file")
                        if l < ll:
                            raise io_mp.LikelihoodError("Mismatch between required values of l in the code and in the delensing file")
                    self.noise_delensing[ll] = float(line.split()[2])/(ll*(ll+1)/2./math.pi)
                    # change 3 to 4 in the above line for CMBxCIB delensing
                    line = delensing_file.readline()

            else:
                raise io_mp.LikelihoodError("Could not find file ",self.delensing_file)

        ###############################################################
        # Read data for TT, EE, TE, [eventually BB or phi-phi, phi-T] #
        ###############################################################

        # default:
        if not self.ExcludeTTTEEE:
            numCls = 3
        # default 0 if excluding TT EE
        else:
            numCls = 0

        # deal with BB:
        if self.Bmodes:
            self.index_B = numCls
            numCls += 1

        # deal with pp, pT (p = CMB lensing potential):
        if self.LensingExtraction:
            self.index_pp = numCls
            numCls += 1
            if not self.ExcludeTTTEEE:
                self.index_tp = numCls
                numCls += 1

            if not self.noise_from_file:
                # provide a file containing NlDD (noise for the extracted
                # deflection field spectrum) This option is temporary
                # because at some point this module will compute NlDD
                # itself, when logging the fiducial model spectrum.
                try:
                    self.temporary_Nldd_file
                except:
                    raise io_mp.LikelihoodError("For lensing extraction, you must provide a temporary_Nldd_file")

                # read the NlDD file
                self.Nldd = np.zeros(self.l_max+1, 'float64')

                if os.path.exists(os.path.join(self.data_directory, self.temporary_Nldd_file)):
                    fid_file = open(os.path.join(self.data_directory, self.temporary_Nldd_file), 'r')
                    line = fid_file.readline()
                    while line.find('#') != -1:
                        line = fid_file.readline()
                    while (line.find('\n') != -1 and len(line) == 1):
                        line = fid_file.readline()
                    for l in range(self.l_min, self.l_max+1):
                        ll = int(float(line.split()[0]))
                        if l != ll:
                            # if l_min is larger than the first l in the delensing file we can skip lines
                            # until we are at the correct l. Otherwise raise error
                            while l > ll:
                                try:
                                    line = fid_file.readline()
                                    ll = int(float(line.split()[0]))
                                except:
                                    raise io_mp.LikelihoodError("Mismatch between required values of l in the code and in the delensing file")
                            if l < ll:
                                raise io_mp.LikelihoodError("Mismatch between required values of l in the code and in the delensing file")
                        # this lines assumes that Nldd is stored in the
                        # 4th column (can be customised)
                        self.Nldd[ll] = float(line.split()[3])/(l*(l+1.)/2./math.pi)
                        line = fid_file.readline()
                else:
                    raise io_mp.LikelihoodError("Could not find file ",self.temporary_Nldd_file)

        # deal with fiducial model:
        # If the file exists, initialize the fiducial values
        self.Cl_fid = np.zeros((numCls, self.l_max+1), 'float64')
        self.fid_values_exist = False
        if os.path.exists(os.path.join(
                self.data_directory, self.fiducial_file)):
            self.fid_values_exist = True
            fid_file = open(os.path.join(
                self.data_directory, self.fiducial_file), 'r')
            line = fid_file.readline()
            while line.find('#') != -1:
                line = fid_file.readline()
            while (line.find('\n') != -1 and len(line) == 1):
                line = fid_file.readline()
            for l in range(self.l_min, self.l_max+1):
                ll = int(line.split()[0])
                if not self.ExcludeTTTEEE:
                    self.Cl_fid[0, ll] = float(line.split()[1])
                    self.Cl_fid[1, ll] = float(line.split()[2])
                    self.Cl_fid[2, ll] = float(line.split()[3])
                # read BB:
                if self.Bmodes:
                    try:
                        self.Cl_fid[self.index_B, ll] = float(line.split()[self.index_B+1])
                    except:
                        raise io_mp.LikelihoodError(
                            "The fiducial model does not have enough columns.")
                # read DD, TD (D = deflection field):
                if self.LensingExtraction:
                    try:
                        self.Cl_fid[self.index_pp, ll] = float(line.split()[self.index_pp+1])
                        if not self.ExcludeTTTEEE:
                            self.Cl_fid[self.index_tp, ll] = float(line.split()[self.index_tp+1])
                    except:
                        raise io_mp.LikelihoodError(
                            "The fiducial model does not have enough columns.")

                line = fid_file.readline()

        # Else the file will be created in the loglkl() function.

        # Explicitly display the flags to be sure that likelihood does what you expect:
        print("Initialised likelihood_mock_cmb with following options:")
        if self.unlensed_clTTTEEE:
            print("  unlensed_clTTTEEE is True")
        else:
            print("  unlensed_clTTTEEE is False")
        if self.Bmodes:
            print("  Bmodes is True")
        else:
            print("  Bmodes is False")
        if self.delensing:
            print("  delensing is True")
        else:
            print("  delensing is False")
        if self.LensingExtraction:
            print("  LensingExtraction is True")
        else:
            print("  LensingExtraction is False")
        if self.neglect_TD:
            print("  neglect_TD is True")
        else:
            print("  neglect_TD is False")
        if self.ExcludeTTTEEE:
            print("  ExcludeTTTEEE is True")
        else:
            print("  ExcludeTTTEEE is False")
        if self.OnlyTT:
            print("  OnlyTT is True")
        else:
            print("  OnlyTT is False")
        print("")

        # end of initialisation
        return

    def loglkl(self, cosmo, data):

        # get Cl's from the cosmological code (returned in muK**2 units)

        # if we want unlensed Cl's
        if self.unlensed_clTTTEEE:
            cl = self.get_unlensed_cl(cosmo)
            # exception: for non-delensed B modes we need the lensed BB spectrum
            # (this case is usually not useful/relevant)
            if self.Bmodes and (not self.delensing):
                    cl_lensed = self.get_cl(cosmo)
                    for l in range(self.lmax+1):
                        cl[l]['bb']=cl_lensed[l]['bb']

        # if we want lensed Cl's
        else:
            cl = self.get_cl(cosmo)
            # exception: for delensed B modes we need the unlensed spectrum
            if self.Bmodes and self.delensing:
                cl_unlensed = self.get_unlensed_cl(cosmo)
                for l in range(self.lmax+1):
                        cl[l]['bb']=cl_unlensed[l]['bb']

        # get likelihood
        lkl = self.compute_lkl(cl, cosmo, data)

        return lkl

    def compute_lkl(self, cl, cosmo, data):

        # Write fiducial model spectra if needed (return an imaginary number in
        # that case)
        if self.fid_values_exist is False:
            # Store the values now.
            fid_file = open(os.path.join(
                self.data_directory, self.fiducial_file), 'w')
            fid_file.write('# Fiducial parameters')
            for key, value in dictitems(data.mcmc_parameters):
                fid_file.write(', %s = %.5g' % (
                    key, value['current']*value['scale']))
            fid_file.write('\n')
            for l in range(self.l_min, self.l_max+1):
                fid_file.write("%5d  " % l)
                if not self.ExcludeTTTEEE:
                    fid_file.write("%.8g  " % (cl['tt'][l]+self.noise_T[l]))
                    fid_file.write("%.8g  " % (cl['ee'][l]+self.noise_P[l]))
                    fid_file.write("%.8g  " % cl['te'][l])
                if self.Bmodes:
                    # next three lines added by S. Clesse for delensing
                    if self.delensing:
                        fid_file.write("%.8g  " % (cl['bb'][l]+self.noise_P[l]+self.noise_delensing[l]))
                    else:
                        fid_file.write("%.8g  " % (cl['bb'][l]+self.noise_P[l]))
                if self.LensingExtraction:
                    # we want to store clDD = l(l+1) clpp
                    # and ClTD = sqrt(l(l+1)) Cltp
                    fid_file.write("%.8g  " % (l*(l+1.)*cl['pp'][l] + self.Nldd[l]))
                    if not self.ExcludeTTTEEE:
                        fid_file.write("%.8g  " % (math.sqrt(l*(l+1.))*cl['tp'][l]))
                fid_file.write("\n")
            print('\n')
            warnings.warn(
                "Writing fiducial model in %s, for %s likelihood\n" % (
                    self.data_directory+'/'+self.fiducial_file, self.name))
            return 1j

        # compute likelihood

        chi2 = 0

        # count number of modes.
        # number of modes is different form number of spectra
        # modes = T,E,[B],[D=deflection]
        # spectra = TT,EE,TE,[BB],[DD,TD]
        # default:
        if not self.ExcludeTTTEEE:
            if self.OnlyTT:
                num_modes=1
            else:
                num_modes=2
        # default 0 if excluding TT EE
        else:
            num_modes=0
        # add B mode:
        if self.Bmodes:
            num_modes += 1
        # add D mode:
        if self.LensingExtraction:
            num_modes += 1

        Cov_obs = np.zeros((num_modes, num_modes), 'float64')
        Cov_the = np.zeros((num_modes, num_modes), 'float64')
        Cov_mix = np.zeros((num_modes, num_modes), 'float64')

        for l in range(self.l_min, self.l_max+1):

            if self.Bmodes and self.LensingExtraction:
                raise io_mp.LikelihoodError("We have implemented a version of the likelihood with B modes, a version with lensing extraction, but not yet a version with both at the same time. You can implement it.")

            # case with B modes:
            elif self.Bmodes:
                Cov_obs = np.array([
                    [self.Cl_fid[0, l], self.Cl_fid[2, l], 0],
                    [self.Cl_fid[2, l], self.Cl_fid[1, l], 0],
                    [0, 0, self.Cl_fid[3, l]]])
                # next 5 lines added by S. Clesse for delensing
                if self.delensing:
                    Cov_the = np.array([
                        [cl['tt'][l]+self.noise_T[l], cl['te'][l], 0],
                        [cl['te'][l], cl['ee'][l]+self.noise_P[l], 0],
                        [0, 0, cl['bb'][l]+self.noise_P[l]+self.noise_delensing[l]]])
                else:
                    Cov_the = np.array([
                        [cl['tt'][l]+self.noise_T[l], cl['te'][l], 0],
                        [cl['te'][l], cl['ee'][l]+self.noise_P[l], 0],
                        [0, 0, cl['bb'][l]+self.noise_P[l]]])

            # case with lensing
            # note that the likelihood is based on ClDD (deflection spectrum)
            # rather than Clpp (lensing potential spectrum)
            # But the Bolztmann code input is Clpp
            # So we make the conversion using ClDD = l*(l+1.)*Clpp
            # So we make the conversion using ClTD = sqrt(l*(l+1.))*Cltp

            # just DD, i.e. no TT or EE.
            elif self.LensingExtraction and self.ExcludeTTTEEE:
                cldd_fid = self.Cl_fid[self.index_pp, l]
                cldd = l*(l+1.)*cl['pp'][l]
                Cov_obs = np.array([[cldd_fid]])
                Cov_the = np.array([[cldd+self.Nldd[l]]])

            # Usual TTTEEE plus DD and TD
            elif self.LensingExtraction:
                cldd_fid = self.Cl_fid[self.index_pp, l]
                cldd = l*(l+1.)*cl['pp'][l]
                if self.neglect_TD:
                    cltd_fid = 0.
                    cltd = 0.
                else:
                    cltd_fid = self.Cl_fid[self.index_tp, l]
                    cltd = math.sqrt(l*(l+1.))*cl['tp'][l]

                Cov_obs = np.array([
                    [self.Cl_fid[0, l], self.Cl_fid[2, l], 0.*self.Cl_fid[self.index_tp, l]],
                    [self.Cl_fid[2, l], self.Cl_fid[1, l], 0],
                    [cltd_fid, 0, cldd_fid]])
                Cov_the = np.array([
                    [cl['tt'][l]+self.noise_T[l], cl['te'][l], 0.*math.sqrt(l*(l+1.))*cl['tp'][l]],
                    [cl['te'][l], cl['ee'][l]+self.noise_P[l], 0],
                    [cltd, 0, cldd+self.Nldd[l]]])

	    # case with TT only (Added by Siavash Yasini)
            elif self.OnlyTT:
                Cov_obs = np.array([[self.Cl_fid[0, l]]])

                Cov_the = np.array([[cl['tt'][l]+self.noise_T[l]]])


            # case without B modes nor lensing:
            else:
                Cov_obs = np.array([
                    [self.Cl_fid[0, l], self.Cl_fid[2, l]],
                    [self.Cl_fid[2, l], self.Cl_fid[1, l]]])
                Cov_the = np.array([
                    [cl['tt'][l]+self.noise_T[l], cl['te'][l]],
                    [cl['te'][l], cl['ee'][l]+self.noise_P[l]]])

            # get determinant of observational and theoretical covariance matrices
            det_obs = np.linalg.det(Cov_obs)
            det_the = np.linalg.det(Cov_the)

            # get determinant of mixed matrix (= sum of N theoretical
            # matrices with, in each of them, the nth column replaced
            # by that of the observational matrix)
            det_mix = 0.
            for i in range(num_modes):
                Cov_mix = np.copy(Cov_the)
                Cov_mix[:, i] = Cov_obs[:, i]
                det_mix += np.linalg.det(Cov_mix)

            chi2 += (2.*l+1.)*self.f_sky *\
                (det_mix/det_the + math.log(det_the/det_obs) - num_modes)

        return -chi2/2


###################################
# MPK TYPE LIKELIHOOD
# --> sdss, wigglez, etc.
###################################
class Likelihood_mpk(Likelihood):

    def __init__(self, path, data, command_line, common=False, common_dict={}):

        Likelihood.__init__(self, path, data, command_line)

        # require P(k) from class
        self.need_cosmo_arguments(data, {'output': 'mPk'})

        if common:
            self.add_common_knowledge(common_dict)

        try:
            self.use_halofit
        except:
            self.use_halofit = False

        if self.use_halofit:
            self.need_cosmo_arguments(data, {'non linear': 'halofit'})

        # sdssDR7 by T. Brinckmann
        # Based on Reid et al. 2010 arXiv:0907.1659 - Note: arXiv version not updated
        try:
            self.use_sdssDR7
        except:
            self.use_sdssDR7 = False

        # read values of k (in h/Mpc)
        self.k_size = self.max_mpk_kbands_use-self.min_mpk_kbands_use+1
        self.mu_size = 1
        self.k = np.zeros((self.k_size), 'float64')
        self.kh = np.zeros((self.k_size), 'float64')

        datafile = open(os.path.join(self.data_directory, self.kbands_file), 'r')
        for i in range(self.num_mpk_kbands_full):
            line = datafile.readline()
            while line.find('#') != -1:
                line = datafile.readline()
            if i+2 > self.min_mpk_kbands_use and i < self.max_mpk_kbands_use:
                self.kh[i-self.min_mpk_kbands_use+1] = float(line.split()[0])
        datafile.close()

        khmax = self.kh[-1]

        # check if need hight value of k for giggleZ
        try:
            self.use_giggleZ
        except:
            self.use_giggleZ = False

        # Try a new model, with an additional nuisance parameter. Note
        # that the flag use_giggleZPP0 being True requires use_giggleZ
        # to be True as well. Note also that it is defined globally,
        # and not for every redshift bin.
        if self.use_giggleZ:
            try:
                self.use_giggleZPP0
            except:
                self.use_giggleZPP0 = False
        else:
            self.use_giggleZPP0 = False

        # If the flag use_giggleZPP0 is set to True, the nuisance parameters
        # P0_a, P0_b, P0_c and P0_d are expected.
        if self.use_giggleZPP0:
            if 'P0_a' not in data.get_mcmc_parameters(['nuisance']):
                raise io_mp.LikelihoodError(
                    "In likelihood %s. " % self.name +
                    "P0_a is not defined in the .param file, whereas this " +
                    "nuisance parameter is required when the flag " +
                    "'use_giggleZPP0' is set to true for WiggleZ")

        if self.use_giggleZ:
            datafile = open(os.path.join(self.data_directory,self.giggleZ_fidpk_file), 'r')

            line = datafile.readline()
            k = float(line.split()[0])
            line_number = 1
            while (k < self.kh[0]):
                line = datafile.readline()
                k = float(line.split()[0])
                line_number += 1
            ifid_discard = line_number-2
            while (k < khmax):
                line = datafile.readline()
                k = float(line.split()[0])
                line_number += 1
            datafile.close()
            self.k_fid_size = line_number-ifid_discard+1
            khmax = k

        if self.use_halofit:
            khmax *= 2

        # require k_max and z_max from the cosmological module
        if self.use_sdssDR7:
            self.need_cosmo_arguments(data, {'z_max_pk': self.zmax})
            self.need_cosmo_arguments(data, {'P_k_max_h/Mpc': 7.5*self.kmax})
        else:
            self.need_cosmo_arguments(
                data, {'P_k_max_h/Mpc': khmax, 'z_max_pk': self.redshift})

        # read information on different regions in the sky
        try:
            self.has_regions
        except:
            self.has_regions = False

        if (self.has_regions):
            self.num_regions = len(self.used_region)
            self.num_regions_used = 0
            for i in range(self.num_regions):
                if (self.used_region[i]):
                    self.num_regions_used += 1
            if (self.num_regions_used == 0):
                raise io_mp.LikelihoodError(
                    "In likelihood %s. " % self.name +
                    "Mpk: no regions begin used in this data set")
        else:
            self.num_regions = 1
            self.num_regions_used = 1
            self.used_region = [True]

        # read window functions
        self.n_size = self.max_mpk_points_use-self.min_mpk_points_use+1

        self.window = np.zeros(
            (self.num_regions, self.n_size, self.k_size), 'float64')

        datafile = open(os.path.join(self.data_directory, self.windows_file), 'r')
        for i_region in range(self.num_regions):
            for i in range(self.num_mpk_points_full):
                line = datafile.readline()
                while line.find('#') != -1:
                    line = datafile.readline()
                if (i+2 > self.min_mpk_points_use and i < self.max_mpk_points_use):
                    for j in range(self.k_size):
                        self.window[i_region, i-self.min_mpk_points_use+1, j] = float(line.split()[j+self.min_mpk_kbands_use-1])
        datafile.close()

        # read measurements
        self.P_obs = np.zeros((self.num_regions, self.n_size), 'float64')
        self.P_err = np.zeros((self.num_regions, self.n_size), 'float64')

        datafile = open(os.path.join(self.data_directory, self.measurements_file), 'r')
        for i_region in range(self.num_regions):
            for i in range(self.num_mpk_points_full):
                line = datafile.readline()
                while line.find('#') != -1:
                    line = datafile.readline()
                if (i+2 > self.min_mpk_points_use and
                    i < self.max_mpk_points_use):
                    self.P_obs[i_region, i-self.min_mpk_points_use+1] = float(line.split()[3])
                    self.P_err[i_region, i-self.min_mpk_points_use+1] = float(line.split()[4])
        datafile.close()

        # read covariance matrices
        try:
            self.covmat_file
            self.use_covmat = True
        except:
            self.use_covmat = False

        try:
            self.use_invcov
        except:
            self.use_invcov = False

        self.invcov = np.zeros(
            (self.num_regions, self.n_size, self.n_size), 'float64')

        if self.use_covmat:
            cov = np.zeros((self.n_size, self.n_size), 'float64')
            invcov_tmp = np.zeros((self.n_size, self.n_size), 'float64')

            datafile = open(os.path.join(self.data_directory, self.covmat_file), 'r')
            for i_region in range(self.num_regions):
                for i in range(self.num_mpk_points_full):
                    line = datafile.readline()
                    while line.find('#') != -1:
                        line = datafile.readline()
                    if (i+2 > self.min_mpk_points_use and i < self.max_mpk_points_use):
                        for j in range(self.num_mpk_points_full):
                            if (j+2 > self.min_mpk_points_use and j < self.max_mpk_points_use):
                                cov[i-self.min_mpk_points_use+1,j-self.min_mpk_points_use+1] = float(line.split()[j])

                if self.use_invcov:
                    invcov_tmp = cov
                else:
                    invcov_tmp = np.linalg.inv(cov)
                for i in range(self.n_size):
                    for j in range(self.n_size):
                        self.invcov[i_region, i, j] = invcov_tmp[i, j]
            datafile.close()
        else:
            for i_region in range(self.num_regions):
                for j in range(self.n_size):
                    self.invcov[i_region, j, j] = \
                        1./(self.P_err[i_region, j]**2)

        # read fiducial model
        if self.use_giggleZ:
            self.P_fid = np.zeros((self.k_fid_size), 'float64')
            self.k_fid = np.zeros((self.k_fid_size), 'float64')
            datafile = open(os.path.join(self.data_directory,self.giggleZ_fidpk_file), 'r')
            for i in range(ifid_discard):
                line = datafile.readline()
            for i in range(self.k_fid_size):
                line = datafile.readline()
                self.k_fid[i] = float(line.split()[0])
                self.P_fid[i] = float(line.split()[1])
            datafile.close()

        # read integral constraint
        if self.use_sdssDR7:
            self.zerowindowfxn = np.zeros((self.k_size), 'float64')
            datafile = open(os.path.join(self.data_directory,self.zerowindowfxn_file), 'r')
            for i in range(self.k_size):
                line = datafile.readline()
                self.zerowindowfxn[i] = float(line.split()[0])
            datafile.close()
            self.zerowindowfxnsubtractdat = np.zeros((self.n_size), 'float64')
            datafile = open(os.path.join(self.data_directory,self.zerowindowfxnsubtractdat_file), 'r')
            line = datafile.readline()
            self.zerowindowfxnsubtractdatnorm = float(line.split()[0])
            for i in range(self.n_size):
                line = datafile.readline()
                self.zerowindowfxnsubtractdat[i] = float(line.split()[0])
            datafile.close()

        # initialize array of values for the nuisance parameters a1,a2
        if self.use_sdssDR7:
            nptsa1=self.nptsa1
            nptsa2=self.nptsa2
            a1maxval=self.a1maxval
            self.a1list=np.zeros(self.nptstot)
            self.a2list=np.zeros(self.nptstot)
            da1 = a1maxval/(nptsa1//2)
            da2 = self.a2maxpos(-a1maxval) / (nptsa2//2)
            count=0
            for i in range(-nptsa1//2, nptsa1//2+1):
                for j in range(-nptsa2//2, nptsa2//2+1):
                    a1val = da1*i
                    a2val = da2*j
                    if ((a2val >= 0.0 and a2val <= self.a2maxpos(a1val) and a2val >= self.a2minfinalpos(a1val)) or \
                        (a2val <= 0.0 and a2val <= self.a2maxfinalneg(a1val) and a2val >= self.a2minneg(a1val))):
                        if (self.testa1a2(a1val,a2val) == False):
                            raise io_mp.LikelihoodError(
                                'Error in likelihood %s ' % (self.name) +
                                'Nuisance parameter values not valid: %s %s' % (a1,a2) )
                        if(count >= self.nptstot):
                            raise io_mp.LikelihoodError(
                                'Error in likelihood %s ' % (self.name) +
                                'count > nptstot failure' )
                        self.a1list[count]=a1val
                        self.a2list[count]=a2val
                        count=count+1

        return

    # functions added for nuisance parameter space checks.
    def a2maxpos(self,a1val):
        a2max = -1.0
        if (a1val <= min(self.s1/self.k1,self.s2/self.k2)):
            a2max = min(self.s1/self.k1**2 - a1val/self.k1, self.s2/self.k2**2 - a1val/self.k2)
        return a2max

    def a2min1pos(self,a1val):
        a2min1 = 0.0
        if(a1val <= 0.0):
            a2min1 = max(-self.s1/self.k1**2 - a1val/self.k1, -self.s2/self.k2**2 - a1val/self.k2, 0.0)
        return a2min1

    def a2min2pos(self,a1val):
        a2min2 = 0.0
        if(abs(a1val) >= 2.0*self.s1/self.k1 and a1val <= 0.0):
            a2min2 = a1val**2/self.s1*0.25
        return a2min2

    def a2min3pos(self,a1val):
        a2min3 = 0.0
        if(abs(a1val) >= 2.0*self.s2/self.k2 and a1val <= 0.0):
            a2min3 = a1val**2/self.s2*0.25
        return a2min3

    def a2minfinalpos(self,a1val):
        a2minpos = max(self.a2min1pos(a1val),self.a2min2pos(a1val),self.a2min3pos(a1val))
        return a2minpos

    def a2minneg(self,a1val):
        if (a1val >= max(-self.s1/self.k1,-self.s2/self.k2)):
            a2min = max(-self.s1/self.k1**2 - a1val/self.k1, -self.s2/self.k2**2 - a1val/self.k2)
        else:
            a2min = 1.0
        return a2min

    def a2max1neg(self,a1val):
        if(a1val >= 0.0):
            a2max1 = min(self.s1/self.k1**2 - a1val/self.k1, self.s2/self.k2**2 - a1val/self.k2, 0.0)
        else:
            a2max1 = 0.0
        return a2max1

    def a2max2neg(self,a1val):
        a2max2 = 0.0
        if(abs(a1val) >= 2.0*self.s1/self.k1 and a1val >= 0.0):
            a2max2 = -a1val**2/self.s1*0.25
        return a2max2

    def a2max3neg(self,a1val):
        a2max3 = 0.0
        if(abs(a1val) >= 2.0*self.s2/self.k2 and a1val >= 0.0):
            a2max3 = -a1val**2/self.s2*0.25
        return a2max3

    def a2maxfinalneg(self,a1val):
        a2maxneg = min(self.a2max1neg(a1val),self.a2max2neg(a1val),self.a2max3neg(a1val))
        return a2maxneg

    def testa1a2(self,a1val, a2val):
        testresult = True
        # check if there's an extremum; either a1val or a2val has to be negative, not both
        if (a2val==0.):
             return testresult #not in the original code, but since a2val=0 returns True this way I avoid zerodivisionerror
        kext = -a1val/2.0/a2val
        diffval = abs(a1val*kext + a2val*kext**2)
        if(kext > 0.0 and kext <= self.k1 and diffval > self.s1):
            testresult = False
        if(kext > 0.0 and kext <= self.k2 and diffval > self.s2):
            testresult = False
        if (abs(a1val*self.k1 + a2val*self.k1**2) > self.s1):
            testresult = False
        if (abs(a1val*self.k2 + a2val*self.k2**2) > self.s2):
            testresult = False
        return testresult


    def add_common_knowledge(self, common_dictionary):
        """
        Add to a class the content of a shared dictionary of attributes

        The purpose of this method is to set some attributes globally for a Pk
        likelihood, that are shared amongst all the redshift bins (in
        WiggleZ.data for instance, a few flags and numbers are defined that
        will be transfered to wigglez_a, b, c and d

        """
        for key, value in dictitems(common_dictionary):
            # First, check if the parameter exists already
            try:
                exec("self.%s" % key)
                warnings.warn(
                    "parameter %s from likelihood %s will be replaced by " +
                    "the common knowledge routine" % (key, self.name))
            except:
                if type(value) != type('foo'):
                    exec("self.%s = %s" % (key, value))
                else:
                    exec("self.%s = '%s'" % (key, value))

    # compute likelihood
    def loglkl(self, cosmo, data):

        # reduced Hubble parameter
        h = cosmo.h()

        # WiggleZ and sdssDR7 specific
        if self.use_scaling:
            # angular diameter distance at this redshift, in Mpc
            d_angular = cosmo.angular_distance(self.redshift)

            # radial distance at this redshift, in Mpc, is simply 1/H (itself
            # in Mpc^-1). Hz is an array, with only one element.
            r, Hz = cosmo.z_of_r([self.redshift])
            d_radial = 1/Hz[0]

            # scaling factor = (d_angular**2 * d_radial)^(1/3) for the
            # fiducial cosmology used in the data files of the observations
            # divided by the same quantity for the cosmology we are comparing with.
            # The fiducial values are stored in the .data files for
            # each experiment, and are truly in Mpc. Beware for a potential
            # difference with CAMB conventions here.
            scaling = pow(
                (self.d_angular_fid/d_angular)**2 *
                (self.d_radial_fid/d_radial), 1./3.)
        else:
            scaling = 1
        # get rescaled values of k in 1/Mpc
        self.k = self.kh*h*scaling

        # get P(k) at right values of k, convert it to (Mpc/h)^3 and rescale it
        P_lin = np.zeros((self.k_size), 'float64')

        # If the flag use_giggleZ is set to True, the power spectrum retrieved
        # from Class will get rescaled by the fiducial power spectrum given by
        # the GiggleZ N-body simulations CITE
        if self.use_giggleZ:
            P = np.zeros((self.k_fid_size), 'float64')
            for i in range(self.k_fid_size):
                P[i] = cosmo.pk(self.k_fid[i]*h, self.redshift)
                power = 0
                # The following create a polynome in k, which coefficients are
                # stored in the .data files of the experiments.
                for j in range(6):
                    power += self.giggleZ_fidpoly[j]*self.k_fid[i]**j
                # rescale P by fiducial model and get it in (Mpc/h)**3
                P[i] *= pow(10, power)*(h/scaling)**3/self.P_fid[i]

            if self.use_giggleZPP0:
                # Shot noise parameter addition to GiggleZ model. It should
                # recover the proper nuisance parameter, depending on the name.
                # I.e., Wigglez_A should recover P0_a, etc...
                tag = self.name[-2:]  # circle over "_a", "_b", etc...
                P0_value = data.mcmc_parameters['P0'+tag]['current'] *\
                    data.mcmc_parameters['P0'+tag]['scale']
                P_lin = np.interp(self.kh,self.k_fid,P+P0_value)
            else:
                # get P_lin by interpolation. It is still in (Mpc/h)**3
                P_lin = np.interp(self.kh, self.k_fid, P)

        elif self.use_sdssDR7:
            kh = np.logspace(math.log(1e-3),math.log(1.0),num=(math.log(1.0)-math.log(1e-3))/0.01+1,base=math.exp(1.0)) # k in h/Mpc
            # Rescale the scaling factor by the fiducial value for h divided by the sampled value
            # h=0.701 was used for the N-body calibration simulations
            scaling = scaling * (0.701/h)
            k = kh*h # k in 1/Mpc

            # Define redshift bins and associated bao 2 sigma value [NEAR, MID, FAR]
            z = np.array([0.235, 0.342, 0.421])
            sigma2bao = np.array([86.9988, 85.1374, 84.5958])
            # Initialize arrays
            # Analytical growth factor for each redshift bin
            D_growth = np.zeros(len(z))
            # P(k) *with* wiggles, both linear and nonlinear
            Plin = np.zeros(len(k), 'float64')
            Pnl = np.zeros(len(k), 'float64')
            # P(k) *without* wiggles, both linear and nonlinear
            Psmooth = np.zeros(len(k), 'float64')
            Psmooth_nl = np.zeros(len(k), 'float64')
            # Damping function and smeared P(k)
            fdamp = np.zeros([len(k), len(z)], 'float64')
            Psmear = np.zeros([len(k), len(z)], 'float64')
            # Ratio of smoothened non-linear to linear P(k)
            nlratio = np.zeros([len(k), len(z)], 'float64')
            # Loop over each redshift bin
            for j in range(len(z)):
                # Compute growth factor at each redshift
                # This growth factor is normalized by the growth factor today
                D_growth[j] = cosmo.scale_independent_growth_factor(z[j])
                # Compute Pk *with* wiggles, both linear and nonlinear
                # Get P(k) at right values of k in Mpc**3, convert it to (Mpc/h)^3 and rescale it
                # Get values of P(k) in Mpc**3
                for i in range(len(k)):
                    Plin[i] = cosmo.pk_lin(k[i], z[j])
                    Pnl[i] = cosmo.pk(k[i], z[j])
                # Get rescaled values of P(k) in (Mpc/h)**3
                Plin *= h**3 #(h/scaling)**3
                Pnl *= h**3 #(h/scaling)**3
                # Compute Pk *without* wiggles, both linear and nonlinear
                Psmooth = self.remove_bao(kh,Plin)
                Psmooth_nl = self.remove_bao(kh,Pnl)
                # Apply Gaussian damping due to non-linearities
                fdamp[:,j] = np.exp(-0.5*sigma2bao[j]*kh**2)
                Psmear[:,j] = Plin*fdamp[:,j]+Psmooth*(1.0-fdamp[:,j])
                # Take ratio of smoothened non-linear to linear P(k)
                nlratio[:,j] = Psmooth_nl/Psmooth

            # Save fiducial model for non-linear corrections using the flat fiducial
            # Omega_m = 0.25, Omega_L = 0.75, h = 0.701
            # Re-run if changes are made to how non-linear corrections are done
            # e.g. the halofit implementation in CLASS
            # To re-run fiducial, set <experiment>.create_fid = True in .data file
            # Can leave option enabled, as it will only compute once at the start
            try:
                self.create_fid
            except:
                self.create_fid = False

            if self.create_fid == True:
                # Calculate relevant flat fiducial quantities
                fidnlratio, fidNEAR, fidMID, fidFAR = self.get_flat_fid(cosmo,data,kh,z,sigma2bao)
                try:
                    existing_fid = np.loadtxt('data/sdss_lrgDR7/sdss_lrgDR7_fiducialmodel.dat')
                    print('sdss_lrgDR7: Checking fiducial deviations for near, mid and far bins:', np.sum(existing_fid[:,1] - fidNEAR),np.sum(existing_fid[:,2] - fidMID), np.sum(existing_fid[:,3] - fidFAR))
                    if np.sum(existing_fid[:,1] - fidNEAR) + np.sum(existing_fid[:,2] - fidMID) + np.sum(existing_fid[:,3] - fidFAR) < 10**-5:
                        self.create_fid = False
                except:
                    pass
                if self.create_fid == True:
                    print('sdss_lrgDR7: Creating fiducial file with Omega_b = 0.25, Omega_L = 0.75, h = 0.701')
                    print('             Required for non-linear modeling')
                    # Save non-linear corrections from N-body sims for each redshift bin
                    arr=np.zeros((np.size(kh),7))
                    arr[:,0]=kh
                    arr[:,1]=fidNEAR
                    arr[:,2]=fidMID
                    arr[:,3]=fidFAR
                    # Save non-linear corrections from halofit for each redshift bin
                    arr[:,4:7]=fidnlratio
                    np.savetxt('data/sdss_lrgDR7/sdss_lrgDR7_fiducialmodel.dat',arr)
                    self.create_fid = False
                    print('             Fiducial created')

            # Load fiducial model
            fiducial = np.loadtxt('data/sdss_lrgDR7/sdss_lrgDR7_fiducialmodel.dat')
            fid = fiducial[:,1:4]
            fidnlratio = fiducial[:,4:7]

            # Put all factors together to obtain the P(k) for each redshift bin
            Pnear=np.interp(kh,kh,Psmear[:,0]*(nlratio[:,0]/fidnlratio[:,0])*fid[:,0]*D_growth[0]**(-2.))
            Pmid =np.interp(kh,kh,Psmear[:,1]*(nlratio[:,1]/fidnlratio[:,1])*fid[:,1]*D_growth[1]**(-2.))
            Pfar =np.interp(kh,kh,Psmear[:,2]*(nlratio[:,2]/fidnlratio[:,2])*fid[:,2]*D_growth[2]**(-2.))

            # Define and rescale k
            self.k=self.kh*h*scaling
            # Weighted mean of the P(k) for each redshift bin
            P_lin=(0.395*Pnear+0.355*Pmid+0.250*Pfar)
            P_lin=np.interp(self.k,kh*h,P_lin)*(1./scaling)**3 # remember self.k is scaled but self.kh isn't

        else:
            # get rescaled values of k in 1/Mpc
            self.k = self.kh*h*scaling
            # get values of P(k) in Mpc**3
            for i in range(self.k_size):
                P_lin[i] = cosmo.pk(self.k[i], self.redshift)
            # get rescaled values of P(k) in (Mpc/h)**3
            P_lin *= (h/scaling)**3

        # infer P_th from P_lin. It is still in (Mpc/h)**3. TODO why was it
        # called P_lin in the first place ? Couldn't we use now P_th all the
        # way ?
        P_th = P_lin

        if self.use_sdssDR7:
            chisq =np.zeros(self.nptstot)
            chisqmarg = np.zeros(self.nptstot)

            Pth = P_th
            Pth_k = P_th*(self.k/h) # self.k has the scaling included, so self.k/h != self.kh
            Pth_k2 = P_th*(self.k/h)**2

            WPth = np.dot(self.window[0,:], Pth)
            WPth_k = np.dot(self.window[0,:], Pth_k)
            WPth_k2 = np.dot(self.window[0,:], Pth_k2)

            sumzerow_Pth = np.sum(self.zerowindowfxn*Pth)/self.zerowindowfxnsubtractdatnorm
            sumzerow_Pth_k = np.sum(self.zerowindowfxn*Pth_k)/self.zerowindowfxnsubtractdatnorm
            sumzerow_Pth_k2 = np.sum(self.zerowindowfxn*Pth_k2)/self.zerowindowfxnsubtractdatnorm

            covdat = np.dot(self.invcov[0,:,:],self.P_obs[0,:])
            covth  = np.dot(self.invcov[0,:,:],WPth)
            covth_k  = np.dot(self.invcov[0,:,:],WPth_k)
            covth_k2  = np.dot(self.invcov[0,:,:],WPth_k2)
            covth_zerowin  = np.dot(self.invcov[0,:,:],self.zerowindowfxnsubtractdat)
            sumDD = np.sum(self.P_obs[0,:] * covdat)
            sumDT = np.sum(self.P_obs[0,:] * covth)
            sumDT_k = np.sum(self.P_obs[0,:] * covth_k)
            sumDT_k2 = np.sum(self.P_obs[0,:] * covth_k2)
            sumDT_zerowin = np.sum(self.P_obs[0,:] * covth_zerowin)

            sumTT = np.sum(WPth*covth)
            sumTT_k = np.sum(WPth*covth_k)
            sumTT_k2 = np.sum(WPth*covth_k2)
            sumTT_k_k = np.sum(WPth_k*covth_k)
            sumTT_k_k2 = np.sum(WPth_k*covth_k2)
            sumTT_k2_k2 = np.sum(WPth_k2*covth_k2)
            sumTT_zerowin = np.sum(WPth*covth_zerowin)
            sumTT_k_zerowin = np.sum(WPth_k*covth_zerowin)
            sumTT_k2_zerowin = np.sum(WPth_k2*covth_zerowin)
            sumTT_zerowin_zerowin = np.sum(self.zerowindowfxnsubtractdat*covth_zerowin)

            currminchisq = 1000.0

            # analytic marginalization over a1,a2
            for i in range(self.nptstot):
                a1val = self.a1list[i]
                a2val = self.a2list[i]
                zerowinsub = -(sumzerow_Pth + a1val*sumzerow_Pth_k + a2val*sumzerow_Pth_k2)
                sumDT_tot = sumDT + a1val*sumDT_k + a2val*sumDT_k2 + zerowinsub*sumDT_zerowin
                sumTT_tot = sumTT + a1val**2.0*sumTT_k_k + a2val**2.0*sumTT_k2_k2 + \
                    zerowinsub**2.0*sumTT_zerowin_zerowin + \
                    2.0*a1val*sumTT_k + 2.0*a2val*sumTT_k2 + 2.0*a1val*a2val*sumTT_k_k2 + \
                    2.0*zerowinsub*sumTT_zerowin + 2.0*zerowinsub*a1val*sumTT_k_zerowin + \
                    2.0*zerowinsub*a2val*sumTT_k2_zerowin
                minchisqtheoryamp = sumDT_tot/sumTT_tot
                chisq[i] = sumDD - 2.0*minchisqtheoryamp*sumDT_tot + minchisqtheoryamp**2.0*sumTT_tot
                chisqmarg[i] = sumDD - sumDT_tot**2.0/sumTT_tot + math.log(sumTT_tot) - \
                    2.0*math.log(1.0 + math.erf(sumDT_tot/2.0/math.sqrt(sumTT_tot)))
                if(i == 0 or chisq[i] < currminchisq):
                    myminchisqindx = i
                    currminchisq = chisq[i]
                    currminchisqmarg = chisqmarg[i]
                    minchisqtheoryampminnuis = minchisqtheoryamp
                if(i == int(self.nptstot/2)):
                    chisqnonuis = chisq[i]
                    minchisqtheoryampnonuis = minchisqtheoryamp
                    if(abs(a1val) > 0.001 or abs(a2val) > 0.001):
                         print('sdss_lrgDR7: ahhhh! violation!!', a1val, a2val)

            # numerically marginalize over a1,a2 now using values stored in chisq
            minchisq = np.min(chisqmarg)
            maxchisq = np.max(chisqmarg)

            LnLike = np.sum(np.exp(-(chisqmarg-minchisq)/2.0)/(self.nptstot*1.0))
            if(LnLike == 0):
                #LnLike = LogZero
                raise io_mp.LikelihoodError(
                    'Error in likelihood %s ' % (self.name) +
                    'LRG LnLike LogZero error.' )
            else:
                chisq = -2.*math.log(LnLike) + minchisq
            #print('DR7 chi2/2=',chisq/2.)

        #if we are not using DR7
        else:
            W_P_th = np.zeros((self.n_size), 'float64')

            # starting analytic marginalisation over bias

            # Define quantities living in all the regions possible. If only a few
            # regions are selected in the .data file, many elements from these
            # arrays will stay at 0.
            P_data_large = np.zeros(
                (self.n_size*self.num_regions_used), 'float64')
            W_P_th_large = np.zeros(
                (self.n_size*self.num_regions_used), 'float64')
            cov_dat_large = np.zeros(
                (self.n_size*self.num_regions_used), 'float64')
            cov_th_large = np.zeros(
                (self.n_size*self.num_regions_used), 'float64')

            normV = 0

            # Loop over all the available regions
            for i_region in range(self.num_regions):
                # In each region that was selected with the array of flags
                # self.used_region, define boundaries indices, and fill in the
                # corresponding windowed power spectrum. All the unused regions
                # will still be set to zero as from the initialization, which will
                # not contribute anything in the final sum.

                if self.used_region[i_region]:
                    imin = i_region*self.n_size
                    imax = (i_region+1)*self.n_size-1

                    W_P_th = np.dot(self.window[i_region, :], P_th)
                    #print(W_P_th)
                    for i in range(self.n_size):
                        P_data_large[imin+i] = self.P_obs[i_region, i]
                        W_P_th_large[imin+i] = W_P_th[i]
                        cov_dat_large[imin+i] = np.dot(
                            self.invcov[i_region, i, :],
                            self.P_obs[i_region, :])
                        cov_th_large[imin+i] = np.dot(
                            self.invcov[i_region, i, :],
                            W_P_th[:])

            # Explain what it is TODO
            normV += np.dot(W_P_th_large, cov_th_large)
            # Sort of bias TODO ?
            b_out = np.sum(W_P_th_large*cov_dat_large) / \
                np.sum(W_P_th_large*cov_th_large)

            # Explain this formula better, link to article ?
            chisq = np.dot(P_data_large, cov_dat_large) - \
                np.dot(W_P_th_large, cov_dat_large)**2/normV
            #print('WiggleZ chi2=',chisq/2.)

        return -chisq/2

    def remove_bao(self,k_in,pk_in):
        # De-wiggling routine by Mario Ballardini

        # This k range has to contain the BAO features:
        k_ref=[2.8e-2, 4.5e-1]

        # Get interpolating function for input P(k) in log-log space:
        _interp_pk = scipy.interpolate.interp1d( np.log(k_in), np.log(pk_in),
                                                 kind='quadratic', bounds_error=False )
        interp_pk = lambda x: np.exp(_interp_pk(np.log(x)))

        # Spline all (log-log) points outside k_ref range:
        idxs = np.where(np.logical_or(k_in <= k_ref[0], k_in >= k_ref[1]))
        _pk_smooth = scipy.interpolate.UnivariateSpline( np.log(k_in[idxs]),
                                                         np.log(pk_in[idxs]), k=3, s=0 )
        pk_smooth = lambda x: np.exp(_pk_smooth(np.log(x)))

        # Find second derivative of each spline:
        fwiggle = scipy.interpolate.UnivariateSpline(k_in, pk_in / pk_smooth(k_in), k=3, s=0)
        derivs = np.array([fwiggle.derivatives(_k) for _k in k_in]).T
        d2 = scipy.interpolate.UnivariateSpline(k_in, derivs[2], k=3, s=1.0)

        # Find maxima and minima of the gradient (zeros of 2nd deriv.), then put a
        # low-order spline through zeros to subtract smooth trend from wiggles fn.
        wzeros = d2.roots()
        wzeros = wzeros[np.where(np.logical_and(wzeros >= k_ref[0], wzeros <= k_ref[1]))]
        wzeros = np.concatenate((wzeros, [k_ref[1],]))
        wtrend = scipy.interpolate.UnivariateSpline(wzeros, fwiggle(wzeros), k=3, s=0)

        # Construct smooth no-BAO:
        idxs = np.where(np.logical_and(k_in > k_ref[0], k_in < k_ref[1]))
        pk_nobao = pk_smooth(k_in)
        pk_nobao[idxs] *= wtrend(k_in[idxs])

        # Construct interpolating functions:
        ipk = scipy.interpolate.interp1d( k_in, pk_nobao, kind='linear',
                                          bounds_error=False, fill_value=0. )

        pk_nobao = ipk(k_in)

        return pk_nobao

    def get_flat_fid(self,cosmo,data,kh,z,sigma2bao):
        # SDSS DR7 LRG specific function
        # Compute fiducial properties for a flat fiducial
        # with Omega_m = 0.25, Omega_L = 0.75, h = 0.701
        param_backup = data.cosmo_arguments
        data.cosmo_arguments = {'P_k_max_h/Mpc': 1.5, 'ln10^{10}A_s': 3.0, 'N_ur': 3.04, 'h': 0.701,
                                'omega_b': 0.035*0.701**2, 'non linear': ' halofit ', 'YHe': 0.24, 'k_pivot': 0.05,
                                'n_s': 0.96, 'tau_reio': 0.084, 'z_max_pk': 0.5, 'output': ' mPk ',
                                'omega_cdm': 0.215*0.701**2, 'T_cmb': 2.726}
        cosmo.empty()
        cosmo.set(data.cosmo_arguments)
        cosmo.compute(['lensing'])
        h = data.cosmo_arguments['h']
        k = kh*h
        # P(k) *with* wiggles, both linear and nonlinear
        Plin = np.zeros(len(k), 'float64')
        Pnl = np.zeros(len(k), 'float64')
        # P(k) *without* wiggles, both linear and nonlinear
        Psmooth = np.zeros(len(k), 'float64')
        Psmooth_nl = np.zeros(len(k), 'float64')
        # Damping function and smeared P(k)
        fdamp = np.zeros([len(k), len(z)], 'float64')
        Psmear = np.zeros([len(k), len(z)], 'float64')
        # Ratio of smoothened non-linear to linear P(k)
        fidnlratio = np.zeros([len(k), len(z)], 'float64')
        # Loop over each redshift bin
        for j in range(len(z)):
            # Compute Pk *with* wiggles, both linear and nonlinear
            # Get P(k) at right values of k in Mpc**3, convert it to (Mpc/h)^3 and rescale it
            # Get values of P(k) in Mpc**3
            for i in range(len(k)):
                Plin[i] = cosmo.pk_lin(k[i], z[j])
                Pnl[i] = cosmo.pk(k[i], z[j])
            # Get rescaled values of P(k) in (Mpc/h)**3
            Plin *= h**3 #(h/scaling)**3
            Pnl *= h**3 #(h/scaling)**3
            # Compute Pk *without* wiggles, both linear and nonlinear
            Psmooth = self.remove_bao(kh,Plin)
            Psmooth_nl = self.remove_bao(kh,Pnl)
            # Apply Gaussian damping due to non-linearities
            fdamp[:,j] = np.exp(-0.5*sigma2bao[j]*kh**2)
            Psmear[:,j] = Plin*fdamp[:,j]+Psmooth*(1.0-fdamp[:,j])
            # Take ratio of smoothened non-linear to linear P(k)
            fidnlratio[:,j] = Psmooth_nl/Psmooth

        # Polynomials to shape small scale behavior from N-body sims
        kdata=kh
        fidpolyNEAR=np.zeros(np.size(kdata))
        fidpolyNEAR[kdata<=0.194055] = (1.0 - 0.680886*kdata[kdata<=0.194055] + 6.48151*kdata[kdata<=0.194055]**2)
        fidpolyNEAR[kdata>0.194055] = (1.0 - 2.13627*kdata[kdata>0.194055] + 21.0537*kdata[kdata>0.194055]**2 - 50.1167*kdata[kdata>0.194055]**3 + 36.8155*kdata[kdata>0.194055]**4)*1.04482
        fidpolyMID=np.zeros(np.size(kdata))
        fidpolyMID[kdata<=0.19431] = (1.0 - 0.530799*kdata[kdata<=0.19431] + 6.31822*kdata[kdata<=0.19431]**2)
        fidpolyMID[kdata>0.19431] = (1.0 - 1.97873*kdata[kdata>0.19431] + 20.8551*kdata[kdata>0.19431]**2 - 50.0376*kdata[kdata>0.19431]**3 + 36.4056*kdata[kdata>0.19431]**4)*1.04384
        fidpolyFAR=np.zeros(np.size(kdata))
        fidpolyFAR[kdata<=0.19148] = (1.0 - 0.475028*kdata[kdata<=0.19148] + 6.69004*kdata[kdata<=0.19148]**2)
        fidpolyFAR[kdata>0.19148] = (1.0 - 1.84891*kdata[kdata>0.19148] + 21.3479*kdata[kdata>0.19148]**2 - 52.4846*kdata[kdata>0.19148]**3 + 38.9541*kdata[kdata>0.19148]**4)*1.03753

        fidNEAR=np.interp(kh,kdata,fidpolyNEAR)
        fidMID=np.interp(kh,kdata,fidpolyMID)
        fidFAR=np.interp(kh,kdata,fidpolyFAR)

        cosmo.empty()
        data.cosmo_arguments = param_backup
        cosmo.set(data.cosmo_arguments)
        cosmo.compute(['lensing'])

        return fidnlratio, fidNEAR, fidMID, fidFAR

class Likelihood_sn(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        # try and import pandas
        try:
            import pandas
        except ImportError:
            raise io_mp.MissingLibraryError(
                "This likelihood has a lot of IO manipulation. You have "
                "to install the 'pandas' library to use it. Please type:\n"
                "`(sudo) pip install pandas --user`")

        # check that every conflicting experiments is not present in the list
        # of tested experiments, in which case, complain
        if hasattr(self, 'conflicting_experiments'):
            for conflict in self.conflicting_experiments:
                if conflict in data.experiments:
                    raise io_mp.LikelihoodError(
                        'conflicting %s measurements, you can ' % conflict +
                        ' have either %s or %s ' % (self.name, conflict) +
                        'as an experiment, not both')

        # Read the configuration file, supposed to be called self.settings.
        # Note that we unfortunately can not
        # immediatly execute the file, as it is not formatted as strings.
        assert hasattr(self, 'settings') is True, (
            "You need to provide a settings file")
        self.read_configuration_file()

    def read_configuration_file(self):
        """
        Extract Python variables from the configuration file

        This routine performs the equivalent to the program "inih" used in the
        original c++ library.
        """
        settings_path = os.path.join(self.data_directory, self.settings)
        with open(settings_path, 'r') as config:
            for line in config:
                # Dismiss empty lines and commented lines
                if line and line.find('#') == -1 and line not in ['\n', '\r\n']:
                    lhs, rhs = [elem.strip() for elem in line.split('=')]
                    # lhs will always be a string, so set the attribute to this
                    # likelihood. The right hand side requires more work.
                    # First case, if set to T or F for True or False
                    if str(rhs) in ['T', 'F']:
                        rhs = True if str(rhs) == 'T' else False
                    # It can also be a path, starting with 'data/'. We remove
                    # this leading folder path
                    elif str(rhs).find('data/') != -1:
                        rhs = rhs.replace('data/', '')
                    else:
                        # Try  to convert it to a float
                        try:
                            rhs = float(rhs)
                        # If it fails, it is a string
                        except ValueError:
                            rhs = str(rhs)
                    # Set finally rhs to be a parameter of the class
                    setattr(self, lhs, rhs)

    def read_matrix(self, path):
        """
        extract the matrix from the path

        This routine uses the blazing fast pandas library (0.10 seconds to load
        a 740x740 matrix). If not installed, it uses a custom routine that is
        twice as slow (but still 4 times faster than the straightforward
        numpy.loadtxt method)

        .. note::

            the length of the matrix is stored on the first line... then it has
            to be unwrapped. The pandas routine read_table understands this
            immediatly, though.

        """
        from pandas import read_table
        path = os.path.join(self.data_directory, path)
        # The first line should contain the length.
        with open(path, 'r') as text:
            length = int(text.readline())

        # Note that this function does not require to skiprows, as it
        # understands the convention of writing the length in the first
        # line
        matrix = read_table(path).as_matrix().reshape((length, length))

        return matrix

    def read_light_curve_parameters(self):
        """
        Read the file jla_lcparams.txt containing the SN data

        .. note::

            the length of the resulting array should be equal to the length of
            the covariance matrices stored in C00, etc...

        """
        from pandas import read_table
        path = os.path.join(self.data_directory, self.data_file)

        # Recover the names of the columns. The names '3rdvar' and 'd3rdvar'
        # will be changed, because 3rdvar is not a valid variable name
        with open(path, 'r') as text:
            clean_first_line = text.readline()[1:].strip()
            names = [e.strip().replace('3rd', 'third')
                     for e in clean_first_line.split()]

        lc_parameters = read_table(
            path, sep=' ', names=names, header=0, index_col=False)
        return lc_parameters


class Likelihood_clocks(Likelihood):
    """Base implementation of H(z) measurements"""

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        # Read the content of the data file, containing z, Hz and error
        total = np.loadtxt(
            os.path.join(self.data_directory, self.data_file))

        # Store the columns separately
        self.z = total[:, 0]
        self.Hz = total[:, 1]
        self.err = total[:, 2]

    def loglkl(self, cosmo, data):

        # Store the speed of light in km/s
        c_light_km_per_sec = const.c/1000.
        chi2 = 0

        # Loop over the redshifts
        for index, z in enumerate(self.z):
            # Query the cosmo module for the Hubble rate (in 1/Mpc), and
            # convert it to km/s/Mpc
            H_cosmo = cosmo.Hubble(z)*c_light_km_per_sec
            # Add to the tota chi2
            chi2 += (self.Hz[index]-H_cosmo)**2/self.err[index]**2

        return -0.5 * chi2

###################################
# ISW-Likelihood
# by B. Stoelzner
###################################
class Likelihood_isw(Likelihood):
    def __init__(self, path, data, command_line):
        # Initialize
        Likelihood.__init__(self, path, data, command_line)
        self.need_cosmo_arguments(data, {'output': 'mPk','P_k_max_h/Mpc' : 300,'z_max_pk' : 5.1})

        # Read l,C_l, and the covariance matrix of the autocorrelation of the survey and the crosscorrelation of the survey with the CMB
        self.l_cross,cl_cross=np.loadtxt(os.path.join(self.data_directory,self.cl_cross_file),unpack=True,usecols=(0,1))
        self.l_auto,cl_auto=np.loadtxt(os.path.join(self.data_directory,self.cl_auto_file),unpack=True,usecols=(0,1))
        cov_cross=np.loadtxt(os.path.join(self.data_directory,self.cov_cross_file))
        cov_auto=np.loadtxt(os.path.join(self.data_directory,self.cov_auto_file))

        # Extract data in the specified range in l.
        self.l_cross=self.l_cross[self.l_min_cross:self.l_max_cross+1]
        cl_cross=cl_cross[self.l_min_cross:self.l_max_cross+1]
        self.l_auto=self.l_auto[self.l_min_auto:self.l_max_auto+1]
        cl_auto=cl_auto[self.l_min_auto:self.l_max_auto+1]
        cov_cross=cov_cross[self.l_min_cross:self.l_max_cross+1,self.l_min_cross:self.l_max_cross+1]
        cov_auto=cov_auto[self.l_min_auto:self.l_max_auto+1,self.l_min_auto:self.l_max_auto+1]

        # Create logarithically spaced bins in l.
        self.bins_cross=np.ceil(np.logspace(np.log10(self.l_min_cross),np.log10(self.l_max_cross),self.n_bins_cross+1))
        self.bins_auto=np.ceil(np.logspace(np.log10(self.l_min_auto),np.log10(self.l_max_auto),self.n_bins_auto+1))

        # Bin l,C_l, and covariance matrix in the previously defined bins
        self.l_binned_cross,self.cl_binned_cross,self.cov_binned_cross=self.bin_cl(self.l_cross,cl_cross,self.bins_cross,cov_cross)
        self.l_binned_auto,self.cl_binned_auto,self.cov_binned_auto=self.bin_cl(self.l_auto,cl_auto,self.bins_auto,cov_auto)

        # Read the redshift distribution of objects in the survey, perform an interpolation of dN/dz(z), and calculate the normalization in this redshift bin
        zz,dndz=np.loadtxt(os.path.join(self.data_directory,self.dndz_file),unpack=True,usecols=(0,1))
        self.dndz=scipy.interpolate.interp1d(zz,dndz,kind='cubic')
        self.norm=scipy.integrate.quad(self.dndz,self.z_min,self.z_max)[0]

    def bin_cl(self,l,cl,bins,cov=None):
        # This function bins l,C_l, and the covariance matrix in given bins in l
        B=[]
        for i in range(1,len(bins)):
            if i!=len(bins)-1:
                a=np.where((l<bins[i])&(l>=bins[i-1]))[0]
            else:
                a=np.where((l<=bins[i])&(l>=bins[i-1]))[0]
            c=np.zeros(len(l))
            c[a]=1./len(a)
            B.append(c)
        l_binned=np.dot(B,l)
        cl_binned=np.dot(B,cl)
        if cov is not None:
            cov_binned=np.dot(B,np.dot(cov,np.transpose(B)))
            return l_binned,cl_binned,cov_binned
        else:
            return l_binned,cl_binned

    def integrand_cross(self,z,cosmo,l):
        # This function will be integrated to calculate the exspected crosscorrelation between the survey and the CMB
        c= const.c/1000.
        H0=cosmo.h()*100
        Om=cosmo.Omega0_m()
        k=lambda z:(l+0.5)/(cosmo.angular_distance(z)*(1+z))
        return (3*Om*H0**2)/((c**2)*(l+0.5)**2)*self.dndz(z)*cosmo.Hubble(z)*cosmo.scale_independent_growth_factor(z)*scipy.misc.derivative(lambda z:cosmo.scale_independent_growth_factor(z)*(1+z),x0=z,dx=1e-4)*cosmo.pk(k(z),0)/self.norm

    def integrand_auto(self,z,cosmo,l):
        # This function will be integrated to calculate the expected autocorrelation of the survey
        c= const.c/1000.
        H0=cosmo.h()*100
        k=lambda z:(l+0.5)/(cosmo.angular_distance(z)*(1+z))
        return (self.dndz(z))**2*(cosmo.scale_independent_growth_factor(z))**2*cosmo.pk(k(z),0)*cosmo.Hubble(z)/(cosmo.angular_distance(z)*(1+z))**2/self.norm**2

    def compute_loglkl(self, cosmo, data,b):
        # Retrieve sampled parameter
        A=data.mcmc_parameters['A_ISW']['current']*data.mcmc_parameters['A_ISW']['scale']

        # Calculate the expected auto- and crosscorrelation by integrating over the redshift.
        cl_binned_cross_theory=np.array([(scipy.integrate.quad(self.integrand_cross,self.z_min,self.z_max,args=(cosmo,self.bins_cross[ll]))[0]+scipy.integrate.quad(self.integrand_cross,self.z_min,self.z_max,args=(cosmo,self.bins_cross[ll+1]))[0]+scipy.integrate.quad(self.integrand_cross,self.z_min,self.z_max,args=(cosmo,self.l_binned_cross[ll]))[0])/3 for ll in range(self.n_bins_cross)])
        cl_binned_auto_theory=np.array([scipy.integrate.quad(self.integrand_auto,self.z_min,self.z_max,args=(cosmo,ll),epsrel=1e-8)[0] for ll in self.l_binned_auto])

        # Calculate the chi-square of auto- and crosscorrelation
        chi2_cross=np.asscalar(np.dot(self.cl_binned_cross-A*b*cl_binned_cross_theory,np.dot(np.linalg.inv(self.cov_binned_cross),self.cl_binned_cross-A*b*cl_binned_cross_theory)))
        chi2_auto=np.asscalar(np.dot(self.cl_binned_auto-b**2*cl_binned_auto_theory,np.dot(np.linalg.inv(self.cov_binned_auto),self.cl_binned_auto-b**2*cl_binned_auto_theory)))
        return -0.5*(chi2_cross+chi2_auto)

###################################
# SPTPol LIKELIHOOD
# TE,EE spectra with window functions
# Henning et al., 2017
###################################
class Likelihood_sptpol_500d_TEEE(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        self.need_cosmo_arguments(data, {'lensing': 'yes', 'output': 'tCl lCl pCl'})

        desc_file = np.loadtxt(os.path.join(self.data_directory, self.sptpol_TEEE_desc_file))
        self.nbin=int(desc_file[0][0])
        nfreq=int(desc_file[0][1])
        spt_windows_lmin=int(desc_file[1][0])
        spt_windows_lmax=int(desc_file[1][1])
        self.spt_lmin = spt_windows_lmin
        self.spt_lrange=spt_windows_lmax-spt_windows_lmin+1
        print("nbin=",self.nbin,nfreq,spt_windows_lmin,spt_windows_lmax)

        if (nfreq != 1):
            raise io_mp.LikelihoodError("Sorry, current code wont work for multiple freqs")
        #Should be three for SPTpol (TT,TE,EE, although mostly ignore TT).
        self.bands_per_freq = 3 

        nband = self.bands_per_freq*(nfreq)
        #Cov doesn't know about TT. Number of raws.
        self.nall = self.nbin*(nfreq * (self.bands_per_freq - 1))

        self.need_cosmo_arguments(data, {'l_max_scalars': spt_windows_lmax+1})
        if (spt_windows_lmin < 2 or spt_windows_lmin >= spt_windows_lmax):
            raise io_mp.LikelihoodError("Invalid lranges for sptpol")

        self.ells = np.zeros(self.spt_lrange+2, 'float64')
        self.cl_to_dl_conversion = np.zeros(self.spt_lrange+2, 'float64')
        self.rawspec_factor = np.zeros(self.spt_lrange+2, 'float64')
        self.deriv_factor = np.zeros(self.spt_lrange, 'float64')
        for i in range(self.spt_lrange+2):
            self.ells[i] = spt_windows_lmin-1+i
            self.cl_to_dl_conversion[i] = self.ells[i]*(self.ells[i] + 1.)/2./np.pi
            self.rawspec_factor[i] = self.ells[i]**2/self.cl_to_dl_conversion[i]
        for i in range(self.spt_lrange):
            self.deriv_factor[i] = 0.5/self.ells[i+1]

        #Read in bandpowers
        #Should be TE, EE, TT, in that order from SPTpol analysis.
        self.spec = np.zeros((self.nbin, nband), 'float64')
        datafile = open(os.path.join(self.data_directory, self.sptpol_TEEE_bp_file), 'r')
        for i in range(nband):
            for j in range(self.nbin):
                line = datafile.readline()
                self.spec[j][i] = float(line.split()[1])
        datafile.close()

        #Read in covariance
        #Should be TE, EE
        self.cov = np.zeros((self.nall, self.nall), 'float64')
        datafile = open(os.path.join(self.data_directory, self.sptpol_TEEE_cov_file), 'r')
        for i in range(self.nall):
            for j in range(self.nall):
                line = datafile.readline()
                self.cov[j][i] = float(line.split()[0])
        datafile.close()

        if (self.sptpol_EEonly or self.sptpol_TEonly):
            print("Zero off-diagonal cov blocks...")
            for i in range(nbin):
                for j in range(nbin):
                    self.cov[j + nbin][i] = 0.
                    self.cov[j][i + nbin] = 0.
            #Explode TE auto-block if we only want EE.
            if self.sptpol_EEonly:
                print("Exploding TE auto-cov block...")
                for i in range(self.nbin):
                    tmp=self.cov[i][i]*1.e24
                    for j in range(self.nbin):
                        self.cov[j][i] = 0.
                    self.cov[i][i] = tmp
            #Explode TT auto-block if we only want EE.
            if self.sptpol_TEonly:
                print("Exploding EE auto-cov block...")
                for i in range(self.nbin,2*self.nbin):
                    tmp=self.cov[i][i]*1.e24
                    for j in range(self.nbin,2*self.nbin):
                        self.cov[j][i] = 0.
                    self.cov[i][i] = tmp

        print("First entry of covariance matrix: ",self.cov[0][0])

        #Read in windows
        #Should be TE, EE
        self.windows = np.zeros((self.spt_lrange, self.nall), 'float64')
        for i in range(self.nall):
            i_s=repr(i+1)
            datafile = open(os.path.join(self.data_directory,self.sptpol_TEEE_window_dir,'window_' + i_s), 'r')
            for j in range(self.spt_lrange):
                line = datafile.readline()
                self.windows[j][i] = float(line.split()[1])
            datafile.close()

        #get beam error term
        neff = self.nall
        n_beam_terms = 2
        self.beam_err = np.zeros((neff, n_beam_terms), 'float64')
        datafile = open(os.path.join(self.data_directory,self.sptpol_TEEE_beam_file), 'r')
        for i in range(n_beam_terms):
            for j in range(neff):
                line = datafile.readline()
                self.beam_err[j][i] = float(line.split()[1])
        datafile.close()

        print("Successfully initialized SPTPOL_TEEE data...")     
        # end of initialisation

        return

    def loglkl(self, cosmo, data):

        #DataParams for SPTpol likelihood are: [kappa, CzeroTE, Czero_EE, Adust_TE, alpha_TE, Adust_EE, alpha_EE,MapTcal, MapPcal, BeamFac]

        dls = np.zeros((self.spt_lrange+2,2),'float64')
        d3000 = 3000.*3001./2./np.pi
        beta = 0.0012309
        dipole_cosine = -0.4033
        deltacb = np.zeros(self.nall,'float64')
        tmp2cb = np.zeros(self.nall,'float64')
        BeamFac = np.zeros(self.nall,'float64')
        tmpcb = np.zeros(self.nbin,'float64')
        PoissonLevels = np.zeros(2,'float64')
        ADust = np.zeros(2,'float64')
        alphaDust = np.zeros(2,'float64')
        CalFactors = np.zeros(3,'float64')
        arr = np.zeros(2,'float64')
        arr7 = np.zeros(7,'float64')
        BeamFactors = np.zeros(2,'float64')

        dl_fgs = np.zeros(self.spt_lrange,'float64')
        cl_derivative = np.zeros((self.spt_lrange,2),'float64')
        aberration = np.zeros((self.spt_lrange,2),'float64')
        raw_spectra = np.zeros((self.spt_lrange+2,2),'float64')

        cl = self.get_cl(cosmo)
        for i in range(self.spt_lrange+2):
            dls[i,0] = cl['te'][int(self.ells[i])]*self.ells[i]*(self.ells[i]+1)/2./np.pi
            dls[i,1] = cl['ee'][int(self.ells[i])]*self.ells[i]*(self.ells[i]+1)/2./np.pi

        #Calculate derivatives for this position in parameter space.
        for i in range(self.spt_lrange+2):
            raw_spectra[i][0] = self.rawspec_factor[i] * dls[i,0]
            raw_spectra[i][1] = self.rawspec_factor[i] * dls[i,1]
        for i in range(self.spt_lrange):
            cl_derivative[i][0] = self.deriv_factor[i] * (raw_spectra[i+2][0] - raw_spectra[i][0])
            cl_derivative[i][1] = self.deriv_factor[i] * (raw_spectra[i+2][1] - raw_spectra[i][1])

        #Also get derives of the Dls for use with aberration corrections.
        if self.correct_aberration:
            for i in range(self.spt_lrange):
                aberration[i][0] = 0.5 * (dls[i+2][0]-dls[i][0])
                aberration[i][0] = (-1*beta*dipole_cosine) * self.ells[i+1] * aberration[i][0]
                aberration[i][1] = 0.5 * (dls[i+2][1]-dls[i][1])
                aberration[i][1] = (-1*beta*dipole_cosine) * self.ells[i+1] * aberration[i][1]

        #DataParams for SPTpol likelihood are: [MapTcal, MapPcal, Czero_EE, CzeroTE, kappa, Adust_TE, alpha_TE, Adust_EE, alpha_EE]
        czero_psTE_150 = data.mcmc_parameters['czero_psTE_150']['current']*data.mcmc_parameters['czero_psTE_150']['scale']
        czero_psEE_150 = data.mcmc_parameters['czero_psEE_150']['current']*data.mcmc_parameters['czero_psEE_150']['scale']
        ADust_TE = data.mcmc_parameters['ADust_TE']['current']*data.mcmc_parameters['ADust_TE']['scale']
        ADust_EE = data.mcmc_parameters['ADust_EE']['current']*data.mcmc_parameters['ADust_EE']['scale']
        alphaDust_TE = data.mcmc_parameters['alphaDust_TE']['current']*data.mcmc_parameters['alphaDust_TE']['scale']
        alphaDust_EE = data.mcmc_parameters['alphaDust_EE']['current']*data.mcmc_parameters['alphaDust_EE']['scale']
        mapTcal = data.mcmc_parameters['mapTcal']['current']*data.mcmc_parameters['mapTcal']['scale']
        mapPcal = data.mcmc_parameters['mapPcal']['current']*data.mcmc_parameters['mapPcal']['scale']
        beam1 = data.mcmc_parameters['beam1']['current']*data.mcmc_parameters['beam1']['scale']
        beam2 = data.mcmc_parameters['beam2']['current']*data.mcmc_parameters['beam2']['scale']
        Kappa = data.mcmc_parameters['kappa']['current']*data.mcmc_parameters['kappa']['scale']
        #TE/EE Poisson
        PoissonLevels[0] = czero_psTE_150/d3000
        PoissonLevels[1] = czero_psEE_150/d3000
        #TE dust amplitude, in Dl, NOT Cl yet...
        ADust[0] = ADust_TE
        #TE dust spectral index (for Dl).
        alphaDust[0] = alphaDust_TE
        #EE dust amplitude, in Dl, NOT Cl yet...
        ADust[1] = ADust_EE
        #EE dust spectral index (for Dl).
        alphaDust[1] = alphaDust_EE
        #TT, TE, EE
        for i in range(3):
            CalFactors[i] = (mapTcal)**2*(mapPcal)**i
        BeamFactors[0] = beam1
        BeamFactors[1] = beam2

        #Don't care about TT. Just use it for leakage, which is turned off by default.
        for k in range(self.bands_per_freq-1):
            #First get model foreground spectrum (in Cl).
            #Note all the foregrounds are recorded in Dl at l=3000, so we divide by d3000 to get to a normalized Cl spectrum.

            #Start with Poisson power and subtract the kappa parameter for super sample lensing.
            for j in range(self.spt_lrange):
                dl_fgs[j] = (PoissonLevels[k] - Kappa*cl_derivative[j][k]) * self.cl_to_dl_conversion[j+1]
                #Now add model CMB.
                dl_fgs[j] = dl_fgs[j] + dls[j+1][k]
                #Do we want to correct for aberration?
                dl_fgs[j] = dl_fgs[j] + aberration[j][k]
                #add dust foreground model (defined in Dl)
                dl_fgs[j] = dl_fgs[j] + ADust[k]*(self.ells[j+1]/80.)**(alphaDust[k]+2.)

            #Now bin into bandpowers with the window functions.
            tmpcb = np.dot(dl_fgs,self.windows[:,self.nbin*k:self.nbin+self.nbin*k])
            #scale theory spectrum by calibration:
            tmpcb = tmpcb/CalFactors[k+1]
            tmp2cb[self.nbin*k:self.nbin+self.nbin*k] = tmpcb[:]

        BeamFac[:] = 1.
        for i in range(2):
            for j in range(self.nall):
                BeamFac[j] = BeamFac[j] * (1. + self.beam_err[j][i] * BeamFactors[i])

        deltacb = tmp2cb[:] * BeamFac[:]
        for k in range(self.bands_per_freq-1):
            deltacb[self.nbin*k:self.nbin+self.nbin*k] = deltacb[self.nbin*k:self.nbin+self.nbin*k] - self.spec[:,k]

        chi2 = 0.
        chi2 = np.dot(deltacb, np.dot(np.linalg.inv(self.cov), deltacb))
        #print("SPTpolEELnLike=", 0.5 * chi2)

        #beam prior
        chi2 += sum(BeamFactors**2)
        #print("PriorLnLike=", 0.5 * sum(BeamFactors**2))

        #Add Gaussian prior for temperature calibration.
        if self.sptpol_tcal_prior:
            chi2 += (np.log(mapTcal/self.sptpol_meanTcal)/np.log(1.+self.sptpol_sigmaTcal))**2
            #print("PriorLnLike=", 0.5 * (np.log(mapTcal/self.sptpol_meanTcal)/np.log(1.+self.sptpol_sigmaTcal))**2)

        #Add Gaussian prior for polarization efficiency.
        if self.sptpol_pcal_prior:
            chi2 += (np.log(mapPcal/self.sptpol_meanPcal)/np.log(1.+self.sptpol_sigmaPcal))**2
            #print("PriorLnLike=", 0.5 * (np.log(mapPcal/self.sptpol_meanPcal)/np.log(1.+self.sptpol_sigmaPcal))**2)

        #Add Gaussian prior for kappa.
        if self.sptpol_kappa_prior:
            chi2 += ((Kappa - self.sptpol_meankappa)/self.sptpol_sigmakappa)**2
            #print("PriorLnLike=", 0.5 * ((Kappa - self.sptpol_meankappa)/self.sptpol_sigmakappa)**2)

        #Add Gaussian prior for alphaTE.
        if self.sptpol_alphaTE_prior:
            chi2 += ((alphaDust_TE - self.sptpol_meanAlphaTE)/self.sptpol_sigmaAlphaTE)**2
            #print("PriorLnLike=", 0.5 * ((alphaDust_TE - self.sptpol_meanAlphaTE)/self.sptpol_sigmaAlphaTE)**2)

        #Add Gaussian prior for alphaEE.
        if self.sptpol_alphaEE_prior:
            chi2 += ((alphaDust_EE - self.sptpol_meanAlphaEE)/self.sptpol_sigmaAlphaEE)**2
            #print("PriorLnLike=", 0.5 * ((alphaDust_EE - self.sptpol_meanAlphaEE)/self.sptpol_sigmaAlphaEE)**2)

        lkl = -0.5 * chi2

        return lkl

#######################MODIFICATIONS FROM HERE by RM 2021#####################

###################################
# SPTPol lensing
# Lensing power spectra with window functions
# Bianchini et al., 2019
###################################
class Likelihood_sptpol_500d_lens(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        self.need_cosmo_arguments(data, {'lensing': 'yes', 'output': 'tCl lCl pCl'})
        self.need_cosmo_arguments(data, {'l_max_scalars': self.cl_lmax})
        self.need_cosmo_arguments(data, {'non linear': 'halofit'})

        self.cl_lrange=self.cl_lmax-self.cl_lmin+1
        self.ells = np.zeros(self.cl_lrange, 'float64')
        for i in range(self.cl_lrange):
            self.ells[i] = self.cl_lmin+i

        #Read bandpowers
        self.ClHat = np.zeros(self.nbins, 'float64')
        datafile = open(os.path.join(self.data_directory, self.cl_hat_file), 'r')
        for i in range(self.nbins):
            line = datafile.readline()
            while line.find('#') != -1:
                line = datafile.readline()
            self.ClHat[i] = float(line.split()[4])
        datafile.close()

        #Read fiducial corrections
        self.FiducialCorrection = np.zeros(self.nbins, 'float64')
        datafile = open(os.path.join(self.data_directory, self.linear_correction_fiducial_file), 'r')
        for i in range(self.nbins):
            line = datafile.readline()
            while line.find('#') != -1:
                line = datafile.readline()
            self.FiducialCorrection[i] = float(line.split()[1])
        datafile.close()

        #Read covariance
        self.cov = np.zeros((self.nbins, self.nbins), 'float64')
        self.cov = np.loadtxt(os.path.join(self.data_directory, self.covmat_fiducial))

        #Read windows
        self.windows = np.zeros((self.cl_lrange, self.nbins), 'float64')
        #Cpp 100<=l<=2000
        delta = 98
        for i in range(self.nbins):
            i_s=repr(i+1)
            datafile = open(os.path.join(self.data_directory,self.bin_window_dir,'window' + i_s + '.dat'), 'r')
            line = datafile.readlines()
            lmin = int(line[0].split()[0])
            lmax = int(line[-1].split()[0])
            for j in range(0,lmax-lmin+1):
                self.windows[j+delta][i] = float(line[j].split()[1])
            delta += lmax-lmin+1
            datafile.close()

        #Read windows correction
        self.windows_corr = np.zeros((self.cl_lrange, 4, self.nbins), 'float64')
        #Ctt,Cee,Cte,Cpp 0<=l<=2001
        for i in range(self.nbins):
            i_s=repr(i+1)
            datafile = open(os.path.join(self.data_directory,self.linear_correction_bin_window_dir,'window' + i_s + '.dat'), 'r')
            line = datafile.readlines()
            lmin = int(line[0].split()[0])
            lmax = int(line[-1].split()[0])
            delta = self.cl_lmin - lmin
            for k in range(0,4):
                for j in range(0,self.cl_lrange):
                    self.windows_corr[j][k][i] = float(line[j+delta].split()[k+1])
            datafile.close()

        return

    def loglkl(self, cosmo, data):

        clth = np.zeros((self.cl_lrange,4),'float64')
        clthb = np.zeros(self.nbins,'float64')
        deltab = np.zeros(self.nbins,'float64')

        cl = self.get_cl(cosmo)
        for i in range(self.cl_lrange):
            clth[i,0] = cl['tt'][int(self.ells[i])]*self.ells[i]*(self.ells[i]+1)/2./np.pi
            clth[i,1] = cl['ee'][int(self.ells[i])]*self.ells[i]*(self.ells[i]+1)/2./np.pi
            clth[i,2] = cl['te'][int(self.ells[i])]*self.ells[i]*(self.ells[i]+1)/2./np.pi
            clth[i,3] = cl['pp'][int(self.ells[i])]*self.ells[i]**2*(self.ells[i]+1)**2/2./np.pi

        #Convert to bandpowers
        clthb = np.dot(clth[:,3],self.windows)

        #Windows correction
        for k in range(4):
            clthb += np.dot(clth[:,k],self.windows_corr[:,k,0:self.nbins])

        #Fiducial correction
        clthb += - self.FiducialCorrection

        Aphiphi = data.mcmc_parameters['Aphiphi']['current']*data.mcmc_parameters['Aphiphi']['scale']
        deltab = Aphiphi*clthb - self.ClHat

        chi2 = np.dot(deltab, np.dot(np.linalg.inv(self.cov), deltab))

        lkl = -0.5 * chi2

        return lkl
