# This is to insure py2.7 compatibility
from __future__ import print_function
from __future__ import division

from montepython.likelihood_class import Likelihood
import io_mp
import re  # Module to handle regular expressions
import sys
import os
import numpy as np
import pickle
from scipy.linalg import block_diag

class Lya_joint(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        print("Initializing Lya likelihood")

        self.need_cosmo_arguments(data, {'output': 'mPk'})
        self.need_cosmo_arguments(data, {'P_k_max_h/Mpc': 1.5*self.kmax})

        # Derived_lkl is a new type of derived parameter calculated in the likelihood, and not known to class.
        # This first initialising avoids problems in the case of an error in the first point of the MCMC
        data.derived_lkl = {'lya_neff':0, 'area':0}

        self.bin_file_path = os.path.join(command_line.folder,self.bin_file_name)
        if not os.path.exists(self.bin_file_path):
            with open(self.bin_file_path, 'w') as bin_file:
                bin_file.write('#')
                for name in data.get_mcmc_parameters(['varying']):
                    name = re.sub('[$*&]', '', name)
                    bin_file.write(' %s\t' % name)
                for name in data.get_mcmc_parameters(['derived']):
                    name = re.sub('[$*&]', '', name)
                    bin_file.write(' %s\t' % name)
                for name in data.get_mcmc_parameters(['derived_lkl']):
                    name = re.sub('[$*&]', '', name)
                    bin_file.write(' %s\t' % name)
                bin_file.write('\n')
                bin_file.close()

        if 'z_reio' not in data.get_mcmc_parameters(['derived']) or 'sigma8' not in data.get_mcmc_parameters(['derived']):
            raise io_mp.ConfigurationError('Error: Lya likelihood need z_reio and sigma8 as derived parameters')

        # Redshift independent parameters - params order: z_reio, sigma_8, n_eff, f_UV, area
        # area is the parameter associated to the extra power wrt to LCDM

        self.zind_param_size = [3, 5, 5, 3, 6] # How many values we have for each param
        self.zind_param_min = np.array([7., 0.754, -2.3474, 0., 0.])   
        self.zind_param_max = np.array([15., 0.904, -2.2674, 1., 1.])  
        zind_param_ref = np.array([9., 0.829, -2.3074, 0., 0.])   
        self.zreio_range = self.zind_param_max[0]-self.zind_param_min[0]
        self.s8_range = self.zind_param_max[1]-self.zind_param_min[1]
        self.neff_range = self.zind_param_max[2]-self.zind_param_min[2]
        self.fuv_range = self.zind_param_max[3]-self.zind_param_min[3]
        self.area_range = self.zind_param_max[4]-self.zind_param_min[4]

        # Redshift dependent parameters - params order: params order: mean_f, t0, slope
        zdep_params_size = [9, 3, 3]  # How many values we have for each param
        zdep_params_refpos = [4, 1, 2]  # Where to store the P_F(ref) DATA

        # Mean flux values
        F_ref = [0.669181, 0.617042, 0.564612, 0.512514, 0.461362, 0.411733, 0.364155, 0.253828, 0.146033, 0.0712724]
        
        # Manage the data sets
        # FIRST (NOT USED) DATASET (19 wavenumbers) ***XQ-100***
        self.zeta_range_XQ = [3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2]  # List of redshifts corresponding to the 19 wavenumbers (k)
        self.k_XQ = [0.003,0.006,0.009,0.012,0.015,0.018,0.021,0.024,0.027,0.03,0.033,0.036,0.039,0.042,0.045,0.048,0.051,0.054,0.057]

        # SECOND DATASET (7 wavenumbers) ***HIRES/MIKE***
        self.zeta_range_mh = [4.2, 4.6, 5.0, 5.4]  # List of redshifts corresponding to the 7 wavenumbers (k)
        self.k_mh = [0.00501187,0.00794328,0.0125893,0.0199526,0.0316228,0.0501187,0.0794328] # Note that k is in s/km

        self.zeta_full_length = (len(self.zeta_range_XQ) + len(self.zeta_range_mh))
        self.kappa_full_length = (len(self.k_XQ) + len(self.k_mh))

        # Which snapshots we use (first 7 for first dataset, last 4 for second one)
        self.redshift = [3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.2, 4.6, 5.0, 5.4]

        #T 0 and slope values
        T_ref = [11251.5, 11293.6, 11229.0, 10944.6, 10421.8, 9934.49, 9227.31, 8270.68, 7890.68, 7959.4]
        g_ref = [1.53919, 1.52894, 1.51756, 1.50382, 1.48922, 1.47706, 1.46909, 1.48025, 1.50814, 1.52578]

        T_values = np.zeros(( 10, zdep_params_size[1] ),'float64')
        T_values[:,0] = [7522.4, 7512.0, 7428.1, 7193.32, 6815.25, 6480.96, 6029.94, 5501.17, 5343.59, 5423.34]
        T_values[:,1] = T_ref[:]
        T_values[:,2] = [14990.1, 15089.6, 15063.4, 14759.3, 14136.3, 13526.2, 12581.2, 11164.9, 10479.4, 10462.6]

        g_values = np.zeros(( 10, zdep_params_size[2] ),'float64')
        g_values[:,0] = [0.996715, 0.979594, 0.960804, 0.938975, 0.915208, 0.89345, 0.877893, 0.8884, 0.937664, 0.970259]
        g_values[:,1] = [1.32706, 1.31447, 1.30014, 1.28335, 1.26545, 1.24965, 1.2392, 1.25092, 1.28657, 1.30854]
        g_values[:,2] = g_ref[:]

        if self.astro_spectra_file == "Lya_grid/LCDM_ratio_matrix_reshaped_expanded.pkl":
            self.T_min = T_values[:,0]*0.1
            self.T_max = T_values[:,2]*1.5
            self.g_min = g_values[:,0]*0.8
            self.g_max = g_values[:,2]*1.2
        else:
            self.T_min = T_values[:,0]
            self.T_max = T_values[:,2]
            self.g_min = g_values[:,0]
            self.g_max = g_values[:,2]


        # Import the grid of PFs for Kriging
        file_path = os.path.join(self.data_directory, self.astro_spectra_file)
        if os.path.exists(file_path):
            try:
                pkl = open(file_path, 'rb')
                self.input_full_matrix_interpolated_ASTRO = pickle.load(pkl)
            except UnicodeDecodeError as e:
                pkl = open(file_path, 'rb')
                self.input_full_matrix_interpolated_ASTRO = pickle.load(pkl, encoding='latin1')
            pkl.close()
        else:
            raise io_mp.ConfigurationError('Error: astro spectra file is missing')

        ALL_zdep_params = len(F_ref) + len(T_ref) + len(g_ref)
        grid_length_ASTRO = len(self.input_full_matrix_interpolated_ASTRO[0,0,:])
        astroparams_number_KRIG = len(self.zind_param_size) + ALL_zdep_params


        # Import the ASTRO GRID (ordering of params: z_reio, sigma_8, n_eff, f_UV, mean_f(z), t0(z), slope(z), area)
        file_path = os.path.join(self.data_directory, self.astro_grid_file)
        if os.path.exists(file_path):
            self.X = np.zeros((grid_length_ASTRO,astroparams_number_KRIG), 'float64')
            for param_index in range(astroparams_number_KRIG):
                self.X[:,param_index] = np.genfromtxt(file_path, usecols=[param_index], skip_header=1)
        else:
            raise io_mp.ConfigurationError('Error: astro grid file is missing')


        # Prepare the interpolation in astro-param space
        self.redshift_list = [3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.6, 5.0, 5.4]  # This corresponds to the combined dataset (MIKE/HIRES + XQ-100)
        self.F_prior_min = [0.569851, 0.508675, 0.44921, 0.392273, 0.338578, 0.28871, 0.243108, 0.146675, 0.0676442, 0.0247793]
        self.F_prior_max = [0.785826, 0.748495, 0.709659, 0.669613, 0.628673, 0.587177, 0.545471, 0.439262, 0.315261, 0.204999]


        # Load the data
        if not self.DATASET == "joint":
            raise io_mp.LikelihoodError('Error: for the time being, only the joint dataset is available')

        file_path = os.path.join(self.data_directory, self.MIKE_spectra_file)
        if os.path.exists(file_path):
            try:
                pkl = open(file_path, 'rb')
                y_M_reshaped = pickle.load(pkl)
            except UnicodeDecodeError as e:
                pkl = open(file_path, 'rb')
                y_M_reshaped = pickle.load(pkl, encoding='latin1')
            pkl.close()
        else:
            raise io_mp.ConfigurationError('Error: MIKE spectra file is missing')

        file_path = os.path.join(self.data_directory, self.HIRES_spectra_file)
        if os.path.exists(file_path):
            try:
                pkl = open(file_path, 'rb')
                y_H_reshaped = pickle.load(pkl)
            except UnicodeDecodeError as e:
                pkl = open(file_path, 'rb')
                y_H_reshaped = pickle.load(pkl, encoding='latin1')
            pkl.close()
        else:
            raise io_mp.ConfigurationError('Error: HIRES spectra file is missing')

        file_path = os.path.join(self.data_directory, self.XQ_spectra_file)
        if os.path.exists(file_path):
            try:
                pkl = open(file_path, 'rb')
                y_XQ_reshaped = pickle.load(pkl)
            except UnicodeDecodeError as e:
                pkl = open(file_path, 'rb')
                y_XQ_reshaped = pickle.load(pkl, encoding='latin1')
            pkl.close()
        else:
            raise io_mp.ConfigurationError('Error: XQ-100 spectra file is missing')

        file_path = os.path.join(self.data_directory, self.MIKE_cov_file)
        if os.path.exists(file_path):
            try:
                pkl = open(file_path, 'rb')
                cov_M_inverted = pickle.load(pkl)
            except UnicodeDecodeError as e:
                pkl = open(file_path, 'rb')
                cov_M_inverted = pickle.load(pkl, encoding='latin1')
            pkl.close()
        else:
            raise io_mp.ConfigurationError('Error: MIKE covariance matrix file is missing')

        file_path = os.path.join(self.data_directory, self.HIRES_cov_file)
        if os.path.exists(file_path):
            try:
                pkl = open(file_path, 'rb')
                cov_H_inverted = pickle.load(pkl)
            except UnicodeDecodeError as e:
                pkl = open(file_path, 'rb')
                cov_H_inverted = pickle.load(pkl, encoding='latin1')
            pkl.close()
        else:
            raise io_mp.ConfigurationError('Error: HIRES covariance matrix file is missing')

        file_path = os.path.join(self.data_directory, self.XQ_cov_file)
        if os.path.exists(file_path):
            try:
                pkl = open(file_path, 'rb')
                cov_XQ_inverted = pickle.load(pkl)
            except UnicodeDecodeError as e:
                pkl = open(file_path, 'rb')
                cov_XQ_inverted = pickle.load(pkl, encoding='latin1')
            pkl.close()
        else:
            raise io_mp.ConfigurationError('Error: XQ-100 covariance matrix file is missing')

        file_path = os.path.join(self.data_directory, self.PF_ref_file)
        if os.path.exists(file_path):
            try:
                pkl = open(file_path, 'rb')
                self.PF_ref = pickle.load(pkl)
            except UnicodeDecodeError as e:
                pkl = open(file_path, 'rb')
                self.PF_ref = pickle.load(pkl, encoding='latin1')
            pkl.close()
        else:
            raise io_mp.ConfigurationError('Error: PF_ref file is missing')

        self.cov_MH_inverted = block_diag(cov_H_inverted,cov_M_inverted)
        self.y_MH_reshaped = np.concatenate((y_H_reshaped, y_M_reshaped))

        self.y_XQ_reshaped = np.array(y_XQ_reshaped)
        self.cov_XQ_inverted = cov_XQ_inverted

        print("Initialization of Lya likelihood done")



    # The following functions are used elsewhere in the code

    # Analytical function for the redshift dependence of t0 and slope
    def z_dep_func(self,parA, parS, z):
        return parA*(np.power(((1.+z)/(1.+self.zp)),parS))

    # Functions for the Kriging interpolation
    def ordkrig_distance(self,p1, p2, p3, p4, p5, p6, p7, p8, v1, v2, v3, v4, v5, v6, v7, v8):
        return (((p1 - v1)**2 + (p2 - v2)**2 + (p3 - v3)**2 + (p4 - v4)**2 + (p5 - v5)**2 + (p6 - v6)**2 + (p7 - v7)**2 + (p8 - v8)**2)**(0.5) + self.epsilon)**self.exponent

    def ordkrig_norm(self,p1, p2, p3, p4, p5, p6, p7, p8, v1, v2, v3, v4, v5, v6, v7, v8):
        return np.sum(1./self.ordkrig_distance(p1, p2, p3, p4, p5, p6, p7, p8, v1, v2, v3, v4, v5, v6, v7, v8))

    def ordkrig_lambda(self,p1, p2, p3, p4, p5, p6, p7, p8, v1, v2, v3, v4, v5, v6, v7, v8):
        return (1./self.ordkrig_distance(p1, p2, p3, p4, p5, p6, p7, p8, v1, v2, v3, v4, v5, v6, v7, v8))/self.ordkrig_norm(p1, p2, p3, p4, p5, p6, p7, p8, v1, v2, v3, v4, v5, v6, v7, v8)

    
    ## p21 = [z_reio,sigma8,neff,F_UV,Fz1,Fz2,Fz3,Fz4,Fz5,Fz6,Fz7,Fz8,Fz9,Fz10,T0a,T0s,gamma_a,gamma_s,area]
    def ordkrig_estimator(self,p21, z):
    	
        pa10 = []; pb10 = []
        for ii in range(10):    
            pa10.append(self.z_dep_func(p21[-5], p21[-4], z[ii])*1e4/(self.T_max[ii]-self.T_min[ii]))
            pb10.append(self.z_dep_func(p21[-3], p21[-2], z[ii])/(self.g_max[ii]-self.g_min[ii]))
        p37 = np.concatenate((np.array(p21[:14]), np.array(pa10[:]), np.array(pb10[:]), np.array([p21[-1]])))
        astrokrig_result = np.zeros((self.zeta_full_length, self.kappa_full_length), 'float64')
        
        for index in range(len(self.redshift)):
            if index < self.num_z_XQ:
                astrokrig_result[index,:] = np.sum(np.multiply(self.ordkrig_lambda(p37[0]/self.zreio_range, p37[1]/self.s8_range, p37[2]/self.neff_range, p37[3]/self.fuv_range, p37[4+index] \
                                                                               /(self.F_prior_max[index]-self.F_prior_min[index]), \
                                                                               p37[14+index], p37[24+index], p37[-1]/self.area_range, \
                                                                               self.X[:,0]/self.zreio_range, self.X[:,1]/self.s8_range, self.X[:,2]/self.neff_range, self.X[:,3]/self.fuv_range, self.X[:,4+index]/(self.F_prior_max[index]-self.F_prior_min[index]), \
                                                                               self.X[:,14+index]/(self.T_max[index]-self.T_min[index]), self.X[:,24+index]/(self.g_max[index]-self.g_min[index]),self.X[:,34]//self.area_range), \
                                                           self.input_full_matrix_interpolated_ASTRO[index,:,:]),axis=1)
            else:
                astrokrig_result[index,:] = np.sum(np.multiply(self.ordkrig_lambda(p37[0]/self.zreio_range, p37[1]/self.s8_range, p37[2]/self.neff_range, p37[3]/self.fuv_range, p37[4+index] \
                                                                               /(self.F_prior_max[index-self.num_z_overlap]-self.F_prior_min[index-self.num_z_overlap]), \
                                                                               p37[14+index], p37[24+index], p37[-1]/self.area_range, \
                                                                               self.X[:,0]/self.zreio_range, self.X[:,1]/self.s8_range, self.X[:,2]/self.neff_range, self.X[:,3]/self.fuv_range, self.X[:,4+index-self.num_z_overlap]/(self.F_prior_max[index-self.num_z_overlap]-self.F_prior_min[index-self.num_z_overlap]), \
                                                                               self.X[:,14+index-self.num_z_overlap]/(self.T_max[index-self.num_z_overlap]-self.T_min[index-self.num_z_overlap]), self.X[:,24+index-self.num_z_overlap]/(self.g_max[index-self.num_z_overlap]-self.g_min[index-self.num_z_overlap]),self.X[:,34]/self.area_range), \
                                                           self.input_full_matrix_interpolated_ASTRO[index-self.num_z_overlap,:,:]),axis=1)
        return astrokrig_result



    # Start of the actual likelihood computation function
    def loglkl(self, cosmo, data):

        k = np.logspace(np.log10(self.kmin), np.log10(self.kmax), num=self.k_size)

        # Initialise the bin file
        if not os.path.exists(self.bin_file_path):
            with open(self.bin_file_path, 'w') as bin_file:
                bin_file.write('#')
                for name in data.get_mcmc_parameters(['varying']):
                    name = re.sub('[$*&]', '', name)
                    bin_file.write(' %s\t' % name)
                for name in data.get_mcmc_parameters(['derived']):
                    name = re.sub('[$*&]', '', name)
                    bin_file.write(' %s\t' % name)
                for name in data.get_mcmc_parameters(['derived_lkl']):
                    name = re.sub('[$*&]', '', name)
                    bin_file.write(' %s\t' % name)
                bin_file.write('\n')
                bin_file.close()

        # Deal with the astro nuisance parameters
        if 'T0a' in data.mcmc_parameters:
            T0a=data.mcmc_parameters['T0a']['current']*data.mcmc_parameters['T0a']['scale']
        else:
            T0a=0.74
        if 'T0s' in data.mcmc_parameters:
            T0s=data.mcmc_parameters['T0s']['current']*data.mcmc_parameters['T0s']['scale']
        else:
            T0s=-4.38
        if 'gamma_a' in data.mcmc_parameters:
            gamma_a=data.mcmc_parameters['gamma_a']['current']*data.mcmc_parameters['gamma_a']['scale']
        else:
            gamma_a=1.45
        if 'gamma_s' in data.mcmc_parameters:
            gamma_s=data.mcmc_parameters['gamma_s']['current']*data.mcmc_parameters['gamma_s']['scale']
        else:
            gamma_s=-1.93
        if 'Fz1' in data.mcmc_parameters:
            Fz1=data.mcmc_parameters['Fz1']['current']*data.mcmc_parameters['Fz1']['scale']
        else:
            Fz1=F_ref[0]
        if 'Fz2' in data.mcmc_parameters:
            Fz2=data.mcmc_parameters['Fz2']['current']*data.mcmc_parameters['Fz2']['scale']
        else:
            Fz2=F_ref[1]
        if 'Fz3' in data.mcmc_parameters:
            Fz3=data.mcmc_parameters['Fz3']['current']*data.mcmc_parameters['Fz3']['scale']
        else:
            Fz3=F_ref[2]
        if 'Fz4' in data.mcmc_parameters:
            Fz4=data.mcmc_parameters['Fz4']['current']*data.mcmc_parameters['Fz4']['scale']
        else:
            Fz4=F_ref[3]
        if 'Fz5' in data.mcmc_parameters:
            Fz5=data.mcmc_parameters['Fz5']['current']*data.mcmc_parameters['Fz5']['scale']
        else:
            Fz5=F_ref[4]
        if 'Fz6' in data.mcmc_parameters:
            Fz6=data.mcmc_parameters['Fz6']['current']*data.mcmc_parameters['Fz6']['scale']
        else:
            Fz2=F_ref[5]
        if 'Fz7' in data.mcmc_parameters:
            Fz7=data.mcmc_parameters['Fz7']['current']*data.mcmc_parameters['Fz7']['scale']
        else:
            Fz7=F_ref[6]
        if 'Fz8' in data.mcmc_parameters:
            Fz8=data.mcmc_parameters['Fz8']['current']*data.mcmc_parameters['Fz8']['scale']
        else:
            Fz8=F_ref[7]
        if 'Fz9' in data.mcmc_parameters:
            Fz9=data.mcmc_parameters['Fz9']['current']*data.mcmc_parameters['Fz9']['scale']
        else:
            Fz9=F_ref[8]
        if 'Fz10' in data.mcmc_parameters:
            Fz10=data.mcmc_parameters['Fz10']['current']*data.mcmc_parameters['Fz10']['scale']
        else:
            Fz10=F_ref[9]
        if 'F_UV' in data.mcmc_parameters:
            F_UV=data.mcmc_parameters['F_UV']['current']*data.mcmc_parameters['F_UV']['scale']
        else:
            F_UV=0.0

        # Get P(k) from CLASS
        h=cosmo.h()
        Plin = np.zeros(len(k), 'float64')
        for index_k in range(len(k)):
            Plin[index_k] = cosmo.pk_lin(k[index_k]*h, 0.0)
        Plin *= h**3

        # Compute the Lya k scale
        Om=cosmo.Omega_m()
        OL=cosmo.Omega_Lambda()
        k_neff=self.k_s_over_km*100./(1.+self.z)*(((1.+self.z)**3*Om+OL)**(1./2.))

        derived = cosmo.get_current_derived_parameters(data.get_mcmc_parameters(['derived']))
        for (name, value) in derived.items():
            data.mcmc_parameters[name]['current'] = value
        for name in derived:
            data.mcmc_parameters[name]['current'] /= data.mcmc_parameters[name]['scale']

        # Obtain current z_reio, sigma_8, and neff from CLASS
        z_reio=data.mcmc_parameters['z_reio']['current']
        # Check that z_reio is in the correct range
        if z_reio<self.zind_param_min[0]:
            z_reio = self.zind_param_min[0]
        if z_reio>self.zind_param_max[0]:
            z_reio=self.zind_param_max[0]
        sigma8=data.mcmc_parameters['sigma8']['current']
        neff=cosmo.pk_tilt(k_neff*h,self.z)

        # Store neff as a derived_lkl parameter
        data.derived_lkl['lya_neff'] = neff

        ###########################################################################
        # Compute area parameter (associated to extra power wrt to LCDM)
        area = 0.
        data.derived_lkl['area'] = area        
        ###########################################################################

        # First (and only) sanity check, to make sure the cosmological parameters are in the correct range
        if ((sigma8<self.zind_param_min[1] or sigma8>self.zind_param_max[1]) or (neff<self.zind_param_min[2] or neff>self.zind_param_max[2])):
            with open(self.bin_file_path, 'a') as bin_file:
                bin_file.write('#Error_cosmo\t')
                for elem in data.get_mcmc_parameters(['varying']):
                    bin_file.write(' %.6e\t' % data.mcmc_parameters[elem]['current'])
                for elem in data.get_mcmc_parameters(['derived']):
                    bin_file.write(' %.6e\t' % data.mcmc_parameters[elem]['current'])
                for elem in data.get_mcmc_parameters(['derived_lkl']):
                    bin_file.write(' %.6e\t' % data.mcmc_parameters[elem]['current'])
                bin_file.write('\n')
                bin_file.close()
            sys.stderr.write('#Error_cosmo\n')
            sys.stderr.flush()
            return data.boundary_loglike

        # chi2 computation
        chi2_MH=0.
        chi2_XQ=0.

        model_H = np.zeros (( len(self.zeta_range_mh), len(self.k_mh) ), 'float64')
        model_M = np.zeros (( len(self.zeta_range_mh)-1, len(self.k_mh) ), 'float64')
        model_XQ = np.zeros (( len(self.zeta_range_XQ), len(self.k_XQ) ), 'float64')
        
        theta=np.array([z_reio,sigma8,neff,F_UV,Fz1,Fz2,Fz3,Fz4,Fz5,Fz6,Fz7,Fz8,Fz9,Fz10,T0a,T0s,gamma_a,gamma_s,area])
        
        model = self.PF_ref*self.ordkrig_estimator(theta, self.redshift_list)
        upper_block = np.vsplit(model, [7,11])[0]
        lower_block = np.vsplit(model, [7,11])[1]

        model_H[:,:] = lower_block[:,19:]
        model_H_reshaped = np.reshape(model_H, -1, order='C')
        model_M[:,:] = lower_block[:3,19:]
        model_M_reshaped = np.reshape(model_M, -1, order='C')
        model_MH_reshaped = np.concatenate((model_H_reshaped,model_M_reshaped))

        model_XQ[:,:] = upper_block[:,:19]
        model_XQ_reshaped = np.reshape(model_XQ, -1, order='C')

        chi2_MH = np.dot((self.y_MH_reshaped - model_MH_reshaped),np.dot(self.cov_MH_inverted,(self.y_MH_reshaped - model_MH_reshaped)))
        chi2_XQ = np.dot((self.y_XQ_reshaped - model_XQ_reshaped),np.dot(self.cov_XQ_inverted,(self.y_XQ_reshaped - model_XQ_reshaped)))

        loglkl = - 0.5 * (chi2_MH + chi2_XQ)

        return loglkl
