import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd

# Script for analysing TGLF simulations to determine the dominant
# turbulent mode type. Reading functions adapted from previous scripts
# by G. Szepesi.

# H.G. Dudding, 07/03/2024


''' Functions to read TGLF output files '''

# ---------------------------------------------------------------------
def convert_string_list_to_float_array(strdat, ncol):
    # Convert list of strings into numpy array of floats:
    data = np.array(strdat, dtype='|S30')
    data = data.astype(float)

    # Total number of elements in the data array:
    ntot = np.size(data)

    # Reshape data array:
    data = np.reshape(data, (int(ntot/ncol), ncol))

    return data

# Returns [NKY, 2 * nmodes] dimensional array, and so to access eigen
# spectra, take "data[:,j]", where j=0 is dominant mode growth rate,
# j=1 dominant mode frequency, j=2 is second mode growth rate, and so
# on. 
# ---------------------------------------------------------------------
def read_eigenvalue(tglf_directory, NMODES):
    # The out.tglf.eigenvalue_spectrul file is organized as follows:
    #   - 2 lines of header,
    #   - (gamma(n),freq(n),n=1,nmodes_in),
    
    filename = tglf_directory + 'out.tglf.eigenvalue_spectrum'

    # Read file by line into a list of strings:
    strdat = []
    with open(filename, 'r') as infile:
        count = 0
        for line in infile:
            count = count + 1
            if count<3:
                continue
            strdat = strdat + line.split()

    # Number of columns in the reshaped data array:
    ncol = 2 * NMODES # (gr, freq) for each mode
    # Convert strings into float array and reshape:
    data = convert_string_list_to_float_array(strdat, ncol)
    
    return data	

# Gives the 5 fluxes: particle, energy, toroidal stress,
# parallel stress, exchange, so "data[:,j]", for 0<=j<=4. Doesn't care about NMODES.
# ---------------------------------------------------------------------
def get_fluxes(tglf_directory, NKY, species, field, NFIELDS):
	
	filename = tglf_directory + 'out.tglf.sum_flux_spectrum'

	strdat = []
	with open(filename, 'r') as infile:
		count = 0
		for line in infile:
			count = count + 1
			if count< (3 + (NFIELDS * species + field) * (3 + 2 * NKY - 1)):
				continue
			if count> ((NFIELDS * species + field + 1) * (3 + 2 * NKY - 1)):
				continue
			strdat = strdat + line.split()

	# Number of columns in the reshaped data array:
	ncol = 5
	# Convert strings into float array and reshape:
	data = convert_string_list_to_float_array(strdat, ncol)
	return data

# Get numerical parameters that help reading files
# ---------------------------------------------------------------------
def get_tglf_run_info(tglf_directory):
	kys = np.loadtxt(tglf_directory + 'out.tglf.ky_spectrum', skiprows = 2)
	NKY = len(kys)
	# check which fields are turned on
	sig_bpar, sig_apar = 0, 0
	with open(tglf_directory + 'input.tglf.gen') as f:
		lines = f.readlines()
		if lines[19] == '.true.  USE_BPAR\n':
			sig_bpar = 1	
		if lines[18] == '.true.  USE_BPER\n':
			sig_apar = 1
		NFIELDS = 1 + sig_apar + sig_bpar # number of fields
		NMODES = int(lines[24][0]) # number of modes
		NMODES = 2
		NSPECIES = int(lines[45][0]) # number of species 
		# problems can arise if the first species is not the electrons
		if lines[53] not in ['-1.0  ZS_1\n', '-1  ZS_1\n']:
			print('First species should be electrons!')
	return NKY, NFIELDS, NMODES, NSPECIES

# the most complicated one, in part due to there being different
# line skips for modes than fields and species. 
# Like with get_fluxes, "data[:,j]" for 0<=j<=4.
# ---------------------------------------------------------------------
def get_weights(tglf_directory, NKY, species, field, mode, NFIELDS, NMODES):

	filename = tglf_directory + 'out.tglf.QL_flux_spectrum'
	
	strdat = []
	with open(filename, 'r') as infile:
		count = 0
		for line in infile:
			count = count + 1
			start = (7 + 2 * NKY * (NFIELDS * NMODES * species + NMODES * field + mode) + 
					(mode + (NMODES + 1) * field + NFIELDS * (NMODES + 1) * species))
			if count< start:
				continue
			if count> (start + 2 * NKY - 1):
				continue
			strdat = strdat + line.split()
	# Number of columns in the reshaped data array:
	ncol = 5

	# Convert strings into float array and reshape:
	data = convert_string_list_to_float_array(strdat, ncol)
	return data

# 6 opening lines to skip. First column is 'vector',
# then phi, A_par, B_par. To access, data[:,j], 0<=j<=3.
# ---------------------------------------------------------------------
def get_field_spec(tglf_directory, NKY, NMODES, mode):
	
	filename = tglf_directory + 'out.tglf.field_spectrum'
	
	strdat = []
	with open(filename, 'r') as infile:
		count = 0
		for line in infile:
			count = count + 1
			if count<7:
				continue
			strdat = strdat + line.split()

	# Number of columns in the reshaped data array:
	ncol = 4

	# Convert strings into float array and reshape:
	data = convert_string_list_to_float_array(strdat, ncol)
	data = data[mode: mode + NKY * NMODES :NMODES]
	return data



# Get index to separate into ETG and ion-scale parts
# ---------------------------------------------------------------------
def get_scale_sep_index(direc, kys):
	# get rho_i
	with open(direc + 'out.tglf.scalar_saturation_parameters') as f:
		lines = f.readlines()
		#df = pd.read_csv(direc + 'out.tglf.scalar_saturation_parameters',skiprows=1, sep='=')
		#df = df.T
		#df = df.rename(columns=lambda x: x.strip())
		#rho_ion = df['rho_ion'].values.astype(float)
		rho_ion = float(lines[17][13:].strip('\n').strip(' ').strip('='))
		print(rho_ion)
	ky_e = 2.0 / rho_ion
	# find index that separates scales
	j = int(np.max(np.where(kys<=ky_e)))+1

	j=0
	while kys[j] <= ky_e:
		j = j + 1
	
	scale_sep_index = j
	return scale_sep_index

# Calculate the flux spectrum for a given mode from the TGLF outputs
# ---------------------------------------------------------------------
def flux_spec_per_mode(direc, kys, mode, species):
	NKY, NFIELDS, NMODES, NSPECIES = get_tglf_run_info(direc)
	field_data = get_field_spec(direc, NKY, NMODES, mode)[:,1]
	all_fluxes = []
	for j in range(5):
		QL_weight_data = get_weights(direc, NKY, species, 0, mode, NFIELDS, NMODES)[:,j]
		pre_integral_flux = field_data * QL_weight_data
		dky0, ky0 = 0.0, 0.0 # initialise integration quantities
		flux0, flux1, flux_out = 0.0, 0.0, 0.0
		fluxes = []
		for i in range(NKY):
			ky1 = kys[i]
			if (i==0):
				dky1=ky1
			else:
				dky = np.log(ky1/ky0)/(ky1-ky0)
				dky1 = ky1*(1.0 - ky0*dky)
				dky0 = ky0*(ky1*dky - 1.0)
			flux1 = pre_integral_flux[i]
			flux_out = dky0*flux0 + dky1*flux1
			fluxes.append(flux_out)
			flux0 = flux1
			ky0 = ky1
		all_fluxes.append(fluxes)
	return np.array(all_fluxes)

# Function to test whether the calculation of the fluxes from the weight
# and field data matches the flux data.
# ---------------------------------------------------------------------	
def test_function(direc):
	NKY, NFIELDS, NMODES, NSPECIES = get_tglf_run_info(direc)
	kys = np.loadtxt(direc + 'out.tglf.ky_spectrum', skiprows = 2)
	for k in range(5):
		for s in range(NSPECIES):
			flux_data = sum(get_fluxes(direc, NKY, s, 0, NFIELDS)[:,k])
			flux_calc = 0.0
			for m in range(NMODES):
				flux_per_mode = sum(flux_spec_per_mode(direc, kys, m, s)[k])
				flux_calc = flux_calc + flux_per_mode
			print('species ' + str(s) + ' k = ' + str(k) + ' flux data = ' + str(flux_data) + 
				  ', flux calc = ' +str(flux_calc))


''' Main functions '''

# Main function of the script, that takes a TGLF directory ('direc')
# and returns an array with information on the weights.
# ---------------------------------------------------------------------
def get_TGLF_weight_info(direc):
	NKY, NFIELDS, NMODES, NSPECIES = get_tglf_run_info(direc)
	kys = np.loadtxt(direc + 'out.tglf.ky_spectrum', skiprows = 2)
	if NFIELDS != 1:
		print('This script only deals with NFIELDS = 1.')
	
	all_weight_info = []
	for m in range(2): # for the 2 modes
		# get array of flux spectra for each species
		species_weights = []
		for species in range(NSPECIES):
			type_weights = [] # 'types' being particle, energy, etc
			for j in range(5):
				QL_weight_data = get_weights(direc, NKY, species, 0, m, NFIELDS, NMODES)[:,j]
				# '0' here is the field phi, so electrostatic
				type_weights.append(QL_weight_data)
			species_weights.append(type_weights)
		species_weights = np.array(species_weights)
		all_weight_info.append(species_weights)
	all_weight_info = np.array(all_weight_info)
	
	# indices:
	# m = 0: electron frequency direction mode
	# m = 1: ion frequency direction mode
	# to get weight spectra for mode m, species s,
	# for flux type k, : all_weight_info[m, s, k]
	
	# flux types: 0 particle
	#			  1 energy
	#			  2 toroidal stress
	#			  3 parallel stress
	#			  4 exchange
	
	return all_weight_info
	
### Get eigenvalue spectra for a given directory. Slicing explained below.
def get_growth_rates_and_frequencies(direc):
	NKY, NFIELDS, NMODES, NSPECIES = get_tglf_run_info(direc)
	kys = np.loadtxt(direc + 'out.tglf.ky_spectrum', skiprows = 2)
	if NFIELDS != 1:
		print('This script only deals with NFIELDS = 1.')
	
	if NMODES != 2:
		print('This script only deals with NMODES = 2.')
	
	eigenvalue_data = read_eigenvalue(direc, NMODES)
	# slice via
	# eigenvalue_data[:, g]
	# where g = 0: electron frequency direction mode growth rate
	#         = 1: electron frequency direction mode frequency
	#         = 2: ion mode frequency direction growth rate
	#         = 3: ion mode frequency direction frequency
	# The ':' is because it's a spectrum of length len(kys).
	return kys, eigenvalue_data
	

''' Two main things we're interested in analysing are 'all_weight_info'
from 'get_TGLF_weight_info' and 'eigenvalue_data' from 'get_growth_rates_and_frequencies'.

The labelling of 'electron frequency mode direction' and 'ion frequency mode direction'
refers to the sign of the real frequency of the instability (whether its positive
or negative, to help with mode identification), and shouldn't be confused
with having species dependence. In TGLF, electron direction is taken to be positive
and ion direction is taken to be negative. One can loosely think of 'ion direction' to be ITG
and 'electron direction' to be TEM (for low ky). The rest of the labelling is explained
at the end of the two functions above. '''


for i in [7]:
	tglf_directory = '/home/hdudding/work/PostDoc/DTP/tglfNN_project/tglf_standalone/GA_std_dgscan/' + str(i) + '/'
	kys, eigen_dat = get_growth_rates_and_frequencies(tglf_directory)
	
	test_function(tglf_directory)
	
	# Growth rate plot
	plt.plot(kys, eigen_dat[:, 0], label = 'Elec. direc. ' + r'$\gamma_{k_y}$') # electron direction mode growth rate.
																			   # should dominate at high ky!
	plt.plot(kys, eigen_dat[:, 2], label = 'Ion direc. ' + r'$\gamma_{k_y}$') # ion direction mode growth rate
	plt.xlabel(r'$k_y \rho_{\mathrm{unit}}$')
	plt.ylabel(r'$\gamma_{k_y} \left(a / c_s\right)$')
	plt.title('Growth rates')
	plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.grid()
	plt.show()
	
	# Frequency plot
	plt.plot(kys, eigen_dat[:, 1], label = 'Elec. direc. ' + r'$\omega_{k_y}$') # electron direction mode frequency,
																			    # should be +ve
	plt.plot(kys, eigen_dat[:, 3], label = 'Ion direc. ' + r'$\omega_{k_y}$') # ion direction mode frequency, should be -ve
	plt.xlabel(r'$k_y \rho_{\mathrm{unit}}$')
	plt.ylabel(r'$\omega_{k_y} \left(a / c_s\right)$')
	plt.title('Frequencies')
	plt.xscale('log')
	plt.legend()
	plt.grid()
	plt.show()
	
	
	# Weight plots
	flux_types = ['Particle', 'Energy', 'Toroidal stress', 
					'Parallel stress', 'Exchange']
	species_labels = ['Electron', 'Ion']
	mode_labels = ['Elec. direc', 'Ion direc.']
	 
	all_weight_info = get_TGLF_weight_info(tglf_directory)
	NKY, NFIELDS, NMODES, NSPECIES = get_tglf_run_info(tglf_directory)
	
	flux_type = 1 # energy, in this example
	for mod in range(2): # mod = 0: electron direction mode, mod = 1: ion direction mode
		for spec in range(2): # spec = 0: electron species, spec = 1: ion species
			plt.plot(kys, all_weight_info[mod, spec, flux_type], label = str(mode_labels[mod]) + ' mode, ' + str(species_labels[spec]) + ' species '
						+ str(flux_types[flux_type]) + ' weights')
		plt.title(str(flux_types[flux_type]) + ' weights')
		plt.xlabel(r'$k_y \rho_{\mathrm{unit}}$')
		plt.ylabel(r'$W_{k_y} / W_{\mathrm{ref}}$')
		plt.xscale('log')
		plt.grid()
		plt.legend()
		plt.show()

