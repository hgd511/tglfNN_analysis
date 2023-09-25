import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd

# Script for analysing TGLF simulations to determine the dominant
# turbulent mode type. Reading functions adapted from previous scripts
# by G. Szepesi.




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
    ncol = 2 * NMODES # (gr, freq) for each eigenmode
    # Convert strings into float array and reshape:
    data = convert_string_list_to_float_array(strdat, ncol)
    
    return data	

# Gives the 5 fluxes: particle, energy, toroidal stress,
# parallel stress, exchange, so "data[:,j]". Doesn't care about NMODES.
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
# line skips for modes than fields and species. "data[:,j]"
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
			if count> (start + 2* NKY - 1):
				continue
			strdat = strdat + line.split()
	# Number of columns in the reshaped data array:
	ncol = 5

	# Convert strings into float array and reshape:
	data = convert_string_list_to_float_array(strdat, ncol)
	return data

# Pretty simple here. 6 opening lines to skip. First column is 'vector',
# then phi, A_par, B_par. But, arranged all n's per ky, and so must be 
# split up, hence extra bit at the end. data_new[:,j] for different columns.
# n is the mode number, 0 <= n <= NMODES - 1.
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
	data = data[mode: mode + (NKY) * NMODES :NMODES]
	return data


''' Functions that test turbulent mode type '''

# Separate into ETG and ion-scale parts
# ---------------------------------------------------------------------
def ES_flux_test(direc, kys, species_fluxes):
	# get rho_i
	with open(direc + 'out.tglf.scalar_saturation_parameters') as f:
		lines = f.readlines()
		df = pd.read_csv(direc + 'out.tglf.scalar_saturation_parameters',skiprows=1, sep='=')
		df = df.T
		df = df.rename(columns=lambda x: x.strip())
		rho_ion = df['rho_ion'].values.astype(float)
		#rho_ion = float(lines[17][13:].strip('\n').strip(' ').strip('='))
	ky_e = 2.0 / rho_ion
	# find index that separates scales
	j = int(np.max(np.where(kys<=ky_e)))+1

	j=0
	while kys[j] <= ky_e:
		j = j + 1
	
	flux_info = []
	for s in range(len(species_fluxes)):
		ion_scales_species_s = []
		electron_scales_species_s = []
		for k in range(5):
			ion_scales_species_s.append(sum(species_fluxes[s,k, :j]))
			electron_scales_species_s.append(sum(species_fluxes[s,k, j:]))
		flux_info.append([ion_scales_species_s, electron_scales_species_s])
	# indices: flux_info[species, scale, flux type]
	# if I want the electron, ion-scale, energy flux: flux_info[0, 0, 1]
	return np.array(flux_info)

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


''' Main '''

# Main function of the script, that takes a TGLF directory ('direc')
# and follows the flowchart to prescribe mode types to each MODE.
# ---------------------------------------------------------------------
def get_TGLF_modes(direc):
	NKY, NFIELDS, NMODES, NSPECIES = get_tglf_run_info(direc)
	kys = np.loadtxt(direc + 'out.tglf.ky_spectrum', skiprows = 2)
	if NFIELDS != 1:
		print('This script only deals with NFIELDS = 1.')
	
	all_flux_info = []
	for m in range(2):
		# get array of flux spectra for each species
		species_fluxes = []
		for species in range(NSPECIES):
			fluxes = flux_spec_per_mode(direc, kys, m, species)
			species_fluxes.append(fluxes)
		species_fluxes = np.array(species_fluxes)
		flux_info = ES_flux_test(direc, kys, species_fluxes)
		all_flux_info.append(flux_info)
	all_flux_info = np.array(all_flux_info)
	
	# indices:
	# m = 0: electron mode
	# m = 1: ion mode
	# to get flux values for mode m, species s, in the ion(0)/electron(1) scale, 
	# for flux type k, : all_flux_info[m, s, 0/1, k]
	
	# flux types: 0 particle
	#			  1 energy
	#			  2 toroidal stress
	#			  3 parallel stress
	#			  4 exchange
	
	# should find that summing over m and j for given s and k gives SOME
	# of the values in out.tglf.run, and should agree with the test function
	#for s in range(NSPECIES):
	#	for k in range(5):
	#		print(np.sum(all_flux_info[:, s, :, k]))
	
	return all_flux_info


direc = '/home/hdudding/Downloads/prova/'
test_function(direc)
all_flux_info = get_TGLF_modes(direc)

