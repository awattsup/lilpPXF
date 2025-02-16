import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import ticker
from matplotlib.lines import Line2D
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.table import Table, hstack
from astropy.stats import sigma_clip
import astropy.constants as ac

from scipy import stats
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
import resource
import copy
import time


try:
	from mpi4py import MPI
except:
	print('mpi4py not found')
	exit()

import glob
import sys, os
from astropy.io import fits

try:
	from ppxf.ppxf import ppxf
	import ppxf.ppxf_util as pputils
except:
	print('No pPXF!')
	exit()

home_dir = os.path.expanduser('~')
sys.path.append(f'{home_dir}/Research/programs/astro-functions')
import astro_functions as astrofunc

# import modified vorbin
from modified_voronoi_2d_binning import mod_voronoi_2d_binning as mv2db
from vorbin.voronoi_2d_binning import voronoi_2d_binning


#global variables
c = 1.e-3*ac.c.value	#km/s

emlines = {		'Hbeta':{	'lambda':[4861.333],			'ratio':[1]},
				'Halpha':{	'lambda':[6562.819],			'ratio':[1]},
				
				'Pa9':{		'lambda':[9229.014],			'ratio':[1]},
				'Pa10':{	'lambda':[9014.909],			'ratio':[1]},
				'Pa11':{	'lambda':[8862.782],			'ratio':[1]},
				'Pa12':{	'lambda':[8750.472],			'ratio':[1]},
				'Pa13':{	'lambda':[8665.019],			'ratio':[1]},			#overlaps with CaT line!!
				'Pa14':{	'lambda':[8598.392],			'ratio':[1]},
				'Pa15':{	'lambda':[8545.383],			'ratio':[1]},			#overlaps with CaT line!!
				'Pa16':{	'lambda':[8502.483],			'ratio':[1]},			#overlaps with CaT line!!
				# 'Pa17':{	'lambda':[8467.254],			'ratio':[1]},
				'Pa18':{	'lambda':[8437.956],			'ratio':[1]},
				# 'Pa19':{	'lambda':[8413.318],			'ratio':[1]},
				# 'Pa20':{	'lambda':[8392.397],			'ratio':[1]},


				'OI':{		'lambda':[6300.304,6363.78],	'ratio':[1,0.33]},					#low ionisation 14.53
				'OI8446': {	'lambda':[8446.359],			'ratio':[1]},
				'OII':{		'lambda':[7319.990, 7330.730],	'ratio':[1,1]}, #?? check 			#low ionisation 13.62
				'OIII':{	'lambda':[4958.911, 5006.843],	'ratio':[0.35,1]}, 					#high ionisation 35.12

				'NI':{ 		'lambda':[5200.257],			'ratio':[1]},
			 	'NII':{		'lambda':[6548.050,6583.460],	'ratio':[0.34,1]},					#low ionisation 15.43


			 	'Fe4993':{	'lambda':[4993.358],			'ratio':[1]},
			 	'Fe5018':{	'lambda':[5018.440],			'ratio':[1]},
			 	
				'HeI5876':{	'lambda':[5875.624],			'ratio':[1]}, 
				'HeI6678':{	'lambda':[6678.151],			'ratio':[1]}, 
				'HeI7065':{'lambda':[7065.196], 			'ratio':[1]},
				'HeII4685':{'lambda':[4685.710],			'ratio':[1]}, 

				'SII6716':{	'lambda':[6716.440],			'ratio':[1]},						#low ionisation 10.36
				'SII6730':{	'lambda':[6730.810],			'ratio':[1]},						#^^
				'SIII9069':{'lambda':[9068.6],				'ratio':[1]},						#high ionisation 23.33
				
				'ArIII7135':{'lambda':[7135.790],			'ratio':[1]}, 						#high ionisation 27.63 #doublet**
				'ArIII7751':{'lambda':[7751.060],			'ratio':[1]},						#high ionisation 27.63
				'ArIV':{	'lambda':[4711.260],			'ratio':[1]}, 						#high ionisation 40.74 #doublet**
				'ArIV':{	'lambda':[4740.120],			'ratio':[1]},						#high ionisation 40.74
				}

emlines_indiv = {'Hbeta':{	'lambda':[4861.333],			'ratio':[1]},
				'Halpha':{	'lambda':[6562.819],			'ratio':[1]},
				
				'Pa9':{		'lambda':[9229.014],			'ratio':[1]},
				'Pa10':{	'lambda':[9014.909],			'ratio':[1]},
				'Pa11':{	'lambda':[8862.782],			'ratio':[1]},
				'Pa12':{	'lambda':[8750.472],			'ratio':[1]},
				'Pa13':{	'lambda':[8665.019],			'ratio':[1]},			#overlaps with CaT line!!
				'Pa14':{	'lambda':[8598.392],			'ratio':[1]},
				'Pa15':{	'lambda':[8545.383],			'ratio':[1]},			#overlaps with CaT line!!
				'Pa16':{	'lambda':[8502.483],			'ratio':[1]},			#overlaps with CaT line!!
				# 'Pa17':{	'lambda':[8467.254],			'ratio':[1]},
				'Pa18':{	'lambda':[8437.956],			'ratio':[1]},
				# 'Pa19':{	'lambda':[8413.318],			'ratio':[1]},
				# 'Pa20':{	'lambda':[8392.397],			'ratio':[1]},

				'OI6300':{	'lambda':[6300.304],	'ratio':[1]},					#low ionisation 14.53
				'OI6364':{	'lambda':[6363.78],		'ratio':[1]},					#low ionisation 14.53
				'OI8446': {	'lambda':[8446.359],			'ratio':[1]},
				'OII7312':{	'lambda':[7319.990],	'ratio':[1]}, #?? check 			#low ionisation 13.62
				'OII7330':{	'lambda':[7330.730],	'ratio':[1]}, #?? check 			#low ionisation 13.62
				'OIII4959':{'lambda':[4958.911],	'ratio':[1]}, 					#high ionisation 35.12
				'OIII5006':{'lambda':[5006.843],	'ratio':[1]}, 					#high ionisation 35.12

				'NI':{ 		'lambda':[5200.257],			'ratio':[1]},
			 	'NII6548':{		'lambda':[6548.050],	'ratio':[1]},					#low ionisation 15.43
			 	'NII6583':{		'lambda':[6583.460],	'ratio':[1]},					#low ionisation 15.43

			 	'Fe4993':{	'lambda':[4993.358],			'ratio':[1]},
			 	'Fe5018':{	'lambda':[5018.440],			'ratio':[1]},
			 	
				'HeI5876':{	'lambda':[5875.624],			'ratio':[1]}, 
				'HeI6678':{	'lambda':[6678.151],			'ratio':[1]}, 
				'HeI7065':{'lambda':[7065.196], 			'ratio':[1]},
				'HeII4685':{'lambda':[4685.710],			'ratio':[1]}, 

				'SII6716':{	'lambda':[6716.440],			'ratio':[1]},						#low ionisation 10.36
				'SII6730':{	'lambda':[6730.810],			'ratio':[1]},						#^^
				'SIII9069':{'lambda':[9068.6],				'ratio':[1]},						#high ionisation 23.33
				
				'ArIII7135':{'lambda':[7135.790],			'ratio':[1]}, 						#high ionisation 27.63 #doublet**
				'ArIII7751':{'lambda':[7751.060],			'ratio':[1]},						#high ionisation 27.63
				'ArIV4711':{	'lambda':[4711.260],			'ratio':[1]}, 						#high ionisation 40.74 #doublet**
				'ArIV4740':{	'lambda':[4740.120],			'ratio':[1]},						#high ionisation 40.74
				}


###Main program callers

def run():
	"""
	The main running function. Reads the provided parameter file and runs the desired modules
	"""

	#Get parameters from the command line
	parameterfile = sys.argv[1]
	parameters = read_parameterfile(parameterfile)

	if parameters['read_data']:
		make_spectra_tables(parameterfile)

	if parameters['run_vorbin']:
		if parameters['vorbin_type'] == 'normal':
			voronoi_bin_cube(parameterfile)
		elif parameters['vorbin_type'] == 'sub':
			voronoi_subbin_cube(parameterfile)

	if parameters['run_fit']:
		parallel_pPXF_fit(parameterfile)


	if parameters['stelkin_outputs']:
		
		comm = MPI.COMM_WORLD
		rank = comm.Get_rank()
		nproc = comm.Get_size()

		if rank == 0:
			print("Making stellar kinematics maps")
			sys.stdout.flush()
			make_stelkin_map(parameters)
		
	if parameters['continuum_outputs']:
		
		comm = MPI.COMM_WORLD
		rank = comm.Get_rank()
		nproc = comm.Get_size()

		if rank == 0:
			make_continuum_subtracted_spectra(parameterfile)
			make_line_subcubes(parameterfile)
			measure_linecubes(parameterfile)
			make_linecube_fluxmaps(parameterfile)
			make_linecube_kinmaps(parameterfile)

	if parameters['binned_contsub']:
		comm = MPI.COMM_WORLD
		rank = comm.Get_rank()
		nproc = comm.Get_size()

		if rank == 0:
			make_line_subcubes(parameterfile)
			measure_linecubes(parameterfile)
			make_linecube_fluxmaps(parameterfile)
			make_linecube_kinmaps(parameterfile)

def parallel_pPXF_fit(parameterfile):
	"""
	The main workhorse, reads spectra, templates, and fits using MPI for parallelisation

	Parameters
	----------
	parameterfile : dict
		Dictionary containing fit inputs and parameters

	Returns
	-------
	Nothing
		Pixel table is updated with fit params and spectra are written to file
	"""
	
	comm = MPI.COMM_WORLD								#MPI set up
	rank = comm.Get_rank()
	nproc = comm.Get_size()

	#Head node reads spectra and templates
	if rank == 0:
		startTimeRun = time.time()
		print(f"Running with {nproc} processors")
		print("Preparing inputs")
		sys.stdout.flush()
		
		#Get spectra and templates
		parameters = read_parameterfile(parameterfile)							
		spax_properties = get_bin_properties(parameters)
		vorbin_nums = np.array(spax_properties['vorbin_num'][:])
		vorbin_nums = np.sort(np.unique(vorbin_nums[vorbin_nums>=0]))
		logLambda_spec, logRebin_spectra, logRebin_noise = get_spectra_to_fit(parameters)
		logLambda_templates, templates, components, gas_components = get_templates(parameters)

		print(f"The velscale of the data being fit is {parameters['galaxy_velscale']:.2f}")

		if 'fit_lrange' in parameters.keys():			#Truncate spectra to fit range
			Lmin = parameters['fit_lrange'][0]
			Lmax = parameters['fit_lrange'][1]
			specrange = np.where((logLambda_spec>=np.log(Lmin)) & 
									(logLambda_spec<=np.log(Lmax)))[0]
			logLambda_spec = logLambda_spec[specrange]
			logRebin_spectra = logRebin_spectra[specrange,:]
			logRebin_noise = logRebin_noise[specrange,:]

			specrange = np.where((logLambda_templates>=np.log(Lmin - 25)) & 
									(logLambda_templates<=np.log(Lmax + 25)))[0]
			logLambda_templates = logLambda_templates[specrange]
			templates = templates[specrange,:]

		#Mask sky lines and bad pixels
		good_pixels = create_spectrum_mask(logLambda_spec,parameters)

		#Get kinematic inputs and constraints
		start, moments, constr_kinem = get_constraints(parameters,spax_properties,vorbin_nums)
		dv = c*(np.nanmean(logLambda_templates[:parameters['velscale_ratio']]) - logLambda_spec[0]) # km/s

		parameters['components'] = components
		parameters['gas_components'] = gas_components
		parameters['dv'] = dv
		parameters['good_pixels'] = good_pixels

	comm.barrier()
	if rank == 0:
		print(f"Head node distributing {len(logRebin_spectra[0,:])} spectra")
	sys.stdout.flush()
	for nn in range(1,nproc):												#Distribute spectra, templates, and parameters to nodes
		if rank == 0:
			proc_spec_num = np.arange(nn,len(vorbin_nums),nproc,dtype=int)	#IDs of spectra/noise to send to each node
			
			proc_parameters = copy.deepcopy(parameters)
			proc_parameters['vorbin_nums'] = vorbin_nums[proc_spec_num]
			proc_parameters['templates'] = templates
			proc_parameters['logRebin_spectra'] = logRebin_spectra[:,proc_spec_num]
			proc_parameters['logRebin_noise'] = logRebin_noise[:,proc_spec_num]
			proc_parameters['start'] = [start[pp] for pp in proc_spec_num]
			proc_parameters['moments'] = [moments[pp] for pp in proc_spec_num]
			proc_parameters['constr_kinem'] =[constr_kinem[pp] for pp in proc_spec_num]

			comm.send(proc_parameters, dest=nn, tag=100+nn)				#Send
			tosend = None

		elif rank == nn:
			torecieve = comm.recv(source=0, tag=100+rank)				#Recieve
			proc_parameters = torecieve

			torecieve = None
		comm.barrier()

	if rank == 0:														#Select head node's spectra
		proc_spec_num = np.arange(0,len(vorbin_nums),nproc,dtype=int)
		proc_parameters = copy.deepcopy(parameters)
		proc_parameters['vorbin_nums'] = vorbin_nums[proc_spec_num]
		proc_parameters['templates'] = templates
		proc_parameters['logRebin_spectra'] = logRebin_spectra[:,proc_spec_num]
		proc_parameters['logRebin_noise'] = logRebin_noise[:,proc_spec_num]
		proc_parameters['start'] = [start[pp] for pp in proc_spec_num]
		proc_parameters['moments'] = [moments[pp] for pp in proc_spec_num]
		proc_parameters['constr_kinem'] =[constr_kinem[pp] for pp in proc_spec_num]

		parameters['spectra_shape'] = logRebin_spectra.shape
		parameters['templates_shape'] = templates.shape

		logRebin_spectra = None											#Free up memory										
		logRebin_noise = None

	if rank == 0:
		print(f"Spectra distributed, running fits")
		outputs_all = []												#To hold all outputs
	sys.stdout.flush()
	comm.barrier()

	timeStart = time.time()												#Start timer
	outputs = []														#Hold outputs from each node
	time_fits = []
	for vb, vorbin_num in enumerate(proc_parameters['vorbin_nums']):	#Loop over all spectra
		spectrum = proc_parameters['logRebin_spectra'][:,vb]
		noise = proc_parameters['logRebin_noise'][:,vb]
		spec_median = np.abs(np.nanmedian(spectrum))
		spectrum = spectrum / spec_median								#Nomalise spectrum to avoid numerical problems

		good_pixels_spec = proc_parameters['good_pixels']				#Adjust bad pixels for individual spectrum
		good_pixels_spec[np.isfinite(spectrum)==False] = False
		good_pixels_spec[(np.isfinite(noise)==False) | (noise <= 0)] = False
		noise[~good_pixels_spec] = 1									#Propagate bad pixels to noise array

		out = ppxf(	proc_parameters['templates'],						#Call fit function
					spectrum, 
					noise,
					proc_parameters['galaxy_velscale'],
					proc_parameters['start'][vb],
				moments = proc_parameters['moments'][vb],
				component = proc_parameters['components'],
				gas_component = proc_parameters['gas_components'],
				constr_kinem = proc_parameters['constr_kinem'][vb],
				degree = proc_parameters['degree'],
				mdegree = proc_parameters['mdegree'],
				velscale_ratio = proc_parameters['velscale_ratio'],
				vsyst = proc_parameters['dv'],
				goodpixels = np.arange(len(spectrum))[good_pixels_spec],
				plot = proc_parameters['plot'],
				quiet = not proc_parameters['plot']
				)
		if proc_parameters['plot']:
			plt.show()

		bestfit_spectrum = out.bestfit*spec_median

		if isinstance(proc_parameters['gas_components'],type(None)):	#Extract gas emission lines if being fit
			bestfit_gas = np.zeros_like(bestfit_spectrum)
			gas_flux = []
			gas_flux_error = []
		else:
			gas_flux = out.gas_flux
			gas_flux_error = out.gas_flux_error
			bestfit_gas = out.gas_bestfit*spec_median
		bestfit_stars = bestfit_spectrum - bestfit_gas
		
		output = [vorbin_num,out.chi2,out.sol,out.error,				
						gas_flux,gas_flux_error,
						bestfit_stars,bestfit_gas,
						out.weights,out.apoly,out.mpoly]
		outputs.append(output)

		if vb%100 == 0 and vb !=0:										#Pause and gather outputs every 100 fits to avoid comm limits
			comm.barrier()
			sys.stdout.flush()

			if rank == 0:								
				timeDiff = time.time()-timeStart						#Estimate runtime
				print(f"Fit is {100*vb/len(proc_parameters['vorbin_nums']):.2f}% through {len(proc_parameters['vorbin_nums'])} spectra")
				print(f"avg. {timeDiff/(vb+1):.3f}s/spec, "\
						f"{int(timeDiff/60):.0f}m{timeDiff%60:.1f}s ellapsed / est. {(timeDiff/(vb+1) * (len(proc_parameters['vorbin_nums']))/60):.1f}m total")
				print(f"Gathering outputs so far")
			sys.stdout.flush()

			outputs = comm.gather(outputs,root=0)
			if rank == 0:
				outputs_all.extend(outputs)
				print(f"Outputs gathered, continuing")
				print(f"-------------------------")
			outputs = []												#Reset node outputs list
			sys.stdout.flush()
	comm.barrier()														#Wait for all fits to finish

	if rank == 0:
		print("pPXF fits finished, gathering last outputs to head node")
		print(f"-------------------------")	
		sys.stdout.flush()

	outputs = comm.gather(outputs,root=0)								#Gather final outputs
	if rank == 0:
		outputs_all.extend(outputs)
		outputs_all = [oo for output in outputs_all for oo in output]
	comm.barrier()
	
	#Memory-freeing
	outputs = None
	proc_logRebin_spectra = None
	proc_logRebin_noise = None

	if rank == 0:														#Save outputs
		distribute_save_outputs(parameters, spax_properties,logLambda_spec, outputs_all)
		runTime = time.time() - startTimeRun
		print(f'Total wallclock runtime {int(runTime/60)}m{runTime%60:.1f}s')

	
### Data reading and binning ###
def read_parameterfile(filename = None):
	"""
	Read a parameter file and creates the parameter dictionary, with defaults
	Parameters
	----------
	filename : string
		Path to parameter file

	Returns
	-------
	parameters : dict
		Dictionary of parameters to be passed to the fitting function
	"""

	if filename is None: 				#Default  file
		filename = "./parameters.param"

	#Set code running path and parameter defaults	
	run_path = os.path.abspath(filename).split("/parameterfiles/")[0]
	parameters = {'read_data':False,'run_vorbin':False,'run_stelkin':False,'run_continuum':False,
					'base': run_path,
					'output': f"outputs",
					'velscale_ratio': 1, 'degree':-1, 'mdegree':-1,
					'spatial_mask': None,'vorbin_type':'normal',
					'run_fit': False, 'stelkin_outputs':False, 'continuum_outputs':False, 'binned_contsub':False,
					'plot': False,
					'gas_groups': None, 'gas_names': [],'gas_Ncomp': [],'gas_constraints': None,
					'fit_CaT':False}

	f = open(filename)
	for line in f:									#Loop over lines in file and read parameters
		if line[0] == "#" or line[0] == " ":		#Skip comment or empty lines
			continue
		else:
			line = line.split("\n")[0].split(" ")

			if line[0] == "read_data":				#Valid parameters
				parameters[line[0]] = eval(line[1])
			if line[0] == "run_vorbin":
				parameters[line[0]] = eval(line[1])
			if line[0] == "run_stelkin":
				parameters[line[0]] = eval(line[1])
			if line[0] == "run_continuum":
				parameters[line[0]] = eval(line[1])
			if line[0] == "stelkin_outputs":
				parameters[line[0]] = eval(line[1])
			if line[0] == "continuum_outputs":
				parameters[line[0]] = eval(line[1])
			if line[0] == "run_fit":
				parameters[line[0]] = eval(line[1])
			if line[0] == "binned_contsub":
				parameters[line[0]] = eval(line[1])
			if line[0] == "plot":
				parameters[line[0]] = eval(line[1])

			if line[0] == "datacube":
				parameters[line[0]] = line[1]
			elif line[0] == "stellar_templates":
				parameters[line[0]] = line[1]
			elif line[0] == "output":
				parameters[line[0]] = line[1]
			elif line[0] == "SN_cont":
				parameters[line[0]] = [float(line[1]),float(line[2])]
			elif line[0] == "SN_line":
				parameters[line[0]] = line[1::]
			elif line[0] == "vorbin_base":
				parameters[line[0]] = line[1]
			elif line[0] == "vorbin_outname":
				parameters[line[0]] = line[1]
			elif line[0] == "vorbin_SNname":
				parameters[line[0]] = line[1::]
			elif line[0] == "fit_CaT":
				parameters[line[0]] = eval(line[1])

			elif "_start" in line[0]:
				parameters[line[0]] = [float(line[1]),float(line[2])]
			elif "_lrange" in line[0]:
				parameters[line[0]] = [float(line[1]),float(line[2])]
			elif "lrange" in line[0]:
				parameters[line[0]] = [float(line[1]),float(line[2])]

			elif "gas_param" in line[0]:					#Read gas emission line parameter file
				gas_properties = read_gaslines_parameterfile(line[1])
				parameters['gas_groups'] = gas_properties[0]
				parameters['gas_names'] = gas_properties[1]
				parameters['gas_Ncomp'] = gas_properties[2]
				parameters['gas_constraints'] = gas_properties[3]
			elif len(line[1::]) == 1:
				try:
					parameters[line[0]] = int(line[1])
				except:
					try:
						parameters[line[0]] = float(line[1])
					except:
						parameters[line[0]] = line[1]

	#Update with specific running paths
	parameters['output'] = f"{parameters['base']}/{parameters['output']}"
	parameters['output_dir'] = f"{parameters['output']}/{parameters['output_dir']}"

	#Allow separate input and output directories. 
	if "input_dir" in parameters.keys():
		parameters['input_dir'] = f"{parameters['output']}/{parameters['input_dir']}"

	else:
		parameters['input_dir'] = parameters['output_dir']


	return parameters

def read_gaslines_parameterfile(filename = None):
	"""
	Read a gas line parameter file and creates the gas lines parameters, with defaults
	Parameters
	----------
	filename : string
		Path to parameter file

	Returns
	-------
	gas_params : list
		List of gas lines and constraints
	"""

	if filename is None:						#default parameter file				
		filename = "./gaslines.param"

	gas_groups = []
	gas_group = []
	constr_kinem = {'A_ineq':[],'b_ineq':[]}

	f = open(filename)
	for line in f:											#loop over lines
		if line[0] == "#" or line[0] == " " or line[0] == "\n":	
			if gas_group != []:								#Identifies a new line kinematic group	
				gas_groups.append(gas_group)
				gas_group = []
			continue
		elif  line.split("\n")[0].split(" ")[0] == "CONST":
			line = line.split("\n")[0].split("CONST ")[-1].split(" : ")
			A = [float(cc) for cc in line[0].split(" ")]
			b = float(line[1])
			
			constr_kinem['A_ineq'].append(A)
			constr_kinem['b_ineq'].extend([b])

		else:
			line = line.split("\n")[0].split(" ")			#Add line to group
			gas_group.extend(line)
	if gas_group != []:	
		gas_groups.append(gas_group)

	gas_Ncomp = []											#Make ppxf-style inputs
	gas_names = []
	for gg, group in enumerate(gas_groups):
		for gl, gas_line in enumerate(group):
			gas_names.extend([f"C{gg+1}-{gas_line}"])
		gas_Ncomp.extend([gl+1])


	gas_params = [gas_groups, gas_names, gas_Ncomp, constr_kinem]
	return gas_params

def make_spectra_tables(parameterfile):
	"""
	Read a datacube and makes a pixel table and 2D spectrum list
	Parameters
	----------
	parameterfile : dict
		Dictionary of parameters

	Returns
	-------
	Nothing 
		Outputs are written to file
	"""

	#Ensure if run in MPI mode that only the head node handles this
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	nproc = comm.Get_size()

	if rank == 0:
		parameters = read_parameterfile(parameterfile)

		#Make directories if not already existing
		if not os.path.isdir(f"{parameters['output']}/indiv"):
			os.mkdir(f"{parameters['output']}/indiv")
		if not os.path.isdir(f"{parameters['output']}/testcube"):
			os.mkdir(f"{parameters['output']}/testcube")
		if not os.path.isdir(f"{parameters['output']}/test"):
			os.mkdir(f"{parameters['output']}/test")


		#Check if these data files have already been made
		if os.path.isfile(f"{parameters['output']}/indiv/spaxel_properties.fits") and\
			os.path.isfile(f"{parameters['output']}/indiv/spectra_indiv.fits") and\
			os.path.isfile(f"{parameters['output']}/indiv/logRebin_spectra.fits"):

			print('Individual spectra products exist, going to test spectra creation')

			spax_properties = Table.read(f"{parameters['output']}/indiv/spaxel_properties.fits")
			Nx = spax_properties.meta['NX']
			Ny = spax_properties.meta['NY']
			Nl = spax_properties.meta['NL']

			hdul = fits.open(f"{parameters['output']}/indiv/spectra_indiv.fits")
			linLambda_obs, spectra_list  = read_spectra_hdul(hdul)
			spectra = spectra_list[0]
			noise = spectra_list[1]
			spectra_list = None
			header = hdul[0].header
			hdul.close()

			hdul = fits.open(f"{parameters['output']}/indiv/logRebin_spectra.fits")
			logLambda, spectra_list  = read_spectra_hdul(hdul)
			logRebin_spectra = spectra_list[0]
			logRebin_noise = spectra_list[1]
			spectra_list = None
			header = hdul[0].header
			velscale = [header['VELSCALE']]
			hdul.close()

			print("All read in")

		else:				#Read the data cube

			hdul = fits.open(parameters['datacube'])
			header = hdul[1].header

			spectra = hdul[1].data
			noise = np.sqrt(hdul[2].data)
			hdul = None				#Free up memory

			Nx = header['NAXIS1']
			Ny = header['NAXIS2']
			Nl = header['NAXIS3']

			spectra = spectra.reshape(Nl, Nx*Ny)
			noise = noise.reshape(Nl, Nx*Ny)

			#Create and de-redshift spectral axis
			linLambda_obs = astrofunc.get_wavelength_axis(header)
			linLambda_obs = linLambda_obs / (1.e0 + parameters['z'])		

			#Truncate to desired wavelength range
			if 'lrange' in parameters.keys():
				Lambda_range = np.logical_and(linLambda_obs >= parameters['lrange'][0] ,
										linLambda_obs <= parameters['lrange'][1])

				spectra = spectra[Lambda_range]
				noise = noise[Lambda_range]
				linLambda_obs = linLambda_obs[Lambda_range]

			#Set up pixel and bin IDs
			spax_number = np.arange(Nx*Ny,dtype='int')
			vorbin_number = np.zeros([Nx*Ny],dtype='int')
			
			#Identify bad pixels
			obs_flags = np.all(np.isfinite(spectra), axis = 0)
			obs_flags_nums = np.zeros(Nx*Ny)
			obs_flags_nums[obs_flags] = 1
			spec_good = np.where(obs_flags_nums == 1)[0]
			spec_bad = np.where(obs_flags_nums == 0)[0]

			#Measure signal to noise over desired range
			SN_cont_Lrange = np.logical_and(linLambda_obs >= parameters['SN_lrange'][0] , 
											linLambda_obs <= parameters['SN_lrange'][1])
			spax_signal_cont = np.nanmedian(spectra[SN_cont_Lrange,:], axis = 0)
			spax_noise_cont = np.abs(np.nanmedian(noise[SN_cont_Lrange,:], axis = 0))

			#Pixel coordinates
			spax_xxyy, spax_RADEC, spax_SP = astrofunc.make_pix_WCS_grids(header)
			spax_size = np.abs(np.diff(spax_SP[0])[0])

			spax_properties = np.column_stack([spax_number.astype(int),
											vorbin_number,
											obs_flags_nums,
											spax_xxyy[0].reshape(Nx*Ny).astype(int),
											spax_xxyy[1].reshape(Nx*Ny).astype(int),
											spax_RADEC[0].reshape(Nx*Ny),
											spax_RADEC[1].reshape(Nx*Ny),
											spax_SP[0].reshape(Nx*Ny),
											spax_SP[1].reshape(Nx*Ny),
											spax_signal_cont,
											spax_noise_cont
											]
											)

			#Trim to only good observed spaxels to save memory
			spax_properties = spax_properties[spec_good]

			#Table metadata
			metadata = {'Nx':Nx,'Ny':Ny,'Nl':Nl,'d_spax':spax_size,'z':parameters['z'],
						'SN_LMIN':parameters['SN_lrange'][0],'SN_LMIN':parameters['SN_lrange'][1]}
			#Set up pixel table
			spax_properties = Table(spax_properties,
						names=['spax_num','vorbin_num','obs_flag','spax_xx','spax_yy',
								'spax_RA','spax_DEC','spax_SPxx','spax_SPyy',
								'spax_signal_cont','spax_noise_cont'],#+\
								# [f'spax_signal_{line}' for line in parameters['SN_line']]+\
								# [f'spax_noise_{line}' for line in parameters['SN_line']],
								meta=metadata)

			#Assign bin numbers only to good spectra
			spax_properties['vorbin_num'] = np.arange(len(spax_properties),dtype='int')
			
			#Save individual spaxel properties
			print("Saving individual spaxel properties")
			sys.stdout.flush()
			spax_properties.write(f"{parameters['output']}/indiv/spaxel_properties.fits",overwrite=True)
			print("Saved")
			sys.stdout.flush()


			#Trim spectra to only good observed spaxels
			spectra = spectra[:,spec_good]
			noise = noise[:,spec_good]

			#Save individual spectra
			indiv_header = fits.Header()
			indiv_header['COMMENT'] = "A.B. Watts"
			indiv_primary_hdu = fits.PrimaryHDU(data = linLambda_obs,
												header = indiv_header)

			indiv_spectra_hdu = fits.BinTableHDU.from_columns(
									fits.ColDefs([
									fits.Column(
									array = spectra.T,
									name='SPEC',format=str(len(spectra))+'D'
									)]))
			indiv_noise_hdu = fits.BinTableHDU.from_columns(
									fits.ColDefs([
									fits.Column(
									array = noise.T,
									name='VAR',format=str(len(noise))+'D' 
									)]))
			hdul_indiv = fits.HDUList([indiv_primary_hdu,
										indiv_spectra_hdu,
										indiv_noise_hdu])
			print("Saving indiv. spectra table")
			sys.stdout.flush()
			hdul_indiv.writeto(f"{parameters['output']}/indiv/spectra_indiv.fits",overwrite=True)
			print("Saved")
			indiv_spectra_hdu = None
			indiv_noise_hdu = None
			hdul_indiv = None
			sys.stdout.flush()

			#log-rebin the individual spectra
			print("log-rebinning the individual spectra")
			sys.stdout.flush()
			logRebin_spectra, logLambda, velscale = pputils.log_rebin(linLambda_obs,
																		spectra)

			logRebin_noise, logLambda1, velscale1 = pputils.log_rebin(linLambda_obs,
																		noise)
			print("Done")
			sys.stdout.flush()
			
			#save log-rebinned individual spectra
			logRebin_header = fits.Header()
			logRebin_header['velscale'] = velscale
			logRebin_header['COMMENT'] = "A.B. Watts"

			logRebin_primary_hdu = fits.PrimaryHDU(data = logLambda,
													header = logRebin_header)
			logRebin_spectra_hdu = fits.BinTableHDU.from_columns(
									fits.ColDefs([
									fits.Column(
									array = logRebin_spectra.T,
									name='SPEC',format=str(len(logRebin_spectra))+'D' 
									)]))
			logRebin_noise_hdu = fits.BinTableHDU.from_columns(
									fits.ColDefs([
									fits.Column(
									array = logRebin_noise.T,
									name='VAR',format=str(len(logRebin_noise))+'D'
									)]))
			hdul_logRebin = fits.HDUList([logRebin_primary_hdu,
											logRebin_spectra_hdu,
											logRebin_noise_hdu])

			print("Saving log-rebinnned indiv. spectra table")
			sys.stdout.flush()
			hdul_logRebin.writeto(f"{parameters['output']}/indiv/logRebin_spectra.fits",overwrite=True)
			print("Saved")
			logRebin_spectra_hdu = None
			logRebin_noise_hdu = None
			hdul_logRebin = None
			sys.stdout.flush()

	comm.barrier()

def read_spectra_hdul(hdul):
	"""
	Read 2D spectral tables
	Parameters
	----------
	hdul : HDUL
		fits HDU list

	Returns
	-------
	wave : array
		Wavelength array
	spectra : 2-list
		List containing spectra and noise 
	"""
	wave = np.asarray(hdul[0].data)
	spectra = []
	for hdu in hdul[1::]:
		spectra.append(np.asarray([dd[0] for dd in hdu.data]).T)

	return wave, spectra


def voronoi_bin_cube(parameterfile):
	"""
	Perform voronoi spatial binning
	Parameters
	----------
	parameterfile : dict
		Dictionary of parameters

	Returns
	-------
	Nothing 
		Outputs are written to file
	"""
	
	#Ensure if run in MPI mode that only the head node handles this
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	nproc = comm.Get_size()

	if rank == 0:
		#Get parameter file and pixel properties
		parameters = read_parameterfile(parameterfile)
		spax_prop_file = f"{parameters['input_dir']}/spaxel_properties.fits"
		spax_properties = Table.read(spax_prop_file)

		if not os.path.isdir(f"{parameters['output_dir']}/"):
			os.mkdir(f"{parameters['output_dir']}/")
			os.mkdir(f"{parameters['output_dir']}/figures")

		#Value for where spectra don't make the min SN cut
		spax_properties['vorbin_num'] =  np.full(len(spax_properties),-1)
		spax_properties['vorbin_xx'] = np.full(len(spax_properties),-1)
		spax_properties['vorbin_yy'] = np.full(len(spax_properties),-1)
		spax_properties['vorbin_SN'] = np.full(len(spax_properties),-1)

		#Voronoi binning parameters
		spax_size = 1
		SN_indiv = parameters['vorbin_SNmin']
		SN_vorbin = parameters['vorbin_SNtarget']

		#Include an additional spatial mask to avoid areas with, e.g. lots of sky or very very little galaxy
		if not isinstance(parameters['spatial_mask'],type(None)):
			with fits.open(parameters['spatial_mask']) as hdul: mask = hdul[1].data.flatten()
			mask = mask[np.asarray(spax_properties['spax_num'].data,dtype=int)]
			spax_properties['spatial_mask'] = np.full(len(spax_properties),0)
			spax_properties['spatial_mask'][np.isfinite(mask)] = mask[np.isfinite(mask)]
		else:
			mask = np.ones(len(spax_properties))

		#Read emission line fluxes if available
		if os.path.isfile(f"{parameters['input_dir']}/spax_emline_fluxes.fits"):
			spax_emline_fluxes = Table.read(f"{parameters['input_dir']}/spax_emline_fluxes.fits")

		#Select what propeties to bin on, e.g. starts, Halpha, [OIII]
		SNnames = parameters['vorbin_SNname']
		spax_signal = np.zeros([len(spax_properties),len(SNnames)])
		spax_noise = np.zeros([len(spax_properties),len(SNnames)])
		spax_xx = np.full([len(SNnames),len(spax_properties)],spax_properties['spax_xx'].data).T
		spax_yy = np.full([len(SNnames),len(spax_properties)],spax_properties['spax_yy'].data).T

		#Loop over SN properties and set values for non-detected data
		for nn, name in enumerate(SNnames):
			if name ==  "cont" :
				spax_signal[:,nn] = spax_properties[f'spax_signal_{name}'].data
				spax_noise[:,nn] = spax_properties[f'spax_noise_{name}'].data

			else:
				spax_signal[:,nn] = spax_emline_fluxes[f'{name}'].data
				spax_noise[:,nn] = spax_emline_fluxes[f'{name}_err'].data

				spax_signal[spax_signal[:,nn]==-99,nn] =  0.5*spax_noise[spax_signal[:,nn]==-99,nn]

		#Only bin pixels above minimum SN value
		spax_SN = np.nanmin(spax_signal / spax_noise,axis=1)
		refs = np.logical_and(spax_SN >= SN_indiv, mask==1)
		
		if len(SNnames) > 1: 			#Modified binning function if enfocing a bin S/N in more than one property	
			sn_func = voronoi_multiSN_function
			spax_signal = spax_signal[refs,:]
			spax_noise = spax_noise[refs,:]
			spax_xx = spax_xx[refs,:]
			spax_yy = spax_yy[refs,:]
		else:
			sn_func = None
			spax_signal = spax_signal[refs].flatten()
			spax_noise = spax_noise[refs].flatten()
			spax_xx = spax_xx[refs].flatten()
			spax_yy = spax_yy[refs].flatten()


				

		#compute Voronoi bins
		print('Computing voronoi bins')
		sys.stdout.flush()
		vorbin_nums, Nxx, Nyy, vorbin_xx, vorbin_yy, vorbin_SN, vorbin_Npix, scale = \
						mv2db(spax_xx, spax_yy, 
											spax_signal, spax_noise, 
											SN_vorbin, pixelsize=spax_size,
											plot = False,
											sn_func = sn_func,
											quiet=True)

		#Record vorbin properties for each spaxel
		spax_properties['vorbin_num'][refs] = vorbin_nums
		spax_properties['vorbin_signal'] = np.full(len(spax_properties),-1)

		#Distribute properites into pixel table
		for vv, num in enumerate(np.sort(np.unique(vorbin_nums))):
			inbin = np.in1d(spax_properties['vorbin_num'].data, num)
			spax_properties['vorbin_xx'][inbin] =\
												vorbin_xx[vv]

			spax_properties['vorbin_yy'][inbin] =\
												vorbin_yy[vv]

			spax_properties['vorbin_SN'][inbin] =\
												vorbin_SN[vv]
			spax_properties['vorbin_signal'][inbin] =\
												np.nansum(spax_signal[np.in1d(vorbin_nums,num)])

		spax_properties.write(f"{parameters['output_dir']}/spaxel_properties.fits",overwrite=True)

		#Make binned spectra
		create_binned_spectra(parameters, contsub=parameters['binned_contsub'])
		#Plot map of spatial bin
		make_vorbins_map(parameters)
	comm.barrier()

def voronoi_multiSN_function(index, signal =  None, noise = None):
	"""
	Custom function for computing the S/N for mulitple properties (e.g., stars, Halpha) and returning the minimum
	Parameters
	----------
	index : array size N
		Array of intergers of values to combine
	signal : array size m x N
		Array of signal values for m properties
	noise : array size m x N
		Array or noise values for m properies

	Returns
	-------
	sn : float
		Minimum S/N across m properties
		
	"""
	if signal.ndim ==1:			#Single property case
		sn = np.sum(signal[index]) / np.sqrt(np.sum(noise**2))

	elif signal.ndim > 1:		#Multiple properties case
		if isinstance(index,int) or isinstance(index,np.int64) or len(index)==1:	#single pixel case
			SN = signal[index,:] / np.sqrt(noise[index,:]**2)

		elif len(index)>1:
			SN = np.sum(signal[index,:],axis = 0) / np.sqrt(np.sum(noise[index,:]**2,axis = 0))

		sn = np.min(SN)

	return sn

def create_binned_spectra(parameters = None, contsub=False):
	"""
	Co-adds spectra based on spatial bins
	
	Parameters
	----------
	parameters : dict 
		Dictionary of parameters
	contsub : bool
		Flag to identify if the spectra being co-added are already log-rebinned

	Returns
	-------
	Nothing
		Outputs are saved to file
		
	"""

	if parameters is None:						#Default parameter file
		parameters = read_parameterfile()

	#Read pixel properties
	spax_prop_file = f"{parameters['output_dir']}/spaxel_properties.fits"
	spax_properties = Table.read(spax_prop_file)

	#Read input spectra
	if contsub:
		spectra_file = f"{parameters['input_dir']}/logRebin_contsub_spectra.fits"
	else:
		spectra_file = f"{parameters['input_dir']}/spectra_indiv.fits"

	hdul = fits.open(spectra_file)
	header = hdul[0].header
	linLambda_obs, spectra_list = read_spectra_hdul(hdul)
	hdul.close()
	spectra = spectra_list[0]
	noise = spectra_list[1]
	if contsub:
		logLambda_obs = linLambda_obs
		with fits.open(f"{parameters['output']}/indiv/logRebin_spectra.fits") as hdul: header1 =hdul[0].header
		velscale = header1['VELSCALE']

	#Get bin numbers
	vorbin_nums = np.unique(spax_properties['vorbin_num'])
	vorbin_nums = np.sort(vorbin_nums[vorbin_nums >= 0])

	#Make binned spectra arrays
	vorbin_spectra = np.zeros([len(spectra),len(vorbin_nums)])
	vorbin_noise = copy.deepcopy(vorbin_spectra)

	#Loop over bins and sum spectra and noise in each bin
	for vb in vorbin_nums:
		inbin = np.in1d(spax_properties['vorbin_num'][:], vb)
		if not contsub:
			vorbin_spectra[:,vb] = np.sum(spectra[:,inbin], axis=1)
			vorbin_noise[:,vb] = np.sqrt(np.sum(noise[:,inbin]**2, axis=1))
		elif contsub:
			vorbin_spectra[:,vb] = np.sum(spectra[:,inbin], axis=1)
			vorbin_noise[:,vb] = np.sqrt(np.sum(noise[:,inbin]**2, axis=1))


	#Save native resoltuion binned spectra
	if not contsub:
		vorbin_header = fits.Header()
		vorbin_header['COMMENT'] = "A.B. Watts"
		vorbin_primary_hdu = fits.PrimaryHDU(data = linLambda_obs,
												header = vorbin_header)

		vorbin_spectra_hdu = fits.BinTableHDU.from_columns(
								fits.ColDefs([
								fits.Column(
								array = vorbin_spectra.T,
								name='SPEC',format=str(len(vorbin_spectra))+'D'
								)]))
		hdul_vorbin = fits.HDUList([vorbin_primary_hdu,
									vorbin_spectra_hdu])

		vorbin_noise_hdu = fits.BinTableHDU.from_columns(
								fits.ColDefs([
								fits.Column(
								array = vorbin_noise.T,
								name='VAR',format=str(len(vorbin_noise))+'D' 
								)]))
		hdul_vorbin.append(vorbin_noise_hdu)
		

		print("Saving vorbin. spectra table")
		hdul_vorbin.writeto(f"{parameters['output_dir']}/spectra_vorbin.fits",overwrite=True)
		print("Saved")

	#Log-rebin binned spectra if needed
	if not contsub:
		logRebin_vorbin_spectra, logLambda_obs, velscale = pputils.log_rebin(linLambda_obs,
																	vorbin_spectra)
		logRebin_vorbin_noise, logLambda_obs1, velscale1 = pputils.log_rebin(linLambda_obs,
																	vorbin_noise)
	elif contsub:
		logRebin_vorbin_spectra = vorbin_spectra
		logRebin_vorbin_noise = vorbin_noise

	if isinstance(velscale,type(np.array([]))):
		velscale = velscale[0]

	#Save log-rebinned vorbin spectra
	logRebin_vorbin_header = fits.Header()
	logRebin_vorbin_header['VELSCALE'] = velscale
	logRebin_vorbin_header['COMMENT'] = "A.B. Watts"
	logRebin_vorbin_primary_hdu = fits.PrimaryHDU(data=logLambda_obs,
												header = logRebin_vorbin_header)
	logRebin_vorbin_spectra_hdu = fits.BinTableHDU.from_columns(
							fits.ColDefs([
							fits.Column(
							array = logRebin_vorbin_spectra.T,
							name='SPEC',format=str(len(logRebin_vorbin_spectra))+'D'
							)]))
	hdul_logRebin_vorbin = fits.HDUList([logRebin_vorbin_primary_hdu,
								logRebin_vorbin_spectra_hdu])
	logRebin_vorbin_noise_hdu = fits.BinTableHDU.from_columns(
							fits.ColDefs([
							fits.Column(
							array = logRebin_vorbin_noise.T,
							name='VAR',format=str(len(logRebin_vorbin_noise))+'D' 
							)]))


	hdul_logRebin_vorbin.append(logRebin_vorbin_noise_hdu)
	print("Saving log-rebinnned. vorbin spectra table")
	if not contsub:
		hdul_logRebin_vorbin.writeto(f"{parameters['output_dir']}/logRebin_spectra.fits",overwrite=True)
	elif contsub:
		hdul_logRebin_vorbin.writeto(f"{parameters['output_dir']}/logRebin_contsub_spectra.fits",overwrite=True)

	print("Saved")

def get_bin_properties(parameters):
	"""
	Reads the spaxel properties file and adds redshift to the parameter dictionary
	Parameters
	----------
	parameters : dict 
		Dictionary of parameters

	Returns
	-------
	spax_properties
		Table of spaxel properties
		
	"""

	spax_properties_file = f"{parameters['input_dir']}/spaxel_properties.fits"
	spax_properties = Table.read(spax_properties_file)
	parameters['z'] = spax_properties.meta['Z']

	return spax_properties


def get_spectra_to_fit(parameters):
	"""
	Reads the log-rebinned spectra and noise and adds VELSCALE to the parameter dictionary
	
	Parameters
	----------
	parameters : dict 
		Dictionary of parameters

	Returns
	-------
	logLambda_spec : array 1xN
		log-spaced spectral axis vector
	logRebin_spectra : array NxS
		Array of log-rebinned spectra, one spectrum per column
	logRebin_noise : array
		Array of log-rebinned noise vectors, one per column
		
	"""
	spectra_file = f"{parameters['input_dir']}/logRebin_spectra.fits"

	hdul = fits.open(spectra_file)
	logLambda_spec, spectra  = read_spectra_hdul(hdul)
	logRebin_spectra = spectra[0]
	logRebin_noise = spectra[1]
	spectra = None					#Free up memory

	header = hdul[0].header
	hdul.close()
	velscale = header['VELSCALE']
	parameters['galaxy_velscale'] = float(velscale)

	return logLambda_spec,logRebin_spectra,logRebin_noise

def distribute_save_outputs(parameters,spax_properties,logLambda_spec,outputs_all):
	"""
	Distributes fit parameters into the pixel table and saves best fit spectra outputs
	
	Parameters
	----------
	parameters : dict 
		Dictionary of parameters
	spax_properties : astropy.Table 
		Table of pixel properties
	logLambda_spec : array 
		Spectral axis of the observed data in log-rebin
	outputs_all : list 
		List of pPXF outputs

	Returns
	-------
	Nothing
		Outputs are saved to disc
		
	"""

	#Names of possible kinematic moments
	moments_list = ['V','sigma','h3','h4','h5','h6']

	#Column names of stellar kinematics and gas outputs
	stars_kin_names = [f'{k}_stellar{ee}'  for k in moments_list[:parameters['stars_moments']] for ee in ['','_err']] 
	gas_kin_names = [f'{k}_gas_C{ii+1}{ee}' for ii in range(len(parameters['gas_Ncomp'])) for k in ['V','sigma'] for ee in ['','_err']]
	gas_flux_names = [f'{ll}_flux{ee}' for ll in parameters['gas_names'] for ee in ['','err']]

	#Total name list and array to hold all data
	names = ['chi2'] + stars_kin_names + gas_kin_names + gas_flux_names		 
	kin_fluxes = np.full([len(spax_properties),len(names)],np.nan)

	#Arrays to hold bestfit spectra outputs and polynomials
	bestfit_stars = np.zeros(parameters['spectra_shape'])
	bestfit_gas = np.zeros(parameters['spectra_shape'])
	template_weights = np.empty((parameters['templates_shape'][1],parameters['spectra_shape'][1]))
	apy = np.empty(parameters['spectra_shape'])
	mpy = np.empty(parameters['spectra_shape'])

	
	for out in outputs_all:											#Loop over outputs
		vorbin_num = out[0]
		ref = np.where(spax_properties['vorbin_num'][:] == vorbin_num)[0]	#Find all pixels in this bin 
		chi2 = out[1]
		kin = np.hstack(out[2])
		err = np.hstack(out[3])*np.sqrt(chi2)						#Scale error by chi squared as instructed by pPXF
		if isinstance(parameters['gas_components'],type(None)):		#Handle inclusion of gas emission lines
			gas_flux =  np.array([])
			gas_flux_err =  np.array([])
		else:
			gas_flux = out[4]
			gas_flux_err = out[5]*np.sqrt(chi2)

		kin_err = np.full(1+len(kin)+len(err),None)					#Populate the array with output fit values
		kin_err[0] = chi2
		kin_err[1::2] = kin
		kin_err[2::2] = err

		flux_err = np.full(len(gas_flux)+len(gas_flux_err),None)
		flux_err[::2] = gas_flux
		flux_err[1::2] = gas_flux_err


		kin_fluxes[ref,:] = np.hstack((kin_err,flux_err))				#Distribute fit values to pixel table

		

		bestfit_stars[:,vorbin_num] = np.array(out[6])						#Save bestfit spectra
		bestfit_gas[:,vorbin_num] = np.array(out[7])
		template_weights[:,vorbin_num] = out[8]
		apy[:,vorbin_num] = out[9]
		mpy[:,vorbin_num] = out[10]

	
	if not os.path.isdir(f"{parameters['output_dir']}"):					#Make folders if they dont exist
		os.mkdir(f"{parameters['output_dir']}")
		os.mkdir(f"{parameters['output_dir']}/figures")
	
	kin_fluxes = Table(kin_fluxes,names=names)
	print("Saving bestfit kinematics and fluxes table")
	sys.stdout.flush()
	kin_fluxes.write(f"{parameters['output_dir']}/bestfit_kinematics_fluxes.fits",overwrite=True)

	#Save fit weights and polynomials
	primary_hdu = fits.PrimaryHDU(data = logLambda_spec)
	primary_hdu.header["COMMENT1"] = "Best fit weights for bins"

	weights_hdu = fits.BinTableHDU.from_columns(
							fits.ColDefs([
							fits.Column(
							array = template_weights.T,
							name='weights',format=str(len(template_weights))+'D'
							)]))
	apoly_hdu = fits.BinTableHDU.from_columns(
							fits.ColDefs([
							fits.Column(
							array = apy.T,
							name='apoly',format=str(len(apy))+'D'
							)]))
	mpoly_hdu = fits.BinTableHDU.from_columns(
							fits.ColDefs([
							fits.Column(
							array = mpy.T,
							name='mpoly',format=str(len(mpy))+'D'
							)]))
	hdul = fits.HDUList([primary_hdu,weights_hdu,apoly_hdu, mpoly_hdu])		
	hdul.writeto(f"{parameters['output_dir']}/bestfit_weights.fits",overwrite=True)


	#Save bestfit spectra
	primary_hdu = fits.PrimaryHDU(data = logLambda_spec)
	primary_hdu.header["COMMENT1"] = "Best fit spectra for bins"
	bestfit_stars_hdu = fits.BinTableHDU.from_columns(
							fits.ColDefs([
							fits.Column(
							array = bestfit_stars.T,
							name='BESTSTARS',format=str(len(bestfit_stars))+'D'
							)]))
	bestfit_gas_hdu = fits.BinTableHDU.from_columns(
							fits.ColDefs([
							fits.Column(
							array = bestfit_gas.T,
							name='BESTGAS',format=str(len(bestfit_gas))+'D'
							)]))

	hdul = fits.HDUList([primary_hdu,
						bestfit_stars_hdu,bestfit_gas_hdu])
	print("Saving best fit spectra table")
	sys.stdout.flush()
	hdul.writeto(f"{parameters['output_dir']}/logRebin_bestfit_spectra.fits",overwrite=True)
	print("Saved")
	sys.stdout.flush()



### Template functions

def get_templates(parameters):
	"""
	Gets all templates

	Parameters
	----------
	parameters : dict
		Dictionary of parameters

	Returns
	-------
	logLambda_templates : array
		log-rebinned spectral axis
	templates : array
		Column vectors of all templates
	components : array
		List of numbers indicating kinematic components each template belongs to
	gas_components
		List of numbers indicating gas line components

	"""

	logLambda_templates, templates = get_stellar_templates(parameters)		#Get stellar templates
	components_stars = [0]*len(templates[0,:])


	if not isinstance(parameters['gas_groups'],type(None)):
		gas_templates = make_gas_templates(parameters,logLambda_templates)		#Get gas templates
		templates = np.column_stack((templates,gas_templates))

		components_gas = [[nn + 1 + components_stars[-1]]*NN for nn,NN in enumerate(parameters['gas_Ncomp'])]
		components_gas = [comp for group in components_gas for comp in group]
		components = components_stars + components_gas

		gas_components = np.array(components) > components_stars[-1]
	else:
		components = components_stars
		gas_components = None


	return logLambda_templates, templates, components, gas_components

def get_stellar_templates(parameters, match_velscale=True, convolve = True, regrid=False):
	"""
	Reads chosen spectral templates, convolve to match the same resoltuion as the data
	
	Parameters
	----------
	parameters : dict
		Dictionary of parameters
	match_velscale : bool
		Optionally match the template velocity scale to the same, or a multiple of, the data
	convolve : bool
		Smooth the templates to match the spectral resolution of the data
	regrid : bool
		Shape the templates into a grid for regularised fitting. 

	Returns
	-------
	logLambda_templates : array
		Log-rebinned spectral axis of the templates
	stellar_templates_final : 2- or 3-D array 
		Log-rebinned and convolved stellar templates 

	"""
	
	if parameters is None:						#Load default parameter file
		parameters = read_parameterfile()

	#Choose which set of templates to load
	if "EMILES" in parameters['stellar_templates']:
		stellar_templates, linLambda_templates, wave_step  = read_EMILES()
		FWHM_template = np.full_like(linLambda_templates,2.51)				#FWHM of EMILES templates is 2.51A

	elif "MILES" in parameters['stellar_templates']:
		stellar_templates, linLambda_templates, wave_step  = read_MILES()
		FWHM_template = np.full_like(linLambda_templates,2.51)				#FWHM of EMILES templates is 2.51A

	elif "BPASS" in parameters['stellar_templates']:
		if parameters['stellar_templates'].split('-')[0] == parameters['stellar_templates']:
			mode = None
		else:
			mode = parameters['stellar_templates'].split('-')[-1]
		stellar_templates, linLambda_templates, wave_step  = read_BPASS(mode=mode)
		FWHM_template = np.full_like(linLambda_templates,0)					#FWHM of BPASS is 0 as theoretical

	#Optionally convolve templates
	stellar_templates_conv = np.zeros_like(stellar_templates)
	for tt in range(stellar_templates.shape[1]):
		template = stellar_templates[:,tt]

		if convolve:
			FWHM_galaxy = MUSE_LSF_Bacon17(linLambda_templates,z = parameters['z'])		#Get MUSE resolution
			FWHM_diffs = np.sqrt(FWHM_galaxy**2.e0 - FWHM_template**2.e0 )				#Difference of sigmas
			FWHM_diffs[np.isfinite(FWHM_diffs)==False] = 1.e-3
			stddev_diffs = FWHM_diffs / (wave_step  * 2.355)					#VARSMOOTH TAKES UNITS OF WAVELENGTH, NOT PIXELS
			stellar_templates_conv[:,tt] = pputils.varsmooth(linLambda_templates,template,stddev_diffs)
		else:
			stellar_templates_conv[:,tt] = template

	#Keep only templates where everyything is finite
	stellar_templates_conv = stellar_templates_conv[:,np.where(np.prod(np.isfinite(stellar_templates_conv),axis=0)==1)[0]]

	if match_velscale:			#Optionally elocity resolution match to data or up sample
		template_velscale = parameters['galaxy_velscale'] / parameters['velscale_ratio']
	else:
		template_velscale = None

	#log-rebin templates
	stellar_templates_logRebin, logLambda_templates, temp_velscale = pputils.log_rebin(linLambda_templates,
															stellar_templates_conv,
															velscale = template_velscale)


	if not regrid: 				#Light-weighted, normalise each by themself
		stellar_templates_final = stellar_templates_logRebin
		stellar_templates_final /= np.median(stellar_templates_final,axis=0)

	elif regrid:				#Mass-weighted, normalise by median of all templates and resort into alpha x Z x Age grid
		stellar_templates_final = np.full([len(stellar_templates_logRebin[:,0]),
											Nages,
											NZs,
											Nalphas],np.nan)
		for tt in range(len(stellar_templates_logRebin[0,:])):
			age_loc = np.where(Ages == template_params[tt,0])[0][0]
			Z_loc = np.where(Zs == template_params[tt,1])[0][0]
			alpha_loc = np.where(Alphas == template_params[tt,2])[0][0]
			
			stellar_templates_final[:,age_loc,Z_loc,alpha_loc] = stellar_templates_logRebin[:,tt]
		stellar_templates_final /= np.median(stellar_templates_final)

	return logLambda_templates, stellar_templates_final

def MUSE_LSF_Bacon17(ll, z = 0):
	"""
	Returns redshift-corrected linespread function for MUSE
	
	Parameters
	----------
	ll : float/array
		wavelength in Angstroms
	z : float
		redshift

	Returns
	-------
	FWHM : float/array
		FWHM at wavelength ll in Angstroms, corrected for redshift
		
	"""
	ll = ll*(1.e0 + z)										#convert rest-frame wavelength to observed frame
	FWHM = 5.866e-8 * ll*ll - 9.187e-4 * ll + 6.040 		#get LSF at observed-frame wavelength
	FWHM = FWHM / (1.e0 + z)								#convert obs-frame LSF to narrower de-redshifted value

	return FWHM
def read_EMILES():
	"""
	Reads EMILES spectral templates from a directory, which should be in the same level as this file
	
	Parameters
	----------
	Nothing

	Returns
	-------
	templates : array
		Stellar templates sorted as column vectors
	linLambda_templates_trunc : array
		Linear spectral axis of the templates
	wave_step : float
		wavelength channel resolution
		
	"""

	#Identify template files
	template_dir = f"{os.path.dirname(os.path.realpath(__file__))}/templates/EMILES"
	files = sorted(glob.glob(template_dir + "/*"))
	
	params = ["ch","Z","T","_iTp","_"] 				#Possible template parameters to identify
	template_params = []
	for ff in range(len(files)):					#Loop over files to get the parameters
	
		#Split filename to get parameters
		a = [files[ff].split("/")[-1].split(params[p],maxsplit=1)[1].split(params[p+1],maxsplit=1)[0]
						for p in range(len(params)-1)]	

		IMF = float(a[0])
		Z = a[1]
		if Z[0] == "p":
			Z = float(Z[1::])
		elif Z[0] == "m":
			Z = -1.e0*float(Z[1::])
		age = float(a[2])
		alpha = float(a[3])

		template_params.append([age,Z,alpha])
	template_params = np.array(template_params)

	temp_sort = np.lexsort((template_params[:,2],		#Sort by alpha, then Z, then age
							template_params[:,1],
							template_params[:,0]))
	template_params = template_params[temp_sort]
	files = np.array(files)[temp_sort]
	
	Ages = np.unique(template_params[:,0])
	Nages = len(Ages)
	Zs = np.unique(template_params[:,1])
	NZs = len(Zs)
	Alphas = np.unique(template_params[2])
	Nalphas = len(Alphas)

	for ff in range(len(files)):						#Read templates in sorted order
		file = files[ff]
		hdu = fits.open(file)
		template = hdu[0].data
		header = hdu[0].header
		hdu.close()

		if ff == 0:										#Get spectral parameters from first template
			wave_step = header['CDELT1']
			lambda0 = header['CRVAL1']
			linLambda_templates = lambda0 + np.arange(header['NAXIS1'])*header['CDELT1']
			templates = np.zeros([header['NAXIS1'],len(files)])
			templates = templates[(linLambda_templates>=4000) & (linLambda_templates<10000),:]			#We arent covering more than this range
			
			linLambda_templates_trunc = linLambda_templates[(linLambda_templates>=4000) & (linLambda_templates<10000)]

		
		template = template[(linLambda_templates>=4000) & (linLambda_templates<10000)]

		templates[:,ff] =  template

	return templates, linLambda_templates_trunc, wave_step

def read_MILES():
	"""
	Reads MILES spectral templates from a directory, which should be in the same level as this file
	
	Parameters
	----------
	Nothing

	Returns
	-------
	templates : array
		Stellar templates sorted as column vectors
	linLambda_templates_trunc : array
		Linear spectral axis of the templates
	wave_step : float
		wavelength channel resolution
		
	"""

	#Identify template files
	template_dir = f"{os.path.dirname(os.path.realpath(__file__))}/templates/MILES"
	files = sorted(glob.glob(template_dir + "/*"))


	params = ["ch","Z","T","_iTp","_"] 				#Possible template parameters to identify
	template_params = []
	for ff in range(len(files)):					#Loop over files to get the parameters
		#Split filename to get parameters
		a = [files[ff].split("/")[-1].split(params[p],maxsplit=1)[1].split(params[p+1],maxsplit=1)[0]
						for p in range(len(params)-1)]

		IMF = float(a[0])
		Z = a[1]
		if Z[0] == "p":
			Z = float(Z[1::])
		elif Z[0] == "m":
			Z = -1.e0*float(Z[1::])
		age = float(a[2])
		alpha = float(a[3])

		template_params.append([age,Z,alpha])
	template_params = np.array(template_params)

	temp_sort = np.lexsort((template_params[:,2],	#Sort by alpha, then Z, then age
							template_params[:,1],
							template_params[:,0]))
	template_params = template_params[temp_sort]
	files = np.array(files)[temp_sort]
	
	Ages = np.unique(template_params[:,0])
	Nages = len(Ages)
	Zs = np.unique(template_params[:,1])
	NZs = len(Zs)
	Alphas = np.unique(template_params[2])
	Nalphas = len(Alphas)

	for ff in range(len(files)):					#Read templates in sorted order

		file = files[ff]
		hdu = fits.open(file)
		template = hdu[0].data
		header = hdu[0].header
		hdu.close()

		if ff == 0:									#Get spectral parameters from first template
			wave_step = header['CDELT1']
			lambda0 = header['CRVAL1']
			linLambda_templates = lambda0 + np.arange(header['NAXIS1'])*header['CDELT1']
			templates = np.zeros([header['NAXIS1'],len(files)])
			templates = templates[(linLambda_templates>=4000) & (linLambda_templates<7300),:]
			
			linLambda_templates_trunc = linLambda_templates[(linLambda_templates>=4000) & (linLambda_templates<7300)]

		template = template[(linLambda_templates>=4000) & (linLambda_templates<7300)]

		templates[:,ff] =  template

	return templates, linLambda_templates_trunc, wave_step

def read_BPASS(mode = None):
	"""
	Need to document!
	"""
	template_dir = f"{os.path.dirname(os.path.realpath(__file__))}//templates/BPASS"

	files = np.array(glob.glob(template_dir + "/*"))
	Zs = np.zeros(len(files))
	for ff, file in enumerate(files):
		z = file.split('.dat')[0].split('.z')[-1]
		if 'em' in z:
			z = z.split('em')[-1]
			z = 1.*10**(-1*float(z))
		else:
			z = 1.e-3 * float(z)
		Zs[ff] = z
	files = files[Zs.argsort()]
	Zs = Zs[Zs.argsort()]
	alphas = np.zeros(len(Zs))
	agesAll = np.arange(6,11.1,0.1)
	agesFlags = (agesAll<=10.2)


	wave_step = 1

	if isinstance(mode,type(None)):
		mode = 'short'		

	if mode == 'emshort':
		agesList = np.array([7.4,7.6,7.8,8,9.2,9.5,9.7,9.9,10.2])
		timeSteps = np.sum([np.isclose(agesAll,aa) for aa in agesList],axis=0,dtype=bool) * agesFlags 
		zList = np.array([1.e-3,6.e-3,1.4e-2,4.e-2])
		refs = np.sum([np.isclose(Zs,zz) for zz in zList],axis=0,dtype=bool)
		files = files[refs]
		Zs = Zs[refs]
		alphas = alphas[refs]
		ages = agesAll[timeSteps]

	if mode == 'short':
		timeSteps = ~np.array((np.arange(len(agesAll)))%2,dtype=bool) * agesFlags		#0.2dex age steps
		zList = np.array([1.e-3,6.e-3,1.4e-2,4.e-2])
		refs = np.sum([np.isclose(Zs,zz) for zz in zList],axis=0,dtype=bool)
		files = files[refs]
		Zs = Zs[refs]
		alphas = alphas[refs]
		ages = agesAll[timeSteps]

	if mode == 'pops':
		timeSteps = ~np.array((np.arange(len(agesAll)))%2,dtype=bool)					#0.2dex age steps
		zList = np.array([1.e-4,1.e-3,2.e-3,4.e-3,8.e-3,1.4e-2,2.e-2,4.e-2])
		refs = np.sum([np.isclose(Zs,zz) for zz in zList],axis=0,dtype=bool)
		files = files[refs]
		Zs = Zs[refs]
		alphas = alphas[refs]
		ages = agesAll[timeSteps]

	elif mode == 'full':
		timeSteps = np.array(np.ones(len(ages)),dtype=bool) * agesFlags					#0.1dex age steps

	timeSteps = np.append(np.array([False]),timeSteps)

	Nspec = len(ages)

	for ff in range(len(files)):

		file = files[ff]
		data = np.loadtxt(file)

		if ff == 0:
			linLambda_templates = data[:,0]

			wave_shorten = np.logical_and((linLambda_templates>=4000), (linLambda_templates<10000))

			linLambda_templates_trunc = linLambda_templates[wave_shorten]
			templates = np.zeros([linLambda_templates_trunc.shape[0],Nspec*len(files)])

		templates[:,ff*Nspec:(ff+1)*Nspec] = data[wave_shorten,:][:,timeSteps]



	return  templates, linLambda_templates_trunc,  wave_step

def make_gas_templates(parameters, logLambda_templates, convolve = True, pixel = True):
	"""
	Makes gas emission line templates

	Parameters
	----------
	parameters : dict
		Dictionary of parameters
	logLambda_templates : array
		Spectra axis of stellar templates
	convolve : bool
		Smooth the templates to match the spectral resolution of the data
	pixel : bool
		Implement pPXF's pixel integration 

	Returns
	-------
	gas_templates : 2D array
		Column vector of gas emission line templates

	"""

	gas_templates = []

	if convolve:								#Get linespread function of the MUSE data
		FWHM = lambda ll: MUSE_LSF_Bacon17(ll, parameters['z'])
	else:
		FWHM = 0

	gas_groups = parameters['gas_groups']
	for gg, group in enumerate(gas_groups):		#Loop over gas emission lines and lien groups
		for gl, gas_line in enumerate(group):
			gas_lambda = emlines[gas_line]['lambda']
			gas_ratio = emlines[gas_line]['ratio']
			line_template = pputils.gaussian(logLambda_templates,
												np.asarray(gas_lambda),
												FWHM,
												pixel = pixel) \
												@ gas_ratio
			gas_templates.append(line_template)
	gas_templates = np.array(gas_templates).T

	return gas_templates

def create_spectrum_mask(logLambda, parameters):
	"""
	Make a list of spectrum channels that should be included in the fit

	Parameters
	----------
	logLambda : array
		Spectral axis
	parameters : dict
		Dictionary of parameters

	Returns
	-------
	goodpix : array
		Array of good pixels to be included in the fit

	"""

	goodpix = np.zeros_like(logLambda,dtype=bool)
	z = parameters['z']
	linLambda = np.exp(logLambda)					#convert to linear channels

	width = 400. / c								#set the window to mask around lines

	gas_fit = False
	if not isinstance(parameters['gas_groups'],type(None)):
		gas_fit = True



	sky = np.array([5577.338, 6300.304, 6363.78])/(1.e0+z)					# OI sky emission lines
																		
	absorption_lines = np.array([5889.95, 5895.92, 8498., 8542., 8662.]) 	# Stellar absorpton lines that can have ISM contribution  NaD CaT
	if parameters['fit_CaT'] == True:										#Remove masks if fitting the CaT triplet
		absorption_lines = absorption_lines[:2]

	if gas_fit:
		emission_lines_fit = [line for group in parameters['gas_groups'] for line in group]
		emission_lines_fit = np.unique(np.array(emission_lines_fit))
	else:
		emission_lines_fit = None

	emlines_temp = copy.deepcopy(emlines)									#List of emission lines to mask out
	if parameters['fit_CaT'] == True:										#Dont mask emission lines that overlap with CaT if fitting it
		del emlines_temp['Pa13']
		del emlines_temp['Pa15']
		del emlines_temp['Pa16']

	emission_lines = np.array([])
	for line in emlines_temp:												#Loop over emission lines but don't mask regions we are fitting
		if not gas_fit:
			emission_lines = np.append(emission_lines,
								np.array(emlines_temp[line]['lambda']))
		elif line not in emission_lines_fit:
			emission_lines = np.append(emission_lines,
								np.array(emlines_temp[line]['lambda']))

	lines = np.concatenate((emission_lines,absorption_lines,sky),axis=None)		#Combine all lines to mask
	in_spec = np.logical_and(lines>=np.min(linLambda), lines<=np.max(linLambda))	#Only choose lines in the wavelength range
	lines = lines[in_spec]
	for line in lines:
		min_Lambda = line - line*width
		max_Lambda = line + line*width
		goodpix += np.logical_and(linLambda>=min_Lambda, linLambda<= max_Lambda)
	
	#Mask the telluric feature
	min_Lambda = (7650 / (1+z)) * (1-2500/c) 
	max_Lambda = (7650 / (1+z)) * (1+2500/c) 
	goodpix += np.logical_and(linLambda>=min_Lambda, linLambda<= max_Lambda)

	goodpix = ~goodpix

	return goodpix

def get_constraints(parameters, spaxel_properties, vorbin_nums):
	"""
	Read kinematic constraints

	Parameters
	----------
	parameters : dict
		Dictionary of parameters
	spaxel_properties : astropy.Table
		Table of pixel properties
	vorbin_nums : array
		List of bin numbers

	Returns
	-------
	start : list
		List of fit input starting guesses
	moments : list
		List of kinematic moments to fit for each kinematic component
	constraints : dict
		Dictionary of kinematic constraints
	"""

	#Read previously fit stellar kinematics if kinematics are being fixed
	if 'stelkin_constr' in parameters.keys():
		stel_kin_file = f"{parameters['output']}/{parameters['stelkin_constr']}/bestfit_kinematics_fluxes.fits"
		stel_kin = Table.read(stel_kin_file)
		print('Found stelkin constraints...')
	else:
		stel_kin = None

	constraints = []
	start = []
	moments = []
	for vv, vb in enumerate(vorbin_nums):				#Loop over each spatial bin

		constr_kinem = {'A_ineq':[],'b_ineq':[]}

		refs = np.where(spaxel_properties['vorbin_num'] == vb)[0]
		fixed_flag = 0
		if not isinstance(stel_kin,type(None)):
			stel_kin_vorbin = stel_kin[refs]

			V_max =  np.max(stel_kin_vorbin['V_stellar'])				#Bound stellar kinematics by all pixel values in bin if e.g. you've re-binned on top of other data
			V_min =  np.min(stel_kin_vorbin['V_stellar'])
			sigma_max =  np.max(stel_kin_vorbin['sigma_stellar'])
			sigma_min =  np.min(stel_kin_vorbin['sigma_stellar'])
			h3_max =  np.max(stel_kin_vorbin['h3_stellar'])
			h3_min =  np.min(stel_kin_vorbin['h3_stellar'])
			h4_max =  np.max(stel_kin_vorbin['h4_stellar'])
			h4_min =  np.min(stel_kin_vorbin['h4_stellar'])

			#If max and min are the same fix the parameter, otherwise give reasonable bounds
			fixed_flag = 0
			if V_max != V_min:			
				if V_min == 0:
					V_min -= 10
				if V_max == 0:
					V_max += 10
				constr_kinem['A_ineq'].append( [1,0,0,0] + [0,0]*len(parameters['gas_Ncomp']) )
				constr_kinem['b_ineq'].extend([ V_max / parameters['galaxy_velscale'] ])

				constr_kinem['A_ineq'].append( [-1,0,0,0] + [0,0]*len(parameters['gas_Ncomp']) )
				constr_kinem['b_ineq'].extend( [ -1.e0* V_min / parameters['galaxy_velscale'] ])
			else:
				fixed_flag += 1
			if sigma_max != sigma_min:
				if sigma_min == 0:
					sigma_min = 10
				if sigma_max == 0:
					sigma_max += 10
				constr_kinem['A_ineq'].append([0,1,0,0] + [0,0]*len(parameters['gas_Ncomp']) )
				constr_kinem['b_ineq'].extend([ sigma_max / parameters['galaxy_velscale'] ])
				constr_kinem['A_ineq'].append([0,-1,0,0] + [0,0]*len(parameters['gas_Ncomp']) )
				constr_kinem['b_ineq'].extend([ -1.e0* sigma_min / parameters['galaxy_velscale'] ])
			else:
				fixed_flag += 1
			if h3_max != h3_min:
				if h3_min == 0:
					h3_min -= 2
				if h3_max == 0:
					h3_max += 2
			else:
				fixed_flag += 1
			if h4_max != h4_min:
				if h4_min == 0:
					h4_min -= 2
				if h4_max == 0:
					h4_max += 2
			else:
				fixed_flag += 1
			start_vorbin = [np.nanmean(stel_kin_vorbin['V_stellar']), np.nanmean(stel_kin_vorbin['sigma_stellar']),np.nanmean(stel_kin_vorbin['h3_stellar']),np.nanmean(stel_kin_vorbin['h4_stellar'])]
		else:
			start_vorbin = parameters['stars_start']


		if not isinstance(parameters['gas_constraints'],type(None)):		#Read gas parameter file constraints
			gas_constraints = parameters['gas_constraints']
			for aa in range(len(gas_constraints['A_ineq'])):
				constr_kinem['A_ineq'].append( [0.,0.,0.,0.] + gas_constraints['A_ineq'][aa] )
				constr_kinem['b_ineq'].extend( [ gas_constraints['b_ineq'][aa] / parameters['galaxy_velscale'] ])

		
		if constr_kinem['A_ineq'] == []:									#Identify if no constraints are being used
			constraints.append(None)
		else:
			constraints.append(constr_kinem)

		#Get starting estimates
		if len(parameters['gas_Ncomp'])>0:
			start_vorbin = [start_vorbin] + [[0,50.]] 
		if len(parameters['gas_Ncomp'])>1:
			start_vorbin = start_vorbin + [[0,300]]*(len(parameters['gas_Ncomp'])-1)
		start.append(start_vorbin)

		#Combine all moments for the bin
		moments_vorbin = [parameters['stars_moments']] + [2] * len(parameters['gas_Ncomp'])
		
		if fixed_flag==4:			#Flag stellar kinematics as fixed
			moments_vorbin[0] *= -1
		moments.append(moments_vorbin)

	print('Starting values and kinematic constraints created')

	return start, moments, constraints




##### Things for analysing outputs, not so much part of the main pipeline #####

def make_continuum_subtracted_spectra(parameterfile):
	"""
	Subtracts the best fit spectra, where the fit is done on spatially binned data, from individual pixel spectra

	Parameters
	----------
	parameters : dict
		Dictionary of parameters

	Returns
	-------
	Nothing
		Writes spectra to file
	"""

	print("Doing continuum subtraction")
	sys.stdout.flush()

	#Read parameters and properties
	parameters = read_parameterfile(parameterfile)
	spax_properties = Table.read(f"{parameters['input_dir']}/spaxel_properties.fits")
	Nx = spax_properties.meta['NX']
	Ny = spax_properties.meta['NY']

	#Read observed and fit spectra
	print("...Reading data")
	sys.stdout.flush()
	spectra_file = f"{parameters['output']}/indiv/logRebin_spectra.fits"
	hdul = fits.open(spectra_file)
	logLambda_obs, spectra = read_spectra_hdul(hdul)
	hdul.close()
	galaxy_spectra = spectra[0]
	noise_spectra = spectra[1]

	bestfit_continuum_file = f"{parameters['output_dir']}/logRebin_bestfit_spectra.fits"
	hdul = fits.open(bestfit_continuum_file)
	logLambda_fit, spectra = read_spectra_hdul(hdul)
	hdul.close()
	bestfit_continuum = spectra[0]
	spectra = None


	#Trim observed data to fit range
	print("...Trimming data")
	sys.stdout.flush()

	galaxy_spectra = galaxy_spectra[np.asarray(logLambda_obs<=logLambda_fit[-1]),:]
	noise_spectra = noise_spectra[np.asarray(logLambda_obs<=logLambda_fit[-1]),:]
	logLambda_obs = logLambda_obs[np.asarray(logLambda_obs<=logLambda_fit[-1])]
	linLambda_obs = np.exp(logLambda_obs)				#convert to linear spectral axis
	print("...Done reading data")
	sys.stdout.flush()


	#Set up arrays for continuum subtracted data
	Nl = logLambda_obs.shape[0]
	contsub_cube = copy.deepcopy(np.zeros([Nl,Nx*Ny]))
	contsub_spectra = np.zeros(galaxy_spectra.shape)

	#Propagate through pixels that were not fit
	notBinned = np.where(spax_properties['vorbin_num'].data < 0)[0]
	contsub_spectra[:,notBinned] = galaxy_spectra[:,notBinned]

	#Define S/N range for rescaling spectra
	SN_cont_Lmin = spax_properties.meta['SN_LMIN']
	SN_cont_Lmax = spax_properties.meta['SN_LMAX']
	SN_cont_Lrange = np.logical_and(linLambda_obs>=SN_cont_Lmin,linLambda_obs<=SN_cont_Lmax)

	print("...Looping over spectra")
	sys.stdout.flush()
	for vv in range(bestfit_continuum.shape[1]):			#Loop over bins
		
		fitSpec = bestfit_continuum[:,vv]
		fitSignal = np.nanmedian(fitSpec[SN_cont_Lrange])

		inbin = np.where(spax_properties['vorbin_num'].data == vv)[0]

		for ss in inbin:
			obsSpec = galaxy_spectra[:,ss]
			obsSignal = np.nanmedian(obsSpec[SN_cont_Lrange])

			fitSpec_scaled = fitSpec * obsSignal / fitSignal			#Scale fit by ratio of emission/signal in binned and individual spectra

			contsub_spectra[:,ss] = obsSpec - fitSpec_scaled
	print("...Done")
	sys.stdout.flush()

	#Reshape into a datacube
	contsub_cube[:,np.asarray(spax_properties['spax_num'].data,dtype=int)] = contsub_spectra
	contsub_cube = contsub_cube.reshape(Nl,Ny,Nx)
	

	#Save continuum-subtracted spectra and datacube
	header = fits.Header()
	header['COMMENT'] = "A.B. Watts"
	header['COMMENT'] = "LOG LAMBDA IS IN THIS HDU "
	primary_hdu = fits.PrimaryHDU(data = logLambda_obs,
									header = header)
	spectra_hdu = fits.BinTableHDU.from_columns(
							fits.ColDefs([
							fits.Column(
							array = contsub_spectra.T,
							name='CONTSUB',format=str(len(contsub_spectra))+'D'
							)]))
	noise_hdu = fits.BinTableHDU.from_columns(
							fits.ColDefs([
							fits.Column(
							array = noise_spectra.T,
							name='NOISE',format=str(len(noise_spectra))+'D'
							)]))

	hdul = fits.HDUList([primary_hdu,
								spectra_hdu, noise_hdu])
	print("Saving continuum-subtracted spectra table")
	hdul.writeto(f"{parameters['output_dir']}/logRebin_contsub_spectra.fits",overwrite=True)
	print("Saved")


	cube_file = parameters['datacube']
	cube_hdu = fits.open(cube_file)
	cube_header = cube_hdu[1].header
	cube_hdu.close()
	cube_header['CRVAL3'] = 1
	cube_header['CRPIX3'] = 1
	print(cube_header.keys())
	if any(key == "CDELT3" for key in cube_header.keys()):
		cube_header['CDELT3'] = 1
	else:
		cube_header['CD3_3'] = 1

	cube_header['CTYPE3'] = "Chan"
	header = fits.Header()
	header['COMMENT'] = "A.B. Watts"
	header['COMMENT'] = "LOG LAMBDA IS IN THIS HDU "
	primary_hdu = fits.PrimaryHDU(header = header,data = logLambda_obs)


	contsub_cube_hdu = fits.ImageHDU(contsub_cube,
							name='CONTSUB',
							header = cube_header)
	hdul = fits.HDUList([primary_hdu,
								contsub_cube_hdu])#,noise_cube_hdu])
	print("Saving continuum-subtracted cube")
	hdul.writeto(f"{parameters['output_dir']}/contsub_cube.fits",overwrite=True)
	print("Saved")

def make_line_subcubes(parameterfile):
	"""
	Extracts sub-cubes around emission lines from a continuum-subtracted cube

	Parameters
	----------
	parameters : dict
		Dictionary of parameters

	Returns
	-------
	Nothing
		Writes spectra to file
	"""

	#Lines to extract
	lines = {'Hbeta':{	'lambda':[4861.333]	},
		'OIII4959':{	'lambda':[4958.911]	}, 
		'OIII5006':{	'lambda':[5006.843]	}, 
		'HeI5876':{	'lambda':[5875.624]		}, 
		'OI6300':{		'lambda':[6300.304]	},
	 	'NII6548':{		'lambda':[6548.050]	},
	 	'NII6583':{		'lambda':[6583.460]	},
		'Halpha':{	'lambda':[6562.819]		},
		'HeI6678':{	'lambda':[6678.151]		}, 
		'SII6716':{	'lambda':[6716.440]		},
		'SII6730':{	'lambda':[6730.810]		}
		}


	print("Making emission-line subcubes")
	sys.stdout.flush()

	#Read parameters and pixel properties
	parameters = read_parameterfile(parameterfile)
	spax_properties = Table.read(f"{parameters['input_dir']}/spaxel_properties.fits")
	Nx = spax_properties.meta['NX']
	Ny = spax_properties.meta['NY']

	#Get continuum-subtracted spectra
	spectra_file = f"{parameters['output_dir']}/logRebin_contsub_spectra.fits"
	hdul = fits.open(spectra_file)
	logLambda, spectra = read_spectra_hdul(hdul)
	linLambda = np.exp(logLambda)
	hdul.close()
	contsub_spectra = spectra[0]
	noise = spectra[1]
	spectra = None
	Nl = contsub_spectra.shape[1]

	#Get header template
	cube_file = parameters['datacube']
	cube_hdu = fits.open(cube_file)
	cube_header = cube_hdu[1].header
	cube_hdu.close()

	#Set up directory
	if not os.path.isdir(f"{parameters['output_dir']}/linecubes"):
		os.mkdir(f"{parameters['output_dir']}/linecubes")

	#Subcube parameters
	clipping_width = 1500 / c
	subcube_width = 500 / c
	for ll, line in enumerate(lines):		#Loop over emission lines

		#Get line range for subcube
		lineLambda = lines[line]['lambda'][0]
		subcube_range =  np.where((logLambda > np.log(lineLambda*(1 - subcube_width)))  & 
								(logLambda < np.log(lineLambda*(1 + subcube_width))))[0]
		clipping_range = np.where((logLambda > np.log(lineLambda*(1 - clipping_width)))  & 
								(logLambda < np.log(lineLambda*(1 + clipping_width))))[0]
		logLambda_subcube = logLambda[subcube_range]
		subcube_spectra = np.full(contsub_spectra[subcube_range,:].shape,np.nan)
		subcube_noise = noise[subcube_range,:]


		for ii in range(contsub_spectra.shape[1]):	#Loop over spectra

			#Sigma-clip spectrum and fit and subtract a polynomial to remove any residual continuum
			logLambda_clipRange = logLambda[clipping_range]
			unclippedSpectrum = contsub_spectra[clipping_range,ii]
			clippedSpectrum = sigma_clip(unclippedSpectrum,2.,2.)
			fit,covar = curve_fit(baselineFunc,
								logLambda_clipRange[~clippedSpectrum.mask],
								clippedSpectrum[~clippedSpectrum.mask])
			spectrum = contsub_spectra[subcube_range,ii]
			spec_noise = noise[subcube_range,ii]
			spectrum = spectrum - baselineFunc(logLambda_subcube,fit[0],fit[1],fit[2])

			#Account for neaby lines
			if line == "OIII5006":	
				line_min_logLambda = np.min(logLambda_subcube)
				line_max_logLambda = np.log(0.5*(lineLambda + emlines_indiv['Fe5018']['lambda'][0])) 
				logLambda_window = np.logical_and(logLambda_subcube > line_min_logLambda,
								logLambda_subcube  < line_max_logLambda)
				spectrum[~logLambda_window] = 0
			elif line == "Halpha":	
				line_min_logLambda = np.log(0.5*(lineLambda + 6548.050))
				line_max_logLambda = np.log(0.5*(lineLambda + 6583.460)) 
				logLambda_window = np.logical_and(logLambda_subcube > line_min_logLambda,
								logLambda_subcube  < line_max_logLambda)
				spectrum[~logLambda_window] = 0
			elif line == "NII6548":	
				line_min_logLambda = np.min(logLambda_subcube)
				line_max_logLambda = np.log(0.5*(lineLambda + 6562.819))
				logLambda_window = np.logical_and(logLambda_subcube > line_min_logLambda,
								logLambda_subcube  < line_max_logLambda)
				spectrum[~logLambda_window] = 0
			elif line == "NII6583":	
				line_min_logLambda = np.log(0.5*(lineLambda + 6562.819)) 
				line_max_logLambda = np.max(np.log(np.exp(logLambda_subcube)))
				logLambda_window = np.logical_and(logLambda_subcube > line_min_logLambda,
								logLambda_subcube  < line_max_logLambda)
				spectrum[~logLambda_window] = 0

			elif line == "SII6716":	
				line_min_logLambda = np.min(logLambda_subcube)
				line_max_logLambda = np.log(0.5*(lineLambda + 6730.810)) 
				logLambda_window = np.logical_and(logLambda_subcube > line_min_logLambda,
								logLambda_subcube  < line_max_logLambda)
				spectrum[~logLambda_window] = 0
			elif line == "SII6730":	
				line_min_logLambda = np.log(0.5*(lineLambda + 6716.440)) 
				line_max_logLambda = np.max(logLambda_subcube)
				logLambda_window = np.logical_and(logLambda_subcube > line_min_logLambda,
								logLambda_subcube  < line_max_logLambda)
				spectrum[~logLambda_window] = 0

			
			subcube_spectra_SN =  spectrum/ spec_noise			#make an S/N spectrum
			mask1 = np.zeros_like(subcube_spectra_SN)
			mask2 = np.zeros_like(subcube_spectra_SN)


			for cc in range(1,len(subcube_spectra_SN)-1):		
				if subcube_spectra_SN[cc] >=3.5:				#Narrow mask, all channels >3 sigma (line detection)
					mask1[cc] = 2
				if subcube_spectra_SN[cc] >= 1.5:				#Broad mask, those with >1.5 sigma in >2 contiguous channels
					if subcube_spectra_SN[cc-1] >=1.5 and subcube_spectra_SN[cc+1] >=1.5 :
						mask2[cc] = 1
						mask2[cc-1] = 1
						mask2[cc+1] = 1

			mask_tot = mask1+mask2
			mask_segments = []
			seg = []

			for mm, val in enumerate(mask_tot):					#Keep only broad mask segments that also have a narrow mask
				if val == 0 or mm == len(mask_tot)-1:
					if len(seg) != 0:
						mask_segments.append(seg)
						seg = []

				if val != 0:
					seg.extend([mm])
			maxlen = 0
			mask_final = np.zeros_like(mask_tot)

			for seg in mask_segments:							#Keep only the largest mask segment
				if np.any(mask_tot[seg] >= 2):
					if len(seg)> maxlen:
						mask_final = np.zeros_like(mask_tot)
						maxlen = len(seg)
						mask_final[seg] = 1

			subcube_spectra[:,ii] = spectrum*mask_final			#Mask emission line


		#Save subcube spectra
		spectra_header = fits.Header()
		spectra_header['COMMENT'] = "A.B. Watts"
		spectra_header['COMMENT'] = "LOG LAMBDA IS IN THIS HDU "
		primary_hdu = fits.PrimaryHDU(data = logLambda_subcube,
									header = spectra_header)

		spectra_hdu = fits.BinTableHDU.from_columns(
								fits.ColDefs([
								fits.Column(
								array = subcube_spectra.T,
								name=line,format=str(len(subcube_spectra))+'D'
								)]))
		sigma_hdu = fits.BinTableHDU.from_columns(
								fits.ColDefs([
								fits.Column(
								array = subcube_noise.T,
								name='sigma',format=str(len(subcube_noise))+'D'
								)]))

		hdul = fits.HDUList([primary_hdu,
								spectra_hdu,sigma_hdu])
		print(f"Saving {line} spectra table")
		hdul.writeto(f"{parameters['output_dir']}/linecubes/{line}_subspectra.fits",overwrite=True)
		print("Saved")
		

		#Save as a cube for visualisation
		cube_header['NAXIS3'] = len(logLambda_subcube)
		cube_header['CRVAL3'] = 1
		cube_header['CRPIX3'] = 1
		if any(key == "CDELT3" for key in cube_header.keys()):
			cube_header['CDELT3'] = 1
		else:
			cube_header['CD3_3'] = 1

		Nl = subcube_spectra.shape[0]

		subcube_cube = np.zeros([Nl,Nx*Ny])
		if parameters['contsub_level'] == 'spax':
			subcube_cube[:,np.array(spax_properties['spax_num'],dtype=int)] = subcube_spectra
		elif parameters['contsub_level'] == 'bin':
			for vv in range(subcube_spectra.shape[1]):
				inbin = np.where(spax_properties['vorbin_num']==vv)[0]
				spax_num_inbin = np.array(spax_properties['spax_num'].data[inbin],dtype=int)
				subcube_cube[:,spax_num_inbin] = np.full((len(inbin),Nl),subcube_spectra[:,vv]).T

		subcube_cube = subcube_cube.reshape(Nl,Ny,Nx)

		# plt.imshow(np.nansum(subcube_cube,axis=0))
		# plt.show()
		header = fits.Header()
		header['COMMENT'] = "A.B. Watts"
		header['COMMENT'] = "LOG LAMBDA IS IN THIS HDU "
		primary_hdu = fits.PrimaryHDU(header = header,data=logLambda_subcube)
		line_cube_hdu = fits.ImageHDU(subcube_cube,
								name=line,
								header = cube_header)
		hdul = fits.HDUList([primary_hdu,
									line_cube_hdu])
		print(f"Saving {line} linecube")
		hdul.writeto(f"{parameters['output_dir']}/linecubes/{line}_linecube.fits",overwrite=True)
		print("Saved")

def measure_linecubes(parameterfile):
	"""
	Measures line fluxes and kinematic moments from sub-cubes

	Parameters
	----------
	filename : dict
		linecube name

	Returns
	-------
	kin
		Emission line kinematics
	names
		Measured column names 
	"""
	print("Measuring fluxes and kinematics for each emission-line subcube")
	sys.stdout.flush()

	#Get parametes and pixel properties
	parameters = read_parameterfile(parameterfile)
	spax_properties = Table.read(f"{parameters['input_dir']}/spaxel_properties.fits")

	#Use all subcubes in working directory
	subspectra = glob.glob(f"{parameters['output_dir']}/linecubes/*_subspectra.fits")
		

	F_names = []
	F_measures = []
	Kin_names = []
	Kin_measures = []

	for subspectra_file in subspectra:			#loop over sub-cubes

		fluxes, names = measure_linecube_fluxes(subspectra_file)
		F_names.extend(names)
		F_measures.extend(fluxes)
			
		kin, names = measure_linecube_kinematics(subspectra_file)
		Kin_names.extend(names)
		Kin_measures.extend(kin)


	F_measures = np.array(F_measures).T
	Kin_measures = np.array(Kin_measures).T

	#Set up arrays for if we are working on the pixel or bin level
	F_measures_expanded = np.zeros([len(spax_properties),F_measures.shape[1]])
	Kin_measures_expanded = np.zeros([len(spax_properties),Kin_measures.shape[1]])
	for bb in range(F_measures.shape[0]):
		if parameters['contsub_level'] == 'spax':			#assign on the pixel level
			F_measures_expanded[bb,:] = F_measures[bb,:]
			Kin_measures_expanded[bb,:] = Kin_measures[bb,:]
		elif parameters['contsub_level'] == 'bin':			#assign on the bin level
			inbin = np.in1d(spax_properties['vorbin_num'].data,bb)
			F_measures_expanded[inbin,:] = F_measures[bb,:]
			Kin_measures_expanded[inbin,:] = Kin_measures[bb,:]


	#save outputs
	fluxes_table = Table(F_measures_expanded,names=F_names)
	kin_table = Table(Kin_measures_expanded,names=Kin_names)
	fluxes_table.write(f"{parameters['output_dir']}/spax_emline_fluxes.fits",overwrite=True)
	kin_table.write(f"{parameters['output_dir']}/spax_emline_kin.fits",overwrite=True)

def measure_linecube_fluxes(filename):
	"""
	Measures line fluxes from sub-cubes

	Parameters
	----------
	filename : dict
		linecube name

	Returns
	-------
	fluxes
		Emission line fluxes and uncertainties for each spectrum
	names
		Measured column names names
	"""

	#Get data
	hdul = fits.open(filename)
	logLambda, spectra_list = read_spectra_hdul(hdul)
	spectra = spectra_list[0]
	noise = spectra_list[1]
	hdul.close()

	#Spectral resolution for integration
	linLambda = np.exp(logLambda)
	diff_lambda = np.append(np.diff(linLambda)[0]-np.diff(np.diff(linLambda))[0],
							np.diff(linLambda))


	#Get line name
	line = filename.split('/')[-1].split('_subspectra')[0]
	names = [line,f"{line}_err"]


	#Set up arrays	
	Fint = np.zeros([spectra.shape[1]])
	Ferr = np.zeros([spectra.shape[1]])

	for ii in range(spectra.shape[1]):			#Loop over spectra
		spectrum = spectra[:,ii]
		sigma = noise[:,ii]
		spectrum_mask = spectrum > 0

		Fint[ii] = np.nansum(spectrum*diff_lambda)				#Integrate masked signal
		Ferr[ii] = np.nansum((sigma*spectrum_mask)*diff_lambda)	#Integrate masked noise

		if Fint[ii] == 0:				#set non-detection to 1 sigma assuming a 3 channel line
			Fint[ii] = -99
			# 3 closest channels to rest wavelength of line
			loc = np.sort(np.argsort(np.abs(linLambda-emlines_indiv[line]['lambda'][0]))[0:3])+1
			Ferr[ii] = np.nansum(sigma[loc]*diff_lambda[loc])
	
	fluxes = [Fint,Ferr]

	return  fluxes, names
def measure_linecube_kinematics(filename):
	"""
	Measures line kinematic moments from sub-cubes

	Parameters
	----------
	filename : dict
		linecube name

	Returns
	-------
	kin
		Emission line kinematics
	names
		Measured column names 
	"""

	#Get data
	hdul = fits.open(filename)
	logLambda, spectra_list = read_spectra_hdul(hdul)
	spectra = spectra_list[0]
	hdul.close()

	#Get line name
	line = filename.split('/')[-1].split('_subspectra')[0]
	names = [f"{line}_V",f"{line}_sigma"]

	#Set up arrays
	V = np.zeros([spectra.shape[1]])
	sigma = np.zeros([spectra.shape[1]])

	for ii in range(spectra.shape[1]):
		spectrum = spectra[:,ii]
		spectrum_mask = spectrum > 0

		d_logLambda = logLambda - np.log(emlines_indiv[line]['lambda']) 			#Offset from rest value

		if np.any(spectrum_mask):
			V[ii] = c*weighted_moment(d_logLambda,weights=spectrum,moment=1) 		#Velocity
			sigma[ii] = c*weighted_moment(d_logLambda,weights=spectrum,moment=2) 	#Dispersion
		else:
			V[ii] = np.nan
			sigma[ii] = np.nan
	
	kin = [V,sigma]

	return  kin, names

### Making maps
def make_vorbins_map(parameters):
	"""
	Saves a map of the spatial bins

	Parameters
	----------
	parameters : dict
		Dictionary of parameters

	Returns
	-------
	Nothing
		Writes image to file
	"""

	spax_prop_file = f"{parameters['output_dir']}/spaxel_properties.fits"
	spax_properties = Table.read(spax_prop_file)
	meta = spax_properties.meta
	NY = meta['NY']
	NX = meta['NX']

	img_grid = np.full([NY,NX],np.nan).flatten()
	img_grid[np.asarray(spax_properties['spax_num'],dtype=int)] = spax_properties['vorbin_num']
	img_grid = img_grid.reshape((NY,NX))

	fig = plt.figure()
	gs = gridspec.GridSpec(1,1)
	ax = fig.add_subplot(gs[0,0])
	ax.pcolormesh(img_grid,cmap='prism')
	fig.savefig(f"{parameters['output_dir']}/figures/vorbin_map.png")

def make_stelkin_map(parameters):
	"""
	Saves maps of the stellar kinematics

	Parameters
	----------
	parameters : dict
		Dictionary of parameters

	Returns
	-------
	Nothing
		Writes image to file
	"""
	spax_prop_file = f"{parameters['input_dir']}/spaxel_properties.fits"

	fit_stellar_kinematics = f"{parameters['output_dir']}/bestfit_kinematics_fluxes.fits"

	spax_properties = Table.read(spax_prop_file)
	bestfit_stelkin = Table.read(fit_stellar_kinematics)

	spax_properties = hstack((spax_properties,bestfit_stelkin))

	meta = spax_properties.meta
	NY = meta['NY']
	NX = meta['NX']

	img_grid = np.full([NY,NX],np.nan).flatten()
	vorbin_nums = np.unique(spax_properties['vorbin_num'])
	
	refs = np.zeros_like(vorbin_nums,dtype=int)
	for vv, vb in enumerate(vorbin_nums):
		refs[vv] = np.where(spax_properties['vorbin_num'] == vb)[0][0]

	#Loop over kinematic components
	for comp in ['V_stellar','sigma_stellar','h3_stellar','h4_stellar']:
		fig = plt.figure()
		gs = gridspec.GridSpec(1,1)
		ax = fig.add_subplot(gs[0,0])

		img = copy.deepcopy(img_grid)
		img[np.asarray(spax_properties['spax_num'],dtype=int)] = spax_properties[comp]
		img = img.reshape((NY,NX))


		if comp == 'V_stellar':
			med_vel = np.nanmedian(spax_properties[comp][refs])
			img -= med_vel
			vmin = np.percentile(spax_properties[comp][refs]-med_vel,5)
			vmax =np.percentile(spax_properties[comp][refs]-med_vel,95)
			cmap = 'RdBu_r'

			iimmgg = ax.pcolormesh(img,vmin=vmin,vmax=vmax,cmap=cmap)

		elif comp == 'sigma_stellar':
			vmin = 0
			vmax = 60
			cmap = 'inferno'

			iimmgg = ax.pcolormesh(img,vmin=vmin,vmax=vmax,cmap=cmap)

		else:
			vmin = 0.005
			vmax = -0.005
			cmap = 'inferno'
			iimmgg = ax.pcolormesh(img,vmin=vmin,vmax=vmax)

		ax.set_aspect('equal')

		fig.colorbar(iimmgg)
		fig.savefig(f"{parameters['output_dir']}/figures/{comp}_map.png")
		plt.close()

def make_linecube_fluxmaps(parameterfile):
	"""
	Saves maps of the emission line fluxes

	Parameters
	----------
	parameters : dict
		Dictionary of parameters

	Returns
	-------
	Nothing
		Writes image to file
	"""
	parameters = read_parameterfile(parameterfile)

	spax_properties = Table.read(f"{parameters['input_dir']}/spaxel_properties.fits")

	emline_fluxes = Table.read(f"{parameters['output_dir']}/spax_emline_fluxes.fits")

	spax_properties = hstack((spax_properties,emline_fluxes))

	metadata = spax_properties.meta
	Nx =  metadata['NX']
	Ny =  metadata['NY']

	img_grid = np.full((Ny,Nx),np.nan).flatten()
	line_names = ['A_V']+emline_fluxes.keys()[2::2]

	for ll, line in enumerate(line_names):
		linemap = copy.deepcopy(img_grid)
		linemap[np.array(spax_properties['spax_num'],dtype=int)] = spax_properties[f"{line}"]
		linemap = linemap.reshape((Ny,Nx))

		if f"{line}_err" in emline_fluxes.keys():
			line_errmap = copy.deepcopy(img_grid)
			line_errmap[np.array(spax_properties['spax_num'],dtype=int)] = spax_properties[f"{line}_err"]
			line_errmap = line_errmap.reshape((Ny,Nx))

		fig = plt.figure(figsize=(12,10))
		gs = gridspec.GridSpec(1,2,width_ratios = [1,0.05])


		ax1 = fig.add_subplot(gs[0,0])
		cb_ax = fig.add_subplot(gs[0,1])

		if line == 'A_V':
			img = ax1.pcolormesh(linemap,vmin=0,vmax=1)
		else:
			img = ax1.pcolormesh(np.log10(linemap),vmin=0.5,vmax=4)

		fig.colorbar(img,cax = cb_ax)
		cb_ax.set_title(f'{line}')

		ax1.set_aspect("equal")
		ax1.set_xlabel("pix",fontsize=15)
		ax1.set_ylabel("pix",fontsize=15)
		ax1.tick_params(which='both',axis='both',direction='in',labelsize=12)

		fig.savefig(f"{parameters['output_dir']}/figures/{line}_linemap.png")

		if f"{line}_err" in emline_fluxes.keys():


			fig = plt.figure(figsize=(12,10))
			gs = gridspec.GridSpec(1,2,width_ratios = [1,0.05])

			ax1 = fig.add_subplot(gs[0,0])
			cb_ax = fig.add_subplot(gs[0,1])


			img = ax1.pcolormesh(linemap/line_errmap,vmin=0,vmax=50)
			# ax1.pcolormesh(HAmap/HA_errmap,vmin=0.5,vmax=5)
			ax1.contour(linemap/line_errmap,levels=[5,10,30],colors=['Black','Blue','Red'])

			fig.colorbar(img,cax = cb_ax)
			cb_ax.plot(cb_ax.get_xlim(),[5,5],color='Black')
			cb_ax.plot(cb_ax.get_xlim(),[10,10],color='Grey')
			cb_ax.plot(cb_ax.get_xlim(),[30,30],color='Red')
			cb_ax.set_title(f'{line} log S/N')

			ax1.set_aspect("equal")
			ax1.set_xlabel("pix",fontsize=15)
			ax1.set_ylabel("pix",fontsize=15)
			ax1.tick_params(which='both',axis='both',direction='in',labelsize=12)

			fig.savefig(f"{parameters['output_dir']}/figures/{line}_SNmap.png")
			plt.close()

def make_linecube_kinmaps(parameterfile):
	"""
	Saves maps of the emission line kinematics

	Parameters
	----------
	parameters : dict
		Dictionary of parameters

	Returns
	-------
	Nothing
		Writes image to file
	"""
	parameters = read_parameterfile(parameterfile)

	spax_properties = Table.read(f"{parameters['input_dir']}/spaxel_properties.fits")

	emline_kin = Table.read(f"{parameters['output_dir']}/spax_emline_kin.fits")

	spax_properties = hstack((spax_properties,emline_kin))

	metadata = spax_properties.meta
	NX =  metadata['NX']
	NY =  metadata['NY']

	vorbin_nums = np.unique(spax_properties['vorbin_num'])
	refs = np.zeros_like(vorbin_nums,dtype=int)
	for vv, vb in enumerate(vorbin_nums):
		refs[vv] = np.where(spax_properties['vorbin_num'] == vb)[0][0]


	img_grid = np.full((NY,NX),np.nan).flatten()

	line_names = [name.split('_V')[0] for name in emline_kin.keys()[::2]]
	for ll, line in enumerate(line_names):
		print(line)
		for comp in [f'{line}_V',f'{line}_sigma']:

			img = copy.deepcopy(img_grid)
			img[np.asarray(spax_properties['spax_num'],dtype=int)] = spax_properties[comp]
			img = img.reshape((NY,NX))


			fig = plt.figure()
			gs = gridspec.GridSpec(1,1)
			ax = fig.add_subplot(gs[0,0])

			if "_V" in comp:
				good = np.isfinite(spax_properties[comp][refs])
				med_vel = np.nanmedian(spax_properties[comp][refs][good])
				img -= med_vel
				vmin = -np.percentile((spax_properties[comp][refs][good]-med_vel),95)
				vmax = np.percentile((spax_properties[comp][refs][good]-med_vel),95)
				cmap = 'RdBu_r'

				iimmgg = ax.pcolormesh(img,vmin=-80,vmax=80,cmap=cmap)

			elif "_sigma" in comp:
				# vmin = 0
				vmin = np.percentile(spax_properties[comp][refs][good],5)
				vmax = np.percentile(spax_properties[comp][refs][good],95)
				cmap = 'inferno'

				iimmgg = ax.pcolormesh(img,vmin=50,vmax=120,cmap=cmap)

			ax.set_aspect('equal')
			fig.colorbar(iimmgg,label='km/s')
			fig.savefig(f"{parameters['output_dir']}/figures/{comp}_map.png")
			plt.close()

	


#Useful things

def EBV_Hlines(F1 ,F2 ,lambda1 ,lambda2 , Rint = 2.83,k_l = None):
	#lambdas in angstrom
	#F1=HA F2 = HB (default)
	
	if isinstance(k_l,type(None)):
		k_l = lambda ll: extinction_curve(ll)

	ratio = np.log10((F1/F2) / Rint)

	kdiff = k_l(lambda2) - k_l(lambda1)

	E_BV = ratio / (0.4 * kdiff)
	# print(np.min(E_BV))

	E_BV[np.isfinite(E_BV)==False] = 0
	# print(np.min(E_BV))

	return E_BV

@np.vectorize
def extinction_curve(ll, RV = 4, extcurve = 'Calzetti00'):
	##ll should be in Angstrom
	
	if extcurve == 'Calzetti00':
		ll *= 1.e-4 		#convert to micron
		llinv = 1.e0/ll

		if  ll >= 0.12 and ll < 0.63:
			k = 2.659*(-2.156 + 1.509*llinv - 0.196*(llinv*llinv) + 0.011*(llinv*llinv*llinv)) + RV

		elif ll >=0.63 and ll <=2.20:
			k = 2.659*(-1.857 + 1.040*llinv) + RV
		else:
			k = np.nan

	return k

def create_spectrum_from_weights(parameterfile, weights, templates):

		parameters = read_parameterfile(parameterfile)
		
		spax_properties_file = f"{parameters['input_dir']}/spaxel_properties.fits"

		spax_properties = Table.read(spax_properties_file)
		# print(spax_properties)
		vorbin_nums = np.array(spax_properties['vorbin_num'][:])
		vorbin_nums = np.sort(np.unique(vorbin_nums[vorbin_nums>=0]))

		spectra_file = f"{parameters['input_dir']}/logRebin_spectra.fits"


		hdul = fits.open(spectra_file)
		logLambda_spec, spectra  = read_spectra_hdul(hdul)
		logRebin_spectra = spectra[0]
		logRebin_noise = spectra[1]
		spectra = None

		header = hdul[0].header
		hdul.close()
		velscale = header['VELSCALE']
		parameters['galaxy_velscale'] = float(velscale)	

		print(parameters['galaxy_velscale'])


		templates, logLambda_templates = read_EMILES_spectra(parameters, match_velscale=False, convolve=False)
		# plt.scatter(logLambda_templates,templates[:,0],s=3)
		velscale_templates = c*np.diff(logLambda_templates)[0]
		# print(velscale_templates)
		# print(templates[:,0])

		# print(vsr_temp)

		# parameters['velscale_ratio'] = vsr_temp

		# templates, logLambda_templates = read_EMILES_spectra(parameters, match_velscale=True, convolve=False)
		# plt.scatter(logLambda_templates,templates[:,0],s=3)
		# velscale_templates = c*np.diff(logLambda_templates)[0]
		# print(velscale_templates)
		# print(templates[:,0])
		
		# templates, logLambda_templates = read_EMILES_spectra(parameters, match_velscale=False, convolve=True)
		# plt.scatter(logLambda_templates,templates[:,0],s=3)
		# velscale_templates = c*np.diff(logLambda_templates)[0]
		# print(velscale_templates)
		# print(templates[:,0])



		# templates, logLambda_templates = read_EMILES_spectra(parameters, match_velscale=True, convolve=True)

		# velscale_templates = c*np.diff(logLambda_templates)[0]
		# print(velscale_templates)
		# plt.scatter(logLambda_templates,templates[:,0],s=3)


		

		spec = pputils.convolve_gauss_hermite(templates, velscale_templates, pp.sol, galaxy.size,
                                      velscale_ratio=ratio, vsyst=dv)

		# The spectrum below is equal to pp.bestfit to machine precision

		spectrum = (spec @ pp.weights)*pp.mpoly + pp.apoly

def weighted_moment(data,weights=None,moment = 1):

	if isinstance(weights,type(None)):
		weights = np.ones_like(data)

	if moment >= 1:
		moment1 =  np.sum(data*weights) / np.sum(weights)
		mom = moment1

	if moment >= 2:
		moment2 = np.sum(weights*(data - moment1)**2) / np.sum(weights)
		moment2 = np.sqrt(moment2)
		mom = moment2

	if moment == 3:
		mom = np.sum(weights * ((data - moment1) / moment2)**3) / np.sum(weights)

	if moment == 4:
		mom = np.sum(weights * ((data - moment1) / moment2)**4) / np.sum(weights) - 3

	
	return mom	

def baselineFunc(xx,aa,bb,cc):
	yy = aa*xx**2 + bb*xx + cc
	return yy


### Old functions for checking things ###
def check_line_ratios(parameterfile):

	parameters = read_parameterfile(parameterfile)
	
	spax_properties = Table.read(f"{parameters['input_dir']}/spaxel_properties.fits")
	emline_fluxes = Table.read(f"{parameters['output_dir']}/spax_emline_fluxes.fits")
	emline_fluxes = hstack((spax_properties,emline_fluxes))

	vorbin_nums = np.unique(spax_properties['vorbin_num'])
	refs = np.zeros_like(vorbin_nums,dtype=int)
	for vv, vb in enumerate(vorbin_nums):
		refs[vv] = np.where(spax_properties['vorbin_num'] == vb)[0][0]

	fig = plt.figure(figsize=(20,10))
	gs = gridspec.GridSpec(4,1,hspace=0)


	ax1 = fig.add_subplot(gs[0,0])
	ax1.scatter(np.log10(emline_fluxes['Halpha']),
				emline_fluxes['Hbeta']/emline_fluxes['Halpha'],
				marker='o',s=0.5,color='Black',label="All")

	emline_fluxes_SN3 = emline_fluxes[
						np.where(
						(emline_fluxes['Halpha']/emline_fluxes['Halpha_err'] >=3) &
						(emline_fluxes['Hbeta']/emline_fluxes['Hbeta_err'] >=3))]

	emline_fluxes_SN5 = emline_fluxes[
						np.where(
						(emline_fluxes['Halpha']/emline_fluxes['Halpha_err'] >=5) &
						(emline_fluxes['Hbeta']/emline_fluxes['Hbeta_err'] >=5))]


	ax1.scatter(np.log10(emline_fluxes_SN3['Halpha']),
				emline_fluxes_SN3['Hbeta']/emline_fluxes_SN3['Halpha'],
				marker='o',s=0.5,color='DodgerBlue',label="SN>3")

	ax1.scatter(np.log10(emline_fluxes_SN5['Halpha']),
				emline_fluxes_SN5['Hbeta']/emline_fluxes_SN5['Halpha'],
				marker='o',s=0.5,color='Red',label="SN>5")

	ax1.plot([0,6],[1/2.83,1/2.83],color='Black',ls='--')


	ax2 = fig.add_subplot(gs[1,0],sharey=ax1)
	ax2.scatter(np.log10(emline_fluxes['Halpha']),
				emline_fluxes['OIII4959']/emline_fluxes['OIII5006'],
				marker='o',s=0.5,color='Black')
	
	emline_fluxes_SN3 = emline_fluxes[
						np.where(
						(emline_fluxes['OIII4959']/emline_fluxes['OIII4959_err'] >=3) &
						(emline_fluxes['OIII5006']/emline_fluxes['OIII5006_err'] >=3))]

	emline_fluxes_SN5 = emline_fluxes[
						np.where(
						(emline_fluxes['OIII4959']/emline_fluxes['OIII4959_err'] >=5) &
						(emline_fluxes['OIII5006']/emline_fluxes['OIII5006_err'] >=5))]


	ax2.scatter(np.log10(emline_fluxes_SN3['Halpha']),
				emline_fluxes_SN3['OIII4959']/emline_fluxes_SN3['OIII5006'],
				marker='o',s=0.5,color="DodgerBlue")

	ax2.scatter(np.log10(emline_fluxes_SN5['Halpha']),
				emline_fluxes_SN5['OIII4959']/emline_fluxes_SN5['OIII5006'],
				marker='o',s=0.5,color="Red")

	ax2.plot([0,6],[0.3355,0.3355],color='Black',ls='--')

	ax3 = fig.add_subplot(gs[2,0],sharey=ax1)
	ax3.scatter(np.log10(emline_fluxes['Halpha']),
				emline_fluxes['NII6548']/emline_fluxes['NII6583'],
				marker='o',s=0.5,color='Black')


	emline_fluxes_SN3 = emline_fluxes[
						np.where(
						(emline_fluxes['NII6548']/emline_fluxes['NII6548_err'] >=3) &
						(emline_fluxes['NII6583']/emline_fluxes['NII6583_err'] >=3))]

	emline_fluxes_SN5 = emline_fluxes[
						np.where(
						(emline_fluxes['NII6548']/emline_fluxes['NII6548_err'] >=5) &
						(emline_fluxes['NII6583']/emline_fluxes['NII6583_err'] >=5))]


	ax3.scatter(np.log10(emline_fluxes_SN3['Halpha']),
				emline_fluxes_SN3['NII6548']/emline_fluxes_SN3['NII6583'],
				marker='o',s=0.5,color="DodgerBlue")

	ax3.scatter(np.log10(emline_fluxes_SN5['Halpha']),
				emline_fluxes_SN5['NII6548']/emline_fluxes_SN5['NII6583'],
				marker='o',s=0.5,color="Red")


	ax3.plot([0,6],[0.337,0.337],color='Black',ls='--')

	ax4 = fig.add_subplot(gs[3,0])
	ax4.scatter(np.log10(emline_fluxes['Halpha']),
				emline_fluxes['SII6716']/emline_fluxes['SII6730'],
				marker='o',s=0.5,color='Black')

	emline_fluxes_SN3 = emline_fluxes[
						np.where(
						(emline_fluxes['SII6716']/emline_fluxes['SII6716_err'] >=3) &
						(emline_fluxes['SII6730']/emline_fluxes['SII6730_err'] >=3))]

	emline_fluxes_SN5 = emline_fluxes[
						np.where(
						(emline_fluxes['SII6716']/emline_fluxes['SII6716_err'] >=5) &
						(emline_fluxes['SII6730']/emline_fluxes['SII6730_err'] >=5))]


	ax4.scatter(np.log10(emline_fluxes_SN3['Halpha']),
				emline_fluxes_SN3['SII6716']/emline_fluxes_SN3['SII6730'],
				marker='o',s=0.5,color="DodgerBlue")

	ax4.scatter(np.log10(emline_fluxes_SN5['Halpha']),
				emline_fluxes_SN5['SII6716']/emline_fluxes_SN5['SII6730'],
				marker='o',s=0.5,color="Red")
	ax4.plot([0,6],[0.45,0.45],color='Black',ls='--')
	ax4.plot([0,6],[1.45,1.45],color='Black',ls='--')


	ax1.set_yscale('log')
	ax4.set_yscale('log')
	# ax1.set_xscale('log')


	ax1.tick_params(which='both',axis='both',direction='in',labelsize=13)
	ax1.tick_params(which='both',axis='x',direction='in',labelsize=0)
	ax1.set_ylabel(r"H$\beta$ / H$\alpha$",fontsize=14)


	ax2.tick_params(which='both',axis='both',direction='in',labelsize=13)
	ax2.tick_params(which='both',axis='x',direction='in',labelsize=0)
	ax2.set_ylabel("[OIII] 4958 / 5006",fontsize=14)

	ax3.tick_params(which='both',axis='both',direction='in',labelsize=13)
	ax3.tick_params(which='both',axis='x',direction='in',labelsize=0)
	ax3.set_ylabel("[NII] 6548 / 6583",fontsize=14)


	ax4.tick_params(which='both',axis='both',direction='in',labelsize=13)
	ax4.set_ylabel("[SII] 6716 / 6730",fontsize=14)

	
	ax4.set_xlabel(r'log F$_{H\alpha}$ [10$^{-20}$ erg s$^{-1}$ cm$^{-2}$]',fontsize=15)

	ax1.set_ylim([0.1,2])
	ax1.set_xlim([1,6])
	ax2.set_ylim([0.1,2])
	ax2.set_xlim([1,6])
	ax3.set_ylim([0.1,2])
	ax3.set_xlim([1,6])
	ax4.set_ylim([0.2,4])
	ax4.set_xlim([1,6])

	legs = [Line2D([0],[0],color='White',marker='o',markerfacecolor='Black'),
			Line2D([0],[0],color='White',marker='o',markerfacecolor='DodgerBlue'),
			Line2D([0],[0],color='White',marker='o',markerfacecolor='Red')]
	ax1.legend(legs,["All","SN>3","SN>5"],fontsize=15)

	fig.savefig(f"{parameters['output_dir']}/figures/EMline_ratios.png")

	# plt.show()
def line_ratio_maps(parameterfile):
	parameters = read_parameterfile(parameterfile)

	spax_properties = Table.read(f"{parameters['input_dir']}/spaxel_properties.fits")

	emline_fluxes = Table.read(f"{parameters['output_dir']}/spax_emline_fluxes.fits")

	spax_properties = hstack((spax_properties,emline_fluxes))

	metadata = spax_properties.meta
	Nx =  metadata['NX']
	Ny =  metadata['NY']


	img_grid = np.zeros((Ny,Nx)).flatten()


	lines = emline_fluxes.keys()
	
	line_pairs = [['NII6548','Halpha'],
				['NII6583','Halpha'],
				['SII6716','Halpha'],
				['SII6730','Halpha'],
				['OIII4959','Hbeta'],
				['OIII5006','Hbeta'],
				['Halpha','Hbeta'],
				['NII6548','NII6583'],
				['OIII4959', 'OIII5006'],
				['SII6716','SII6730']
				]
	ratio_limits = [[-1.5,-0.5],
					[-1.,0.],
					[-2,0.5],
					[-2,0.5],
					[-2,1.5],
					[-2,1.5],
					[np.log10(0.75 * 2.83),np.log10(1.25 * 2.83)],
					[np.log10(0.337*0.75),np.log10(0.337*1.25)],
					[np.log10(0.3355*0.75),np.log10(0.3355*1.25)],
					[np.log10(1*0.75),np.log10(1*1.25)],
	]



	for ii,pair in enumerate(line_pairs):
		line1_flux = emline_fluxes[pair[0]]
		line1_flux_err = emline_fluxes[f'{pair[0]}_err']
		line2_flux = emline_fluxes[pair[1]]
		line2_flux_err = emline_fluxes[f'{pair[1]}_err']

		line1_SN = line1_flux / line1_flux_err
		line2_SN = line2_flux / line2_flux_err

		line_ratio = line1_flux / line2_flux

		emlines_SN5 = np.logical_and(line1_SN>=4,
									line2_SN >=4 
									)


		linemap = copy.deepcopy(img_grid)
		linemap[np.array(spax_properties['spax_num'],dtype=int)[emlines_SN5]] = \
							line_ratio[emlines_SN5]

		p90 = np.percentile(np.log10(line_ratio[emlines_SN5]),90)
		p50 = np.percentile(np.log10(line_ratio[emlines_SN5]),50)
		p10 = np.percentile(np.log10(line_ratio[emlines_SN5]),10)
	
		linemap = linemap.reshape((Ny,Nx))


		fig = plt.figure(figsize=(12,10))
		gs = gridspec.GridSpec(1,2,width_ratios = [1,0.05])


		ax1 = fig.add_subplot(gs[0,0])
		cb_ax = fig.add_subplot(gs[0,1])

		# img = ax1.pcolormesh(np.log10(linemap),vmin=ratio_limits[ii][0],vmax=ratio_limits[ii][1])
		img = ax1.pcolormesh(np.log10(linemap),vmin=p10,vmax=p90)
		# ax1.pcolormesh(HAmap/HA_errmap,vmin=0.5,vmax=5)
		# ax1.contour(np.log10(linemap),levels=[p25,p50,p75],linewidths = 1,colors='Black')

		fig.colorbar(img,cax = cb_ax)
		# cb_ax.plot(cb_ax.get_xlim(),[p25,p25],color='Black')
		# cb_ax.plot(cb_ax.get_xlim(),[p50,p50],color='Black')
		# cb_ax.plot(cb_ax.get_xlim(),[p75,p75],color='Black')

		cb_ax.set_title(f'log {pair[0]}/{pair[1]}')
		cb_ax.tick_params(which='both',axis='both',direction='in')


		ax1.set_aspect("equal")
		ax1.set_xlabel("pix",fontsize=15)
		ax1.set_ylabel("pix",fontsize=15)
		ax1.tick_params(which='both',axis='both',direction='in',labelsize=12)

		fig.savefig(f"{parameters['output_dir']}/figures/{pair[0]}_{pair[1]}_ratio_map.png")
		# plt.show()

		if pair == ['Halpha','Hbeta']:
			linemap = copy.deepcopy(img_grid)
			
			EBV = 4*EBV_Hlines(line1_flux,line2_flux,6562.819,4861.333,2.83)


			linemap[np.array(spax_properties['spax_num'],dtype=int)[emlines_SN5]] = \
								EBV[emlines_SN5]

			p90 = np.percentile(np.log10(EBV[emlines_SN5]),75)
			p50 = np.percentile(np.log10(EBV[emlines_SN5]),50)
			p10 = np.percentile(np.log10(EBV[emlines_SN5]),25)
		
			linemap = linemap.reshape((Ny,Nx))


			fig = plt.figure(figsize=(12,10))
			gs = gridspec.GridSpec(1,2,width_ratios = [1,0.05])


			ax1 = fig.add_subplot(gs[0,0])
			cb_ax = fig.add_subplot(gs[0,1])

			# img = ax1.pcolormesh(np.log10(linemap),vmin=ratio_limits[ii][0],vmax=ratio_limits[ii][1])
			img = ax1.pcolormesh(linemap,vmin=0)#,vmax=p90)
			# ax1.pcolormesh(HAmap/HA_errmap,vmin=0.5,vmax=5)
			# ax1.contour(np.log10(linemap),levels=[p25,p50,p75],linewidths = 1,colors='Black')

			fig.colorbar(img,cax = cb_ax)
			# cb_ax.plot(cb_ax.get_xlim(),[p25,p25],color='Black')
			# cb_ax.plot(cb_ax.get_xlim(),[p50,p50],color='Black')
			# cb_ax.plot(cb_ax.get_xlim(),[p75,p75],color='Black')

			cb_ax.set_title(f'A$_V$')
			cb_ax.tick_params(which='both',axis='both',direction='in')


			ax1.set_aspect("equal")
			ax1.set_xlabel("pix",fontsize=15)
			ax1.set_ylabel("pix",fontsize=15)
			ax1.tick_params(which='both',axis='both',direction='in',labelsize=12)
			fig.savefig(f"{parameters['output_dir']}/figures/AV_map.png")
def make_BPT_diagram(parameterfile):
	parameters = read_parameterfile(parameterfile)

	spax_properties = Table.read(f"{parameters['input_dir']}/spaxel_properties.fits")

	emline_fluxes = Table.read(f"{parameters['output_dir']}/spax_emline_fluxes.fits")

	spax_properties = hstack((spax_properties,emline_fluxes))

	metadata = spax_properties.meta
	Nx =  metadata['NX']
	Ny =  metadata['NY']


	img_grid = np.zeros((Ny,Nx)).flatten()


	lines = emline_fluxes.keys()
	
	SNmin = 4

	SNgood = np.logical_and.reduce((emline_fluxes['Halpha'] / emline_fluxes['Halpha_err'] >= SNmin,
									emline_fluxes['NII6583'] / emline_fluxes['NII6583_err'] >= SNmin,
									emline_fluxes['Hbeta'] / emline_fluxes['Hbeta_err'] >= SNmin,
									emline_fluxes['OIII5006'] / emline_fluxes['OIII5006_err'] >= SNmin,
									emline_fluxes['SII6716'] / emline_fluxes['SII6716_err'] >= SNmin
									))

	emlines_SNgood = emline_fluxes[SNgood]

	fig = plt.figure(figsize=(10,5))
	gs = gridspec.GridSpec(1,2)


	BPT_ax1 = fig.add_subplot(gs[0,0])
	BPT_ax2 = fig.add_subplot(gs[0,1])

	N2HA = np.log10(emlines_SNgood['NII6583'] / emlines_SNgood['Halpha'])
	S2HA = np.log10(emlines_SNgood['SII6716'] / emlines_SNgood['Halpha'])
	O3HB = np.log10(emlines_SNgood['OIII5006'] / emlines_SNgood['Hbeta'])


	BPT_ax1.hexbin(N2HA,O3HB,bins='log',gridsize=(int(1.4*100),int(2.*100)) )
	BPT_ax2.hexbin(S2HA,O3HB,bins='log',gridsize=(int(1.4*100),int(2.*100)) )

	# BPT_ax1.scatter(N2HA,O3HB, s=0.5)
	# BPT_ax2.scatter(S2HA,O3HB, s=0.5)


	BPT_ax1.tick_params(axis='both',which='both',direction='in',labelsize=10)
	BPT_ax2.tick_params(axis='both',which='both',direction='in',labelsize=10)

	BPT_ax1.plot(np.arange(-1.5,0.3,0.01),0.61 / (np.arange(-1.5,0.3,0.01) - 0.47) + 1.19,ls=':',color='Black')
	BPT_ax1.text(-0.1,-0.95,'Kew+01',color='Black')
	
	BPT_ax1.plot(np.arange(-1.5,0.,0.01),0.61 / (np.arange(-1.5,0.,0.01) - 0.05) + 1.3,ls='--',color='Black')
	BPT_ax1.text(-0.5,-0.95,'Kauf+03',color='Black')


	BPT_ax2.plot(np.arange(-1.5,0.2,0.01),0.72 / (np.arange(-1.5,0.2,0.01) - 0.32) + 1.30,ls=':',color='Black')
	BPT_ax2.text(-0.3,-0.95,'Kew+01',color='Black')


	BPT_ax1.set_xlabel("log NII6583 / Halpha")
	BPT_ax2.set_xlabel("log SII6730 / Halpha")
	BPT_ax1.set_ylabel("log OIII5006 / Hbeta")

	BPT_ax1.set_xlim([-1.25,0.25])
	BPT_ax2.set_xlim(BPT_ax1.get_xlim())
	BPT_ax1.set_ylim([-1.,1.])
	BPT_ax2.set_ylim(BPT_ax1.get_ylim())

	fig.savefig(f"{parameters['output_dir']}/figures/BPTs.png")
	# plt.show()

if __name__ == '__main__':
	print("Fuctions not called this way, load library and use e.g. lilpPXF.run()")
	