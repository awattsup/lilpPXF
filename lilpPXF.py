import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import ticker
from matplotlib.lines import Line2D
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.table import Table, hstack
from scipy import stats
import resource
import copy
# import datetime
# import psutil

from scipy import ndimage
# from mpi4py import MPI

try:
	from mpi4py import MPI
except:
	print('mpi4py not found')

# from guppy import hpy
import glob
import sys, os
# import pandas as pd
from astropy.io import fits

from ppxf.ppxf import ppxf
import ppxf.ppxf_util as pputils

home_dir = os.path.expanduser('~')
sys.path.append(f'{home_dir}/Research/programs/astro-functions')
import astro_functions as astrofunc

from spectral_cube import SpectralCube


# import vorbin
from vorbin.voronoi_2d_binning import voronoi_2d_binning


#global variables
c = 299792.458

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
###program caller
def run():

	parameterfile = sys.argv[1]
	parameters = read_parameterfile(parameterfile)

	if parameters['read_data']:
		make_spectra_tables(parameterfile)

	if parameters['run_vorbin']:
		voronoi_bin_cube(parameterfile)

	if parameters['run_stelkin']:
		fit_stellar_kinematics(parameterfile)
###


#main fitting functions
def fit_stellar_kinematics(parameterfile):

	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	nproc = comm.Get_size()

	if rank == 0:
		print(f"Running with {nproc} processors")
		print("Preparing inputs")
		sys.stdout.flush()
		parameters = read_parameterfile(parameterfile)
		
		spax_properties_file = f"{parameters['input_dir']}/spaxel_properties.fits"

		spax_properties = Table.read(spax_properties_file)
		parameters['z'] = spax_properties.meta['Z']

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

		logLambda_templates, templates, components, gas_components = get_templates(parameters)


		#truncate to fit range
		if 'stars_lrange' in parameters.keys():
			Lmin = parameters['stars_lrange'][0]
			Lmax = parameters['stars_lrange'][1]
			specrange = np.where((logLambda_spec>=np.log(Lmin)) & 
									(logLambda_spec<=np.log(Lmax)))[0]
			logLambda_spec = logLambda_spec[specrange]
			logRebin_spectra = logRebin_spectra[specrange,:]
			logRebin_noise = logRebin_noise[specrange,:]

			specrange = np.where((logLambda_templates>=np.log(Lmin*0.9)) & 
									(logLambda_templates<=np.log(Lmax*1.1)))[0]
			logLambda_templates = logLambda_templates[specrange]
			templates = templates[specrange,:]



		good_pixels = create_spectrum_mask(logLambda_spec,parameters)

		start, moments, constr_kinem = get_constraints(parameters,spax_properties,vorbin_nums)


		dv = c*(np.nanmean(logLambda_templates[:parameters['velscale_ratio']]) - logLambda_spec[0]) # km/s

	else:
		parameters = None
		# start = None
		# moments = None
		# constraints = None
		components = None
		gas_components = None
		dv = None
		templates = None
		good_pixels = None

	parameters = comm.bcast(parameters,root=0)
	# start = comm.bcast(start,root=0)
	# moments = comm.bcast(moments,root=0)
	components = comm.bcast(components,root=0)
	gas_components = comm.bcast(gas_components,root=0)
	dv = comm.bcast(dv,root=0)
	templates = comm.bcast(templates,root=0)
	good_pixels = comm.bcast(good_pixels,root=0)
	
	comm.barrier()
	if rank == 0:
		print(f"Head node distributing {len(logRebin_spectra[0,:])} spectra")
		vorbin_nums = np.array(spax_properties['vorbin_num'][:])
		vorbin_nums = np.sort(np.unique(vorbin_nums[vorbin_nums>=0]))
	sys.stdout.flush()
	for nn in range(1,nproc):
		if rank == 0:
			proc_spax = np.arange(nn,len(logRebin_spectra[0,:]),nproc,dtype=int)
			# proc_spax_properies = spax_properties[proc_spax]
			proc_vorbin_nums = vorbin_nums[proc_spax]
			proc_logRebin_spectra = logRebin_spectra[:,proc_spax]
			proc_logRebin_noise = logRebin_noise[:,proc_spax]
			proc_start = [start[pp] for pp in proc_spax]
			proc_moments = [moments[pp] for pp in proc_spax]
			proc_constr_kinem = [constr_kinem[pp] for pp in proc_spax]
			tosend = [proc_vorbin_nums, #proc_spax_properies,
						proc_logRebin_spectra,
						proc_logRebin_noise,
						proc_start,
						proc_moments,
						proc_constr_kinem]
			comm.send(tosend, dest=nn, tag=100+nn)
			tosend = None

		elif rank == nn:
			torecieve = comm.recv(source=0, tag=100+rank)
			# proc_spax_properties = torecieve[0]
			proc_vorbin_nums = torecieve[0]
			proc_logRebin_spectra = torecieve[1]
			proc_logRebin_noise = torecieve[2]
			proc_start = torecieve[3]
			proc_moments = torecieve[4]
			proc_constr_kinem = torecieve[5]
			torecieve = None

	if rank == 0:
		proc_spax = np.arange(0,len(logRebin_spectra[0,:]),nproc,dtype=int)
		# proc_spax_properties = spax_properties[proc_spax]
		proc_vorbin_nums = vorbin_nums[proc_spax]
		proc_logRebin_spectra = logRebin_spectra[:,proc_spax]
		proc_logRebin_noise = logRebin_noise[:,proc_spax]
		proc_start = [start[pp] for pp in proc_spax]
		proc_moments = [moments[pp] for pp in proc_spax]
		proc_constr_kinem = [constr_kinem[pp] for pp in proc_spax]

		spectra_shape = logRebin_spectra.shape


		logRebin_spectra = None
		logRebin_noise = None
		hdul = None

	if rank == 0:
		print(f"Spectra distributed, running fits")
	sys.stdout.flush()

	comm.barrier()

	outputs = []
	outputs_all = []

	# for ss in range(len(proc_logRebin_spectra[0,:])):
	for vb, vorbin_num in enumerate(proc_vorbin_nums):

		spectrum = np.array(proc_logRebin_spectra[:,vb])
		noise = np.array(proc_logRebin_noise[:,vb])
		noise = np.sqrt(noise)
		good_pixels_spec = good_pixels.copy()
		good_pixels_spec[np.isfinite(spectrum)==False] = False
		good_pixels_spec[(np.isfinite(noise)==False) | (noise <= 0)] = False


		spec_median = np.abs(np.nanmedian(spectrum))
		spectrum = spectrum / spec_median			#nomalise spectrum
		# var_median = np.abs(np.nanmedian(noise))
		# noise = noise / var_median			#nomalise noise
		noise[~good_pixels_spec] = 1.e10		

		# print(proc_constraints)
		# exit()
		# print(proc_start)


		out = ppxf(templates, spectrum, noise,
				velscale = parameters['galaxy_velscale'],
				start = proc_start[vb],
				moments = proc_moments[vb],
				component = components,
				gas_component = gas_components,
				constr_kinem = proc_constr_kinem[vb],
				degree = parameters['stars_degree'],
				mdegree = parameters['stars_mdegree'],
				velscale_ratio = parameters['velscale_ratio'],
				vsyst = dv,
				goodpixels=np.arange(len(spectrum))[good_pixels_spec],
				plot=False,
				# clean=True,
				quiet=True)
		# print(out.weights)
		# print(np.where(out.weights > 0))
		# plt.show()
		# plt.plot(np.exp(logLambda_templates),templates)
		# plt.show()
		# exit()
		# print(np.median(out.galaxy[good_pixels]))
		# print(np.median(out.galaxy[good_pixels]-out.bestfit[good_pixels]))
		# print(np.std(out.galaxy[good_pixels]-out.bestfit[good_pixels]))
		# print(out.polyweights)
		# print(out.mpolyweights)
		# plt.plot(out.galaxy[good_pixels]-out.bestfit[good_pixels])
		# plt.show()
		# exit()

		sol = out.sol
		error = out.error
		if not isinstance(out.sol[0],float):
			sol = out.sol[0]
			error = out.error[0]

		outputs.append([vorbin_num, out.chi2,sol,error*np.sqrt(out.chi2),out.bestfit*spec_median, out.weights,out.apoly,out.mpoly])
	
		if vb%100 == 0 and vb !=0:
			comm.barrier()
			if rank == 0:
				print(f"Proc {rank} is {100*vb/len(proc_logRebin_spectra[0,:]):.2f}% through {len(proc_logRebin_spectra[0,:])} spectra")
				print(f"Gathering outputs so far")
				# print(f"-------------------------")
			sys.stdout.flush()

			outputs = comm.gather(outputs,root=0)
			if rank == 0:
				outputs_all.extend(outputs)
			outputs = []
			print(f"Outputs gathered, continuing")
			print(f"-------------------------")
			sys.stdout.flush()
	
	comm.barrier()
	if rank == 0:
		print("pPXF fits finished, gathering last outputs to head node")
		print(f"-------------------------")	
		sys.stdout.flush()

	outputs = comm.gather(outputs,root=0)
	if rank == 0:
		outputs_all.extend(outputs)
		outputs_all = [oo for output in outputs_all for oo in output]
	comm.barrier()


	if rank == 0:
			

		# spax_properties = Table.read(spax_properties_file)
		stellar_kin = np.full([len(spax_properties),9],np.nan)
		bestfit_spectra = np.zeros(spectra_shape)
		template_weights = np.empty((templates.shape[1],spectra_shape[1]))
		apy = np.empty(spectra_shape)
		mpy = np.empty(spectra_shape)

		# for k in ['V','sigma','h3','h4']:
			# spax_properties[f'{k}_stellar'] = np.full(len(spax_properties['spax_num'][:]), -9999.)
		
		for out in outputs_all:
			vorbin_num = out[0]
			chi2 = out[1]
			kin = out[2]
			err = out[3]

			ref = np.where(spax_properties['vorbin_num'][:] == vorbin_num)[0]
			stellar_kin[ref,:] = np.hstack((chi2,kin,err))
			# print(out[2])
			bestfit_spectra[:,vorbin_num] = np.array(out[4])
			# for v,k in enumerate(['V','sigma','h3','h4']):
					# spax_properties[f'{k}_stellar'][ref] = fit.sol[v]


			template_weights[:,vorbin_num] = out[5]
			apy[:,vorbin_num] = out[6]
			mpy[:,vorbin_num] = out[7]

		
		if not os.path.isdir(f"{parameters['output_dir']}"):
			os.mkdir(f"{parameters['output_dir']}")
			os.mkdir(f"{parameters['output_dir']}/figures")
		
		stellar_kin = Table(stellar_kin,names=
					('chi2','V_stellar','sigma_stellar','h3_stellar','h4_stellar',
						'V_stellar_err','sigma_stellar_err','h3_stellar_err','h4_stellar_err'))
		print("Saving bestfit stellar kinematics table")
		sys.stdout.flush()
		stellar_kin.write(f"{parameters['output_dir']}/bestfit_stellar_kinematics.fits",overwrite=True)
		
		# weights = Table(np.asarray(template_weights),names=
		# 			('weights','apoly','mpoly'))
		# print("Saving templates table")
		# weights.write(f"{parameters['output_dir']}/bestfit_weights.fits",overwrite=True)
		header["COMMENT1"] = "Best fit weights for bins"
		primary_hdu = fits.PrimaryHDU(data = logLambda_spec,header = header)

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



		header["COMMENT1"] = "Best fit stellar spectra for bins"
		primary_hdu = fits.PrimaryHDU(data = logLambda_spec,header = header)

		bestfit_stars_hdu = fits.BinTableHDU.from_columns(
								fits.ColDefs([
								fits.Column(
								array = bestfit_spectra.T,
								name='BESTSTARS',format=str(len(bestfit_spectra))+'D'
								)]))

		hdul = fits.HDUList([primary_hdu,
							bestfit_stars_hdu])
		print("Saving best fit spectra table")
		sys.stdout.flush()
		hdul.writeto(f"{parameters['output_dir']}/logRebin_stelkin_spectra.fits",overwrite=True)
		print("Saved")
		sys.stdout.flush()

		print("Making stellar kinematics maps")
		sys.stdout.flush()
		make_stelkin_map(parameters)

def fit_individual_continuum(parameterfile):
	# import mpi4py
	# print(mpi4py.get_config())
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	nproc = comm.Get_size()

	if rank == 0:
		print(f"Running with {nproc} processors")
		print("Preparing inputs")
		sys.stdout.flush()
		parameters = read_parameterfile(parameterfile)
		
		spax_properties_file = f"{parameters['input_dir']}/spaxel_properties.fits"
		spax_properties = Table.read(spax_properties_file)

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
		# print(velscale)

		gas_flag = False
		if not isinstance(parameters['gas_groups'],type(None)):
			gas_flag = True

		vorbin_nums = np.array(spax_properties['vorbin_num'][:])
		vorbin_nums = np.sort(np.unique(vorbin_nums[vorbin_nums>=0]))

		logLambda_templates, stellar_templates = read_EMILES_spectra(parameters)
		components_stars = [0]*len(stellar_templates[0,:])

		if gas_flag:
			gas_templates, gas_names, Ncomp_gas = make_gas_templates(parameters,logLambda_templates)
			parameters['Ncomp_gas'] = Ncomp_gas
			components_gas = [[nn + 1 + components_stars[-1]]*NN for nn,NN in enumerate(Ncomp_gas)]
			components_gas = [comp for group in components_gas for comp in group]
			good_pixels = create_spectrum_mask(logLambda_spec,parameters,gas_fit=True)

			components = components_stars + components_gas
			templates = np.column_stack((stellar_templates,gas_templates))
		else:
			Ncomp_gas = []
			gas_names = []
			good_pixels = create_spectrum_mask(logLambda_spec,parameters)
			components = components_stars
			templates = stellar_templates


		gas_components = np.array(components) > 0
	
		

		start, moments, constraints = get_constraints(parameters,spax_properties,vorbin_nums)
		# print(len(constraints))
		# print(constraints)
		# exit()


		parameters['dv'] = c*(np.nanmean(logLambda_templates[:parameters['velscale_ratio']]) - logLambda_spec[0]) # km/s


	else:
		parameters = None
		templates = None
		good_pixels = None
		components = None
		moments = None
		start = None
		gas_flag = None
		constraints = None
		gas_components = None

		# Ncomp_gas = None

	parameters = comm.bcast(parameters,root=0)
	templates = comm.bcast(templates,root=0)
	good_pixels = comm.bcast(good_pixels,root=0)
	components = comm.bcast(components,root=0)
	moments = comm.bcast(moments,root=0)
	start = comm.bcast(start,root=0)
	gas_flag = comm.bcast(gas_flag,root=0)
	constraints = comm.bcast(constraints,root=0)
	gas_components = comm.bcast(gas_components,root=0)
	
	# Ncomp_gas = comm.bcast(Ncomp_gas,root=0)
	

	comm.barrier()
	if rank == 0:
		print(f"Head node distributing {len(logRebin_spectra[0,:])} spectra")
	sys.stdout.flush()
	for nn in range(1,nproc):
		if rank == 0:
			proc_spax = np.arange(nn,len(logRebin_spectra[0,:]),nproc,dtype=int)
			# proc_spax_properies = spax_properties[proc_spax]
			proc_vorbin_nums = vorbin_nums[proc_spax]
			proc_logRebin_spectra = logRebin_spectra[:,proc_spax]
			proc_logRebin_noise = logRebin_noise[:,proc_spax]

			tosend = [proc_vorbin_nums, #proc_spax_properies,
						proc_logRebin_spectra,
						proc_logRebin_noise]
			comm.send(tosend, dest=nn, tag=100+nn)
			tosend = None

		elif rank == nn:
			torecieve = comm.recv(source=0, tag=100+rank)
			# proc_spax_properties = torecieve[0]
			proc_vorbin_nums = torecieve[0]
			proc_logRebin_spectra = torecieve[1]
			proc_logRebin_noise = torecieve[2]
			torecieve = None

	if rank == 0:
		proc_spax = np.arange(0,len(logRebin_spectra[0,:]),nproc,dtype=int)
		# proc_spax_properties = spax_properties[proc_spax]
		proc_vorbin_nums = vorbin_nums[proc_spax]
		proc_logRebin_spectra = logRebin_spectra[:,proc_spax]
		proc_logRebin_noise = logRebin_noise[:,proc_spax]

		spectra_shape = logRebin_spectra.shape


		logRebin_spectra = None
		logRebin_noise = None
		hdul = None

	if rank == 0:
		print(f"Spectra distributed, running fits")
	sys.stdout.flush()

	comm.barrier()


	outputs = []
	outputs_all = []

	# proc_vorbin_nums = proc_vorbin_nums[2292::]

	# for ss in range(len(proc_logRebin_spectra[0,:])):
	for vb, vorbin_num in enumerate(proc_vorbin_nums):
		# print(vorbin_num)

		spectrum = np.array(proc_logRebin_spectra[:,vb])
		noise = np.array(proc_logRebin_noise[:,vb])
		noise = np.sqrt(noise)
		good_pixels_spec = good_pixels.copy()
		good_pixels_spec[np.isfinite(spectrum)==False] = False
		good_pixels_spec[(np.isfinite(noise)==False) | (noise <= 0)] = False


		spec_median = np.abs(np.nanmedian(spectrum))
		spectrum = spectrum / spec_median			#nomalise spectrum
		# var_median = np.abs(np.nanmedian(noise))
		# noise = noise / var_median			#nomalise noise
		noise[~good_pixels_spec] = 1.e10	

		constr = constraints[vorbin_num]
		if isinstance(constr,list):
			constr = constr[0]
		# print(constr)
		# print(start[vorbin_num])
		# print(moments[vorbin_num])
		# exit()

		# start = [list(stel_kin[['V_stellar','sigma_stellar','h3_stellar','h4_stellar']][int(vorbin_num)])] + [[0,30.]] + [[0,50]]*(len(parameters['Ncomp_gas'])-1)

		# try:
		if gas_flag:

			out = ppxf(templates, spectrum, noise,
				velscale = parameters['galaxy_velscale'],
				start = start[vorbin_num],
				moments = moments[vorbin_num],
				component = components,
				# degree = parameters['stars_degree'],
				degree = -1,					# parameters['gas_degree'] #additive can affect Balmer fluxes
				mdegree= parameters['cont_mdegree'],
				gas_component = gas_components,
				velscale_ratio = parameters['velscale_ratio'],
				vsyst = parameters['dv'],
				constr_kinem = constr,
				goodpixels=np.arange(len(spectrum))[good_pixels_spec],
				plot=False,
				quiet=True
				)


		else:

			out = ppxf(templates, spectrum, noise,
				velscale = parameters['galaxy_velscale'],
				start = start[vorbin_num],
				moments = moments[vorbin_num],
				component = components,
				# degree = parameters['stars_degree'],
				degree = -1,					# parameters['gas_degree'] #additive can affect Balmer fluxes
				mdegree= parameters['cont_mdegree'],
				velscale_ratio = parameters['velscale_ratio'],
				vsyst = dv,
				constr_kinem = constr,
				goodpixels=np.arange(len(spectrum))[good_pixels_spec],
				plot=False,
				quiet=True
				)
		# except:
			# print(vorbin_num)
			# print(constr)
			# print(np.array(start[vorbin_num][0])/velscale)
			# exit()

		# plt.show()

		bestfit_spectrum = out.bestfit*spec_median
		if gas_flag:
			bestfit_gas = out.gas_bestfit*spec_median
		else:
			bestfit_gas = np.zeros_like(bestfit_spectrum)
		bestfit_stars = bestfit_spectrum - bestfit_gas

		# outputs.append([vorbin_num,out.gas_flux,out.gas_flux_error,out.sol])

		outputs.append([vorbin_num,bestfit_stars,bestfit_gas])
		# plt.show()		
	
		if vb%100 == 0 and vb !=0:
			comm.barrier()
			if rank == 0:
				print(f"Proc {rank} is {100*vb/len(proc_logRebin_spectra[0,:]):.2f}% through {len(proc_logRebin_spectra[0,:])} spectra")
				print(f"Gathering outputs so far")
				print(f"-------------------------")
			sys.stdout.flush()

			outputs = comm.gather(outputs,root=0)
			if rank == 0:
				outputs_all.extend(outputs)
			outputs = []
	
	comm.barrier()
	if rank == 0:
		print("pPXF fits finished, gathering last outputs to head node")
	outputs = comm.gather(outputs,root=0)
	if rank == 0:
		outputs_all.extend(outputs)
		outputs_all = [oo for output in outputs_all for oo in output]
	comm.barrier()


	if rank == 0:
		bestfit_continuum = np.zeros(spectra_shape)
		bestfit_gas = np.zeros(spectra_shape)
		for out in outputs_all:
			vorbin_num = out[0]
			stars = out[1]
			gas = out[2]
			ref = np.where(spax_properties['vorbin_num'][:] == vorbin_num)[0][0]
			bestfit_continuum[:,ref] = stars
			bestfit_gas[:,ref] = gas

		header["COMMENT1"] = "Best fit continuum spectra for individual spaxels"
		primary_hdu = fits.PrimaryHDU(data = logLambda_spec,header = header)

		bestfit_continuum_hdu = fits.BinTableHDU.from_columns(
								fits.ColDefs([
								fits.Column(
								array = bestfit_continuum.T,
								name='BESTSTARS',format=str(len(bestfit_continuum))+'D'
								)]))
		bestfit_gas_hdu = fits.BinTableHDU.from_columns(
								fits.ColDefs([
								fits.Column(
								array = bestfit_gas.T,
								name='BESTGAS',format=str(len(bestfit_gas))+'D'
								)]))

		hdul = fits.HDUList([primary_hdu,
							bestfit_continuum_hdu,
							bestfit_gas_hdu])
			
		if not os.path.isdir(f"{parameters['output_dir']}"):
			os.mkdir(f"{parameters['output_dir']}")
			os.mkdir(f"{parameters['output_dir']}/figures")

		print("Saving continuum spectra table")
		sys.stdout.flush()
		hdul.writeto(f"{parameters['output_dir']}/bestfit_continuum_spectra.fits",overwrite=True)
		print("Saved")
		sys.stdout.flush()

def fit_continuum(parameterfile):

	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	nproc = comm.Get_size()


	if rank == 0:
		print(f"Running with {nproc} processors")
		print("Preparing inputs")
		sys.stdout.flush()
		parameters = read_parameterfile(parameterfile)
		print(parameters.keys())
		
		if 'gas_groups' in parameters.keys() and not isinstance(parameters['gas_groups'],type(None)):
			gas_flag = True
		else:
			gas_flag = False

		spax_properties_file = f"{parameters['input_dir']}/spaxel_properties.fits"

		spectra_file = f"{parameters['input_dir']}/logRebin_spectra.fits"


		spax_properties = Table.read(spax_properties_file)

		hdul = fits.open(spectra_file)
		logLambda_spec, spectra  = read_spectra_hdul(hdul)
		logRebin_spectra = spectra[0]
		logRebin_noise = spectra[1]
		spectra = None

		header = hdul[0].header
		hdul.close()
		velscale = header['VELSCALE']
		parameters['galaxy_velscale'] = float(velscale)

		logLambda_templates, stellar_templates = read_EMILES_spectra(parameters)
		components_stars = [0]*len(stellar_templates[0,:])

		if gas_flag:
			gas_templates, gas_names, Ncomp_gas = make_gas_templates(parameters,logLambda_templates)
			components_gas = [[nn + 1 + components_stars[-1]]*NN for nn,NN in enumerate(Ncomp_gas)]
			components_gas = [comp for group in components_gas for comp in group]
			good_pixels = create_spectrum_mask(logLambda_spec,parameters,gas_fit=True)

			components = components_stars + components_gas
			templates = np.column_stack((stellar_templates,gas_templates))
		else:
			Ncomp_gas = []
			gas_names = []
			good_pixels = create_spectrum_mask(logLambda_spec,parameters)
			components = components_stars
			templates = stellar_templates

		constr_kinem = get_constraints(parameters)
		# print(constr_kinem)
		# exit()



		gas_components = np.array(components) > 0
	
		moments  = [parameters['continuum_moments']] + \
					[parameters['continuum_gas_moments']] * len(Ncomp_gas)

		start = [[0,50.]] + [[0,50.]]*len(Ncomp_gas)
		if len(start)==1:
			start = start[0]
			# moments = moments[0]

		# print(len(moments))
		# print(len(start))

		# print(start)
		# print(moments)

		dv = c*(np.nanmean(logLambda_templates[:parameters['velscale_ratio']]) - logLambda_spec[0]) # km/s
		# start =[V,100]
		Nspec = len(logRebin_spectra[0,:])

	else:
		parameters = None
		start = None
		dv = None
		templates = None
		good_pixels = None
		Nspec = None
		components = None
		moments = None
		constr_kinem = None
		
		gas_components = None

	parameters = comm.bcast(parameters,root=0)
	start = comm.bcast(start,root=0)
	dv = comm.bcast(dv,root=0)
	templates = comm.bcast(templates,root=0)
	good_pixels = comm.bcast(good_pixels,root=0)
	Nspec = comm.bcast(Nspec,root=0)
	components = comm.bcast(components,root=0)
	moments = comm.bcast(moments,root=0)
	constr_kinem = comm.bcast(constr_kinem,root=0)
	
	gas_components = comm.bcast(gas_components,root=0)
	

	comm.barrier()
	if rank == 0:
		print(f"Head node distributing {len(logRebin_spectra[0,:])} spectra")
	sys.stdout.flush()
	for nn in range(1,nproc):
		if rank == 0:
			proc_spax = np.arange(nn,len(logRebin_spectra[0,:]),nproc,dtype=int)
			proc_spax_properies = spax_properties[proc_spax]
			proc_logRebin_spectra = logRebin_spectra[:,proc_spax]
			proc_logRebin_noise = logRebin_noise[:,proc_spax]

			tosend = [proc_spax_properies,
						proc_logRebin_spectra,
						proc_logRebin_noise]
			comm.send(tosend, dest=nn, tag=100+nn)
			tosend = None

		elif rank == nn:
			torecieve = comm.recv(source=0, tag=100+rank)
			proc_spax_properties = torecieve[0]
			proc_logRebin_spectra = torecieve[1]
			proc_logRebin_noise = torecieve[2]
			torecieve = None

	if rank == 0:
		proc_spax = np.arange(0,len(logRebin_spectra[0,:]),nproc,dtype=int)
		proc_spax_properties = spax_properties[proc_spax]
		proc_logRebin_spectra = logRebin_spectra[:,proc_spax]
		proc_logRebin_noise = logRebin_noise[:,proc_spax]

		spectra_shape = logRebin_spectra.shape


		logRebin_spectra = None
		logRebin_noise = None
		hdul = None

	if rank == 0:
		print(f"Spectra distributed, running fits")
	sys.stdout.flush()

	comm.barrier()

	outputs = []
	outputs_all = []

	for ss in range(len(proc_logRebin_spectra[0,:])):

		spectrum = np.array(proc_logRebin_spectra[:,ss])
		noise = np.array(proc_logRebin_noise[:,ss])
		good_pixels_spec = good_pixels.copy()
		good_pixels_spec[np.isfinite(spectrum)==False] = False
		good_pixels_spec[(np.isfinite(noise)==False) | (noise <= 0)] = False

		spec_median = np.abs(np.nanmedian(spectrum))
		var_median = np.abs(np.nanmedian(noise))
		spectrum = spectrum / spec_median			#nomalise spectrum
		noise = noise / var_median			#nomalise noise
		noise[~good_pixels_spec] = 1.e-5			#pPXF doesnt apply goodpix to noise -_-


		if np.any(gas_components):
			out = ppxf(templates, spectrum, noise,
				velscale = parameters['galaxy_velscale'],
				start = start,
				moments = moments,
				component=components,
				degree = -1,										# additive can affect Balmer fluxes
				mdegree = parameters['continuum_mdegree'],
				gas_component = gas_components,
				velscale_ratio = parameters['velscale_ratio'],
				vsyst = dv,
				goodpixels=np.arange(len(spectrum))[good_pixels_spec],
				clean=True,
				constr_kinem = constr_kinem,
				plot=False,
				quiet=True)
		else:
			out = ppxf(templates, spectrum, noise,
				velscale = parameters['galaxy_velscale'],
				start = start,
				moments = moments,
				component=components,
				degree = -1,										# additive can affect Balmer fluxes
				mdegree = parameters['continuum_mdegree'],
				# gas_component = gas_components,
				# velscale_ratio = parameters['velscale_ratio'],
				vsyst = dv,
				goodpixels=np.arange(len(spectrum))[good_pixels_spec],
				clean=True,
				constr_kinem = constr_kinem,
				plot=False,
				quiet=True)
		# plt.show()


		
		bestfit_spectrum = out.bestfit*spec_median
		if gas_flag:
			bestfit_gas = out.gas_bestfit*spec_median
		else:
			bestfit_gas = np.zeros_like(bestfit_spectrum)
		bestfit_stars = bestfit_spectrum - bestfit_gas


		outputs.append([int(proc_spax_properties['vorbin_num'][ss]),bestfit_stars,bestfit_gas])
			
	
		if ss%100 == 0 and ss !=0:
			comm.barrier()
			if rank == 0:
				print(f"Proc {rank} is {100*ss/len(proc_logRebin_spectra[0,:]):.2f}% through {len(proc_logRebin_spectra[0,:])} spectra")
				print(f"Gathering outputs so far")
				print(f"-------------------------")
			sys.stdout.flush()

			outputs = comm.gather(outputs,root=0)
			if rank == 0:
				outputs_all.extend(outputs)
			outputs = []
				

			# if rank == 0:
			# 	mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
			# 	print(mem)
			# 	h = hpy().heap()
			# 	print(h.byrcs)


	comm.barrier()
	if rank == 0:
		print("pPXF fits finished, gathering last outputs to head node")
	outputs = comm.gather(outputs,root=0)
	if rank == 0:
		outputs_all.extend(outputs)
		outputs_all = [oo for output in outputs_all for oo in output]
	comm.barrier()
		
	if rank == 0:
		bestfit_continuum = np.zeros(spectra_shape)
		bestfit_gas = np.zeros(spectra_shape)
		for out in outputs_all:
			vorbin_num = out[0]
			stars = out[1]
			gas = out[2]
			ref = np.where(spax_properties['vorbin_num'][:] == vorbin_num)[0][0]
			# ref = np.in1d(spax_properties['vorbin_num'][:], vorbin_num)
			bestfit_continuum[:,ref] = stars
			bestfit_gas[:,ref] = gas

		header["COMMENT1"] = "Best fit continuum spectra for individual spaxels"
		primary_hdu = fits.PrimaryHDU(data = logLambda_spec,header = header)

		bestfit_stars_hdu = fits.BinTableHDU.from_columns(
								fits.ColDefs([
								fits.Column(
								array = bestfit_continuum.T,
								name='BESTSTARS',format=str(len(bestfit_continuum))+'D'
								)]))
		bestfit_gas_hdu = fits.BinTableHDU.from_columns(
								fits.ColDefs([
								fits.Column(
								array = bestfit_gas.T,
								name='BESTGAS',format=str(len(bestfit_gas))+'D'
								)]))

		hdul = fits.HDUList([primary_hdu,
							bestfit_stars_hdu,
							bestfit_gas_hdu])
		print("Saving continuum spectra table")
		sys.stdout.flush()
		hdul.writeto(f"{parameters['output_dir']}/bestfit_continuum_spectra.fits",overwrite=True)
		print("Saved")
		sys.stdout.flush()
			
def fit_gas_lines(parameterfile):

	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	nproc = comm.Get_size()

	if rank == 0:
		print(f"Running with {nproc} processors")
		print("Preparing inputs")
		sys.stdout.flush()
		parameters = read_parameterfile(parameterfile)
		
		spax_properties_file = f"{parameters['input_dir']}/spaxel_properties.fits"

		spectra_file = f"{parameters['input_dir']}/logRebin_spectra.fits"


		spax_properties = Table.read(spax_properties_file)

		hdul = fits.open(spectra_file)
		logLambda_spec, spectra  = read_spectra_hdul(hdul)
		logRebin_spectra = spectra[0]
		logRebin_noise = spectra[1]
		spectra = None

		header = hdul[0].header
		hdul.close()
		velscale = header['VELSCALE']
		parameters['galaxy_velscale'] = float(velscale)

		logLambda_templates, stellar_templates = read_EMILES_spectra(parameters)

		gas_templates, gas_names, Ncomp_gas = make_gas_templates(parameters,logLambda_templates)

		components_stars = [0]*len(stellar_templates[0,:])
		components_gas = [[nn+components_stars[-1]+1]*NN for nn,NN in enumerate(Ncomp_gas)]
		components_gas = [comp for group in components_gas for comp in group]

		# print(components_gas)
		# exit()

		good_pixels = create_spectrum_mask(logLambda_spec,parameters,gas_fit=True)

		stellar_kin_file = f"{parameters['output_dir']}/bestfit_stellar_kinematics.fits"
		stel_kin = Table.read(stellar_kin_file)
		# stel_kin = stel_kin['V_stellar','sigma_stellar',
											# 'h3_stellar','h4_stellar'][:]
		V = 0
		dv = c*(np.nanmean(logLambda_templates[:parameters['velscale_ratio']]) - logLambda_spec[0]) # km/s
		start =[V,50]
		Nspec = len(logRebin_spectra[0,:])
		# print(dv)
		# exit()

	else:
		parameters = None
		start = None
		dv = None
		stellar_templates = None
		good_pixels = None
		Nspec = None
		components_stars = None
		stel_kin = None
		
		gas_templates = None
		components_gas = None
		Ncomp_gas = None

	parameters = comm.bcast(parameters,root=0)
	start = comm.bcast(start,root=0)
	dv = comm.bcast(dv,root=0)
	stellar_templates = comm.bcast(stellar_templates,root=0)
	good_pixels = comm.bcast(good_pixels,root=0)
	Nspec = comm.bcast(Nspec,root=0)
	components_stars = comm.bcast(components_stars,root=0)
	stel_kin = comm.bcast(stel_kin,root=0)
	
	gas_templates = comm.bcast(gas_templates,root=0)
	components_gas = comm.bcast(components_gas,root=0)
	Ncomp_gas = comm.bcast(Ncomp_gas,root=0)
	

	comm.barrier()
	if rank == 0:
		print(f"Head node distributing {len(logRebin_spectra[0,:])} spectra")
	sys.stdout.flush()
	for nn in range(1,nproc):
		if rank == 0:
			proc_spax = np.arange(nn,len(logRebin_spectra[0,:]),nproc,dtype=int)
			proc_spax_properies = spax_properties[proc_spax]
			proc_spax_stelkin = stel_kin[proc_spax]
			# print((proc_spax_stelkin))
			proc_logRebin_spectra = logRebin_spectra[:,proc_spax]
			proc_logRebin_noise = logRebin_noise[:,proc_spax]

			tosend = [proc_spax_properies,
						proc_spax_stelkin,
						proc_logRebin_spectra,
						proc_logRebin_noise]
			comm.send(tosend, dest=nn, tag=100+nn)
			tosend = None

		elif rank == nn:
			torecieve = comm.recv(source=0, tag=100+rank)
			proc_spax_properties = torecieve[0]
			proc_spax_stelkin = torecieve[1]
			proc_logRebin_spectra = torecieve[2]
			proc_logRebin_noise = torecieve[3]
			torecieve = None

	if rank == 0:
		proc_spax = np.arange(0,len(logRebin_spectra[0,:]),nproc,dtype=int)
		proc_spax_properties = spax_properties[proc_spax]
		proc_spax_stelkin = stel_kin[proc_spax]
		proc_logRebin_spectra = logRebin_spectra[:,proc_spax]
		proc_logRebin_noise = logRebin_noise[:,proc_spax]

		spectra_shape = logRebin_spectra.shape


		# logRebin_spectra = None
		logRebin_noise = None
		hdul = None

	if rank == 0:
		print(f"Spectra distributed, running fits")
	sys.stdout.flush()

	comm.barrier()

	outputs = []
	outputs_all = []


	for ss in range(len(proc_logRebin_spectra[0,:])):

		#spaxel has stellar kinematics fit
		if proc_spax_properties['vorbin_num'][ss] >=0:
		# if True:
			templates = np.column_stack((stellar_templates,gas_templates))

			components = components_stars + components_gas
							
			gas_components = np.array(components) > 0

			moments  = [-parameters['stars_moments']] + \
						[parameters['gas_moments']] * len(Ncomp_gas)

			start = [list(proc_spax_stelkin[ss])] + [[0,50.]]*len(Ncomp_gas)


		elif proc_spax_properties['vorbin_num'][ss] < 0: #spaxel does not have stellar kinematics
			
			templates = gas_templates

			components = components_gas
							
			gas_components = np.array(components) > 0

			moments  = [parameters['gas_moments']] * len(Ncomp_gas)

			if len(Ncomp_gas)>1:
				start = [[0.,50.]]*len(Ncomp_gas)
			else:
				start = [[0.,50.]]


		# print(components)

		spectrum = np.array(proc_logRebin_spectra[:,ss])
		noise = np.array(proc_logRebin_noise[:,ss])
		good_pixels_spec = good_pixels.copy()
		good_pixels_spec[np.isfinite(spectrum)==False] = False
		good_pixels_spec[(np.isfinite(noise)==False) | (noise <= 0)] = False

		spec_median = np.abs(np.nanmedian(spectrum))
		var_median = np.abs(np.nanmedian(noise))
		spectrum = spectrum / spec_median			#nomalise spectrum
		noise = noise / var_median			#nomalise noise
		noise[~good_pixels_spec] = 1.e-5			#pPXF doesnt apply goodpix to noise -_-


		out = ppxf(templates, spectrum, noise,
				velscale = parameters['galaxy_velscale'],
				start = start,
				moments = moments,
				component = components,
				# degree = parameters['stars_degree'],
				degree = -1,					# parameters['gas_degree'] #additive can affect Balmer fluxes
				mdegree= parameters['gas_mdegree'],
				gas_component = gas_components,
				velscale_ratio = parameters['velscale_ratio'],
				vsyst = dv,
				goodpixels=np.arange(len(spectrum))[good_pixels],
				plot=False,
				quiet=True
				)
		# plt.show()

		outputs.append([int(proc_spax_properties['vorbin_num'][ss]),out.gas_flux,out.gas_flux_error,out.sol])
		# plt.show()		
	
		if ss%100 == 0 and ss !=0:
			comm.barrier()
			if rank == 0:
				print(f"Proc {rank} is {100*ss/len(proc_logRebin_spectra[0,:]):.2f}% through {len(proc_logRebin_spectra[0,:])} spectra")
				print(f"Gathering outputs so far")
				print(f"-------------------------")
			sys.stdout.flush()

			outputs = comm.gather(outputs,root=0)
			if rank == 0:
				outputs_all.extend(outputs)
			outputs = []
	
	comm.barrier()
	if rank == 0:
		print("pPXF fits finished, gathering last outputs to head node")
	outputs = comm.gather(outputs,root=0)
	if rank == 0:
		outputs_all.extend(outputs)
		outputs_all = [oo for output in outputs_all for oo in output]
	comm.barrier()


	if rank == 0:
		

		spax_properties = Table.read(spax_properties_file)
		gas_kin = np.full([len(spax_properties),2*len(Ncomp_gas)+2*len(gas_names)],np.nan)
		names = [f'{k}_gas_C{ii+1}' for ii in range(len(Ncomp_gas)) for k in ['V','sigma']] + \
				 [f'{ll}_flux{ee}' for ll in gas_names for ee in ['','err']]

		for out in outputs_all:
			vorbin_num = out[0]
			flux =out[1]
			flux_err = out[2]
			kin = out[3]
			if isinstance(kin[0],float):
				kin = np.array(kin).flatten()
			else:
				kin = np.array(kin[len(kin) - len(Ncomp_gas):]).flatten()

			ref = np.where(spax_properties['vorbin_num'][:] == vorbin_num)[0]
			# print(names)
			# print(kin,flux,flux_err)
			# print(np.hstack((kin,flux,flux_err)))

			gas_kin[ref,:] = np.hstack((kin,flux,flux_err))

			# for v,k in enumerate(['V','sigma','h3','h4']):
					# spax_properties[f'{k}_stellar'][ref] = fit.sol[v]

				
		gas_kin = Table(gas_kin,names=names)
		print("Saving bestfit gas kinematics table")
		gas_kin.write(f"{parameters['output_dir']}/bestfit_gas_kinematics.fits",overwrite=True)
		# spax_properties.write(spax_properties_file,overwrite=True)

def fit_individual_spectrum(parameterfile):
	parameters = read_parameterfile(parameterfile)
		
	spax_properties_file = f"{parameters['input_dir']}/spaxel_properties.fits"

	spectra_file = f"{parameters['input_dir']}/logRebin_spectra.fits"


	spax_properties = Table.read(spax_properties_file)
	
	hdul = fits.open(spectra_file)
	logLambda_spec, spectra  = read_spectra_hdul(hdul)
	logRebin_spectra = spectra[0]
	logRebin_noise = spectra[1]
	spectra = None

	ref = np.where((spax_properties['spax_xx'] == parameters['indiv_xx']) &
						(spax_properties['spax_yy'] == parameters['indiv_yy']))[0]

	vorbin_num = spax_properties['vorbin_num'][ref]


	spectrum = np.asarray(logRebin_spectra[:,vorbin_num]).flatten()
	noise = np.asarray(logRebin_noise[:,vorbin_num]).flatten()

	#truncate to fit range
	if 'indiv_lrange' in parameters.keys():
		Lmin = parameters['indiv_lrange'][0]
		Lmax = parameters['indiv_lrange'][1]
		specrange = np.where((logLambda_spec>=np.log(Lmin)) & 
								(logLambda_spec<=np.log(Lmax)))[0]
		logLambda_spec = logLambda_spec[specrange]
		spectrum = spectrum[specrange]
		noise = noise[specrange]

	header = hdul[0].header
	hdul.close()
	velscale = header['VELSCALE']
	parameters['galaxy_velscale'] = float(velscale)


	logLambda_templates, stellar_templates = read_EMILES_spectra(parameters)
	components_stars = [0]*len(stellar_templates[0,:])


	if not isinstance(parameters['gas_groups'],type(None)):
		gas_templates, gas_names, Ncomp_gas = make_gas_templates(parameters,logLambda_templates)
		components_gas = [[nn + 1 + components_stars[-1]]*NN for nn,NN in enumerate(Ncomp_gas)]
		components_gas = [comp for group in components_gas for comp in group]

		components = components_stars + components_gas
		templates = np.column_stack((stellar_templates,gas_templates))


		good_pixels = create_spectrum_mask(logLambda_spec,parameters,gas_fit=True)
	else:
		Ncomp_gas = []
		components = components_stars
		templates = stellar_templates

		moments  = [parameters['indiv_moments']]
				
	constr_kinem = get_constraints(parameters)

	# print(constr_kinem)

	moments  = [parameters['indiv_moments']] + \
				[parameters['indiv_gas_moments']] * len(Ncomp_gas)
	gas_components = np.array(components) > 0
	
	# print(moments)

	dv = c*(np.nanmean(logLambda_templates[:parameters['velscale_ratio']]) - logLambda_spec[0]) # km/s
	start = parameters['indiv_start']
	start = [start] + [[0,50.]] + [[0,200]]*(len(Ncomp_gas)-1)


	good_pixels_spec = good_pixels.copy()
	good_pixels_spec[np.isfinite(spectrum)==False] = False
	good_pixels_spec[(np.isfinite(noise)==False) | (noise <= 0)] = False

	spec_median = np.abs(np.nanmedian(spectrum))
	spectrum = spectrum / spec_median			#nomalise spectrum
	# var_median = np.abs(np.nanmedian(noise))
	# noise = noise / var_median			#nomalise noise
	noise[~good_pixels_spec] = 1.e10		
	

	# templates = np.hstack((np.zeros(len(spectrum)+10),np.ones(len(spectrum)+10))).T
	if not isinstance(parameters['gas_groups'],type(None)):
		out = ppxf(templates, spectrum, noise,
				velscale = parameters['galaxy_velscale'],
				start = start,
				lam = np.exp(logLambda_spec),
				moments = moments,
				component = components,
				degree = -1,
				mdegree = parameters['indiv_mdegree'],
				velscale_ratio = parameters['velscale_ratio'],
				vsyst = dv,
				goodpixels=np.arange(len(spectrum))[good_pixels_spec],
				gas_component = gas_components,
				constr_kinem = constr_kinem,
				plot=True,
				clean=False,
				quiet=False)
	else:

		out = ppxf(templates, spectrum, noise,
				velscale = parameters['galaxy_velscale'],
				start = start,
				lam = np.exp(logLambda_spec),
				moments = parameters['indiv_moments'],
				degree = parameters['indiv_degree'],
				mdegree = parameters['indiv_mdegree'],
				velscale_ratio = parameters['velscale_ratio'],
				vsyst = dv,
				goodpixels=np.arange(len(spectrum))[good_pixels_spec],
				constr_kinem = constr_kinem,
				plot=True,
				clean=False,
				quiet=False)
	# print(out.error*np.sqrt(out.chi2))
	plt.show()
###


###data reading and binning
def read_parameterfile(filename = None):

	if filename is None:
		filename = "./parameters.param"


	#default values
	parameters = {'read_data':False,'run_vorbin':False,'run_stelkin':False,
					'base':home_dir,
					'velscale_ratio': 1,
					'gas_groups': None,'gas_names': None,'gas_Ncomp': [],'gas_constraints': None,
					'fit_CaT':False}


	f = open(filename)
	for line in f:
		# print(line)
		if line[0] == "#" or line[0] == " ":
				
			continue
		else:
			line = line.split("\n")[0].split(" ")

			if line[0] == "read_data":
				parameters[line[0]] = eval(line[1])
			if line[0] == "run_vorbin":
				parameters[line[0]] = eval(line[1])
			if line[0] == "run_stelkin":
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
			# elif line[0] == "test":
			# 	parameters[line[0]] = np.asarray(int(line[1]),dtype=bool)
			# elif line[0] == "continuum_subcube":
			# 	parameters[line[0]] = np.asarray(int(line[1]),dtype=bool)
			# elif line[0] == "input_dir":
			# 	parameters[line[0]] = line[1]
			# elif line[0] == "output_dir":
			# 	parameters[line[0]] = line[1]
			elif "_start" in line[0]:
				parameters[line[0]] = [float(line[1]),float(line[2])]
			elif "_lrange" in line[0]:
				parameters[line[0]] = [float(line[1]),float(line[2])]
			elif "lrange" in line[0]:
				parameters[line[0]] = [float(line[1]),float(line[2])]


			# elif line[0] == "vorbin_spectra_file":
				# if line[1] == "":
					# parameters[line[0]] = None
				# else:
					# parameters[line[0]] = line[1]
			# elif line[0] == "linemaps_spectra_file":
				# if line[1] == "":
					# parameters[line[0]] = None
				# else:
					# parameters[line[0]] = line[1]
			elif "_gaslines" in line[0]:
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

			# else:
			# 	parameters[line[0]] = [int(x) for x in line[1::]]

			# elif line[0] in ["z","Lmin","Lmax","SN_Lmin","SN_Lmax","SN_indiv","SN_vorbin"]:
			# 	parameters[line[0]] = float(line[1])
			# else:
			# 	parameters[line[0]] = [int(x) for x in line[1::]]

	parameters['output'] = f"{parameters['base']}/{parameters['output']}"
	parameters['output_dir'] = f"{parameters['output']}/{parameters['output_dir']}"
	if "input_dir" in parameters.keys():
		parameters['input_dir'] = f"{parameters['output']}/{parameters['input_dir']}"

	else:
		parameters['input_dir'] = parameters['output_dir']

	return parameters

def read_gaslines_parameterfile(filename = None):
	if filename is None:
		filename = "./gaslines.param"

	parameters = {}

	f = open(filename)


	gas_groups = []
	gas_group = []
	constr_kinem = {'A_ineq':[],'b_ineq':[]}
	for line in f:
		if line[0] == "#" or line[0] == " " or line[0] == "\n":
			if gas_group != []:	
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
			line = line.split("\n")[0].split(" ")
			# print(line)
			gas_group.extend(line)
	if gas_group != []:	
		gas_groups.append(gas_group)


	gas_Ncomp = []
	gas_names = []
	for gg, group in enumerate(gas_groups):
		for gl, gas_line in enumerate(group):
			gas_names.extend([f"C{gg+1}-{gas_line}"])
		gas_Ncomp.extend([gl+1])


	return [gas_groups, gas_names, gas_Ncomp, constr_kinem]

def make_spectra_tables(parameterfile):

	parameters = read_parameterfile(parameterfile)

	if not os.path.isdir(f"{parameters['output']}/indiv"):
		os.mkdir(f"{parameters['output']}/indiv")
	if not os.path.isdir(f"{parameters['output']}/testcube"):
		os.mkdir(f"{parameters['output']}/testcube")
	if not os.path.isdir(f"{parameters['output']}/test"):
		os.mkdir(f"{parameters['output']}/test")



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

	else:

		hdu = fits.open(parameters['datacube'])
		header = hdu[1].header

		spectra = hdu[1].data
		noise = np.sqrt(hdu[2].data)

		Nx = header['NAXIS1']
		Ny = header['NAXIS2']
		Nl = header['NAXIS3']

		spectra = spectra.reshape(Nl, Nx*Ny)
		noise = noise.reshape(Nl, Nx*Ny)

		linLambda_obs = astrofunc.get_wavelength_axis(header)
		#de-redshift spectra
		linLambda_obs = linLambda_obs / (1.e0 + parameters['z'])

		if 'lrange' in parameters.keys():
			Lambda_range = np.logical_and(linLambda_obs >= parameters['lrange'][0] ,
									linLambda_obs <= parameters['lrange'][1])

			spectra = spectra[Lambda_range]
			noise = noise[Lambda_range]
			linLambda_obs = linLambda_obs[Lambda_range]



		spax_number = np.arange(Nx*Ny,dtype='int')
		vorbin_number = np.zeros([Nx*Ny],dtype='int')
		obs_flags = np.all(np.isfinite(spectra), axis = 0)
		obs_flags_nums = np.zeros(Nx*Ny)
		obs_flags_nums[obs_flags] = 1

		spec_good = np.where(obs_flags_nums == 1)[0]
		spec_bad = np.where(obs_flags_nums == 0)[0]


		SN_cont_Lrange = np.logical_and(linLambda_obs >= parameters['SN_lrange'][0] , 
										linLambda_obs <= parameters['SN_lrange'][1])
		spax_signal_cont = np.nanmedian(spectra[SN_cont_Lrange,:], axis = 0)
		spax_noise_cont = np.abs(np.nanmedian(noise[SN_cont_Lrange,:], axis = 0))

		# if 'SN_line' in parameters.keys():
		# 	spax_signal_line = np.zeros([Nx*Ny,len(parameters['SN_line'])])
		# 	spax_noise_line = np.zeros([Nx*Ny,len(parameters['SN_line'])])
		# 	spax_line_names = []

		# 	for ii in range(len(parameters['SN_line'])):
		# 		print(emlines[parameters['SN_line'][ii]]['lambda'])
		# 		print(emlines[parameters['SN_line'][ii]]['lambda'][-1])
				
		# 		lineLambda = emlines[parameters['SN_line'][ii]]['lambda'][-1]

		# 		SN_line_Lrange = np.logical_and(linLambda_obs >= lineLambda*(1-500/c) , 
		# 								linLambda_obs <= lineLambda*(1+500/c) )
		# 		spax_signal_line[:,ii] = np.nansum(spectra[SN_line_Lrange,:], axis = 0)
		# 		spax_noise_line[:,ii] = np.nansum(noise[SN_line_Lrange,:], axis = 0)


		spax_xxyy , spax_RADEC, spax_SP = astrofunc.make_pix_WCS_grids(header)
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
										spax_noise_cont,
										# spax_signal_line,
										# spax_noise_line
										]
										)

		#trim to only good observed spaxels to save memory
		spax_properties = spax_properties[spec_good]

		metadata = {'Nx':Nx,'Ny':Ny,'Nl':Nl,'d_spax':spax_size,'z':parameters['z']}

		spax_properties = Table(spax_properties,
					names=['spax_num','vorbin_num','obs_flag','spax_xx','spax_yy',
							'spax_RA','spax_DEC','spax_SPxx','spax_SPyy',
							'spax_signal_cont','spax_noise_cont'],#+\
							# [f'spax_signal_{line}' for line in parameters['SN_line']]+\
							# [f'spax_noise_{line}' for line in parameters['SN_line']],
							meta=metadata)

		spax_properties['vorbin_num'] = np.arange(len(spax_properties),dtype='int')
		
		print("Saving individual spaxel properties")
		#save individual spaxel properties
		spax_properties.write(f"{parameters['output']}/indiv/spaxel_properties.fits",overwrite=True)
		print("Saved")


		#trim spectra to only observed spaxels
		spectra = spectra[:,spec_good]
		noise = noise[:,spec_good]

		#save individual spectra
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
		hdul_indiv.writeto(f"{parameters['output']}/indiv/spectra_indiv.fits",overwrite=True)
		print("Saved")

		#log-rebin the individual spectra
		print("log-rebinning the individual spectra")
		logRebin_spectra, logLambda, velscale = pputils.log_rebin(linLambda_obs,
																	spectra)

		logRebin_noise, logLambda1, velscale1 = pputils.log_rebin(linLambda_obs,
																	noise)
		print("Done")
		
		#save log-rebinned individual spectra
		logRebin_header = fits.Header()
		logRebin_header['velscale'] = velscale[0]
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
		hdul_logRebin.writeto(f"{parameters['output']}/indiv/logRebin_spectra.fits",overwrite=True)
		print("Saved")




	print('Extracting testcube of 20K spectra')

	index_cen_xx = int(Nx*0.5)
	index_cen_yy = int(Ny*0.5)

	yy_num = int(2.e4 / index_cen_xx)
	xx_edge = int(Nx)
	yy_edge = int(index_cen_yy + yy_num)

	index_xx = [index_cen_xx,xx_edge]
	index_yy = [index_cen_yy,yy_edge]


	testcube = extract_subcube(parameters['datacube'],["all",index_xx,index_yy],hdu=1)
	
	data_testcube = testcube.unmasked_data[:].value
	data_header = testcube.header

	testcube = extract_subcube(parameters['datacube'],["all",index_xx,index_yy],hdu=2)
	
	noise_testcube = testcube.unmasked_data[:].value
	noise_header = testcube.header

	data_testcube = np.swapaxes(data_testcube,1,2)
	noise_testcube = np.swapaxes(noise_testcube,1,2)


	hdu0 = fits.PrimaryHDU(header=fits.Header())
	hdu1 = fits.ImageHDU(data_testcube,header=data_header)
	hdu2 = fits.ImageHDU(noise_testcube,header=noise_header)

	hdul = fits.HDUList([hdu0,hdu1,hdu2])
	hdul.writeto(f"{parameters['output']}/testcube/testcube.fits",overwrite=True)


	# spax_xx_testcube, spax_yy_testcube = np.meshgrid(np.arange(index_cen_xx,index_edge_xx))

	spec_nums = [(yy-1)*Nx + (xx-1) for xx in range(index_cen_xx,xx_edge) for yy in range(index_cen_yy,yy_edge)]
	spec_nums = np.sort(spec_nums)

	testcube_locs = np.in1d(spax_properties['spax_num'][:],spec_nums)

	spax_properties_testcube = spax_properties[testcube_locs]
	# spax_properties_testcube['spax_num'] = np.arange(len(spax_properties_testcube))[testcube_locs]
	# print(spax_properties_testcube['spax_num'])
	NX =  np.abs(index_xx[1] - index_xx[0])
	NY = np.abs(index_yy[1] - index_yy[0])
	spax_properties_testcube['spax_xx'] -= index_cen_xx -1
	spax_properties_testcube['spax_yy'] -= index_cen_yy -1
	spax_properties_testcube['spax_num'] = (spax_properties_testcube['spax_yy']-1)*NX + (spax_properties_testcube['spax_xx']-1)
	spax_properties_testcube.meta['NX'] = NX
	spax_properties_testcube.meta['NY'] = NY

	print("Saving testcube individual spaxel properties")
	#save individual spaxel properties
	spax_properties_testcube.write(f"{parameters['output']}/testcube/spaxel_properties.fits",overwrite=True)
	print("Saved")


	spectra_testcube = spectra[:,testcube_locs]
	noise_testcube = noise[:,testcube_locs]

	logRebin_spectra_testcube = logRebin_spectra[:,testcube_locs]
	logRebin_noise_testcube = logRebin_noise[:,testcube_locs]


	#save individual testcube spectra
	indiv_header = fits.Header()
	indiv_header['COMMENT'] = "A.B. Watts"
	indiv_primary_hdu = fits.PrimaryHDU(data = linLambda_obs,
										header = indiv_header)

	indiv_spectra_hdu = fits.BinTableHDU.from_columns(
							fits.ColDefs([
							fits.Column(
							array = spectra_testcube.T,
							name='SPEC',format=str(len(spectra))+'D'
							)]))

	indiv_noise_hdu = fits.BinTableHDU.from_columns(
							fits.ColDefs([
							fits.Column(
							array = noise_testcube.T,
							name='VAR',format=str(len(noise))+'D' 
							)]))

	hdul_indiv = fits.HDUList([indiv_primary_hdu,
								indiv_spectra_hdu,
								indiv_noise_hdu])
	print("Saving indiv. testcube spectra table")
	hdul_indiv.writeto(f"{parameters['output']}/testcube/spectra_indiv.fits",overwrite=True)
	print("Saved")

	
	#save log-rebinned individual testcube spectra
	logRebin_header = fits.Header()
	logRebin_header['velscale'] = velscale[0]
	logRebin_header['COMMENT'] = "A.B. Watts"

	logRebin_primary_hdu = fits.PrimaryHDU(data = logLambda,
											header = logRebin_header)
	logRebin_spectra_hdu = fits.BinTableHDU.from_columns(
							fits.ColDefs([
							fits.Column(
							array = logRebin_spectra_testcube.T,
							name='SPEC',format=str(len(logRebin_spectra))+'D' 
							)]))

	logRebin_noise_hdu = fits.BinTableHDU.from_columns(
							fits.ColDefs([
							fits.Column(
							array = logRebin_noise_testcube.T,
							name='VAR',format=str(len(logRebin_noise))+'D'
							)]))


	hdul_logRebin = fits.HDUList([logRebin_primary_hdu,
									logRebin_spectra_hdu,
									logRebin_noise_hdu])
	
	print("Saving log-rebinnned indiv. testcube spectra table")
	hdul_logRebin.writeto(f"{parameters['output']}/testcube/logRebin_spectra.fits",overwrite=True)
	print("Saved")




	print("Extracting and saving 5 test spectra across cube")

	SNsort = np.argsort(np.array(spax_properties['spax_signal_cont'])/np.array(spax_properties['spax_noise_cont']))
	SN_ids = [SNsort[-1],
			SNsort[int(len(SNsort)*0.9)],
			SNsort[int(len(SNsort)*0.8)],
			SNsort[int(len(SNsort)*0.7)],
			SNsort[int(len(SNsort)*0.5)]]

	locs = [[int(spax_properties['spax_xx'][ii]),int(spax_properties['spax_yy'][ii])]
				for ii in SN_ids]

	spax_properties_test = spax_properties[SN_ids]
	spax_properties_test['spax_num'] = np.arange(len(spax_properties_test),dtype=int)
	spax_properties_test['vorbin_num'] = np.arange(len(spax_properties_test),dtype=int)
	spax_properties_test.meta['NX'] = len(SN_ids)
	spax_properties_test.meta['NY'] = 1

	
	print("Saving test individual spaxel properties")
	spax_properties_test.write(f"{parameters['output']}/test/spaxel_properties.fits",overwrite=True)
	print("Saved")


	spectra_test = spectra[:,SN_ids]
	noise_test = noise[:,SN_ids]

	logRebin_spectra_test = logRebin_spectra[:,SN_ids]
	logRebin_noise_test = logRebin_noise[:,SN_ids]



	#save individual test spectra
	indiv_header = fits.Header()
	indiv_header['COMMENT'] = "A.B. Watts"
	indiv_primary_hdu = fits.PrimaryHDU(data = linLambda_obs,
										header = indiv_header)

	indiv_spectra_hdu = fits.BinTableHDU.from_columns(
							fits.ColDefs([
							fits.Column(
							array = spectra_test.T,
							name='SPEC',format=str(len(spectra))+'D'
							)]))

	indiv_noise_hdu = fits.BinTableHDU.from_columns(
							fits.ColDefs([
							fits.Column(
							array = noise_test.T,
							name='VAR',format=str(len(noise))+'D' 
							)]))

	hdul_indiv = fits.HDUList([indiv_primary_hdu,
								indiv_spectra_hdu,
								indiv_noise_hdu])
	print("Saving indiv. test spectra table")
	hdul_indiv.writeto(f"{parameters['output']}/test/spectra_indiv.fits",overwrite=True)
	print("Saved")

	
	#save log-rebinned individual test spectra
	logRebin_header = fits.Header()
	logRebin_header['velscale'] = velscale[0]
	logRebin_header['COMMENT'] = "A.B. Watts"

	logRebin_primary_hdu = fits.PrimaryHDU(data = logLambda,
											header = logRebin_header)
	logRebin_spectra_hdu = fits.BinTableHDU.from_columns(
							fits.ColDefs([
							fits.Column(
							array = logRebin_spectra_test.T,
							name='SPEC',format=str(len(logRebin_spectra))+'D' 
							)]))

	logRebin_noise_hdu = fits.BinTableHDU.from_columns(
							fits.ColDefs([
							fits.Column(
							array = logRebin_noise_test.T,
							name='VAR',format=str(len(logRebin_noise))+'D'
							)]))


	hdul_logRebin = fits.HDUList([logRebin_primary_hdu,
									logRebin_spectra_hdu,
									logRebin_noise_hdu])
	
	print("Saving log-rebinnned indiv. test spectra table")
	hdul_logRebin.writeto(f"{parameters['output']}/test/logRebin_spectra.fits",overwrite=True)
	print("Saved")

def read_spectra_hdul(hdul):

	wave = np.asarray(hdul[0].data)
	spectra = []
	for hdu in hdul[1::]:
		spectra.append(np.asarray([dd[0] for dd in hdu.data]).T)

	return wave, spectra

def voronoi_bin_cube(parameterfile):

	parameters = read_parameterfile(parameterfile)

	spax_prop_file = f"{parameters['input_dir']}/spaxel_properties.fits"
	SN_indiv = parameters['vorbin_SNmin']
	SN_vorbin = parameters['vorbin_SNtarget']


	spax_properties = Table.read(spax_prop_file)
	spax_size = spax_properties.meta['D_SPAX'] * 3600.

	# #value for where specta don't make min SN cut
	spax_properties['vorbin_num'] =  np.full(len(spax_properties),-1)
	spax_properties['vorbin_xx'] = np.full(len(spax_properties),-1)
	spax_properties['vorbin_yy'] = np.full(len(spax_properties),-1)
	spax_properties['vorbin_SN'] = np.full(len(spax_properties),-1)

	#currently not saving non-observed spectra
	# #value for where spectra are not observed
	# spax_properties['vorbin_num'][spax_properties['obs_flag'][:] == 0] = -99
	# spax_properties['vorbin_xx'][spax_properties['obs_flag'][:] == 0] = -99
	# spax_properties['vorbin_yy'][spax_properties['obs_flag'][:] == 0] = -99
	# spax_properties['vorbin_SN'][spax_properties['obs_flag'][:] == 0] = -99

	SNnames = parameters['vorbin_SNname']

	if len(SNnames) == 1:

		#get spaxels above desired SN
		spax_SN = spax_properties[f'spax_signal_{SNnames[0]}'] /  spax_properties[f'spax_noise_{SNnames[0]}']

		refs = np.where(spax_SN >= SN_indiv)[0]

		spax_xx = spax_properties['spax_xx'][refs]
		spax_yy = spax_properties['spax_yy'][refs]
		spax_signal = spax_properties[f'spax_signal_{SNnames[0]}'][refs]
		spax_noise = spax_properties[f'spax_noise_{SNnames[0]}'][refs]

		sn_func = None
	elif len(SNnames) > 1:

		refs =  np.arange(len(spax_properties),dtype=int)
		spax_emline_fluxes = Table.read(f"{parameters['input_dir']}/spax_emline_fluxes.fits")

		# spax_properties_temp = hstack((spax_properties,spax_emline_fluxes))
		spax_xx = np.full([len(SNnames),len(spax_properties)],spax_properties['spax_xx']).T
		spax_yy = np.full([len(SNnames),len(spax_properties)],spax_properties['spax_yy']).T

		spax_signal = np.zeros([len(spax_properties),len(SNnames)])
		spax_noise = np.zeros([len(spax_properties),len(SNnames)])
		
		for nn, name in enumerate(SNnames):
			if name ==  "cont" :
				spax_signal[:,nn] = spax_properties[f'spax_signal_{name}']
				spax_noise[:,nn] = spax_properties[f'spax_noise_{name}']

			else:
				spax_signal[:,nn] = spax_emline_fluxes[name]
				spax_noise[:,nn] = spax_emline_fluxes[f'{name}_err']

				noise_median = np.nanmedian(spax_noise[spax_noise[:,nn]>0,nn])
				print(noise_median)
				print(0.5*noise_median)
				spax_signal[spax_noise[:,nn]==0,nn] =  0.5*noise_median
				spax_noise[spax_noise[:,nn]==0,nn] =  noise_median



		sn_func = voronoi_multiSN_function


	# print(spax_signal)
	# print(spax_noise)
	# plt.hist(spax_noise[:,1],bins=100)
	# plt.show()
	# print(spax_noise[np.argsort((spax_signal/spax_noise)[:,1])[0],1])
	# exit()

	#compute Voronoi bins
	print('Computing voronoi bins')
	vorbin_nums, Nxx, Nyy, vorbin_xx, vorbin_yy, vorbin_SN, vorbin_Npix, scale = \
					voronoi_2d_binning(spax_xx, spax_yy, 
										spax_signal, spax_noise, 
										SN_vorbin, pixelsize=spax_size,
										plot = False,
										sn_func = sn_func,
										quiet=True)


	#record vorbin properties for each spaxel
	spax_properties['vorbin_num'][refs] = vorbin_nums
	# spax_properties['vorbin_xx'][spax_SN >= SN_indiv] = vorbin_xx
	# spax_properties['vorbin_yy'][spax_SN >= SN_indiv] = vorbin_yy
	# spax_properties['vorbin_SN'][spax_SN >= SN_indiv] = vorbin_SN
	for vv, num in enumerate(np.sort(np.unique(vorbin_nums))):

		spax_properties['vorbin_xx'][vorbin_nums == num] =\
																		vorbin_xx[vv]

		spax_properties['vorbin_yy'][vorbin_nums == num] =\
																		vorbin_yy[vv]

		spax_properties['vorbin_SN'][vorbin_nums == num] =\
																		vorbin_SN[vv]


	if not os.path.isdir(f"{parameters['output_dir']}/"):
		os.mkdir(f"{parameters['output_dir']}/")
		os.mkdir(f"{parameters['output_dir']}/figures")

	# spax_properties = spax_properties[spax_properties.argsort(spax_properties['vorbin_num'])]
	spax_properties.write(f"{parameters['output_dir']}/spaxel_properties.fits",overwrite=True)

	create_binned_spectra(parameters)
	make_vorbins_map(parameters)

def voronoi_multiSN_function(index, signal =  None, noise = None):

	if signal.ndim ==1:
		sn = np.sum(signal[index]) / np.sqrt(np.sum(noise**2))


	elif signal.ndim > 1:
		# print(signal)

		# print(noise)
		# print('flag')
		# print(signal.ndim)
		# print(signal.shape)
		# print(signal[index])
		# print(noise[index])
		# exit()
		if isinstance(index,int) or isinstance(index,np.int64) or len(index)==1:
			SN = signal[index,:] / np.sqrt(noise[index,:]**2)
			# print('flag1')


		elif len(index)>1:
			SN = np.sum(signal[index,:],axis = 0) / np.sqrt(np.sum(noise[index,:]**2,axis = 0))
			# print('flag2')
		
		sn = np.min(SN)
			# print(SN)
			# print(sn)
			# exit()
	# exit()


	return sn

def create_binned_spectra(parameters = None):

	if parameters is None:
		parameters = read_parameterfile()

	# print(parameters)
	spax_prop_file = f"{parameters['output_dir']}/spaxel_properties.fits"
	
	spax_properties = Table.read(spax_prop_file)

	# spectra_file = parameters['vorbin_spectra_file']
	# if isinstance(spectra_file,type(None)):
	spectra_file = f"{parameters['input_dir']}/spectra_indiv.fits"

	hdul = fits.open(spectra_file)
	header = hdul[0].header
	#is there a faster way to read these?
	linLambda_obs, spectra_list = read_spectra_hdul(hdul)
	hdul.close()
	spectra = spectra_list[0]
	noise = spectra_list[1]
	# varbin_flag = True
	# else:
	# 	hdul = fits.open(spectra_file)
	# 	header = hdul[0].header
	# 	#is there a faster way to read these?
	# 	linLambda_obs, spectra_list = read_spectra_hdul(hdul)
	# 	spectra = spectra_list[0]
	# 	varbin_flag = False
	# 	# print(header)
	
	# if "logRebin" in spectra_file:				#you're summing something already log-rebinned, like a cont. sub. cube
	# 	logRebin_flag = False
	# else:
	# 	logRebin_flag = True



	vorbin_nums = np.unique(spax_properties['vorbin_num'])
	vorbin_nums = np.sort(vorbin_nums[vorbin_nums >= 0])


	# if logRebin_flag:

	vorbin_spectra = np.zeros([len(spectra),len(vorbin_nums)])
	vorbin_noise = vorbin_spectra.copy()

	for nn in vorbin_nums:

		inbin = np.in1d(spax_properties['vorbin_num'][:], nn)

		vorbin_spectra[:,nn] = np.sum(spectra[:,inbin], axis=1)
		vorbin_noise[:,nn] = np.sum(noise[:,inbin], axis=1)

	# else:
	# 	logRebin_vorbin_spectra = np.zeros([len(spectra),len(vorbin_nums)])
	# 	for nn in vorbin_nums:
	# 		inbin = np.where(spax_properties['vorbin_num'][:] == nn)[0]
	# 		logRebin_vorbin_spectra[:,nn] = np.sum(spectra[:,inbin], axis=1)
	# 	logLambda = linLambda_obs

	# if logRebin_flag:
		#save vorbin spectra
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

	# if varbin_flag:
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

	logRebin_vorbin_spectra, logLambda, velscale = pputils.log_rebin(linLambda_obs,
																vorbin_spectra)
	
	# if varbin_flag:
	logRebin_vorbin_noise, logLambda1, velscale1 = pputils.log_rebin(linLambda_obs,
																vorbin_noise)

	#save log-rebinned vorbin spectra
	logRebin_vorbin_header = fits.Header()
	# logRebin_vorbin_header['VELSCALE'] = velscale[0]
	logRebin_vorbin_header['VELSCALE'] = velscale[0]
	logRebin_vorbin_header['COMMENT'] = "A.B. Watts"
	logRebin_vorbin_primary_hdu = fits.PrimaryHDU(data=logLambda,
												header = logRebin_vorbin_header)
	logRebin_vorbin_spectra_hdu = fits.BinTableHDU.from_columns(
							fits.ColDefs([
							fits.Column(
							array = logRebin_vorbin_spectra.T,
							name='SPEC',format=str(len(logRebin_vorbin_spectra))+'D'
							)]))
	hdul_logRebin_vorbin = fits.HDUList([logRebin_vorbin_primary_hdu,
								logRebin_vorbin_spectra_hdu])
	# if varbin_flag:
	logRebin_vorbin_noise_hdu = fits.BinTableHDU.from_columns(
							fits.ColDefs([
							fits.Column(
							array = logRebin_vorbin_noise.T,
							name='VAR',format=str(len(logRebin_vorbin_noise))+'D' 
							)]))

	
	hdul_logRebin_vorbin.append(logRebin_vorbin_noise_hdu)
	print("Saving log-rebinnned. vorbin spectra table")
	hdul_logRebin_vorbin.writeto(f"{parameters['output_dir']}/logRebin_spectra.fits",overwrite=True)
	print("Saved")
###


###template stuff
def read_EMILES():
	template_dir = f"{os.path.dirname(os.path.realpath(__file__))}/templates/EMILES"

	files = glob.glob(template_dir + "/*")

	# template_params = np.array([[],[],[]])
	template_params = []

	params = ["ch","Z","T","_iTp","_"]

	for ff in range(len(files)):
	
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

	temp_sort = np.lexsort((template_params[:,2],
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

	


	for ff in range(len(files)):

		file = files[ff]
		hdu = fits.open(file)
		template = hdu[0].data
		header = hdu[0].header
		hdu.close()

		if ff == 0:
			wave_step = header['CDELT1']
			lambda0 = header['CRVAL1']
			linLambda_templates = lambda0 + np.arange(header['NAXIS1'])*header['CDELT1']
			templates = np.zeros([header['NAXIS1'],len(files)])
			templates = templates[(linLambda_templates>=4000) & (linLambda_templates<10000),:]
			
			linLambda_templates_trunc = linLambda_templates[(linLambda_templates>=4000) & (linLambda_templates<10000)]

		
		template = template[(linLambda_templates>=4000) & (linLambda_templates<10000)]

		templates[:,ff] =  template

	return templates, linLambda_templates_trunc, wave_step

def read_XSL():	
	template_dir = f"{os.path.dirname(os.path.realpath(__file__))}/templates/XSL_sorted"

	files = glob.glob(template_dir + "/*X064*")

	for ff in range(len(files)):

		file = files[ff]
		hdu = fits.open(file)
		data = hdu[1].data
		header = hdu[1].header
		hdu.close()

		linLambda_templates = np.array(data.field(0))*10.
		template = np.array(data.field(2))
		# err = np.array(data.field(3))
		wave_step = np.diff(linLambda_templates)
		wave_step = wave_step[(linLambda_templates[0:-1]>=4000) & (linLambda_templates[0:-1]<10000)]


		if ff == 0:
			
			templates = np.zeros([len(linLambda_templates),len(files)])			
			templates = templates[(linLambda_templates>=4000) & (linLambda_templates<10000),:]
			linLambda_templates_trunc = linLambda_templates[(linLambda_templates>=4000) & (linLambda_templates<10000)]

		
		template = template[(linLambda_templates>=4000) & (linLambda_templates<10000)]

		templates[:,ff] =  template

	

	return  templates, linLambda_templates_trunc,  wave_step, files

def read_BPASS():
	template_dir = f"{os.path.dirname(os.path.realpath(__file__))}//templates/BPASS"

	files = glob.glob(template_dir + "/*")


	wave_step = 1

	for ff in range(len(files)):

		file = files[ff]
		data = np.loadtxt(file)

		if ff == 0:
			linLambda_templates = data[:,0]

			wave_shorten = np.logical_and((linLambda_templates>=4000), (linLambda_templates<10000))

			linLambda_templates_trunc = linLambda_templates[wave_shorten]
			templates = np.zeros([linLambda_templates_trunc.shape[0],51*len(files)])


		templates[:,ff*51:(ff+1)*51] = data[wave_shorten,1::]



	return  templates, linLambda_templates_trunc,  wave_step

def get_stellar_templates(parameters, match_velscale=True, convolve = True, regrid=False):

	if parameters is None:
		parameters = read_parameterfile()

	if parameters['stellar_templates'] == "EMILES":
		stellar_templates, linLambda_templates, wave_step  = read_EMILES()
		FWHM_template = np.full_like(linLambda_templates,2.51)


	if parameters['stellar_templates'] == "XSL":
		stellar_templates, linLambda_templates, wave_step,files  = read_XSL()
		# FWHM_template = np.zeros_like(linLambda_templates)
	# 	UV_range = np.logical_and(linLambda_templates>, linLambda_templates<)
	# 	VIS_range = np.logical_and(linLambda_templates>, linLambda_templates<)
	# 	NIR_range = np.logical_and(linLambda_templates>, linLambda_templates<)
	# 	FWHHM_template[UV_range] = (13/c)
	# 	11/c
	# 	15/c

	if parameters['stellar_templates'] == "BPASS":
		stellar_templates, linLambda_templates, wave_step  = read_BPASS()
		FWHM_template = np.full_like(linLambda_templates,1)


	stellar_templates_conv = np.zeros_like(stellar_templates)
	for tt in range(stellar_templates.shape[1]):
		template = stellar_templates[:,tt]

		if convolve:
			FWHM_galaxy = MUSE_LSF_Bacon17(linLambda_templates,z = parameters['z'])
			FWHM_diffs = np.sqrt(FWHM_galaxy**2.e0 - FWHM_template**2.e0 )
			FWHM_diffs[np.isfinite(FWHM_diffs)==False] = 0.01

			# stddev_diffs = FWHM_diffs / (wave_step  * 2.355)					###VARSMOOTH TAKES UNITS OF WAVELENGTH, NOT PIXELS
			stddev_diffs = FWHM_diffs / (wave_step  * 2.355)

			stellar_templates_conv[:,tt] = pputils.varsmooth(linLambda_templates,template,stddev_diffs)

		else:
			stellar_templates_conv[:,tt] = template

		# template_conv = np.zeros(len(linLambda_templates))
		# for ll in range(len(linLambda_templates)):
		# 	template_temp = np.zeros(len(linLambda_templates))
		# 	template_temp[ll] = template[ll]
		# 	if np.isfinite(stddev_diffs[ll]):
		# 		template_conv += ndimage.gaussian_filter1d(template_temp, stddev_diffs[ll])
		# 	else:
		# 		template_conv += template_temp
		# if tt==14:
			# plt.plot(template_conv)
			# plt.show()
			# exit()

	stellar_templates_conv = stellar_templates_conv[:,np.where(np.prod(np.isfinite(stellar_templates_conv),axis=0)==1)[0]]

	if match_velscale:
		template_velscale = parameters['galaxy_velscale'] / parameters['velscale_ratio']
	else:
		template_velscale = None


	stellar_templates_logRebin, logLambda_templates, temp_velscale = pputils.log_rebin(linLambda_templates,
															stellar_templates_conv,
															velscale = template_velscale)

	if not regrid:
		stellar_templates_final = stellar_templates_logRebin
		stellar_templates_final /= np.median(stellar_templates_final,axis=0)

	elif regrid:

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

def make_gas_templates(parameters, logLambda_templates, convolve = True, pixel = True):
	
	gas_templates = []

	if convolve:
		FWHM = lambda ll: MUSE_LSF_Bacon17(ll, parameters['z'])
	else:
		FWHM = 0

	gas_groups = parameters['gas_groups']
	for gg, group in enumerate(gas_groups):
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

def get_templates(parameters):

	logLambda_templates, templates = get_stellar_templates(parameters)
	components_stars = [0]*len(templates[0,:])


	if not isinstance(parameters['gas_groups'],type(None)):
		gas_templates = make_gas_templates(parameters,logLambda_templates)
		templates = np.column_stack((templates,gas_templates))

		components_gas = [[nn + 1 + components_stars[-1]]*NN for nn,NN in enumerate(parameters['gas_Ncomp'])]
		components_gas = [comp for group in components_gas for comp in group]
		components = components_stars + components_gas

		gas_components = np.array(components) > components_stars[-1]
	else:
		components = components_stars
		gas_components = None


	return logLambda_templates, templates, components, gas_components

def get_stellar_templates_testing(parameters, match_velscale=True, convolve = True, logRebin = True, regrid=False):

	if parameters is None:
		parameters = read_parameterfile()

	if parameters['stellar_templates'] == "EMILES":
		stellar_templates, linLambda_templates, wave_step  = read_EMILES()
		FWHM_template = np.full_like(linLambda_templates,2.51)


	if parameters['stellar_templates'] == "XSL":
		stellar_templates, linLambda_templates, wave_step,files  = read_XSL()
		# FWHM_template = np.zeros_like(linLambda_templates)
	# 	UV_range = np.logical_and(linLambda_templates>, linLambda_templates<)
	# 	VIS_range = np.logical_and(linLambda_templates>, linLambda_templates<)
	# 	NIR_range = np.logical_and(linLambda_templates>, linLambda_templates<)
	# 	FWHHM_template[UV_range] = (13/c)
	# 	11/c
	# 	15/c

	if parameters['stellar_templates'] == "BPASS":
		stellar_templates, linLambda_templates, wave_step  = read_BPASS()
		FWHM_template = np.full_like(linLambda_templates,1)


	stellar_templates_conv = np.zeros_like(stellar_templates)
	for tt in range(stellar_templates.shape[1]):
		template = stellar_templates[:,tt]

		if convolve:
			FWHM_galaxy = MUSE_LSF_Bacon17(linLambda_templates,z = parameters['z'])
			FWHM_diffs = np.sqrt(FWHM_galaxy**2.e0 - FWHM_template**2.e0 )
			FWHM_diffs[np.isfinite(FWHM_diffs)==False] = 0.01

			# stddev_diffs = FWHM_diffs / (wave_step  * 2.355)					###VARSMOOTH TAKES UNITS OF WAVELENGTH, NOT PIXELS
			stddev_diffs = FWHM_diffs / (wave_step  * 2.355)

			stellar_templates_conv[:,tt] = pputils.varsmooth(linLambda_templates,template,stddev_diffs)

		else:
			stellar_templates_conv[:,tt] = template

		# template_conv = np.zeros(len(linLambda_templates))
		# for ll in range(len(linLambda_templates)):
		# 	template_temp = np.zeros(len(linLambda_templates))
		# 	template_temp[ll] = template[ll]
		# 	if np.isfinite(stddev_diffs[ll]):
		# 		template_conv += ndimage.gaussian_filter1d(template_temp, stddev_diffs[ll])
		# 	else:
		# 		template_conv += template_temp
		# if tt==14:
			# plt.plot(template_conv)
			# plt.show()
			# exit()

	stellar_templates_conv = stellar_templates_conv[:,np.where(np.prod(np.isfinite(stellar_templates_conv),axis=0)==1)[0]]

	if match_velscale:
		template_velscale = parameters['galaxy_velscale'] / parameters['velscale_ratio']
	else:
		template_velscale = None


	if logRebin:
		lambda_range = [np.min(linLambda_templates),np.max(linLambda_templates)]
		stellar_templates_logRebin, logLambda_templates, temp_velscale = pputils.log_rebin(linLambda_templates,
																stellar_templates_conv,
																velscale = template_velscale)
	else:
		stellar_templates_logRebin = stellar_templates_conv
		logLambda_templates = linLambda_templates


	if not regrid:
		stellar_templates_final = stellar_templates_logRebin
		stellar_templates_final /= np.median(stellar_templates_final,axis=0)

	elif regrid:

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

def create_spectrum_mask(logLambda, parameters):

	goodpix = np.zeros_like(logLambda,dtype=bool)
	z = parameters['z']
	linLambda = np.exp(logLambda)

	width = 400. / c

	gas_fit = False
	if not isinstance(parameters['gas_groups'],type(None)):
		gas_fit = True



	#					OI 			OI 		OI 
	sky = np.array([5577.338, 6300.304, 6363.78])/(1.e0+z)
	#				weird sky feature?
	# sk1 = np.array([5890.5])/(1.e0+z)
	#									NaD 			CaT
	absorption_lines = np.array([5889.95, 5895.92, 8498., 8542., 8662.]) 
	if parameters['fit_CaT'] == True:
		absorption_lines = absorption_lines[:2]


	if gas_fit:
		emission_lines_fit = [line for group in parameters['gas_groups'] for line in group]
		emission_lines_fit = np.unique(np.array(emission_lines_fit))
	else:
		emission_lines_fit = None

	emlines_temp = emlines.copy()
	if parameters['fit_CaT'] == True:
		del emlines_temp['Pa13']
		del emlines_temp['Pa15']
		del emlines_temp['Pa16']


	emission_lines = np.array([])
	for line in emlines_temp:

		if not gas_fit:
			emission_lines = np.append(emission_lines,
								np.array(emlines_temp[line]['lambda']))
		elif line not in emission_lines_fit:
			emission_lines = np.append(emission_lines,
								np.array(emlines_temp[line]['lambda']))

	lines = np.concatenate((emission_lines,absorption_lines,sky),axis=None)

	in_spec = np.logical_and(lines>=np.min(linLambda), lines<=np.max(linLambda))
	
	lines = lines[in_spec]
	for line in lines:
		min_Lambda = line - line*width
		max_Lambda = line + line*width

		goodpix += np.logical_and(linLambda>=min_Lambda, linLambda<= max_Lambda)
	

	#telluric feature
	min_Lambda = (7650 / (1+z)) * (1-2500/c) 
	max_Lambda = (7650 / (1+z)) * (1+2500/c) 
	goodpix += np.logical_and(linLambda>=min_Lambda, linLambda<= max_Lambda)

	goodpix = ~goodpix

	return goodpix

def get_constraints(parameters, spaxel_properties, vorbin_nums):

	# if 'gas_constraints' not in parameters.keys():
	# 	# print('entering default dummy constraint')

	# 	# constr_kinem = {'A_ineq':[[0.,-1.,0.,0.]],'b_ineq':[0]}
	# 	constr_kinem = None
	# elif parameters['gas_constraints']['A_ineq'] == []:
	# 		# print('entering default dummy constraint')
	# 		# constr_kinem = {'A_ineq':[[0.,-1.,0.,0.] + [0.,0.]*len(parameters['gas_groups'])],'b_ineq':[0]}
	# 		constr_kinem = None

	# else:
		

	# print(parameters['gas_constraints'])


	if 'cont_stelkin' in parameters.keys():
		stel_kin_file = f"{parameters['output']}/{parameters['cont_stelkin']}/bestfit_stellar_kinematics.fits"
		stel_kin = Table.read(stel_kin_file)
		# stel_kin_keys = stel_kin.keys()[1::]
		# stel_kin_keys = stel_kin_keys[['err' not in key for key in stel_kin_keys]]
	else:
		stel_kin = None
	# print(stel_kin)

	constraints = []
	start = []
	moments = []

	for vv, vb in enumerate(vorbin_nums):

		# if np.all(spax_properties['gas_Ncomp'][refs] == spax_properties['gas_Ncomp'][refs][0])  # for when I put gas Ncomp keywords in
		
		constr_kinem = {'A_ineq':[],'b_ineq':[]}

		refs = np.where(spaxel_properties['vorbin_num'] == vb)[0]
		fixed_flag = 0
		if not isinstance(stel_kin,type(None)):
			stel_kin_vorbin = stel_kin[refs]
			# print(vb,'flag')
			# maxmin = 
			# for key in stel_kin_keys:


			V_max =  np.max(stel_kin_vorbin['V_stellar'])
			V_min =  np.min(stel_kin_vorbin['V_stellar'])
			sigma_max =  np.max(stel_kin_vorbin['sigma_stellar'])
			sigma_min =  np.min(stel_kin_vorbin['sigma_stellar'])
			h3_max =  np.max(stel_kin_vorbin['h3_stellar'])
			h3_min =  np.min(stel_kin_vorbin['h3_stellar'])
			h4_max =  np.max(stel_kin_vorbin['h4_stellar'])
			h4_min =  np.min(stel_kin_vorbin['h4_stellar'])

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
				# print('V the same')
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
				# print('sigma the same')
			if h3_max != h3_min:
				if h3_min == 0:
					h3_min -= 2
				if h3_max == 0:
					h3_max += 2
				constr_kinem['A_ineq'].append([0,0,1,0] + [0,0]*len(parameters['gas_Ncomp']) )
				constr_kinem['b_ineq'].extend([ (h3_max+0.1) / parameters['galaxy_velscale'] ])
				constr_kinem['A_ineq'].append([0,0,-1,0] + [0,0]*len(parameters['gas_Ncomp']))
				constr_kinem['b_ineq'].extend([ -1.e0* (h3_min-0.1) / parameters['galaxy_velscale'] ])
			else:
				fixed_flag += 1
				# print('h3 the same')
			if h4_max != h4_min:
				if h4_min == 0:
					h4_min -= 2
				if h4_max == 0:
					h4_max += 2
				# constr_kinem['A_ineq'].append([0,0,0,1] + [0,0]*len(parameters['gas_Ncomp']))
				# constr_kinem['b_ineq'].extend([ h4_max / parameters['galaxy_velscale'] ])
				# constr_kinem['A_ineq'].append([0,0,0,-1] + [0,0]*len(parameters['gas_Ncomp']) )
				# constr_kinem['b_ineq'].extend([ -1.e0* h4_min / parameters['galaxy_velscale'] ])
			else:
				fixed_flag += 1
				# print('h4 the same')


			start_vorbin = [[np.mean(stel_kin_vorbin['V_stellar']), np.mean(stel_kin_vorbin['sigma_stellar']),np.mean(stel_kin_vorbin['h3_stellar']),np.mean(stel_kin_vorbin['h4_stellar'])] ] 

		else:
			start_vorbin = parameters['stars_start']



		if not isinstance(parameters['gas_constraints'],type(None)):
			gas_constraints = parameters['gas_constraints']
			for aa in range(len(gas_constraints['A_ineq'])):
				constr_kinem['A_ineq'].append( [0.,0.,0.,0.] + gas_constraints['A_ineq'][aa] )
				constr_kinem['b_ineq'].extend( [ gas_constraints['b_ineq'][aa] / parameters['galaxy_velscale'] ])


		if constr_kinem['A_ineq'] == []:
			constraints.append(None)
		else:
			# if len(constr_kinem['A_ineq']) == 1:
				# constr_kinem['A_ineq'] = constr_kinem['A_ineq'][0]
			constraints.append(constr_kinem)

		#get starting estimates
		# print(parameters['gas_Ncomp'])
		if len(parameters['gas_Ncomp'])>0:
			start_vorbin = [start_vorbin] + [[0,30.]] 
		if len(parameters['gas_Ncomp'])>1:
			start_vorbin = start_vorbin +[[0,500]]*(len(parameters['gas_Ncomp'])-1)

		start.append(start_vorbin)

		moments_vorbin = [parameters['stars_moments']] + [2] * len(parameters['gas_Ncomp'])
		if fixed_flag==4:
			moments_vorbin[0] *= -1
		moments.append(moments_vorbin)



	# constr_kinem_spec = copy.deepcopy(constraints)
	# if parameters['cont_stelkin_Vfrac'] != 1:
	# 	# A_ineq = constraints['A_ineq']
	# 	# b_ineq = constraints['b_ineq']
	# 	if stel_kin['V_stellar'][vorbin_num] == 0:
	# 		upper = 10 / parameters['galaxy_velscale']
	# 		lower = -10 / parameters['galaxy_velscale']
	# 	else:
	# 		upper = stel_kin['V_stellar'][vorbin_num] * parameters['cont_stelkin_Vfrac'] / parameters['galaxy_velscale']
	# 		lower = stel_kin['V_stellar'][vorbin_num] / parameters['cont_stelkin_Vfrac'] / parameters['galaxy_velscale']

	# 	constr_kinem_spec['A_ineq'].append([1,0,0,0] + [0,0]*len(Ncomp_gas))
	# 	constr_kinem_spec['b_ineq'].extend([upper])

	# 	constr_kinem_spec['A_ineq'].append([-1,0,0,0] + [0,0]*len(Ncomp_gas))
	# 	constr_kinem_spec['b_ineq'].extend([-lower])

	# 	# constr_kinem_spec = {'A_ineq':A_ineq, 'b_ineq' : b_ineq}

	# else:
	# 	moments[0] = -moments[0]



	return start, moments, constraints
###


###making maps
def make_vorbins_map(parameters):
	spax_prop_file = f"{parameters['output_dir']}/spaxel_properties.fits"


	spax_properties = Table.read(spax_prop_file)
	meta = spax_properties.meta
	NY = meta['NY']
	NX = meta['NX']

	img_grid = np.zeros([NY,NX]).flatten()
	# print(np.asarray(spax_properties['spax_num'],dtype=int))
	img_grid[np.asarray(spax_properties['spax_num'],dtype=int)] = spax_properties['vorbin_num']
	img_grid = img_grid.reshape((NY,NX))



	fig = plt.figure()
	gs = gridspec.GridSpec(1,1)

	ax = fig.add_subplot(gs[0,0])
	ax.pcolormesh(img_grid,cmap='prism')
	# ax.scatter(spax_properties['spax_xx'],spax_properties['spax_yy'],c=spax_properties['vorbin_num'])
	# plt.show()
	fig.savefig(f"{parameters['output_dir']}/figures/vorbin_map.pdf")

def make_stelkin_map(parameters):
	spax_prop_file = f"{parameters['input_dir']}/spaxel_properties.fits"

	fit_stellar_kinematics = f"{parameters['output_dir']}/bestfit_stellar_kinematics.fits"

	spax_properties = Table.read(spax_prop_file)
	bestfit_stelkin = Table.read(fit_stellar_kinematics)


	spax_properties = hstack((spax_properties,bestfit_stelkin))

	meta = spax_properties.meta
	NY = meta['NY']
	NX = meta['NX']

	img_grid = np.full([NY,NX],np.nan).flatten()
	# print(np.asarray(spax_properties['spax_num'],dtype=int))

	vorbin_nums = np.unique(spax_properties['vorbin_num'])
	refs = np.zeros_like(vorbin_nums,dtype=int)
	for vv, vb in enumerate(vorbin_nums):
		refs[vv] = np.where(spax_properties['vorbin_num'] == vb)[0][0]


	for comp in ['V_stellar','sigma_stellar','h3_stellar','h4_stellar']:


		fig = plt.figure()
		gs = gridspec.GridSpec(1,1)
		ax = fig.add_subplot(gs[0,0])

		img = img_grid.copy()
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
			# vmax = np.percentile(img[np.isfinite(img)].flatten(),84)
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
		# ax.scatter(spax_properties['spax_xx'],spax_properties['spax_yy'],c=spax_properties['vorbin_num'])
		# plt.show()
		fig.savefig(f"{parameters['output_dir']}/figures/{comp}_map.pdf")
		plt.close()
###


##### things for analysing outputs #####
def make_continuum_subtracted_spectra(parameterfile):

	parameters = read_parameterfile(parameterfile)

	spax_properties = Table.read(f"{parameters['input_dir']}/spaxel_properties.fits")
	Nx = spax_properties.meta['NX']
	Ny = spax_properties.meta['NY']

	spectra_file = f"{parameters['input_dir']}/logRebin_spectra.fits"
	hdul = fits.open(spectra_file)
	logLambda, spectra = read_spectra_hdul(hdul)
	hdul.close()
	galaxy_spectra = spectra[0]
	noise_spectra = spectra[1]

	bestfit_continuum_file = f"{parameters['output_dir']}/bestfit_continuum_spectra.fits"
	hdul = fits.open(bestfit_continuum_file)
	logLambda, spectra = read_spectra_hdul(hdul)
	hdul.close()
	bestfit_continuum = spectra[0]
	# logLambda = np.log(logLambda)

	contsub_spectra = galaxy_spectra - bestfit_continuum

	Nl = logLambda.shape[0]

	empty_cube = np.zeros([Nl,Nx*Ny])

	galaxy_cube = empty_cube.copy()
	noise_cube = empty_cube.copy()
	continuum_cube = empty_cube.copy()
	contsub_cube = empty_cube.copy()


	for vv in range(galaxy_spectra.shape[1]):
		inbin = np.where(spax_properties['vorbin_num']==vv)[0]
		spax_num_inbin = np.array(spax_properties['spax_num'][inbin],dtype=int)
		
		galaxy_cube[:,spax_num_inbin] = np.full((len(inbin),Nl),galaxy_spectra[:,vv]).T
		noise_cube[:,spax_num_inbin] = np.full((len(inbin),Nl),noise_spectra[:,vv]).T
		continuum_cube[:,spax_num_inbin] = np.full((len(inbin),Nl),bestfit_continuum[:,vv]).T
		contsub_cube[:,spax_num_inbin] =  np.full((len(inbin),Nl),contsub_spectra[:,vv]).T


	# vorbin_nums = np.unique(spaxel_properties['vorbin_num'][:])
	# vorbin_nums = vorbin_nums[vorbin_nums>=0]
	# for bb in vorbin_nums:
	# 	inbin = np.where(spaxel_properties['vorbin_num'][:] == bb)[0]

	# 	galaxy_cube[:,inbin] = galaxy_spectra[bb]
	# 	continuum_cube[:,inbin] = bestfit_continuum[bb]
	# 	contsub_cube[:,inbin] =  contsub_spectra[bb]


	galaxy_cube = galaxy_cube.reshape(Nl,Ny,Nx)
	noise_cube = noise_cube.reshape(Nl,Ny,Nx)
	continuum_cube = continuum_cube.reshape(Nl,Ny,Nx)
	contsub_cube = contsub_cube.reshape(Nl,Ny,Nx)
	

	header = fits.Header()
	header['COMMENT'] = "A.B. Watts"
	header['COMMENT'] = "LOG LAMBDA IS IN THIS HDU "
	primary_hdu = fits.PrimaryHDU(data = logLambda,
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


	# print("Saving datacubes")

	# continuum_cube = np.zeros([Nl,Nx*Ny])
	# continuum_cube[:,np.array(spax_properties['spax_num'][:],dtype=int)] = bestfit_continuum
	# continuum_cube = continuum_cube.reshape(Nl,Ny,Nx)

	# contsub_cube = np.zeros([Nl,Nx*Ny])
	# contsub_cube[:,np.array(spax_properties['spax_num'][:],dtype=int)] = contsub_spectra
	# contsub_cube = contsub_cube.reshape(Nl,Ny,Nx)


	cube_file = "/Users/00088350/Research/projects/NGC4383/MUSE/spectral_fitting/outputs/testcube/testcube.fits"
	cube_hdu = fits.open(cube_file)
	cube_header = cube_hdu[1].header
	cube_hdu.close()
	# cube_header['NAXIS3'] = len(logLambda)
	cube_header['CRVAL3'] = logLambda[0]
	cube_header['CRPIX3'] = 1
	print(cube_header.keys())
	if any(key == "CDELT3" for key in cube_header.keys()):
		cube_header['CDELT3'] = np.abs(np.diff(logLambda))[0]
	else:
		cube_header['CD3_3'] = np.abs(np.diff(logLambda))[0]

	cube_header['CTYPE3'] = "AWAV-LOG"


	# primary_hdu = fits.PrimaryHDU(header = fits.Header())
	# galaxy_cube_hdu = fits.ImageHDU(galaxy_cube,
	# 						name='galaxy',
	# 						header = cube_header)
	# hdul = fits.HDUList([primary_hdu,
	# 							galaxy_cube_hdu])
	# print("Saving log-rebinned galaxy cube")
	# hdul.writeto(f"{parameters['output_dir']}/logRebingalaxy_cube.fits",overwrite=True)
	# print("Saved")



	# primary_hdu = fits.PrimaryHDU(header = fits.Header())
	# continuum_cube_hdu = fits.ImageHDU(continuum_cube,
	# 						name='CONTINUUM',
	# 						header = cube_header)
	# hdul = fits.HDUList([primary_hdu,
	# 							continuum_cube_hdu])
	# print("Saving bestfit continuum cube")
	# hdul.writeto(f"{parameters['output_dir']}/bestfit_continuum_cube.fits",overwrite=True)
	# print("Saved")


	primary_hdu = fits.PrimaryHDU(header = fits.Header())
	contsub_cube_hdu = fits.ImageHDU(contsub_cube,
							name='CONTSUB',
							header = cube_header)
	noise_cube_hdu = fits.ImageHDU(noise_cube,
							name='NOISE',
							header = cube_header)
	hdul = fits.HDUList([primary_hdu,
								contsub_cube_hdu,noise_cube_hdu])
	print("Saving continuum-subtracted cube")
	hdul.writeto(f"{parameters['output_dir']}/contsub_cube.fits",overwrite=True)
	print("Saved")


	##this is just the bestfit continuum cube again?
	# linefree_spectra = galaxy_spectra - contsub_spectra
	# linefree_cube = np.zeros([Nl,Nx*Ny])
	# linefree_cube[:,np.array(spax_properties['spax_num'][:],dtype=int)] = linefree_spectra
	# linefree_cube = linefree_cube.reshape(Nl,Ny,Nx)

	# primary_hdu = fits.PrimaryHDU(header = fits.Header())
	# linefree_cube_hdu = fits.ImageHDU(linefree_cube,
	# 						name='LINEFREE',
	# 						header = cube_header)
	# hdul = fits.HDUList([primary_hdu,
	# 							linefree_cube_hdu])
	# print("Saving linefree cube")
	# hdul.writeto(f"{parameters['output_dir']}/linefree_cube.fits",overwrite=True)
	# print("Saved")
	
def check_continuum_at_emissionlines(parameters):
	
	spax_properties = Table.read(f"{parameters['output_dir']}/spaxel_properties.fits")
	bright_spectra = np.argsort(spax_properties['spax_signal_cont']/spax_properties['spax_noise_cont'])[::-1][0:5]

	spectra_file = f"{parameters['output_dir']}/logRebin_spectra.fits"
	hdul = fits.open(spectra_file)
	logLambda, spectra = read_spectra_hdul(hdul)
	linLambda = np.exp(logLambda)
	hdul.close()
	spectra = spectra[0]
	spectra = spectra[:,bright_spectra]

	continuum_spectra_file = f"{parameters['output_dir']}/bestfit_continuum_spectra.fits"
	hdul = fits.open(continuum_spectra_file)
	logLambda, continuum_spectra = read_spectra_hdul(hdul)
	hdul.close()
	gasline_spectra = continuum_spectra[1]
	continuum_spectra = continuum_spectra[0]
	continuum_spectra = continuum_spectra[:,bright_spectra]
	gasline_spectra = gasline_spectra[:,bright_spectra]

	width = 3500/3.e5

	emlines = {'Hbeta':{	'lambda':[4861.333],			'ratio':[1]},
			'OIII':{	'lambda':[4958.911, 5006.843],	'ratio':[0.35,1]}, 
			'HeI5876':{	'lambda':[5875.624],			'ratio':[1]}, 
			'OI':{		'lambda':[6300.304,6363.78],	'ratio':[1,0.33]},
			'Halpha+NII':{	'lambda':[6562.819,6548.050,6583.460],			'ratio':[1]},
			# 'HeI6678':{	'lambda':[6678.151],			'ratio':[1]}, 
			'SII6716':{	'lambda':[6716.440,6730.810],			'ratio':[1]}
			# 'ArIII7135':{'lambda':[7135.790],			'ratio':[1]}, 
			# 'OII':{		'lambda':[7319.990, 7330.730],	'ratio':[1,1]}, #?? check
			# 'ArIII7751':{'lambda':[7751.060],			'ratio':[1]},
			# 'SIII':{	'lambda':[9068.6],				'ratio':[1]}
			}

	lines = []
	for nn,line in enumerate(emlines):
		if np.max(emlines[line]['lambda']) < np.max(linLambda) and np.min(emlines[line]['lambda'])>np.min(linLambda):
			lines.extend([line])


	fig1 = plt.figure(figsize=(4*len(spectra[0,:]),3*len(lines)))
	gs1 = gridspec.GridSpec(len(lines),len(spectra[0,:]))

	fig2 = 	plt.figure(figsize=(4*len(spectra[0,:]),3*len(lines)))
	gs2 = gridspec.GridSpec(len(lines),len(spectra[0,:]),)


	for ll, line in enumerate(lines):

		specrange = np.logical_and(logLambda < np.log(np.max(emlines[line]['lambda'])*(1 + width)),
									logLambda > np.log(np.min(emlines[line]['lambda'])*(1-width)))


		for ss in range(len(spectra[0,:])):
			ax1 = fig1.add_subplot(gs1[ll,ss])


			ax1.plot(linLambda[specrange],spectra[specrange,ss])
			ax1.plot(linLambda[specrange],continuum_spectra[specrange,ss])
			# ax1.set_ylim([0.85*np.median(continuum_spectra[specrange,ss]),
									# 1.1*np.median(continuum_spectra[specrange,ss])])
			ax1.set_ylim([(np.min(continuum_spectra[specrange,ss])),
									np.median(continuum_spectra[specrange,ss]) + 0.1*np.max(spectra[specrange,ss])])
			ax1.set_aspect(0.5*np.abs(np.diff(ax1.get_xlim()))/np.abs(np.diff(ax1.get_ylim())))
			if ss == 0:
				ax1.text(0.01,0.1,line,transform=ax1.transAxes)
				ax1.set_ylabel("Flux")
			ax1.tick_params(which='both',axis='both',direction='in')
			if ll == len(lines)-1:
				ax1.set_xlabel("Wavelength")
			if ll == 0:
				ax1.text(0.7,0.2,"Galaxy",transform=ax1.transAxes,color='Blue')
				ax1.text(0.7,0.1,"Cont. fit",transform=ax1.transAxes,color='Orange')


			ax2 = fig2.add_subplot(gs1[ll,ss])
			ax2.plot(linLambda[specrange],spectra[specrange,ss])
			ax2.plot(linLambda[specrange],continuum_spectra[specrange,ss]+gasline_spectra[specrange,ss])
			ax2.set_ylim([0.85*np.median(continuum_spectra[specrange,ss]),
									1.1*np.median(continuum_spectra[specrange,ss]+gasline_spectra[specrange,ss])])
			ax2.set_aspect(0.5*np.abs(np.diff(ax2.get_xlim()))/np.abs(np.diff(ax2.get_ylim())))
			if ss == 0:
				ax2.text(0.01,0.1,line,transform=ax2.transAxes)
				ax2.set_ylabel("Flux")
			ax2.tick_params(which='both',axis='both',direction='in')
			if ll == len(lines)-1:
				ax2.set_xlabel("Wavelength")
			if ll == 0:
				ax2.text(0.7,0.2,"Galaxy",transform=ax2.transAxes,color='Blue')
				ax2.text(0.7,0.1,"Cont. fit",transform=ax2.transAxes,color='Orange')


	fig1.tight_layout()
	fig2.tight_layout()

	# plt.show()
	fig1.savefig(f"{parameters['output_dir']}/figures/absline_check.pdf")
	fig2.savefig(f"{parameters['output_dir']}/figures/emline_check.pdf")

def check_contsub_emlines(parameters):
	spax_properties = Table.read(f"{parameters['output_dir']}/spaxel_properties.fits")
	bright_spectra = np.argsort(spax_properties['spax_signal_cont']/spax_properties['spax_noise_cont'])[::-1][0:5]

	spectra_file = f"{parameters['output_dir']}/logRebin_contsub_spectra.fits"
	hdul = fits.open(spectra_file)
	logLambda, spectra = read_spectra_hdul(hdul)
	linLambda = np.exp(logLambda)
	hdul.close()
	spectra = spectra[0]
	spectra = spectra[:,bright_spectra]

	# continuum_spectra_file = f"{parameters['output_dir']}/bestfit_continuum_spectra.fits"
	# hdul = fits.open(continuum_spectra_file)
	# logLambda, continuum_spectra = read_spectra_hdul(hdul)
	# hdul.close()
	# gasline_spectra = continuum_spectra[1]
	# continuum_spectra = continuum_spectra[0]
	# continuum_spectra = continuum_spectra[:,bright_spectra]
	# gasline_spectra = gasline_spectra[:,bright_spectra]

	width = 800/3.e5

	emlines = {'Hbeta':{	'lambda':[4861.333],			'ratio':[1]},
			'OIII5007':{	'lambda':[5006.843],	'ratio':[0.35,1]}, 
			# 'HeI5876':{	'lambda':[5875.624],			'ratio':[1]}, 
			# 'OI':{		'lambda':[6300.304,6363.78],	'ratio':[1,0.33]},
			'Halpha':{	'lambda':[6562.819],			'ratio':[1]},
			# 'HeI6678':{	'lambda':[6678.151],			'ratio':[1]}, 
			'SII6716':{	'lambda':[6716.440,6730.810],			'ratio':[1]}
			# 'ArIII7135':{'lambda':[7135.790],			'ratio':[1]}, 
			# 'OII':{		'lambda':[7319.990, 7330.730],	'ratio':[1,1]}, #?? check
			# 'ArIII7751':{'lambda':[7751.060],			'ratio':[1]},
			# 'SIII':{	'lambda':[9068.6],				'ratio':[1]}
			}

	lines = []
	for nn,line in enumerate(emlines):
		if np.max(emlines[line]['lambda']) < np.max(linLambda) and np.min(emlines[line]['lambda'])>np.min(linLambda):
			lines.extend([line])


	fig1 = plt.figure(figsize=(4*len(spectra[0,:]),3*len(lines)))
	gs1 = gridspec.GridSpec(len(lines),len(spectra[0,:]))

	# fig2 = plt.figure(figsize=(3*len(spectra[0,:]),4*len(lines)))
	# gs2 = gridspec.GridSpec(len(lines),len(spectra[0,:]),hspace=0.08,wspace=0.3,
	# 		left=0.05,right=0.99,top=0.99,bottom=0.08)


	for ll, line in enumerate(lines):

		specrange = np.logical_and(logLambda < np.log(np.max(emlines[line]['lambda'])*(1 + width)),
									logLambda > np.log(np.min(emlines[line]['lambda'])*(1-width)))


		for ss in range(len(spectra[0,:])):
			ax1 = fig1.add_subplot(gs1[ll,ss])


			ax1.plot(linLambda[specrange],spectra[specrange,ss])
			# ax1.plot(linLambda[specrange],continuum_spectra[specrange,ss])
			# ax1.set_ylim([-10*np.abs(np.median(spectra[specrange,ss])),
									# 1.1*np.max(spectra[specrange,ss])])
			ax1.set_ylim([-0.2*np.max(spectra[specrange,ss]),
									0.75*np.max(spectra[specrange,ss])])
			ax1.set_aspect(0.5*np.abs(np.diff(ax1.get_xlim()))/np.abs(np.diff(ax1.get_ylim())))
			if ss == 0:
				ax1.text(0.01,0.1,line,transform=ax1.transAxes)
				ax1.set_ylabel("Flux")
			ax1.tick_params(which='both',axis='both',direction='in')
			if ll == len(lines)-1:
				ax1.set_xlabel("Wavelength")
			if ll == 0:
				# ax1.text(0.7,0.2,"Galaxy",transform=ax1.transAxes,color='Blue')
				ax1.text(0.7,0.1,"Cont. sub.",transform=ax1.transAxes,color='Blue')


			# ax2 = fig2.add_subplot(gs1[ll,ss])
			# ax2.plot(linLambda[specrange],spectra[specrange,ss])
			# ax2.plot(linLambda[specrange],continuum_spectra[specrange,ss]+gasline_spectra[specrange,ss])
			# ax2.set_ylim([0.85*np.median(continuum_spectra[specrange,ss]),
			# 						1.1*np.median(continuum_spectra[specrange,ss]+gasline_spectra[specrange,ss])])
			# ax2.set_aspect(0.5*np.abs(np.diff(ax2.get_xlim()))/np.abs(np.diff(ax2.get_ylim())))
			# if ss == 0:
			# 	ax2.text(0.01,0.1,line,transform=ax2.transAxes)
			# 	ax2.set_ylabel("Flux")
			# ax2.tick_params(which='both',axis='both',direction='in')
			# if ll == len(lines)-1:
			# 	ax2.set_xlabel("Wavelength")
			# if ll == 0:
			# 	ax2.text(0.7,0.2,"Galaxy",transform=ax2.transAxes,color='Blue')
			# 	ax2.text(0.7,0.1,"Cont. fit",transform=ax2.transAxes,color='Orange')


	fig1.tight_layout()

	# plt.show()
	fig1.savefig(f"{parameters['output_dir']}/figures/contsub_spectra_check.pdf")
	# fig2.savefig(f"{parameters['output_dir']}/figures/emline_check.pdf")

def find_signal_free_spaxels(parameters):
	spectra_file = f"{parameters['output_dir']}/logRebin_contsub_spectra.fits"
	hdul = fits.open(spectra_file)
	logLambda, spectra = read_spectra_hdul(hdul)
	linLambda = np.exp(logLambda)
	hdul.close()
	spectra = spectra[0]


	HA_lambda = 6562.819
	width1 = 400/c

	specrange1 = np.logical_and(logLambda < np.log(HA_lambda)*(1 + width1),
									logLambda > np.log(HA_lambda)*(1-width1))

	width = 300/c

	specrange = np.logical_and(logLambda < np.log(HA_lambda)*(1 + width),
									logLambda > np.log(HA_lambda)*(1-width))

	HA_signal = np.zeros([len(spectra[0,:])])

	for ss in range(len(spectra[0,:])):
		# HA_sigma = np.std(spectra[specrange,ss])
		# HA_signal[ss] = np.max(spectra[specrange,ss])
		# HA_signal[ss] = np.max(np.abs(spectra[specrange,ss])) / HA_sigma
		# HA_signal[ss] = np.sum(spectra[specrange,ss][spectra[specrange,ss]>0])
		HA_signal[ss] = np.sum(spectra[specrange,ss])
		# HA_signal[ss] = np.median(spectra[specrange,ss])




	HA_argsort =  np.argsort(np.abs(HA_signal))
	# HA_signal = HA_signal[HA_argsort]
	# plt.scatter(range(len(HA_signal)),HA_signal)
	# plt.show()
	# exit()

	# lowsig = np.where(np.abs(HA_signal)<1.5)[0]
	lowsig = HA_argsort[0:200]
	print(len(lowsig))
	
	sigmas = np.zeros([len(spectra)])

	for ii in range(len(spectra)):
		signal_free = spectra[ii,lowsig]
		# signal_free =  np.append(signal_free[signal_free<np.median(signal_free)],-1*signal_free[signal_free<np.median(signal_free)])

		med = np.median(signal_free)
		MAD = astrofunc.median_absolute_deviation(
					signal_free[signal_free<np.percentile(signal_free,68)])
		signal_free =  signal_free[signal_free< med + 4*MAD]
		med = np.median(signal_free)
		MAD = astrofunc.median_absolute_deviation(signal_free)
		sigmas[ii] = MAD
		# print(ii,np.exp(logLambda[ii]))
		# print(med)
		# print(MAD,np.std(signal_free))
		# plt.hist(signal_free,bins=20)
		# plt.show()
	plt.plot(linLambda,sigmas)
	plt.show()

	np.savetxt(f"{parameters['output_dir']}/wavelength_sigma.txt",np.vstack([logLambda,sigmas]).T)


	# for ii in range(len(lowsig))[::-1]:
	# 	print(HA_signal[lowsig[ii]])
	# 	print(np.std(spectra[specrange1,lowsig[ii]]))
	# 	plt.plot(np.arange(len(logLambda))[specrange1],spectra[specrange1,lowsig[ii]])
	# 	plt.ylim([-20,20])
	# 	plt.fill_between(np.arange(len(logLambda))[specrange],y1=20,y2=-20,color='Grey',alpha=0.5)
	# 	plt.show()
	# exit()
	
	
def make_line_subcubes(parameterfile):
	parameters = read_parameterfile(parameterfile)


	spax_properties = Table.read(f"{parameters['input_dir']}/spaxel_properties.fits")
	Nx = spax_properties.meta['NX']
	Ny = spax_properties.meta['NY']

	# sigmas =  np.loadtxt(f"{parameters['output_dir']}/wavelength_sigma.txt")
	# sigmaLambda =  sigmas[:,1]

	spectra_file = f"{parameters['output_dir']}/logRebin_contsub_spectra.fits"

	hdul = fits.open(spectra_file)
	logLambda, spectra = read_spectra_hdul(hdul)
	# logLambda = np.exp(logLambda)
	linLambda = np.exp(logLambda)
	hdul.close()
	contsub_spectra = spectra[0]
	noise = np.sqrt(spectra[1])
	spectra = None
	Nl = contsub_spectra.shape[1]


	spectrum_mask = np.zeros_like(contsub_spectra,dtype=bool)
	# print(spectrum_mask)
	# print(sigmaLambda)

	line_shifts = np.zeros([contsub_spectra.shape[1]])

	for ii in range(contsub_spectra.shape[1]):
		spectrum_mask[:,ii] = np.greater(np.abs(contsub_spectra[:,ii]), 3*noise[:,ii])
		spectrum = contsub_spectra[:,ii].copy()
		# plt.plot(linLambda,spectrum)
		# plt.plot(linLambda,noise[:,ii]*3)
		# plt.show()
		spectrum[spectrum_mask[:,ii]==False] = 0

		HA_window = np.logical_and(logLambda < np.log(6562.816*(1 + (300./c))),
			logLambda > np.log(6562.816*(1 - (300./c))))
		# print(spectrum[HA_window])

		if np.all(spectrum[HA_window] <= 0):
			line_shifts[ii] = np.nan
		else:
			HAloc = np.where(spectrum[HA_window] == np.max(spectrum[HA_window]))[0]
			# print(spectrum[HA_window][HAloc])
			HAloc_logLambda = logLambda[HA_window][HAloc]
			line_shift = HAloc_logLambda - np.log(6562.816)
			line_shifts[ii] = line_shift


	# print(line_shifts)
	# exit()


	cube_file = parameters['datacube']
	cube_hdu = fits.open(cube_file)
	cube_header = cube_hdu[1].header
	cube_hdu.close()


	lines = {'Hbeta':{	'lambda':[4861.333]			},
			'OIII4959':{	'lambda':[4958.911]	}, 
			'OIII5006':{	'lambda':[5006.843]	}, 
			# 'HeI5876':{	'lambda':[5875.624]			}, 
			# 'OI6300':{		'lambda':[6300.304]	},
			# 'OI6363':{		'lambda':[6363.78]	},
		 	'NII6548':{		'lambda':[6548.050]	},
		 	'NII6583':{		'lambda':[6583.460]	},
			'Halpha':{	'lambda':[6562.819]			},
			'HeI6678':{	'lambda':[6678.151]			}, 
			'SII6716':{	'lambda':[6716.440]			},
			'SII6730':{	'lambda':[6730.810]			},
			# 'ArIII7135':{'lambda':[7135.790]			}, 
			# 'OII':{		'lambda':[7319.990]	}
			# 'OII':{		'lambda':[7330.730]	}
			# 'ArIII7751':{'lambda':[7751.060]			},
			# 'SIII':{	'lambda':[9068.6]				}
			}

			
	# lines = {'OI6300':{		'lambda':[6300.304]	}}

	if not os.path.isdir(f"{parameters['output_dir']}/linecubes"):
		os.mkdir(f"{parameters['output_dir']}/linecubes")

	subcube_width = 1000 / c
	for ll, line in enumerate(lines):

		lineLambda = lines[line]['lambda'][0]

		subcube_range =  np.where((logLambda > np.log(lineLambda*(1 - subcube_width)))  & 
								(logLambda < np.log(lineLambda*(1 + subcube_width))))[0]

		logLambda_subcube = logLambda[subcube_range]

		subcube_spectra = contsub_spectra[subcube_range,:]
		subcube_noise = noise[subcube_range,:]
		# sigmaLambda_spectra = np.full(subcube_spectra.T.shape,subcube_noise).T
		# print(sigmaLambda_spectra)
		# exit()


		for ii in range(subcube_spectra.shape[1]):
			line_shift = line_shifts[ii]
		
			if np.isfinite(line_shift):

				if line == "Halpha":	
					line_min_logLambda = np.log(0.5*(lineLambda + 6548.050)) + line_shift
					line_max_logLambda = np.log(0.5*(lineLambda + 6583.460)) + line_shift
					logLambda_window = np.logical_and(logLambda_subcube > line_min_logLambda,
									logLambda_subcube  < line_max_logLambda)
					subcube_spectra[~logLambda_window,ii] = 0
					# subcube_noise[~logLambda_window,ii] = 0
				elif line == "NII6548":	
					line_min_logLambda = np.min(logLambda_subcube)
					line_max_logLambda = np.log(0.5*(lineLambda + 6562.819)) + line_shift
					logLambda_window = np.logical_and(logLambda_subcube > line_min_logLambda,
									logLambda_subcube  < line_max_logLambda)
					subcube_spectra[~logLambda_window,ii] = 0
					# subcube_noise[~logLambda_window,ii] = 0
				elif line == "NII6583":	
					line_min_logLambda = np.log(0.5*(lineLambda + 6562.819)) + line_shift
					line_max_logLambda = np.max(logLambda_subcube)
					logLambda_window = np.logical_and(logLambda_subcube > line_min_logLambda,
									logLambda_subcube  < line_max_logLambda)
					subcube_spectra[~logLambda_window,ii] = 0
					# subcube_noise[~logLambda_window,ii] = 0

				elif line == "SII6716":	
					line_min_logLambda = np.min(logLambda_subcube)
					line_max_logLambda = np.log(0.5*(lineLambda + 6730.810)) + line_shift
					logLambda_window = np.logical_and(logLambda_subcube > line_min_logLambda,
									logLambda_subcube  < line_max_logLambda)
					subcube_spectra[~logLambda_window,ii] = 0
					# subcube_noise[~logLambda_window,ii] = 0
				elif line == "SII6730":	
					line_min_logLambda = np.log(0.5*(lineLambda + 6716.440)) + line_shift
					line_max_logLambda = np.max(logLambda_subcube)
					logLambda_window = np.logical_and(logLambda_subcube > line_min_logLambda,
									logLambda_subcube  < line_max_logLambda)
					subcube_spectra[~logLambda_window,ii] = 0
					# subcube_noise[~logLambda_window,ii] = 0
				
				
				subcube_spectra_SN = subcube_spectra[:,ii] / subcube_noise[:,ii]
				mask1 = np.zeros_like(subcube_spectra_SN)
				mask2 = np.zeros_like(subcube_spectra_SN)


				for cc,chan in enumerate(subcube_spectra_SN[1:-1]):
					if chan >= 3.5:
						if subcube_spectra_SN[cc] >=3.5 or subcube_spectra_SN[cc+2] >=3.5:
							mask1[cc+1] = 1

					if chan >= 2:
						if subcube_spectra_SN[cc] >=2 or subcube_spectra_SN[cc+2] >=2:
							mask2[cc+1] = 1

				
				mask_tot = mask1+mask2
				mask_segments = []
				seg = []

				for mm, val in enumerate(mask_tot):
					if val == 0:
						if len(seg) != 0:
							mask_segments.append(seg)
							seg = []

					if val != 0:
						seg.extend([mm])

				mask_final = np.zeros_like(mask_tot)
				
				for seg in mask_segments:
					# print(seg)
					# print(mask_tot[seg])
					if np.any(mask_tot[seg] == 2):
						# seg.extend([seg[0]-1,seg[-1]+1])
						mask_final[seg] = 1

				subcube_spectra[:,ii] *= mask_final
				# subcube_noise[:,ii] *= mask_final

				# subcube_spectra[:,ii][subcube_spectra[:,ii] < 3.*subcube_noise] = 0
				# plt.plot(np.exp(logLambda_subcube),subcube_spectra[:,ii])
				# plt.plot(np.exp(logLambda_subcube),mask1*np.max(subcube_spectra[:,ii]))
				# plt.plot(np.exp(logLambda_subcube),mask2*np.max(subcube_spectra[:,ii]),ls=':')
				# plt.plot(np.exp(logLambda_subcube),mask_final*np.max(subcube_spectra[:,ii]),ls='--')
				# plt.show()
				# exit()
			else:
				subcube_spectra[:,ii] = 0
				# subcube_noise[:,ii] = 0



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
		

		cube_header['NAXIS3'] = len(logLambda_subcube)
		cube_header['CRVAL3'] = logLambda_subcube[0]
		cube_header['CRPIX3'] = 1
		if any(key == "CDELT3" for key in cube_header.keys()):
			cube_header['CDELT3'] = np.abs(np.diff(logLambda_subcube))[0]
		else:
			cube_header['CD3_3'] = np.abs(np.diff(logLambda_subcube))[0]

		Nl = subcube_spectra.shape[0]

		subcube_cube = np.zeros([Nl,Nx*Ny])
		
		for vv in range(subcube_spectra.shape[1]):
			inbin = np.where(spax_properties['vorbin_num']==vv)[0]
			spax_num_inbin = np.array(spax_properties['spax_num'][inbin],dtype=int)
			subcube_cube[:,spax_num_inbin] = np.full((len(inbin),Nl),subcube_spectra[:,vv]).T

		subcube_cube = subcube_cube.reshape(Nl,Ny,Nx)

		# plt.imshow(np.nansum(subcube_cube,axis=0))
		# plt.show()

		primary_hdu = fits.PrimaryHDU(header = fits.Header())
		line_cube_hdu = fits.ImageHDU(subcube_cube,
								name=line,
								header = cube_header)
		hdul = fits.HDUList([primary_hdu,
									line_cube_hdu])
		print(f"Saving {line} linecube")
		hdul.writeto(f"{parameters['output_dir']}/linecubes/{line}_linecube.fits",overwrite=True)
		print("Saved")

def measure_line_fluxes(parameterfile):

	parameters = read_parameterfile(parameterfile)

	spax_properties = Table.read(f"{parameters['input_dir']}/spaxel_properties.fits")

	# sigmas =  np.loadtxt(f"{parameters['output_dir']}/wavelength_sigma.txt")
	# sigmaLambda =  sigmas[:,1]

	subspectra = glob.glob(f"{parameters['output_dir']}/linecubes/*_subspectra.fits")

	RV = 4
	k_l = lambda ll: extinction_curve(ll,RV=RV,extcurve='Calzetti00')

	Av_flag = False
	if any(['Halpha' in file for file in subspectra]) & any(['Hbeta' in file for file in subspectra]):
		AV_flag = True
	
		Halpha_filename = f"{parameters['output_dir']}/linecubes/Halpha_subspectra.fits"
		Hbeta_filename = f"{parameters['output_dir']}/linecubes/Hbeta_subspectra.fits"

		F_Ha, names_Ha = measure_linecube_fluxes(Halpha_filename)
		F_Hb, names_Hb = measure_linecube_fluxes(Hbeta_filename)

		flux_HA = F_Ha[0]
		flux_HB = F_Hb[0]

		EBV_l = EBV_Hlines(flux_HA,flux_HB,6562.819,4861.333,2.83,
							k_l = k_l)

		Av = RV*EBV_l

	else:
		Av_flag = False
		Av = 0
		EBV_l = 0
		

	names = []
	F_measures = []

	for subspectra_file in subspectra:

		fluxes, line_names = measure_linecube_fluxes(subspectra_file,EBV= EBV_l,k_l = k_l)
		

		names.extend(line_names)
		F_measures.extend(fluxes)


	F_measures = np.array(F_measures).T

	if Av_flag:
		reddening = np.array([EBV_l,Av]).T
	else:
		reddening = np.full([2,len(F_measures)],np.array([EBV_l,Av])).T


	F_measures = np.hstack((reddening,F_measures))
	names = ['EBV_l',"A_V"] + names

	F_measures_expanded = np.zeros([len(spax_properties),F_measures.shape[1]])

	for bb in range(F_measures.shape[0]):
		# fluxes_row = np.array([fluxes[ii][bb] for ii in range(len(fluxes))])
		inbin = np.where(spax_properties['vorbin_num'] == bb)[0]
		# spax_num_inbin = spax_properties['spax_num'][inbin]
		
		F_measures_expanded[inbin,:] = F_measures[bb,:]


	fluxes_table = Table(F_measures_expanded,names=names)
	# print(fluxes_table)
	# exit()


	fluxes_table.write(f"{parameters['output_dir']}/spax_emline_fluxes.fits",overwrite=True)

def measure_linecube_fluxes(filename,EBV = 0,k_l = None):

	if isinstance(k_l,type(None)):
		k_l = lambda ll: extinction_curve(ll,RV=4,extcurve='Calzetti00')
		# A_l = lambda ll: EBV * k_l(ll)

	hdul = fits.open(filename)
	logLambda, spectra_list = read_spectra_hdul(hdul)
	spectra = spectra_list[0]
	noise = spectra_list[1]
	hdul.close()

	line = filename.split('/')[-1].split('_subspectra')[0]
	names = [line,f"{line}_err",f"{line}_extcorr",f"{line}_err_extcorr"]

	linLambda = np.exp(logLambda)
	# print(linLambda)


	# print(line)
	# plt.plot(linLambda,k_l(linLambda))
	# plt.show()
	diff_lambda = np.append(np.diff(linLambda)[0]-np.diff(np.diff(linLambda))[0],
							np.diff(linLambda))

	# spectra_extcorr = spectra * np.power(10,0.4*EBV*k_l(linLambda))

	Fint = np.zeros([spectra.shape[1]])
	Ferr = np.zeros([spectra.shape[1]])
	Fint_extcorr = np.zeros([spectra.shape[1]])
	Ferr_extcorr = np.zeros([spectra.shape[1]])

	for ii in range(spectra.shape[1]):
		spectrum = spectra[:,ii]
		sigma = noise[:,ii]

		spectrum_mask = spectrum > 0

		if isinstance(EBV,int):
			E_BV = EBV
		else:
			E_BV = EBV[ii]


		Fint[ii] = np.nansum(spectrum*diff_lambda)
		Ferr[ii] = np.nansum((sigma*spectrum_mask)*diff_lambda)

		if Fint[ii] == 0:				#set non-detection to 1 sigma assuming a 3 channel line
			Fint[ii] = -99
			# 3 closest channel to rest wavelength of line
			loc = np.sort(np.argsort(np.abs(linLambda-emlines_indiv[line]['lambda'][0]))[0:6])+1
			Ferr[ii] = np.nansum(sigma[loc]*diff_lambda[loc])
			# plt.plot(linLambda,spectrum)
			# plt.plot(linLambda,sigma)
			# plt.fill_between(linLambda[loc],np.full_like(linLambda[loc],np.max(sigma[loc])),
									# np.zeros_like(linLambda[loc]),
									# color='Black',alpha=0.2)
			# plt.show()
			# exit()

		# else:
			# print(len(spectrum[spectrum>0]))
			# loc = np.sort(np.argsort(np.abs(linLambda-emlines[line]['lambda'][0]))[0:6])+1
			# plt.plot(linLambda,spectrum)

			# plt.plot(linLambda,sigma)
			# plt.fill_between(linLambda[loc],np.full_like(linLambda[loc],np.max(spectrum[loc])),
									# np.zeros_like(linLambda[loc]),color='Black',alpha=0.2)

			# plt.show()

			# exit()


		spectrum_extcorr = spectrum * np.power(10,0.4*E_BV*k_l(linLambda))
		sigma_extcorr = sigma * np.power(10,0.4*E_BV*k_l(linLambda))

		Fint_extcorr[ii] = np.nansum(spectrum_extcorr*diff_lambda)
		Ferr_extcorr[ii] = np.nansum((sigma_extcorr*spectrum_mask)*diff_lambda)
	
	fluxes = [Fint,Ferr,Fint_extcorr,Ferr_extcorr]

	return  fluxes, names





	# contsub_spectra_file = f"{parameters['output']}/logRebin_contsub_spectra.fits"
	# hdul = fits.open(contsub_spectra_file)
	# logLambda, spectra = read_spectra_hdul(hdul)
	# hdul.close()
	# contsub_spectra = spectra[0]

	# # wavelength = 6562.8
	# wavelength = 4861.333

	# xx = np.array([-200/3.e5,200/3.e5]) +  np.log(wavelength)
	# logLrange = np.logical_and(logLambda >= xx[0], logLambda <= xx[1])
	# specmax = np.nanmax(contsub_spectra[logLrange,0])
	# maxloc = logLambda[logLrange][contsub_spectra[logLrange,0]==specmax]

	# xx = np.array([-250/3.e5,250/3.e5]) +  maxloc
	# logLrange = np.logical_and(logLambda >= xx[0], logLambda <= xx[1])
	
	# # plt.plot(logLambda,contsub_spectra[:,200],lw=0.5)
	# # plt.plot([maxloc,maxloc],plt.gca().get_ylim(),lw=0.5)
	# # plt.fill_between(xx, y1=plt.gca().get_ylim()[1])
	# # plt.show()

	# upper = np.logical_and(logLambda > maxloc, logLambda<xx[1])
	# lower = np.logical_and(logLambda < maxloc, logLambda>xx[0])
	# FWHM1 = np.interp(0.5*specmax,contsub_spectra[upper,0][::-1],logLambda[upper][::-1])
	# FWHM2 = np.interp(0.5*specmax,contsub_spectra[lower,0],logLambda[lower])
	# FWHM = FWHM1 - FWHM2
	
	# # xx = np.array([-1.25*FWHM,1.25*FWHM]) +  maxloc
	# # logLrange = np.logical_and(logLambda >= xx[0], logLambda <= xx[1])

	# print(specmax)
	# # print(maxloc)

	# plt.plot(logLambda,contsub_spectra[:,0],lw=0.5)
	# plt.plot([maxloc,maxloc],plt.gca().get_ylim(),lw=0.5)
	# plt.fill_between(xx, y1=plt.gca().get_ylim()[1],alpha=0.5,color='Black')
	# plt.show()

def measure_linecube_kinematics(filename):

	hdul = fits.open(filename)
	logLambda, spectra_list = read_spectra_hdul(hdul)
	spectra = spectra_list[0]
	noise = spectra_list[1]
	hdul.close()

	line = filename.split('/')[-1].split('_subspectra')[0]
	names = [f"V_{line}",f"sigma_{line}"]

	linLambda = np.exp(logLambda)


	V = np.zeros([spectra.shape[1]])
	sigma = np.zeros([spectra.shape[1]])

	for ii in range(spectra.shape[1]):
		spectrum = spectra[:,ii]
		sigma = noise[:,ii]

		spectrum_mask = spectrum > 0

		V = np.moment(linLambda)


		Fint[ii] = np.nansum(spectrum*diff_lambda)
		Ferr[ii] = np.nansum((sigma*spectrum_mask)*diff_lambda)

		if Fint[ii] == 0:				#set non-detection to 1 sigma assuming a 3 channel line
			Fint[ii] = -99
			# 3 closest channel to rest wavelength of line
			loc = np.sort(np.argsort(np.abs(linLambda-emlines_indiv[line]['lambda'][0]))[0:6])+1
			Ferr[ii] = np.nansum(sigma[loc]*diff_lambda[loc])
			# plt.plot(linLambda,spectrum)
			# plt.plot(linLambda,sigma)
			# plt.fill_between(linLambda[loc],np.full_like(linLambda[loc],np.max(sigma[loc])),
									# np.zeros_like(linLambda[loc]),
									# color='Black',alpha=0.2)
			# plt.show()
			# exit()

		# else:
			# print(len(spectrum[spectrum>0]))
			# loc = np.sort(np.argsort(np.abs(linLambda-emlines[line]['lambda'][0]))[0:6])+1
			# plt.plot(linLambda,spectrum)

			# plt.plot(linLambda,sigma)
			# plt.fill_between(linLambda[loc],np.full_like(linLambda[loc],np.max(spectrum[loc])),
									# np.zeros_like(linLambda[loc]),color='Black',alpha=0.2)

			# plt.show()

			# exit()


		spectrum_extcorr = spectrum * np.power(10,0.4*E_BV*k_l(linLambda))
		sigma_extcorr = sigma * np.power(10,0.4*E_BV*k_l(linLambda))

		Fint_extcorr[ii] = np.nansum(spectrum_extcorr*diff_lambda)
		Ferr_extcorr[ii] = np.nansum((sigma_extcorr*spectrum_mask)*diff_lambda)
	
	fluxes = [Fint,Ferr,Fint_extcorr,Ferr_extcorr]

	return  fluxes, names

def check_line_ratios(parameterfile):

	parameters = read_parameterfile(parameterfile)


	emline_fluxes = Table.read(f"{parameters['output_dir']}/spax_emline_fluxes.fits")

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
				emline_fluxes['OIII4958']/emline_fluxes['OIII5006'],
				marker='o',s=0.5,color='Black')
	
	emline_fluxes_SN3 = emline_fluxes[
						np.where(
						(emline_fluxes['OIII4958']/emline_fluxes['OIII4958_err'] >=3) &
						(emline_fluxes['OIII5006']/emline_fluxes['OIII5006_err'] >=3))]

	emline_fluxes_SN5 = emline_fluxes[
						np.where(
						(emline_fluxes['OIII4958']/emline_fluxes['OIII4958_err'] >=5) &
						(emline_fluxes['OIII5006']/emline_fluxes['OIII5006_err'] >=5))]


	ax2.scatter(np.log10(emline_fluxes_SN3['Halpha']),
				emline_fluxes_SN3['OIII4958']/emline_fluxes_SN3['OIII5006'],
				marker='o',s=0.5,color="DodgerBlue")

	ax2.scatter(np.log10(emline_fluxes_SN5['Halpha']),
				emline_fluxes_SN5['OIII4958']/emline_fluxes_SN5['OIII5006'],
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
	ax4.set_ylim([0.5,5])
	ax4.set_xlim([1,6])

	legs = [Line2D([0],[0],color='White',marker='o',markerfacecolor='Black'),
			Line2D([0],[0],color='White',marker='o',markerfacecolor='DodgerBlue'),
			Line2D([0],[0],color='White',marker='o',markerfacecolor='Red')]
	ax1.legend(legs,["All","SN>3","SN>5"],fontsize=15)

	fig.savefig(f"{parameters['output_dir']}/figures/EMline_ratios.png")

	# plt.show()
def check_line_SNmaps(parameterfile):
	parameters = read_parameterfile(parameterfile)

	spax_properties = Table.read(f"{parameters['input_dir']}/spaxel_properties.fits")

	emline_fluxes = Table.read(f"{parameters['output_dir']}/spax_emline_fluxes.fits")

	spax_properties = hstack((spax_properties,emline_fluxes))

	metadata = spax_properties.meta
	Nx =  metadata['NX']
	Ny =  metadata['NY']


	img_grid = np.zeros((Ny,Nx)).flatten()

	line_names = emline_fluxes.keys()[2::2]
	# line_names = [line.split("_err")[0] for line in emline_fluxes.keys()]
	# line_names = np.unique(line_names)

	for ll, line in enumerate(line_names):


		linemap = img_grid.copy()
		linemap[np.array(spax_properties['spax_num'],dtype=int)] = spax_properties[f"{line}"]
		linemap = linemap.reshape((Ny,Nx))

		if f"{line}_err" in emline_fluxes.keys():
			line_errmap = img_grid.copy()
			line_errmap[np.array(spax_properties['spax_num'],dtype=int)] = spax_properties[f"{line}_err"]
			line_errmap = line_errmap.reshape((Ny,Nx))

		fig = plt.figure(figsize=(12,10))
		gs = gridspec.GridSpec(1,2,width_ratios = [1,0.05])


		ax1 = fig.add_subplot(gs[0,0])
		cb_ax = fig.add_subplot(gs[0,1])


		img = ax1.pcolormesh(np.log10(linemap),vmin=1,vmax=4.5)
		if f"{line}_err" in emline_fluxes.keys():
			# ax1.pcolormesh(HAmap/HA_errmap,vmin=0.5,vmax=5)
			ax1.contour(linemap/line_errmap,levels=[5],colors=['Black'])

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
				['OIII4958','Hbeta'],
				['OIII5006','Hbeta'],
				['Halpha','Hbeta'],
				['NII6548','NII6583'],
				['OIII4958', 'OIII5006'],
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


		linemap = img_grid.copy()
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
			linemap = img_grid.copy()
			
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


def metallicity_Curti17(parameterfile,method='O3N2'):	
	#R2 = OII3727/Hbeta
	#R3= OIII5007/Hbeta
	#R23 = (OII3727+OIII4959,5007)/Hbeta
	#O32 = OIII5007/OII3727
	#N2 = NII6584/Halpha
	#O3N2 = OIII5007/Hbeta / NII6584/Halpha

	methods = {'R2':{'coeffs':[0.418,-0.961, -3.505,-1.949],'lims':[7.6,8.3]},
				'R3':{'coeffs':[-0.277 ,-3.549 ,-3.593 ,-0.981],'lims':[8.3,8.85]},
				'O32':{'coeffs':[-0.691 ,-2.944 ,-1.308],'lims':[7.6,8.85]},
				'R23':{'coeffs':[0.527 ,-1.569 ,-1.652 ,-0.421],'lims':[8.4,8.85]},
				'N2':{'coeffs':[-0.489, 1.513,-2.554, -5.293, -2.867],'lims':[7.6,8.85]},
				'O3N2':{'coeffs':[0.281,-4.765,-2.268],'lims':[7.6,8.85]}
				}


	parameters = read_parameterfile(parameterfile)

	spax_properties = Table.read(f"{parameters['input_dir']}/spaxel_properties.fits")

	emline_fluxes = Table.read(f"{parameters['output_dir']}/spax_emline_fluxes.fits")

	spax_properties = hstack((spax_properties,emline_fluxes))

	
	if method =='R2':
		SNgood = np.where((spax_properties['OII3727']/spax_properties['OII3727_err'] >=4) & 
						(spax_properties['Hbeta']/spax_properties['Hbeta_err'] >=4))[0]

		ratio = (spax_properties['OII3727_extcorr'] / spax_properties['Hbeta_extcorr'])[SNgood]
		log_ratio = np.log10(ratio)

	elif method == 'R3':

		SNgood = np.where((spax_properties['OIII5006']/spax_properties['OIII5006_err'] >=4) & 
						(spax_properties['Hbeta']/spax_properties['Hbeta_err'] >=4))[0]

		ratio = (spax_properties['OIII5006_extcorr'] / spax_properties['Hbeta_extcorr'])[SNgood]
		log_ratio = np.log10(ratio)

	elif method == 'R23':
		SNgood = np.where((spax_properties['OIII5006']/spax_properties['OIII5006_err'] >=4) & 
						(spax_properties['OII3727']/spax_properties['OII3727_err'] >=4) &
						(spax_properties['Hbeta']/spax_properties['Hbeta_err'] >=4))[0]

		ratio = ((spax_properties['OIII5006_extcorr'] + spax_properties['OII3727_extcorr']) /
					spax_properties['Hbeta_extcorr'] )[SNgood]
		log_ratio = np.log10(ratio)

	elif method == 'O32':
		SNgood = np.where((spax_properties['OIII5006']/spax_properties['OIII5006_err'] >=4) & 
						(spax_properties['OII3727']/spax_properties['OII3727_err'] >=4))[0]

		ratio = (spax_properties['OIII5006_extcorr'] / spax_properties['OII3727_extcorr'])[SNgood]
		log_ratio = np.log10(ratio)

	elif method == 'N2':

		SNgood = np.where((spax_properties['NII6583']/spax_properties['NII6583_err'] >=4) & 
						(spax_properties['Halpha']/spax_properties['Halpha_err'] >=4))[0]

		ratio = (spax_properties['NII6583_extcorr'] / spax_properties['Halpha_extcorr'])[SNgood]
		log_ratio = np.log10(ratio)

	elif method == 'O3N2':

		SNgood = np.where((spax_properties['NII6583']/spax_properties['NII6583_err'] >=4) & 
					(spax_properties['Halpha']/spax_properties['Halpha_err'] >=4) &
					(spax_properties['OIII5006']/spax_properties['OIII5006_err'] >=4) & 
					(spax_properties['Hbeta']/spax_properties['Hbeta_err'] >=4))[0]


		ratio = ( (spax_properties['OIII5006_extcorr'][:] / spax_properties['Hbeta_extcorr'][:]) 
				/ (spax_properties['NII6583_extcorr'][:] / spax_properties['Halpha_extcorr'][:]) )[SNgood]
		log_ratio = np.log10(ratio)



	# plt.hist(log_ratio,bins=100)
	# plt.show()
	# exit()

	method_coeffs = methods[method]
	logOH_range = np.linspace(method_coeffs['lims'][0],method_coeffs['lims'][1],1000)
	logRatio_range  = np.zeros_like(logOH_range)
	for ii in range(len(method_coeffs['coeffs'])):
		# print(method_coeffs['coeffs'][ii])
		logRatio_range += method_coeffs['coeffs'][ii] * np.power(logOH_range-8.69,ii)

	# plt.plot(logOH_range,logRatio_range)
	# plt.show()
	# exit()

	min_logRatio = np.min(logRatio_range)
	max_logRatio = np.max(logRatio_range)

	# print(min_logRatio,max_logRatio)

	# spaxel_logOH = np.zeros_like(log_ratio)
	# spaxel_logOH = np.full_like(log_ratio,-1)

	# print(logRatio_range[::-1])


	if logRatio_range[0]>logRatio_range[-1]:
		logRatio_range = logRatio_range[::-1]
		logOH_range = logOH_range[::-1]
	spaxel_logOH = np.interp(log_ratio,logRatio_range,logOH_range)
	# bad_spax = 
	spaxel_logOH[(log_ratio < min_logRatio)] = np.nan
	spaxel_logOH[ (log_ratio > max_logRatio)] = np.nan



	spaxel_logOH_full = np.full(len(spax_properties),np.nan)

	#re-distribute QC spectra to whole list
	for nn,vv in enumerate(SNgood):
		spaxel_logOH_full[vv] = spaxel_logOH[nn]


	metadata = spax_properties.meta
	Nx =  metadata['NX']
	Ny =  metadata['NY']


	img_grid = np.full((Ny,Nx),np.nan).flatten()

	img_grid[np.array(spax_properties['spax_num'],dtype=int)] =	spaxel_logOH_full
	img_grid = img_grid.reshape((Ny,Nx))


	p90 = np.percentile(spaxel_logOH[np.isfinite(spaxel_logOH)],90)
	p50 = np.percentile(spaxel_logOH[np.isfinite(spaxel_logOH)],50)
	p10 = np.percentile(spaxel_logOH[np.isfinite(spaxel_logOH)],10)


	fig = plt.figure(figsize=(12,10))
	gs = gridspec.GridSpec(1,2,width_ratios = [1,0.05])


	ax1 = fig.add_subplot(gs[0,0])
	cb_ax = fig.add_subplot(gs[0,1])

	img = ax1.pcolormesh(img_grid,vmin=p10,vmax=p90)


	fig.colorbar(img,cax = cb_ax)

	cb_ax.set_title(f'12+ log(O/H) [{method}]')
	cb_ax.tick_params(which='both',axis='both',direction='in')


	ax1.set_aspect("equal")
	ax1.set_xlabel("pix",fontsize=15)
	ax1.set_ylabel("pix",fontsize=15)
	ax1.tick_params(which='both',axis='both',direction='in',labelsize=12)

	fig.savefig(f"{parameters['output_dir']}/figures/logOH_map_{method}.png")
	# plt.show()
def metallicity_Dopita16(parameterfile):

	parameters = read_parameterfile(parameterfile)

	spax_properties = Table.read(f"{parameters['input_dir']}/spaxel_properties.fits")

	emline_fluxes = Table.read(f"{parameters['output_dir']}/spax_emline_fluxes.fits")

	spax_properties = hstack((spax_properties,emline_fluxes))


	SNgood = np.where((spax_properties['OIII5006']/spax_properties['OIII5006_err'] >=4) & 
					(spax_properties['NII6583']/spax_properties['NII6583_err'] >=4) & 
					(spax_properties['Halpha']/spax_properties['Halpha_err'] >=4) &
					(spax_properties['SII6716']/spax_properties['SII6716_err'] >=4))[0]




	y = (np.log10(spax_properties['NII6583_extcorr']/spax_properties['SII6716_extcorr']) + \
		0.264 * np.log10(spax_properties['NII6583_extcorr'] / spax_properties['Halpha_extcorr']))[SNgood]

	# plt.hist(y,bins=100)
	# plt.show()

	spaxel_logOH = 8.77 + y
	# spaxel_logOH = 8.77 + y + 0.45*(y+0.3)**5



	spaxel_logOH_full = np.full(len(spax_properties),np.nan)

	#re-distribute QC spectra to whole list
	for nn,vv in enumerate(SNgood):
		spaxel_logOH_full[vv] = spaxel_logOH[nn]


	metadata = spax_properties.meta
	Nx =  metadata['NX']
	Ny =  metadata['NY']


	img_grid = np.full((Ny,Nx),np.nan).flatten()

	img_grid[np.array(spax_properties['spax_num'],dtype=int)] =	spaxel_logOH_full
	img_grid = img_grid.reshape((Ny,Nx))


	p90 = np.percentile(spaxel_logOH[np.isfinite(spaxel_logOH)],90)
	p50 = np.percentile(spaxel_logOH[np.isfinite(spaxel_logOH)],50)
	p10 = np.percentile(spaxel_logOH[np.isfinite(spaxel_logOH)],10)


	fig = plt.figure(figsize=(12,10))
	gs = gridspec.GridSpec(1,2,width_ratios = [1,0.05])


	ax1 = fig.add_subplot(gs[0,0])
	cb_ax = fig.add_subplot(gs[0,1])

	img = ax1.pcolormesh(img_grid,vmin=p10,vmax=p90)


	fig.colorbar(img,cax = cb_ax)

	cb_ax.set_title(f'12+ log(O/H) [Dopita16]')
	cb_ax.tick_params(which='both',axis='both',direction='in')


	ax1.set_aspect("equal")
	ax1.set_xlabel("pix",fontsize=15)
	ax1.set_ylabel("pix",fontsize=15)
	ax1.tick_params(which='both',axis='both',direction='in',labelsize=12)

	fig.savefig(f"{parameters['output_dir']}/figures/logOH_map_Dopita16.png")
	# plt.show()


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



def fit_O3_line_components(parameterfile):
	parameters =  read_parameterfile(parameterfile)


	spectra_file = f"{parameters['input_dir']}/linecubes/OIII5006_subspectra.fits"
	hdul = fits.open(spectra_file)
	logLambda_spec, spectra  = read_spectra_hdul(hdul)
	logRebin_spectra = spectra[0]
	logRebin_noise = spectra[1]
	spectra = None


	spectrum = logRebin_spectra[:,10500]

	from scipy.optimize import curve_fit

	coeffs = []
	covars = []
	fits_all = []
	for ii in range(4):
		if ii+1 == 1:
			fit_coeff, covar = curve_fit(gaussian1,logLambda_spec,spectrum,p0=[0.5*np.sum(spectrum),np.mean(logLambda_spec),0.0005])
			fit = [gaussian1(logLambda_spec,fit_coeff[0],fit_coeff[1],fit_coeff[2])]
			coeffs.append(fit_coeff)
			covars.append(covar)
			fits_all.append(fit)
		if ii+1 == 2:
			fit_coeff, covar = curve_fit(gaussian2,logLambda_spec,spectrum,
												p0=[0.25*np.sum(spectrum),np.mean(logLambda_spec),0.0005,
												0.25*np.sum(spectrum),np.mean(logLambda_spec),0.001])
			fit = [gaussian1(logLambda_spec,fit_coeff[0],fit_coeff[1],fit_coeff[2]),
					gaussian1(logLambda_spec,fit_coeff[3],fit_coeff[4],fit_coeff[5]) ]
			coeffs.append(fit_coeff)
			covars.append(covar)
			fits_all.append(fit)
		if ii+1 == 3:
			fit_coeff, covar = curve_fit(gaussian3,logLambda_spec,spectrum,
											p0=[0.25*np.sum(spectrum),np.mean(logLambda_spec),0.0005,
												0.25*np.sum(spectrum),np.mean(logLambda_spec),0.001,
												0.25*np.sum(spectrum),np.mean(logLambda_spec),0.001])
			fit = [gaussian1(logLambda_spec,fit_coeff[0],fit_coeff[1],fit_coeff[2]),
					gaussian1(logLambda_spec,fit_coeff[3],fit_coeff[4],fit_coeff[5]),
					gaussian1(logLambda_spec,fit_coeff[6],fit_coeff[7],fit_coeff[8])]
			coeffs.append(fit_coeff)
			covars.append(covar)
			fits_all.extend(fit)
		if ii+1 == 4:
			fit_coeff, covar = curve_fit(gaussian4,logLambda_spec,spectrum,
												p0=[0.25*np.sum(spectrum),np.mean(logLambda_spec),0.0005,
												0.25*np.sum(spectrum),np.mean(logLambda_spec),0.001,
												0.25*np.sum(spectrum),np.mean(logLambda_spec),0.001,
												0.25*np.sum(spectrum),np.mean(logLambda_spec),0.001])
			fit = [gaussian1(logLambda_spec,fit_coeff[0],fit_coeff[1],fit_coeff[2]),
					gaussian1(logLambda_spec,fit_coeff[3],fit_coeff[4],fit_coeff[5]),
					gaussian1(logLambda_spec,fit_coeff[6],fit_coeff[7],fit_coeff[8]),
					gaussian1(logLambda_spec,fit_coeff[9],fit_coeff[10],fit_coeff[11])]
			coeffs.append(fit_coeff)
			covars.append(covar)
			fits_all.extend(fit)



		print(fit_coeff)
		plt.plot(logLambda_spec,spectrum)
		# plt.plot(logLambda_spec,gaussian1(logLambda_spec,7000,8.519,0.001))
		for ff in range(len(fit)):
			plt.plot(logLambda_spec,fit[ff])
		plt.show()



def gaussian1(xx,A1, mu1, sigma1):
	# prob = 1. / (sigma*np.sqrt(2.e0 * np.pi)) * \
			# np.exp(-0.5e0*( ((xx - mu) / sigma) *((xx - mu) / sigma) ))
	gg = np.abs(A1)*np.exp(-0.5e0*( ((xx - mu1) / sigma1) *((xx - mu1) / sigma1) ))
	return gg
def gaussian2(xx,A1, mu1, sigma1,A2, mu2, sigma2):
	gg = np.abs(A1)*np.exp(-0.5e0*( ((xx - mu1) / sigma1) *((xx - mu1) / sigma1) )) +\
		np.abs(A2)*np.exp(-0.5e0*( ((xx - mu2) / sigma2) *((xx - mu2) / sigma2) ))
	return gg
def gaussian3(xx,A1, mu1, sigma1,A2, mu2, sigma2,A3, mu3, sigma3):
	gg = np.abs(A1)*np.exp(-0.5e0*( ((xx - mu1) / sigma1) *((xx - mu1) / sigma1) )) +\
		np.abs(A2)*np.exp(-0.5e0*( ((xx - mu2) / sigma2) *((xx - mu2) / sigma2) )) +\
		np.abs(A3)*np.exp(-0.5e0*( ((xx - mu3) / sigma3) *((xx - mu3) / sigma3) ))
	return gg
def gaussian4(xx,A1, mu1, sigma1,A2, mu2, sigma2,A3, mu3, sigma3,A4, mu4, sigma4):
	gg = np.abs(A1)*np.exp(-0.5e0*( ((xx - mu1) / sigma1) *((xx - mu1) / sigma1) )) +\
		np.abs(A2)*np.exp(-0.5e0*( ((xx - mu2) / sigma2) *((xx - mu2) / sigma2) )) +\
		np.abs(A3)*np.exp(-0.5e0*( ((xx - mu3) / sigma3) *((xx - mu3) / sigma3) ))+\
		np.abs(A4)*np.exp(-0.5e0*( ((xx - mu4) / sigma4) *((xx - mu4) / sigma4) ))

	return gg

def norm_gaussian(xx,A,mu,sigma):
	
	prob = 1. / (sigma*np.sqrt(2.e0 * np.pi)) * \
			np.exp(-0.5e0*( ((xx - mu) / sigma) *((xx - mu) / sigma) ))
	return prob

#useful things


def MUSE_LSF_Bacon17(ll, z = 0):
	ll = ll*(1.e0 + z)										#convert rest-frame wavelength to observed frame
	FWHM = 5.866e-8 * ll*ll - 9.187e-4 * ll + 6.040 		#get LSF at observed-frame wavelength
	FWHM = FWHM / (1.e0 + z)								#convert obs-frame LSF to narrower de-redshifted value

	return FWHM


def EBV_Hlines(F1 ,F2 ,lambda1 ,lambda2 , Rint = 2.83,k_l = None):
	#lambdas in angstrom
	#F1=HA F2 = HB (default)
	
	if isinstance(k_l,type(None)):
		k_l = lambda ll: extinction_curve(ll)

	ratio = np.log10((F1/F2) / Rint)

	kdiff = k_l(lambda2) - k_l(lambda1)

	E_BV = ratio / (0.4 * kdiff)
	print(np.min(E_BV))

	E_BV[np.isfinite(E_BV)==False] = 0
	print(np.min(E_BV))

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


def extract_subcube(datacube,subcube_index,filepath = True,hdu=0):

	if filepath:
		cube = SpectralCube.read(datacube,hdu=hdu)
	elif not filepath:
		cube = SpectralCube(data=datacube[0],wcs=datacube[1])

	if subcube_index[0]=='all':
		subcube = cube[:,
					int(subcube_index[1][0]):int(subcube_index[1][1]),
					int(subcube_index[2][0]):int(subcube_index[2][1])]
	else:
		subcube = cube[int(subcube_index[0][0]):int(subcube_index[0][1]),
					int(subcube_index[1][0]):int(subcube_index[1][1]),
					int(subcube_index[2][0]):int(subcube_index[2][1])]

	return subcube


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




if __name__ == '__main__':
	main()
	# read_EMILES_spectra()
	# read_parameterfile()
	# read_gaslines_parameterfile()

	# make_spectra_tables()


	# voronoi_bin_cube(SN_indiv = 5, SN_vorbin = 40)

	# create_binned_spectra()