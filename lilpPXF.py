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
# import datetime
# import psutil

from scipy import ndimage
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

sys.path.append('/Users/00088350/Research/programs/astro-functions')
import astro_functions as astrofunc

from spectral_cube import SpectralCube


# import vorbin
from vorbin.voronoi_2d_binning import voronoi_2d_binning


#global variables
c = 299792.458

emlines = {		'Hbeta':{	'lambda':[4861.333],			'ratio':[1]},
				'Halpha':{	'lambda':[6562.819],			'ratio':[1]},
				
				'OI':{		'lambda':[6300.304,6363.78],	'ratio':[1,0.33]},					#low ionisation 14.53
				'OII':{		'lambda':[7319.990, 7330.730],	'ratio':[1,1]}, #?? check 			#low ionisation 13.62
				'OIII':{	'lambda':[4958.911, 5006.843],	'ratio':[0.35,1]}, 					#high ionisation 35.12

			 	'NII':{		'lambda':[6548.050,6583.460],	'ratio':[0.34,1]},					#low ionisation 15.43


				'HeI5876':{	'lambda':[5875.624],			'ratio':[1]}, 
				'HeI6678':{	'lambda':[6678.151],			'ratio':[1]}, 
				'HeI7065':{'lambda':[7065.196], 			'ratio':[1]},

				'SII6716':{	'lambda':[6716.440],			'ratio':[1]},						#low ionisation 10.36
				'SII6730':{	'lambda':[6730.810],			'ratio':[1]},						#^^
				'SIII9069':{	'lambda':[9068.6],				'ratio':[1]},					#high ionisation 23.33
				
				'ArIII7135':{'lambda':[7135.790],			'ratio':[1]}, 						#high ionisation 27.63 #doublet**
				'ArIII7751':{'lambda':[7751.060],			'ratio':[1]},						#high ionisation 27.63
				}


#main fitting functions

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

		spax_properties_file = f"{parameters['output']}/{parameters['input_dir']}/spaxel_properties.fits"

		spectra_file = f"{parameters['output']}/{parameters['input_dir']}/logRebin_spectra.fits"


		spax_properties = Table.read(spax_properties_file)

		hdul = fits.open(spectra_file)
		logLambda_spec, spectra  = read_spectra_hdul(hdul)
		logRebin_spectra = spectra[0]
		logRebin_variance = spectra[1]
		spectra = None

		header = hdul[0].header
		hdul.close()
		velscale = header['VELSCALE']
		parameters['galaxy_velscale'] = float(velscale)

		stellar_templates, logLambda_templates = read_EMILES_spectra(parameters)
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
			proc_logRebin_variance = logRebin_variance[:,proc_spax]

			tosend = [proc_spax_properies,
						proc_logRebin_spectra,
						proc_logRebin_variance]
			comm.send(tosend, dest=nn, tag=100+nn)
			tosend = None

		elif rank == nn:
			torecieve = comm.recv(source=0, tag=100+rank)
			proc_spax_properties = torecieve[0]
			proc_logRebin_spectra = torecieve[1]
			proc_logRebin_variance = torecieve[2]
			torecieve = None

	if rank == 0:
		proc_spax = np.arange(0,len(logRebin_spectra[0,:]),nproc,dtype=int)
		proc_spax_properties = spax_properties[proc_spax]
		proc_logRebin_spectra = logRebin_spectra[:,proc_spax]
		proc_logRebin_variance = logRebin_variance[:,proc_spax]

		spectra_shape = logRebin_spectra.shape


		# logRebin_spectra = None
		logRebin_variance = None
		hdul = None

	if rank == 0:
		print(f"Spectra distributed, running fits")
	sys.stdout.flush()

	comm.barrier()

	outputs = []
	outputs_all = []

	for ss in range(len(proc_logRebin_spectra[0,:])):

		spectrum = np.array(proc_logRebin_spectra[:,ss])
		variance = np.array(proc_logRebin_variance[:,ss])
		good_pixels_spec = good_pixels.copy()
		good_pixels_spec[np.isfinite(spectrum)==False] = False
		good_pixels_spec[(np.isfinite(variance)==False) | (variance <= 0)] = False

		spec_median = np.abs(np.nanmedian(spectrum))
		var_median = np.abs(np.nanmedian(variance))
		spectrum = spectrum / spec_median			#nomalise spectrum
		variance = variance / var_median			#nomalise variance
		variance[~good_pixels_spec] = 1.e-5			#pPXF doesnt apply goodpix to variance -_-


		if np.any(gas_components):
			out = ppxf(templates, spectrum, variance,
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
			out = ppxf(templates, spectrum, variance,
				velscale = parameters['galaxy_velscale'],
				start = start,
				moments = moments,
				component=components,
				degree = -1,										# additive can affect Balmer fluxes
				mdegree = parameters['continuum_mdegree'],
				# gas_component = gas_components,
				velscale_ratio = parameters['velscale_ratio'],
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
		hdul.writeto(f"{parameters['output']}/{parameters['output_dir']}/bestfit_continuum_spectra.fits",overwrite=True)
		print("Saved")
		sys.stdout.flush()
			
def fit_stellar_kinematics(parameterfile):

	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	nproc = comm.Get_size()

	if rank == 0:
		print(f"Running with {nproc} processors")
		print("Preparing inputs")
		sys.stdout.flush()
		parameters = read_parameterfile(parameterfile)
		
		spax_properties_file = f"{parameters['output']}/{parameters['input_dir']}/spaxel_properties.fits"

		spectra_file = f"{parameters['output']}/{parameters['input_dir']}/logRebin_spectra.fits"


		spax_properties = Table.read(spax_properties_file)

		hdul = fits.open(spectra_file)
		logLambda_spec, spectra  = read_spectra_hdul(hdul)
		logRebin_spectra = spectra[0]
		logRebin_variance = spectra[1]
		spectra = None

		header = hdul[0].header
		hdul.close()
		velscale = header['VELSCALE']
		parameters['galaxy_velscale'] = float(velscale)

		templates, logLambda_templates = read_EMILES_spectra(parameters)


		#truncate to fit range
		if 'stars_lrange' in parameters.keys():
			Lmin = parameters['stars_lrange'][0]
			Lmax = parameters['stars_lrange'][1]
			specrange = np.where((logLambda_spec>=np.log(Lmin)) & 
									(logLambda_spec<=np.log(Lmax)))[0]
			logLambda_spec = logLambda_spec[specrange]
			logRebin_spectra = logRebin_spectra[specrange,:]
			logRebin_variance = logRebin_variance[specrange,:]

			# specrange = np.where((logLambda_templates>=np.log(Lmin)) & 
									# (logLambda_templates<=np.log(Lmax)))[0]



		good_pixels = create_spectrum_mask(logLambda_spec,parameters)

		dv = c*(np.nanmean(logLambda_templates[:parameters['velscale_ratio']]) - logLambda_spec[0]) # km/s
		if 'stars_start' in parameters.keys():
			start = parameters['stars_start']
		else:
			start = [0, 100]
		# print(start)
		# exit()

	else:
		parameters = None
		start = None
		dv = None
		templates = None
		good_pixels = None

	parameters = comm.bcast(parameters,root=0)
	start = comm.bcast(start,root=0)
	dv = comm.bcast(dv,root=0)
	templates = comm.bcast(templates,root=0)
	good_pixels = comm.bcast(good_pixels,root=0)
	

	comm.barrier()
	if rank == 0:
		print(f"Head node distributing {len(logRebin_spectra[0,:])} spectra")
		vorbin_nums = spax_properties['vorbin_num'][:]
		vorbin_nums = np.sort(np.unique(vorbin_nums[vorbin_nums>=0]))
	sys.stdout.flush()
	for nn in range(1,nproc):
		if rank == 0:
			proc_spax = np.arange(nn,len(logRebin_spectra[0,:]),nproc,dtype=int)
			# proc_spax_properies = spax_properties[proc_spax]
			proc_vorbin_nums = vorbin_nums[proc_spax]
			proc_logRebin_spectra = logRebin_spectra[:,proc_spax]
			proc_logRebin_variance = logRebin_variance[:,proc_spax]

			tosend = [proc_vorbin_nums, #proc_spax_properies,
						proc_logRebin_spectra,
						proc_logRebin_variance]
			comm.send(tosend, dest=nn, tag=100+nn)
			tosend = None

		elif rank == nn:
			torecieve = comm.recv(source=0, tag=100+rank)
			# proc_spax_properties = torecieve[0]
			proc_vorbin_nums = torecieve[0]
			proc_logRebin_spectra = torecieve[1]
			proc_logRebin_variance = torecieve[2]
			torecieve = None

	if rank == 0:
		proc_spax = np.arange(0,len(logRebin_spectra[0,:]),nproc,dtype=int)
		# proc_spax_properties = spax_properties[proc_spax]
		proc_vorbin_nums = vorbin_nums[proc_spax]
		proc_logRebin_spectra = logRebin_spectra[:,proc_spax]
		proc_logRebin_variance = logRebin_variance[:,proc_spax]

		spectra_shape = logRebin_spectra.shape


		logRebin_spectra = None
		logRebin_variance = None
		hdul = None

	if rank == 0:
		print(f"Spectra distributed, running fits")
	sys.stdout.flush()

	comm.barrier()

	outputs = []
	outputs_all = []

	# for ss in range(len(proc_logRebin_spectra[0,:])):
	for vb, vorbin_num in enumerate(proc_vorbin_nums[0:5]):


		spectrum = np.array(proc_logRebin_spectra[:,vb])
		variance = np.array(proc_logRebin_variance[:,vb])
		good_pixels_spec = good_pixels.copy()
		good_pixels_spec[np.isfinite(spectrum)==False] = False
		good_pixels_spec[(np.isfinite(variance)==False) | (variance <= 0)] = False

		spec_median = np.abs(np.nanmedian(spectrum))
		var_median = np.abs(np.nanmedian(variance))
		spectrum = spectrum / spec_median			#nomalise spectrum
		variance = variance / var_median			#nomalise variance
		variance[~good_pixels_spec] = 1.e-5			#pPXF doesnt apply goodpix to variance -_-


		out = ppxf(templates, spectrum, variance,
				velscale = parameters['galaxy_velscale'],
				start = start,
				moments = parameters['stars_moments'],
				degree = parameters['stars_degree'],
				velscale_ratio = parameters['velscale_ratio'],
				vsyst = dv,
				goodpixels=np.arange(len(spectrum))[good_pixels],
				plot=False,
				# clean=True,
				quiet=False)
		# plt.show()
		# print(np.median(out.galaxy[good_pixels]))
		# print(np.median(out.galaxy[good_pixels]-out.bestfit[good_pixels]))
		# print(np.std(out.galaxy[good_pixels]-out.bestfit[good_pixels]))
		# print(out.polyweights)
		# print(out.mpolyweights)
		# plt.plot(out.galaxy[good_pixels]-out.bestfit[good_pixels])
		# plt.show()

		# print(vorbin_num)
		outputs.append([vorbin_num,out.sol])
		
	
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
		

		# spax_properties = Table.read(spax_properties_file)
		stellar_kin = np.full([len(spax_properties),4],np.nan)

		# for k in ['V','sigma','h3','h4']:
			# spax_properties[f'{k}_stellar'] = np.full(len(spax_properties['spax_num'][:]), -9999.)
		
		for out in outputs_all:
			vorbin_num = out[0]
			kin = out[1]

			ref = np.where(spax_properties['vorbin_num'][:] == vorbin_num)[0]
			stellar_kin[ref,:] = kin
			# for v,k in enumerate(['V','sigma','h3','h4']):
					# spax_properties[f'{k}_stellar'][ref] = fit.sol[v]
		
		if not os.path.isdir(f"{parameters['output']}/{parameters['output_dir']}"):
			os.mkdir(f"{parameters['output']}/{parameters['output_dir']}")
		
		stellar_kin = Table(stellar_kin,names=('V_stellar','sigma_stellar','h3_stellar','h4_stellar'))
		print("Saving bestfit stellar kinematics table")
		stellar_kin.write(f"{parameters['output']}/{parameters['output_dir']}/bestfit_stellar_kinematics.fits",overwrite=True)
		# spax_properties.write(spax_properties_file,overwrite=True)

def fit_gas_lines(parameterfile):

	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	nproc = comm.Get_size()

	if rank == 0:
		print(f"Running with {nproc} processors")
		print("Preparing inputs")
		sys.stdout.flush()
		parameters = read_parameterfile(parameterfile)
		
		spax_properties_file = f"{parameters['output']}/{parameters['input_dir']}/spaxel_properties.fits"

		spectra_file = f"{parameters['output']}/{parameters['input_dir']}/logRebin_spectra.fits"


		spax_properties = Table.read(spax_properties_file)

		hdul = fits.open(spectra_file)
		logLambda_spec, spectra  = read_spectra_hdul(hdul)
		logRebin_spectra = spectra[0]
		logRebin_variance = spectra[1]
		spectra = None

		header = hdul[0].header
		hdul.close()
		velscale = header['VELSCALE']
		parameters['galaxy_velscale'] = float(velscale)

		stellar_templates, logLambda_templates = read_EMILES_spectra(parameters)

		gas_templates, gas_names, Ncomp_gas = make_gas_templates(parameters,logLambda_templates)

		components_stars = [0]*len(stellar_templates[0,:])
		components_gas = [[nn+components_stars[-1]+1]*NN for nn,NN in enumerate(Ncomp_gas)]
		components_gas = [comp for group in components_gas for comp in group]

		# print(components_gas)
		# exit()

		good_pixels = create_spectrum_mask(logLambda_spec,parameters,gas_fit=True)

		stellar_kin_file = f"{parameters['output']}/{parameters['output_dir']}/bestfit_stellar_kinematics.fits"
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
			proc_logRebin_variance = logRebin_variance[:,proc_spax]

			tosend = [proc_spax_properies,
						proc_spax_stelkin,
						proc_logRebin_spectra,
						proc_logRebin_variance]
			comm.send(tosend, dest=nn, tag=100+nn)
			tosend = None

		elif rank == nn:
			torecieve = comm.recv(source=0, tag=100+rank)
			proc_spax_properties = torecieve[0]
			proc_spax_stelkin = torecieve[1]
			proc_logRebin_spectra = torecieve[2]
			proc_logRebin_variance = torecieve[3]
			torecieve = None

	if rank == 0:
		proc_spax = np.arange(0,len(logRebin_spectra[0,:]),nproc,dtype=int)
		proc_spax_properties = spax_properties[proc_spax]
		proc_spax_stelkin = stel_kin[proc_spax]
		proc_logRebin_spectra = logRebin_spectra[:,proc_spax]
		proc_logRebin_variance = logRebin_variance[:,proc_spax]

		spectra_shape = logRebin_spectra.shape


		# logRebin_spectra = None
		logRebin_variance = None
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
		variance = np.array(proc_logRebin_variance[:,ss])
		good_pixels_spec = good_pixels.copy()
		good_pixels_spec[np.isfinite(spectrum)==False] = False
		good_pixels_spec[(np.isfinite(variance)==False) | (variance <= 0)] = False

		spec_median = np.abs(np.nanmedian(spectrum))
		var_median = np.abs(np.nanmedian(variance))
		spectrum = spectrum / spec_median			#nomalise spectrum
		variance = variance / var_median			#nomalise variance
		variance[~good_pixels_spec] = 1.e-5			#pPXF doesnt apply goodpix to variance -_-


		out = ppxf(templates, spectrum, variance,
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
		gas_kin.write(f"{parameters['output']}/{parameters['output_dir']}/bestfit_gas_kinematics.fits",overwrite=True)
		# spax_properties.write(spax_properties_file,overwrite=True)



#data reading and binning

def read_parameterfile(filename = None):
	if filename is None:
		filename = "./parameters.param"

	parameters = {}


	f = open(filename)
	for line in f:
		# print(line)
		if line[0] == "#" or line[0] == " ":
				
			continue
		else:
			line = line.split("\n")[0].split(" ")
			# print(line)
			# print(line[0])

			if line[0] == "datacube":
				parameters[line[0]] = line[1]
			elif line[0] == "stelkin_templates":
				parameters[line[0]] = line[1]
			elif line[0] == "continuum_templates":
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
				parameters[line[0]] = line[1]
			# elif line[0] == "test":
			# 	parameters[line[0]] = np.asarray(int(line[1]),dtype=bool)
			# elif line[0] == "continuum_subcube":
			# 	parameters[line[0]] = np.asarray(int(line[1]),dtype=bool)
			elif line[0] == "input_dir":
				parameters[line[0]] = line[1]
			elif line[0] == "output_dir":
				parameters[line[0]] = line[1]
			elif line[0] == "stars_start":
				parameters[line[0]] = [float(line[1]),float(line[2])]
			elif line[0] == "stars_lrange":
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
			elif line[0] == "continuum_gaslines":
				gas_lines, gas_constraints = read_gaslines_parameterfile(line[1])
				parameters['gas_groups'] = gas_lines
				parameters['gas_constraints'] = gas_constraints
			elif line[0] == "gas_gaslines":
				gas_lines, gas_constraints = read_gaslines_parameterfile(line[1])
				parameters['gas_groups'] = gas_lines
				parameters['gas_constraints'] = gas_constraints
			elif len(line[1::]) == 1:
				try:
					parameters[line[0]] = int(line[1])
				except:
					parameters[line[0]] = float(line[1])



			# else:
			# 	parameters[line[0]] = [int(x) for x in line[1::]]

			# elif line[0] in ["z","Lmin","Lmax","SN_Lmin","SN_Lmax","SN_indiv","SN_vorbin"]:
			# 	parameters[line[0]] = float(line[1])
			# else:
			# 	parameters[line[0]] = [int(x) for x in line[1::]]

	# print(parameters)
	# exit()
	# parameters['velscale_ratio'] = int(parameters['velscale_ratio'])

	if 'gas_groups' not in parameters.keys():
		parameters['gas_groups'] = None

	if "input_dir" not in parameters.keys():
		parameters['input_dir'] = parameters['output_dir']
	
	# if "stars_lrange" not in parameters.keys():
	# 	parameters['stars_lrange'] = [parameters['Lmin'],parameters['Lmax']]


	return parameters

def read_gaslines_parameterfile(filename = None):
	if filename is None:
		filename = "./gaslines.param"

	parameters = {}

	f = open(filename)


	linegroups = []
	linegroup = []
	constr_kinem = {'A_ineq':[],'b_ineq':[]}
	for line in f:
		if line[0] == "#" or line[0] == " " or line[0] == "\n":
			if linegroup != []:	
				linegroups.append(linegroup)
				linegroup = []
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
			linegroup.extend(line)
	if linegroup != []:	
		linegroups.append(linegroup)

	# print(linegroups)
	# print(constr_kinem)
	# exit()
	return linegroups, constr_kinem

def get_constraints(parameters = None):
	if parameters is None:
		parameters = read_parameterfile()


	if 'gas_constraints' not in parameters.keys():
		# print('entering default dummy constraint')

		constr_kinem = {'A_ineq':[[0.,-1.,0.,0.]],'b_ineq':[0]}
	elif parameters['gas_constraints']['A_ineq'] == []:
			# print('entering default dummy constraint')
			constr_kinem = {'A_ineq':[[0.,-1.,0.,0.] + [0.,0.]*len(parameters['gas_groups'])],'b_ineq':[0]}
	else:
		constr_kinem = parameters['gas_constraints']
		for aa in range(len(constr_kinem['A_ineq'])):
			constr_kinem['A_ineq'][aa] = [0.,0.,0.,0.] + constr_kinem['A_ineq'][aa]
			constr_kinem['b_ineq'][aa] /= parameters['galaxy_velscale']

	return constr_kinem



def make_spectra_tables(parameters = None):

	if parameters is None:
		parameters = read_parameterfile()


	if not os.path.isdir(f"{parameters['output']}/indiv"):
		os.mkdir(f"{parameters['output']}/indiv")
	if not os.path.isdir(f"{parameters['output']}/testcube"):
		os.mkdir(f"{parameters['output']}/testcube")
	if not os.path.isdir(f"{parameters['output']}/test"):
		os.mkdir(f"{parameters['output']}/test")



	if os.path.isfile(f"{parameters['output']}/indiv/spaxel_properties.fits") and\
		os.path.isfile(f"{parameters['output']}/indiv/spectra_indiv.fits") and\
		os.path.isfile(f"{parameters['output']}/indiv/logRebin_spectra.fits"):

		print('Individual spectra produects exist, going to test spectra creation')

		spax_properties = Table.read(f"{parameters['output']}/indiv/spaxel_properties.fits")
		Nx = spax_properties.meta['NX']
		Ny = spax_properties.meta['NY']
		Nl = spax_properties.meta['NL']

		hdul = fits.open(f"{parameters['output']}/indiv/spectra_indiv.fits")
		obsLambda, spectra_list  = read_spectra_hdul(hdul)
		spectra = spectra_list[0]
		variance = spectra_list[1]
		spectra_list = None
		header = hdul[0].header
		hdul.close()

		hdul = fits.open(f"{parameters['output']}/indiv/logRebin_spectra.fits")
		logLambda, spectra_list  = read_spectra_hdul(hdul)
		logRebin_spectra = spectra_list[0]
		logRebin_variance = spectra_list[1]
		spectra_list = None
		header = hdul[0].header
		velscale = [header['VELSCALE']]
		hdul.close()

		print("All read in")


	else:

		hdu = fits.open(parameters['datacube'])
		header = hdu[1].header

		spectra = hdu[1].data
		variance = hdu[2].data


		Nx = header['NAXIS1']
		Ny = header['NAXIS2']
		Nl = header['NAXIS3']

		spectra = spectra.reshape(Nl, Nx*Ny)
		variance = variance.reshape(Nl, Nx*Ny)

		obsLambda = astrofunc.get_wavelength_axis(header)
		#de-redshift spectra
		obsLambda = obsLambda / (1.e0 + parameters['z'])

		Lambda_range = np.logical_and(obsLambda >= parameters['Lmin'] ,
									obsLambda <= parameters['Lmax'])

		spectra = spectra[Lambda_range]
		variance = variance[Lambda_range]
		obsLambda = obsLambda[Lambda_range]


		spax_number = np.arange(Nx*Ny,dtype='int')
		vorbin_number = np.arange(Nx*Ny,dtype='int')
		obs_flags = np.all(np.isfinite(spectra), axis = 0)
		obs_flags_nums = np.zeros(Nx*Ny)
		obs_flags_nums[obs_flags] = 1

		spec_good = np.where(obs_flags_nums == 1)[0]
		spec_bad = np.where(obs_flags_nums == 0)[0]



		SN_cont_Lrange = np.logical_and(obsLambda >= parameters['SN_cont'][0] , 
										obsLambda <= parameters['SN_cont'][1])
		spax_signal_cont = np.nanmedian(spectra[SN_cont_Lrange,:], axis = 0)
		spax_noise_cont = np.abs(np.nanmedian(np.sqrt(variance[SN_cont_Lrange,:])), axis = 0)

		if 'SN_line' in parameters.keys():
			spax_signal_line = np.zeros([Nx*Ny,len(parameters['SN_line'])])
			spax_noise_line = np.zeros([Nx*Ny,len(parameters['SN_line'])])
			spax_line_names = []

			for ii in range(len(parameters['SN_line'])):
				print(emlines[parameters['SN_line'][ii]]['lambda'])
				print(emlines[parameters['SN_line'][ii]]['lambda'][-1])
				
				lineLambda = emlines[parameters['SN_line'][ii]]['lambda'][-1]

				SN_line_Lrange = np.logical_and(obsLambda >= lineLambda*(1-500/c) , 
										obsLambda <= lineLambda*(1+500/c) )
				spax_signal_line[:,ii] = np.nansum(spectra[SN_line_Lrange,:], axis = 0)
				spax_noise_line[:,ii] = np.nansum(variance[SN_line_Lrange,:], axis = 0)



		spax_xxyy , spax_RADEC, spax_SP = astrofunc.make_pix_WCS_grids(header)
		spax_size = np.abs(np.diff(spax_SP[0])[0])

		spax_properties = np.column_stack([spax_number,
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
										spax_signal_line,
										spax_noise_line]
										)

		#trim to only good observed spaxels to save memory
		spax_properties = spax_properties[spec_good]

		metadata = {'Nx':Nx,'Ny':Ny,'Nl':Nl,'d_spax':spax_size}

		spax_properties = Table(spax_properties,
					names=['spax_num','vorbin_num','obs_flag','spax_xx','spax_yy',
							'spax_RA','spax_DEC','spax_SPxx','spax_SPyy',
							'spax_signal_cont','spax_noise_cont']+\
							[f'spax_signal_{line}' for line in parameters['SN_line']]+\
							[f'spax_noise_{line}' for line in parameters['SN_line']],
							meta=metadata)

		
		print("Saving individual spaxel properties")
		#save individual spaxel properties
		spax_properties.write(f"{parameters['output']}/indiv/spaxel_properties.fits",overwrite=True)
		print("Saved")


		#trim spectra to only observed spaxels
		spectra = spectra[:,spec_good]
		variance = variance[:,spec_good]

		#save individual spectra
		indiv_header = fits.Header()
		indiv_header['COMMENT'] = "A.B. Watts"
		indiv_primary_hdu = fits.PrimaryHDU(data = obsLambda,
											header = indiv_header)

		indiv_spectra_hdu = fits.BinTableHDU.from_columns(
								fits.ColDefs([
								fits.Column(
								array = spectra.T,
								name='SPEC',format=str(len(spectra))+'D'
								)]))

		indiv_variance_hdu = fits.BinTableHDU.from_columns(
								fits.ColDefs([
								fits.Column(
								array = variance.T,
								name='VAR',format=str(len(variance))+'D' 
								)]))

		hdul_indiv = fits.HDUList([indiv_primary_hdu,
									indiv_spectra_hdu,
									indiv_variance_hdu])
		print("Saving indiv. spectra table")
		hdul_indiv.writeto(f"{parameters['output']}/indiv/spectra_indiv.fits",overwrite=True)
		print("Saved")

		#log-rebin the individual spectra
		obsLambda_range = obsLambda[[0,-1]]
		print("log-rebinning the individual spectra")
		logRebin_spectra, logLambda, velscale = pputils.log_rebin(obsLambda_range,
																	spectra)
		
		logRebin_variance, logLambda1, velscale1 = pputils.log_rebin(obsLambda_range,
																	variance)
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

		logRebin_variance_hdu = fits.BinTableHDU.from_columns(
								fits.ColDefs([
								fits.Column(
								array = logRebin_variance.T,
								name='VAR',format=str(len(logRebin_variance))+'D'
								)]))


		hdul_logRebin = fits.HDUList([logRebin_primary_hdu,
										logRebin_spectra_hdu,
										logRebin_variance_hdu])
		
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
	
	variance_testcube = testcube.unmasked_data[:].value
	variance_header = testcube.header

	data_testcube = np.swapaxes(data_testcube,1,2)
	variance_testcube = np.swapaxes(variance_testcube,1,2)


	hdu0 = fits.PrimaryHDU(header=fits.Header())
	hdu1 = fits.ImageHDU(data_testcube,header=data_header)
	hdu2 = fits.ImageHDU(variance_testcube,header=variance_header)

	hdul = fits.HDUList([hdu0,hdu1,hdu2])
	hdul.writeto("./outputs/testcube/testcube.fits",overwrite=True)


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
	variance_testcube = variance[:,testcube_locs]

	logRebin_spectra_testcube = logRebin_spectra[:,testcube_locs]
	logRebin_variance_testcube = logRebin_variance[:,testcube_locs]


	#save individual testcube spectra
	indiv_header = fits.Header()
	indiv_header['COMMENT'] = "A.B. Watts"
	indiv_primary_hdu = fits.PrimaryHDU(data = obsLambda,
										header = indiv_header)

	indiv_spectra_hdu = fits.BinTableHDU.from_columns(
							fits.ColDefs([
							fits.Column(
							array = spectra_testcube.T,
							name='SPEC',format=str(len(spectra))+'D'
							)]))

	indiv_variance_hdu = fits.BinTableHDU.from_columns(
							fits.ColDefs([
							fits.Column(
							array = variance_testcube.T,
							name='VAR',format=str(len(variance))+'D' 
							)]))

	hdul_indiv = fits.HDUList([indiv_primary_hdu,
								indiv_spectra_hdu,
								indiv_variance_hdu])
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

	logRebin_variance_hdu = fits.BinTableHDU.from_columns(
							fits.ColDefs([
							fits.Column(
							array = logRebin_variance_testcube.T,
							name='VAR',format=str(len(logRebin_variance))+'D'
							)]))


	hdul_logRebin = fits.HDUList([logRebin_primary_hdu,
									logRebin_spectra_hdu,
									logRebin_variance_hdu])
	
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
	variance_test = variance[:,SN_ids]

	logRebin_spectra_test = logRebin_spectra[:,SN_ids]
	logRebin_variance_test = logRebin_variance[:,SN_ids]



	#save individual test spectra
	indiv_header = fits.Header()
	indiv_header['COMMENT'] = "A.B. Watts"
	indiv_primary_hdu = fits.PrimaryHDU(data = obsLambda,
										header = indiv_header)

	indiv_spectra_hdu = fits.BinTableHDU.from_columns(
							fits.ColDefs([
							fits.Column(
							array = spectra_test.T,
							name='SPEC',format=str(len(spectra))+'D'
							)]))

	indiv_variance_hdu = fits.BinTableHDU.from_columns(
							fits.ColDefs([
							fits.Column(
							array = variance_test.T,
							name='VAR',format=str(len(variance))+'D' 
							)]))

	hdul_indiv = fits.HDUList([indiv_primary_hdu,
								indiv_spectra_hdu,
								indiv_variance_hdu])
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

	logRebin_variance_hdu = fits.BinTableHDU.from_columns(
							fits.ColDefs([
							fits.Column(
							array = logRebin_variance_test.T,
							name='VAR',format=str(len(logRebin_variance))+'D'
							)]))


	hdul_logRebin = fits.HDUList([logRebin_primary_hdu,
									logRebin_spectra_hdu,
									logRebin_variance_hdu])
	
	print("Saving log-rebinnned indiv. test spectra table")
	hdul_logRebin.writeto(f"{parameters['output']}/test/logRebin_spectra.fits",overwrite=True)
	print("Saved")

def voronoi_bin_cube(parameters =  None):

	if parameters is None:
		parameters = read_parameterfile()

	spax_prop_file = f"{parameters['output']}/{parameters['input_dir']}/spaxel_properties.fits"
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

	SNname = parameters['vorbin_SNname']

	#get spaxels above desired SN
	spax_SN = spax_properties[f'spax_signal_{SNname}'] /  spax_properties[f'spax_noise_{SNname}']

	spax_xx = spax_properties['spax_xx'][spax_SN >= SN_indiv]
	spax_yy = spax_properties['spax_yy'][spax_SN >= SN_indiv]
	spax_signal = spax_properties[f'spax_signal_{SNname}'][spax_SN >= SN_indiv]
	spax_noise = spax_properties[f'spax_noise_{SNname}'][spax_SN >= SN_indiv]

	#compute Voronoi bins
	print('Computing voronoi bins')
	vorbin_nums, Nxx, Nyy, vorbin_xx, vorbin_yy, vorbin_SN, vorbin_Npix, scale = \
					voronoi_2d_binning(spax_xx, spax_yy, 
										spax_signal, spax_noise, 
										SN_vorbin, pixelsize=spax_size,
										plot = False)


	#record vorbin properties for each spaxel
	spax_properties['vorbin_num'][spax_SN >= SN_indiv] = vorbin_nums
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


	target_dir = f"{parameters['output']}/{parameters['output_dir']}/"
	if not os.path.isdir(target_dir):
		os.mkdir(target_dir)
		os.mkdir(f"{target_dir}/figures")

	# spax_properties = spax_properties[spax_properties.argsort(spax_properties['vorbin_num'])]
	spax_properties.write(f"{target_dir}/spaxel_properties.fits",overwrite=True)

	create_binned_spectra(parameters)

	make_vorbins_map(parameters)

def create_binned_spectra(parameters = None):

	if parameters is None:
		parameters = read_parameterfile()

	# print(parameters)
	spax_prop_file = f"{parameters['output']}/"+\
						f"{parameters['output_dir']}/"+\
						f"spaxel_properties.fits"
	
	spax_properties = Table.read(spax_prop_file)

	# spectra_file = parameters['vorbin_spectra_file']
	# if isinstance(spectra_file,type(None)):
	spectra_file = f"{parameters['output']}/{parameters['input_dir']}/spectra_indiv.fits"

	hdul = fits.open(spectra_file)
	header = hdul[0].header
	#is there a faster way to read these?
	obsLambda, spectra_list = read_spectra_hdul(hdul)
	hdul.close()
	spectra = spectra_list[0]
	variance = spectra_list[1]
	# varbin_flag = True
	# else:
	# 	hdul = fits.open(spectra_file)
	# 	header = hdul[0].header
	# 	#is there a faster way to read these?
	# 	obsLambda, spectra_list = read_spectra_hdul(hdul)
	# 	spectra = spectra_list[0]
	# 	varbin_flag = False
	# 	# print(header)
	
	# if "logRebin" in spectra_file:				#you're summing something already log-rebinned, like a cont. sub. cube
	# 	logRebin_flag = False
	# else:
	# 	logRebin_flag = True

	
	obsLambda_range = [obsLambda[0],obsLambda[-1]]

	vorbin_nums = np.unique(spax_properties['vorbin_num'])
	vorbin_nums = np.sort(vorbin_nums[vorbin_nums >= 0])


	# if logRebin_flag:

	vorbin_spectra = np.zeros([len(spectra),len(vorbin_nums)])
	vorbin_variance = vorbin_spectra.copy()

	for nn in vorbin_nums:

		inbin = np.in1d(spax_properties['vorbin_num'][:], nn)

		vorbin_spectra[:,nn] = np.sum(spectra[:,inbin], axis=1)
		vorbin_variance[:,nn] = np.sum(variance[:,inbin], axis=1)

	# else:
	# 	logRebin_vorbin_spectra = np.zeros([len(spectra),len(vorbin_nums)])
	# 	for nn in vorbin_nums:
	# 		inbin = np.where(spax_properties['vorbin_num'][:] == nn)[0]
	# 		logRebin_vorbin_spectra[:,nn] = np.sum(spectra[:,inbin], axis=1)
	# 	logLambda = obsLambda

	# if logRebin_flag:
		#save vorbin spectra
	vorbin_header = fits.Header()
	vorbin_header['COMMENT'] = "A.B. Watts"
	vorbin_primary_hdu = fits.PrimaryHDU(data = obsLambda,
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
	vorbin_variance_hdu = fits.BinTableHDU.from_columns(
							fits.ColDefs([
							fits.Column(
							array = vorbin_variance.T,
							name='VAR',format=str(len(vorbin_variance))+'D' 
							)]))
	hdul_vorbin.append(vorbin_variance_hdu)
	

	print("Saving vorbin. spectra table")
	hdul_vorbin.writeto(f"{parameters['output']}/{parameters['output_dir']}/spectra_vorbin.fits",overwrite=True)
	print("Saved")

	logRebin_vorbin_spectra, logLambda, velscale = pputils.log_rebin(obsLambda_range,
																vorbin_spectra)
	
	# if varbin_flag:
	logRebin_vorbin_variance, logLambda1, velscale1 = pputils.log_rebin(obsLambda_range,
																vorbin_variance)

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
	logRebin_vorbin_variance_hdu = fits.BinTableHDU.from_columns(
							fits.ColDefs([
							fits.Column(
							array = logRebin_vorbin_variance.T,
							name='VAR',format=str(len(logRebin_vorbin_variance))+'D' 
							)]))

	
	hdul_logRebin_vorbin.append(logRebin_vorbin_variance_hdu)
	print("Saving log-rebinnned. vorbin spectra table")
	hdul_logRebin_vorbin.writeto(f"{parameters['output']}/{parameters['output_dir']}/logRebin_spectra.fits",overwrite=True)
	print("Saved")

def read_spectra_hdul(hdul):

	wave = np.asarray(hdul[0].data)
	spectra = []
	for hdu in hdul[1::]:
		spectra.append(np.asarray([dd[0] for dd in hdu.data]).T)

	return wave, spectra


def read_EMILES_spectra(parameters,regrid=False):

	if parameters is None:
		parameters = read_parameterfile()

	FWHM_template = 2.51
	try:
		template_dir = parameters['stelkin_templates']
	except:
		template_dir = parameters['continuum_templates']

	files = glob.glob(template_dir + "/*")

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


	template_velscale = parameters['galaxy_velscale'] / parameters['velscale_ratio']


	for ff in range(len(files)):

		file = files[ff]
		hdu = fits.open(file)
		template = hdu[0].data
		header = hdu[0].header
		hdu.close()

		if ff == 0:

			lambda0 = header['CRVAL1']
			lambda_array = lambda0 + np.arange(header['NAXIS1'])*header['CDELT1']
			# print(lambda_array[-1])
			# lambda_range = np.array([0, (header['NAXIS1']-1)*header['CDELT1']]) + lambda0
			templates = np.zeros([header['NAXIS1'],len(files)])
			templates = templates[(lambda_array>=4000) & (lambda_array<10000),:]
			
			lambda_array_trunc = lambda_array[(lambda_array>=4000) & (lambda_array<10000)]
			lambda_range = [np.min(lambda_array_trunc),np.max(lambda_array_trunc)]


		template = template[(lambda_array>=4000) & (lambda_array<10000)]

		FWHM_diffs = np.sqrt(MUSE_LSF_Bacon17(lambda_array_trunc,z = parameters['z'])**2.e0 - 
						np.full_like(lambda_array_trunc,FWHM_template)**2.e0 )
		FWHM_diffs[np.isfinite(FWHM_diffs)==False] = 0.01
							
		stddev_diffs = FWHM_diffs / (header["CDELT1"]  * 2.355)

		template_conv = pputils.varsmooth(lambda_array_trunc,template,stddev_diffs)

		# template_conv = np.zeros(len(lambda_array_trunc))

		# for ll in range(len(lambda_array_trunc)):
		
		# 	template_temp = np.zeros(len(lambda_array_trunc))
		# 	template_temp[ll] = template[ll]

		# 	if np.isfinite(stddev_diffs[ll]):
		# 		template_conv += ndimage.gaussian_filter1d(template_temp, stddev_diffs[ll])
		# 	else:
		# 		template_conv += template_temp

		
		templates[:,ff] =  template_conv

	templates_logRebin, logLambda, temp_velscale = pputils.log_rebin(lambda_range,
																templates,
																velscale = template_velscale)
	
	if not regrid:
		templates_combined = templates_logRebin
		templates_combined /= np.median(templates_combined,axis=0)


	elif regrid:

		templates_combined = np.full([len(templates_logRebin[:,0]),
											Nages,
											NZs,
											Nalphas],np.nan)


		for tt in range(len(templates_logRebin[0,:])):
			age_loc = np.where(Ages == template_params[tt,0])[0][0]
			Z_loc = np.where(Zs == template_params[tt,1])[0][0]
			alpha_loc = np.where(Alphas == template_params[tt,2])[0][0]
			

			templates_combined[:,age_loc,Z_loc,alpha_loc] = templates_logRebin[:,tt]
			
		templates_combined /= np.median(templates_combined)


	return templates_combined, logLambda

def make_gas_templates(parameters,logLambda_templates):

	# emlines = {'Hbeta':{	'lambda':[4861.333],			'ratio':[1]},
	# 			'OIII':{	'lambda':[4958.911, 5006.843],	'ratio':[0.33,1]}, 
	# 			'HeI5876':{	'lambda':[5875.624],			'ratio':[1]}, 
	# 			'OI':{		'lambda':[6300.304,6363.78],	'ratio':[1,0.33]},
	# 		 	'NII':{		'lambda':[6548.050,6583.460],	'ratio':[0.33,1]},
	# 			'Halpha':{	'lambda':[6562.819],			'ratio':[1]},
	# 			'HeI6678':{	'lambda':[6678.151],			'ratio':[1]}, 
	# 			'SII6716':{	'lambda':[6716.440],			'ratio':[1]},
	# 			'SII6730':{	'lambda':[6730.810],			'ratio':[1]},
	# 			'ArIII7135':{'lambda':[7135.790],			'ratio':[1]}, 
	# 			'OII':{		'lambda':[7319.990, 7330.730],	'ratio':[1,1]}, #?? check
	# 			'ArIII7751':{'lambda':[7751.060],			'ratio':[1]},
	# 			'SIII':{	'lambda':[9068.6],				'ratio':[1]}
	# 			}


	def FWHM_temp_obsframe(ll):
		ff = MUSE_LSF_Bacon17(ll*(1.+parameters['z'])) / (1.e0 + parameters['z'])
		return ff



	Ncomp = []
	gas_names = []
	templates = []

	gas_groups = parameters['gas_groups']
	for gg, group in enumerate(gas_groups):
		for gl, gas_line in enumerate(group):
			gas_lambda = emlines[gas_line]['lambda']
			gas_ratio = emlines[gas_line]['ratio']
			

			line_template = pputils.gaussian(logLambda_templates,
												np.asarray(gas_lambda),
												# FWHM_temp_obsframe,
												0,
												pixel = True) \
												@ gas_ratio

			gas_names.extend([f"C{gg+1}-{gas_line}"])
			templates.append(line_template)
		Ncomp.extend([gl+1])

	templates = np.array(templates).T
	
	return templates, gas_names, Ncomp

def create_spectrum_mask(logLambda, parameters, gas_fit = False):

	goodpix = np.zeros_like(logLambda,dtype=bool)
	z = parameters['z']
	linLambda = np.exp(logLambda)

	width = 400 / 299792.458

	# #list of all emission lines
	# emlines = {'Hbeta':{	'lambda':[4861.333],			'ratio':[1]},
	# 		'OIII':{	'lambda':[4958.911, 5006.843],	'ratio':[0.35,1]}, 
	# 		'HeI5876':{	'lambda':[5875.624],			'ratio':[1]}, 
	# 		'OI':{		'lambda':[6300.304,6363.78],	'ratio':[1,0.33]},
	# 	 	'NII':{		'lambda':[6548.050,6583.460],	'ratio':[0.34,1]},
	# 		'Halpha':{	'lambda':[6562.819],			'ratio':[1]},
	# 		'HeI6678':{	'lambda':[6678.151],			'ratio':[1]}, 
	# 		'SII6716':{	'lambda':[6716.440],			'ratio':[1]},
	# 		'SII6730':{	'lambda':[6730.810],			'ratio':[1]},
	# 		'ArIII7135':{'lambda':[7135.790],			'ratio':[1]}, 
	# 		'OII':{		'lambda':[7319.990, 7330.730],	'ratio':[1,1]}, #?? check
	# 		'ArIII7751':{'lambda':[7751.060],			'ratio':[1]},
	# 		'SIII':{	'lambda':[9068.6],				'ratio':[1]}
	# 		}


	if not isinstance(parameters['gas_groups'],type(None)):
		emission_lines_fit = [line for group in parameters['gas_groups'] for line in group]
		emission_lines_fit = np.unique(np.array(emission_lines_fit))
	else:
		emission_lines_fit = None


	emission_lines = np.array([])
	for line in emlines:
		if not gas_fit:
			emission_lines = np.append(emission_lines,
								np.array(emlines[line]['lambda']))
		elif line not in emission_lines_fit:
			emission_lines = np.append(emission_lines,
								np.array(emlines[line]['lambda']))


	#		OI 			OI 		OI 	
	sky = np.array([5577.338, 6300.304, 6363.78])/(1.e0+z)		
	#					NaD 				CaT
	absorption_lines = np.array([5889.95,5895.92, 8498, 8542, 8662]) 


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



#making maps
def make_vorbins_map(parameters):
	spax_prop_file = f"{parameters['output']}/"+\
						f"{parameters['output_dir']}/"+\
						f"spaxel_properties.fits"


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
	ax.pcolormesh(img_grid)
	# ax.scatter(spax_properties['spax_xx'],spax_properties['spax_yy'],c=spax_properties['vorbin_num'])
	plt.show()
	fig.savefig(f"{parameters['output']}/"+\
						f"{parameters['output_dir']}/"+\
						f"figures/vorbin_map.pdf")

def make_stelkin_map(parameters):
	spax_prop_file = f"{parameters['output']}/"+\
						f"{parameters['input_dir']}/"+\
						f"spaxel_properties.fits"

	fit_stellar_kinematics = f"{parameters['output']}/"+\
						f"{parameters['output_dir']}/"+\
						f"bestfit_stellar_kinematics.fits"

	spax_properties = Table.read(spax_prop_file)
	bestfit_stelkin = Table.read(fit_stellar_kinematics)


	spax_properties = hstack((spax_properties,bestfit_stelkin))

	meta = spax_properties.meta
	NY = meta['NY']
	NX = meta['NX']

	img_grid = np.full([NY,NX],np.nan).flatten()
	# print(np.asarray(spax_properties['spax_num'],dtype=int))


	for comp in ['V_stellar','sigma_stellar','h3_stellar','h4_stellar']:


		fig = plt.figure()
		gs = gridspec.GridSpec(1,1)
		ax = fig.add_subplot(gs[0,0])

		img = img_grid.copy()
		img[np.asarray(spax_properties['spax_num'],dtype=int)] = spax_properties[comp]
		img = img.reshape((NY,NX))

		if comp == 'V_stellar':
			img -= np.nanmedian(img)
			vmin = np.percentile(img[np.isfinite(img)].flatten(),16)
			vmax = np.percentile(img[np.isfinite(img)].flatten(),84)
			cmap = 'RdBu_r'

			iimmgg = ax.pcolormesh(img,vmin=vmin,vmax=vmax,cmap=cmap)

		elif comp == 'sigma_stellar':
			vmin = 0
			vmax = np.percentile(img[np.isfinite(img)].flatten(),84)
			cmap = 'inferno'

			iimmgg = ax.pcolormesh(img,vmin=vmin,vmax=vmax,cmap=cmap)

		else:
			vmin = 0.2
			vmax = -0.2
			cmap = 'inferno'
			iimmgg = ax.pcolormesh(img,vmin=vmin,vmax=vmax)



		fig.colorbar(iimmgg)
		# ax.scatter(spax_properties['spax_xx'],spax_properties['spax_yy'],c=spax_properties['vorbin_num'])
		# plt.show()
		fig.savefig(f"{parameters['output']}/"+\
							f"{parameters['output_dir']}/"+\
							f"figures/{comp}_map.pdf")
		plt.close()




##### things for analysing outputs #####
def make_continuum_subtracted_spectra(parameters):

	spax_properties = Table.read(f"{parameters['output']}/{parameters['output_dir']}/spaxel_properties.fits")
	Nx = spax_properties.meta['NX']
	Ny = spax_properties.meta['NY']

	bestfit_continuum_file = f"{parameters['output']}/{parameters['output_dir']}/bestfit_continuum_spectra.fits"
	hdul = fits.open(bestfit_continuum_file)
	logLambda, spectra = read_spectra_hdul(hdul)
	hdul.close()
	bestfit_continuum = spectra[0]
	logLambda = np.log(logLambda)

	spectra_file = f"{parameters['output']}/{parameters['output_dir']}/logRebin_spectra.fits"
	hdul = fits.open(spectra_file)
	logLambda, spectra = read_spectra_hdul(hdul)
	hdul.close()
	galaxy_spectra = spectra[0]

	contsub_spectra = galaxy_spectra - bestfit_continuum

	Nl = logLambda.shape[0]

	empty_cube = np.zeros([Nl,Nx*Ny])

	galaxy_cube = empty_cube.copy()
	continuum_cube = empty_cube.copy()
	contsub_cube = empty_cube.copy()


	for vv in range(galaxy_spectra.shape[1]):
		inbin = np.where(spax_properties['vorbin_num']==vv)[0]
		spax_num_inbin = np.array(spax_properties['spax_num'][inbin],dtype=int)
		
		galaxy_cube[:,spax_num_inbin] = np.full((len(inbin),Nl),galaxy_spectra[:,vv]).T
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

	hdul = fits.HDUList([primary_hdu,
								spectra_hdu])
	print("Saving continuum-subtracted spectra table")
	hdul.writeto(f"{parameters['output']}/{parameters['output_dir']}/logRebin_contsub_spectra.fits",overwrite=True)
	print("Saved")


	print("Saving datacubes")

	continuum_cube = np.zeros([Nl,Nx*Ny])
	continuum_cube[:,np.array(spax_properties['spax_num'][:],dtype=int)] = bestfit_continuum
	continuum_cube = continuum_cube.reshape(Nl,Ny,Nx)

	contsub_cube = np.zeros([Nl,Nx*Ny])
	contsub_cube[:,np.array(spax_properties['spax_num'][:],dtype=int)] = contsub_spectra
	contsub_cube = contsub_cube.reshape(Nl,Ny,Nx)


	cube_file = parameters['datacube']
	cube_hdu = fits.open(cube_file)
	cube_header = cube_hdu[1].header
	cube_hdu.close()
	cube_header['NAXIS3'] = len(logLambda)
	cube_header['CRVAL3'] = logLambda[0]
	cube_header['CRPIX3'] = 1
	print(cube_header.keys())
	if any(key == "CDELT3" for key in cube_header.keys()):
		cube_header['CDELT3'] = np.abs(np.diff(logLambda))[0]
	else:
		cube_header['CD3_3'] = np.abs(np.diff(logLambda))[0]

	# cube_header['CTYPE3'] = "AWAV-LOG"


	primary_hdu = fits.PrimaryHDU(header = fits.Header())
	galaxy_cube_hdu = fits.ImageHDU(galaxy_cube,
							name='galaxy',
							header = cube_header)
	hdul = fits.HDUList([primary_hdu,
								galaxy_cube_hdu])
	print("Saving log-rebinned galaxy cube")
	hdul.writeto(f"{parameters['output']}/{parameters['output_dir']}/logRebingalaxy_cube.fits",overwrite=True)
	print("Saved")



	primary_hdu = fits.PrimaryHDU(header = fits.Header())
	continuum_cube_hdu = fits.ImageHDU(continuum_cube,
							name='CONTINUUM',
							header = cube_header)
	hdul = fits.HDUList([primary_hdu,
								continuum_cube_hdu])
	print("Saving bestfit continuum cube")
	hdul.writeto(f"{parameters['output']}/{parameters['output_dir']}/bestfit_continuum_cube.fits",overwrite=True)
	print("Saved")


	primary_hdu = fits.PrimaryHDU(header = fits.Header())
	contsub_cube_hdu = fits.ImageHDU(contsub_cube,
							name='CONTSUB',
							header = cube_header)
	hdul = fits.HDUList([primary_hdu,
								contsub_cube_hdu])
	print("Saving continuum-subtracted cube")
	hdul.writeto(f"{parameters['output']}/{parameters['output_dir']}/contsub_cube.fits",overwrite=True)
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
	# hdul.writeto(f"{parameters['output']}/{parameters['output_dir']}/linefree_cube.fits",overwrite=True)
	# print("Saved")
	

def check_continuum_at_emissionlines(parameters):
	
	spax_properties = Table.read(f"{parameters['output']}/{parameters['output_dir']}/spaxel_properties.fits")
	bright_spectra = np.argsort(spax_properties['spax_signal_cont']/spax_properties['spax_noise_cont'])[::-1][0:5]

	spectra_file = f"{parameters['output']}/{parameters['output_dir']}/logRebin_spectra.fits"
	hdul = fits.open(spectra_file)
	logLambda, spectra = read_spectra_hdul(hdul)
	linLambda = np.exp(logLambda)
	hdul.close()
	spectra = spectra[0]
	spectra = spectra[:,bright_spectra]

	continuum_spectra_file = f"{parameters['output']}/{parameters['output_dir']}/bestfit_continuum_spectra.fits"
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
	fig1.savefig(f"{parameters['output']}/{parameters['output_dir']}/figures/absline_check.pdf")
	fig2.savefig(f"{parameters['output']}/{parameters['output_dir']}/figures/emline_check.pdf")


def check_contsub_emlines(parameters):
	spax_properties = Table.read(f"{parameters['output']}/{parameters['output_dir']}/spaxel_properties.fits")
	bright_spectra = np.argsort(spax_properties['spax_signal_cont']/spax_properties['spax_noise_cont'])[::-1][0:5]

	spectra_file = f"{parameters['output']}/{parameters['output_dir']}/logRebin_contsub_spectra.fits"
	hdul = fits.open(spectra_file)
	logLambda, spectra = read_spectra_hdul(hdul)
	linLambda = np.exp(logLambda)
	hdul.close()
	spectra = spectra[0]
	spectra = spectra[:,bright_spectra]

	# continuum_spectra_file = f"{parameters['output']}/{parameters['output_dir']}/bestfit_continuum_spectra.fits"
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
	fig1.savefig(f"{parameters['output']}/{parameters['output_dir']}/figures/contsub_spectra_check.pdf")
	# fig2.savefig(f"{parameters['output']}/{parameters['output_dir']}/figures/emline_check.pdf")


def find_signal_free_spaxels(parameters):
	spectra_file = f"{parameters['output']}/{parameters['output_dir']}/logRebin_contsub_spectra.fits"
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

	np.savetxt(f"{parameters['output']}/{parameters['output_dir']}/wavelength_sigma.txt",np.vstack([logLambda,sigmas]).T)


	# for ii in range(len(lowsig))[::-1]:
	# 	print(HA_signal[lowsig[ii]])
	# 	print(np.std(spectra[specrange1,lowsig[ii]]))
	# 	plt.plot(np.arange(len(logLambda))[specrange1],spectra[specrange1,lowsig[ii]])
	# 	plt.ylim([-20,20])
	# 	plt.fill_between(np.arange(len(logLambda))[specrange],y1=20,y2=-20,color='Grey',alpha=0.5)
	# 	plt.show()
	# exit()
	
	

def make_line_subcubes(parameters):

	spax_properties = Table.read(f"{parameters['output']}/{parameters['output_dir']}/spaxel_properties.fits")
	Nx = spax_properties.meta['NX']
	Ny = spax_properties.meta['NY']

	sigmas =  np.loadtxt(f"{parameters['output']}/{parameters['output_dir']}/wavelength_sigma.txt")
	sigmaLambda =  sigmas[:,1]

	spectra_file = f"{parameters['output']}/{parameters['output_dir']}/logRebin_contsub_spectra.fits"

	hdul = fits.open(spectra_file)
	logLambda, spectra = read_spectra_hdul(hdul)
	linLambda = np.exp(logLambda)
	hdul.close()
	contsub_spectra = spectra[0]
	spectra = None
	Nl = contsub_spectra.shape[1]


	spectrum_mask = np.zeros_like(contsub_spectra,dtype=bool)
	# print(spectrum_mask)
	# print(sigmaLambda)

	line_shifts = np.zeros([contsub_spectra.shape[1]])

	for ii in range(contsub_spectra.shape[1]):
		spectrum_mask[:,ii] = np.greater(np.abs(contsub_spectra[:,ii]), 3*sigmaLambda)
		spectrum = contsub_spectra[:,ii].copy()
		# plt.plot(linLambda,spectrum)
		# plt.plot(linLambda,sigmaLambda)
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



	cube_file = parameters['datacube']
	cube_hdu = fits.open(cube_file)
	cube_header = cube_hdu[1].header
	cube_hdu.close()


	lines = {'Hbeta':{	'lambda':[4861.333]			},
			'OIII4958':{	'lambda':[4958.911]	}, 
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


	# lines = {"Halpha":{"lambda":[6562.618]}}

	subcube_width = 1000 / c
	for ll, line in enumerate(lines):

		lineLambda = lines[line]['lambda'][0]

		subcube_range =  np.where((logLambda > np.log(lineLambda*(1 - subcube_width)))  & 
								(logLambda < np.log(lineLambda*(1 + subcube_width))))[0]

		logLambda_subcube = logLambda[subcube_range]

		subcube_spectra = contsub_spectra[subcube_range,:]

		sigmaLambda_subcube = sigmaLambda[subcube_range]
		sigmaLambda_spectra = np.full(subcube_spectra.T.shape,sigmaLambda_subcube).T
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
				elif line == "NII6548":	
					line_min_logLambda = np.min(logLambda_subcube)
					line_max_logLambda = np.log(0.5*(lineLambda + 6562.819)) + line_shift
					logLambda_window = np.logical_and(logLambda_subcube > line_min_logLambda,
									logLambda_subcube  < line_max_logLambda)
					subcube_spectra[~logLambda_window,ii] = 0
				elif line == "NII6583":	
					line_min_logLambda = np.log(0.5*(lineLambda + 6562.819)) + line_shift
					line_max_logLambda = np.max(logLambda_subcube)
					logLambda_window = np.logical_and(logLambda_subcube > line_min_logLambda,
									logLambda_subcube  < line_max_logLambda)
					subcube_spectra[~logLambda_window,ii] = 0

				elif line == "SII6716":	
					line_min_logLambda = np.min(logLambda_subcube)
					line_max_logLambda = np.log(0.5*(lineLambda + 6730.810)) + line_shift
					logLambda_window = np.logical_and(logLambda_subcube > line_min_logLambda,
									logLambda_subcube  < line_max_logLambda)
					subcube_spectra[~logLambda_window,ii] = 0
				elif line == "SII6730":	
					line_min_logLambda = np.log(0.5*(lineLambda + 6716.440)) + line_shift
					line_max_logLambda = np.max(logLambda_subcube)
					logLambda_window = np.logical_and(logLambda_subcube > line_min_logLambda,
									logLambda_subcube  < line_max_logLambda)
					subcube_spectra[~logLambda_window,ii] = 0
				
				
				subcube_spectra_SN = subcube_spectra[:,ii] / sigmaLambda_subcube
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

				subcube_spectra[:,ii]*= mask_final
				sigmaLambda_spectra[:,ii] = sigmaLambda_subcube*mask_final

				# subcube_spectra[:,ii][subcube_spectra[:,ii] < 3.*sigmaLambda_subcube] = 0
				# plt.plot(np.exp(logLambda_subcube),subcube_spectra[:,ii])
				# plt.plot(np.exp(logLambda_subcube),mask1)
				# plt.plot(np.exp(logLambda_subcube),mask2,ls=':')
				# plt.show()
			else:
				subcube_spectra[:,ii] = 0
				sigmaLambda_spectra[:,ii] = 0


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
								array = sigmaLambda_spectra.T,
								name='sigma',format=str(len(sigmaLambda_spectra))+'D'
								)]))

		hdul = fits.HDUList([primary_hdu,
								spectra_hdu,sigma_hdu])
		print(f"Saving {line} spectra table")
		hdul.writeto(f"{parameters['output']}/{parameters['output_dir']}/linecubes/{line}_subspectra.fits",overwrite=True)
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

		primary_hdu = fits.PrimaryHDU(header = fits.Header())
		line_cube_hdu = fits.ImageHDU(subcube_cube,
								name='H-alpha',
								header = cube_header)
		hdul = fits.HDUList([primary_hdu,
									line_cube_hdu])
		print(f"Saving {line} linecube")
		hdul.writeto(f"{parameters['output']}/{parameters['output_dir']}/linecubes/{line}_linecube.fits",overwrite=True)
		print("Saved")



def measure_line_fluxes(parameters):

	spax_properties = Table.read(f"{parameters['output']}/{parameters['output_dir']}/spaxel_properties.fits")

	sigmas =  np.loadtxt(f"{parameters['output']}/{parameters['output_dir']}/wavelength_sigma.txt")
	sigmaLambda =  sigmas[:,1]

	subspectra = glob.glob(f"{parameters['output']}/{parameters['output_dir']}/linecubes/*_subspectra.fits")

	RV = 4
	k_l = lambda ll: extinction_curve(ll,RV=RV,extcurve='Calzetti00')

	Av_flag = False
	if any(['Halpha' in file for file in subspectra]) & any(['Hbeta' in file for file in subspectra]):
		AV_flag = True
	
		Halpha_filename = f"{parameters['output']}/{parameters['output_dir']}/linecubes/Halpha_subspectra.fits"
		Hbeta_filename = f"{parameters['output']}/{parameters['output_dir']}/linecubes/Hbeta_subspectra.fits"

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

	# print(reddening)

	F_measures = np.hstack((reddening,F_measures))
	names = ['EBV_l',"A_V"] + names


	F_measures_expanded = np.zeros(spax_properties.size[0],F_measures.size[1])

	for bb in range(F_measures.size[0]):
		# fluxes_row = np.array([fluxes[ii][bb] for ii in range(len(fluxes))])
		inbin = np.where(spax_properties['vorbin_num'] == bb)[0]
		spax_num_inbin = spax_properties['spax_num'][inbin]
		
		F_measures_expanded[spax_num_inbin,:] = F_measures[bb,:]


	fluxes_table = Table(F_measures_expanded,names=names)
	print(fluxes_table)
	# exit()


	fluxes_table.write(f"{parameters['output']}/{parameters['output_dir']}/spax_emline_fluxes.fits",overwrite=True)



def measure_linecube_fluxes(filename,EBV = 0,k_l = None):

	if isinstance(k_l,type(None)):
		k_l = lambda ll: extinction_curve(ll,RV=4,extcurve='Calzetti00')
		# A_l = lambda ll: EBV * k_l(ll)

	hdul = fits.open(filename)
	logLambda, spectra_list = read_spectra_hdul(hdul)
	spectra = spectra_list[0]
	spec_sigma = spectra_list[1]
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
		sigma = spec_sigma[:,ii]

		# print(spectrum)

		# plt.plot(spectrum)
		# plt.show()
		# exit()
		if isinstance(EBV,int):
			E_BV = EBV
		else:
			E_BV = EBV[ii]


		Fint[ii] = np.nansum(spectrum*diff_lambda)
		Ferr[ii] = np.nansum(sigma*diff_lambda)

		spectrum_extcorr = spectrum * np.power(10,0.4*E_BV*k_l(linLambda))
		sigma_extcorr = sigma * np.power(10,0.4*E_BV*k_l(linLambda))

		Fint_extcorr[ii] = np.nansum(spectrum_extcorr*diff_lambda)
		Ferr_extcorr[ii] = np.nansum(sigma_extcorr*diff_lambda)
	
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


def check_line_ratios(parameters):

	emline_fluxes = Table.read(f"{parameters['output']}/{parameters['output_dir']}/spax_emline_fluxes.fits")

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

	ax4 = fig.add_subplot(gs[3,0],sharey=ax1)
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

	ax1.set_ylim([-0.8,8])
	ax1.set_xlim([1,6])
	ax2.set_ylim([-0.8,8])
	ax2.set_xlim([1,6])
	ax3.set_ylim([-0.8,8])
	ax3.set_xlim([1,6])
	ax4.set_ylim([-0.8,8])
	ax4.set_xlim([1,6])

	legs = [Line2D([0],[0],color='White',marker='o',markerfacecolor='Black'),
			Line2D([0],[0],color='White',marker='o',markerfacecolor='DodgerBlue'),
			Line2D([0],[0],color='White',marker='o',markerfacecolor='Red')]
	ax1.legend(legs,["All","SN>3","SN>5"],fontsize=15)

	fig.savefig(f"{parameters['output']}/{parameters['output_dir']}/figures/EMline_ratios.png")

	# plt.show()


def check_line_SNmaps(parameters):

	spax_properties = Table.read(f"{parameters['output']}/{parameters['output_dir']}/spaxel_properties.fits")

	emline_fluxes = Table.read(f"{parameters['output']}/{parameters['output_dir']}/spax_emline_fluxes.fits")

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

		fig.savefig(f"{parameters['output']}/{parameters['output_dir']}/figures/{line}_linemap.png")

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

			fig.savefig(f"{parameters['output']}/{parameters['output_dir']}/figures/{line}_SNmap.png")
			


		# plt.show()


def line_ratio_maps(parameters):

	spax_properties = Table.read(f"{parameters['output']}/{parameters['output_dir']}/spaxel_properties.fits")

	emline_fluxes = Table.read(f"{parameters['output']}/{parameters['output_dir']}/spax_emline_fluxes.fits")

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

		fig.savefig(f"{parameters['output']}/{parameters['output_dir']}/figures/{pair[0]}_{pair[1]}_ratio_map.png")
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
			fig.savefig(f"{parameters['output']}/{parameters['output_dir']}/figures/AV_map.png")


def metallicity_Curti17(parameters,method='O3N2'):	
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


	spax_properties = Table.read(f"{parameters['output']}/{parameters['output_dir']}/spaxel_properties.fits")

	emline_fluxes = Table.read(f"{parameters['output']}/{parameters['output_dir']}/spax_emline_fluxes.fits")

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



	plt.hist(log_ratio,bins=100)
	plt.show()
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

	fig.savefig(f"{parameters['output']}/{parameters['output_dir']}/figures/logOH_map_{method}.png")
	# plt.show()


def metallicity_Dopita16(parameters):

	spax_properties = Table.read(f"{parameters['output']}/{parameters['output_dir']}/spaxel_properties.fits")

	emline_fluxes = Table.read(f"{parameters['output']}/{parameters['output_dir']}/spax_emline_fluxes.fits")

	spax_properties = hstack((spax_properties,emline_fluxes))


	SNgood = np.where((spax_properties['OIII5006']/spax_properties['OIII5006_err'] >=4) & 
					(spax_properties['NII6583']/spax_properties['NII6583_err'] >=4) & 
					(spax_properties['Halpha']/spax_properties['Halpha_err'] >=4) &
					(spax_properties['SII6716']/spax_properties['SII6716_err'] >=4))[0]




	y = (np.log10(spax_properties['NII6583_extcorr']/spax_properties['SII6716_extcorr']) + \
		0.264 * np.log10(spax_properties['NII6583_extcorr'] / spax_properties['Halpha_extcorr']))[SNgood]

	plt.hist(y,bins=100)
	plt.show()

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

	fig.savefig(f"{parameters['output']}/{parameters['output_dir']}/figures/logOH_map_Dopita16.png")
	# plt.show()




def make_BPT_diagram(parameters):
	spax_properties = Table.read(f"{parameters['output']}/{parameters['output_dir']}/spaxel_properties.fits")

	emline_fluxes = Table.read(f"{parameters['output']}/{parameters['output_dir']}/spax_emline_fluxes.fits")

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

	fig.savefig(f"{parameters['output']}/{parameters['output_dir']}/figures/BPTs.png")
	# plt.show()






#useful things


def MUSE_LSF_Bacon17(ll, z = 0):
	FWHM = 5.866e-8 * ll*ll - 9.187e-4 * ll + 6.040
	FWHM = FWHM / (1.e0 + z)

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





if __name__ == '__main__':
	main()
	# read_EMILES_spectra()
	# read_parameterfile()
	# read_gaslines_parameterfile()

	# make_spectra_tables()


	# voronoi_bin_cube(SN_indiv = 5, SN_vorbin = 40)

	# create_binned_spectra()