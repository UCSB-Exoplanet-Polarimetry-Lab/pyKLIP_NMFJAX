import numpy as np

def klip_math(sci, ref_psfs, numbasis):
	"""
	Helper function for KLIP that does the linear algebra
	
	Inputs:
		sci: array of length p containing the science data
		ref_psfs: N x p array of the N reference PSFs that 
							 characterizes the PSF of the p pixels
		numbasis: number of KLIP basis vectors to use

	Outputs:
		sub_img: array of lenght p that is the PSF subtracted data

	TODO:
		make numbasis to be any number of KLIP cutoffs and return all of them
	"""
	import numpy.linalg as la
	#import pdb	

	#for the science image, subtract the mean and mask bad pixels
	sci_mean_sub = sci - np.nanmean(sci)
	sci_nanpix = np.where(np.isnan(sci_mean_sub))
	sci_mean_sub[sci_nanpix] = 0

	#do the same for the reference PSFs
	#playing some tricks to vectorize the subtraction
	ref_psfs_mean_sub = ref_psfs - np.nanmean(ref_psfs, axis=1)[:,None]
	ref_psfs_mean_sub[np.where(np.isnan(ref_psfs_mean_sub))] = 0

	#calculate the covariance matrix for the reference PSFs
	#note that numpy.cov normalizes by p-1 to get the NxN covariance matrix
	#we have to correct for that a few lines down when consturcting the KL 
	#vectors since that's not part of the equation in the KLIP paper
	covar_psfs = np.cov(ref_psfs_mean_sub)

	#calculate eigenvalues and eigenvectors of covariance matrix
	evals, evecs = la.eigh(covar_psfs) #function for symmetric matrices
	
	#sort the eigenvalues and eigenvectors (unfortunately smallest first)
	eig_args_all = np.argsort(evals)
	
	#calculate the KL basis vectors
	Z = np.dot(evecs, ref_psfs_mean_sub)
	Z = Z * (1./np.sqrt(evals*(np.size(sci)-1)))[:,None] #multiply a value from each row
	
	#pick the largest however many to model PSF
	tot_basis = np.size(evals)
	#truncation either based on user input or maximum number of PSFs
	trunc_basis = np.min([numbasis, tot_basis])
	#remember that sorting sorted the smallest eigenvalues first
	eig_args = eig_args_all[tot_basis - trunc_basis : tot_basis]

	Z = Z[eig_args, :]

	#project KL vectors onto science image to construct model PSF
	inner_products = np.dot(sci_mean_sub, Z.T)
	klip_psf = np.dot(inner_products, Z)
	
	#subtract from original image to get final image
	sub_img = sci_mean_sub - klip_psf

	#restore NANs
	sub_img[sci_nanpix] = np.nan

	#pdb.set_trace()

	return sub_img

def align_and_scale(img, new_center, old_center=None, scale_factor=1):
	"""
	Helper function that realigns and/or scales the image

	Inputs:
		img: 2D image to perform manipulation on
		new_center: 2 element tuple (xpos, ypos) of new image center
		old_center: 2 element tuple (xpos, ypos) of old image center
		scale_factor: how much the stretch/contract the image. Will we
									scaled w.r.t the new_center (done after relaignment).
									We will adopt the convention
											>1: stretch image (shorter to longer wavelengths)
											<1: contract the image (longer to shorter wvs)

	Outputs:
		resampled_img: shifted and/or scaled 2D image
	"""	
	import scipy.ndimage as ndimage
	import scipy.interpolate as interp
	import pdb

	#create the coordinate system of the image to manipulate for the transform
	dims = img.shape
	x,y = np.meshgrid(np.arange(dims[0], dtype=np.float32), np.arange(dims[1], dtype=np.float32))

	#if old_center is specified, realign the images
	if old_center is not None:
		dx = new_center[0] - old_center[0]
		dy = new_center[1] - old_center[1]
		x -= dx
		y -= dy

	#if scale_factor is specified, scale the images
	if scale_factor != 1:
		#conver to polar for scaling
		r = np.sqrt((x - new_center[0])**2 + (y - new_center[1])**2) 
		theta = np.arctan2(y-new_center[1],x-new_center[0]) #theta range is [-pi,pi]
		
		r /= scale_factor
	
		#convert back to cartesian
		x = r*np.cos(theta) + new_center[0]
		y = r*np.sin(theta) + new_center[1]

	#resample image based on new coordinates
	#scipy uses y,x convention when meshgrid uses x,y
	#stupid scipy functions can't work with masked arrays (NANs)
	#and trying to use interp2d with sparse arrays is way to slow
	#hack my way out of this by picking a really small value for NANs and try to detect them again after the interpolation
	minval = np.min([np.nanmin(img), 0.0])
	nanpix = np.where(np.isnan(img))
	img = np.copy(img)
	img[nanpix] = minval*2.0
	resampled_img = ndimage.map_coordinates(img, [y,x], cval=np.nan)
	resampled_img[np.where(resampled_img <= minval)] = np.nan
	resampled_img[nanpix] = np.nan

	#broken attempt at using sparse arrays with interp2d. Warning: takes forever to run
	#good_dat = np.where(~(np.isnan(img)))
	##recreate old coordinate system
	#x0,y0 = np.meshgrid(np.arange(dims[0], dtype=np.float32), np.arange(dims[1], dtype=np.float32))
	#interpolated = interp.interp2d(x0[good_dat], y0[good_dat], img[good_dat], kind='cubic')
	#resampled_img = np.ones(img.shape) + np.nan
	#resampled_img[good] = interpolated(y[good],x[good])

	return resampled_img
		
		
