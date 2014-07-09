import numpy as np

def klip_math(sci, ref_psfs, numbasis):
	"""
	Helper function for KLIP that does the linear algebra
	
	Inputs:
		sci : array of length p containing the science data
		ref_psfs : N x p array of the N reference PSFs that 
							 characterizes the PSF of the p pixels
		numbasis : number of KLIP basis vectors to use

	Outputs:
		sub_img : array of lenght p that is the PSF subtracted data
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
	
