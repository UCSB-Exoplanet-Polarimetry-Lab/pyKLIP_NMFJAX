#KLIP Forward Modelling

def klip_math(sci, refs, models, numbasis, return_basis=False): #Zack
    """
    linear algebra of KLIP with linear perturbation 
    disks and point sources
    
    Args:
        sci: array of length p containing the science data
        refs: N x p array of the N reference PSFs that 
                  characterizes the extended source with p pixels
        numbasis: number of KLIP basis vectors to use (can be an int or an array of ints of length b)  
        return_basis: If true, return KL basis vectors (used when onesegment==True)

    Returns:
        sub_img_rows_selected: array of shape (p,b) that is the PSF subtracted data for each of the b KLIP basis
                               cutoffs. If numbasis was an int, then sub_img_row_selected is just an array of length p

    """
    sci_mean_sub = sci - np.nanmean(sci)
    sci_nanpix = np.where(np.isnan(sci_mean_sub))
    sci_mean_sub[sci_nanpix] = 0
        
    refs_mean_sub = refs - np.nanmean(refs, axis=1)[:, None]
    refs_mean_sub[np.where(np.isnan(refs_mean_sub))] = 0

    models_mean_sub = models - np.nanmean(models, axis=1)[:,None]
    models_mean_sub[np.where(np.isnan(models_mean_sub))] = 0

    #if covar_psfs is None, disregarded scaling term:
    covar_psfs = np.dot(refs_mean_sub,refs_mean_sub.T)  
          
    tot_basis = covar_psfs.shape[0]
    numbasis = np.clip(numbasis - 1, 0, tot_basis-1)
    max_basis = np.max(numbasis) + 1
        
    evals, evecs = la.eigh(covar_psfs, eigvals = (tot_basis-max_basis, tot_basis-1))
    evals = np.copy(evals[::-1])
    evecs = np.copy(evecs[:,::-1])
    
    KL_basis = np.dot(refs_mean_sub.T,evecs)
    KL_basis = KL_basis * (1. / np.sqrt(evals))[None,:]
    
    sci_mean_sub_rows = np.tile(sci_mean_sub, (max_basis,1))
    sci_rows_selected = np.tile(sci_mean_sub, (np.size(numbasis),1))
    
    sci_nanpix = np.where(np.isnan(sci_mean_sub_rows))
    sci_mean_sub_rows[sci_nanpix] = 0
    sci_nanpix = np.where(np.isnan(sci_rows_selected))
    sci_rows_selected[sci_nanpix] = 0
    
    CAdeltaI = np.dot(refs_mean_sub,models_mean_sub.T) + np.dot(models_mean_sub,refs_mean_sub.T)
    
    Nrefs = evals.shape[0]
    KL_perturb = np.zeros(KL_basis.shape)
    Pert1 = np.zeros(KL_basis.shape)
    Pert2 = np.zeros(KL_basis.shape)
    CoreffsKL = np.zeros((Nrefs,Nrefs))
    Mult = np.zeros((Nrefs,Nrefs))
    cross = np.zeros((Nrefs,Nrefs))
    
    for i in range(Nrefs):
        for j in range(Nrefs):
            cross[i,j] = np.transpose(evecs[:,j]).dot(CAdeltaI).dot(evecs[:,i])
        
    for i in range(Nrefs):
        for j in range(Nrefs):
            if j == i:
                Mult[i,j] = -1./(2.*evals[j])
            else:
                Mult[i,j] = np.sqrt(evals[j]/evals[i])/(evals[i]-evals[j])
        CoreffsKL[i,:] = cross[i]*Mult[i]

     
    for j in range(Nrefs):
        Pert1[:,j] = np.dot(CoreffsKL[j],KL_basis.T).T
        Pert2[:,j] = (1./np.sqrt(evals[j])*np.transpose(evecs[:,j]).dot(models_mean_sub))
        KL_perturb[:,j] = (Pert1[:,j]+Pert2[:,j])
    
    KL_pert = KL_perturb + KL_basis
    
    inner_products = np.dot(sci_mean_sub_rows, KL_pert)
    lower_tri = np.tril(np.ones([max_basis,max_basis]))
    inner_products = inner_products * lower_tri
    klip = np.dot(inner_products[numbasis,:], KL_pert.T)
    
    sub_img_rows_selected = sci_rows_selected - klip
    sub_img_rows_selected[sci_nanpix] = np.nan
    
    if return_basis is True:
        return sub_img_rows_selected.transpose(), KL_basis.transpose()
    else:
        return sub_img_rows_selected.transpose()
		
def klip_adi(imgs, models, centers, parangs, IWA, annuli=5, subsections=4, movement=3, numbasis=None, aligned_center=None, minrot=0): #Zack
    """
    KLIP PSF Subtraction using angular differential imaging, perturbed

    Args:
        imgs: array of 2D images for ADI. Shape of array (N,y,x)
        models: array of 2D models for ADI corresponding to imgs. Shape of array (N,y,x)
        centers: N by 2 array of (x,y) coordinates of image centers
        parangs: N legnth array detailing parallactic angle of each image
        IWA: inner working angle (in pixels)
        anuuli: number of annuli to use for KLIP
        subsections: number of sections to break each annuli into
        movement: minimum amount of movement (in pixels) of an astrophysical source
                  to consider using that image for a refernece PSF
        numbasis: number of KL basis vectors to use (can be a scalar or list like). Length of b
        aligned_center: array of 2 elements [x,y] that all the KLIP subtracted images will be centered on for image
                        registration
        minrot: minimum PA rotation (in degrees) to be considered for use as a reference PSF (good for disks)

    Returns:
        sub_imgs: array of [array of 2D images (PSF subtracted)] using different number of KL basis vectors as
                    specified by numbasis. Shape of (b,N,y,x). Exception is if b==1. Then sub_imgs has the first
                    array stripped away and is shape of (N,y,x).
    """
    
    if numbasis is None:
        totalimgs = imgs.shape[0]
        numbasis = np.arange(1,totalimgs + 5,5)
        print numbasis
    else:
        if hasattr(numbasis,"__len__"):
            numbasis = np.array(numbasis)
        else:
            numbasis = np.array([numbasis])
        
    if aligned_center is None:
        aligned_center = [int(imgs.shape[2]//2),int(imgs.shape[1]//2)]
        
    allnans = np.where(np.isnan(imgs))
    
    #annuli
    dims = imgs.shape
    x,y = np.meshgrid(np.arange(dims[2] * 1.0),np.arange(dims[1]*1.0))
    nanpix = np.where(np.isnan(imgs[0]))
    OWA = np.sqrt(np.min((x[nanpix] - centers[0][0]) ** 2 + (y[nanpix] - centers[0][1]) ** 2))
    dr = float(OWA - IWA) / (annuli) 
    
    rad_bounds = [(dr * rad + IWA, dr * (rad + 1) + IWA) for rad in range(annuli)]
    rad_bounds[annuli - 1] = (rad_bounds[annuli - 1][0], imgs[0].shape[0] / 2)

    dphi = 2 * np.pi / subsections
    phi_bounds = [[dphi * phi_i - np.pi, dphi * (phi_i + 1) - np.pi] for phi_i in range(subsections)]
    phi_bounds[-1][1] = 2. * np.pi
    
    sub_imgs = np.zeros([dims[0], dims[1] * dims[2], numbasis.shape[0]])

    #begin KLIP process for each image
    for img_num, pa in enumerate(parangs):
        recenteredimgs = np.array([align_and_scale(frame, aligned_center, oldcenter) for frame, oldcenter in zip(imgs, centers)])
        recenteredmodels = np.array([align_and_scale(frame, aligned_center, oldcenter) for frame, oldcenter in zip(models, centers)])
        
        #create coordinate system 
        r = np.sqrt((x - aligned_center[0]) ** 2 + (y - aligned_center[1]) ** 2)
        phi = np.arctan2(y - aligned_center[1], x - aligned_center[0])

        #flatten img dimension
        flattenedimgs = recenteredimgs.reshape((dims[0], dims[1] * dims[2]))
        flattenedmodels = recenteredmodels.reshape((dims[0],dims[1] * dims[2]))
        
        r.shape = (dims[1] * dims[2])
        phi.shape = (dims[1] * dims[2])
        #iterate over the different sections
        for radstart, radend in rad_bounds:
            for phistart, phiend in phi_bounds:
                #grab the pixel location of the section we are going to anaylze
                section_ind = np.where((r >= radstart) & (r < radend) & (phi >= phistart) & (phi < phiend))
                if np.size(section_ind) == 0:
                    continue
                #grab the files suitable for reference PSF
                avg_rad = (radstart + radend) / 2.0
                moves = estimate_movement(avg_rad, parang0=pa, parangs=parangs)
                file_ind = np.where((moves >= movement) & (np.abs(parangs - pa) > minrot))

                if np.size(file_ind) < 2:
                    print("less than 2 reference PSFs available, skipping...")
                    print((sub_imgs[img_num, section_ind]).shape)
                    print(np.zeros(np.size(section_ind)).shape)
#                    sub_imgs[img_num, section_ind] = np.zeros(np.size(section_ind))
                    continue
                images_math = flattenedimgs[file_ind[0], :]
                images_math = images_math[:, section_ind[0]]
                models_math = flattenedmodels[file_ind[0], :]
                models_math = models_math[:, section_ind[0]]

                sub_imgs[img_num, section_ind, :] = klip_math(flattenedimgs[img_num, section_ind][0], images_math, models_math, numbasis)
    
    #move number of KLIP modes as leading axis (i.e. move from shape (N,y,x,b) to (b,N,y,x)
    sub_imgs = np.rollaxis(sub_imgs.reshape((dims[0], dims[1], dims[2], numbasis.shape[0])), 3)
    #if we only passed in one value for numbasis (i.e. only want one PSF subtraction), strip off the number of basis)
    if sub_imgs.shape[0] == 1:
        sub_imgs = sub_imgs[0]
    #restore bad pixels
    sub_imgs[:,allnans[0], allnans[1], allnans[2]] = np.nan

    #derotate images
    #imgs_list = []
    #for a in sub_imgs:
    #    imgs_list.append(np.array([rotate(img, pa, (140,140), center) for img,pa,center in zip(sub_imgs, parangs, centers)]))
    #subimgs = np.asarray(imgs_list)
    #all of the image centers are now at aligned_center
    centers[:,0] = aligned_center[0]
    centers[:,1] = aligned_center[1]
    
    return sub_imgs
    