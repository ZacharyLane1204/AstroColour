import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astropy.convolution import convolve
# from astropy.convolution import convolve_fft

from astroscrappy import detect_cosmics

from scipy.fft import fft2, ifft2
from scipy.ndimage import fourier_shift

from photutils.psf.matching import create_matching_kernel, TukeyWindow

from skimage.registration import phase_cross_correlation
from skimage import morphology, measure
from skimage.filters import sobel
from skimage.morphology import binary_dilation

# from AstroColour.image_rotation import unrotate_image
from AstroColour.psf_analysis import PSF_Analysis
# from AstroColour.spatial_psf_matcher import SpatialPSFMatcher
from AstroColour.pdastro import pdastrostatsclass
from AstroColour.hidden_prints import hidden_prints
# from AstroColour.new_spatial_matcher import spatially_matched_difference_pipeline, iterative_fourier_shift

import os
from copy import deepcopy
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

class Image_Analysis():
    def __init__(self, images, epsf_plot = False, bkg_plot = False, temp_save = False, epsf = True):
        
        self.epsf_plot = epsf_plot
        self.bkg_plot = bkg_plot
        self.temp_save = temp_save
        self.epsf = epsf
        
        self.process_images(images)

    def process_images(self, images):
        """
        Process the images.
        
        Parameters
        ----------
        images : List
            List of Numpy Arrays of data images.
        colours : List
            List of Tuples or Strings of colour choice.
        intensities : List
            List of Floats between 0 and 1 for image intensities.
        uppers : List
            List of Floats between 0 and 100 for upper percentile.
        lowers : List
            List of Floats between 0 and 100 for lower percentile
        """
        
        calib_images = []
        diff_images = []
        
        # try:
        if self.epsf:
            fwhms = []
            epsfs = []
            positions = []
            for image in images:
                
                psf_analysis = PSF_Analysis(image, epsf_plot=self.epsf_plot)
                
                fwhms.append(psf_analysis.fwhm_estimate)
                epsfs.append(psf_analysis.epsf_data)
                positions.append(psf_analysis.positions)
            
            idx = np.nanargmax(fwhms)
            
            self.epsf = True
        
        # crmask, cleaned_image_master = self.cosmic_ray_removal(images[idx])
        # cleaned_image_master = self.backgrounding(cleaned_image_master, plot=self.bkg_plot)
        
        
        with hidden_prints():
            aligned_images = []
            for i, image in enumerate(images):                
                
                crmask, cleaned_image = self.cosmic_ray_removal(image)
                sat_mask = self.remove_satellite_trails(cleaned_image, sigma=3)
                
                bkg_image = self.backgrounding(cleaned_image, plot=self.bkg_plot)
                
                taper = TukeyWindow(alpha=0.5)
                kernel = create_matching_kernel(epsfs[i], epsfs[idx], window = taper)
                corrected_image = self.match_psf(bkg_image, kernel)
                
                # diff_images.append(cleaned_image_master - corrected_image)
                # diff_images.append(diffs)
                calib_images.append(corrected_image)
            self.diff_images = [calib_images[i] - calib_images[idx] for i in range(len(calib_images))]
            self.calib_images = calib_images

            if self.temp_save:
                np.save('calib_images.npy', calib_images)
                
    def cosmic_ray_removal(self, image_data):
        crmask, cleaned = detect_cosmics(image_data, 
                                  sigclip=4.5,       # sigma threshold for detection
                                  sigfrac=0.3,        # neighboring pixels fraction
                                  objlim=5.0,         # contrast between CR and object
                                  gain=1.0,           # if not gain-corrected, set appropriately
                                  readnoise=5.0,      # your camera's read noise (in e-)
                                  satlevel=50000.0,   # saturation limit (to avoid false positives)
                                  niter=4,            # number of iterations
                                  cleantype='medmask' # how to interpolate over CRs
                                  )
        
        return crmask, cleaned
    
    def remove_satellite_trails(self, image, sigma=3):
        
        edges = sobel(image)
        mask = edges > sigma * np.nanstd(edges)
        mask = binary_dilation(mask)
        # clean = self.inpaint_masked_pixels(image, mask.astype(np.uint8))
        return mask
    
    def backgrounding(self, data, plot = None):
        
        pdac = pdastrostatsclass()
        
        cut_data = data.copy()
        
        thresh_hi = np.nanpercentile(cut_data, 70)
        thresh_low = np.nanpercentile(cut_data, 5)
        
        cut_data = cut_data[(cut_data < thresh_hi)]
        cut_data = cut_data[(cut_data > thresh_low)]
        
        
        cut_data_med = np.nanpercentile(cut_data, 40)
        
        df = pd.DataFrame()
        
        pdac.t = df
        pdac.t['dm']=  cut_data - cut_data_med
        
        pdac.calcaverage_sigmacutloop('dm', verbose=0, percentile_cut_firstiteration=50)
        
        bkg = pdac.statparams['mean'] + cut_data_med
        
        return data - bkg
    
    def phase_corr_shift(self, img_ref, img_target, mask=None, upsample=100):
        """
        Compute subpixel shift between img_target -> img_ref
        """
        shift, error = phase_cross_correlation(img_ref, img_target, 
                                                          upsample_factor=upsample, 
                                                          reference_mask=mask)
        aligned = np.real(ifft2(fourier_shift(fft2(img_target), shift)))
        return aligned, shift
    
    def match_psf(self, image, psf):
        """
        Match the PSF of two images.
        
        Parameters
        ----------
        image : 2d array
            Image to be processed.
        psf : 2d array
            PSF kernel of the largest image.
            
        Returns
        -------
        corrected_image : 2d array
            Image with matched PSF.
        """

        corrected_image = convolve(image, psf)
        
        return corrected_image