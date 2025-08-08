import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

from astropy.io import fits
from astropy.visualization import simple_norm
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.wcs import WCS
from astropy.nddata import NDData
from astropy.table import Table
from astropy.convolution import convolve

from scipy.signal import fftconvolve, convolve2d

from photutils.detection import find_peaks
from photutils.psf import extract_stars
from photutils.psf import EPSFBuilder
from photutils.detection import DAOStarFinder
from photutils.psf import PSFPhotometry, IterativePSFPhotometry
from photutils.background import LocalBackground, MMMBackground
from photutils.psf import SourceGrouper

from sklearn.cluster import DBSCAN

import os
from copy import deepcopy
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

plt.rc('font', family='serif')

fig_width_pt = 244.0  # Get this from LaTeX using \the\columnwidth
text_width_pt = 508.0 # Get this from LaTeX using \the\textwidth

inches_per_pt = 1.0/72.27               # Convert pt to inches
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt*1.5 # width in inches
fig_width_full = text_width_pt*inches_per_pt*1.5  # 17
fig_height =fig_width*golden_mean # height in inches
fig_size = [fig_width,fig_height] #(9,5.5) #(9, 4.5)
fig_height_full = fig_width_full*golden_mean

class RGB():
    def __init__(self, images, 
                 colours = ['red', 'green', 'blue'], 
                 intensities = [1, 1, 1],
                 uppers = [98, 98, 98], lowers = [2, 2, 2],
                 save = False, save_name = '', save_folder = '',
                 figure_size = None, manual_override = None, dpi = 900, 
                 norm = 'linear', gamma = 1, epsf = True, min_separation = 34, 
                 star_size = 15, epsf_plot = False, cross_corr = True):
        '''
        Create a RGB image from three images.

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
        save : Boolean
            Whether to save the image.
        save_name : String
            Detail to add in the saved filename.
        save_folder : String
            Folder to save the image.
        figure_size : Float
            Dimension of the image.
        manual_override : Boolean
            Whether to manually override the limits.
        dpi : Integer
            DPI of the saved image.
        norm : String
            Normalisation of the image. An option of 'linear', 'sqrt', 'log', 'asinh' or 'sinh'.
        gamma : Float
            Gamma correction of the image. Power to raise the image to.
        epsf : Boolean
            Whether to use the EPSF method.
        min_separation : Float
            Minimum separation between stars.
        star_size : Float
            Size of the stars for the EPSF method.
        epsf_plot : Boolean
            Whether to plot the EPSF kernel.
        cross_corr : Boolean
            Whether to cross correlate the images.
        '''
        length = len(images)
        
        if len(colours) != length:
            raise ValueError("Length of colours does not match the number of images.")
        if len(intensities) != length:
            raise ValueError("Length of intensities does not match the number of images.")
        if len(uppers) != length:
            raise ValueError("Length of uppers does not match the number of images.")
        if len(lowers) != length:
            raise ValueError("Length of lowers does not match the number of images.")

        self.save = save
        self.save_name = save_name
        self.save_folder = save_folder
        self.dpi = dpi
        self.min_separation = min_separation
        self.epsf = epsf
        self.star_size = star_size
        self.epsf_plot = epsf_plot
        self.cross_corr = cross_corr

        self.figure_size = figure_size
        if self.figure_size == None:
            self.figure_size = fig_width_full

        if manual_override == None:
            manual_override = 100
        self.manual_override = manual_override
        
        if isinstance(norm, str):
            if ['linear', 'sqrt', 'log', 'asinh', 'sinh'].__contains__(norm):
                norms = [norm]*len(images)
            else:
                raise ValueError("Not a valid norm. Try 'linear', 'sqrt', 'log', 'asinh' or 'sinh'.")
            
        elif isinstance(norm, list) & (len(norm) == length):
            norms = []
            for i in range(len(norm)):
                if not ['linear', 'sqrt', 'log', 'asinh', 'sinh'].__contains__(norm[i]):
                    raise ValueError("Not a valid norm. Try 'linear', 'sqrt', 'log', 'asinh' or 'sinh'.")
                else:
                    norms.append(norm[i])
        
        self.gamma = gamma
        
        self.process_images(images, colours, intensities, uppers, lowers, norms)

    def process_images(self, images, colours, intensities, uppers, lowers, norms):
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
        calib_colours = []
        calib_intensities = []
        calib_uppers = []
        calib_lowers = []
        files = []
        
        try:
            if self.epsf:
                epsf_kernels = []
                over_sampled_epsf = []
                
                for image in images:
                    epsf_kernel, over_sampled = self.create_epsf_kernel(image)
                    epsf_kernels.append(epsf_kernel)
                    over_sampled_epsf.append(over_sampled)
                fwhms = [self.calculate_fwhm(kernel) for kernel in over_sampled_epsf]  # Calculate FWHM for each EPSF
                largest_fwhm_kernel = epsf_kernels[np.nanargmax(fwhms)]  # Identify the EPSF with the largest FWHM
                
                idx = np.nanargmax(fwhms)
                self.epsf = True
            else:
                self.epsf = False
                idx = 0
        except:
            self.epsf = False
            idx = 0
            print('EPSF failed to create. Skipping...')
            
        for i in tqdm(range(len(images)), desc = 'Processing Images'):
            
            if i == 0:
                if self.epsf & (idx != 0):
                    image = self.match_psf(images[i], largest_fwhm_kernel)
                else:
                    image = images[i]
                calib_images.append(image)
                
            else:
                if self.epsf:
                    image = self.subsequent_images(images[i], largest_fwhm_kernel, calib_images[0])
                else:
                    image = images[i]
                calib_images.append(image)
            
            p_norm = self.percent_norm(image, lower = lowers[i], upper = uppers[i], norm = norms[i])
            colour = self.colour_check(colours[i])
            calib_colours.append(colour)
            
            files.append(self.colourise(p_norm, colour, intensities[i]))
            
            calib_intensities.append(intensities[i])
            calib_uppers.append(uppers[i])
            calib_lowers.append(lowers[i])
            
        self.images = calib_images
        self.colours = calib_colours
        self.intensities = calib_intensities
        self.uppers = calib_uppers
        self.lowers = calib_lowers
        self.files = files
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            im_composite = np.clip(np.nansum(files, axis=0), 0, 10)
    
        self.im_composite = im_composite
        
    def subsequent_images(self, image, largest_fwhm_kernel, match_image):
        """
        Process subsequent images.
        
        Parameters
        ----------
        image : 2d array
            Image to be processed.
        epsf_kernel : 2d array
            EPSF kernel.
        largest_fwhm_kernel : 2d array
            EPSF kernel with the largest FWHM.
        match_image : 2d array
            Image to match the PSF to.
            
        Returns
        -------
        new_image : 2d array
            Processed image.
        """
        if self.cross_corr:
            new_image = self.register(image, match_image)
            if new_image is None:
                    new_image = image
        else:
            new_image = image
                
        if self.epsf:
            new_image = self.match_psf(new_image, largest_fwhm_kernel)
            
        return new_image

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

    def _star_extraction(self, data):
        '''
        Extract stars from the image.
        
        Parameters
        ----------
        data : 2d array
            FITS Image.
        '''
        
        nddata = NDData(data=data) 
        mean, median, std = sigma_clipped_stats(data, sigma=3.0) 

        fwhm = 5

        daofind = DAOStarFinder(fwhm=fwhm, threshold=10.*std)  
        sources = daofind(data) 

        sources.sort('flux', reverse=True)

        positions = np.zeros((len(sources), 2))
        positions[:,0] = sources['xcentroid'].value
        positions[:,1] = sources['ycentroid'].value

        db = DBSCAN(eps=self.min_separation, min_samples=2).fit(positions) # Apply DBSCAN clustering

        labels = db.labels_ # Get the labels assigned by DBSCAN (-1 means noise)

        filtered_positions_temp = positions[labels == -1] # Filter out the points that are considered noise

        condition = np.logical_and(np.logical_and(filtered_positions_temp[:, 0] >= self.min_separation, 
                                                filtered_positions_temp[:, 0] <= (data.shape[0] - self.min_separation)), 
                                    np.logical_and(filtered_positions_temp[:, 1] >= self.min_separation, 
                                                filtered_positions_temp[:, 1] <= (data.shape[1] - self.min_separation)))

        filtered_positions = filtered_positions_temp[condition]

        stars_tbl = Table()
        stars_tbl['x'] = filtered_positions[:,0]
        stars_tbl['y'] = filtered_positions[:,1]

        stars = extract_stars(nddata, stars_tbl, size=self.star_size)

        return stars

    def create_epsf_kernel(self, data):
        '''
        Create an EPSF kernel based on Anderson and King (2000).

        Returns
        -------
        epsf_kernel : 2d array
            EPSF kernel.
        '''
        
        stars = self._star_extraction(data)
        epsf_builder = EPSFBuilder(oversampling=1, maxiters = 20, progress_bar=False, 
                                   recentering_maxiters = 10, smoothing_kernel='quartic')
        epsf, _ = epsf_builder(stars)
        
        epsf_builder_4 = EPSFBuilder(oversampling=4, maxiters = 20, progress_bar=False, 
                                     recentering_maxiters = 10, smoothing_kernel='quartic')
        epsf_4, _ = epsf_builder_4(stars)
        
        if self.epsf_plot:
            plt.figure()
            plt.imshow(epsf.data)
            plt.show()
        
        return epsf.data, epsf_4.data
    
    def calculate_fwhm(self, kernel):
        '''
        Calculate the Full Width at Half Maximum (FWHM) of an EPSF kernel in two dimensions.

        Parameters
        ----------
        kernel : 2d array
            EPSF kernel.

        Returns
        -------
        fwhm : float
            FWHM of the EPSF kernel.
        '''
        
        half_max = np.nanmax(kernel) / 2
        # indices = np.where(kernel >= half_max)
        contour = kernel > half_max
        
        y, x = np.where(contour)
        
        fwhm_x = np.ptp(x)
        fwhm_y = np.ptp(y)
        
        return max(fwhm_x, fwhm_y)

    def colourise(self, im, colour, intensity):
        '''
        Colourise the image. and scale the intensity.

        Parameters
        ----------
        im : 2d array
            FITS Image.
        colour : Tuple
            3 tuple scaled between 0 and 1.
        intensity : Float
            Multiply brightness by this amount

        Returns
        -------
        3d image
            Final 3d array in colour channel form.
        '''
        
        im_scaled = np.atleast_3d(im)

        colour = np.asarray(colour).reshape((1, 1, -1)) # Reshape the color (here, we assume channels last)
        return im_scaled * colour * intensity

    def colour_check(self, colour):
        '''
        Change the colour to a RGB tuple.

        Parameters
        ----------
        colour : Tuple or String
            Defining what colour choice for the channel

        Raises
        ------
        ValueError
            Needs to be a named colour in matplotlib or a tuple in the form (255,255,255).

        Returns
        -------
        colour : Tuple
            RGB colour tuple scaled between 0 and 1.
        '''

        if isinstance(colour, tuple):
            colour = tuple(ci/255 for ci in colour)
        elif isinstance(colour, str):
            colour = tuple(int(c*255) for c in mcolors.to_rgb(colour))
            colour = tuple(ci/255 for ci in colour)
        else:
            raise ValueError("Not a valid input. Try a named colour in matplotlib or a tuple in the form (255,255,255).")
        return colour

    def percent_norm(self, x, lower = 2, upper = 98, norm = 'linear'):
        '''
        Rescale the image to a percentage scale.

        Parameters
        ----------
        x : 2d array
            FITS image.
        lower : float
            lowest percent data to cut out.
        upper : float
            highest percent data to cut out.

        Returns
        -------
        arr_rescaled : 2d array
            Normalised percentage scaled 2d array.
        '''

        # x_low = np.nanpercentile(x, lower)
        # x_hi = np.nanpercentile(x, upper)
        
        # Scale the array so that its minimum and maximum values correspond to the 2nd and 98th percentile values, respectively
        # arr_rescaled = np.interp(x, (x_low, x_hi), (0, 1))
        
        normed = simple_norm(x, stretch=norm, min_percent=lower, max_percent=upper)
        arr_rescaled = normed(x)
        
        arr_rescaled = np.power(arr_rescaled, self.gamma)
        
        return arr_rescaled

    def plot(self):
        '''
        Plot the final image.

        Parameters
        ----------
        im_composite : 3d array
            2d arrays that are stacked in a third axis.
        '''
        im_composite = self.im_composite

        plt.figure(figsize = (self.figure_size,self.figure_size))
        plt.imshow(im_composite)
        plt.axis('off')
        plt.xlim(self.manual_override , im_composite.shape[1] - self.manual_override )
        plt.ylim(self.manual_override , im_composite.shape[0] - self.manual_override )
        if self.save:
            plt.savefig(os.path.join(self.save_folder, self.save_name + '.pdf'), format = 'pdf', bbox_inches = 'tight')
            plt.savefig(os.path.join(self.save_folder, self.save_name + '.png'), dpi = self.dpi, bbox_inches = 'tight')
        plt.show()
        
        return im_composite

    def register(self, T, R):
        """
        Register two images using cross-correlation.

        Parameters
        ----------
        T : 2d array
            Image to be transformed.
        R : 2d array
            Reference image.
        Returns
        -------
        R_new : 2d array
        """ 

        Rcm = R - np.median(R)
        Tcm = T - np.median(T)
        c = fftconvolve(Rcm, Tcm[::-1, ::-1])
        kernel = np.ones((3,3))
        c = convolve2d(c,kernel,mode='same')
        cind = np.where(c == np.max(c))
        try:
            xshift = cind[0][0]-Rcm.shape[0]+1
        except IndexError:
            print('Error: image failed to register.')
            return None
        yshift = cind[1][0]-Rcm.shape[1]+1
        imint = max(0,-xshift)
        imaxt = min(R.shape[0],R.shape[0]-xshift)
        jmint = max(0,-yshift)
        jmaxt = min(R.shape[1],R.shape[1]-yshift)
        iminr = max(0,xshift)
        imaxr = min(R.shape[0],R.shape[0]+xshift)
        jminr = max(0,yshift)
        jmaxr = min(R.shape[1],R.shape[1]+yshift)
        R_new = np.zeros_like(T)
        R_new[iminr:imaxr,jminr:jmaxr] = T[imint:imaxt,jmint:jmaxt]
        return R_new

# Flux calibration