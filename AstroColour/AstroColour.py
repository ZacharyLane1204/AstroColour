import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import matplotlib.colors as mcolors
import os
import pandas as pd
from astropy.io import fits
from scipy.signal import fftconvolve, convolve2d

# %matplotlib widget

plt.rc('text', usetex=True)
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
    def __init__(self, image1, image2, image3, colour1 = 'red', colour2 = 'green', colour3 = 'blue', intensity1 = 1, intensity2 = 1, intensity3 = 1, 
        upper1 = 98, lower1 = 2, upper2 = 98, lower2 = 2, upper3 = 98, lower3 = 2,
        save = False, savename = '', figure_size = None, manual_override = False):
        '''
        Create a RGB image from three images.

        Parameters
        ----------
        image1 : 2d array
            FITS Image.
        image2 : 2d array
            FITS Image.
        image3 : 2d array
            FITS Image.
        colour1 : Tuple or String
            Defining what colour choice for the channel.
        colour2 : Tuple or String
            Defining what colour choice for the channel.
        colour3 : Tuple or String
            Defining what colour choice for the channel.
        intensity1 : Float
            Multiply brightness by this amount.
        intensity2 : Float
            Multiply brightness by this amount.
        intensity3 : Float
            Multiply brightness by this amount.
        upper1 : Float
            Highest percent data to cut out.
        lower1 : Float
            Lowest percent data to cut out.
        upper2 : Float
            Highest percent data to cut out.
        lower2 : Float
            Lowest percent data to cut out.
        upper3 : Float
            Highest percent data to cut out.
        lower3 : Float
            Lowest percent data to cut out.
        save : Boolean
            Whether to save the image.
        savename : String
            Detail to add in the saved filename.
        figure_size : Float
            Dimension of the image.
        manual_override : Boolean
            Whether to manually override the limits.
        '''

        self.intensity1 = intensity1
        self.intensity2 = intensity2
        self.intensity3 = intensity3
        self.upper1 = upper1
        self.lower1 = lower1
        self.upper2 = upper2
        self.lower2 = lower2
        self.upper3 = upper3
        self.lower3 = lower3
        self.save = save
        self.savename = savename
        self.figure_size = figure_size
        if self.figure_size == None:
            self.figure_size = fig_width_full
        self.manual_override = manual_override

        self.image1 = image1
        self.image2 = self.register(image2, image1)
        self.image3 = self.register(image3, image1)

        self.colour1 = self.colour_check(colour1)
        self.colour2 = self.colour_check(colour2)
        self.colour3 = self.colour_check(colour3)

    def colour(self):
        '''
        Create the RGB image.
        '''
        
        p_norm1 = self.percent_norm(self.image1, lower = self.lower1, upper = self.upper1)
        p_norm2 = self.percent_norm(self.image2, lower = self.lower2, upper = self.upper2)
        p_norm3 = self.percent_norm(self.image3, lower = self.lower3, upper = self.upper3)

        file_1 = self.colourise(p_norm1, self.colour1, self.intensity1)
        file_2 = self.colourise(p_norm2, self.colour2, self.intensity2)
        file_3 = self.colourise(p_norm3, self.colour3, self.intensity3)

        im_composite = np.clip(file_1 + file_2 + file_3, 0, 5)

        self.final_plot(im_composite, self.figure_size, self.savename, self.save, self.manual_override)

        return im_composite

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

        # Reshape the color (here, we assume channels last)
        colour = np.asarray(colour).reshape((1, 1, -1))
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

    def percent_norm(self, x, lower = 2, upper = 98):
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

        x_low = np.nanpercentile(x, lower)
        x_hi = np.nanpercentile(x, upper)
        
        # Scale the array so that its minimum and maximum values correspond to the 2nd and 98th percentile values, respectively
        arr_rescaled = np.interp(x, (x_low, x_hi), (0, 1))
        return arr_rescaled

    def final_plot(self, im_composite, figure_size, savename, save, manual_override):
        '''
        Plot the final image.

        Parameters
        ----------
        im_composite : 3d array
            2d arrays that are stacked in a third axis.
        figure_size : Integer
            Dimension of the image
        savename : String
            Detail to add in the saved filename.
        save : Boolean
            Whether to save the image.
        manual_override : Boolean
            Whether to manually override the limits.
        '''

        plt.figure(figsize = (self.figure_size,self.figure_size))
        plt.imshow(im_composite)
        plt.axis('off')
        plt.show()

    def register(self, T, R):
        """
        Register two images using cross-correlation.

        Parameters
        ----------
        T : 2d array
            Image to be registered.
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
