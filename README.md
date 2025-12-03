# AstroColour
**Colour the Astro Images in bands/colours.**

## Install Instructions
```
git clone https://github.com/ZacharyLane1204/AstroColour.git
```
In the Folder: e.g. 
```cd AstroColour/``` Run: ```pip install .```
If you are in a Windows environment you should use powershell.

## 

in your script import the module. Here is an example:
```
from AstroColour.AstroColour import RGB

rgb = RGB(colour_cube,
          save = False, save_name = 'test', save_folder = './', 
          epsf_plot=False, epsf = False,
          bkg_plot = False, temp_save = True, run = False, manual_override=0)

colour = rgb.master_plot(colour_cube, 
                         colours = ['red', 'green', 'blue'],
                         intensities = [0.6, 1, 0.56], 
                         gamma = [0.95, 0.95, 0.95],
                         norms = ['asinh', 'asinh', 'asinh'], 
                         uppers = [99, 99, 99],
                         lowers = [5, 5, 5], 
                         interactive=False)
```

```

RGB Class

Create a RGB image from three images.

Parameters
----------
images : List
    List of Numpy Arrays of data images.
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
epsf : Boolean
    Whether to use the EPSF method.
epsf_plot : Boolean
    Whether to plot the EPSF kernel.
run : Boolean
    Whether to process images or just use the framework
        
```

```

master_plot function

Master RGB composite plotter.

Parameters
----------
cleaned_images : List
    List of cleaned 2D numpy arrays.
lowers : Float or List of Float
    Lower percentile(s) (or counts) for normalization.
uppers : Float or List of Float
    Upper percentile(s) (or counts) for normalization.
norms : String or List of String
    Normalization method(s) ('linear', 'sqrt', 'log', 'asinh', 'sinh').
colours : List
    List of colour choices (tuples or strings).
intensities : Float or List of Float
    Intensity scaling factor(s).
gamma : Float or List of Float
    Gamma correction factor(s).
interactive : Boolean
    Whether to use interactive widgets for parameter adjustment.
method : String
    Normalization method type ('percent', 'sigma', 'counts').

Returns
-------
im_comp : 2D array
    The final RGB composite image.
        
```


Versions:
- numpy == 1.26.4
- matplotlib == matplotlib
- astropy == 7.1.0
- pandas == 2.2.3
- photutils == 1.13.0
- scikit-learn == 1.5.1
- opencv-python == 4.9.0.80
- astroscrappy == 1.2.0
- ipympl == 0.9.7
- ipython == 8.28.0
- ipywidgets == 7.8.4