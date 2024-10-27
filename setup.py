from setuptools import setup, find_packages

setup(name = 'AstroColour',
      version = '1.1.0',
      author = 'Zachary G. Lane', 
      author_email = 'zacastronomy@gmail.com', 
      description = 'Create colour images of astronomical objects.', 
      packages = find_packages(), 
      scripts=['AstroColour/AstroColour.py'], 
      install_requires = ['numpy', 'matplotlib', 'astropy', 'scipy', 
                          'pandas', 'scikit-learn', 'photutils'])