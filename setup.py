from setuptools import setup,find_packages

setup(
    name            =  'hierarchical_grid_opt',
    version         =  '1.0.0',
    packages        = ['hierarchical_grid_opt'], # can also write: find_packages()
    url             =  '',
    license         =  '',
    author          ='Sandro C. Lera',
    author_email    ='sandrolera@gmail.com',
    description     ='hierarchical brute force optimization',
    python_requires ='>3.5.2',
    install_requires=[
                        "numpy>=1.20.0",
                     ]
)
