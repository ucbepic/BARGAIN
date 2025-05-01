from setuptools import setup


setup(
    name='bargain',
    version='0.1.0',    
    description='BARGAIN: Guaranteed Accurate AI for Less',
    url='https://github.com/szeighami/BARGAIN',
    author='Sepanta Zeighami',
    author_email='zeighami@berkeley.edu',
    license='MIT',
    packages=['BARGAIN', 'BARGAIN.process', 'BARGAIN.bounds', 'BARGAIN.models', 'BARGAIN.sampler'],
    install_requires=['pandas',
                      'numpy',                     
                      'tqdm',                     
                      'openai',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: OS Independent',        
        'Programming Language :: Python :: 3',
    ],
)
