from setuptools import setup


setup(
    name='ai-prism',
    version='0.1.0',    
    description='PRISM: Low-Cost LLM-Powered Data Processing',
    url='https://github.com/szeighami/prism',
    author='Sepanta Zeighami',
    author_email='zeighami@berkeley.edu',
    license='MIT',
    packages=['PRISM', 'PRISM.process', 'PRISM.bounds', 'PRISM.models', 'PRISM.sampler'],
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
