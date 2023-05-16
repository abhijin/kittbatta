import setuptools

setuptools.setup(name='kittbatta',
        version='0.6',
        description="AA's helper functions",
        url='#',
        author='Abhijin Adiga',
        install_requires=['opencv-python'],
        author_email='',
        packages=setuptools.find_packages(),
        zip_safe=False,
        include_package_data=True,
        package_data={'': ['aadata/datasets/usa/*', 'aadata/datasets/world/*']})
