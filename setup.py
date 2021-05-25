import setuptools

setuptools.setup(
    name='PhyPy802dot11adComponents',
    version='0.1',
    description='802.11ad PHY component definitions',
    packages=setuptools.find_packages(),
	include_package_data=True,
	package_data={
        'PhyPy802dot11adComponents': ['*.npz', '*.npy']
    }
)
