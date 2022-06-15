from setuptools import setup

setup(
    name="ur_gym",
    author="Thomas Lips",
    author_email="thomas.lips@ugent.be",
    version="0.1",
    description="Gym environments for in-vivo learning using the AIRO Universal Robot",
    url="https://github.com/tlpss/ur-gym",
    packages=["ur_gym"],
    # install_requires=[], # requirements are not handled by this package, since its use is mostly to provide easier use of imports.
)