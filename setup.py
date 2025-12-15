from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'manriix_photo_va'

setup(
    name=package_name,
    version='2.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Configuration files
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        # Launch files
        ('share/' + package_name + '/launch', glob('launch/*.py') if os.path.exists('launch') else []),
        # Model files
        ('share/' + package_name + '/models', glob('models/*') if os.path.exists('models') else []),
    ],
    install_requires=[
        'setuptools',
    ],
    zip_safe=True,
    maintainer='Hype-Lasantha',
    maintainer_email='lasantha.k@hypeinvention.com',
    description='Manriix Automated Photography Robot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'photo_capture_node = mipcs.nodes.photo_capture_node:main',
        ],
    },
)