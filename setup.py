from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'manriix_photo_va'

setup(
    name=package_name,
    version='2.0.0',
    packages=find_packages(exclude=['test']),
    package_data={
        'config': [
            '*.py',
            'shared/*.yaml',
            'hardware/*.yaml',
            'algorithms/*.yaml',
            'algorithms/auto_framing/*.yaml',
            'workflows/*.yaml',
            'system/*.yaml',
        ],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Launch files
        ('share/' + package_name + '/launch', glob('launch/*.py') if os.path.exists('launch') else []),
        # Model files
        ('share/' + package_name + '/models', glob('models/*') if os.path.exists('models') else []),
        # Config files (for ROS2 share directory access)
        ('share/' + package_name + '/config', glob('config/*.py')),
        ('share/' + package_name + '/config/shared', glob('config/shared/*.yaml')),
        ('share/' + package_name + '/config/hardware', glob('config/hardware/*.yaml')),
        ('share/' + package_name + '/config/algorithms', glob('config/algorithms/*.yaml')),
        ('share/' + package_name + '/config/algorithms/auto_framing', glob('config/algorithms/auto_framing/*.yaml')),
        ('share/' + package_name + '/config/workflows', glob('config/workflows/*.yaml')),
        ('share/' + package_name + '/config/system', glob('config/system/*.yaml')),
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