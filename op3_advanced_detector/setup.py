from setuptools import setup
import os
from glob import glob

package_name = 'op3_advanced_detector'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ROBOTIS',
    maintainer_email='robotis@robotis.com',
    description='Advanced ball and face detector for ROBOTIS-OP3',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'advanced_detector = op3_advanced_detector.op3_advanced_detector:main',
        ],
    },
)