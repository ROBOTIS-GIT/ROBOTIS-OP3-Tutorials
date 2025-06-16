from setuptools import setup
import os
from glob import glob

package_name = 'op3_voice_control'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robotis',
    maintainer_email='robotis@robotis.com',
    description='ROBOTIS OP3 Voice Control Package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'speech_recognition_node = op3_voice_control.speech_recognition_node:main',
            'command_processor = op3_voice_control.command_processor:main',
        ],
    },
)