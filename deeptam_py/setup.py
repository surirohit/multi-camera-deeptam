# Copyright 2019 Mayank Mittal. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Installation script for the 'deeptam_tracker' python package."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from setuptools import setup, find_packages

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    'pylint',
    'minieigen',
    'numpy',
    'scipy',
    'matplotlib', 
    'scikit-image',
    'tensorflow-gpu==1.9.0'
]


# Helper scripts provided by this package
SCRIPTS = [
    'bin/single_camera_tracking.py'
]

# Installation operation
setup(name='deeptam_tracker',
      version='0.0.0',
      description='Tracking and Mapping using Deep Learning',
      keywords=["robotics", "machine learning", "visual odometry"],
      packages=[package for package in find_packages() if package.startswith('deeptam')],
      scripts=SCRIPTS,
      install_requires=INSTALL_REQUIRES,
      zip_safe=False)

# EOF
