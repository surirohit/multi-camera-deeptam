#!/usr/bin/env python

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

from setuptools import setup

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    'minieigen==0.5.4',
    'numpy==1.16.2',
    'scipy',
    'pyyaml',
    'matplotlib', 
    'scikit-image==0.14.2',
    'tensorflow-gpu==1.9.0',
    'opencv-python',
    'pillow==6.0.0'
]

# List of packages to install
PACKAGES = [
    'deeptam_tracker',
    'lmbspecialops'
]

# List of packages and their directories
PACKAGES_DIR = {
    '': 'lib',
    'deeptam_tracker': 'lib/deeptam_tracker',
    'lmbspecialops':'lib/lmbspecialops/python'
}

# Installation operation
setup(name='deeptam',
      version='0.0.0',
      description='Tracking and Mapping using Deep Learning',
      keywords=["robotics", "machine learning", "visual odometry"],
      include_package_data=True,
      python_requires='==3.5.*',
      packages=PACKAGES,
      package_dir=PACKAGES_DIR,
      install_requires=INSTALL_REQUIRES,
      zip_safe=False)

# EOF
