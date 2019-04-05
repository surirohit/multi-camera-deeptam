#!/bin/bash

function install_system_deps_ubuntu_1604()
{
  set -e
  set -o xtrace
  APT_GET_FLAGS=-qq
  sudo apt update

  ## GCC v4.9
  sudo apt install $APT_GET_FLAGS gcc-4.9 g++-4.9

  ### Build dependencies
  sudo apt install $APT_GET_FLAGS cmake build-essential unzip

  ## Install python3
  sudo apt install $APT_GET_FLAGS python3-dev python3-pip
}

function install_python_deps()
{
  # Install virtualenv virtualenvwrapper
  sudo pip install virtualenv virtualenvwrapper

  # Setup virtualenv and virtualenvwrapper
  WORKON_HOME=$HOME/.virtualenvs
  printf '\n%s\n%s\n%s' '# virtualenv' 'export WORKON_HOME=$WORKON_HOME' \
  'source /usr/local/bin/virtualenvwrapper.sh \n' >> ~/.bashrc
  source $HOME/.bashrc
  mkdir -p $WORKON_HOME

  # Create the virtualenv for tensorflow (called deeptam_py)
  VIRTUALENV_NAME=deeptam_py
  echo "Creating Virtual Environment: ${VIRTUALENV_NAME}"
  echo "==============================================="
  rm -rf "$WORKON_HOME/$VIRTUALENV_NAME"
  virtualenv --no-site-packages --python=python3 $VIRTUALENV_NAME || true

  # Activate virtualenv and update pip
  source $HOME/.bashrc
  source $HOME/.virtualenvs/$VIRTUALENV_NAME/bin/activate
  pip install -U pip
}

##
# Main script commands
##

## Preparations

### Back-up the local BASH configuration file
cp $HOME/.bashrc $HOME/.bashrc.backup

## Install dependencies

### Check Ubuntu version
UBUNTU_VERSION=$( lsb_release -r | awk '{ print $2 }' )

### Install dependencies depending on the version of Ubuntu
if [[ "$UBUNTU_VERSION" == "16.04" ]]; then
#  install_system_deps_ubuntu_1604
  echo "Already installed stuff!"
else
  echo "Error: Unknown or unsupported Linux distribution."
  exit
fi

### Install common Python dependencies
install_python_deps

## Install DeepTAM

### Add environment variable for DEEPTAM_ROOT
DEEPTAM_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
sed -i "/\b\DEEPTAM_ROOT\\b/d" $HOME/.bashrc
printf 'export DEEPTAM_ROOT='$DEEPTAM_ROOT'\n' >> $HOME/.bashrc
bash
source $HOME/.bashrc

### Install the lmbdspecial ops submodule
git submodule init
git submodule update
LMBSPECIALOPS_ROOT=$DEEPTAM_ROOT/lib/lmbspecialops
cd $LMBSPECIALOPS_ROOT
#### change compiler version
CC=gcc-4.9
CXX=g++-4.9
#### build the submodule
mkdir build
cd build
cmake ..
make

#### add to path of virtual environment
add2virtualenv $LMBSPECIALOPS_ROOT/python

### Install the python-packages
cd lib
pip install -e .

exit

# EOF
