# Furuta Pendulum

Furuta Pendulum is a project containing the instructions and code for the Furuta Pendulum using policies created by RL Sim-To-Real methods.

In this document you will only find information regarding the software of the Furuta Pendulum which should be installed on the Nvidia Jetson and the remaining hardware instructions can be found here:

# Installation

First of all you have to follow the instructions to set up Nvidia Jetson, which can be found here (https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit)

After that you have to install Python 3.8 on the Nvidia Jetson. 

Clone this repository into a new directory (for example ~/Documents/furuta):

  git clone https://github.com/threethirds/furuta-pendulum

Install virtual environments:

  sudo apt install python3-venv
  cd /Documents/furuta
  python3 -m venv venv

Activate the virtual environment and install the required site-packages:
  
  source venv/bin/activate
  pip install -r requirements.txt
  
# Checking if the demos work 

## Angle Calculation

To check if the angle calculation works you have to run ads.py file

## Motor

Before that you will have to set up PWM on the Jetson (https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/hw_setup_jetson_io.html)

  sudo /opt/nvidia/jetson-io/jetson-io.py
  Configure 40-pin expansion header
  turn on pwm0 and pwm2 (with space bar)
  Back 
  Save and reboot to reconfigure pins

To check if the motor works you have to run pwm.py file

# The main file 

Run sactest.py
  

