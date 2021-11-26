# Furuta Pendulum

Furuta Pendulum is a project containing the instructions and code for the Furuta Pendulum using policies created by RL Sim-To-Real methods.

In this document you will only find information regarding the software of the Furuta Pendulum which should be installed on the Nvidia Jetson.

# Preparing Jetson nano

Follow the instructions to set up Nvidia Jetson, which can be found here 
(https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit)

## Enabling the pwm pins

This project uses pwm pins that need to be configured. Configuration is buggy. 
It should work as follows:

    sudo /opt/nvidia/jetson-io/jetson-io.py
    # -> Configure 40-pin expansion header
    # -> turn on pwm0 and pwm2 (with space bar)
    # -> Save and reboot to reconfigure pins

The first command will likely fail. First failure is that a barely visible window
blinks and disappears. One way to tray to fix it is by commenting out this line
in `jetson-io.py` script:

    # curses.resizeterm(height, width)

Second failure is a complaint that `no DTB found for NVIDIA Jetson` This in our case
was fixed by changing this line in  `/opt/nvidia/jetson-io/Jetson/board.py`:

    #dtbdir = os.path.join(self.bootdir, 'dtb')
    dtbdir = os.path.join(self.bootdir, '')

# Installation

Install Python 3.8:

    sudo apt install python3.8 python3.8-venv python3.8-dev

Then install this project into a virtual environment

    git clone https://github.com/threethirds/furuta-pendulum
    cd furuta-pendulum
    python3.8 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
  
# Checking if the demos work 

## Angle Calculation

To check if the angle calculation works run 

    cd furuta-pendulum/Demo
    python ads.py

To check if the motor control works run

    cd furuta-pendulum/Demo
    python pwm.py

# The main file 

Run 

    cd furuta-pendulum
    python sactest.py
  

