import Jetson.GPIO as GPIO
import time 
GPIO.setmode(GPIO.BOARD)
GPIO.setup(33, GPIO.OUT)
GPIO.setup(38, GPIO.OUT, initial=1)
my_pwm = GPIO.PWM(33, 100)
my_pwm.start(15)

try:
  while True:
    #time.sleep(2)
    '''
    for i in range(500):
        GPIO.output(38,0)
        time.sleep(1./240)
        GPIO.output(38,1)
        time.sleep(1./240)
    for i in range(500):
        GPIO.output(38,0)
        time.sleep(1./30)
        GPIO.output(38,1)
        time.sleep(1./30)
    '''
    #GPIO.output(38, 1)
    #time.sleep(2)
    #GPIO.output(38, 0)
    #my_pwm.start(15)
    #my_pwm.start(70)
    #time.sleep(3)
    #GPIO.output(38, 1)
    #time.sleep(3)
    #GPIO.output(38, 0)
    #my_pwm.start(50)
except Exception as e:
  print("Shut down", str(e))
finally:
  GPIO.cleanup()
  print("clean")
