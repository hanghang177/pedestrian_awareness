#!/usr/bin/env python

import time
import rospy
from std_msgs.msg import Float64

from DFRobot_RaspberryPi_Expansion_Board import DFRobot_Expansion_Board_IIC as Board
from DFRobot_RaspberryPi_Expansion_Board import DFRobot_Expansion_Board_Servo as Servo

board = Board(1, 0x10)    # Select i2c bus 1, set address to 0x10
servo = Servo(board)

servo_pin = 0
esc_pin = 1

def esc_callback(esc_val):
    esc_sval = int(esc_val.data * 0.9 + 90)
    if esc_sval < 0:
        esc_sval = 0
    elif esc_sval > 180:
        esc_sval = 180
    servo.move(esc_pin, esc_sval)

def servo_callback(servo_ang):
    servo_sval = int(servo_ang.data * 2 + 90)
    if servo_sval < 0:
        servo_sval = 0
    elif servo_sval > 180:
        servo_sval = 180
    servo.move(servo_pin, servo_sval)

if __name__ == '__main__':
    # Set up board
    board.begin()
    servo.begin()

    # Initialize servo and esc
    servo.move(servo_pin, 90)
    servo.move(esc_pin, 90)

    # Set up listener
    rospy.init_node('pwm_listener', anonymous=True)
    rospy.Subscriber('esc', Float64, esc_callback)
    rospy.Subscriber('servo', Float64, servo_callback)

    # Spin
    rospy.spin()
