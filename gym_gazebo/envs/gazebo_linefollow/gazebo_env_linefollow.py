
import cv2
import gym
import math
import rospy
import roslaunch
import time
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import Image
from time import sleep

from gym.utils import seeding


class Gazebo_Linefollow_Env(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        LAUNCH_FILE = '/home/fizzer/enph353_gym-gazebo-noetic/gym_gazebo/envs/ros_ws/src/linefollow_ros/launch/linefollow_world.launch'
        gazebo_env.GazeboEnv.__init__(self, LAUNCH_FILE)
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world',
                                              Empty)

        self.action_space = spaces.Discrete(3)  # F,L,R
        self.reward_range = (-np.inf, np.inf)
        self.episode_history = []

        self._seed()

        self.bridge = CvBridge()
        self.timeout = 0  # Used to keep track of images with no line detected


    def process_image(self, data):
        '''
            @brief Coverts data into a opencv image and displays it
            @param data : Image data from ROS

            @retval (state, done)
        '''
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # cv2.imshow("raw", cv_image)

        NUM_BINS = 3
        state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        done = False

        # TODO: Analyze the cv_image and compute the state array and
        # episode termination condition.
        #
        # The state array is a list of 10 elements indicating where in the
        # image the line is:
        # i.e.
        #    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] indicates line is on the left
        #    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] indicates line is in the center
        #
        # The episode termination condition should be triggered when the line
        # is not detected for more than 30 frames. In this case set the done
        # variable to True.
        #
        # You can use the self.timeout variable to keep track of which frames
        # have no line detected.

        '''
        @brief Detect the location of the line in the robot view and mark it with a circle. This code comes from our Week 2 Lab.
        '''
        THRESHOLD = 140
        ROW = 200

        def find_line(pixel_row):
            on_line = False

            left_edge = 0
            right_edge = 0

            for i in range(320):
                if pixel_row[i] < THRESHOLD and (not on_line):
                    left_edge = i
                    on_line = True
                elif pixel_row[i] > THRESHOLD and on_line:
                    right_edge = i
                    on_line = False

            # When the line runs off to the right, some odd behaviour occurred. This is an attempt to mitigate that.
            if left_edge != 0 and right_edge == 0:
                right_edge = 319

            return [left_edge, right_edge]

        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        line_edges = find_line(gray_image[ROW])
        # print(line_edges)

        circ_mid = round((line_edges[0] + line_edges[1]) / 2)
        circ_rad = abs(round((line_edges[0] - line_edges[1]) / 3))    

        if not (line_edges[0] == 0 and line_edges[1] == 0):
            cv_image = cv2.circle(cv_image, (circ_mid, ROW), circ_rad, (255,0,0), -1)

        '''
        @brief Draw lines over the robot camera feed to better visualize the 10 sections of the robot view.
        '''
        LINES_START = 31
        for i in range(9):
            cv_image = cv2.line(cv_image, (LINES_START + 32 * i, 0), (LINES_START + 32 * i, 239), (0, 0, 0), 1)

        '''
        @brief Determine which section of the image the drawn circle is in. If in none, then we do not see the line.
        '''
        if self.timeout >= 30:
            done = True

        if line_edges[0] == 0 and line_edges[1] == 0:
            self.timeout += 1
        else:
            section = circ_mid // 32
            state = np.eye(10)[section]

        cv_image = cv2.putText(cv_image, ', '.join(str(num) for num in state), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

        ###############################

        # cv2.imshow("gray", gray_image)
        cv2.imshow("raw", cv_image)
        cv2.waitKey(1) # Without this line of code, only a black screen is displayed

        return state, done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        self.episode_history.append(action)

        vel_cmd = Twist()

        if action == 0:  # FORWARD
            vel_cmd.linear.x = 0.4
            vel_cmd.angular.z = 0.0
        elif action == 1:  # LEFT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.5
        elif action == 2:  # RIGHT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = -0.5

        self.vel_pub.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw', Image,
                                              timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state, done = self.process_image(data)

        # Set the rewards for your action
        if not done:
            if action == 0:  # FORWARD
                reward = 4
            elif action == 1:  # LEFT
                reward = 2
            else:
                reward = 2  # RIGHT
        else:
            reward = -200

        return state, reward, done, {}

    def reset(self):

        print("Episode history: {}".format(self.episode_history))
        self.episode_history = []
        print("Resetting simulation...")
        # Resets the state of the environment and returns an initial
        # observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            # reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            # resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # read image data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw',
                                              Image, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        self.timeout = 0
        state, done = self.process_image(data)

        return state
