import rospy
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from models import DuelingDQN

# Hyperparameters
IMAGE_HEIGHT, IMAGE_WIDTH = 128, 160
MODEL_PATH = "/home/kagan/models_dynamic/dqn_model_episode_5.keras"

class TurtleBotTest:
    def __init__(self):
        rospy.init_node("turtlebot_test")
        self.bridge = CvBridge()

        # ROS Subscribers
        self.rgb_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.rgb_callback)
        self.depth_sub = rospy.Subscriber("/depth_camera/depth/image_raw", Image, self.depth_callback)

        # ROS Publisher
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        # Load the trained model
        self.model = load_model(MODEL_PATH, custom_objects={"DuelingDQN": DuelingDQN})
        self.rgb_image = None
        self.depth_image = None

    def rgb_callback(self, msg):
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.rgb_image = cv2.resize(rgb_image, (IMAGE_WIDTH, IMAGE_HEIGHT)) / 255.0
        except Exception as e:
            rospy.logerr(f"RGB Callback Error: {e}")

    def depth_callback(self, msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            depth_image_resized = cv2.resize(depth_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
            self.depth_image = depth_image_resized[..., np.newaxis] / 255.0
        except Exception as e:
            rospy.logerr(f"Depth Callback Error: {e}")

    def get_action(self):
        if self.rgb_image is None or self.depth_image is None:
            return None  # No image received yet
        state = np.concatenate((self.depth_image, self.rgb_image), axis=-1)
        state = np.expand_dims(state, axis=0)  # (1, 128, 160, 4)
        q_values = self.model.predict(state)
        action = np.argmax(q_values)
        print("action : ", action)
        return action

    def execute_action(self, action):
        twist = Twist()

        if action == 0:  # Move forward
            twist.linear.x = 0.2
            twist.angular.z = 0.0
        elif action == 1:  # Forward left turn
            twist.linear.x = 0.1
            twist.angular.z = 0.5
        elif action == 2:  # Forward right turn
            twist.linear.x = 0.1
            twist.angular.z = -0.5
        elif action == 3:  # Stop
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        elif action == 4:  # Slow down
            twist.linear.x = 0.1  # Move forward more slowly
            twist.angular.z = 0.0
        elif action == 5:  # Full left turn
            twist.linear.x = 0.0
            twist.angular.z = 1.0
        elif action == 6:  # Full right turn
            twist.linear.x = 0.0
            twist.angular.z = -1.0

        self.cmd_vel_pub.publish(twist)


    def test_loop(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            action = self.get_action()
            if action is not None:
                self.execute_action(action)
            rate.sleep()


if __name__ == "__main__":
    bot_test = TurtleBotTest()
    print("Starting test...")
    bot_test.test_loop()
