import rospy
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

# dataset directories
output_dir_rgb = "/path/to/turtlebot_rgb_images" # /home/username/turtlebot_rgb_images
output_dir_depth = "/path/to/turtlebot_depth_images"
os.makedirs(output_dir_rgb, exist_ok=True)
os.makedirs(output_dir_depth, exist_ok=True)

# ROS and OpenCV bridge
bridge = CvBridge()

index = 0

def callback(rgb_msg, depth_msg):
    global index
    try:
        if index<4999:
            # convert ROS messages to OpenCV format
            rgb_image = bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth_image = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

            # normalize depth images
            depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # save images
            rgb_path = os.path.join(output_dir_rgb, f"{index}.png")
            depth_path = os.path.join(output_dir_depth, f"{index}.png")
            cv2.imwrite(rgb_path, rgb_image)
            cv2.imwrite(depth_path, depth_image_normalized)

            print(f"RGB and depth images saved: {index}")

            index += 1
        else:
            print("Dataset saved successfully.")

    except Exception as e:
        rospy.logerr(f"Image processing error: {e}")

if __name__ == "__main__":
    rospy.init_node("synchronized_image_saver")

    # listen RGB and depth topics
    rgb_sub = message_filters.Subscriber("/camera/rgb/image_raw", Image)
    depth_sub = message_filters.Subscriber("/depth_camera/depth/image_raw", Image)

    # synchronize RGB and depth topics
    ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=10, slop=0.1)
    ts.registerCallback(callback)

    print("Synchronous image collection started. To close Ctrl+C.")
    rospy.spin()
