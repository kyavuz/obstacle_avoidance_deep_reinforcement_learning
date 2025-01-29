# Dynamic Obstacle Avoidance for Mobile Robots Using Vision-Based Deep Reinforcement Learning
This repository contains the implementation of a Double Deep Q-Network (D3QN)-based dynamic obstacle avoidance system for mobile robots. The project is developed using ROS, Gazebo, and TensorFlow to train a TurtleBot3 robot to navigate safely in environments containing both static and dynamic obstacles.

The study focuses on vision-based navigation using synchronized RGB and depth images instead of LIDAR. The dataset is collected in a simulated environment and used to train a reinforcement learning agent that can adapt to moving obstacles.

## Table of Contents
* Project Overview
* Installation and Setup
* Dataset
* Training the Model
* Testing the Model
* Results
* References

## Project Overview
Traditional obstacle avoidance methods rely on LIDAR sensors or handcrafted rules for navigation. However, vision-based deep reinforcement learning (DRL) offers a more flexible and adaptive solution for real-world applications.

This project develops a D3QN-based navigation policy using synchronized RGB and depth images collected from Gazebo simulations. The TurtleBot3 robot is trained to avoid obstacles dynamically without explicit path planning.

### Key Features:
- [ ] Uses Double Deep Q-Network (D3QN) for obstacle avoidance
- [ ] Vision-based learning using RGB and depth images
- [ ] Handles both static and dynamic obstacles
- [ ] Implemented in ROS Noetic, Gazebo 11, and TensorFlow
- [ ] Dataset collected in TurtleBot3_house environment

## Installation and Setup
* This project was made using Ubuntu 20.04.
* Versions and dependencies may change.
1. Install ROS Noetic and Gazebo 11
2. Dependencies:
  ```sh
tensorflow==2.13.1
numpy==1.24.3
opencv-python==4.10.0.84
rospkg==1.5.1
message-filters==1.17.0
catkin-tools==0.9.4
torch==2.4.1
torch_tensorrt==2.4.0
torchvision==0.19.1
pillow==10.4.0
  ```
3. You will need to attach an rgb and a depth camera to TurtleBot3. To do that you need to edit and add rgb and camera xml the TurtleBot3's urdf files. After adding cameras, you can see the camera topics as:
  ```sh
/camera/rgb/image_raw
/depth_camera/depth/image_raw
  ```
when running the following command:
  ```sh
rostopic list
  ```

## Dataset
The dataset is collected using Gazebo simulations with the TurtleBot3 robot. It contains synchronized RGB and depth images in the following structure:
  ```sh
datasets/
│── turtlebot_rgb_images_static/    # 5000 RGB images (static obstacles)
│── turtlebot_depth_images_static/  # 5000 depth images (static obstacles)
│── turtlebot_rgb_images_dynamic/   # 5000 RGB images (dynamic obstacles)
│── turtlebot_depth_images_dynamic/ # 5000 depth images (dynamic obstacles)
  ```
To generate your own dataset, run the get_dataset.py file after configuring dataset paths in it.
  ```sh
python3 get_dataset.py
  ```
This script will subscribe to the robot’s RGB and depth camera topics and save the images in the dataset folders.

## Training the Model
1. Run the Training Script

    To train the D3QN model, run the D3QN_training.py file after configuring dataset and output model paths in it.:
  ```sh
python3 D3QN_training.py
  ```
  This will:
  - [ ] Load RGB and depth images from the dataset
  - [ ] Train the D3QN model for dynamic obstacle avoidance
  - [ ] Save the trained model

2. Training Configuration
    You can modify the training parameters inside D3QN_training.py:
  ```sh
MAX_EPISODES = 10  # Number of episodes to train
MAX_STEPS = 4999   # Steps per episode
BATCH_SIZE = 16    # Minibatch size
MEMORY_SIZE = 5000 # Replay buffer size
LEARNING_RATE = 1e-4
  ```
## Testing the Model
1. Run Gazebo
2. Run the Trained Model
  To deploy the trained model in Gazebo, run:
  ```sh
python3 D3QN_test.py
  ```
  This will:
  - [ ] Load the trained D3QN model
  - [ ] Subscribe to RGB and depth camera topics
  - [ ] Control the TurtleBot3 in the simulated world

2. Select a Model for Testing
  Trained models are saved in the trained_models/ directory:
  ```sh
trained_models/
│── models_static/              # Trained with static obstacles
│── models_dynamic/             # Trained with dynamic obstacles
│── models_dynamic_and_static/  # Trained with mixed obstacles
  ```
  Modify D3QN_test.py to use a specific model:
  ```sh
MODEL_PATH = "trained_models/models_dynamic/dqn_model_episode_5.keras"
  ```
## Results
### Static Obstacles
* The best model achieved a reward of 2498.57.
* Successfully avoided tables, chairs, and walls.
* Failure case: Collided with table legs, as the camera lost sight of obstacles after impact.
### Dynamic Obstacles
* The best model achieved a reward of 2514.92.
* Successfully avoided moving obstacles like a rolling ball.
* Failure case: Occasionally collided with walls due to depth misinterpretation.
### Static & Dynamic Obstacles Combined
* The best model achieved a reward of 2521.41.
* Handled static obstacles well but struggled with moving objects.
* Future improvement: More training on dynamic obstacles could enhance performance.

## Helpful notes
The roscore can be launched using the roscore executable:
  ```sh
roscore
  ```
To start Gazebo with TurtleBot3 and turtlebot3_house world:
  ```sh
export TURTLEBOT3_MODEL=waffle_pi
roslaunch turtlebot3_gazebo turtlebot3_house.launch
  ```

To start camera view:
  ```sh
rqt_image_view
  ```

To control TurtleBot3 with keyboard:
  ```sh
export TURTLEBOT3_MODEL=waffle_pi
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
  ```

## References
1. Zamora, I., Lopez, N. G., Vilches, V. M., & Cordero, A. H. (2017). Extending the OpenAI gym for robotics: A toolkit for reinforcement learning using ROS and Gazebo. arXiv preprint arXiv:1608.05742v2.

2. Rickstaa. (n.d.). RICKSTAA/Ros-gazebo-gym: Framework for integrating ROS and Gazebo with Gymnasium, streamlining the development and training of RL algorithms in realistic robot simulations. GitHub. Retrieved January 27, 2025, from https://github.com/rickstaa/ros-gazebo-gym

3. Robotis-Git. (n.d.). Robotis-git/TurtleBot3: ROS packages for TurtleBot3. GitHub. Retrieved January 27, 2025, from https://github.com/ROBOTIS-GIT/turtlebot3

4. Wenzel, P., Schon, T., Leal-Taixe, L., & Cremers, D. (2021). Vision-based mobile robotics obstacle avoidance with deep reinforcement learning. In 2021 IEEE International Conference on Robotics and Automation (ICRA) (pp. 14360–14366). https://doi.org/10.1109/icra48506.2021.9560787

5. Xie, L., Wang, S., Markham, A., & Trigoni, N. (2017). Towards monocular vision-based obstacle avoidance through deep reinforcement learning. In RSS 2017 Workshop on New Frontiers for Deep Learning in Robotics. Retrieved January 27, 2025, from http://dblp.uni-trier.de/db/journals/corr/corr1706.html#XieWMT17

6. Xie, L. (n.d.). Xie9187/Monocular-Obstacle-Avoidance: Codebase for "Towards Monocular Vision Based Obstacle Avoidance through Deep Reinforcement Learning." GitHub. Retrieved January 27, 2025, from https://github.com/xie9187/Monocular-Obstacle-Avoidance
