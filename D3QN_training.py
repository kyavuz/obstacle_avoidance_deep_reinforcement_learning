import tensorflow as tf
import numpy as np
import os
import random
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.optimizers import Adam
from collections import deque
from natsort import natsorted
from models import DuelingDQN
import cv2

# Dataset directories
RGB_STATIC_DIR = "/path/to/turtlebot_rgb_images_static" # /home/username/turtlebot_rgb_images_static
DEPTH_STATIC_DIR = "/path/to/turtlebot_depth_images_static"
RGB_DYNAMIC_DIR = "/path/to/turtlebot_rgb_images_dynamic"
DEPTH_DYNAMIC_DIR = "/path/to/turtlebot_depth_images_dynamic"
TRAINING_RESUTS_TXT = "/path/to/models/training_results.txt"

# Hyperparameters
ACTIONS = 7  # Number of valid actions
GAMMA = 0.99  # Discount factor
BATCH_SIZE = 16  # Batch size
MEMORY_SIZE = 5000  # Replay memory size
LEARNING_RATE = 1e-4  # Learning rate
TAU = 0.01  # Target network update rate
MAX_EPISODES = 20

IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS = 128, 160, 4

# Load a single image (RGB or Depth)
def load_image(file_path, target_size):
    if "rgb" in file_path:
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)  # RGB
        image = cv2.resize(image, target_size)
    else:
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Depth
        image = cv2.resize(image, target_size)
        image = image[..., np.newaxis]  # 2D -> 3D (128, 160 -> 128, 160, 1)
    #print(file_path)
    return image / 255.0


# Generator to load dataset in batches
def data_generator(rgb_dir, depth_dir, batch_size):
    rgb_files = natsorted(os.listdir(rgb_dir))
    depth_files = natsorted(os.listdir(depth_dir))
    total_files = len(rgb_files)

    for i in range(0, total_files, batch_size):
        batch_rgb = []
        batch_depth = []
        for j in range(i, min(i + batch_size, total_files)):
            rgb_path = os.path.join(rgb_dir, rgb_files[j])
            depth_path = os.path.join(depth_dir, depth_files[j])
            batch_rgb.append(load_image(rgb_path, (IMAGE_WIDTH, IMAGE_HEIGHT)))
            batch_depth.append(load_image(depth_path, (IMAGE_WIDTH, IMAGE_HEIGHT)))
        yield np.array(batch_rgb), np.array(batch_depth)


# Initialize D3QN
online_model = DuelingDQN(ACTIONS)
target_model = DuelingDQN(ACTIONS)
optimizer = Adam(learning_rate=LEARNING_RATE)
target_model.set_weights(online_model.get_weights())


# Training step
def train_step(states, actions, rewards, next_states, dones):
    with tf.GradientTape() as tape:
        q_values = online_model(states)  # (BATCH_SIZE, ACTIONS)
        next_q_values = target_model(next_states)  # (BATCH_SIZE, ACTIONS)

        # Q-value update
        q_update = rewards + GAMMA * tf.reduce_max(next_q_values, axis=1) * (1 - dones)
        q_update = tf.stop_gradient(q_update)

        # Correctly index Q-values for actions taken
        batch_indices = tf.range(BATCH_SIZE, dtype=tf.int32)  # [0, 1, ..., BATCH_SIZE-1]
        indices = tf.stack([batch_indices, actions], axis=1)  # Shape: (BATCH_SIZE, 2)
        chosen_q_values = tf.gather_nd(q_values, indices)  # (BATCH_SIZE,)

        # Loss calculation
        loss = tf.reduce_mean(tf.square(q_update - chosen_q_values))

    # Apply gradients
    grads = tape.gradient(loss, online_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, online_model.trainable_variables))
    return loss


# Replay memory
memory = deque(maxlen=MEMORY_SIZE)


def add_to_memory(state, action, reward, next_state, done):
    action = np.clip(action, 0, ACTIONS - 1)  # limit action to valid range
    memory.append((state, action, reward, next_state, done))


# Main training loop
def train_d3qn():
    with open(TRAINING_RESUTS_TXT, "w") as log_file:
        for episode in range(MAX_EPISODES):
            # Dataset order: static, dynamic
            dataset_type = "static" if episode % 2 == 0 else "dynamic"
            print(f"=======Dataset type: {dataset_type}")

            if dataset_type == "static":
                rgb_dir = RGB_STATIC_DIR
                depth_dir = DEPTH_STATIC_DIR
                max_steps = len(os.listdir(RGB_STATIC_DIR))
            else:
                rgb_dir = RGB_DYNAMIC_DIR
                depth_dir = DEPTH_DYNAMIC_DIR
                max_steps = len(os.listdir(RGB_DYNAMIC_DIR))

            # Initialize state
            generator = data_generator(rgb_dir, depth_dir, batch_size=1)
            try:
                rgb_image, depth_image = next(generator)
            except StopIteration:
                # workaround : If generator ends, restart it
                generator = data_generator(rgb_dir, depth_dir, batch_size=1)
                rgb_image, depth_image = next(generator)

            state = np.concatenate((depth_image[0], rgb_image[0]), axis=-1)  # (128, 160, 4)

            total_reward = 0

            for step in range(max_steps):
                #print("episode : ", episode)
                #print("step : ", step)
                #print(f"episode: {episode}, step: {step}")
                # Choose random action (replace with epsilon-greedy policy for advanced usage)
                action = np.random.randint(0, ACTIONS)
                # Simulate reward and next state
                reward = np.random.random()  # Placeholder reward
                done = step == (max_steps - 1)  # Simulated done condition

                try:
                    next_rgb_image, next_depth_image = next(generator)
                except StopIteration:
                    # workaround : If generator ends, restart it
                    generator = data_generator(rgb_dir, depth_dir, batch_size=1)
                    next_rgb_image, next_depth_image = next(generator)

                next_state = np.concatenate((next_depth_image[0], next_rgb_image[0]), axis=-1)

                # Add to memory
                add_to_memory(state, action, reward, next_state, done)
                state = next_state

                # Train when memory is sufficient
                if len(memory) > BATCH_SIZE:
                    batch = random.sample(memory, BATCH_SIZE)
                    states, actions, rewards, next_states, dones = zip(*batch)
                    states = np.array(states)
                    next_states = np.array(next_states)
                    actions = np.array(actions)
                    rewards = np.array(rewards)
                    dones = np.array(dones).astype(np.float32)
                    loss = train_step(states, actions, rewards, next_states, dones)

                total_reward += reward
            
            if (episode + 1) % 2 == 0:
                online_model.save(f"/path/to/models/dqn_model_episode_{episode + 1}.keras")
            log_string = f"Episode {episode + 1}/{MAX_EPISODES}, Total Reward: {total_reward:.2f}, Loss: {loss:.4f}\n"
            log_file.write(log_string)
            print(log_string)

            # Update target model weights
            target_model.set_weights(
                [TAU * online + (1 - TAU) * target for online, target in zip(online_model.get_weights(), target_model.get_weights())]
            )


if __name__ == "__main__":
    train_d3qn()
