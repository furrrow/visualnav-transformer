"""ROS2 version of create_topomap.py that samples images from a topic and saves them to a topomap directory."""

import argparse
import os
import shutil
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Joy

from utils import msg_to_pil

DEFAULT_IMAGE_TOPIC = "/camera/image_raw"
TOPOMAP_IMAGES_DIR = "../topomaps/images"


def remove_files_in_dir(dir_path: str):
    for f in os.listdir(dir_path):
        file_path = os.path.join(dir_path, f)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


class TopomapRecorder(Node):
    def __init__(self, image_topic: str, dt: float, topomap_name: str):
        super().__init__("create_topomap_ros2")
        self.image_topic = image_topic
        self.dt = dt
        self.topomap_dir = os.path.join(TOPOMAP_IMAGES_DIR, topomap_name)

        if not os.path.isdir(self.topomap_dir):
            os.makedirs(self.topomap_dir)
        else:
            self.get_logger().info(
                f"{self.topomap_dir} already exists. Removing previous images..."
            )
            remove_files_in_dir(self.topomap_dir)

        self.obs_img = None
        self.last_msg_time: Optional[float] = None
        self.image_sub = self.create_subscription(
            Image, image_topic, self.callback_obs, 10
        )
        self.joy_sub = self.create_subscription(Joy, "joy", self.callback_joy, 10)
        self.timer = self.create_timer(self.dt, self.save_image_tick)
        self.idx = 0

        self.get_logger().info(
            f"Listening to {self.image_topic}; saving every {self.dt}s to {self.topomap_dir}"
        )

    def callback_obs(self, msg: Image):
        self.obs_img = msg_to_pil(msg)
        self.last_msg_time = time.time()

    def callback_joy(self, msg: Joy):
        if msg.buttons and msg.buttons[0]:
            self.get_logger().info("Joystick stop requested; shutting down.")
            rclpy.shutdown()

    def save_image_tick(self):
        if self.obs_img is not None:
            save_path = os.path.join(self.topomap_dir, f"{self.idx}.png")
            self.obs_img.save(save_path)
            self.get_logger().info(f"Saved image {self.idx}")
            self.idx += 1
            self.obs_img = None
        elif self.last_msg_time and (time.time() - self.last_msg_time) > (2 * self.dt):
            self.get_logger().info(
                f"Topic {self.image_topic} not publishing; shutting down."
            )
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Generate topomaps from an image topic (ROS2)"
    )
    parser.add_argument(
        "--dir",
        "-d",
        default="topomap",
        type=str,
        help=f"path to topological map images in {TOPOMAP_IMAGES_DIR} (default: topomap)",
    )
    parser.add_argument(
        "--dt",
        "-t",
        default=1.0,
        type=float,
        help="time between images sampled from the image topic (default: 1.0)",
    )
    parser.add_argument(
        "--image-topic",
        default=DEFAULT_IMAGE_TOPIC,
        type=str,
        help=f"image topic to subscribe to (default: {DEFAULT_IMAGE_TOPIC})",
    )
    args = parser.parse_args()

    rclpy.init()
    node = TopomapRecorder(args.image_topic, args.dt, args.dir)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
