"""ROS2 version of navigate.py to run navigation over a prebuilt topomap."""

import argparse
import os
import time
from typing import List

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
import torch
import yaml
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from PIL import Image as PILImage

from utils import msg_to_pil, to_numpy, transform_images, load_model
from vint_train.training.train_utils import get_action
from topic_names import (
    IMAGE_TOPIC as DEFAULT_IMAGE_TOPIC,
    WAYPOINT_TOPIC as DEFAULT_WAYPOINT_TOPIC,
    SAMPLED_ACTIONS_TOPIC,
    REACHED_GOAL_TOPIC,
)

# Paths/constants
TOPOMAP_IMAGES_DIR = "../topomaps/images"
MODEL_WEIGHTS_PATH = "../model_weights"
ROBOT_CONFIG_PATH = "../config/robot.yaml"
MODEL_CONFIG_PATH = "../config/models.yaml"


def load_robot_params():
    with open(ROBOT_CONFIG_PATH, "r") as f:
        robot_config = yaml.safe_load(f)
    return robot_config["max_v"], robot_config["max_w"], robot_config["frame_rate"]


class Navigator(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("navigator_ros2")
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        self.max_v, self.max_w, self.frame_rate = load_robot_params()

        # Load model config/weights
        with open(MODEL_CONFIG_PATH, "r") as f:
            model_paths = yaml.safe_load(f)
        model_config_path = model_paths[args.model]["config_path"]
        with open(model_config_path, "r") as f:
            self.model_params = yaml.safe_load(f)
        self.context_size = self.model_params["context_size"]

        ckpt_path = model_paths[args.model]["ckpt_path"]
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model weights not found at {ckpt_path}")
        self.get_logger().info(f"Loading model from {ckpt_path}")
        self.model = load_model(ckpt_path, self.model_params, self.device).to(self.device)
        self.model.eval()

        # Load topomap images
        topomap_dir = os.path.join(TOPOMAP_IMAGES_DIR, args.dir)
        topomap_filenames = sorted(
            os.listdir(topomap_dir), key=lambda x: int(x.split(".")[0])
        )
        self.topomap: List[PILImage.Image] = []
        for fname in topomap_filenames:
            self.topomap.append(PILImage.open(os.path.join(topomap_dir, fname)))

        self.closest_node = 0
        assert -1 <= args.goal_node < len(self.topomap), "Invalid goal index"
        self.goal_node = len(self.topomap) - 1 if args.goal_node == -1 else args.goal_node

        # ROS2 comms
        self.context_queue: List[PILImage.Image] = []
        self.image_sub = self.create_subscription(
            Image, args.image_topic, self.callback_obs, 1
        )
        self.waypoint_pub = self.create_publisher(Float32MultiArray, args.waypoint_topic, 1)
        self.sampled_actions_pub = self.create_publisher(
            Float32MultiArray, SAMPLED_ACTIONS_TOPIC, 1
        )
        self.goal_pub = self.create_publisher(Bool, REACHED_GOAL_TOPIC, 1)

        # Timers
        self.timer = self.create_timer(1.0 / self.frame_rate, self.navigate_step)

        # Diffusion scheduler cache (for NoMaD)
        if self.model_params["model_type"] == "nomad":
            self.num_diffusion_iters = self.model_params["num_diffusion_iters"]
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.model_params["num_diffusion_iters"],
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                prediction_type="epsilon",
            )

        self.get_logger().info(
            f"Navigator ready; waiting for images on {args.image_topic}. Goal node: {self.goal_node}"
        )

    def callback_obs(self, msg: Image):
        obs_img = msg_to_pil(msg)
        if len(self.context_queue) < self.context_size + 1:
            self.context_queue.append(obs_img)
        else:
            self.context_queue.pop(0)
            self.context_queue.append(obs_img)

    def navigate_step(self):
        if len(self.context_queue) <= self.context_size:
            return

        chosen_waypoint = np.zeros(4)

        if self.model_params["model_type"] == "nomad":
            obs_images = transform_images(
                self.context_queue, self.model_params["image_size"], center_crop=False
            )
            obs_images = torch.split(obs_images, 3, dim=1)
            obs_images = torch.cat(obs_images, dim=1)
            obs_images = obs_images.to(self.device)
            mask = torch.zeros(1).long().to(self.device)

            start = max(self.closest_node - self.args.radius, 0)
            end = min(self.closest_node + self.args.radius + 1, self.goal_node)
            goal_image = [
                transform_images(g_img, self.model_params["image_size"], center_crop=False).to(
                    self.device
                )
                for g_img in self.topomap[start : end + 1]
            ]
            goal_image = torch.concat(goal_image, dim=0)

            obsgoal_cond = self.model(
                "vision_encoder",
                obs_img=obs_images.repeat(len(goal_image), 1, 1, 1),
                goal_img=goal_image,
                input_goal_mask=mask.repeat(len(goal_image)),
            )
            dists = self.model("dist_pred_net", obsgoal_cond=obsgoal_cond)
            dists = to_numpy(dists.flatten())
            min_idx = np.argmin(dists)
            self.closest_node = min_idx + start
            sg_idx = min(min_idx + int(dists[min_idx] < self.args.close_threshold), len(obsgoal_cond) - 1)
            obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)

            with torch.no_grad():
                if len(obs_cond.shape) == 2:
                    obs_cond = obs_cond.repeat(self.args.num_samples, 1)
                else:
                    obs_cond = obs_cond.repeat(self.args.num_samples, 1, 1)

                noisy_action = torch.randn(
                    (self.args.num_samples, self.model_params["len_traj_pred"], 2), device=self.device
                )
                naction = noisy_action
                self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
                for k in self.noise_scheduler.timesteps[:]:
                    noise_pred = self.model(
                        "noise_pred_net", sample=naction, timestep=k, global_cond=obs_cond
                    )
                    naction = self.noise_scheduler.step(
                        model_output=noise_pred, timestep=k, sample=naction
                    ).prev_sample

            naction = to_numpy(get_action(naction))
            sampled_actions_msg = Float32MultiArray()
            sampled_actions_msg.data = np.concatenate((np.array([0]), naction.flatten()))
            self.sampled_actions_pub.publish(sampled_actions_msg)
            naction = naction[0]
            chosen_waypoint = naction[self.args.waypoint]
        else:
            start = max(self.closest_node - self.args.radius, 0)
            end = min(self.closest_node + self.args.radius + 1, self.goal_node)
            batch_obs_imgs = []
            batch_goal_data = []
            for sg_img in self.topomap[start : end + 1]:
                transf_obs_img = transform_images(
                    self.context_queue, self.model_params["image_size"]
                )
                goal_data = transform_images(sg_img, self.model_params["image_size"])
                batch_obs_imgs.append(transf_obs_img)
                batch_goal_data.append(goal_data)

            batch_obs_imgs = torch.cat(batch_obs_imgs, dim=0).to(self.device)
            batch_goal_data = torch.cat(batch_goal_data, dim=0).to(self.device)

            distances, waypoints = self.model(batch_obs_imgs, batch_goal_data)
            distances = to_numpy(distances)
            waypoints = to_numpy(waypoints)
            min_dist_idx = np.argmin(distances)
            if distances[min_dist_idx] > self.args.close_threshold:
                chosen_waypoint = waypoints[min_dist_idx][self.args.waypoint]
                self.closest_node = start + min_dist_idx
            else:
                chosen_waypoint = waypoints[min(min_dist_idx + 1, len(waypoints) - 1)][
                    self.args.waypoint
                ]
                self.closest_node = min(start + min_dist_idx + 1, self.goal_node)

        if self.model_params.get("normalize", False):
            chosen_waypoint[:2] *= (self.max_v / self.frame_rate)

        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = chosen_waypoint
        self.waypoint_pub.publish(waypoint_msg)

        reached_goal = self.closest_node == self.goal_node
        self.goal_pub.publish(Bool(data=reached_goal))
        if reached_goal:
            self.get_logger().info("Reached goal! Stopping...")
            # Optionally stop publishing by shutting down
            # rclpy.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROS2 navigation over a topomap")
    parser.add_argument(
        "--model",
        "-m",
        default="nomad",
        type=str,
        help="model name (check ../config/models.yaml)",
    )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2,
        type=int,
        help="index of the waypoint used for navigation",
    )
    parser.add_argument(
        "--dir",
        "-d",
        default="topomap",
        type=str,
        help="path to topomap images",
    )
    parser.add_argument(
        "--goal-node",
        "-g",
        default=-1,
        type=int,
        help="goal node index (-1 uses last node)",
    )
    parser.add_argument(
        "--close-threshold",
        "-t",
        default=3,
        type=int,
        help="distance threshold to switch to next node",
    )
    parser.add_argument(
        "--radius",
        "-r",
        default=4,
        type=int,
        help="temporal radius of nodes to consider for localization",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=8,
        type=int,
        help="Number of actions sampled from the exploration model",
    )
    parser.add_argument(
        "--image-topic",
        default=DEFAULT_IMAGE_TOPIC,
        type=str,
        help=f"image topic to subscribe to (default: {DEFAULT_IMAGE_TOPIC})",
    )
    parser.add_argument(
        "--waypoint-topic",
        default=DEFAULT_WAYPOINT_TOPIC,
        type=str,
        help=f"waypoint topic to publish to (default: {DEFAULT_WAYPOINT_TOPIC})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = Navigator(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
