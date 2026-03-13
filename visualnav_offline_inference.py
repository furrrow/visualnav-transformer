import argparse
import os
import time

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
matplotlib.use("TkAgg")
import yaml

import pickle
from PIL import Image as PILImage
import argparse
import torchdiffeq
from pathlib import Path
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# Custom Imports
from deployment.src.offline_utils import plot_trajs_and_points
from deployment.src.offline_utils import (to_numpy, transform_images, load_model,
                           load_calibration, overlay_path, get_action)
from deployment.src.offline_utils import RGB_color_dict as color_dict
"""
offline_inference.py
custom inference script to test out visualnav,
"""

# CONSTANTS
TOPOMAP_IMAGES_DIR = "/home/jim/Projects/prune/deployment/topomaps/images"
CAMERA_MATRIX_DIR = "/home/jim/Projects/prune/deployment/camera_matrix.json"
ROBOT_CONFIG_PATH ="./deployment/config/robot.yaml"
MODEL_CONFIG_PATH = "./deployment/config/models.yaml"

model_paths = {
        "vint" : {
            "config_path": "./train/config/vint.yaml",
            "ckpt_path": "./deployment/model_weights/vint.pth",
        },
        "gnm": {
            "config_path": "./train/config/gnm.yaml",
            "ckpt_path": "./deployment/model_weights/gnm.pth",
        },
        "nomad": {
            "config_path": "./train/config/nomad.yaml",
            "ckpt_path": "./deployment/model_weights/nomad.pth",
        },
    }

with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"]

def main(config: dict) -> None:
    # Set up the device
    if torch.cuda.is_available():
        gpu_id = "0"
    device = torch.device(
        f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    # load model parameters
    model_config_path = model_paths[args.model]["config_path"]
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

    # load model weights
    ckpth_path = model_paths[args.model]["ckpt_path"]
    if os.path.exists(ckpth_path):
        print(f"Loading model from {ckpth_path}")
    else:
        raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
    model = load_model(
        ckpth_path,
        model_params,
        device,
    )
    model = model.to(device)
    model.eval()

    # load topomap
    topomap_filenames = sorted(os.listdir(os.path.join(
        TOPOMAP_IMAGES_DIR, args.dir)), key=lambda x: int(x.split(".")[0]))
    topomap_dir = f"{TOPOMAP_IMAGES_DIR}/{args.dir}"
    num_nodes = len(os.listdir(topomap_dir))
    topomap = []
    for i in range(num_nodes):
        image_path = os.path.join(topomap_dir, topomap_filenames[i])
        topomap.append(PILImage.open(image_path))

    context_size = model_params["context_size"]
    context_queue = topomap[:context_size+1]
    # if len(context_queue) < self.context_size + 1:
    #     self.context_queue.append(self.obs_img)
    # else:
    #     self.context_queue.pop(0)
    #     self.context_queue.append(self.obs_img)

    cam_matrix, dist_coeffs, T_base_from_cam = load_calibration(CAMERA_MATRIX_DIR)
    T_cam_from_base = np.linalg.inv(T_base_from_cam)

    if model_params["model_type"] == "nomad":
        num_diffusion_iters = model_params["num_diffusion_iters"]
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=model_params["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

    # run_navigation_loop, once.
    chosen_waypoint = np.zeros(4)
    obs_images = transform_images(context_queue, model_params["image_size"], center_crop=False)
    obs_images = torch.split(obs_images, 3, dim=1)
    obs_images = torch.cat(obs_images, dim=1)
    obs_images = obs_images.to(device) # [1, 15, 96, 96]
    # Definition of the goal mask (convention: 0 = no mask, 1 = mask)
    one_mask = torch.ones(1).long().to(device)
    no_mask = torch.zeros(1).long().to(device)

    closest_node = 0
    assert -1 <= args.goal_node < len(topomap), "Invalid goal index"
    if args.goal_node == -1:
        goal_node = len(topomap) - 1
    else:
        goal_node = args.goal_node
    start = max(closest_node - args.radius, 0)
    end = min(closest_node + args.radius + 1, goal_node)
    goal_image = [transform_images(g_img, model_params["image_size"], center_crop=False).to(device) for g_img in
                  topomap[start:end + 1]]
    goal_image = torch.concat(goal_image, dim=0)

    # navigation
    obsgoal_cond = model('vision_encoder', obs_img=obs_images.repeat(len(goal_image), 1, 1, 1),
                              goal_img=goal_image, input_goal_mask=no_mask.repeat(len(goal_image)))
    dists = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
    dists = to_numpy(dists.flatten())

    # exploration
    obs_cond = model('vision_encoder', obs_img=obs_images.repeat(len(goal_image), 1, 1, 1),
                         goal_img=goal_image, input_goal_mask=one_mask.repeat(len(goal_image)))
    min_idx = np.argmin(dists)
    closest_node = min_idx + start

    # infer action
    with torch.no_grad():
        start_time = time.time()
        obs_cond = obs_cond[
            min(min_idx + int(dists[min_idx] < args.close_threshold), len(obsgoal_cond) - 1)].unsqueeze(0)
        obsgoal_cond = obsgoal_cond[
            min(min_idx + int(dists[min_idx] < args.close_threshold), len(obsgoal_cond) - 1)].unsqueeze(0)

        if len(obs_cond.shape) == 2:
            obs_cond = obs_cond.repeat(args.num_samples, 1)
            obsgoal_cond = obsgoal_cond.repeat(args.num_samples, 1)
        else:
            obs_cond = obs_cond.repeat(args.num_samples, 1, 1)
            obsgoal_cond = obsgoal_cond.repeat(args.num_samples, 1, 1)

        # Navigation
        noisy_action = torch.randn(
            (args.num_samples, model_params["len_traj_pred"], 2), device=device)
        naction = noisy_action

        # init scheduler
        noise_scheduler.set_timesteps(num_diffusion_iters)
        start_time = time.time()
        for k in noise_scheduler.timesteps[:]:
            # predict noise
            noise_pred = model(
                'noise_pred_net',
                sample=naction,
                timestep=k,
                global_cond=obs_cond
            )
            # inverse diffusion step (remove noise)
            naction = noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample
        print("time elapsed:", time.time() - start_time)

        # proc_time = time.time() - start_time
        # mean_proc_time = proc_time / noisy_action.shape[0]
        # print("Mean Processing Time UC", mean_proc_time)
        # print("Processing Time UC", proc_time)

        gc_actions = to_numpy(get_action(naction))
        message_data = np.concatenate((np.array([0]), gc_actions.flatten()))
        # sampled_actions_msg.data = message_data.tolist()
        # sampled_actions_pub.publish(sampled_actions_msg)
        # print("sampled_actions_msg", message_data)
        current_action = gc_actions[0]
        chosen_waypoint = current_action[args.waypoint]

    # plot distribution:
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 3, figure=fig)
    fig.suptitle(f"trajectory visualization with {args.model}")
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[:, 1:])

    # uc_actions = list(uc_actions)
    gc_actions = list(gc_actions)
    action_label = gc_actions[0]
    traj_list = np.concatenate(
        [
            gc_actions,
            action_label[None],
        ],
        axis=0,
    ) # [17,8,2]
    traj_colors = (
            ["green"] * len(gc_actions) + ["magenta"]
    )
    mock_goal_pos = np.array([10, 0])
    traj_alphas = [0.1] * (len(gc_actions)) + [1.0]
    point_list = [np.array([0, 0]), torch.Tensor(mock_goal_pos)]
    point_colors = ["green", "red"]
    point_alphas = [1.0, 1.0]
    plot_trajs_and_points(
        ax=ax00,
        list_trajs=traj_list,
        list_points=point_list,
        traj_colors=traj_colors,
        point_colors=point_colors,
        traj_labels=None,
        point_labels=None,
        quiver_freq=0,
        traj_alphas=traj_alphas,
        point_alphas=point_alphas,
    )
    obs_image = np.array(context_queue[0])
    goal_image = np.array(topomap[-1])
    # obs_image = np.moveaxis(obs_image, 0, -1)
    # goal_image = np.moveaxis(goal_image, 0, -1)
    obs_image = overlay_path(np.array(gc_actions[1:]), obs_image, cam_matrix, T_cam_from_base, color_dict['GREEN'])
    obs_image = overlay_path(np.array(gc_actions[0]), obs_image, cam_matrix, T_cam_from_base, color_dict['BLUE'])
    ax11.imshow(obs_image)
    ax00.set_title("action predictions \n green goal, magenta best_path red notional goal")
    ax11.set_title("observation, blue best path")

    ax01.imshow(goal_image)
    ax01.set_title(f"goal")
    plt.show()


    # waypoint_msg = Float32MultiArray()
    waypoint_msg = chosen_waypoint.flatten().tolist()
    # waypoint_msg.data = chosen_waypoint.flatten().tolist()
    # waypoint_pub.publish(waypoint_msg)
    print("goal reached message", waypoint_msg)

    print(f"CHOSEN WAYPOINT: {chosen_waypoint}")

    reached_goal = closest_node == goal_node
    # goal_reached_msg = Bool()
    # goal_reached_msg.data = bool(reached_goal)
    # goal_pub.publish(goal_reached_msg)
    print("goal reached message")

    if reached_goal:
        print("Reached goal! Stopping...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Offline script to run flownav")
    # Parse command line arguments
    parser.add_argument(
        "--model",
        "-m",
        default="nomad",
        type=str,
        help="Model to run: Only nomad is supported currently",
    )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2,  # close waypoints exihibit straight line motion (the middle waypoint is a good default)
        type=int,
        help=f"""index of the waypoint used for navigation (between 0 and 4 or 
            how many waypoints your model predicts) (default: 2)""",
    )
    parser.add_argument(
        "--dir",
        "-d",
        default="antonov",
        type=str,
        help="folder title in the topomap image folder",
    )
    parser.add_argument(
        "--goal-node",
        "-g",
        default=-1,
        type=int,
        help="""goal node index in the topomap (if -1, then the goal node is 
            the last node in the topomap) (default: -1)""",
    )
    parser.add_argument(
        "--close-threshold",
        "-t",
        default=3,
        type=int,
        help="""temporal distance within the next node in the topomap before 
            localizing to it (default: 3)""",
    )
    parser.add_argument(
        "--radius",
        "-r",
        default=4,
        type=int,
        help="""temporal number of locobal nodes to look at in the topopmap for
            localization (default: 2)""",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=8,
        type=int,
        help=f"Number of actions sampled from the exploration model (default: 8)",
    )

    args = parser.parse_args()
    main(args)

