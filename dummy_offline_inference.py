import argparse
import os
import time

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
from visualnav_inference_point_based import RewardInferenceRunner
from frechetdist import frdist
"""
offline_inference.py
custom inference script to test out visualnav,
"""

# CONSTANTS
TOPOMAP_IMAGES_DIR = "/home/jim/Projects/prune/deployment/topomaps/images"
CAMERA_MATRIX_DIR = "/home/jim/Projects/prune/deployment/camera_matrix.json"
# TOPOMAP_IMAGES_DIR = "/workspace/prune/deployment/topomaps/images"
# CAMERA_MATRIX_DIR = "/workspace/prune/deployment/camera_matrix.json"
ROBOT_CONFIG_PATH ="./deployment/config/robot.yaml"
MODEL_CONFIG_PATH = "./deployment/config/models.yaml"

dummy_actions = np.array(
[[[ 6.11471534e-01,  2.28075981e-02],
  [ 1.23913908e+00, -1.16825104e-04],
  [ 1.88055801e+00,  1.61237717e-02],
  [ 2.54769731e+00,  1.59904957e-02],
  [ 3.22852659e+00,  4.91845608e-02],
  [ 3.91691875e+00,  8.81326199e-02],
  [ 4.63910770e+00,  1.69375181e-01],
  [ 5.36809731e+00,  2.63122320e-01]],
 [[ 6.08049929e-01, -9.69552994e-03],
  [ 1.22131133e+00,  2.52723694e-05],
  [ 1.84367132e+00,  4.32252884e-03],
  [ 2.46937156e+00,  2.21924782e-02],
  [ 3.10458779e+00,  6.04500771e-02],
  [ 3.73051548e+00,  9.94310379e-02],
  [ 4.37741470e+00,  1.61967278e-01],
  [ 5.02405787e+00,  2.70364285e-01]],
 [[ 6.27055168e-01,  1.87706947e-02],
  [ 1.27188396e+00,  4.26325798e-02],
  [ 1.94885838e+00,  5.13057709e-02],
  [ 2.62499022e+00,  1.08020306e-01],
  [ 3.28061485e+00,  1.75201416e-01],
  [ 3.88464141e+00,  2.45447159e-01],
  [ 4.46694565e+00,  3.64558697e-01],
  [ 5.04016972e+00,  4.53863144e-01]],
 [[ 7.48751044e-01,  2.71315575e-02],
  [ 1.46656895e+00,  2.18515396e-02],
  [ 2.17804003e+00,  2.13041306e-02],
  [ 2.84093666e+00,  1.53980255e-02],
  [ 3.48469210e+00, -3.32434177e-02],
  [ 4.10397577e+00, -6.42757416e-02],
  [ 4.72613287e+00, -7.59413242e-02],
  [ 5.34765816e+00, -6.77120686e-02]],
 [[ 7.04163313e-01,  8.03565979e-03],
  [ 1.37333655e+00,  1.64804459e-02],
  [ 2.02007008e+00,  1.47776604e-02],
  [ 2.65990758e+00,  3.05585861e-02],
  [ 3.32704902e+00,  7.10954666e-02],
  [ 4.02168560e+00,  1.30295277e-01],
  [ 4.74540377e+00,  2.28003979e-01],
  [ 5.47259521e+00,  3.42406750e-01]],
 [[ 4.75286603e-01,  5.64765930e-03],
  [ 9.55439866e-01,  7.57312775e-03],
  [ 1.45891404e+00,  2.10742950e-02],
  [ 1.93941021e+00,  8.06713104e-02],
  [ 2.42008090e+00,  2.09338665e-01],
  [ 2.92968440e+00,  3.98213863e-01],
  [ 3.48213840e+00,  6.73923492e-01],
  [ 4.01696014e+00,  9.91996288e-01]],
 [[ 6.49355054e-01,  8.19349289e-03],
  [ 1.29916918e+00, -2.03967094e-03],
  [ 1.97712350e+00, -1.64127350e-02],
  [ 2.67512894e+00, -9.10854340e-03],
  [ 3.38129783e+00,  1.20463371e-02],
  [ 4.08308458e+00,  4.63843346e-02],
  [ 4.78140974e+00,  1.31647110e-01],
  [ 5.46730661e+00,  2.31839657e-01]],
 [[ 7.77929425e-01, -1.30581856e-03],
  [ 1.57163489e+00, -1.95932388e-03],
  [ 2.38158727e+00, -2.35500336e-02],
  [ 3.19476151e+00, -5.76543808e-03],
  [ 4.01301718e+00,  1.13821030e-02],
  [ 4.79053736e+00,  8.51106644e-02],
  [ 5.53158665e+00,  1.99385643e-01],
  [ 6.23126841e+00,  3.66864204e-01]]])

def generate_trajectory(curvature=0.0, num_points=8, step=1.0):
    """
    curvature:
        >0 : curve left
        <0 : curve right
         0 : straight
    """
    traj = []

    for i in range(num_points):
        x = i * step + 0.5

        # quadratic lateral offset
        y = curvature * (i ** 2)

        traj.append([round(x, 3), round(y, 3)])

    return traj


def make_video_writer(video_path: str, fps: float):
    suffix = Path(video_path).suffix.lower()
    if suffix == ".gif":
        return animation.PillowWriter(fps=fps), video_path

    if animation.writers.is_available("ffmpeg"):
        return animation.FFMpegWriter(fps=fps), video_path

    fallback_path = str(Path(video_path).with_suffix(".gif"))
    print(f"ffmpeg is not available; saving GIF instead: {fallback_path}")
    return animation.PillowWriter(fps=fps), fallback_path

def resample_path_2d(path: np.ndarray, k: int) -> np.ndarray:
    """
    Evenly resample a sequence of 2D points to length k using linear interpolation.
    Expects ``path`` shape (n, 2); returns float32 array shape (k, 2).
    """
    if path.size == 0:
        return np.zeros((k, 2), dtype=np.float32)
    if len(path) == 1:
        return np.repeat(path, k, axis=0)
    deltas = path[1:] - path[:-1]
    seg_len = np.linalg.norm(deltas, axis=1)
    cum = np.concatenate([np.array([0.0], dtype=np.float32), np.cumsum(seg_len, dtype=np.float32)])
    total = cum[-1]
    if total == 0:
        return np.repeat(path[:1], k, axis=0)

    target = np.linspace(0.0, float(total), num=k, dtype=np.float32)
    out = np.empty((k, path.shape[1]), dtype=np.float32)
    for i, t in enumerate(target):
        j = np.searchsorted(cum, t, side="right") - 1
        j = int(np.clip(j, 0, len(seg_len) - 1))
        t0, t1 = cum[j], cum[j + 1]
        alpha = 0.0 if t1 == t0 else float((t - t0) / (t1 - t0))
        out[i] = path[j] * (1 - alpha) + path[j + 1] * alpha
    return out

def prune_distance(points: np.ndarray, cutoff: float, n_paths=8, k: int=8):
    """

    :param points: trajectories of xy points, (num_trajectories, traj_length, 2)
    :param cutoff: a distance measure
    :return: updated trajectory points
    """
    paths = np.array(points, dtype=np.float32) # (8, 8, 2)
    n_pts, pt_len, _ = paths.shape
    first_points = np.expand_dims(paths[:, 0, :], 1).repeat(pt_len, axis=1)
    deltas = paths - first_points
    seg_len = np.linalg.norm(deltas, axis=-1)
    path_check = seg_len > cutoff
    paths = paths[:n_paths]
    for i, i_path in enumerate(paths):
        if not True in path_check[i]:
            continue
        sub_path = i_path[:np.argmax(path_check[i])]
        paths[i] = resample_path_2d(sub_path, k)
    return paths

model_paths = {
        "vint" : {
            "config_path": "./train/config/vint.yaml",
            "ckpt_path": "./weights/vint.pth",
        },
        "gnm": {
            "config_path": "./train/config/gnm.yaml",
            "ckpt_path": "./weights/gnm.pth",
        },
        "nomad": {
            "config_path": "./train/config/nomad.yaml",
            "ckpt_path": "./weights/nomad.pth",
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
    exp_dir = args.exp_dir
    os.makedirs(exp_dir, exist_ok=True)

    if args.steer:
        # Reward model
        # rm_ckpt_path = "./weights/epoch_029.pt"
        # rm_ckpt_path = "../../weights/model_150_epoch_34.pth"
        rm_ckpt_path = "../../weights/model_151_epoch_22.pth"
        # rm_config_path = "/home/jim/Projects/prune/config/config_point_based.yaml"
        rm_config_path = "/home/jim/Projects/prune/config/setting.yaml"
        reward_runner = RewardInferenceRunner(checkpoint_path=rm_ckpt_path, config_path=rm_config_path, verbose=True)

    distance_cutoff = 2.5 # won't consider paths beyond this distance when doing steering
    ckpt_path = model_paths[args.model]["ckpt_path"]
    if os.path.exists(ckpt_path):
        print(f"Loading model from {ckpt_path}")
    else:
        raise FileNotFoundError(f"Model weights not found at {ckpt_path}")
    cur_exp_dir = f"{exp_dir}/{args.model}_{args.dir}_{args.goal_node}_{distance_cutoff}"
    if args.steer:
        cur_exp_dir = f"{exp_dir}/{args.model}_{Path(rm_ckpt_path).name}_{args.dir}_{args.goal_node}_{distance_cutoff}"
    os.makedirs(cur_exp_dir, exist_ok=True)

    cur_exp_im_dir = f"{cur_exp_dir}/images"
    os.makedirs(cur_exp_im_dir, exist_ok=True)

    cur_exp_pkl_dir = f"{cur_exp_dir}/pkl"
    os.makedirs(cur_exp_pkl_dir, exist_ok=True)

    video_path = args.video_path
    if video_path is None:
        video_path = os.path.join(cur_exp_dir, "navigation.mp4")
    video_parent = os.path.dirname(video_path)
    if video_parent:
        os.makedirs(video_parent, exist_ok=True)
    video_writer, video_path = make_video_writer(video_path, args.video_fps)
    # load model parameters
    model_config_path = model_paths[args.model]["config_path"]
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

    if args.mode == "navigate":
        mode = "navigate"
    else:
        mode = "explore"
    # load topomap
    topomap_filenames = sorted(os.listdir(os.path.join(
        TOPOMAP_IMAGES_DIR, args.dir)), key=lambda x: int(x.split(".")[0]))
    topomap_dir = f"{TOPOMAP_IMAGES_DIR}/{args.dir}"
    num_nodes = len(os.listdir(topomap_dir))
    topomap = []
    for i in range(num_nodes):
        image_path = os.path.join(topomap_dir, topomap_filenames[i])
        topomap.append(PILImage.open(image_path))

    cam_matrix, dist_coeffs, T_base_from_cam = load_calibration(CAMERA_MATRIX_DIR)
    T_cam_from_base = np.linalg.inv(T_base_from_cam)

    assert -1 <= args.goal_node < len(topomap), "Invalid goal index"
    if args.goal_node == -1:
        goal_node = len(topomap) - 1
    else:
        goal_node = args.goal_node

    context_size = model_params["context_size"]
    last_start_img = len(topomap) - context_size - 1
    if args.start_img > last_start_img:
        raise ValueError(
            f"start_img={args.start_img} does not leave enough images for "
            f"context_size={context_size}; last valid start is {last_start_img}"
        )
    closest_node = args.start_img
    plt.ion()
    fig = plt.figure(figsize=(16, 8))
    video_writer.setup(fig, video_path, dpi=args.video_dpi)
    for nav_idx, start_img in enumerate(range(args.start_img, last_start_img + 1)):
        context_queue = topomap[start_img:context_size + start_img + 1]
        # context_queue = topomap[0:context_size + 0 + 1]
        rewards = None
        chosen_waypoint = np.zeros(4)
        actions = np.array([
            generate_trajectory(-0.02),
            generate_trajectory(-0.03),
            generate_trajectory(-0.04),
            generate_trajectory(-0.05),
            generate_trajectory(0.0),
            generate_trajectory(0.02),
            generate_trajectory(0.03),
            generate_trajectory(0.04),
            generate_trajectory(0.05),
            # generate_trajectory(0.10),
            # generate_trajectory(0.15),

            # generate_trajectory(-0.10),
            # generate_trajectory(-0.15),
        ])
        actions = dummy_actions
        current_action = actions[0]
        chosen_waypoint = current_action[args.waypoint]
        obs_image = np.array(context_queue[-1])  # not sure which img is the best one to show...

        if args.steer:
            pruned_actions = prune_distance(actions, distance_cutoff, 8)
            # pruned_actions = actions
            image_tensor = torch.from_numpy(obs_image).permute(2, 0, 1).contiguous()  # (3, H, W)
            points_tensor = torch.from_numpy(pruned_actions)  # (M, K, 2)
            rewards = reward_runner.predict_rewards(image_tensor=image_tensor, points_tensor=points_tensor)
            best_action = torch.argmax(rewards).item()
            # print("Predicted rewards:", rewards, "best reward action(red) :", best_action)
            # different distrance metric to make sure selected action does not veer too far:
            eval_dict = {}
            for idx, action in enumerate(pruned_actions):
                eval_dict[idx] = {}
                eval_dict[idx]["reward"] = rewards[idx].item()
                eval_dict[idx]["frdist"] = frdist(action, pruned_actions[0])

        # plot distribution:
        fig.clf()
        gs = GridSpec(2, 3, figure=fig)
        ax00 = fig.add_subplot(gs[0, 0])
        ax01 = fig.add_subplot(gs[1, 0])
        ax11 = fig.add_subplot(gs[:, 1:])

        fig.suptitle(f"trajectory visualization with {args.model} | iteration {nav_idx}")
        actions = list(actions)
        traj_list = np.concatenate([actions], axis=0, )
        traj_list = traj_list[:, :, ::-1] # flip y-x for visualization purposes
        traj_list[:, :, 0] = -traj_list[:, :, 0] # flip x about 0 for visualization purposes
        traj_colors = ["blue"] + ["green"] * (len(actions)-1)
        traj_alphas = [0.75] + [0.1] * (len(actions)-1)
        print("first action:", np.array2string(actions[0][0], precision=1, suppress_small=True))
        if rewards is not None:
            new_traj_list = np.concatenate([pruned_actions], axis=0, )
            new_traj_list = new_traj_list[:, :, ::-1]  # flip y-x for visualization purposes
            new_traj_list[:, :, 0] = -new_traj_list[:, :, 0]  # flip x about 0 for visualization purposes
            traj_list = np.concatenate((traj_list, new_traj_list), axis=0, )
            new_colors = ["blue"] + ["green"] * (len(actions)-1)
            traj_colors[best_action] = "red"
            new_colors[best_action] = "red"
            traj_colors += new_colors
            traj_alphas += [0.5] * len(pruned_actions)

        point_list = [np.array([0, 0])]
        point_colors = ["green", "red"]
        point_alphas = [1.0, 1.0]
        plot_trajs_and_points(
            ax=ax01,
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
        display_goal_image = np.array(topomap[closest_node])
        if args.steer:
            overlay_img = overlay_path(np.array(pruned_actions), obs_image, cam_matrix, T_cam_from_base, color_dict['GREEN'],
                                       metrics=eval_dict)
        else:
            overlay_img = overlay_path(np.array(actions), obs_image, cam_matrix, T_cam_from_base, color_dict['GREEN'])
        if overlay_img is not None:
            ax11.imshow(overlay_img)
        else:
            ax11.imshow(obs_image)

        ax00.imshow(display_goal_image)
        ax00.set_title(f"intermediate goal node {closest_node}")
        ax01.set_title("action predictions")
        ax11.set_title("observation, blue best path")

        fig.canvas.draw()
        fig.canvas.flush_events()
        image_path = os.path.join(cur_exp_im_dir, f"navigation_{nav_idx:04d}.png")
        fig.savefig(image_path, dpi=args.video_dpi, bbox_inches="tight")
        video_writer.grab_frame()
        # plt.pause(0.5)

        print(f"CHOSEN WAYPOINT: {chosen_waypoint}")

        reached_goal = closest_node == goal_node

        if reached_goal:
            print("Reached goal; saving and exiting.")
            break

    print(f"Finished {nav_idx + 1} navigation iterations.")
    video_writer.finish()
    print(f"Saved navigation video to {video_path}")
    plt.ioff()
    plt.show()


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
        default=2,
        type=int,
        help="index of the waypoint used for navigation (default: 2)",
    )
    parser.add_argument(
        "--dir",
        "-topo_dir",
        default="mrc_vint_ft3",
        type=str,
        help="path to topomap images",
    )
    parser.add_argument(
        "--goal-node",
        "-g",
        default=-1,
        type=int,
        help="goal node index in the topomap (if -1, then the goal node is the last node in the topomap)",
    )
    parser.add_argument(
        "--start-img",
        "-s",
        default=0,
        type=int,
        help="which topomap image to use as observation",
    )
    parser.add_argument(
        "--close-threshold",
        "-t",
        default=3,
        type=int,
        help="temporal distance within the next node in the topomap before localizing to it",
    )
    parser.add_argument(
        "--radius",
        "-r",
        default=10,
        type=int,
        help="temporal number of locobal nodes to look at in the topopmap for localization",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=8,
        type=int,
        help="Number of actions sampled from the exploration model",
    )
    parser.add_argument(
        "--exp_dir",
        "-d",
        default="./nav_experiments",
        type=str,
        help="Path to log experiment results",
    )
    parser.add_argument("-robo", "--robot", type=str, help="Robot Name",
                        default="ghost")
    parser.add_argument(
        "--mode",
        default="navigate",
        help="navigate or explore"
    )
    parser.add_argument(
        "--steer",
        action="store_true",
        help="whether to use the reward model steering"
    )
    parser.add_argument(
        "--video-path",
        default=None,
        type=str,
        help="path to save the navigation video; defaults to navigation.mp4 in the experiment directory",
    )
    parser.add_argument(
        "--video-fps",
        default=1.0,
        type=float,
        help="frames per second for the saved navigation video",
    )
    parser.add_argument(
        "--video-dpi",
        default=120,
        type=int,
        help="DPI used when encoding the saved navigation video",
    )

    args = parser.parse_args()
    main(args)
