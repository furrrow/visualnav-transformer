import os
import sys
import cv2
import math
import json
import yaml
# pytorch
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from PIL import Image
from typing import List, Tuple, Dict, Optional

# models
from train.vint_train.models.gnm.gnm import GNM
from train.vint_train.models.vint.vint import ViNT
from train.vint_train.models.vint.vit import ViT
from train.vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from train.vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from train.vint_train.data.data_utils import IMAGE_ASPECT_RATIO

BGR_color_dict = { # BGR
    "RED" : (0, 0, 255),
    "GREEN" : (0, 255, 0),
    "BLUE" : (255, 0, 0),
    "CYAN" : (255, 255, 0),
    "YELLOW" : (0, 255, 255),
    "CUSTOM" : (125, 125, 125),
}

RGB_color_dict = { # RGB
    "RED" : (255, 0, 0),
    "GREEN" : (0, 255, 0),
    "BLUE" : (0, 0, 255),
    "CYAN" : (0, 255, 255),
    "YELLOW" : (255, 255, 0),
    "CUSTOM" : (125, 125, 125),
}

def load_data_stats() -> dict:
    with open(
        os.path.join(os.path.dirname(__file__), "../../train/vint_train/data/data_config.yaml"), "r"
    ) as f:
        data_config = yaml.safe_load(f)
    action_stats = {}
    for key in data_config["action_stats"]:
        action_stats[key] = np.array(data_config["action_stats"][key])
    return action_stats


ACTION_STATS = load_data_stats()

def load_model(
    model_path: str,
    config: dict,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Load a model from a checkpoint file (works with models trained on multiple GPUs)"""
    model_type = config["model_type"]
    
    if model_type == "gnm":
        model = GNM(
            config["context_size"],
            config["len_traj_pred"],
            config["learn_angle"],
            config["obs_encoding_size"],
            config["goal_encoding_size"],
        )
    elif model_type == "vint":
        model = ViNT(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
    elif config["model_type"] == "nomad":
        if config["vision_encoder"] == "nomad_vint":
            vision_encoder = NoMaD_ViNT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        elif config["vision_encoder"] == "vit": 
            vision_encoder = ViT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                image_size=config["image_size"],
                patch_size=config["patch_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        else: 
            raise ValueError(f"Vision encoder {config['vision_encoder']} not supported")
        
        noise_pred_net = ConditionalUnet1D(
                input_dim=2,
                global_cond_dim=config["encoding_size"],
                down_dims=config["down_dims"],
                cond_predict_scale=config["cond_predict_scale"],
            )
        dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
        
        model = NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if model_type == "nomad":
        state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        loaded_model = checkpoint["model"]
        try:
            state_dict = loaded_model.module.state_dict()
            model.load_state_dict(state_dict, strict=False)
        except AttributeError as e:
            state_dict = loaded_model.state_dict()
            model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model


def msg_to_pil(msg: Image) -> PILImage.Image:
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
        msg.height, msg.width, -1)
    pil_image = PILImage.fromarray(img)
    return pil_image


def pil_to_msg(pil_img: PILImage.Image, encoding="mono8") -> Image:
    img = np.asarray(pil_img)  
    ros_image = Image(encoding=encoding)
    ros_image.height, ros_image.width, _ = img.shape
    ros_image.data = img.ravel().tobytes() 
    ros_image.step = ros_image.width
    return ros_image


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def transform_images(pil_imgs: List[PILImage.Image], image_size: List[int], center_crop: bool = False) -> torch.Tensor:
    """Transforms a list of PIL image to a torch tensor."""
    transform_type = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                    0.229, 0.224, 0.225]),
        ]
    )
    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        w, h = pil_img.size
        if center_crop:
            if w > h:
                pil_img = TF.center_crop(pil_img, (h, int(h * IMAGE_ASPECT_RATIO)))  # crop to the right ratio
            else:
                pil_img = TF.center_crop(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))
        pil_img = pil_img.resize(image_size) 
        transf_img = transform_type(pil_img)
        transf_img = torch.unsqueeze(transf_img, 0)
        transf_imgs.append(transf_img)
    return torch.cat(transf_imgs, dim=1)
    

# clip angle between -pi and pi
def clip_angle(angle):
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

def get_action(diffusion_output, action_stats=ACTION_STATS):
    # diffusion_output: (B, 2*T+1, 1)
    # return: (B, T-1)
    device = diffusion_output.device
    ndeltas = diffusion_output
    ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
    ndeltas = to_numpy(ndeltas)
    ndeltas = unnormalize_data(ndeltas, action_stats)
    actions = np.cumsum(ndeltas, axis=1)
    return torch.from_numpy(actions).float().to(device)

VIZ_IMAGE_SIZE = (640, 480)
RED = np.array([1, 0, 0])
GREEN = np.array([0, 1, 0])
BLUE = np.array([0, 0, 1])
CYAN = np.array([0, 1, 1])
YELLOW = np.array([1, 1, 0])
MAGENTA = np.array([1, 0, 1])

def plot_trajs_and_points(
    ax: plt.Axes,
    list_trajs: list,
    list_points: list,
    traj_colors: list = [CYAN, MAGENTA],
    point_colors: list = [RED, GREEN],
    traj_labels: Optional[list] = ["prediction", "ground truth"],
    point_labels: Optional[list] = ["robot", "goal"],
    traj_alphas: Optional[list] = None,
    point_alphas: Optional[list] = None,
    quiver_freq: int = 1,
    default_coloring: bool = True,
):
    assert len(list_trajs) <= len(traj_colors) or default_coloring, (
        "Not enough colors for trajectories"
    )
    assert len(list_points) <= len(point_colors), "Not enough colors for points"
    assert (
        traj_labels is None or len(list_trajs) == len(traj_labels) or default_coloring
    ), "Not enough labels for trajectories"
    assert point_labels is None or len(list_points) == len(point_labels), (
        "Not enough labels for points"
    )

    for i, traj in enumerate(list_trajs):
        if traj_labels is None:
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                color=traj_colors[i],
                alpha=traj_alphas[i] if traj_alphas is not None else 1.0,
                marker="o",
            )
        else:
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                color=traj_colors[i],
                label=traj_labels[i],
                alpha=traj_alphas[i] if traj_alphas is not None else 1.0,
                marker="o",
            )
    for i, pt in enumerate(list_points):
        if point_labels is None:
            ax.plot(
                pt[0],
                pt[1],
                color=point_colors[i],
                alpha=point_alphas[i] if point_alphas is not None else 1.0,
                marker="o",
                markersize=7.0,
            )
        else:
            ax.plot(
                pt[0],
                pt[1],
                color=point_colors[i],
                alpha=point_alphas[i] if point_alphas is not None else 1.0,
                marker="o",
                markersize=7.0,
                label=point_labels[i],
            )
    if traj_labels is not None or point_labels is not None:
        ax.legend()
        ax.legend(bbox_to_anchor=(0.0, -0.5), loc="upper left", ncol=2)
    ax.set_aspect("equal", "box")

# my custom utils
def overlay_path(pts_cur: np.ndarray, img: Optional[np.ndarray] = None, cam_matrix: Optional[np.ndarray] = None,
                 T_cam_from_base: Optional[np.ndarray] = None,
                 path_color=(0, 0, 255), policy_color= RGB_color_dict['BLUE'], steer_color=RGB_color_dict['RED'],
                 metrics=None):
    if pts_cur.size == 0:
        print("pts_cur.size is zero...")
        return None
    if cam_matrix is None or T_cam_from_base is None:
        print("cam_matrix:", cam_matrix)
        print("T_cam_from_base:", T_cam_from_base)
        print("returning...")
        return None
    if img is None:
        print("img is none...")
        return None

    if len(pts_cur.shape) == 2:
        n_trajectories = 1
        pts_cur = np.expand_dims(pts_cur, 0)
    elif len(pts_cur.shape) == 3:
        n_trajectories = pts_cur.shape[0]
    else:
        print("error, unable to process pts_cur dimension", pts_cur.shape)
        return None
    metric_labels = {}
    if metrics is not None:
        reward_values = []
        for i in range(n_trajectories):
            metric = metrics.get(i, metrics.get(str(i), None))
            if metric is None:
                reward_values.append(np.nan)
                continue
            reward_values.append(metric.get("reward", np.nan))

            label_lines = [f"{i}"]
            if "reward" in metric:
                label_lines.append(f"rew {metric['reward']:.2f}")
            if "frdist" in metric:
                label_lines.append(f"frd {metric['frdist']:.2f}")
            if "dtw" in metric:
                label_lines.append(f"dtw {metric['dtw']:.2f}")
            metric_labels[i] = label_lines
        rewards = np.asarray(reward_values, dtype=np.float32)

    # Points in base frame -> camera frame -> pixels
    R_cb = T_cam_from_base[:3, :3]
    t_cb = T_cam_from_base[:3, 3]
    rvec, _ = cv2.Rodrigues(R_cb)
    overlay = img.copy()
    reward_labels = []
    for i in range(n_trajectories):
        pts_3d = np.hstack([pts_cur[i], np.zeros((pts_cur[i].shape[0], 1))])  # z=0 in base frame
        img_pts, _ = cv2.projectPoints(pts_3d, rvec, t_cb, cam_matrix, None)
        img_pts = img_pts.reshape(-1, 2)

        # Keep points in front of camera and inside image
        pts_cam = (R_cb @ pts_3d.T + t_cb.reshape(3, 1)).T
        valid_z = pts_cam[:, 2] > 0
        h, w = img.shape[:2]
        valid_xy = (
            (img_pts[:, 0] >= 0) & (img_pts[:, 0] < w) &
            (img_pts[:, 1] >= 0) & (img_pts[:, 1] < h)
        )
        keep = valid_z & valid_xy
        if not keep.any():
            print(f"out of {pts_cam.shape} points, no points kept in front of camera...")
            continue

        pts_pix = img_pts[keep].astype(int)
        my_color = path_color
        if i == 0:
            my_color = policy_color
        if metrics is not None:
            if i == np.argmax(rewards):
                my_color = steer_color
        if len(pts_pix) >= 2:
            cv2.polylines(overlay, [pts_pix], isClosed=False, color=my_color, thickness=2)
        else:
            for pt in pts_pix:
                cv2.circle(overlay, tuple(pt), radius=3, color=my_color, thickness=-1)
        if metrics is not None:
            label_lines = metric_labels.get(i, [f"{i}: {float(rewards[i]):.2f}"])
            label_anchor = pts_pix[-1]
            reward_labels.append({
                "label_lines": label_lines,
                "anchor": label_anchor,
                "color": my_color,
                "traj_idx": i,
            })

    # Sort by x coordinate
    reward_labels.sort(key=lambda item: item["anchor"][0])

    if reward_labels:
        trajectory_y_values = [item["anchor"][1] for item in reward_labels]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        pad = 4
        gap = 6
        bg_alpha = 0.55
        text_sizes = [
            cv2.getTextSize(line, font, font_scale, thickness)
            for item in reward_labels
            for line in item["label_lines"]
        ]
        text_w = max((size[0][0] for size in text_sizes), default=80)
        text_h = max((size[0][1] for size in text_sizes), default=15)
        baseline = max((size[1] for size in text_sizes), default=5)
        line_gap = 4
        max_lines = max((len(item["label_lines"]) for item in reward_labels), default=1)
        box_w = text_w + 2 * pad
        box_h = max_lines * text_h + (max_lines - 1) * line_gap + baseline + 2 * pad
        box_y_gap = 90

        top_y = int(min(trajectory_y_values)) if trajectory_y_values else 0
        box_top = int(np.clip(top_y - box_h - box_y_gap, 0, max(0, h - box_h - 1)))
        total_row_w = len(reward_labels) * box_w + max(0, len(reward_labels) - 1) * gap
        leftmost_anchor_x = min(int(item["anchor"][0]) for item in reward_labels)
        leftmost_anchor_x = min(leftmost_anchor_x, w//3)
        max_start_x = max(0, w - total_row_w - 1)
        current_x = int(np.clip(leftmost_anchor_x - box_w // 2, 0, max_start_x))

        for idx, item in enumerate(reward_labels):
            label_lines, anchor, color = item['label_lines'], item['anchor'], item['color']
            x = current_x
            best_box = (x, box_top, x + box_w, box_top + box_h)
            current_x = best_box[2] + gap

            anchor_xy = (int(anchor[0]), int(anchor[1]))
            label_center = ((best_box[0] + best_box[2]) // 2, (best_box[1] + best_box[3]) // 2)
            cv2.line(overlay, anchor_xy, label_center, color=color, thickness=1, lineType=cv2.LINE_AA)
            cv2.circle(overlay, anchor_xy, radius=2, color=color, thickness=-1)
            x1, y1, x2, y2 = best_box
            box_roi = overlay[y1:y2, x1:x2]
            black_fill = np.zeros_like(box_roi)
            cv2.addWeighted(black_fill, bg_alpha, box_roi, 1 - bg_alpha, 0, dst=box_roi)
            cv2.rectangle(overlay, (best_box[0], best_box[1]), (best_box[2], best_box[3]), color, thickness=1)
            for line_idx, line in enumerate(label_lines):
                line_y = box_top + pad + text_h + line_idx * (text_h + line_gap)
                cv2.putText(overlay, line, (x + pad, line_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return overlay


def load_calibration(json_path: str):
    """
    Builds:
      K (3x3), dist=None, T_cam_from_base (4x4)
    from tf.json with H_cam_bl: pitch(deg), x,y,z.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if data is None or "H_cam_bl" not in data:
        raise ValueError(f"Missing H_cam_bl in {json_path}")

    h = data["H_cam_bl"]
    roll = math.radians(float(h["roll"]))
    xt, yt, zt = float(h["x"]), float(h["y"]), float(h["z"])

    # Rotation about +y (camera pitched down is positive pitch if y up/right-handed)
    Ry = np.array([
        [0.0, math.sin(roll), math.cos(roll)],
        [-1.0, 0.0, 0.0],
        [0.0, -math.cos(roll), math.sin(roll)]
    ], dtype=np.float64)

    T_base_from_cam = np.eye(4, dtype=np.float64)
    T_base_from_cam[:3, :3] = Ry
    T_base_from_cam[:3, 3] = np.array([xt, yt, zt], dtype=np.float64)

    fx = data["Intrinsics"]["fx"]
    fy = data["Intrinsics"]["fy"]
    cx = data["Intrinsics"]["cx"]
    cy = data["Intrinsics"]["cy"]

    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    dist = None  # explicitly no distortion
    return K, dist, T_base_from_cam