import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from global_config import ROOT_DIR
from configs.tinker_constraint_him import TinkerConstraintHimRoughCfg
import torch
import pygame

# 初始化手柄
pygame.init()
pygame.joystick.init()
if pygame.joystick.get_count() == 0:
    raise RuntimeError("No joystick detected")
joystick = pygame.joystick.Joystick(0)
joystick.init()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class cmd:
    vx = 0.0  # 前进/后退速度，由手柄控制
    vy = 0.0  # 横向速度，固定为 0
    dyaw = 0.0  # 偏航角速度，由手柄控制

default_dof_pos = [0.0,-0.07,0.56,-1.12,0.57,  0.0,0.07,0.56,-1.12,0.57]

def quaternion_to_euler_array(quat):
    x, y, z, w = quat[[1, 2, 3, 0]]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    return np.array([roll_x, pitch_y, yaw_z])

def get_obs(data):
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True)[[1, 0, 2]]  # 交换 x 和 y 轴
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

def _low_pass_action_filter(actions, last_actions):
    return last_actions * 0.3 + actions * 0.7

def update_joystick_commands(cfg):
    pygame.event.clear()
    pygame.event.pump()
    # 读取轴值并应用死区
    DEAD_ZONE = 0.15
    vx_raw = -joystick.get_axis(1)  # 左摇杆 y 轴，需确认索引
    dyaw_raw = joystick.get_axis(3)  # 右摇杆 x 轴，需确认索引
    cmd.vx = 0.0 if abs(vx_raw) < DEAD_ZONE else vx_raw * cfg.commands.ranges.lin_vel_x[1]
    cmd.dyaw = 0.0 if abs(dyaw_raw) < DEAD_ZONE else dyaw_raw * cfg.commands.ranges.ang_vel_yaw[1]
    cmd.vy = 0.0
    cmd.vx = np.clip(cmd.vx, cfg.commands.ranges.lin_vel_x[0], cfg.commands.ranges.lin_vel_x[1])
    cmd.dyaw = np.clip(cmd.dyaw, cfg.commands.ranges.ang_vel_yaw[0], cfg.commands.ranges.ang_vel_yaw[1])
    return cmd.vx, cmd.vy, cmd.dyaw

def run_mujoco(policy, cfg):
    global default_dof_pos
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)

    # Set initial position to ensure robot is on ground
    data.qpos[0:3] = [0.0, 0.0, 0.25]  # Adjust z to match simple_play.py
    data.qpos[7:] = default_dof_pos  # Set initial joint positions
    mujoco.mj_resetData(model, data)

    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)
    action_flt = np.zeros((cfg.env.num_actions), dtype=np.double)
    last_actions = np.zeros((cfg.env.num_actions), dtype=np.double)
    hist_obs = deque()
    data_log = []

    # 初始化历史观测
    q, dq, quat, v, omega, gvec = get_obs(data)
    obs = np.zeros([1, cfg.env.n_proprio], dtype=np.float32)
    eu_ang = quaternion_to_euler_array(quat)
    eu_ang[eu_ang > math.pi] -= 2 * math.pi
    obs[0, 0:3] = omega * cfg.normalization.obs_scales.ang_vel
    obs[0, 3:6] = eu_ang * cfg.normalization.obs_scales.quat
    obs[0, 6:9] = [cmd.vx * cfg.normalization.obs_scales.lin_vel, cmd.vy * cfg.normalization.obs_scales.lin_vel, cmd.dyaw * cfg.normalization.obs_scales.ang_vel]
    obs[0, 9:19] = (q[-cfg.env.num_actions:] - default_dof_pos) * cfg.normalization.obs_scales.dof_pos
    obs[0, 19:29] = dq[-cfg.env.num_actions:] * cfg.normalization.obs_scales.dof_vel
    obs[0, 29:39] = last_actions
    for _ in range(cfg.env.history_len):
        hist_obs.append(obs.copy())

    count_lowlevel = 0
    for step in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
        # 更新手柄指令
        cmd.vx, cmd.vy, cmd.dyaw = 0.0, 0.0, 0.0
        if count_lowlevel % cfg.sim_config.decimation == 0:
            vx, vy, dyaw = update_joystick_commands(cfg)
            print(f"Step {step}: cmd.vx={vx:.4f}, cmd.vy={vy:.4f}, cmd.dyaw={dyaw:.4f}")

        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-cfg.env.num_actions:]
        dq = dq[-cfg.env.num_actions:]

        if count_lowlevel % cfg.sim_config.decimation == 0:
            obs = np.zeros([1, cfg.env.n_proprio], dtype=np.float32)
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi
            obs[0, 0:3] = omega * cfg.normalization.obs_scales.ang_vel
            obs[0, 3:6] = eu_ang * cfg.normalization.obs_scales.quat
            obs[0, 6:9] = [cmd.vx * cfg.normalization.obs_scales.lin_vel, cmd.vy * cfg.normalization.obs_scales.lin_vel, cmd.dyaw * cfg.normalization.obs_scales.ang_vel]
            obs[0, 9:19] = (q - default_dof_pos) * cfg.normalization.obs_scales.dof_pos
            obs[0, 19:29] = dq * cfg.normalization.obs_scales.dof_vel
            obs[0, 29:39] = last_actions
            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

            hist_obs.append(obs)
            hist_obs.popleft()

            policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float16)
            policy_input[0, 0:cfg.env.n_proprio] = obs
            for i in range(cfg.env.history_len):
                policy_input[0, cfg.env.n_proprio + cfg.env.n_priv_latent + cfg.env.n_scan + i * cfg.env.n_proprio : cfg.env.n_proprio + cfg.env.n_priv_latent + cfg.env.n_scan + (i + 1) * cfg.env.n_proprio] = hist_obs[i][0, :]

            action[:] = policy.act_teacher(torch.tensor(policy_input, device=device).half())[0].detach().cpu().numpy()
            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
            # action = np.clip(action, -0.5, 0.5)

            action_flt = _low_pass_action_filter(action, last_actions)
            last_actions = action

            target_q = action_flt + default_dof_pos

            print(f"Step {step}: obs[0, 6:9]={obs[0, 6:9]}, action={action}, target_q={target_q}")

            if np.any(np.abs(action) > 2.0) or np.any(np.abs(target_q) > 5.0):
                print(f"Warning: Large action={action}, target_q={target_q}")

        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
        tau = pd_control(target_q, q, cfg.robot_config.kps, target_dq, dq, cfg.robot_config.kds)
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
        # tau = np.clip(tau, -1, 1)

        data.ctrl = tau

        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1

        data_log.append({
            'step': step,
            'base_pos': data.qpos[:3].copy(),
            'base_vel': v.copy(),
            'action': action.copy()
        })

    np.save("sim2sim_trajectory.npy", data_log)
    viewer.close()
    pygame.quit()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, default='/home/rot/original_isaacgym/modelt2.pt')
    parser.add_argument('--terrain', action='store_true', default=False)
    args = parser.parse_args()

    class Sim2simCfg(TinkerConstraintHimRoughCfg):
        class sim_config:
            mujoco_model_path = f'{ROOT_DIR}/resources/tinker/xml/world.xml'
            sim_duration = 60.0
            dt = 0.001
            decimation = 20

        class robot_config:
            # kp_all = 15.0
            # kd_all = 0.1
            # kps = np.array([kp_all, kp_all, kp_all, kp_all, kp_all, kp_all, kp_all, kp_all, kp_all, kp_all], dtype=np.double)#PD和isacc内部一致
            # kds = np.array([kd_all, kd_all, kd_all, kd_all, kd_all, kd_all, kd_all, kd_all, kd_all, kd_all], dtype=np.double)
            kps = np.array([25.0, 25.0, 25.0, 25.0, 25.0,     25.0, 25.0, 25.0, 25.0, 25.0], dtype=np.double)
            kds = np.array([0.8, 1.2, 1.2, 0.8, 1.2,          0.8, 1.2, 1.2, 0.8, 1.2], dtype=np.double)
            tau_limit = 12.0 * np.ones(10, dtype=np.double)

        class normalization:
            class obs_scales:
                lin_vel = 2.0
                ang_vel = 0.5
                dof_pos = 1.0
                dof_vel = 0.05
                quat = 0.05
            clip_observations = 100.0
            clip_actions = 1.0

        class commands:
            class ranges:
                lin_vel_x = [-0.5, 0.5]  # 前进/后退速度范围
                lin_vel_y = [-0.2, 0.2]  # 横向速度范围（未使用）
                ang_vel_yaw = [-1.0, 1.0]  # 偏航角速度范围

    policy = torch.load(args.load_model, map_location=device)
    policy.to(device)
    run_mujoco(policy, Sim2simCfg())