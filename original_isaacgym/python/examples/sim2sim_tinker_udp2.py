# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from global_config import ROOT_DIR
from configs.tinker_constraint_him import TinkerConstraintHimRoughCfg
import time
import threading
import socket
import struct


class cmd:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0

def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])

def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    
    return (q, dq, quat, v, omega, gvec)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd

def _low_pass_action_filter(actions, last_actions):
    actons_filtered = last_actions * 0.2 + actions * 0.8
    return actons_filtered

def generate_gait(t, frequency=1.0, amplitude=0.2, vx=0.0, vy=0.0, dyaw=0.0, roll_angle=0.0):
    '''Generates sinusoidal joint trajectories for quadruped walking with velocity and roll compensation'''
    gait = np.zeros(10)
    phase = 2 * np.pi * frequency * t
    
    # Adjust amplitude based on forward velocity
    velocity_scale = min(max(abs(vx) / 0.3, 0.5), 1.5)
    adjusted_amplitude = amplitude * velocity_scale
    
    # Trot gait with phase offsets
    gait[0] = adjusted_amplitude * np.sin(phase)  # Left front hip
    gait[2] = -adjusted_amplitude * np.sin(phase + np.pi)  # Right front hip
    gait[4] = -adjusted_amplitude * np.sin(phase + np.pi)  # Left rear hip
    gait[6] = adjusted_amplitude * np.sin(phase)  # Right rear hip
    
    # Knee and ankle joints
    gait[1] = 0.5 * adjusted_amplitude * np.cos(phase)
    gait[3] = -0.5 * adjusted_amplitude * np.cos(phase + np.pi)
    gait[5] = -0.5 * adjusted_amplitude * np.cos(phase + np.pi)
    gait[7] = 0.5 * adjusted_amplitude * np.cos(phase)
    gait[8] = 0.2 * adjusted_amplitude * np.cos(phase)
    gait[9] = -0.2 * adjusted_amplitude * np.cos(phase + np.pi)
    
    # Roll compensation: adjust hip roll joints
    roll_compensation = -0.5 * roll_angle  # Proportional to roll angle, negative to counteract
    gait[0] += roll_compensation  # Left front hip
    gait[2] -= roll_compensation  # Right front hip
    gait[4] += roll_compensation  # Left rear hip
    gait[6] -= roll_compensation  # Right rear hip
    
    # Lateral and yaw adjustments
    if abs(vy) > 0.01:
        lateral_adjust = vy * 0.1
        gait[0] += lateral_adjust
        gait[2] -= lateral_adjust
        gait[4] += lateral_adjust
        gait[6] -= lateral_adjust
    
    if abs(dyaw) > 0.01:
        yaw_adjust = dyaw * 0.1
        gait[0] += yaw_adjust
        gait[2] -= yaw_adjust
        gait[4] -= yaw_adjust
        gait[6] += yaw_adjust
    
    return gait

tx_data_udp = [0]*500
rx_num_now = 0

def decode_int(rx_Buf, start_Byte_num):
    global rx_num_now
    temp = bytes([rx_Buf[start_Byte_num], rx_Buf[start_Byte_num+1], rx_Buf[start_Byte_num+2], rx_Buf[start_Byte_num+3]])
    rx_num_now = rx_num_now + 4
    return struct.unpack('i', temp)[0]

def decode_float(rx_Buf, start_Byte_num):
    global rx_num_now
    temp = bytes([rx_Buf[start_Byte_num], rx_Buf[start_Byte_num+1], rx_Buf[start_Byte_num+2], rx_Buf[start_Byte_num+3]])
    rx_num_now = rx_num_now + 4
    return struct.unpack('f', temp)[0]

def send_float(tx_Buf, data):
    temp_B = struct.pack('f', float(data))
    tx_Buf.append(temp_B[0])
    tx_Buf.append(temp_B[1])
    tx_Buf.append(temp_B[2])
    tx_Buf.append(temp_B[3])

def send_int(tx_Buf, data):
    temp_B = struct.pack('i', int(data))
    tx_Buf.append(temp_B[0])
    tx_Buf.append(temp_B[1])
    tx_Buf.append(temp_B[2])
    tx_Buf.append(temp_B[3])

def run_mujoco(cfg):
    global action_rl, rx_num_now
    # Adjusted default_dof_pos for stable standing pose, optimized for forward walking
    default_dof_pos = [0.0,-0.07,0.56,-1.12,0.57,  0.0,0.07,0.56,-1.12,0.57]
    action_rl = default_dof_pos.copy()

    udp_addr = ('127.0.0.1', 8888)
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind(udp_addr)
    udp_socket.settimeout(10)

    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)
    action_flt = np.zeros((cfg.env.num_actions), dtype=np.double)
    last_actions = np.zeros((cfg.env.num_actions), dtype=np.double)
    hist_obs = deque()
    for _ in range(cfg.env.history_len):
        hist_obs.append(np.zeros([1, cfg.env.n_proprio], dtype=np.double))

    count_lowlevel = 0
    simulation_time = 0.0

    with open("robot_data1.txt", "w") as txt_file:
        txt_file.write("=== 机器人仿真数据 ===\n")
        txt_file.write("格式: 步数 | 关节位置 (q, 弧度) | 关节速度 (dq, 弧度/秒) | 四元数 (quat, [x,y,z,w]) | 欧拉角 (eu_ang, 弧度, [滚转,俯仰,偏航]) | 角速度 (omega, 弧度/秒) | 重力向量 (gvec) | 控制力矩 (tau, Nm) | 命令 (vx, vy, dyaw)\n")
        txt_file.write("====================\n")

    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
        simulation_time += cfg.sim_config.dt
        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-cfg.env.num_actions:]
        dq = dq[-cfg.env.num_actions:]

        eu_ang = quaternion_to_euler_array(quat)
        eu_ang[eu_ang > math.pi] -= 2 * math.pi

        print("=== 仿真数据 ===")
        print(f"关节位置 (q, 弧度): {q}")
        print(f"关节速度 (dq, 弧度/秒): {dq}")
        print(f"四元数 (quat, [x, y, z, w]): {quat}")
        print(f"欧拉角 (eu_ang, 弧度, [滚转, 俯仰, 偏航]): {eu_ang}")
        print(f"角速度 (omega, 弧度/秒): {omega}")
        print(f"重力向量 (gvec): {gvec}")
        print(f"控制力矩 (tau, Nm): {data.ctrl}")
        print(f"命令 (vx, vy, dyaw): [{cmd.vx}, {cmd.vy}, {cmd.dyaw}]")
        print("================\n")

        if count_lowlevel % cfg.sim_config.decimation == 0:
            with open("robot_data1.txt", "a") as txt_file:
                txt_file.write(f"步数: {count_lowlevel}\n")
                txt_file.write(f"关节位置 (q, 弧度): {np.array2string(np.round(q, 2), separator=', ')}\n")
                txt_file.write(f"关节速度 (dq, 弧度/秒): {np.array2string(np.round(dq, 2), separator=', ')}\n")
                txt_file.write(f"四元数 (quat, [x, y, z, w]): {np.array2string(np.round(quat, 2), separator=', ')}\n")
                txt_file.write(f"欧拉角 (eu_ang, 弧度, [滚转, 俯仰, 偏航]): {np.array2string(np.round(eu_ang, 2), separator=', ')}\n")
                txt_file.write(f"角速度 (omega, 弧度/秒): {np.array2string(np.round(omega, 2), separator=', ')}\n")
                txt_file.write(f"重力向量 (gvec): {np.array2string(np.round(gvec, 2), separator=', ')}\n")
                txt_file.write(f"控制力矩 (tau, Nm): {np.array2string(np.round(data.ctrl, 2), separator=', ')}\n")
                txt_file.write(f"命令 (vx, vy, dyaw): [{cmd.vx}, {cmd.vy}, {cmd.dyaw}]\n")
                txt_file.write("--------------------\n")

        if 1:
            if count_lowlevel % cfg.sim_config.decimation == 0:
                udp_get = 0
                try:
                    data_udp, addr = udp_socket.recvfrom(144)
                    if data_udp:
                        udp_get = 1
                        rx_num_now = 0
                        # Read velocity commands first
                        cmd.vx = decode_float(data_udp, rx_num_now)
                        cmd.vy = decode_float(data_udp, rx_num_now)
                        cmd.dyaw = decode_float(data_udp, rx_num_now)
                        # Then read joint actions
                        for i in range(10):
                            action_rl[i] = decode_float(data_udp, rx_num_now)
                except socket.timeout:
                    print("UDP timeout, using default commands")
                    cmd.vx = 0.3  # Fallback to default forward velocity
                    cmd.vy = 0.0
                    cmd.dyaw = 0.0
                except Exception as e:
                    print(f"UDP error: {e}, using default commands")
                    cmd.vx = 0.3
                    cmd.vy = 0.0
                    cmd.dyaw = 0.0

                # Send observations via UDP
                tx_data_udp_temp = []
                send_float(tx_data_udp_temp, 1)
                send_float(tx_data_udp_temp, cmd.vx)
                send_float(tx_data_udp_temp, cmd.vy)
                send_float(tx_data_udp_temp, cmd.dyaw)
                send_float(tx_data_udp_temp, 0)
                send_float(tx_data_udp_temp, eu_ang[0])
                send_float(tx_data_udp_temp, eu_ang[1])
                send_float(tx_data_udp_temp, eu_ang[2])
                send_float(tx_data_udp_temp, omega[0])
                send_float(tx_data_udp_temp, omega[1])
                send_float(tx_data_udp_temp, omega[2])
                send_float(tx_data_udp_temp, 0)
                send_float(tx_data_udp_temp, 0)
                send_float(tx_data_udp_temp, 0)
                for i in range(10):
                    send_float(tx_data_udp_temp, q[i])
                for i in range(10):
                    send_float(tx_data_udp_temp, dq[i])
                for i in range(10):
                    send_float(tx_data_udp_temp, 0)

                for i in range(len(tx_data_udp_temp)):
                    tx_data_udp[i] = tx_data_udp_temp[i]
                len_tx = len(tx_data_udp_temp)

                tx_data_temp = [0] * len_tx
                for i in range(len_tx):
                    tx_data_temp[i] = tx_data_udp[i]
                try:
                    udp_socket.sendto(bytearray(tx_data_temp), addr)
                except:
                    print("Sending Error!!!\n")

                # Generate walking gait with UDP commands
                gait = generate_gait(simulation_time, frequency=1.0, amplitude=0.2, vx=cmd.vx, vy=cmd.vy, dyaw=cmd.dyaw, roll_angle=eu_ang[0])
                # Combine gait with UDP actions
                action = np.clip(action_rl, -1, 1) * 0.1 + gait
                action = _low_pass_action_filter(action, last_actions)
                last_actions = action.copy()
                target_q = action + default_dof_pos

            target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
            tau = pd_control(target_q, q, cfg.robot_config.kps,
                            target_dq, dq, cfg.robot_config.kds)
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
            data.ctrl = tau
        else:
            obs = np.zeros([1, cfg.env.n_proprio], dtype=np.float32)
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi
            target_q = default_dof_pos
            target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
            tau = pd_control(target_q, q, cfg.robot_config.kps,
                            target_dq, dq, cfg.robot_config.kds)
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
            data.ctrl = tau

        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1

    viewer.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--terrain', action='store_true', default='/home/rot/original_isaacgym/modelt2.pt')
    args = parser.parse_args()
    
    class Sim2simCfg(TinkerConstraintHimRoughCfg):
        class sim_config:
            if args.terrain:
                mujoco_model_path = f'{ROOT_DIR}/resources/tinker/xml/world_terrain.xml'
            else:
                mujoco_model_path = f'{ROOT_DIR}/resources/tinker/xml/world.xml'
            sim_duration = 60.0
            dt = 0.001
            decimation = 18

        class robot_config:
            kp_all = 80.0  # Increased for better tracking
            kd_all = 2.0   # Slightly increased for stability
            kps = np.array([kp_all, kp_all, kp_all, kp_all, kp_all, kp_all, kp_all, kp_all, kp_all, kp_all], dtype=np.double)
            kds = np.array([kd_all, kd_all, kd_all, kd_all, kd_all, kd_all, kd_all, kd_all, kd_all, kd_all], dtype=np.double)
            tau_limit = 25.0 * np.ones(10, dtype=np.double)  # Increased torque limit for forward motion
    run_mujoco(Sim2simCfg())