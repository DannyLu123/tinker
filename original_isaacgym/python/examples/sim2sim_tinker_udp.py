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
import numpy as np
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
    
    # 将 q, dq, quat, r, v, omega, gvec 按照 State_Rl.cpp需要的方式组装起来
    
    
    return (q, dq, quat, v, omega, gvec)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd

def _low_pass_action_filter(actions,last_actions):
    actons_filtered = last_actions * 0.2 + actions * 0.8
    return actons_filtered


tx_data_udp =  [0]*500
# struct _msg_request
# {
#     int trigger;
#     float command[10];
#     float eu_ang[3];
#     float omega[3];
#     float acc[3];
#     float q[12];
#     float dq[12];
#     float tau[12];
# };

# struct _msg_response
# {
#     float q_exp[12];
#     float dq_exp[12];
#     float tau_exp[12];
# };
rx_num_now=0
def decode_int(rx_Buf,start_Byte_num):
    global rx_num_now
    temp=bytes([rx_Buf[start_Byte_num],rx_Buf[start_Byte_num+1],rx_Buf[start_Byte_num+2],rx_Buf[start_Byte_num+3]])
    rx_num_now= rx_num_now+4
    return struct.unpack('i',temp)[0]

def decode_float(rx_Buf,start_Byte_num):
    global rx_num_now
    temp=bytes([rx_Buf[start_Byte_num],rx_Buf[start_Byte_num+1],rx_Buf[start_Byte_num+2],rx_Buf[start_Byte_num+3]])
    rx_num_now= rx_num_now+4
    return struct.unpack('f',temp)[0]

def send_float(tx_Buf,data):
    temp_B= struct.pack('f',float(data))
    tx_Buf.append(temp_B[0])
    tx_Buf.append(temp_B[1])
    tx_Buf.append(temp_B[2])
    tx_Buf.append(temp_B[3])

def send_int(tx_Buf,data):
    temp_B= struct.pack('i',int(data))
    tx_Buf.append(temp_B[0])
    tx_Buf.append(temp_B[1])
    tx_Buf.append(temp_B[2])
    tx_Buf.append(temp_B[3])

def run_mujoco(cfg):
    global action_rl,rx_num_now
    default_dof_pos=[0.0,-0.07,0.56,-1.12,0.57,  0.0,0.07,0.56,-1.12,0.57]#默认角度需要与isacc一致
    #default_dof_pos=[0.0,0.0,0.0,0.0,0.0,  0.0,0.0,0.0,0.0,0.0]
    action_rl=default_dof_pos#[0.0,-0.1,-0.56,1.12,-0.57,  0.0,0.1,-0.56,1.12,-0.57]#默认角度需要与isacc一致

    udp_addr = ('127.0.0.1', 8888)
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)# SOCK_DGRAM指的就是UDP通信类型
    udp_socket.bind(udp_addr)
    udp_socket.settimeout(10)  # 设置一个时间提示，如果10秒钟没接到数据进行提示

    """
    通过LCM作为通用接口传输传感器原始数据，接受网络直接输出
    """
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)#载入初始化位置由XML决定
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
    cnt_send=0

    # 初始化 TXT 文件
    with open("robot_data1.txt", "w") as txt_file:
        txt_file.write("=== 机器人仿真数据 ===\n")
        txt_file.write("格式: 步数 | 关节位置 (q, 弧度) | 关节速度 (dq, 弧度/秒) | 四元数 (quat, [x,y,z,w]) | 欧拉角 (eu_ang, 弧度, [滚转,俯仰,偏航]) | 角速度 (omega, 弧度/秒) | 重力向量 (gvec) | 控制力矩 (tau, Nm)\n")
        txt_file.write("====================\n")

    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
        
        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data)#从mujoco获取仿真数据
        q = q[-cfg.env.num_actions:]
        dq = dq[-cfg.env.num_actions:]

        # 转换为欧拉角
        eu_ang = quaternion_to_euler_array(quat)
        eu_ang[eu_ang > math.pi] -= 2 * math.pi
        
        # obs_buf =torch.cat((self.base_ang_vel  * self.obs_scales.ang_vel,
        #                     self.base_euler_xyz * self.obs_scales.quat,
        #                     self.commands[:, :3] * self.commands_scale,#xy+航向角速度
        #                     self.reindex((self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos),
        #                     self.reindex(self.dof_vel * self.obs_scales.dof_vel),
        #                     self.action_history_buf[:,-1]),dim=-1)#列表最后一项 [:-1]也就是上一次的
        # 打印机器人数据
        print("=== 仿真数据 ===")
        print(f"关节位置 (q, 弧度): {q}")
        print(f"关节速度 (dq, 弧度/秒): {dq}")
        print(f"四元数 (quat, [x, y, z, w]): {quat}")
        print(f"欧拉角 (eu_ang, 弧度, [滚转, 俯仰, 偏航]): {eu_ang}")
        print(f"角速度 (omega, 弧度/秒): {omega}")
        print(f"重力向量 (gvec): {gvec}")
        print(f"控制力矩 (tau, Nm): {data.ctrl}")
        print("================\n")

        # 在低频控制周期保存数据到 TXT 文件，保留两位小数
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
                txt_file.write("--------------------\n")
        

        if 1:
            # 1000hz -> 100hz
            if count_lowlevel % cfg.sim_config.decimation == 0:
                udp_get=0
                try:
                    data_udp, addr = udp_socket.recvfrom(144)
                    # float q_exp[12];
                    # float dq_exp[12];
                    # float tau_exp[12];
                    if data_udp:
                        udp_get=1
                        rx_num_now = 0
                        for i in range(10):
                            action_rl[i]=decode_float(data_udp,rx_num_now)
                        #print ("got data from", addr, 'len=',len(data))#在此对遥控器下发指令进行解码
                        #print ("data_udp:",data_udp)
                        #print(action_rl)
                except:
                    print ("err UDP data from", addr, 'len=',len(data_udp))
                    # udp_get=1
                    # rx_num_now = 0
                    # for i in range(12):
                    #     action_rl[i]=decode_float(data_udp,rx_num_now)
                
                eu_ang = quaternion_to_euler_array(quat)
                eu_ang[eu_ang > math.pi] -= 2 * math.pi

                cmd.vx=0.3
                cmd.vy=0.0
                cmd.dyaw= 0.0
                #---send obs
                if 1:
                    tx_data_udp_temp=[]
                #     int trigger;
                #     float command[4];
                #     float eu_ang[3];
                #     float omega[3];
                #     float acc[3];
                #     float q[12];
                #     float dq[12];
                #     float tau[12];
                    send_float(tx_data_udp_temp,1)
                    send_float(tx_data_udp_temp,cmd.vx)
                    send_float(tx_data_udp_temp,cmd.vy)
                    send_float(tx_data_udp_temp,cmd.dyaw)
                    send_float(tx_data_udp_temp,0)
                    send_float(tx_data_udp_temp,eu_ang[0])
                    send_float(tx_data_udp_temp,eu_ang[1])
                    send_float(tx_data_udp_temp,eu_ang[2])
                    send_float(tx_data_udp_temp,omega[0])
                    send_float(tx_data_udp_temp,omega[1])
                    send_float(tx_data_udp_temp,omega[2])
                    send_float(tx_data_udp_temp,0)
                    send_float(tx_data_udp_temp,0)
                    send_float(tx_data_udp_temp,0)
                    for i in range(10):
                        send_float(tx_data_udp_temp,q[i])
                    for i in range(10):
                        send_float(tx_data_udp_temp,dq[i])
                    for i in range(10):
                        send_float(tx_data_udp_temp,0)        

                    for i in range(len(tx_data_udp_temp)):
                        tx_data_udp[i]=tx_data_udp_temp[i]
                    len_tx=len(tx_data_udp_temp) 

                    tx_data_temp =  [0]*len_tx
                    for i in range(len_tx):
                        tx_data_temp[i]=tx_data_udp[i]
                    try:
                        udp_socket.sendto(bytearray (tx_data_temp), addr)#在此将机器人状态数据回传OCU
                    except:
                        print("Sending Error!!!\n")
                    
                #--UDP传输RL滤波后的输出
                action = np.clip(action_rl, -1, 1)
                #action = action_rl
                #print("clip_actions:", cfg.normalization.clip_actions)
                target_q = action * 0.1 + default_dof_pos
                #target_q = action * cfg.control.action_scale + default_dof_pos
                #target_q = [0.0,-0.07,0.57,-1.12,0.56,  0.0,0.07,0.57,-1.12,0.56]
            target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
            # Generate PD control
            # print("action=",action)
            # print("target_q=",target_q)
            print("default_dof_pos=",default_dof_pos)
            tau = pd_control(target_q, q, cfg.robot_config.kps,
                             target_dq, dq, cfg.robot_config.kds)  # Calc torques
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
            data.ctrl = tau
        else:#air mode test
            obs = np.zeros([1, cfg.env.n_proprio], dtype=np.float32) #1,45
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi
            target_q = default_dof_pos
            # target_q[0]=0
            # target_q[1]=3
            # target_q[2]=3
            # target_q[3]=0
            # target_q[4]=3
            # target_q[5]=3     
            #print(eu_ang*57.3)
            target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
            # Generate PD control
            tau = pd_control(target_q, q, cfg.robot_config.kps,
                            target_dq, dq, cfg.robot_config.kds)  # Calc torques
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
            data.ctrl = tau

        
        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1

    viewer.close()
    

 
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--terrain', action='store_true', default='/home/rot/original_isaacgym/modelt.pt')
    args = parser.parse_args()
    
    class Sim2simCfg(TinkerConstraintHimRoughCfg):

        class sim_config:
            if args.terrain:
                mujoco_model_path = f'{ROOT_DIR}/resources/tinker/xml/world_terrain.xml'
            else:
                mujoco_model_path = f'{ROOT_DIR}/resources/tinker/xml/world.xml'
            sim_duration = 60.0
            dt = 0.001 #1Khz底层
            decimation = 1 # 50Hz 网络滞后需要加速

        class robot_config:
            kp_all = 8.0
            kd_all = 0.8
            kps = np.array([kp_all, kp_all, kp_all, kp_all, kp_all, kp_all, kp_all, kp_all, kp_all, kp_all], dtype=np.double)#PD和isacc内部一致
            kds = np.array([kd_all, kd_all, kd_all, kd_all, kd_all, kd_all, kd_all, kd_all, kd_all, kd_all], dtype=np.double)
            tau_limit = 10.0 * np.ones(10, dtype=np.double)#nm
    
    run_mujoco(Sim2simCfg())
