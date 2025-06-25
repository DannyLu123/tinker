#include "udp_publish.h"
#include <iostream>
#include <valarray>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <pthread.h>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <unistd.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include <sys/shm.h>
#include <arpa/inet.h>
#include <time.h>
using namespace std;
using namespace torch::indexing; // 确保使用了正确的命名空间
RL_Tinymal_UDP tinymal_rl;
//RL
struct _msg_request msg_request;
struct _msg_response msg_response;
float limit(float input,float min,float max){
    if(input>max)
        return max;
    if(input<min)
        return min;
    return input;
} 


void RL_Tinymal_UDP::handleMessage(_msg_request request)//获取机器人反馈
{              
    // #单次观测
    // obs[0, 0] = omega[0] *cfg.normalization.obs_scales.ang_vel
    // obs[0, 1] = omega[1] *cfg.normalization.obs_scales.ang_vel
    // obs[0, 2] = omega[2] *cfg.normalization.obs_scales.ang_vel
    // obs[0, 3] = eu_ang[0] *cfg.normalization.obs_scales.quat
    // obs[0, 4] = eu_ang[1] *cfg.normalization.obs_scales.quat
    // obs[0, 5] = eu_ang[2] *cfg.normalization.obs_scales.quat
    // obs[0, 6] = cmd.vx * cfg.normalization.obs_scales.lin_vel
    // obs[0, 7] = cmd.vy * cfg.normalization.obs_scales.lin_vel
    // obs[0, 8] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
    // obs[0, 9:21] = (q-default_dof_pos) * cfg.normalization.obs_scales.dof_pos #g关节角度顺序依据修改为样机
    // obs[0, 21:33] = dq * cfg.normalization.obs_scales.dof_vel
    // obs[0, 33:45] = last_actions#上次控制指令
    #if 0
        cout<<"cmd:";
        cout<<request.command[0]<<" ";
        cout<<request.command[1]<<" ";
        cout<<request.command[2]<<" ";
        cout<<endl;
        cout<<"att:";
        cout<<request.eu_ang[0]<<" ";
        cout<<request.eu_ang[1]<<" ";
        cout<<request.eu_ang[2]<<" ";
        cout<<endl;
        cout<<"rate:";
        cout<<request.omega[0]<<" ";
        cout<<request.omega[1]<<" ";
        cout<<request.omega[2]<<" ";
        cout<<endl;
        cout<<"q:";
        for(int i=0;i<12;i++)
            cout<<request.q[i]<<" ";
        cout<<endl;
        cout<<"dq:";
        for(int i=0;i<12;i++)
            cout<<request.dq[i]<<" ";
        cout<<endl;    
        cout<<"trigger:"<<request.trigger<<" ";
        
    #endif
    // 将 data 转为 tensor 类型，输入到模型
    if(request.trigger==1){
        request.trigger=0;
        std::vector<float> obs;
        for(int i=0;i<10;i++)
            init_pos[i]=request.init_pos[i];
            // init_pos[0]=0.0;
            // init_pos[1]=-0.07;
            // init_pos[2]=0.57;
            // init_pos[3]=-1.12;
            // init_pos[4]=0.56;

            // init_pos[5]=0.0;
            // init_pos[6]=0.07;
            // init_pos[7]=0.57;
            // init_pos[8]=-1.12;
            // init_pos[9]=0.56;

            // init_pos[0]=0.0;
            // init_pos[1]=0.0;
            // init_pos[2]=0.0;
            // init_pos[3]=0.0;
            // init_pos[4]=0.0;

            // init_pos[5]=0.0;
            // init_pos[6]=0.0;
            // init_pos[7]=0.0;
            // init_pos[8]=0.0;
            // init_pos[9]=0.0;

            
        //---------------Push data into obsbuf--------------------
        obs.push_back(request.omega[0]*omega_scale);
        obs.push_back(request.omega[1]*omega_scale);
        obs.push_back(request.omega[2]*omega_scale);

        obs.push_back(request.eu_ang[0]*eu_ang_scale);
        obs.push_back(request.eu_ang[1]*eu_ang_scale);
        obs.push_back(request.eu_ang[2]*eu_ang_scale);

        // cmd
        float max = 1.0;
        float min = -1.0;

        cmd_x = cmd_x * (1 - smooth) + (std::fabs(request.command[0]) < dead_zone ? 0.0 : request.command[0]) * smooth;
        cmd_y = cmd_y * (1 - smooth) + (std::fabs(request.command[1]) < dead_zone ? 0.0 : request.command[1]) * smooth;
        cmd_rate = cmd_rate * (1 - smooth) + (std::fabs(request.command[2]) < dead_zone ? 0.0 : request.command[2]) * smooth;

        obs.push_back(cmd_x*lin_vel);//控制指令x
        obs.push_back(cmd_y*lin_vel);//控制指令y
        obs.push_back(cmd_rate*ang_vel);//控制指令yaw rate


        // pos q joint
        for (int i = 0; i < 10; ++i)
        {
            float pos = (request.q[i]  - init_pos[i])* pos_scale;//需要修改init pose
            obs.push_back(pos);
        }
        // vel q joint
        for (int i = 0; i < 10; ++i)
        {
            float vel = request.dq[i] * vel_scale;
            obs.push_back(vel);
        }
        // last action
        for (int i = 0; i < 10; ++i)
        {
            obs.push_back(action_temp[i]);// 
        }
        // std::cout<<("----------------obs---------------")<<std::endl;
        // cout<<obs<<endl;
        // std::cout<<("--------------------------------")<<std::endl;

        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor obs_tensor = torch::from_blob(obs.data(),{1,39},options).to(device);
        //----------------------------------------------------------------
        auto obs_buf_batch = obs_buf.unsqueeze(0);

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(obs_tensor.to(torch::kHalf));
        inputs.push_back(obs_buf_batch.to(torch::kHalf));
        
        //---------------------------网络推理----------------------------- Execute the model and turn its output into a tensor
        //cout<<"obs_tensor1:"<<endl<<obs_tensor<<endl;
        //std::cout<<("*****************")<<std::endl;
        //cout<<"obs_buf_batch:"<<endl<<obs_buf_batch<<endl;
        torch::Tensor action_tensor = model.forward(inputs).toTensor();
        action_buf = torch::cat({action_buf.index({ Slice(1,None),Slice()}),action_tensor},0);
        //cout<<"[action out]:"<<endl<<action_tensor<<endl;
        bool has_nan = false;
        for (float val : obs) {
            //cout << val << " ";
            if (std::isnan(val)) {
                has_nan = true;
            }
        }
        if (has_nan) {
            cout << "NaN detected in obs. Press any key to continue..." << endl;
            getchar(); // 等待键盘输入
        }

        //-----------------------------网络输出滤波--------------------------------
        torch::Tensor action_blend_tensor = 0.8*action_tensor + 0.2*last_action;
        last_action = action_tensor.clone();
    
        this->obs_buf = torch::cat({this->obs_buf.index({Slice(1, None), Slice()}), obs_tensor}, 0); // 历史观测移位
        // //obs_buf = torch::cat({obs_buf.index({Slice(1,None),Slice()}),obs_tensor},0);//历史观测移位
        // //----------------------------------------------------------------
        torch::Tensor action_raw = action_blend_tensor.squeeze(0);
        // move to cpu
        action_raw = action_raw.to(torch::kFloat32);
        action_raw = action_raw.to(torch::kCPU);
        // // assess the result
        auto action_getter = action_raw.accessor <float,1>();//bug
        for (int j = 0; j < 10; j++)
        {
            action[j] = limit(action_getter[j],-15,15)*0.2; 
            action_temp[j] = limit(action_getter[j],-15,15);//原始值
        }
        // action[0] = limit(action_getter[0],-15,15)*0.2; 
        // action_temp[0] = limit(action_getter[0],-15,15);//原始值

        // action[1] = limit(action_getter[1],-15,15)*0.2; 
        // action_temp[1] = limit(action_getter[1],-15,15);//原始值

        // action[2] = limit(action_getter[2],-15,15)*0.2; 
        // action_temp[2] = limit(action_getter[2],-15,15);//原始值

        // action[3] = limit(action_getter[3],-15,15)*0.25; 
        // action_temp[3] = limit(action_getter[3],-15,15);//原始值

        // action[4] = limit(action_getter[4],-15,15)*0.2; 
        // action_temp[4] = limit(action_getter[4],-15,15);//原始值

        // action[5] = limit(action_getter[5],-15,15)*0.2; 
        // action_temp[5] = limit(action_getter[5],-15,15);//原始值

        // action[6] = limit(action_getter[6],-15,15)*0.2; 
        // action_temp[6] = limit(action_getter[6],-15,15);//原始值

        // action[7] = limit(action_getter[7],-15,15)*0.2; 
        // action_temp[7] = limit(action_getter[7],-15,15);//原始值

        // action[8] = limit(action_getter[8],-15,15)*0.25; 
        // action_temp[8] = limit(action_getter[8],-15,15);//原始值

        // action[9] = limit(action_getter[9],-15,15)*0.2; 
        // action_temp[9] = limit(action_getter[9],-15,15);//原始值
    
        
        action_refresh=1;
    }
}

int RL_Tinymal_UDP::load_policy()
{   
    std::cout << model_path << std::endl;
    // load model from check point
    std::cout << "cuda::is_available():" << torch::cuda::is_available() << std::endl;
    device= torch::kCPU;
    if (torch::cuda::is_available()&&1){
        device = torch::kCUDA;
        printf("device = torch::kCUDA\n");
    }
    std::cout<<"device:"<<device<<endl;
    model = torch::jit::load(model_path);
    std::cout << "load model is successed!" << std::endl;
    model.to(device);
    std::cout << "LibTorch Version: " << TORCH_VERSION_MAJOR << "." 
              << TORCH_VERSION_MINOR << "." 
              << TORCH_VERSION_PATCH << std::endl;
    model.to(torch::kHalf);
    std::cout << "load model to device!" << std::endl;
    model.eval();
}
 
int RL_Tinymal_UDP::init_policy() {
    std::cout << "RL model thread start" << endl;
    cout << "cuda_is_available:" << torch::cuda::is_available() << endl;
    cout << "cudnn_is_available:" << torch::cuda::cudnn_is_available() << endl;

    model_path = "/home/rot/original_isaacgym/model_jitt.pt";
    load_policy();

    // 初始化记录
    action_buf = torch::zeros({history_length, 10}, device);
    obs_buf = torch::zeros({history_length, 39}, device);
    last_action = torch::zeros({1, 10}, device);

    action_buf.to(torch::kFloat32);
    obs_buf.to(torch::kFloat32);
    last_action.to(torch::kFloat32);

    // 初始化 action 和 action_temp 为 init_pos
    action.clear();
    action_temp.clear();
    prev_action.clear();
    for (int j = 0; j < 10; j++) {
        action.push_back(init_pos[j]);
        action_temp.push_back(init_pos[j]);
        prev_action.push_back(init_pos[j]);
    }

    // 热启动：初始化历史观测
    for (int i = 0; i < history_length; i++) {
        std::vector<float> obs;
        obs.push_back(0); // omega
        obs.push_back(0);
        obs.push_back(0);
        obs.push_back(0); // eu_ang
        obs.push_back(0);
        obs.push_back(0);
        obs.push_back(0); // cmd
        obs.push_back(0);
        obs.push_back(0);
        for (int j = 0; j < 10; ++j) {
            obs.push_back(0); // pos
        }
        for (int j = 0; j < 10; ++j) {
            obs.push_back(0); // vel
        }
        for (int j = 0; j < 10; ++j) {
            obs.push_back(0); // last action
        }
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor obs_tensor = torch::from_blob(obs.data(), {1, 39}, options).to(device);
        obs_buf = torch::cat({obs_buf.index({Slice(1, None), Slice()}), obs_tensor}, 0);
    }

    return 0;
}

bool RL_Tinymal_UDP::moveToInitPos(_msg_request request) {
    // 检查所有关节角度是否接近初始角度
    bool all_joints_ready = true;
    for (int i = 0; i < 10; i++) {
        float error = std::abs(request.q[i] - init_pos[i]);
        if (error > init_pos_tolerance) {
            all_joints_ready = false;
            break;
        }
    }

    // 如果所有关节都接近初始角度，返回 true 表示初始化完成
    if (all_joints_ready) {
        std::cout << "All joints have reached initial positions. Switching to RL control." << std::endl;
        return true;
    }

    // 否则，发送初始角度作为期望角度
    for (int i = 0; i < 10; i++) {
        msg_response.q_exp[i] = init_pos[i]; // 发送初始角度
    }
    action_refresh = 1; // 触发发送
    std::cout << "Sending initial positions: ";
    for (int i = 0; i < 10; i++) {
        std::cout << msg_response.q_exp[i] << " ";
    }
    std::cout << std::endl;

    return false;
}

int main(int argc, char** argv) {
    int sock_fd;
    sock_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock_fd < 0) {
        perror("Socket creation failed");
        exit(1);
    }

    struct sockaddr_in addr_serv;
    int len;
    memset(&addr_serv, 0, sizeof(addr_serv));
    addr_serv.sin_family = AF_INET;
#if 1
    string UDP_IP = "127.0.0.1"; // local test
    int SERV_PORT = 8888;
#else
    string UDP_IP = "192.168.10.55";
    int SERV_PORT = 8000;
#endif
    addr_serv.sin_addr.s_addr = inet_addr(UDP_IP.c_str());
    addr_serv.sin_port = htons(SERV_PORT);
    len = sizeof(addr_serv);

    int recv_num = 0, send_num = 0;
    char send_buf[500] = {0}, recv_buf[500] = {0};

    tinymal_rl.init_policy();
    for (int i = 0; i < 10; i++)
        msg_response.q_exp[i] = tinymal_rl.init_pos[i]; // 初始化时发送初始角度
    printf("Thread UDP RL-Tinker\n");

    while (1) {
        // 接收机器人状态
        recv_num = recvfrom(sock_fd, recv_buf, sizeof(recv_buf), MSG_WAITALL, (struct sockaddr *)&addr_serv, (socklen_t *)&len);
        if (recv_num > 0) {
            memcpy(&msg_request, recv_buf, sizeof(msg_request));

            // 初始化阶段：移动到初始角度
            if (!tinymal_rl.is_initialized) {
                tinymal_rl.is_initialized = tinymal_rl.moveToInitPos(msg_request);
            }
            // RL控制阶段
            else {
                tinymal_rl.handleMessage(msg_request);
            }
        }

        // 发送期望角度
        if (tinymal_rl.action_refresh) {
            tinymal_rl.action_refresh = 0;
            std::cout.precision(2);
            std::cout << "Sending q_exp: ";
            for (int i = 0; i < 10; i++) {
                std::cout << msg_response.q_exp[i] << " ";
            }
            std::cout << std::endl;
        }
        memcpy(send_buf, &msg_response, sizeof(msg_response));
        send_num = sendto(sock_fd, send_buf, sizeof(msg_response), MSG_WAITALL, (struct sockaddr *)&addr_serv, len);

        if (send_num < 0) {
            perror("Robot sendto error:");
            exit(1);
        }

        usleep(2 * 1000); // 2ms 休眠
    }
    return 0;
}