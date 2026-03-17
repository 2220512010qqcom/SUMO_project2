

# 架构设计：
# sumoController ：专门用来统计sumo环境中的车辆信息，并且打印详细日志到文件中
# agent1-4：四个路口智能体，每个智能体都有自己的即时缓冲区，可以实现训练和预测功能，即使缓冲区使用训练轮次作为时间戳，训练时使用全局缓冲区中的数据
# 每个若干个轮次，全局缓冲区收集各个智能体的即时缓冲区数据，按照时间戳对齐
#
#
#
#
#
#TODO:1.消除急救车警告 2.思考多层缓冲架构及其实现
import os
import random
from matplotlib import pyplot as plt
import torch
from myscripts.sumoController import SumoController
from myscripts.myagent import myAgent
from myscripts.logger import myLogger

class mytrainer:
    def __init__(self):
        self.episode = 0
        self.max_episodes = 3000          # 最大训练回合数
        self.step_per_episode = 3600     # 每个回合的最大步数

        self.plot_dir = './outputs/output1'
        self.total_reward = []
        self.max_wait_time = []
        self.emer_max_wait_time = []
        self.avarage_wait_times = []   # 每轮次的平均等待时间
        self.max_wait_time_in_one_episode = 0
        self.total_reward_in_one_episode = 0       #一个轮次中的奖励之和
        self.emer_max_wait_time_in_one_episode = 0
        self.total_waiting_time = 0  # 车辆等待的总时间
        self.total_vehical_count = 0


        self.sumo_controller = SumoController()     #初始化一个sumo控制器
        self.sumo_controller.start_sumo()
        self.lane_state_num = 2         # 每个车道有多少状态信息要计算
        self.ignore_arg_num = 2         # 额外叠加的信号灯，统计信号灯的相位和时长信息，不计入奖励函数，所以叫ignore_arg
        self.agent_list = []
        self.duration_options = [10, 40,70]  # 动作对应的持续时间选项
        self.initialize_agents()        # 根据交通灯数量初始化智能体，并设置每个智能体控制的车道列表
        pass


    def initialize_agents(self):
        '''根据交通灯ID列表初始化智能体，并设置每个智能体控制的车道列表'''
        tls_IDlist = self.sumo_controller.get_trafficlight_IDlist()
        for tls_id in tls_IDlist:
            controlled_links = self.sumo_controller.get_controlled_lanes(tls_id)   # [[NS],[EW]]
            # 计算输入维度
            num_links = len(controlled_links[0]) + len(controlled_links[1])
            input_dim = num_links * self.lane_state_num + self.ignore_arg_num
            input_dim = input_dim * 4
            self.agent_list.append(myAgent(tls_id,input_dim))
            self.agent_list[-1].set_controlled_lanes(controlled_links)
        self.agent_num = len(self.agent_list)
        print(f"智能体初始化完成，数量为{self.agent_num}")
        return
    
    def set_logger(self, logger):
        self.logger = logger
        self.logger.intialize_agent_num(self.agent_list)

    def train_step(self):
        '''进行一步仿真，达到智能体变灯时间则进行经验存储'''
        current_time = self.sumo_controller.get_current_time()
        # 对每个智能体进行判断
        for agent in self.agent_list:
            dif_time = current_time - agent.start_time
            # 当时间到达持续时间时，选择新动作
            if dif_time >= agent.duration:
                # print(f"current time is {current_time} and curren agent is{agent.id}")
                # 选择新动作,获取当前状态向量
                # current_states = []
                # NS_current_state = []
                # EW_current_state = []
                # # 获取所有南北方向受控车道的车辆信息,添加到状态向量
                # for lane in agent.controlled_lanes[agent.NS_lanes_index]:
                #     state = self.sumo_controller.get_vehicles_in_area(lane)   # 获取各个车道的车辆信息
                #     # print(state)
                #     NS_current_state.extend([state["vehicle_count"], state["max_waiting_time"],state['emergency_count'],state['emergency_min_speed'],state['emergency_max_wait_time']])
                #     self.max_wait_time_in_one_episode = max(self.max_wait_time_in_one_episode,state["max_waiting_time"])  # 更新最大等待时间的统计数据，用于画图
                #     self.emer_max_wait_time_in_one_episode = max(self.emer_max_wait_time_in_one_episode,state['emergency_max_wait_time'])
                #     self.total_waiting_time += state['total_waiting_time']
                #     self.total_vehical_count += state['vehicle_count']
                # # 获取所有东西方向受控车道的车辆信息,添加到状态向量
                # for lane in agent.controlled_lanes[agent.EW_lanes_index]:
                #     state = self.sumo_controller.get_vehicles_in_area(lane)   # 获取各个车道的车辆信息
                #     EW_current_state.extend([state["vehicle_count"], state["max_waiting_time"],state['emergency_count'],state['emergency_min_speed'],state['emergency_max_wait_time']])
                #     self.max_wait_time_in_one_episode = max(self.max_wait_time_in_one_episode,state["max_waiting_time"])  # 更新最大等待时间的统计数据，用于画图
                #     self.emer_max_wait_time_in_one_episode = max(self.emer_max_wait_time_in_one_episode,state['emergency_max_wait_time'])
                #     self.total_waiting_time += state['total_waiting_time']
                #     self.total_vehical_count += state['vehicle_count']
                # current_states.extend(NS_current_state)
                # current_states.extend(EW_current_state)
                # # 添加相位和持续时间信息到状态向量中
                # current_states.extend([agent.phase, agent.duration])  
                
                # print(len(current_states))
                current_states = self.get_agent_current_state(agent) # 使用全局信息训练数据
                prev_state = agent.last_state
                prev_action = agent.last_action
                # 计算奖励，这里简单使用负的车辆总数作为奖励
                # print(current_states)
                reward = self.reward_function(current_states, ignore_args_number=self.ignore_arg_num)

                # 添加奖励到智能体的奖励列表中
                agent.add_reward(reward)

                # print(reward)
                current_states = self.get_global_state() # 使用全局信息训练数据
                self.total_reward_in_one_episode += reward  # 更新统计数据，用于统计画图
                next_state = current_states
                # print(f"current_states: {current_states}, reward: {reward}")
                experience = (prev_state, prev_action, reward, next_state)
                # 存储经验到即时缓冲区
                if prev_state is not None and prev_action is not None:
                    agent.store_experience(experience)
                    # 当缓冲区大小达到数量要求时对神经网络进行训练
                    if agent.immiediate_buffer_length() > agent.train_experience_number and agent.immiediate_buffer_length() % 4 == 0:
                        agent.update_behavior_network()
                    if agent.immiediate_buffer_length() > agent.train_experience_number and agent.immiediate_buffer_length() % 16 == 0:
                        agent.update_target_network()

                
                state_tensor = torch.tensor(current_states, dtype=torch.float32).unsqueeze(0)
                action = agent.select_action(state_tensor)
                # 智能体设置新的相位和持续时间
                duration = self.duration_options[action]
                agent.set_phase(1 - agent.phase)  # 切换相位
                agent.set_duration(duration)
                agent.set_start_time(current_time)
                agent.set_last_state(current_states)
                agent.set_last_action(action)

                # 将动作实施到环境中
                self.sumo_controller.apply_agent(agent)
                

        self.sumo_controller.step_sumo()
        return
    
    def get_agent_current_state(self, agent):
        '''获取一个智能体当前的状态向量'''
        current_state = []
        # 获取所有受控车道的车辆信息,添加到状态向量
        for lane in agent.controlled_lanes[agent.NS_lanes_index]:
            state = self.sumo_controller.get_vehicles_in_area(lane)   # 获取各个车道的车辆信息
            current_state.extend([state["vehicle_count"], state["max_waiting_time"],state['emergency_count'],state['emergency_min_speed'],state['emergency_max_wait_time']])
        for lane in agent.controlled_lanes[agent.EW_lanes_index]:
            state = self.sumo_controller.get_vehicles_in_area(lane)   # 获取各个车道的车辆信息
            current_state.extend([state["vehicle_count"], state["max_waiting_time"],state['emergency_count'],state['emergency_min_speed'],state['emergency_max_wait_time']])
        # 添加相位和持续时间信息到状态向量中
        current_state.extend([agent.phase, agent.duration])  
        return current_state

    def get_global_state(self):
        current_states = []
        for agent in self.agent_list:
            NS_current_state = []
            EW_current_state = []

            # 获取所有受控车道的车辆信息,添加到状态向量
            for lane in agent.controlled_lanes[agent.NS_lanes_index]:
                state = self.sumo_controller.get_vehicles_in_area(lane)   # 获取各个车道的车辆信息
                # NS_current_state.extend([state["vehicle_count"], state["max_waiting_time"],state['emergency_count'],state['emergency_min_speed'],state['emergency_max_wait_time']])
                NS_current_state.extend([state["vehicle_count"],state['emergency_count']])
            # 获取所有受控车道的车辆信息,添加到状态向量
            for lane in agent.controlled_lanes[agent.EW_lanes_index]:
                state = self.sumo_controller.get_vehicles_in_area(lane)   # 获取各个车道的车辆信息
                # EW_current_state.extend([state["vehicle_count"], state["max_waiting_time"],state['emergency_count'],state['emergency_min_speed'],state['emergency_max_wait_time']])
                EW_current_state.extend([state["vehicle_count"],state['emergency_count']])
            
            current_states.extend(NS_current_state)
            current_states.extend(EW_current_state)
            # 添加相位和持续时间信息到状态向量中
            current_states.extend([agent.phase, agent.duration]) 
        return current_states


    # state为[车数量，等待时间，急救车数量，急救车最小速度, 急救车最大等待时间 ,ignore_args]
    def reward_function(self, state, ignore_args_number=0):
        if ignore_args_number > 0:
            relevant_state = state[:-ignore_args_number]
        if len(relevant_state) < 5:
            raise ValueError("State must have 5 elements: [veh_count, wait, emer_cnt, emer_min_speed, emer_max_wait]")

        # === 归一化参数 ===
        MAX_VEH = 30.0
        MAX_WAIT = 100.0
        MAX_EMER = 5.0
        MAX_EMER_WAIT = 100.0
        MAX_SPEED = 100.0  # m/s 
        # 权重（现在可以设为相近值，因为已归一化）
        W_NORMAL = 1.0
        W_EMERGENCY = 10.0  # 仍可略高，但不用 20 那么夸张
        normal_cost = 0
        emergency_cost = 0
        for i in range(0,len(relevant_state),5):

            veh_count, wait_time, emer_count, emer_speed, emer_wait_time = relevant_state[i:i+5]
            # 归一化到 [0, 1]
            norm_veh = min(veh_count / MAX_VEH, 1.0) * 10
            norm_wait = min(wait_time / MAX_WAIT, 1.0)
            norm_emer_cnt = min(emer_count / MAX_EMER, 1.0)* 10
            norm_emer_wait = min(emer_wait_time / MAX_EMER_WAIT, 1.0) * 3
            norm_emer_speed = min(emer_speed / MAX_SPEED, 1.0)  # 注意：速度越大越好
            normal_cost += W_NORMAL * (norm_veh *  norm_wait)
            # emergency_cost += W_EMERGENCY * norm_emer_cnt * ( norm_emer_wait * 5 - norm_emer_speed)
            emergency_cost += W_EMERGENCY * norm_emer_cnt * norm_emer_wait * 5 
            # print(normal_cost,emergency_cost,norm_emer_speed)

        total_reward = -(normal_cost + emergency_cost)
        return total_reward

    def train(self):
        for ep in range(self.max_episodes):
            self.episode += 1
            self.sumo_controller.reset_simulation()

            # 每个轮次进行若干步的训练
            for _ in range(self.step_per_episode):
                self.train_step()
                
            # 每个回合结束后，可以在这里进行经验回放和网络更新
            for agent in self.agent_list:
                print(f"Agent {agent.id} Episode {self.episode} Immediate Buffer Size: {len(agent.immediate_buffer)}")
                agent.reset_all()
                agent.update_epsilon()
                print(f"当前探索率为{agent.epsilon}")
            
            # 保存每个轮次的奖励之和
            self.total_reward.append(self.total_reward_in_one_episode)
            self.total_reward_in_one_episode = 0
            self.max_wait_time.append(self.max_wait_time_in_one_episode)
            self.max_wait_time_in_one_episode = 0
            self.emer_max_wait_time.append(self.emer_max_wait_time_in_one_episode)
            self.emer_max_wait_time_in_one_episode = 0
            self.avarage_wait_times.append(self.total_waiting_time/self.total_vehical_count)
            self.total_vehical_count = 0
            self.total_waiting_time = 0


            # 计算每轮次各个智能体获得的总奖励和
            for agent in self.agent_list:
                episode_reward = sum(agent.get_reward_list())
                print(f"Agent {agent.id} Episode {self.episode} Total Reward: {episode_reward}")
                agent.clear_reward_list()  # 清空奖励列表，为下一轮次做准备
                self.logger.log_agent_rewards(agent, episode_reward)  # 将奖励数据记录到logger中

            # 每10个轮次进行一次画图
            if ep % 10 == 0 :
                self.logger.plot_rewards(self.total_reward)
                self.logger.plot_max_wait_time(self.max_wait_time)
                self.logger.plot_emer_max_wait_time(self.emer_max_wait_time)
                self.logger.plot_avarage_wait_time(self.avarage_wait_times)
                for agent in self.agent_list:
                    agent.save_weight_args()
                self.logger.save_rewards_to_file()  # 将奖励数据保存到文件中
        
        return





# 定义main函数 
def main():
    trainer = mytrainer()
    logger = myLogger(trainer.plot_dir)
    trainer.set_logger(logger)
    
    trainer.train()
    print("*" * 60)
    print("训练完毕！")
    return
    
if __name__ == '__main__':
    main()