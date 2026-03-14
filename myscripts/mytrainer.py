

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

        self.agent_list = []
        self.duration_options = [10, 40,10,40]  # 动作对应的持续时间选项
        self.initialize_agents()        # 根据交通灯数量初始化智能体，并设置每个智能体控制的车道列表
        self.duration = [0] * self.agent_num          # 每个智能体当前剩余的变灯时间,初始为0，表示可以立即选择动作
        self.total_reward = [0] * self.agent_num   # 每个轮次的总奖励列表

        pass


    def initialize_agents(self):
        '''根据交通灯ID列表初始化智能体，并设置每个智能体控制的车道列表'''
        tls_IDlist = self.sumo_controller.get_trafficlight_IDlist()
        for tls_id in tls_IDlist:
            controlled_links = self.sumo_controller.get_controlled_lanes(tls_id)   # [[NS],[EW]]
            # 计算输入维度
            num_links = len(controlled_links[0]) + len(controlled_links[1])
            input_dim = 32  # 每条车道的状态信息维度
            self.agent_list.append(myAgent(tls_id,input_dim))
            self.agent_list[-1].set_controlled_lanes(controlled_links)
        self.agent_num = len(self.agent_list)
        print(f"智能体初始化完成，数量为{self.agent_num}")
        return
    
    def set_logger(self, logger):
        self.logger = logger
        self.logger.intialize_agent_num(self.agent_list)
    
    def should_change_light(self):
        '''判断是否应该更新智能体的动作'''
        for duration in self.duration:
            if duration == 0:
                return True
        return False
    def update_duration(self):
        '''更新每个智能体的剩余变灯时间'''
        for i in range(self.agent_num):
            if self.duration[i] > 0:
                self.duration[i] -= 1
            else:
                self.duration[i] = 0

    def change_light(self,current_state,current_global_reward):
        '''更新指定智能体的变灯时间'''
        for i in range(self.agent_num):
            if self.duration[i] == 0:  # 更新持续时间 变灯智能体存储经验，并更新智能体网络参数
                current_reward= sum(current_global_reward) + current_global_reward[i] * 3  # 当前智能体的奖励在全局奖励中的权重更大一些
                action = self.agent_list[i].select_action(torch.tensor(current_state, dtype=torch.float32))
                last_state = self.agent_list[i].last_state
                last_action = self.agent_list[i].last_action
                if last_state is not None and last_action is not None:
                    self.agent_list[i].store_experience([last_state, last_action, current_reward, current_state])
                    self.agent_list[i].update_behavior_network()  # 在每次存储经验后进行训练
                self.agent_list[i].last_state = current_state
                self.agent_list[i].last_action = action
                self.duration[i] = self.duration_options[action]
                if action < len(self.duration_options) // 2: self.agent_list[i].phase = 0  # 南北绿
                else: self.agent_list[i].phase = 1  # 东西绿
                # self.agent_list[i].phase = 1 # 东西绿
                self.sumo_controller.apply_agent(self.agent_list[i])  # 将智能体的动作应用到sumo环境中
                self.agent_list[i].add_reward(current_reward)  # 将当前奖励添加到智能体的奖励列表中
                print(f"智能体{self.agent_list[i].id}选择了动作{action}，持续时间为{self.duration_options[action]}秒，当前奖励为{current_reward}")
                # self.logger.log(f"智能体{self.agent_list[i].id}选择了动作{action}，持续时间为{self.duration_options[action]}秒，当前奖励为{current_reward}")
                self.logger.plot_agent_rewards(self.agent_list[i].id, self.agent_list[i].reward_list)  # 记录智能体的奖励数据到日志中
                self.logger.log_agent_state(self.agent_list[i].id, current_state)  # 记录智能体的状态信息到日志中
    

    def step_to_next_light_change(self):
        '''进行仿真直到下一个智能体需要变灯'''
        while not self.should_change_light():
            self.sumo_controller.step_sumo()
            self.update_duration()

    def train_step(self):
        '''进行一步仿真，达到智能体变灯时间则进行经验存储'''        
        self.step_to_next_light_change()  # 进行仿真直到下一个智能体需要变灯
        # 使用全局信息训练数据 [南北车辆等级]，[东西车辆等级]，[当前相位]，[剩余变灯时间] * 智能体数量
        current_state, current_global_reward = self.get_global_state_and_reward()
        self.change_light(current_state, current_global_reward)  # 更新智能体的变灯时间,并顺便将经验保存到智能体的存储空间
        self.sumo_controller.step_sumo()        #  进行下一步仿真模拟


    def culculate_traffic_index(self, count,weight,average_speed,basic_index,max_waiting_time):
        '''计算交通指数的函数,作为当前状态的一部分'''
        return (count * weight) / (average_speed + basic_index) + max_waiting_time
        
    def get_lane_traffic_index(self, agent, is_NS):
        current_state = [0,0,0]  # 每个车道的状态信息 [车辆数量，急救车数量]
        basic_index = 2  # 基础平衡因子，防止分母为0
        vehicle_weight = 1.0
        emergency_weight = 10.0
        if is_NS: lanes_choice = agent.NS_lanes_index
        else: lanes_choice = agent.EW_lanes_index
        lane_num = len(agent.controlled_lanes[lanes_choice])
         # 获取所有受控车道的车辆信息,添加到状态向量
        for lane in agent.controlled_lanes[lanes_choice ]:
            state = self.sumo_controller.get_vehicles_in_area(lane)   # 获取各个车道的车辆信息 
            occupancy = state["occupancy"]  # 车辆占有率
            vehicle_count = state["vehicle_count"]  # 车辆数量
            emergency_count = state['emergency_count']  # 急救车数量
            average_speed = state['average_speed']  # 平均速度
            vehicle_max_wait_time = state['max_waiting_time']  # 车辆最大等待时间
            emergency_max_wait_time = state['emergency_max_wait_time']  # 急救
            current_state[0] += occupancy # 车辆占有率求和,反应整体拥堵情况
            current_state[1] += self.culculate_traffic_index(vehicle_count, vehicle_weight, average_speed, basic_index, vehicle_max_wait_time)  # 车辆交通指数求和，反应整体交通状况
            current_state[2] += self.culculate_traffic_index(emergency_count, emergency_weight, average_speed, basic_index, emergency_max_wait_time)  # 急救车交通指数求和，反应急救车的紧急程度
        
        for i in range(len(current_state)):
            current_state[i] /= lane_num  # 取平均值，反应每条车道的平均状况
        return current_state


    def get_agent_lane_traffic_index(self, agent):
        '''计算智能体控制车道的交通指数，作为状态输入的一部分'''
        NS_state = self.get_lane_traffic_index(agent, True)  # 计算南北车道的交通指数
        EW_state = self.get_lane_traffic_index(agent, False)  # 计算东西车道的交通指数
        print(f"智能体{agent.id}的NS车道状态：{NS_state}，EW车道状态：{EW_state}")
        return NS_state, EW_state

    def get_global_state_and_reward(self):
        '''遍历所有车道，获取全局状态和奖励'''
        current_states = []
        current_reward = []
        for agent in self.agent_list:
            NS_state, EW_state = self.get_agent_lane_traffic_index(agent)  # 获取智能体控制车道的交通指数
            current_reward.append(self.reward_function(NS_state) + self.reward_function(EW_state))  # 计算当前智能体的奖励并累加到全局奖励中
            current_states.extend(NS_state)
            current_states.extend(EW_state)
            # 添加相位和持续时间信息到状态向量中
            current_states.extend([agent.phase, agent.duration]) 
        print(f"当前全局状态：{current_states}，当前全局奖励：{current_reward}")
        print(f"全局状态维度：{len(current_states)}")
        return current_states, current_reward


    # state为[车数量，等待时间，急救车数量，急救车最小速度, 急救车最大等待时间 ,ignore_args]
    def reward_function(self, state):
        
        return 1 - state[0]

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
                self.logger.plot_agent_rewards(agent.id, agent.reward_list)  # 绘制当前智能体的奖励曲线
                agent.reset_all()
                agent.update_epsilon()
                print(f"当前探索率为{agent.epsilon}")
                
            
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