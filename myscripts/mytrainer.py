

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
import numpy as np
from myscripts.sumoController import SumoController
from myscripts.myagent import myAgent
from myscripts.logger import myLogger

class mytrainer:

    def __init__(self):
        self.episode = 0
        self.max_episodes = 100          # 最大训练回合数
        self.step_per_episode = 800     # 每个回合的最大步数

        self.plot_dir = './outputs/output2'
        self.sumo_controller = SumoController()     #初始化一个sumo控制器
        self.sumo_controller.start_sumo()
        self.lane_state_num = 2         # 每个车道有多少状态信息要计算

        self.agent_list = []
        # self.duration_options = [10, 40,10,40]  # 动作对应的持续时间选项
        self.duration_options = [10, 40]  # 动作对应的持续时间选项
        self.initialize_agents()        # 根据交通灯数量初始化智能体，并设置每个智能体控制的车道列表
        self.duration = [0] * self.agent_num          # 每个智能体当前剩余的变灯时间,初始为0，表示可以立即选择动作

    def initialize_agents(self):
        '''根据交通灯ID列表初始化智能体，并设置每个智能体控制的车道列表'''
        tls_IDlist = self.sumo_controller.get_trafficlight_IDlist()
        for tls_id in tls_IDlist:
            controlled_links = self.sumo_controller.get_controlled_lanes(tls_id)   # [[NS],[EW]]
            # 计算输入维度
            num_links = len(controlled_links[0]) + len(controlled_links[1])
            # input_dim = 26  # 每条车道的状态信息维度 6.net版本
            input_dim = 50  # 每条车道的状态信息维度 
            self.agent_list.append(myAgent(tls_id,input_dim))
            self.agent_list[-1].set_controlled_lanes(controlled_links)
            self.agent_list[-1].init_RV_list(num_links)
        self.agent_num = len(self.agent_list)
        print(f"智能体初始化完成，数量为{self.agent_num}")
        return

    def set_logger(self, logger):
        self.logger = logger
        self.logger.intialize_agent_num(self.agent_list)

    # --- 仿真主流程 ---
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
                agent.update_target_network()
                print(f"当前探索率为{agent.epsilon}")
        return

    def train_step(self):
        '''进行一步仿真，达到智能体变灯时间则进行经验存储'''
        selected_agent_indexs = self.step_to_next_light_change()  # 进行仿真直到下一个智能体需要变灯
        # 获取需要变灯智能体的状态信息state 一维数组 
        selected_agent_states = self.get_selected_agents_state(selected_agent_indexs)
        # print(selected_agent_states)
        selected_agent_actions = self.get_selected_agents_action(selected_agent_indexs,selected_agent_states)
        selected_agent_cur_RV_list = self.get_selected_agents_RV_list(selected_agent_indexs)
        # selected_agent_cur_RP_list = self.get_selected_agents_RV_list(selected_agent_indexs) TODO
        # selected_agent_cur_RE_list = self.get_selected_agents_RV_list(selected_agent_indexs) TODO
        selected_agent_RV_rewards = self.get_selected_agents_RV_reward(selected_agent_indexs,selected_agent_cur_RV_list)
        self.update_RV_list(selected_agent_indexs,selected_agent_cur_RV_list)  # 更新智能体的RV列表
        selected_agent_rewards = self.get_selected_agents_reward(selected_agent_RV_rewards)  # 计算总体奖励值
        for i, idx in enumerate(selected_agent_indexs):
            self.agent_list[idx].add_reward(selected_agent_rewards[i])
        self.store_experience(selected_agent_indexs, selected_agent_states, selected_agent_rewards, selected_agent_actions)  # 将经验存储到对应智能体的即时缓冲区中
        self.change_light(selected_agent_indexs,selected_agent_states,selected_agent_actions)  # 更新智能体的变灯时间,并顺便将经验保存到智能体的存储空间
        self.sumo_controller.step_sumo()        #  进行下一步仿真模拟
        for agent in self.agent_list:
            agent.update_behavior_network()


    def step_to_next_light_change(self):
        '''进行仿真直到下一个智能体需要变灯，返回此次需要变灯的智能体下标数组'''
        while not self.should_change_light():
            self.sumo_controller.step_sumo()
            self.update_duration()
        agent_indexs = []  #存储需要变灯的智能体下标
        for i in range(self.agent_num):
            if self.duration[i] == 0:
                agent_indexs.append(i)
        return agent_indexs

        

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

    def change_light(self,selected_agent_indexs,selected_agent_states,selected_agent_actions):
        '''更新指定智能体的变灯时间'''
        selected_agent_num = len(selected_agent_indexs)
        for i in range(selected_agent_num):
            index = selected_agent_indexs[i]
            action = selected_agent_actions[i]
            self.duration[index] = self.duration_options[action]
            # 注意，这里一定会改变红绿灯的方向
            if self.agent_list[index].phase == 1:  # 如果原本东西绿   
                self.agent_list[index].phase = 0  # 修改为南北绿   
            else:    
                self.agent_list[index].phase = 1         
            self.sumo_controller.apply_agent(self.agent_list[index])  # 将智能体的动作应用到sumo环境中
   

    
    # --- 状态与奖励相关 ---
    #TODO 添加相邻智能体的状态信息
    def get_selected_agents_state(self,selected_agent_indexs):
        '''遍历所有车道，获取全局状态和奖励'''
        current_states = []
        for i in selected_agent_indexs:
            current_lane_states = []
            NS_state, EW_state = self.get_agent_lane_traffic_index(self.agent_list[i])  # 获取智能体控制车道的交通指数
            current_lane_states.extend(NS_state)
            current_lane_states.extend(EW_state)
            # 添加相位和持续时间信息到状态向量中
            self.agent_list[i].duration = self.duration[i]
            current_lane_states.extend([self.agent_list[i].phase, self.agent_list[i].duration])
            current_states.append(current_lane_states)
        return current_states


    def get_agent_lane_traffic_index(self, agent):
        '''计算智能体控制车道的交通指数，作为状态输入的一部分'''
        NS_state = self.get_lane_traffic_index(agent, True)  # 计算南北车道的交通指数
        EW_state = self.get_lane_traffic_index(agent, False)  # 计算东西车道的交通指数
        # print(f"智能体{agent.id}的NS车道状态：{NS_state}，EW车道状态：{EW_state}")
        return NS_state, EW_state

    def get_lane_traffic_index(self, agent, is_NS):
        current_state = []  # 每个车道的状态信息 [车辆数量，总等待时间，最大等待时间]
        if is_NS:
            lanes_choice = agent.NS_lanes_index
        else:
            lanes_choice = agent.EW_lanes_index
        lane_num = len(agent.controlled_lanes[lanes_choice])
        # 获取所有受控车道的车辆信息,添加到状态向量
        for lane in agent.controlled_lanes[lanes_choice]:
            current_lane_state = [0,0,0]
            state = self.sumo_controller.get_vehicles_in_area(lane)  # 获取各个车道的车辆信息
            vehicle_count = state["vehicle_count"]  # 车辆数量
            vehicle_total_waiting_time = state["total_waiting_time"]  # 车辆数量
            vehicle_max_wait_time = state['max_waiting_time']  # 车辆最大等待时间
            current_lane_state[0] += vehicle_count  # 车辆数量
            current_lane_state[1] += vehicle_total_waiting_time  # 总体等待时间 
            current_lane_state[2] = max(current_lane_state[2],vehicle_max_wait_time)
            current_state.extend(current_lane_state)
        return current_state

    def culculate_traffic_index(self, count, weight, average_speed, basic_index, max_waiting_time):
        '''计算交通指数的函数,作为当前状态的一部分'''
        return (count * weight) / (average_speed + basic_index) + max_waiting_time

    def reward_function(self, state):
        # state为[车数量，等待时间，急救车数量，急救车最小速度, 急救车最大等待时间 ,ignore_args]
        return 1 - state[0] - state[1] - state[2]  # 车辆数量和等待时间越多，奖励越低；急救车数量越多，奖励大幅降低

    def get_selected_agents_rewards(self,selected_agent_indexs,selected_agent_states):
        '''根据当前状态计算上一步的奖励'''
        for i in selected_agent_indexs:
            last_RV_list = self.agent_list[i].RV_list

#    [ [ [lane1],[lane2],...], [],[] ] 
#      ----------------------
#              ⬆
#        这是一个智能体的RV_list 不同智能体对应着不同的列表，访问时需要使用下标取出智能体的RV_list再做下一步的计算
    def get_selected_agents_RV_list(self,selected_agent_indexs):
        '''获取当前智能体的RV列表'''
        selected_agent_RV_list = []  
        for i in selected_agent_indexs:
            cur__agent_RV_list = []
            for lane in self.agent_list[i].controlled_lanes[self.agent_list[i].NS_lanes_index]:  # 遍历智能体控制的南北车道
                current_lane_state = [0,0]
                state = self.sumo_controller.get_vehicles_in_area(lane)  # 获取各个车道的车辆信息
                vehicle_count = state["vehicle_count"]  # 车辆数量
                vehicle_total_waiting_time = state["total_waiting_time"]  # 车辆数量
                vehicle_max_wait_time = state['max_waiting_time']  # 车辆最大等待时间
                current_lane_state[0] += vehicle_count  # 车辆数量
                current_lane_state[1] += vehicle_total_waiting_time  # 总体等待时间 
                cur__agent_RV_list.append(current_lane_state)
            for lane in self.agent_list[i].controlled_lanes[self.agent_list[i].EW_lanes_index]:  # 遍历智能体控制的东西车道
                current_lane_state = [0,0,0]
                state = self.sumo_controller.get_vehicles_in_area(lane)  # 获取各个车道的车辆信息
                vehicle_count = state["vehicle_count"]  # 车辆数量
                vehicle_total_waiting_time = state["total_waiting_time"]  # 车辆数量
                vehicle_max_wait_time = state['max_waiting_time']  # 车辆最大等待时间
                current_lane_state[0] += vehicle_count  # 车辆数量
                current_lane_state[1] += vehicle_total_waiting_time  # 总体等待时间 
                cur__agent_RV_list.append(current_lane_state)
            selected_agent_RV_list.append(cur__agent_RV_list)
        return selected_agent_RV_list

    def get_selected_agents_RV_reward(self,selected_agent_indexs,selected_agent_cur_RV_list):
        ''''''
        selected_agent_num = len(selected_agent_indexs)
        RV_reward_list = []
        for i in range(selected_agent_num):
            agent_index = selected_agent_indexs[i]
            last_RV_list = self.agent_list[agent_index].RV_list
            cur_RV_list = selected_agent_cur_RV_list[i]
            RV_reward = 0
            VT_max_last = 0
            VT_max_cur = 0
            for j in range(len(cur_RV_list)):
                VT_diff_sum = 0
                VN_diff_sum = 0
                VT_diff_sum += cur_RV_list[j][0] -  last_RV_list[j].VT 
                VN_diff_sum += cur_RV_list[j][1] - last_RV_list[j].NT
                VT_max_last = max(VT_max_last,last_RV_list[j].VT)
                VT_max_cur = max(VT_max_cur,cur_RV_list[j][0])
                if VN_diff_sum == 0:
                    VN_diff_sum = 1
            RV_reward += VT_diff_sum / VN_diff_sum    ###  没啥用？
            RV_reward += VT_max_last - VT_max_cur  
            RV_reward = - VT_max_cur   ###  尝试用等待时间代替奖励函数   
            RV_reward_list.append(RV_reward)
        return RV_reward_list
    
    def update_RV_list(self,selected_agent_indexs,selected_agent_cur_RV_list):
        selected_agent_num = len(selected_agent_indexs)
        for i in range(selected_agent_num):
            index = selected_agent_indexs[i]
            num = len(self.agent_list[index].RV_list)
            agent_cur_RV_list = selected_agent_cur_RV_list[i]
            for j in range(num):
                self.agent_list[index].RV_list[j].VT = agent_cur_RV_list[j][0]
                self.agent_list[index].RV_list[j].NT = agent_cur_RV_list[j][1]
    
    def get_selected_agents_reward(self,selected_agent_RV_rewards):
        '''根据各个分项奖励，加权形成智能体奖励'''
        # TODO 增加RP RE 两个分项
        return selected_agent_RV_rewards

    #TODO:增加一个函数，专门用来计算RP奖励和RE奖励，输入是智能体的RV列表和之前存储的RP RE列表，输出是当前的RP RE奖励值，然后在get_selected_agents_reward函数中进行加权计算得到最终奖励值
    def store_experience(self, selected_agent_indexs, selected_agent_states, selected_agent_rewards, selected_agent_actions):
        selected_agent_num = len(selected_agent_indexs)
        for i in range(selected_agent_num):
            index = selected_agent_indexs[i]
            agent = self.agent_list[index]
        
            # 第一次执行时跳过（没有上一步的状态）
            if agent.last_state is None:
                agent.last_state = selected_agent_states[i]
                agent.last_action = selected_agent_actions[i]
                continue
        
            experience = (agent.last_state, agent.last_action, selected_agent_rewards[i], selected_agent_states[i])
            agent.store_experience(experience)
            agent.last_state = selected_agent_states[i]
            agent.last_action = selected_agent_actions[i]

    def get_selected_agents_action(self,selected_agent_indexs,selected_agent_states):
        selected_agent_actions = []
        selected_agent_num = len(selected_agent_indexs)
        for i in range(selected_agent_num):
            index = selected_agent_indexs[i]
            action = self.agent_list[index].select_action(torch.tensor(selected_agent_states[i], dtype=torch.float32))
            selected_agent_actions.append(action)
        return selected_agent_actions
    
    #新增方法收集数据以便于可视化
    def collect_traffic_data(self):
        """收集当前交通数据用于可视化"""
        queue_data = {}
        waiting_data = {}
        
        for agent in self.agent_list:
            ns_queues = 0
            ew_queues = 0
            ns_waiting = 0
            ew_waiting = 0
            ns_count = 0
            ew_count = 0
            
            # 收集NS方向数据
            for lane in agent.controlled_lanes[agent.NS_lanes_index]:
                state = self.sumo_controller.get_vehicles_in_area(lane)
                ns_queues += state["vehicle_count"]
                ns_waiting += state["total_waiting_time"]
                ns_count += 1
            
            # 收集EW方向数据
            for lane in agent.controlled_lanes[agent.EW_lanes_index]:
                state = self.sumo_controller.get_vehicles_in_area(lane)
                ew_queues += state["vehicle_count"]
                ew_waiting += state["total_waiting_time"]
                ew_count += 1
            
            queue_data[agent.id] = {
                'NS': ns_queues,
                'EW': ew_queues
            }
            waiting_data[agent.id] = {
            #     'NS': ns_waiting / ns_count if ns_count > 0 else 0,
            #     'EW': ew_waiting / ew_count if ew_count > 0 else 0
                'NS': ns_waiting,
                'EW': ew_waiting
            }
        
        return queue_data, waiting_data
    
    def collect_emergency_vehicle_data(self):
        """收集当前紧急车辆数据用于可视化"""
        emergency_data = {}
        
        for agent in self.agent_list:
            ns_emergency_count = 0
            ew_emergency_count = 0
            ns_emergency_min_speed = 99  # 初始化为大值
            ew_emergency_min_speed = 99
            # ns_emergency_max_wait = 0
            # ew_emergency_max_wait = 0
            ns_emergency_total_wait = 0
            ew_emergency_total_wait = 0

            # 收集NS方向紧急车辆数据
            for lane in agent.controlled_lanes[agent.NS_lanes_index]:
                state = self.sumo_controller.get_vehicles_in_area(lane)
                ns_emergency_count += state["emergency_count"]
                ns_emergency_min_speed = min(ns_emergency_min_speed, state["emergency_min_speed"])
                ns_emergency_total_wait += state["emergency_total_wait_time"]
            
            # 收集EW方向紧急车辆数据
            for lane in agent.controlled_lanes[agent.EW_lanes_index]:
                state = self.sumo_controller.get_vehicles_in_area(lane)
                ew_emergency_count += state["emergency_count"]
                ew_emergency_min_speed = min(ew_emergency_min_speed, state["emergency_min_speed"])
                ew_emergency_total_wait += state["emergency_total_wait_time"]
            
            emergency_data[agent.id] = {
                'NS': {
                    'count': ns_emergency_count,
                    'min_speed': ns_emergency_min_speed if ns_emergency_min_speed != 99 else 0,
                    'total_wait': ns_emergency_total_wait
                },
                'EW': {
                    'count': ew_emergency_count,
                    'min_speed': ew_emergency_min_speed if ew_emergency_min_speed != 99 else 0,
                    'total_wait': ew_emergency_total_wait
                },
                'total': {
                    'count': ns_emergency_count + ew_emergency_count,
                    'min_speed': min(ns_emergency_min_speed, ew_emergency_min_speed),
                    'total_wait': ns_emergency_total_wait + ew_emergency_total_wait
                }
            }
        
        return emergency_data

    def run_visualization(self):
        """运行可视化数据收集 - 每回合统计一次，带预热"""
    
        print("="*60)
        print("开始收集可视化数据...")
        print(f"总回合数: {self.max_episodes}")
        print(f"每回合步数: {self.step_per_episode}")
        print("="*60)
    

         # ===== 预热：让环境空转，产生车辆 =====
        print("\n预热阶段: 让环境运行产生车辆...")
        self.sumo_controller.reset_simulation()
        warmup_steps = 300  # 空转300步
        for step in range(warmup_steps):
            self.sumo_controller.step_sumo()
        print("预热完成！\n")


        # ===== 正式收集数据：每个回合统计一次 =====
        for episode in range(self.max_episodes):
            self.episode = episode + 1
            print(f"\n--- 回合 {self.episode}/{self.max_episodes} ---")
        
        # 重置仿真
            self.sumo_controller.reset_simulation()
        
        # 每个回合开始前短暂预热
            for _ in range(100):
                self.sumo_controller.step_sumo()
        
        # 收集本回合的步数据
            step_queues = {agent.id: {'NS': [], 'EW': []} for agent in self.agent_list}
            # step_waitings = {agent.id: {'NS': [], 'EW': []} for agent in self.agent_list}
            # step_emergency = {agent.id: {'NS': [], 'EW': []} for agent in self.agent_list}  # 紧急车数据
            step_waitings = {agent.id: {'NS': 0, 'EW': 0} for agent in self.agent_list}      # 改为累加
            step_emergency = {agent.id: {'NS': 0, 'EW': 0} for agent in self.agent_list}      # 改为累加
        
            for step in range(self.step_per_episode):
            # 执行一步训练
                self.train_step()
            
            # 收集当前步的数据
                queue_data, waiting_data = self.collect_traffic_data()
                emergency_data = self.collect_emergency_vehicle_data()  # 获取紧急车数据  
                # for agent_id, data in emergency_data.items():
                #     if data['total']['count'] > 0:
                #         print(f"Agent {agent_id} 检测到急救车! 数量: {data['total']['count']}, 最大等待: {data['total']['max_wait']}")
            
                for agent_id in queue_data:
                    step_queues[agent_id]['NS'].append(queue_data[agent_id]['NS'])
                    step_queues[agent_id]['EW'].append(queue_data[agent_id]['EW'])
                    # step_waitings[agent_id]['NS'].append(waiting_data[agent_id]['NS'])
                    # step_waitings[agent_id]['EW'].append(waiting_data[agent_id]['EW'])
                    step_waitings[agent_id]['NS'] += waiting_data[agent_id]['NS']
                    step_waitings[agent_id]['EW'] += waiting_data[agent_id]['EW']
                    if agent_id in emergency_data:
                        # step_emergency[agent_id]['NS'].append(emergency_data[agent_id]['NS']['max_wait'])
                        # step_emergency[agent_id]['EW'].append(emergency_data[agent_id]['EW']['max_wait'])
                        step_emergency[agent_id]['NS'] += emergency_data[agent_id]['NS']['total_wait']
                        step_emergency[agent_id]['EW'] += emergency_data[agent_id]['EW']['total_wait']
                        
                # if (step + 1) % 400 == 0:
                #     print(f"  步数进度: {step + 1}/{self.step_per_episode}")
                #     # 每个回合结束后，可以在这里进行经验回放和网络更新
                #     for agent in self.agent_list:
                #         print(f"Agent {agent.id} Episode {self.episode} Immediate Buffer Size: {len(agent.immediate_buffer)}")
                #         # agent.reset_all()
                #         agent.update_epsilon()
                #         agent.update_target_network()
                        # print(f"当前探索率为{agent.epsilon}")
        
        # ===== 计算本回合的平均值（合并NS和EW方向）=====
            for agent_id in step_queues:
            # 计算NS和EW的平均队列长度，再取整体平均
                avg_ns_queue = np.mean(step_queues[agent_id]['NS']) if step_queues[agent_id]['NS'] else 0
                avg_ew_queue = np.mean(step_queues[agent_id]['EW']) if step_queues[agent_id]['EW'] else 0
                avg_queue = (avg_ns_queue + avg_ew_queue) / 2
            
            # 计算NS和EW的平均等待时间
                # avg_ns_wait = np.mean(step_waitings[agent_id]['NS']) if step_waitings[agent_id]['NS'] else 0
                # avg_ew_wait = np.mean(step_waitings[agent_id]['EW']) if step_waitings[agent_id]['EW'] else 0
                # avg_waiting = (avg_ns_wait + avg_ew_wait) / 2

                # avg_ns_emergency = np.mean(step_emergency[agent_id]['NS']) if step_emergency[agent_id]['NS'] else 0
                # avg_ew_emergency = np.mean(step_emergency[agent_id]['EW']) if step_emergency[agent_id]['EW'] else 0
                # avg_emergency = (avg_ns_emergency + avg_ew_emergency) / 2

                total_ns_wait = step_waitings[agent_id]['NS']
                total_ew_wait = step_waitings[agent_id]['EW']
                total_waiting = total_ns_wait + total_ew_wait   # 总等待时间

                total_ns_emergency = step_emergency[agent_id]['NS']
                total_ew_emergency = step_emergency[agent_id]['EW']
                total_emergency = total_ns_emergency + total_ew_emergency   # 急救车总延误

            # 记录到logger
                self.logger.log_episode_metrics(agent_id, avg_queue, total_waiting) 
                self.logger.log_episode_emergency(agent_id, total_emergency)  # 记录紧急车数据      
            
            # 记录奖励（从智能体获取）
                # for agent in self.agent_list:
                #     if agent.id == agent_id and agent.reward_list:
                #         last_reward = agent.reward_list[-1] if agent.reward_list else 0
                #         self.logger.log_agent_rewards(agent, last_reward)
                for agent in self.agent_list:
                        if agent.id == agent_id and agent.reward_list:
                            total_reward = sum(agent.reward_list)
                            self.logger.log_agent_rewards(agent, total_reward)
            
            print(f"  回合完成！")

            # episode结束后再reset
            for agent in self.agent_list:

                agent.update_epsilon()
                agent.update_target_network()

                print(f"当前探索率为{agent.epsilon}")

                # 最后清空
                agent.reset_all()
    
    # ===== 生成所有图表 =====
        print("\n数据收集完成！正在生成图表...")
        self.logger.finalize()
        for agent in self.agent_list:
            print(f"Agent {agent.id} Episode {self.episode} Immediate Buffer Size: {len(agent.immediate_buffer)}")
            agent.reset_all()
            agent.update_epsilon()
            agent.update_target_network()
            print(f"当前探索率为{agent.epsilon}")
    
        print(f"\n所有图表已保存到: {self.plot_dir}")
        print("生成的文件:")
        print("  - agent_*_reward_curve.png (每个智能体的奖励曲线)")
        print("  - agent_*_queue_length_curve.png (每个智能体的队列长度曲线)")
        print("  - agent_*_waiting_time_curve.png (每个智能体的等待时间曲线)")
        print("  - agent_*_combined_metrics.png (每个智能体的综合指标图)")
        print("  - agent_*_metrics.csv (每个智能体的数据CSV)")
        print("  - all_agents_reward_comparison.png (所有智能体奖励对比)")
        print("  - all_agents_queue_comparison.png (所有智能体队列对比)")
        print("  - all_agents_waiting_comparison.png (所有智能体等待时间对比)")
    
        return self.logger.agent_queue_data, self.logger.agent_waiting_data
    

# 定义main函数 
def main():
    trainer = mytrainer()
    logger = myLogger(trainer.plot_dir)
    trainer.set_logger(logger)
    
    print("*" * 60)
    print("快速可视化模式")
    print(f"每回合步数: {trainer.step_per_episode}")
    print("*" * 60)
    
    # 运行可视化数据收集
    trainer.run_visualization()
    
    print("\n" + "*" * 60)
    print("完成！请查看输出目录中的图片文件")
    print(f"输出目录: {trainer.plot_dir}")
    print("*" * 60)
    
if __name__ == '__main__':
    main()