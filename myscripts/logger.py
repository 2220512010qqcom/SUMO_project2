import os
import pandas as pd
from matplotlib import pyplot as plt


class myLogger():
    def __init__(self, log_file_path):
        self.plot_dir = log_file_path
        self.all_agent_reward_dict = {}  # 存储所有智能体的奖励数据，格式为{agent_id: [reward1, reward2, ...], ...}

    def intialize_agent_num(self, agent_list):
        self.agent_num = len(agent_list)
        for agent in agent_list:
            self.all_agent_reward_dict[agent.id] = []  # 初始化每个智能体的奖励列表
    def log_agent_rewards(self, agent, reward):
        """记录每个智能体的奖励数据"""
        self.all_agent_reward_dict[agent.id].append(reward)

    def save_rewards_to_file(self):
        """将所有智能体的奖励数据保存到文件中"""
        log_file_path = os.path.join(self.plot_dir, "agent_rewards.csv")
        df = pd.DataFrame(self.all_agent_reward_dict)
        df.columns = [f"Agent_{agent_id}" for agent_id in self.all_agent_reward_dict.keys()]  # 可选：为列添加更具描述性的名称
        df.insert(0, 'Episode', range(1, len(df) + 1))  # 添加一个 'Episode' 列，表示回合数
        df.to_csv(log_file_path, index=False)



    def log(self, message):
        log_file_path = os.path.join(self.plot_dir, "log.txt")
        with open(log_file_path, 'a') as log_file:
            log_file.write(message + '\n')


    def plot_agent_rewards(self, agent_id, reward_list):    
        """绘制单个智能体的奖励曲线"""
        episodes = list(range(1, len(reward_list) + 1))
        rewards = reward_list

        plt.figure(figsize=(8, 5))
        plt.plot(episodes, rewards, 'b-', linewidth=1.2, marker='o', markersize=3)
        plt.title(f'Agent {agent_id} Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # 可选：加平滑曲线（消除噪声）
        if len(rewards) > 20:
            import numpy as np
            window = min(20, len(rewards) // 2)
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(episodes[window-1:], smoothed, 'r-', linewidth=2, label=f'Smoothed (window={window})')
            plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f'agent_{agent_id}_reward_curve.png'), dpi=150)
        plt.close()  # 防止内存泄漏 



    def plot_rewards(self,total_reward):
        """绘制训练奖励曲线"""
        episodes = list(range(1, len(total_reward) + 1))
        rewards = total_reward

        plt.figure(figsize=(8, 5))
        plt.plot(episodes, rewards, 'b-', linewidth=1.2, marker='o', markersize=3)
        plt.title('Training Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # 可选：加平滑曲线（消除噪声）
        if len(rewards) > 20:
            import numpy as np
            window = min(20, len(rewards) // 2)
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(episodes[window-1:], smoothed, 'r-', linewidth=2, label=f'Smoothed (window={window})')
            plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'training_reward_curve.png'), dpi=150)
        plt.close()  # 防止内存泄漏


    def plot_max_wait_time(self,max_wait_time):
        """绘制训练奖励曲线"""
        episodes = list(range(1, len(max_wait_time) + 1))
        wait_times = max_wait_time

        plt.figure(figsize=(8, 5))
        plt.plot(episodes, wait_times, 'b-', linewidth=1.2, marker='o', markersize=3)
        plt.title('Training Vehicle Max Wait Time per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Vehicle Wait  Time')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # 可选：加平滑曲线（消除噪声）
        if len(wait_times) > 20:
            import numpy as np
            window = min(20, len(wait_times) // 2)
            smoothed = np.convolve(wait_times, np.ones(window)/window, mode='valid')
            plt.plot(episodes[window-1:], smoothed, 'r-', linewidth=2, label=f'Smoothed (window={window})')
            plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'training_vehicle_max_wait_time_curve.png'), dpi=150)
        plt.close()  # 防止内存泄漏

    def plot_emer_max_wait_time(self, emer_max_wait_time):
        """绘制训练奖励曲线"""
        episodes = list(range(1, len(emer_max_wait_time) + 1))
        wait_times = emer_max_wait_time

        plt.figure(figsize=(8, 5))
        plt.plot(episodes, wait_times, 'b-', linewidth=1.2, marker='o', markersize=3)
        plt.title('Training Emergency Max Wait Time per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Vehicle Wait  Time')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # 可选：加平滑曲线（消除噪声）
        if len(wait_times) > 20:
            import numpy as np
            window = min(20, len(wait_times) // 2)
            smoothed = np.convolve(wait_times, np.ones(window)/window, mode='valid')
            plt.plot(episodes[window-1:], smoothed, 'r-', linewidth=2, label=f'Smoothed (window={window})')
            plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'training_emergency_max_wait_time_curve.png'), dpi=150)
        plt.close()  # 防止内存泄漏

    def plot_avarage_wait_time(self, avarage_wait_times):
        """绘制训练奖励曲线"""
        episodes = list(range(1, len(avarage_wait_times) + 1))
        wait_times = avarage_wait_times

        plt.figure(figsize=(8, 5))
        plt.plot(episodes, wait_times, 'b-', linewidth=1.2, marker='o', markersize=3)
        plt.title('Training average Wait Time per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Vehicle Average Wait  Time')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # 可选：加平滑曲线（消除噪声）
        if len(wait_times) > 20:
            import numpy as np
            window = min(20, len(wait_times) // 2)
            smoothed = np.convolve(wait_times, np.ones(window)/window, mode='valid')
            plt.plot(episodes[window-1:], smoothed, 'r-', linewidth=2, label=f'Smoothed (window={window})')
            plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'training_vehicle_average_wait_time_curve.png'), dpi=150)
        plt.close()  # 防止内存泄漏

    
    
