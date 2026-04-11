import os
import pandas as pd
from matplotlib import pyplot as plt


class myLogger():
    def __init__(self, log_file_path):
        self.plot_dir = log_file_path
        os.makedirs(self.plot_dir, exist_ok=True)  # 确保日志目录存在
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

    def log_agent_state(self, agent_id, state):
        """记录智能体的状态信息"""
        log_file_path = os.path.join(self.plot_dir, f"agent_{agent_id}_state_log.txt")
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"Episode {agent_id}: State: {state}\n")
    
    def log_agent_duration(self, agent_id, duration):
        """记录智能体的持续时间信息"""
        log_file_path = os.path.join(self.plot_dir, f"agent_{agent_id}_duration_log.txt")
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"Episode {agent_id}: Duration: {duration}\n")

    


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

    # ==================== 新增可视化方法 ====================
    
    def plot_queue_length_heatmap(self, queue_data, save_name='queue_length_heatmap.png'):
        """
        绘制队列长度热力图
        queue_data: 格式 {'agent_id': {'NS': [列表], 'EW': [列表]}}
        """
        import numpy as np
        
        fig, axes = plt.subplots(1, len(queue_data), figsize=(6*len(queue_data), 5))
        if len(queue_data) == 1:
            axes = [axes]
        
        for idx, (agent_id, data) in enumerate(queue_data.items()):
            ns_queues = np.array(data['NS'])
            ew_queues = np.array(data['EW'])
            
            # 创建热力图数据
            heatmap_data = np.vstack([ns_queues, ew_queues])
            
            im = axes[idx].imshow(heatmap_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
            axes[idx].set_yticks([0, 1])
            axes[idx].set_yticklabels(['North-South', 'East-West'])
            axes[idx].set_xlabel('Time Step')
            axes[idx].set_title(f'Agent {agent_id} Queue Length')
            plt.colorbar(im, ax=axes[idx], label='Queue Length')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, save_name), dpi=150, bbox_inches='tight')
        plt.close()

    def plot_queue_length_curve(self, queue_data, save_name='queue_length_curve.png'):
        """
        绘制队列长度曲线
        """
        import numpy as np
        
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['blue', 'green', 'red', 'purple']
        
        for idx, (agent_id, data) in enumerate(queue_data.items()):
            time_steps = np.arange(len(data['NS']))
            color = colors[idx % len(colors)]
            ax.plot(time_steps, data['NS'], '--', color=color, alpha=0.7, label=f'Agent {agent_id} NS')
            ax.plot(time_steps, data['EW'], '-', color=color, alpha=0.7, label=f'Agent {agent_id} EW')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Queue Length (vehicles)')
        ax.set_title('Queue Length Over Time')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, save_name), dpi=150, bbox_inches='tight')
        plt.close()

    def plot_waiting_time_curve(self, waiting_data, save_name='waiting_time_curve.png'):
        """
        绘制等待时间曲线
        """
        import numpy as np
        
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['blue', 'green', 'red', 'purple']
        
        for idx, (agent_id, data) in enumerate(waiting_data.items()):
            time_steps = np.arange(len(data['NS']))
            color = colors[idx % len(colors)]
            ax.plot(time_steps, data['NS'], '--', color=color, alpha=0.7, label=f'Agent {agent_id} NS')
            ax.plot(time_steps, data['EW'], '-', color=color, alpha=0.7, label=f'Agent {agent_id} EW')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Waiting Time (seconds)')
        ax.set_title('Vehicle Waiting Time Over Time')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, save_name), dpi=150, bbox_inches='tight')
        plt.close()

    def plot_waiting_time_boxplot(self, waiting_data, save_name='waiting_time_boxplot.png'):
        """
        绘制等待时间箱线图
        """
        import numpy as np
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data_to_plot = []
        labels = []
        
        for agent_id, data in waiting_data.items():
            if len(data['NS']) > 0:
                data_to_plot.append(data['NS'])
                labels.append(f'Agent {agent_id} NS')
                data_to_plot.append(data['EW'])
                labels.append(f'Agent {agent_id} EW')
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        
        # 设置颜色
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightskyblue']
        for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Waiting Time (seconds)')
        ax.set_title('Waiting Time Distribution')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, save_name), dpi=150, bbox_inches='tight')
        plt.close()

    def plot_comparison_bar(self, queue_data, waiting_data, save_name='comparison_bar.png'):
        """
        绘制对比柱状图
        """
        import numpy as np
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        agents = list(queue_data.keys())
        
        # 平均队列长度
        avg_ns_queue = [np.mean(queue_data[a]['NS']) for a in agents]
        avg_ew_queue = [np.mean(queue_data[a]['EW']) for a in agents]
        
        x = np.arange(len(agents))
        width = 0.35
        
        ax1.bar(x - width/2, avg_ns_queue, width, label='North-South', color='steelblue')
        ax1.bar(x + width/2, avg_ew_queue, width, label='East-West', color='coral')
        ax1.set_xlabel('Agent')
        ax1.set_ylabel('Average Queue Length')
        ax1.set_title('Average Queue Length')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'Agent {a}' for a in agents])
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 平均等待时间
        avg_ns_wait = [np.mean(waiting_data[a]['NS']) for a in agents]
        avg_ew_wait = [np.mean(waiting_data[a]['EW']) for a in agents]
        
        ax2.bar(x - width/2, avg_ns_wait, width, label='North-South', color='steelblue')
        ax2.bar(x + width/2, avg_ew_wait, width, label='East-West', color='coral')
        ax2.set_xlabel('Agent')
        ax2.set_ylabel('Average Waiting Time (s)')
        ax2.set_title('Average Waiting Time')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'Agent {a}' for a in agents])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, save_name), dpi=150, bbox_inches='tight')
        plt.close()