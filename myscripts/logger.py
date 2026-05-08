import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class myLogger():
    def __init__(self, log_file_path):
        self.plot_dir = log_file_path
        os.makedirs(self.plot_dir, exist_ok=True)
        self.all_agent_reward_dict = {}
        # 存储每个智能体的队列长度和等待时间数据（每回合统计一次）
        self.agent_queue_data = {}   # {agent_id: {'NS_EW': [每回合平均值]}}
        self.agent_waiting_data = {} # {agent_id: {'NS_EW': [每回合平均值]}}
        # 新增：存储急救车延误数据
        self.agent_emergency_delay_data = {}  # {agent_id: [每回合平均延误]}

    def intialize_agent_num(self, agent_list):
        self.agent_num = len(agent_list)
        for agent in agent_list:
            self.all_agent_reward_dict[agent.id] = []
            self.agent_queue_data[agent.id] = {'NS_EW': []}
            self.agent_waiting_data[agent.id] = {'NS_EW': []}
            self.agent_emergency_delay_data[agent.id] = []  # 新增: 初始化急救车延误数据存储结构


    def log_agent_rewards(self, agent, reward):
        """记录每个智能体的奖励数据"""
        self.all_agent_reward_dict[agent.id].append(reward)

    def log_episode_metrics(self, agent_id, avg_queue, total_waiting):
        """记录每个回合的指标（合并NS和EW方向，传入的已经是平均值）"""
        self.agent_queue_data[agent_id]['NS_EW'].append(avg_queue)
        self.agent_waiting_data[agent_id]['NS_EW'].append(total_waiting)

    def log_episode_emergency(self, agent_id, total_emergency_delay):
        """记录每个回合的急救车延误"""
        self.agent_emergency_delay_data[agent_id].append(total_emergency_delay)


    def save_agent_data_to_csv(self):
        """每个智能体单独保存数据到CSV文件"""
        # 保存奖励数据（所有智能体汇总）
        if self.all_agent_reward_dict:
            reward_df = pd.DataFrame(self.all_agent_reward_dict)
            reward_df.columns = [f"Agent_{agent_id}" for agent_id in self.all_agent_reward_dict.keys()]
            reward_df.insert(0, 'Episode', range(1, len(reward_df) + 1))
            reward_df.to_csv(os.path.join(self.plot_dir, "agent_rewards.csv"), index=False)
        
        # 每个智能体单独保存队列、等待时间和急救车数据
        for agent_id in self.agent_queue_data.keys():
            queue_list = self.agent_queue_data[agent_id]['NS_EW']
            waiting_list = self.agent_waiting_data[agent_id]['NS_EW']
            # 新增：获取急救车数据
            emergency_list = self.agent_emergency_delay_data.get(agent_id, [])
            
            if queue_list and waiting_list:
                # 补齐长度（如果急救车数据不够）
                max_len = max(len(queue_list), len(waiting_list), len(emergency_list))
                emergency_padded = emergency_list + [np.nan] * (max_len - len(emergency_list))
                
                agent_df = pd.DataFrame({
                    'Episode': range(1, len(queue_list) + 1),
                    'Queue_Length_Avg': queue_list,
                    'Waiting_Time_Total': waiting_list,
                    'Emergency_Delay_Total': emergency_padded[:len(queue_list)]
                })
                agent_df.to_csv(os.path.join(self.plot_dir, f"agent_{agent_id}_metrics.csv"), index=False)

    def log(self, message):
        log_file_path = os.path.join(self.plot_dir, "log.txt")
        with open(log_file_path, 'a') as log_file:
            log_file.write(message + '\n')

    def log_agent_state(self, agent_id, state):
        log_file_path = os.path.join(self.plot_dir, f"agent_{agent_id}_state_log.txt")
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"Step: State: {state}\n")
    
    def log_agent_duration(self, agent_id, duration):
        log_file_path = os.path.join(self.plot_dir, f"agent_{agent_id}_duration_log.txt")
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"Duration: {duration}\n")

    # ==================== 绘图方法 ====================
    
    def _create_figure(self, figsize=(10, 6)):
        """创建统一的图形样式"""
        fig, ax = plt.subplots(figsize=figsize)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlabel('Episode')
        return fig, ax

    def _add_smoothing(self, ax, x, y, window=10, color='red', label=None):
        """添加平滑曲线"""
        if len(y) >= window:
            smoothed = np.convolve(y, np.ones(window)/window, mode='valid')
            ax.plot(x[window-1:], smoothed, color=color, linewidth=2, 
                   label=label or f'Smoothed (window={window})')

    def _save_figure(self, fig, filename):
        """保存图形"""
        filepath = os.path.join(self.plot_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def plot_agent_rewards(self, agent_id, reward_list, window=10):
        """绘制单个智能体的奖励曲线"""
        if not reward_list:
            return
            
        fig, ax = self._create_figure()
        episodes = list(range(1, len(reward_list) + 1))
        
        ax.plot(episodes, reward_list, 'b-', linewidth=1.5, alpha=0.7, label='Original')
        self._add_smoothing(ax, episodes, reward_list, window, 'red', 'Smoothed')
        
        # 添加均值线
        if len(reward_list) > 50:
            mean_reward = np.mean(reward_list[-50:])
            ax.axhline(y=mean_reward, color='green', linestyle='--', alpha=0.5, 
                      label=f'Mean (last 50): {mean_reward:.4f}')
        
        ax.set_title(f'Agent {agent_id} Reward per Episode', fontsize=14, fontweight='bold')
        ax.set_ylabel('Reward')
        ax.legend(loc='best')
        
        self._save_figure(fig, f'agent_{agent_id}_reward_curve.png')

    def plot_agent_queue_length(self, agent_id, queue_list, window=10):
        """绘制单个智能体的队列长度曲线"""
        if not queue_list:
            return
            
        fig, ax = self._create_figure()
        episodes = list(range(1, len(queue_list) + 1))
        
        ax.plot(episodes, queue_list, 'orange', linewidth=1.5, alpha=0.7, label='Queue Length')
        self._add_smoothing(ax, episodes, queue_list, window, 'red')
        
        ax.set_title(f'Agent {agent_id} Queue Length per Episode', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Queue Length (vehicles)')
        ax.legend(loc='best')
        
        self._save_figure(fig, f'agent_{agent_id}_queue_length_curve.png')

    def plot_agent_waiting_time(self, agent_id, waiting_list, window=10):
        """绘制单个智能体的等待时间曲线"""
        if not waiting_list:
            return
            
        fig, ax = self._create_figure()
        episodes = list(range(1, len(waiting_list) + 1))
        
        ax.plot(episodes, waiting_list, 'green', linewidth=1.5, alpha=0.7, label='Waiting Time')
        self._add_smoothing(ax, episodes, waiting_list, window, 'red')
        
        ax.set_title(f'Agent {agent_id} Waiting Time per Episode', fontsize=14, fontweight='bold')
        ax.set_ylabel('Total Waiting Time (seconds)')
        ax.legend(loc='best')
        
        self._save_figure(fig, f'agent_{agent_id}_waiting_time_curve.png')

    def plot_agent_emergency_delay(self, agent_id, delay_list, window=10):
        """绘制单个智能体的急救车延误曲线"""
        if not delay_list:
            return
            
        fig, ax = self._create_figure()
        episodes = list(range(1, len(delay_list) + 1))
        
        ax.plot(episodes, delay_list, 'purple', linewidth=1.5, alpha=0.7, label='Emergency Delay')
        self._add_smoothing(ax, episodes, delay_list, window, 'red')
        
        # 添加均值线
        if len(delay_list) > 20:
            mean_delay = np.mean(delay_list[-20:])
            ax.axhline(y=mean_delay, color='purple', linestyle='--', alpha=0.5, 
                      label=f'Mean (last 20): {mean_delay:.2f}s')
        
        ax.set_title(f'Agent {agent_id} Emergency Vehicle Delay per Episode', fontsize=14, fontweight='bold')
        ax.set_ylabel('Emergency Delay (seconds)')
        ax.legend(loc='best')
        
        self._save_figure(fig, f'agent_{agent_id}_emergency_delay.png')

    def plot_all_agents_emergency_delay(self, window=10):
        """所有智能体急救车延误对比图"""
        if not self.agent_emergency_delay_data:
            print("没有急救车延误数据")
            return
        
        fig, ax = self._create_figure(figsize=(12, 6))
        colors = ['purple', 'orange', 'green', 'red', 'blue', 'brown']
        
        idx = 0
        for agent_id, delay_list in self.agent_emergency_delay_data.items():
            if delay_list:
                episodes = list(range(1, len(delay_list) + 1))
                color = colors[idx % len(colors)]
                ax.plot(episodes, delay_list, color=color, linewidth=1.5, alpha=0.7, label=f'Agent {agent_id}')
                if len(delay_list) >= window:
                    smoothed = np.convolve(delay_list, np.ones(window)/window, mode='valid')
                    ax.plot(episodes[window-1:], smoothed, color=color, linewidth=2, linestyle='--', alpha=0.5)
                idx += 1
        
        ax.set_title('All Agents Emergency Vehicle Delay Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Emergency Delay (seconds)')
        ax.legend(loc='best')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        self._save_figure(fig, 'all_agents_emergency_delay.png')

    def plot_agent_combined_metrics(self, agent_id, queue_list, waiting_list, reward_list, emergency_list=None, window=10):
        """绘制单个智能体的综合指标图（四合一：队列+等待+奖励+急救车延误）"""
        if not queue_list or not waiting_list:
            return
        
        # 确定子图数量
        n_plots = 4 if emergency_list and reward_list else (3 if (emergency_list or reward_list) else 2)
        
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4*n_plots))
        if n_plots == 1:
            axes = [axes]
        
        episodes_q = list(range(1, len(queue_list) + 1))
        episodes_w = list(range(1, len(waiting_list) + 1))
        
        plot_idx = 0
        
        # 队列长度
        axes[plot_idx].plot(episodes_q, queue_list, 'orange', linewidth=1.5, alpha=0.7)
        self._add_smoothing(axes[plot_idx], episodes_q, queue_list, window, 'red')
        axes[plot_idx].set_ylabel('Queue Length (vehicles)')
        axes[plot_idx].set_title(f'Agent {agent_id} - Queue Length')
        axes[plot_idx].grid(True, linestyle='--', alpha=0.6)
        axes[plot_idx].legend(['Original', 'Smoothed'], loc='best')
        plot_idx += 1
        
        # 等待时间
        axes[plot_idx].plot(episodes_w, waiting_list, 'green', linewidth=1.5, alpha=0.7)
        self._add_smoothing(axes[plot_idx], episodes_w, waiting_list, window, 'red')
        axes[plot_idx].set_ylabel('Waiting Time (seconds)')
        axes[plot_idx].set_title(f'Agent {agent_id} - Waiting Time')
        axes[plot_idx].grid(True, linestyle='--', alpha=0.6)
        axes[plot_idx].legend(['Original', 'Smoothed'], loc='best')
        plot_idx += 1
        
        # 奖励
        if reward_list:
            episodes_r = list(range(1, len(reward_list) + 1))
            axes[plot_idx].plot(episodes_r, reward_list, 'blue', linewidth=1.5, alpha=0.7)
            self._add_smoothing(axes[plot_idx], episodes_r, reward_list, window, 'red')
            axes[plot_idx].set_ylabel('Reward')
            axes[plot_idx].set_title(f'Agent {agent_id} - Reward')
            axes[plot_idx].grid(True, linestyle='--', alpha=0.6)
            axes[plot_idx].legend(['Original', 'Smoothed'], loc='best')
            plot_idx += 1
        
        # 急救车延误
        if emergency_list:
            episodes_e = list(range(1, len(emergency_list) + 1))
            axes[plot_idx].plot(episodes_e, emergency_list, 'purple', linewidth=1.5, alpha=0.7)
            self._add_smoothing(axes[plot_idx], episodes_e, emergency_list, window, 'red')
            axes[plot_idx].set_ylabel('Emergency Delay (seconds)')
            axes[plot_idx].set_title(f'Agent {agent_id} - Emergency Delay')
            axes[plot_idx].grid(True, linestyle='--', alpha=0.6)
            axes[plot_idx].legend(['Original', 'Smoothed'], loc='best')
            plot_idx += 1
        
        axes[-1].set_xlabel('Episode')
        plt.tight_layout()
        self._save_figure(fig, f'agent_{agent_id}_combined_metrics.png')

    def plot_all_agents_comparison(self, metric_type='reward', window=10):
        """绘制所有智能体的对比图
        metric_type: 'reward', 'queue', 'waiting'， 'emergency'
        """
        fig, ax = self._create_figure(figsize=(12, 6))
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
        
        idx = 0
        if metric_type == 'reward':
            for agent_id, rewards in self.all_agent_reward_dict.items():
                if rewards:
                    episodes = list(range(1, len(rewards) + 1))
                    color = colors[idx % len(colors)]
                    ax.plot(episodes, rewards, color=color, linewidth=1.5, alpha=0.7, label=f'Agent {agent_id}')
                    if len(rewards) >= window:
                        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                        ax.plot(episodes[window-1:], smoothed, color=color, linewidth=2, linestyle='--', alpha=0.5)
                    idx += 1
            ylabel = 'Reward'
            title = 'All Agents Reward Comparison'
            
        elif metric_type == 'queue':
            for agent_id, data in self.agent_queue_data.items():
                queue_list = data['NS_EW']
                if queue_list:
                    episodes = list(range(1, len(queue_list) + 1))
                    color = colors[idx % len(colors)]
                    ax.plot(episodes, queue_list, color=color, linewidth=1.5, alpha=0.7, label=f'Agent {agent_id}')
                    if len(queue_list) >= window:
                        smoothed = np.convolve(queue_list, np.ones(window)/window, mode='valid')
                        ax.plot(episodes[window-1:], smoothed, color=color, linewidth=2, linestyle='--', alpha=0.5)
                    idx += 1
            ylabel = 'Queue Length (vehicles)'
            title = 'All Agents Queue Length Comparison'
            
        elif metric_type == 'waiting':
            for agent_id, data in self.agent_waiting_data.items():
                waiting_list = data['NS_EW']
                if waiting_list:
                    episodes = list(range(1, len(waiting_list) + 1))
                    color = colors[idx % len(colors)]
                    ax.plot(episodes, waiting_list, color=color, linewidth=1.5, alpha=0.7, label=f'Agent {agent_id}')
                    if len(waiting_list) >= window:
                        smoothed = np.convolve(waiting_list, np.ones(window)/window, mode='valid')
                        ax.plot(episodes[window-1:], smoothed, color=color, linewidth=2, linestyle='--', alpha=0.5)
                    idx += 1
            ylabel = 'Waiting Time (seconds)'
            title = 'All Agents Waiting Time Comparison'

        elif metric_type == 'emergency':
            for agent_id, delay_list in self.agent_emergency_delay_data.items():
                if delay_list:
                    episodes = list(range(1, len(delay_list) + 1))
                    color = colors[idx % len(colors)]
                    ax.plot(episodes, delay_list, color=color, linewidth=1.5, alpha=0.7, label=f'Agent {agent_id}')
                    if len(delay_list) >= window:
                        smoothed = np.convolve(delay_list, np.ones(window)/window, mode='valid')
                        ax.plot(episodes[window-1:], smoothed, color=color, linewidth=2, linestyle='--', alpha=0.5)
                    idx += 1
            ylabel = 'Emergency Delay (seconds)'
            title = 'All Agents Emergency Vehicle Delay Comparison'

        else:
            return
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.legend(loc='best')
        
        self._save_figure(fig, f'all_agents_{metric_type}_comparison.png')

    def generate_all_agent_plots(self):
        """为每个智能体生成所有图表"""
        for agent_id in self.agent_queue_data.keys():
            queue_list = self.agent_queue_data[agent_id]['NS_EW']
            waiting_list = self.agent_waiting_data[agent_id]['NS_EW']
            reward_list = self.all_agent_reward_dict.get(agent_id, [])
            emergency_list = self.agent_emergency_delay_data.get(agent_id, [])
            
            if queue_list:
                self.plot_agent_queue_length(agent_id, queue_list)
            if waiting_list:
                self.plot_agent_waiting_time(agent_id, waiting_list)
            if emergency_list:
                self.plot_agent_emergency_delay(agent_id, emergency_list)
            if reward_list:
                self.plot_agent_rewards(agent_id, reward_list)
            
            # 生成综合图
            if queue_list and waiting_list:
                self.plot_agent_combined_metrics(agent_id, queue_list, waiting_list, reward_list, emergency_list)
        
        # 生成对比图
        self.plot_all_agents_comparison('reward')
        self.plot_all_agents_comparison('queue')
        self.plot_all_agents_comparison('waiting')
        if self.agent_emergency_delay_data:
            self.plot_all_agents_comparison('emergency')
            self.plot_all_agents_emergency_delay()

    # 临时添加，检查是否有数据
        print(f"Emergency data: {self.agent_emergency_delay_data}")

    def finalize(self):
        """完成训练后的最终处理"""
        self.save_agent_data_to_csv()
        self.generate_all_agent_plots()
        self.log("All plots and data saved successfully!")