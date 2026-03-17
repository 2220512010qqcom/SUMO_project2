import random
import torch
import torch.nn as nn

class myDQN(nn.Module):
    """基于CNN的DQN网络"""
    
    def __init__(self,input_dim: int, action_dim: int):
        super().__init__()
        self.q_layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        """前向传播"""      
        return self.q_layers(state_features)
    

class myAgent:
    def __init__(self, id, input_dim=4):
        self.id = id
        self.immediate_buffer = []                                  # 即时缓冲区
        self.action_space = [0,1,2,3]                                   # 0:东西绿 10秒 1:东西绿 40秒 2:南北绿 10秒 3:南北绿 40秒
        self.input_dim = input_dim                                        # 输入维度
        self.start_time = 0                                         # 开始时间
        self.phase = 0                                              # 当前相位 0为南北红东西绿  1为南北绿东西红
        self.duration = 0                                           # 当前持续时间
        self.action_dim = len(self.action_space)                    # 动作维度
        self.behavior = myDQN(self.input_dim, self.action_dim)      # 行为网络
        self.target = myDQN(self.input_dim, self.action_dim)        # 目标网络
        self.learning_rate = 0.01
        self.train_experience_number = 320                           # 每次训练使用的经验数量

        self.reward_list = []                                              # 奖励列表


        self.controlled_lanes = []                                       # 智能体控制的车道列表, [[NS道路],[EW道路]]
        self.NS_lanes_index = 0
        self.EW_lanes_index = 1
        self.last_state = None                                        # 上一个状态
        self.last_action = None                                       # 上一个动作

        self.gamma = 0.95
        self.epsilon = 1.0                                          # 探索率
        self.epsilon_min = 0.01     
        self.epsilon_decay = 0.95   # 衰减系数（用于指数衰减）
        # ds补充
        self.optimizer = torch.optim.Adam(self.behavior.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()                               # 用于DQN的损失函数
        # 初始化其他属性
        pass
    
    ## ========================================== 奖励列表相关函数  ==========================================
    def add_reward(self, reward):
        """添加奖励到奖励列表"""
        self.reward_list.append(reward)
    def clear_reward_list(self):
        """清空奖励列表"""
        self.reward_list = []
    def get_reward_list(self):
        """获取奖励列表"""
        return self.reward_list
    
    # ========================================== 车道控制相关函数  ==========================================

    def set_controlled_lanes(self, lane_list):
        """设置智能体控制的车道列表"""
        self.controlled_lanes = lane_list
    def select_action(self, state: torch.Tensor) -> int:
        """根据当前状态选择动作"""

        if random.random() < self.epsilon:
            return random.choice(self.action_space)
        with torch.no_grad():
            q_values = self.behavior(state)
            action_index = torch.argmax(q_values).item()
        return self.action_space[action_index]
    
    def update_target_network(self):
        """更新目标网络的参数"""
        # print("更新目标网络参数")
        self.target.load_state_dict(self.behavior.state_dict())
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def update_behavior_network(self):
        """使用即时缓冲区的经验更新行为网络"""
        if len(self.immediate_buffer) < self.train_experience_number:
            return  # 如果经验数量不足，直接返回
        # print("更新行为网络参数")
        # 随机采样经验
        sampled_experiences = random.sample(self.immediate_buffer, self.train_experience_number)
        
        # 形成矩阵
        states = torch.tensor([exp[0] for exp in sampled_experiences], dtype=torch.float32)
        actions = torch.tensor([exp[1] for exp in sampled_experiences], dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor([exp[2] for exp in sampled_experiences], dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor([exp[3] for exp in sampled_experiences], dtype=torch.float32)
        
        # 计算当前Q值
        current_q_values = self.behavior(states).gather(1, actions)
        
        # 计算目标Q值
        with torch.no_grad():
            max_next_q_values = self.target(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + self.gamma * max_next_q_values  # 折扣因子gamma=0.99
        
        # 计算损失
        loss = self.criterion(current_q_values, target_q_values)
        
        # 优化行为网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_weight_args(self):
        '''保存神经网络权重参数'''
        torch.save(self.target.state_dict(), f"trained_agent{self.id}.pth")
    def load_weight_args(self):
        '''加载神经网络权重参数'''
        # 加载权重
        self.target.load_state_dict(torch.load(f"trained_agent{self.id}.pth", map_location=torch.device('cuda')))
        self.target.eval()  # 切换到评估模式（关闭 dropout/batchnorm 更新）
        print("✅ Model loaded successfully!")

    def store_experience(self, experience):
        """存储经验到即时缓冲区"""
        # （状态，动作，奖励，下一个状态）
        self.immediate_buffer.append(experience)
    def immiediate_buffer_length(self):
        return len(self.immediate_buffer)
    def clear_immediate_buffer(self):
        """清空即时缓冲区"""
        self.immediate_buffer = []
    def reset_all(self):
        """重置智能体的所有属性"""
        self.start_time = 0
        self.phase = 0
        self.duration = 0
        self.last_state = None
        self.last_action = None
    

    def set_duration(self, duration):
        """设置当前持续时间"""
        self.duration = duration
    def set_phase(self, phase):
        """设置当前相位"""
        self.phase = phase
    def set_start_time(self, start_time):
        """设置开始时间"""
        self.start_time = start_time

    def set_last_state(self, state):
        """设置上一个状态"""
        self.last_state = state
    def set_last_action(self, action):
        """设置上一个动作"""
        self.last_action = action

    
    

