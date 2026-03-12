import math
import traci
from myscripts.myagent import myAgent

# TODO：实现根据智能体查询车道数据，并进行模糊编码
class SumoController:
    def __init__(self):
        self.sumo_binary = "sumo-gui" # 使用图形化界面
        # self.sumo_binary = "sumo"  # 使用无图形化界面333
        # self.config_file = "./config/sumo_config.sumocfg"     # 使用模块化方式启动时的配置路径
        self.config_file = "./config/5.sumocfg"     # 使用模块化方式启动时的配置路径
        # self.config_file = "../config/sumo_config.sumocfg"       # 使用文件方式启动时的配置路径
        self.process = None
        # self.light_state = ['rrrrrGGGggrrrrrGGGgg','GGGggrrrrrGGGggrrrrr']    # GGGggrrrrrGGGggrrrrr 南北通行，东西红灯   rrrrrGGGggrrrrrGGGgg 东西通行，南北红灯
        self.light_state = ['GGGgrrrrGGGgrrrr','rrrrGGGgrrrrGGGg']    # GGGggrrrrrGGGggrrrrr 南北通行，东西红灯   rrrrrGGGggrrrrrGGGgg 东西通行，南北红灯

        self.last_emr_min_speed = 50

    def start_sumo(self):
        traci.start([self.sumo_binary, "-c", self.config_file])
        print("SUMO started with config:", self.config_file)
    def stop_sumo(self):
        traci.close()
        print("SUMO stopped.")
    def step_sumo(self):
        traci.simulationStep()
    def get_current_time(self):
        return traci.simulation.getTime()
    def reset_simulation(self):
        traci.load(["-c", self.config_file])
        print("SUMO simulation reset.")

# 将智能体接触到的所有车道提取出来
# 所有车道上的车辆id提取出来
# 将所有车辆按照路口距离进行统计，统计每个区域的车辆数量，形成对应的网格化编码
# 因为智能体要进行训练和学习，所以这些区域不能简单的按照距离区分，而是各个方向的车辆都进行单独统计
# 同时，因为路口红绿灯的职责主要是阻止和限制车流通过，所以通过路口的车将不进行统计，免得对模型训练造成干扰
# 因此，每个智能体要统计4个进入方向车道的车辆信息，每个车辆信息汇总到一起进行处理
# 形成的向量维度是：4个方向 * 2个车道 * 4个区域 = 32维向量
# 使用32维向量表示一个状态，而一条经验的表示方法是：32维状态 + 8维动作（独热编码） + 1维奖励（数值） + 32维下一个状态 = 73维向量
# 同时每次训练要录入所有邻居信息，73维向量 * 2个邻居 + 2 * 2维邻居相位信息（相位，时长）= 150维向量
# 所以每次输入的维度为：150 + 73 = 223维向量
# 
# 
# 

    # 获取lane_id 车道上区域一的车辆数量以及车辆等待时间
    # TODO：实现单一区域的车辆统计  ok
    # TODO：实现多个区域的车辆统计
    # TODO：实现包含行人的统计
    # TODO：包含急救车统计   ok

    def get_vehicle_count_by_lane(self, lane_id):
        vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)
        return vehicle_count
    
    def get_max_waiting_time_by_lane(self, lane_id):
        vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
        max_waiting_time = 0
        total_waiting_time = 0
        for vid in vehicle_ids:
            waiting_time = traci.vehicle.getWaitingTime(vid)
            total_waiting_time += waiting_time
            if waiting_time > max_waiting_time:
                max_waiting_time = waiting_time
        return max_waiting_time, total_waiting_time
    
    def get_vehicles_in_area(self, lane_id):
        '''获取一个车道上的所有车辆信息，用于计算state'''
        vehicle_count = self.get_vehicle_count_by_lane(lane_id)
        max_waiting_time, total_waiting_time = self.get_max_waiting_time_by_lane(lane_id)
        emergency_count, emergency_min_speed, emergency_max_wait_time = self.get_emergency_count_speed_waitTime(lane_id)
        average_speed = traci.lane.getLastStepMeanSpeed(lane_id)
        Occupancy = traci.lane.getLastStepOccupancy(lane_id)        # 车道占有率
        
        
        vehicle_info = {
            "lane_id":lane_id,
            "vehicle_count": vehicle_count,
            "max_waiting_time": max_waiting_time,
            "total_waiting_time":total_waiting_time,
            "emergency_count": emergency_count,
            "emergency_min_speed": emergency_min_speed,
            "emergency_max_wait_time":emergency_max_wait_time,
            "average_speed": average_speed ,
            "occupancy": Occupancy  
        }
        self.last_emr_min_speed = emergency_min_speed
        return vehicle_info




    def get_emergency_count_speed_waitTime(self,lane_id):
        emergency_vehicles = []
        emergency_speeds = []  # 收集所有急救车的速度
        emergency_wait_times = []  # 收集所有急救车的等待时间
        # 获取该车道上所有车辆 ID
        vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
        emergency_count = len(vehicle_ids)
        for veh_id in vehicle_ids:
            # 1. 判断是否是急救车
            vtype = traci.vehicle.getTypeID(veh_id)
            if vtype in ["emergency", "ambulance", "firebrigade", "police"]:  # 根据你的 .rou.xml 定义调整
                emergency_count += 1
                emergency_vehicles.append(veh_id)
                emergency_speed = traci.vehicle.getSpeed(veh_id) 
                emergency_speeds.append(emergency_speed)
                emergency_wait_time = traci.vehicle.getWaitingTime(veh_id)
                emergency_wait_times.append(emergency_wait_time)
        # 计算急救车的最大等待时间（如果没有急救车，则为 0）
        emergency_min_speed = min(emergency_speeds) if emergency_speeds else 99
        emergency_max_wait_time = max(emergency_wait_times) if emergency_wait_times else 0
        return emergency_count,emergency_min_speed,emergency_max_wait_time
        
    def get_trafficlight_IDlist(self):
        tls_ids = traci.trafficlight.getIDList()
        return tls_ids
    
    def classify_lane_direction(self, lane_id, tolerance=45):
        """根据车道几何形状判断主方向：'NS'、'EW' 或 'OTHER'"""
        try:
            shape = traci.lane.getShape(lane_id)
            if len(shape) < 2:
                return "OTHER"
            
            start_x, start_y = shape[0]
            end_x, end_y = shape[-1]
            dx = end_x - start_x
            dy = end_y - start_y
            
            angle_deg = math.degrees(math.atan2(dy, dx)) % 360
            
            # 判断东西向（0° ± tolerance 和 180° ± tolerance）
            if (angle_deg < tolerance) or (angle_deg >= 360 - tolerance) or \
            (180 - tolerance <= angle_deg < 180 + tolerance):
                return "EW"
            # 判断南北向（90° ± tolerance 和 270° ± tolerance）
            elif (90 - tolerance <= angle_deg < 90 + tolerance) or \
                (270 - tolerance <= angle_deg < 270 + tolerance):
                return "NS"
            else:
                return "OTHER"
        except Exception as e:
            # 如果车道不存在或出错，保守归为 OTHER
            return "OTHER"
    def get_controlled_lanes(self,tls_id):
        """从 TraCI 获取某信号灯控制的所有进口车道（from_edge），去重返回。"""
        controlled_links = traci.trafficlight.getControlledLinks(tls_id)
        # 提取每个连接的 from_edge（即车辆来自的车道）
        lanes = []
        for link_group in controlled_links:
            # link_group 是 [('from', 'to', 'via')]，取第一个元素的第0项
            if link_group:  # 非空
                from_edge = link_group[0][0]  # 字符串，如 'south1_to_1_0'
                lanes.append(from_edge)
        # 去重（保持顺序）
        unique_lanes = list(dict.fromkeys(lanes))
        # Step 2: 按方向分类
        ns_lanes = []
        ew_lanes = []
        for lane in unique_lanes:
            direction = self.classify_lane_direction(lane)
            if direction == "NS":
                ns_lanes.append(lane)
            else:
                ew_lanes.append(lane)      
        return [ns_lanes, ew_lanes]

    def get_controlled_links(self,tls_id):
        # 获取某 TLS 控制的所有连接（车道）
        controlled_links = traci.trafficlight.getControlledLinks(tls_id)
        print(controlled_links)  # 返回 [(from_edge, to_edge, via), ...]
        return controlled_links

    def set_light_phase(self,tls_id,tls_state):
        traci.trafficlight.setRedYellowGreenState(tls_id, tls_state)
    def apply_agent(self,agent:myAgent):
        '''应用智能体策略'''
        self.set_light_phase(agent.id,self.light_state[agent.phase])





    def set_all_traffic_lights_to_red(self):
        """
        将 SUMO 中所有交通信号灯强制设置为全红状态。
        """
        # 获取所有交通灯 ID
        tls_ids = traci.trafficlight.getIDList()
        
        for tls_id in tls_ids:
            try:
                # 方法1：通过 controlled links 获取信号灯状态长度（推荐）
                controlled_links = traci.trafficlight.getControlledLinks(tls_id)
                num_signals = len(controlled_links)
                
                # 构造全红状态字符串（每个受控连接对应一个 'r'）
                # red_state = 'r' * num_signals
                # red_state = 'GGGggrrrrrGGGggrrrrr'  # 南北通行，东西红灯 
                red_state = 'rrrrrGGGggrrrrrGGGgg'  # 东西通行，南北红灯 
                
                # 强制设置当前信号状态
                traci.trafficlight.setRedYellowGreenState(tls_id, red_state)
                
                print(f"✅ TLS '{tls_id}' set to all-red ({num_signals} signals).")
                
            except traci.TraCIException as e:
                print(f"⚠️ Failed to set TLS '{tls_id}': {e}")
     
def main():
    sumo_controller = SumoController()
    sumo_controller.start_sumo()
    lane_id = "south1_to_1_0"  
    count = 500
    while count >0:
        sumo_controller.step_sumo()
        if count == 300:
            sumo_controller.set_all_traffic_lights_to_red()
        count -=1
    vehicle_count = sumo_controller.get_viehicle_count_by_lane(lane_id)
    vehicle_info = sumo_controller.get_vehicles_in_area(lane_id)
    print(f"Vehicle info on lane {lane_id}: {vehicle_info}")
    import time
    time.sleep(1)  # 等待一段时间以确保SUMO已经启动
    print(f"Vehicle count on lane {lane_id}: {vehicle_count}")

    sumo_controller.get_trafficlight_IDlist()
    sumo_controller.stop_sumo()

if __name__ == "__main__":
    main()
