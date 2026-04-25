import traci
import os

# 检查文件是否存在
config_file = "./config/5.sumocfg"
if not os.path.exists(config_file):
    print(f"配置文件不存在: {config_file}")
    # 尝试其他路径
    config_file = "./sumo_config.sumocfg"
    if not os.path.exists(config_file):
        print("找不到配置文件！")
        exit(1)

print(f"使用配置文件: {config_file}")

sumo_cmd = ['sumo', '-c', config_file, '--start', '--quit-on-end']
traci.start(sumo_cmd)

print("所有车辆:")
all_vehicles = traci.vehicle.getIDList()
if all_vehicles:
    for veh_id in all_vehicles:
        vtype = traci.vehicle.getTypeID(veh_id)
        print(f"  {veh_id}: {vtype}")
else:
    print("没有车辆！请检查配置文件中的路线文件是否正确。")

traci.close()