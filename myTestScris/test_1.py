from myscripts.mytrainer import mytrainer


def test_x():
    tempTrainer = mytrainer()
    tempTrainer.sumo_controller.start_sumo()
    for i in range(100):
        tempTrainer.sumo_controller.step_sumo()
        tempTrainer.step_to_next_light_change()
    print(tempTrainer.agent_list)


