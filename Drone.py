

# 定义无人机类
class Drone:

    def __init__(self):
        self.ID = 0
        self.load = 2.5
        self.speed = 40
        self.range = 10
        self.remaining_SOC = 100       # state of charge
        # self.status_category = ['idle', 'delivery', 'back']     # 定义无人机状态的类别
        self.status = None      # 用于在 delivery_orders_and_generate_trajectory 中生成轨迹时存放 status data

        self.current_station_ID = None
        self.assigned_locker_ID = None
        self.assigned_order_ID = None

        self.current_location = None
        self.current_trajectory_timestamp_idx = None

        self.finished_trajectory = None
        self.remaining_trajectory = None

        self.org_lng = None
        self.org_lat = None

        self.des_lng = None
        self.des_lat = None

        self.trajectory = None

        self.task_cnt = 1


    def reset(self):
        """
        重置无人机的状态
        :return:
        """
        self.status = 'idle'     # 用于在 delivery_orders_and_generate_trajectory 中生成轨迹时存放 status data

        self.assigned_locker_ID = None
        self.assigned_order_ID = None

        self.current_trajectory_timestamp_idx = None

        self.finished_trajectory = None
        self.remaining_trajectory = None

        self.des_lng = None
        self.des_lat = None

        self.trajectory = None
        self.remaining_trajectory = None
