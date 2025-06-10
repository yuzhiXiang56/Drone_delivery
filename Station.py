# 定义停机坪类
class Station:

    def __init__(self):
        self.ID = 0
        self.name = None
        self.lng = None
        self.lat = None

        self.assigned_but_not_scheduled_order_set = {}        # 被指派到某个停机坪中，但未被分配无人机的订单
        self.assigned_and_scheduled_order_set = {}        # 被指派到某个停机坪中，但已被分配无人机的订单
        self.assigned_drone_set = {}        # 被指派到某个停机坪中的无人机
        self.assigned_but_not_scheduled_order_queue = []  # 被指派到某个停机坪中的订单的队列



        self.idle_drone_set = {}            # 做动态
        self.occupied_drone_set = {}        # 做动态

        self.waiting_order_set = {}         # 当前停机坪中等待的订单队列
        self.finished_order_set = {}        # 当前停机坪中已完成的订单队列
