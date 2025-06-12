import copy
from gurobipy import *
from tqdm import tqdm
import pandas as pd
from Station import *
from datetime import datetime
import os

class ModelBuilder():

    def __init__(self, data=None):
        self.data = data

    def build_FLP_model(self):
        # 创建模型
        self.FLP_model = Model("Facility_Location_Problem")
        print("========================= 模型建立成功！=========================")

        """
        在这里定义模型所需的决策变量
        :param
        x[i]: 是否在第 i 个点设立外卖柜的决策变量
        y[i, j]: 第 i 个点接收来自第 j 个点多少比例订单的决策变量
        e[j]: 用于 fix 因商务住宅距离外卖柜距离过远导致的选址问题的决策变量
        """
        x = {}
        y = {}
        e = {}

        """
        在这里定义外卖柜选址优化模型所需要的决策变量
        """
        for i in range(len(self.data.home_dict.keys())):
            x[i] = self.FLP_model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f'x_{i}')  # 定义决策变量 x_i
            for j in range(len(self.data.h3_data_dict.keys())):
                y[i, j] = self.FLP_model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f'y_{i}_{j}')  # 定义决策变量 y_i_j

        for j in range(len(self.data.h3_data_dict.keys())):
            e[j] = self.FLP_model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f'e_{j}')

        print("========================= 决策变量定义成功！=========================")

        """
        在这里定义外卖柜选址优化模型所需要的参数

        :param
        D[j]: 第 j 个 h3_area 中 order_num 的数量
        d[i, j]: i 点到 j 点的距离
        S: 所需要生成的外卖柜最小数量
        M: 用于 fix 因商务住宅距离外卖柜距离过远导致的选址问题
        fixed_cost: 每修设一个外卖柜的成本
        delivery_fee_per_km: 无人机配送每 km 的成本
        """
        D = {}
        d = {}
        S = 100
        M = 1000000
        fixed_cost = 5000
        delivery_fee_per_km = 2

        for j in tqdm(range(len(self.data.h3_data_dict.keys()))):
            D[j] = self.data.h3_data_dict[j]['order_num']  # 定义参数 D_j
            for i in range(len(self.data.home_dict.keys())):
                d[i, j] = self.data.compute_geo_dis(  # 定义参数 d_i_j
                    lng1=self.data.h3_data_dict[j]['center_coor'][1],
                    lat1=self.data.h3_data_dict[j]['center_coor'][0],
                    lng2=self.data.home_dict[i].lng,
                    lat2=self.data.home_dict[i].lat
                )
        print('\n')
        print("========================== 参数定义成功！=========================")

        """
        在这里定义外卖柜选址优化模型的目标函数
        """
        # 目标函数
        obj = LinExpr()
        for i in tqdm(range(len(self.data.home_dict.keys()))):
            obj.addTerms(fixed_cost, x[i])
            for j in range(len(self.data.h3_data_dict.keys())):
                obj.addTerms(delivery_fee_per_km * d[i, j] * D[j], y[i, j])
                obj.addTerms(M, e[j])

        self.FLP_model.setObjective(obj, GRB.MINIMIZE)
        print('\n')
        print("========================= 目标函数建立成功！=========================")

        """
        在这里定义外卖柜选址优化模型的约束条件
        """
        lhs1 = LinExpr(0)
        for i in range(len(self.data.home_dict.keys())):
            lhs1.addTerms(1, x[i])
        self.FLP_model.addConstr(lhs1 >= S, name="Constraint_1")

        for i in range(len(self.data.home_dict.keys())):
            for j in range(len(self.data.h3_data_dict.keys())):
                self.FLP_model.addConstr(y[i, j] <= x[i], name="Constraint_2")

        for j in range(len(self.data.h3_data_dict.keys())):
            lhs3 = LinExpr(0)
            for i in range(len(self.data.home_dict.keys())):
                lhs3.addTerms(1, y[i, j])
            self.FLP_model.addConstr(lhs3 + e[j] == 1, name="Constraint_3")
            
        for i in range(len(self.data.home_dict.keys())):
            for j in range(len(self.data.h3_data_dict.keys())):
                if (d[i, j] >= 2):
                    self.FLP_model.addConstr(y[i, j] == 0, name="Constraint_4")

        print("========================= 约束条件设立成功！=========================")

        # 求解该模型
        self.FLP_model.optimize()

        # 可视化外卖柜选址输出
        if (self.FLP_model.SolCount <= 0):
            self.FLP_model.computeIIS()
            self.FLP_model.write('self.FLP_model.ilp')

        print(f'ObjVal: {self.FLP_model.ObjVal}')
        print("=========  选址决策 ========")
        print(" Home ID |   value  |  name, ")
        locker_num = 0

        # 用于存储导出的 locker_data
        locker_results = []
        for home_id in self.data.home_dict.keys():
            if (x[home_id].x >= 0.5):
                locker_results.append({
                    'locker_ID': self.data.home_dict[home_id].ID,
                    'locker_name': self.data.home_dict[home_id].name,
                    'lng': self.data.home_dict[home_id].lng,
                    'lat': self.data.home_dict[home_id].lat
                })

                print(f'{home_id} = {x[home_id].x}  {self.data.home_dict[home_id].name}')
                locker_num += 1

        # 导出到 csv 文件当中(id, name, lng, lat)
        locker_results_file = pd.DataFrame(locker_results)
        locker_file_name = f'locker_results_{S}.csv'
        locker_results_file.to_csv(locker_file_name, index=False, encoding='utf-8-sig')

        for h3_index in self.data.h3_data_dict.keys():
            if (e[h3_index].x > 0):
                print(f'{e[h3_index].VarName} = {e[h3_index].x}')

        print(f'选择的外卖柜数量: {locker_num}')



    # 新增动态约束后的订单指派优化模型
    def build_AP_model_old_version(self, TimeLimit=None):

        """========================= creating model ========================="""
        self.AP_model = Model("Assignment_Problem")


        """========================= define parameter ========================="""
        self.K = 50                          # 每个停机坪最多可容纳订单量
        self.accept_time = {}                # accept_time[i, j]: 订单 i 外卖商家到停机坪 j 的时间
        self.delivery_time = {}              # delivery_time[j, m]: 订单 i 从停机坪 j 到外卖柜 m 的时间
        self.dis = {}                        # dis[x, y]: 用于计算 x, y 两地点之间的距离
        self.start_time = {}                 # start_time[i]:每个订单 i 的 产生时间
        self.drone_speed = 36                # 无人机飞行速度
        self.delivery_person_speed = 20      # 外卖小哥配送速度
        self.wait_time = 1                   # 同一个停机坪每个订单开始配送的时间窗


        # 定义外卖商家到停机坪的时间参数 (T_i_j)
        for i in self.data.order_set_instance.keys():
            for j in self.data.station_dict.keys():
                self.dis[i, j] = self.data.compute_geo_dis(
                    lng1=self.data.order_set_instance[i].org_lng,
                    lat1=self.data.order_set_instance[i].org_lat,
                    lng2=self.data.station_dict[j].lng,
                    lat2=self.data.station_dict[j].lat
                )
                self.accept_time[i, j] = 60 * self.dis[i, j] / self.delivery_person_speed

        # 定义停机坪到外卖柜的时间参数 (T_j_m)
        for j in self.data.station_dict.keys():
            for m in range(len(self.data.locker_dict.keys())):
                self.dis[j, m] = self.data.compute_geo_dis(
                    lng1=self.data.station_dict[j].lng,
                    lat1=self.data.station_dict[j].lat,
                    lng2=self.data.locker_dict[m].lng,
                    lat2=self.data.locker_dict[m].lat
                )
                self.delivery_time[j, m] = (60 * self.dis[j, m] / self.drone_speed)+1

        # 定义第 i 个订单的发起时间 (S_i)
        for i in self.data.order_set_instance.keys():
            self.start_time[i] = self.data.order_set_instance[i].start_time


        """========================= define decision variables ========================="""
        self.x = {}         # x[i, j, k, m]: 订单分配决策变量

        self.p = {}         # p[i]: 第 i 个订单在停机坪的等待时间
        self.RT = {}        # RT[j, k]: 任一订单从第 j 个停机坪的第 k 个位置的离开时间
        self.a = {}         # a[i]: 第 i 个订单的开始配送时间（停机坪-外卖柜）
        self.b = {}         # b[i]: 第 i 个订单送至外卖柜的时间
        self.dt = {}        # dt[i]: 第 i 个订单从发起到配送至外卖柜的时间

        # 决定第 i 个订单是否被指派到第 j 个停机坪的第 k 个位置，并配送至第 m 个外卖柜的决策变量
        for i in tqdm(self.data.order_set_instance.keys()):
            for j in self.data.station_dict.keys():
                for k in range(self.K):
                    for m in range(len(self.data.locker_dict.keys())):
                        self.x[i, j, k, m] = self.AP_model.addVar(
                            lb=0, ub=1,
                            vtype=GRB.BINARY,
                            name=f"Decision variables: x_{i}_{j}_{k}_{m}"
                        )

        # 任一订单离开第 j 个停机坪中第 k 个位置的时间
        for j in self.data.station_dict.keys():
            for k in range(self.K):
                self.RT[j, k] = self.AP_model.addVar(
                    lb=0, ub=GRB.INFINITY,
                    vtype=GRB.CONTINUOUS,
                    name=f"Decision variables: RT_{j}_{k}"
                )

        # 决策变量的定义
        for i in self.data.order_set_instance.keys():
            self.p[i] = self.AP_model.addVar(
                lb=0, ub=GRB.INFINITY,
                vtype=GRB.CONTINUOUS,
                name=f"Decision variables: p_{i}"
            )   # 订单 i 在停机坪的等待时间
            self.a[i] = self.AP_model.addVar(
                lb=0, ub=GRB.INFINITY,
                vtype=GRB.CONTINUOUS,
                name=f"Decision variables: a_{i}"
            )   # 订单 i 的开始配送时间（停机坪-外卖柜）
            self.b[i] = self.AP_model.addVar(
                lb=0, ub=GRB.INFINITY,
                vtype=GRB.CONTINUOUS,
                name=f"Decision variables: b_{i}"
            )   # 订单 i 送达外卖柜的时间

            self.dt[i] = self.AP_model.addVar(
                lb=0, ub=GRB.INFINITY,
                vtype=GRB.CONTINUOUS,
                name=f"Decision variables: b_{i}"
            )   # 订单 i 从商家发起到送达外卖柜的时间戳


        """========================= creating an objective function ========================="""
        # 优化方向：使得所有订单从发起到送达的时间戳总和最小
        obj = LinExpr(0)
        for i in tqdm(self.data.order_set_instance.keys()):
            obj.addTerms(1, self.dt[i])

        self.AP_model.setObjective(obj, GRB.MINIMIZE)


        """========================= describe constraints ========================="""
        # 决策变量总和为 1
        for i in self.data.order_set_instance.keys():
            lhs = LinExpr(0)
            for j in self.data.station_dict.keys():
                for k in range(self.K):
                    for m in range(len(self.data.locker_dict.keys())):
                        lhs.addTerms(1, self.x[i, j, k, m])

            self.AP_model.addConstr(lhs == 1, name="Constraint_1")

        # 对于任一订单 i ，在任一停机坪 j 中最多之占用一个派单顺序位置 k
        for i in self.data.order_set_instance.keys():
            for j in self.data.station_dict.keys():
                lhs = LinExpr(0)
                for k in range(self.K):
                    for m in range(len(self.data.locker_dict.keys())):
                        lhs.addTerms(1, self.x[i, j, k, m])
                self.AP_model.addConstr(lhs <= 1, name="Constraint_2")

        # 对于任一停机坪 j 中的任一派单顺序位置 k 来说，最多只能被一个订单 i 占用
        for j in self.data.station_dict.keys():
            for k in range(self.K):
                lhs = LinExpr(0)
                for i in self.data.order_set_instance.keys():
                    for m in range(len(self.data.locker_dict.keys())):
                        lhs.addTerms(1, self.x[i, j, k, m])
                self.AP_model.addConstr(lhs <= 1, name="Constraint_3")

        # 对于任一停机坪 j 中的任一派单顺序位置 k 来说，只有当第 j 个停机坪的第 k-1 个位置使用后才有使用权
        for j in self.data.station_dict.keys():
            for k in range(self.K - 1):
                lhs1 = LinExpr(0)
                lhs2 = LinExpr(0)
                for i in self.data.order_set_instance.keys():
                    for m in range(len(self.data.locker_dict.keys())):
                        lhs1.addTerms(1, self.x[i, j, k, m])
                        lhs2.addTerms(1, self.x[i, j, k + 1, m])
                self.AP_model.addConstr(lhs1 >= lhs2, name="Constraint_4")

       # 对于每个订单 i 的等待时间约束刻画
        for i in self.data.order_set_instance.keys():
            lhs = LinExpr(0)
            for j in self.data.station_dict.keys():
                for k in range(self.K):
                    for m in range(len(self.data.locker_dict.keys())):
                        # lhs.addTerms(1, k * x[i, j, k, m])
                        lhs.addTerms(k, self.x[i, j, k, m])
            self.AP_model.addConstr(self.p[i] == lhs, name="Constraint_5")

        # 对于任一停机坪的第 k 个位置，需满足: 第 k-1 个位置订单的离开时间 + 时间窗 <= 第 k 个位置订单的离开时间
        for j in self.data.station_dict.keys():
            for k in range(1, self.K):
                lhs = LinExpr(0)
                for i in self.data.order_set_instance.keys():
                    for m in range(len(self.data.locker_dict.keys())):
                        lhs.addTerms(1, self.x[i, j, k, m])
                self.AP_model.addConstr(
                    self.RT[j, k-1] + lhs * self.wait_time <= self.RT[j, k],
                    name="Constraint_6"
                )

        # 对任一停机坪的第 k 个位置: 订单离开时间 >= 订单发起时间 + 停机坪接受订单时间
        for j in self.data.station_dict.keys():
            for k in range(self.K):
                lhs = LinExpr(0)
                for i in self.data.order_set_instance.keys():
                    for m in range(len(self.data.locker_dict.keys())):
                        lhs.addTerms(self.start_time[i] + self.accept_time[i, j], self.x[i, j, k, m])
                self.AP_model.addConstr(
                    self.RT[j, k] >= lhs,
                    name="Constraint_7"
                )

        # RT[j, 0] >= sum x[i, j, 0, m] * (start_time[i] + accept_time[i, j])
        # 任一停机坪的第 0 个位置的离开时间 >= (被指派到该 0 位置)订单的发起时间 + (被指派到该 0 位置)停机坪接收订单时间
        for j in self.data.station_dict.keys():
            lhs = LinExpr(0)
            lhs2 = LinExpr(0)
            for i in self.data.order_set_instance.keys():
                for m in range(len(self.data.locker_dict.keys())):
                    coef = self.start_time[i] + self.accept_time[i, j]
                    lhs.addTerms(coef, self.x[i, j, 0, m])
            self.AP_model.addConstr(
                self.RT[j, 0] >= lhs,
                name="Constraint_8"
            )

        # 对于任一订单，他从停机坪中某个位置的离开时间 <= 该订单的开始配送时间
        for i in self.data.order_set_instance.keys():
            for j in self.data.station_dict.keys():
                for k in range(self.K):
                    lhs = LinExpr(0)
                    for m in range(len(self.data.locker_dict.keys())):
                        lhs.addTerms(1, self.x[i, j, k, m])

                    # if lhs = 1, then RT[j, k] <= a[i]
                    self.AP_model.addConstr(
                        self.RT[j, k] - self.a[i] <= 10000 * (1 - lhs),
                        name="Constraint_9"
                    )

        # 对于每个订单 i: 从 j 停机坪的 k 位置离开时间 + 停机坪往外卖柜配送时间 <= 订单送达时间
        for i in self.data.order_set_instance.keys():
            for j in self.data.station_dict.keys():
                for k in range(self.K):
                    for m in range(len(self.data.locker_dict.keys())):

                        # if x[i,j,k,m] = 1, then RT[j, k] + delivery_time <= b[i]
                        self.AP_model.addConstr(
                            self.RT[j, k] + self.delivery_time[j, m] - self.b[i] <= 10000 * (1 - self.x[i, j, k, m]),
                            name="Constraint_10"
                        )

        # 对于每个订单 i: 全程时间戳 >= 送达时间 - 发起时间
        for i in self.data.order_set_instance.keys():
            self.AP_model.addConstr(
                self.dt[i] >= self.b[i] - self.start_time[i],
                name="Constraint_11"
            )


        """========================= solving the model ========================="""
        if(TimeLimit == None):
            self.AP_model.setParam("TimeLimit", 120)
        else:
            self.AP_model.setParam("TimeLimit", TimeLimit)
        # self.AP_model.write('model.lp')
        self.AP_model.optimize()


        """========================= visual of solution results ========================="""
        print(f"ObjVal: {self.AP_model.ObjVal}")
        for j in self.data.station_dict.keys():
            print(f"-------- 停机坪 {j} ---------")
            print("     x     |", end='')
            print("   Value  |", end='')
            print(" k   |", end='')
            print(" Start Time | ", end = '')
            print(" Delivery Time |", end='')
            print(" Arrive Time |")
            for k in range(self.K):
                for i in self.data.order_set_instance.keys():
                    for m in range(len(self.data.locker_dict.keys())):
                        if (self.x[i, j, k, m].x >= 0.5):
                            print(f'x_{i}_{j}_{k}_{m}:    {self.x[i, j, k, m].x}', end='  ')
                            print(f'  {k} ', end='     ')
                            print(f'  {self.start_time[i]} ', end='        ')
                            print(f'  {round(self.delivery_time[j, m], 3)} ', end='       ')
                            print(f'  {round(self.b[i].x, 3)}  ')



    def set_solving_para_in_simulation(
            self,
            numerical_dict=None
    ):
        # print(numerical_dict[Expr_id])

        self.cut_1 = numerical_dict['cut_1_flag']
        self.cut_2 = numerical_dict['cut_2_flag']
        self.cut_3 = numerical_dict['cut_3_flag']
        self.obj_lb = numerical_dict['obj_lb_flag']
        self.TimeLimit = numerical_dict['Time_limit']
        self.numerical_dict = numerical_dict
        self.M_flag = numerical_dict['M_flag']
        self.output_bound_flag = numerical_dict['output_bound_flag']



        print("========================================")
        print("          求解参数设置成功           ")
        print("========================================")

    def set_solving_para(
            self,
            Expr_id=None,
            numerical_dict=None,
            numerical_result_path=None,
            output_bound_path=None,
            output_log_path=None,
            output_plot_csv_flag=None
    ):
        # print(numerical_dict[Expr_id])

        self.Expr_id = Expr_id
        self.setting_id = numerical_dict[Expr_id]['setting_id']
        self.numerical_flag = True
        self.numerical_path = numerical_result_path
        self.cut_1 = numerical_dict[Expr_id]['cut_1_flag']
        self.cut_2 = numerical_dict[Expr_id]['cut_2_flag']
        self.cut_3 = numerical_dict[Expr_id]['cut_3_flag']
        self.obj_lb = numerical_dict[Expr_id]['obj_lb_flag']
        self.TimeLimit = numerical_dict[Expr_id]['Time_limit']
        self.numerical_dict = numerical_dict[Expr_id]
        self.M_flag = numerical_dict[Expr_id]['M_flag']
        self.output_bound_flag = numerical_dict[Expr_id]['output_bound_flag']
        self.output_bound_path = output_bound_path
        self.output_log_path = output_log_path

        self.output_plot_csv_flag = output_plot_csv_flag


        print("========================================")
        print("          求解参数设置成功           ")
        print("========================================")


    
    def build_AP_model_with_cut(
            self,
            Expr_id=None,
            numerical_dict=None,
            numerical_result_path=None,
            output_bound_path=None,
            output_log_path=None,
            output_plot_csv_flag=None
    ):

        """
        设置参数
        """
        self.set_solving_para(
            Expr_id=Expr_id,
            numerical_dict=numerical_dict,
            numerical_result_path=numerical_result_path,
            output_bound_path=output_bound_path,
            output_log_path=output_log_path,
            output_plot_csv_flag=output_plot_csv_flag
        )

        """ 开始建模求解 """
        self.last_record_time = -5
        record_data = []

        """========================= callback ========================="""
        # 定义回调函数
        def my_callback(model, where):

            if where == GRB.Callback.MIPSOL:
                # 获取当前时间
                current_time = model.cbGet(GRB.Callback.RUNTIME)

                # 每五秒钟记录一次
                # print(current_time - self.last_record_time)
                # if current_time - self.last_record_time >= 5 or round(current_time) == self.Time_limit:
                    # 获取当前的目标值和上下界

                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)  # 获取下界（最优值）
                obj_bound = model.cbGet(GRB.Callback.MIPSOL_OBJBND)  # 获取下界

                # 记录当前时间、目标值、下界和松弛根节点的最优值
                record_data.append([current_time, obj_val, obj_bound])

        """========================= creating model ========================="""
        self.AP_model = Model("Assignment_Problem")

        """========================= define parameter ========================="""
        self.K = 50                          # 每个停机坪最多可容纳订单量
        self.accept_time = {}                # accept_time[i, j]: 订单 i 外卖商家到停机坪 j 的时间
        self.delivery_time = {}              # delivery_time[j, m]: 订单 i 从停机坪 j 到外卖柜 m 的时间
        self.dis = {}                        # dis[x, y]: 用于计算 x, y 两地点之间的距离
        self.start_time = {}                 # start_time[i]:每个订单 i 的 产生时间
        self.drone_speed = 36                # 无人机飞行速度
        self.delivery_person_speed = 20      # 外卖小哥配送速度
        self.wait_time = 1                   # 同一个停机坪每个订单开始配送的时间窗
        self.dis_d_l = {} # 客户到外卖柜的距离参数 (dis_d_l)

        # 定义客户到外卖柜的距离参数 (dis_d_l)
        for i in self.data.order_set_instance.keys():
            for m in list(self.data.locker_dict.keys()):
                self.dis_d_l[i, m] = self.data.compute_geo_dis(
                    lng1=self.data.order_set_instance[i].des_lng,
                    lat1=self.data.order_set_instance[i].des_lat,
                    lng2=self.data.locker_dict[m].lng,
                    lat2=self.data.locker_dict[m].lat
                )
        # print(self.dis_d_l)
        self.dis_d_l_true = {i: [] for i in self.data.order_set_instance.keys()}  #定义一下每个订单的可选locker集合
        for i in self.data.order_set_instance.keys():
            for m in list(self.data.locker_dict.keys()):
                if self.dis_d_l[i, m] < 0.5:
                    self.dis_d_l_true[i].append(m)

        # print(self.dis_d_l_true)


        # 定义外卖商家到停机坪的时间参数 (T_i_j)
        for i in self.data.order_set_instance.keys():
            for j in self.data.station_dict.keys():
                self.dis[i, j] = self.data.compute_geo_dis(
                    lng1=self.data.order_set_instance[i].org_lng,
                    lat1=self.data.order_set_instance[i].org_lat,
                    lng2=self.data.station_dict[j].lng,
                    lat2=self.data.station_dict[j].lat
                )
                self.accept_time[i, j] = 60 * self.dis[i, j] / self.delivery_person_speed
        # print(self.dis)

        # 定义停机坪到外卖柜的时间参数 (T_j_m)
        for j in self.data.station_dict.keys():
            for m in list(self.data.locker_dict.keys()):
                self.dis[j, m] = self.data.compute_geo_dis(
                    lng1=self.data.station_dict[j].lng,
                    lat1=self.data.station_dict[j].lat,
                    lng2=self.data.locker_dict[m].lng,
                    lat2=self.data.locker_dict[m].lat
                )
                self.delivery_time[j, m] = 60 * self.dis[j, m] / self.drone_speed
        # print(self.delivery_time)

        # 定义第 i 个订单的发起时间 (S_i)
        for i in self.data.order_set_instance.keys():
            self.start_time[i] = self.data.order_set_instance[i].start_time


        """========================= define decision variables ========================="""
        if self.M_flag:
            M = 1440 + max(self.delivery_time.values()) - min(self.start_time.values())
            print(f'经过计算后，M的取值为：{M}')
        else:
            M = 1440

        self.x = {}         # x[i, j, k, m]: 订单分配决策变量
        self.p = {}         # p[i]: 第 i 个订单在停机坪的等待时间
        self.RT = {}        # RT[j, k]: 任一订单从第 j 个停机坪的第 k 个位置的离开时间
        self.a = {}         # a[i]: 第 i 个订单的开始配送时间（停机坪-外卖柜）
        self.b = {}         # b[i]: 第 i 个订单送至外卖柜的时间
        self.dt = {}        # dt[i]: 第 i 个订单从发起到配送至外卖柜的时间

        # 决定第 i 个订单是否被指派到第 j 个停机坪的第 k 个位置，并配送至第 m 个外卖柜的决策变量
        for i in tqdm(self.data.order_set_instance.keys()):
            for j in self.data.station_dict.keys():
                for k in range(self.K):
                    for m in list(self.data.locker_dict.keys()):
                        self.x[i, j, k, m] = self.AP_model.addVar(
                            lb=0, ub=1,
                            vtype=GRB.BINARY,
                            name=f"Decision variables: x_{i}_{j}_{k}_{m}"
                        )

        # 任一订单离开第 j 个停机坪中第 k 个位置的时间
        for j in self.data.station_dict.keys():
            for k in range(self.K):
                self.RT[j, k] = self.AP_model.addVar(
                    lb=0, ub=1440,
                    vtype=GRB.CONTINUOUS,
                    name=f"Decision variables: RT_{j}_{k}"
                )

        # 决策变量的定义
        for i in self.data.order_set_instance.keys():
            self.a[i] = self.AP_model.addVar(
                lb=0, ub=1440,
                vtype=GRB.CONTINUOUS,
                name=f"Decision variables: a_{i}"
            )   # 订单 i 的开始配送时间（停机坪-外卖柜）
            self.b[i] = self.AP_model.addVar(
                lb=self.start_time[i], ub=1440,
                vtype=GRB.CONTINUOUS,
                name=f"Decision variables: b_{i}"
            )   # 订单 i 送达外卖柜的时间

            self.dt[i] = self.AP_model.addVar(
                lb=0, ub=1440,
                vtype=GRB.CONTINUOUS,
                name=f"Decision variables: b_{i}"
            )   # 订单 i 从商家发起到送达外卖柜的时间戳


        """========================= creating an objective function ========================="""
        # 优化方向：使得所有订单从发起到送达的时间戳总和最小
        obj = LinExpr(0)
        for i in tqdm(self.data.order_set_instance.keys()):
            obj.addTerms(1, self.dt[i])

        self.AP_model.setObjective(obj, GRB.MINIMIZE)


        """========================= describe constraints ========================="""
        # 决策变量总和为 1
        for i in self.data.order_set_instance.keys():
            lhs = LinExpr(0)
            for j in self.data.station_dict.keys():
                for k in range(self.K):
                    for m in list(self.data.locker_dict.keys()):
                        lhs.addTerms(1, self.x[i, j, k, m])

            self.AP_model.addConstr(lhs == 1, name=f"Constraint_1_{i}")
            
        #对于候选locker来说，只会配送到距离客户位置小于某个值的locker
        for i in self.data.order_set_instance.keys():
            lhs = LinExpr(0)
            for j in self.data.station_dict.keys():
                for k in range(self.K):
                    for m in self.dis_d_l_true[i]:
                        lhs.addTerms(1, self.x[i, j, k, m])
            self.AP_model.addConstr(lhs == 1, name=f"Constraint_buchong_{i}")


        # 对于任一停机坪 j 中的任一派单顺序位置 k 来说，最多只能被一个订单 i 占用
        for j in self.data.station_dict.keys():
            for k in range(self.K):
                lhs = LinExpr(0)
                for i in self.data.order_set_instance.keys():
                    for m in list(self.data.locker_dict.keys()):
                        lhs.addTerms(1, self.x[i, j, k, m])
                self.AP_model.addConstr(lhs <= 1, name=f"Constraint_2_{j}_{k}")

        # 对任一停机坪的第 k 个位置: 订单离开时间 >= 订单发起时间 + 停机坪接受订单时间
        for j in self.data.station_dict.keys():
            for k in range(self.K):
                lhs = LinExpr(0)
                for i in self.data.order_set_instance.keys():
                    for m in list(self.data.locker_dict.keys()):
                        lhs.addTerms(self.start_time[i] + self.accept_time[i, j], self.x[i, j, k, m])
                self.AP_model.addConstr(
                    self.RT[j, k] >= lhs,
                    name=f"Constraint_4_{j}_{k}"
                )

        # 对于任一停机坪的第 k 个位置，需满足: 第 k-1 个位置订单的离开时间 + 时间窗 <= 第 k 个位置订单的离开时间
        for j in self.data.station_dict.keys():
            for k in range(1, self.K):
                lhs = LinExpr(0)
                for i in self.data.order_set_instance.keys():
                    for m in list(self.data.locker_dict.keys()):
                        lhs.addTerms(1, self.x[i, j, k-1, m])
                self.AP_model.addConstr(
                    self.RT[j, k - 1] + lhs * self.wait_time <= self.RT[j, k],
                    name=f"Constraint_5_{j}_{k}"
                )

        # 对于每个订单 i: 从 j 停机坪的 k 位置离开时间 + 停机坪往外卖柜配送时间 <= 订单送达时间
        for i in self.data.order_set_instance.keys():
            for j in self.data.station_dict.keys():
                for k in range(self.K):
                    for m in list(self.data.locker_dict.keys()):

                        # if x[i,j,k,m] = 1, then RT[j, k] + delivery_time <= b[i]
                        self.AP_model.addConstr(
                            self.RT[j, k] + self.delivery_time[j, m] - self.b[i] <= M * (1 - self.x[i, j, k, m]),
                            name=f"Constraint_7_{i}_{j}_{k}_{m}"
                        )

        # 对于每个订单 i: 全程时间戳 >= 送达时间 - 发起时间
        for i in self.data.order_set_instance.keys():
            self.AP_model.addConstr(
                self.dt[i] >= self.b[i] - self.start_time[i],
                name=f"Constraint_8_{i}"
            )


        self.cuts_num = {'cut1_num': 0, 'cut2_num': 0, 'cut3_num': 0, 'obj_lb_num': 0}
        """========================= obj_lb ========================="""
        min_order_time = {}
        if self.obj_lb:
            for i in self.data.order_set_instance.keys():
                obj_i_list = []
                for j in self.data.station_dict.keys():
                    for m in list(self.data.locker_dict.keys()):
                        obj_i_list.append(self.accept_time[i, j] + self.delivery_time[j, m])
                min_order_time[i] = min(obj_i_list)
                self.AP_model.addConstr(self.dt[i] >= min(obj_i_list), name=f"theta_lb_{i}")
                self.cuts_num['obj_lb_num'] += 1

            # self.AP_model.addConstr(obj >= sum(min_order_time.values()),name="obj_lb")  这种形式不太好其实
            # print(sum(min_order_time.values()))
        """========================= cuts ========================="""

        if self.cut_1:
            # 对于任一停机坪 j 中的任一派单顺序位置 k 来说，只有当第 j 个停机坪的第 k-1 个位置使用后才有使用权
            for j in self.data.station_dict.keys():
                for k in range(self.K - 1):
                    lhs1 = LinExpr(0)
                    lhs2 = LinExpr(0)
                    for i in self.data.order_set_instance.keys():
                        for m in list(self.data.locker_dict.keys()):
                            lhs1.addTerms(1, self.x[i, j, k, m])
                            lhs2.addTerms(1, self.x[i, j, k + 1, m])
                    self.AP_model.addConstr(lhs1 >= lhs2, name=f"Constraint_3_{j}_{k} or Non Skip Departure Inequality")
                    self.cuts_num['cut1_num'] += 1
        if self.cut_2:
            l_best_for_j_flag = {(i,j): 10000 for i in self.data.order_set_instance.keys() for j in
                                 self.data.station_dict.keys()}

            l_best_for_j = {}
            for i in self.data.order_set_instance.keys():
                for j in self.data.station_dict.keys():
                    for m in self.dis_d_l_true[i]:
                        if self.delivery_time[j, m] <= l_best_for_j_flag[i, j]:
                            l_best_for_j_flag[i, j] = self.delivery_time[j, m]
                            l_best_for_j[i, j] = m

            for i in self.data.order_set_instance.keys():
                for j in self.data.station_dict.keys():
                    for m in list(self.data.locker_dict.keys()):
                        if m != l_best_for_j[i, j]:
                            lhs1 = LinExpr(0)
                            lhs2 = LinExpr(0)
                            for k in range(self.K):
                                lhs1.addTerms(1, self.x[i, j, k, l_best_for_j[i, j]])
                                lhs2.addTerms(1, self.x[i, j, k, m])
                            self.AP_model.addConstr(lhs1 >= lhs2, name=f"Nestest Locker Cuts {i} {j} {m}")
                            self.cuts_num['cut2_num'] += 1

        if self.cut_3:
            l_best_for_j_flag = {(i, j): 10000 for i in self.data.order_set_instance.keys() for j in
                                 self.data.station_dict.keys()}
            l_best_for_j = {}
            for i in self.data.order_set_instance.keys():
                for j in self.data.station_dict.keys():
                    for m in self.dis_d_l_true[i]:
                        if self.delivery_time[j, m] <= l_best_for_j_flag[i, j]:
                            l_best_for_j_flag[i, j] = self.delivery_time[j, m]
                            l_best_for_j[i, j] = m

            dominance_set = {}
            for i in self.data.order_set_instance.keys():
                per_dominance_list = []
                for j in self.data.station_dict.keys():
                    for h in self.data.station_dict.keys():
                        if j != h:
                            if (self.accept_time[i, j] + self.wait_time * (len(self.data.order_set_instance) - 1) +
                                    l_best_for_j_flag[i, j] < self.accept_time[i, h]):
                                per_dominance_list.append((j, h))
                dominance_set[i] = per_dominance_list

            for i in self.data.order_set_instance.keys():
                for j, h in dominance_set[i]:
                    lhs1 = LinExpr(0)
                    lhs2 = LinExpr(0)
                    for k in range(self.K):
                        for m in list(self.data.locker_dict.keys()):
                            lhs1.addTerms(1, self.x[i, j, k, m])
                            lhs2.addTerms(1, self.x[i, h, k, m])
                    self.AP_model.addConstr(lhs1 >= lhs2, name=f"Dominance Cuts {i} {j} {h}")
                    self.cuts_num['cut3_num'] += 1

        """========================= solving the model ========================="""
        if(numerical_dict == None):
            self.AP_model.setParam("TimeLimit", 120)
        else:
            self.AP_model.setParam("TimeLimit", self.TimeLimit)
        # self.AP_model.write('model.lp')

        self.AP_model.update()
        self.opt_info = {'B_presolve_NumVars': self.AP_model.NumVars, 'B_presolve_NumConstrs': self.AP_model.NumConstrs,
        'A_presolve_NumVars': self.AP_model.presolve().NumVars, 'A_presolve_NumConstrs': self.AP_model.presolve().NumConstrs}

        #创建文件夹
        if not os.path.exists(output_log_path):
            os.makedirs(output_log_path)

        log_file_name = output_log_path + str(f'bound_Expr{Expr_id}_s{numerical_dict[Expr_id]["station_num"]}_l{numerical_dict[Expr_id]["locker_num"]}'+
                        f'_s{numerical_dict[Expr_id]["start_hour"]}_e{numerical_dict[Expr_id]["end_hour"]}_o{numerical_dict[Expr_id]["order_num"]}'+
                        f'_d{numerical_dict[Expr_id]["drone_num"]}_t{int(numerical_dict[Expr_id]["Time_limit"])}_s{numerical_dict[Expr_id]["seed"]}'+
                        f'_{str(self.cut_1)}_{str(self.cut_2)}_{str(self.cut_3)}' +
                        f'_{str(self.obj_lb)}'+'.log')
        self.AP_model.setParam(GRB.Param.LogFile, log_file_name)

        if self.output_bound_flag:
            self.AP_model.optimize(my_callback)
            # print(self.opt_info)
        else:
            self.AP_model.optimize()

        """========================= visual of solution results ========================="""
        print(f"ObjVal: {self.AP_model.ObjVal}")
        for j in self.data.station_dict.keys():
            print(f"-------- 停机坪 {j} ---------")
            print("     x     |", end='')
            print("   Value  |", end='')
            print(" k   |", end='')
            print(" Start Time | ", end = '')
            print(" Delivery Time |", end='')
            print(" Arrive Time |")
            for k in range(self.K):
                for i in self.data.order_set_instance.keys():
                    for m in list(self.data.locker_dict.keys()):
                        if (self.x[i, j, k, m].x >= 0.5):
                            print(f'x_{i}_{j}_{k}_{m}:    {self.x[i, j, k, m].x}', end='  ')
                            print(f'  {k} ', end='     ')
                            print(f'  {round(self.RT[j, k].x)} ', end='        ')
                            print(f'  {round(self.delivery_time[j, m], 3)} ', end='       ')
                            print(f'  {round(self.b[i].x, 3)}  ')

        if self.output_plot_csv_flag:
            df_result = pd.DataFrame(columns=['O_lng', 'O_lat', 'S_lng', 'S_lat', 'L_lng', 'L_lat', 'D_lng', 'D_lat'])
            for j in self.data.station_dict.keys():
                for k in range(self.K):
                    for i in self.data.order_set_instance.keys():
                        for m in list(self.data.locker_dict.keys()):
                            if (self.x[i, j, k, m].x >= 0.5):
                                row = {'O_lng': self.data.order_set_instance[i].org_lng,
                                       'O_lat': self.data.order_set_instance[i].org_lat,
                                        'S_lng': self.data.station_dict[j].lng,
                                        'S_lat': self.data.station_dict[j].lat,
                                        'L_lng': self.data.locker_dict[m].lng,
                                        'L_lat': self.data.locker_dict[m].lat,
                                        'D_lng': self.data.order_set_instance[i].des_lng,
                                         'D_lat': self.data.order_set_instance[i].des_lat}
                                df_result = pd.concat([df_result,pd.DataFrame([row])], ignore_index=True)
            df_result.to_csv(f'{Expr_id}.csv',index=False)

        if self.numerical_flag:
            if os.path.exists(self.numerical_path):
                results_df = pd.read_csv(self.numerical_path)
            else:

                # 初始化一个新的 DataFrame
                columns = ['Expr_id','setting_id', 'station_num', 'locker_num', 'start_hour', 'end_hour', 'order_num', 'drone_num',
                'Time_limit', 'seed','M_flag', 'cut_1', 'cut_2', 'cut_3', 'obj_lb', 'cut1_num', 'cut2_num', 'cut3_num',
                'obj_lb_num','nodes_num','B_presolve_NumVars','B_presolve_NumConstrs','A_presolve_NumVars',
                           'A_presolve_NumConstrs', 'Gap', 'Obj', 'Time']
                results_df = pd.DataFrame(columns=columns)

            # 检查求解状态
            if self.AP_model.SolCount != 0:
                obj = self.AP_model.ObjVal
                gap = self.AP_model.MIPGap
                time = self.AP_model.Runtime
                node_count = self.AP_model.getAttr('nodecount')
                obj_bound = self.AP_model.ObjBound
                # record_data.append([time,obj,obj_bound])
            else:
                obj = '-'
                gap = '-'
                time = self.TimeLimit
                node_count = self.AP_model.getAttr('nodecount')
            new_row = {'Expr_id':self.Expr_id,
                        'setting_id': self.setting_id,
                       'cut_1': self.cut_1,
                       'cut_2': self.cut_2,
                       'cut_3': self.cut_3,
                       'M_flag': self.M_flag,
                       'cut1_num': self.cuts_num['cut1_num'],
                       'cut2_num': self.cuts_num['cut2_num'],
                       'cut3_num': self.cuts_num['cut3_num'],
                       'obj_lb_num': self.cuts_num['obj_lb_num'],
                       'station_num': numerical_dict[Expr_id]['station_num'],
                       'locker_num': numerical_dict[Expr_id]['locker_num'],
                        'start_hour': numerical_dict[Expr_id]['start_hour'],
                        'end_hour': numerical_dict[Expr_id]['end_hour'],
                        'order_num': numerical_dict[Expr_id]['order_num'],
                        'drone_num': numerical_dict[Expr_id]['drone_num'],
                        'Time_limit': int(numerical_dict[Expr_id]['Time_limit']),
                        'seed': numerical_dict[Expr_id]['seed'],
                        'obj_lb': self.obj_lb,
                        'nodes_num': node_count,
                        'B_presolve_NumVars':self.opt_info['B_presolve_NumVars'],
                        'B_presolve_NumConstrs':self.opt_info['B_presolve_NumConstrs'],
                        'A_presolve_NumVars':self.opt_info['A_presolve_NumVars'],
                        'A_presolve_NumConstrs':self.opt_info['A_presolve_NumConstrs'],
                        'Gap': round(gap, 4),
                        'Obj': obj,
                        'Time': time
            }
            results_df = pd.concat([results_df, pd.DataFrame([new_row])],ignore_index=True)
            # 将追加后的结果保存回 CSV 文件
            results_df.to_csv(self.numerical_path, index=False)  # 保存为 CSV 文件

        if self.output_bound_flag:
            if not os.path.exists(output_bound_path):
                os.makedirs(output_bound_path)
            df_bound = pd.DataFrame(columns=['time', 'cur_obj', 'cur_bound', 'root_obj'])
            for i in range(len(record_data)):
                row = {
                    'time': record_data[i][0],
                    'cur_obj': record_data[i][1],
                    'cur_bound': record_data[i][2],
                }
                if i == 0:
                    row['root_obj'] = record_data[i][2]
                df_bound = pd.concat([df_bound, pd.DataFrame([row])], ignore_index=True)
            file_name = str(f'bound_Expr{Expr_id}_s{numerical_dict[Expr_id]["station_num"]}_l{numerical_dict[Expr_id]["locker_num"]}'+
                            f'_s{numerical_dict[Expr_id]["start_hour"]}_e{numerical_dict[Expr_id]["end_hour"]}_o{numerical_dict[Expr_id]["order_num"]}'+
                            f'_d{numerical_dict[Expr_id]["drone_num"]}_t{int(numerical_dict[Expr_id]["Time_limit"])}_s{numerical_dict[Expr_id]["seed"]}'+
                            f'_{str(self.cut_1)}_{str(self.cut_2)}_{str(self.cut_3)}' +
                            f'_{str(self.obj_lb)}.csv')
            df_bound.to_csv(output_bound_path+file_name, index=False)
        # print(record_data)




    def compute_waiting_for_delivery_order_num(self,station):
        num = 0
        for order in list(station.assigned_but_not_scheduled_order_set.values()):
            if order.status == 2 or order.status == 3:
                num += 1

        for order in list(station.assigned_and_scheduled_order_set.values()):
            if order.status == 2 or order.status == 3:
                num += 1
        return num




    def build_AP_model_with_cut_simulation(
            self,
            numerical_dict=None
    ):

        """
        设置参数
        """
        self.set_solving_para_in_simulation(numerical_dict=numerical_dict)

        """ 开始建模求解 """
        """========================= creating model ========================="""
        self.AP_model = Model("Assignment_Problem")

        """========================= define parameter ========================="""
        self.K = 50                          # 每个停机坪最多可容纳订单量
        self.accept_time = {}                # accept_time[i, j]: 订单 i 外卖商家到停机坪 j 的时间
        self.delivery_time = {}              # delivery_time[j, m]: 订单 i 从停机坪 j 到外卖柜 m 的时间
        self.dis = {}                        # dis[x, y]: 用于计算 x, y 两地点之间的距离
        self.start_time = {}                 # start_time[i]:每个订单 i 的 产生时间
        self.drone_speed = 36                # 无人机飞行速度
        self.delivery_person_speed = 20      # 外卖小哥配送速度
        self.wait_time = 1                   # 同一个停机坪每个订单开始配送的时间窗
        self.dis_d_l = {} # 客户到外卖柜的距离参数 (dis_d_l)

        # 定义客户到外卖柜的距离参数 (dis_d_l)
        for i in self.data.order_set_instance.keys():
            for m in list(self.data.locker_dict.keys()):
                self.dis_d_l[i, m] = self.data.compute_geo_dis(
                    lng1=self.data.order_set_instance[i].des_lng,
                    lat1=self.data.order_set_instance[i].des_lat,
                    lng2=self.data.locker_dict[m].lng,
                    lat2=self.data.locker_dict[m].lat
                )
        # print(self.dis_d_l)
        self.dis_d_l_true = {i: [] for i in self.data.order_set_instance.keys()}  # 定义一下每个订单的可选 locker集合
        for i in self.data.order_set_instance.keys():
            for m in list(self.data.locker_dict.keys()):
                if self.dis_d_l[i, m] < 0.5:
                    self.dis_d_l_true[i].append(m)

        # 定义外卖商家到停机坪的时间参数 (T_i_j)
        for i in self.data.order_set_instance.keys():
            for j in self.data.station_dict.keys():
                self.dis[i, j] = self.data.compute_geo_dis(
                    lng1=self.data.order_set_instance[i].org_lng,
                    lat1=self.data.order_set_instance[i].org_lat,
                    lng2=self.data.station_dict[j].lng,
                    lat2=self.data.station_dict[j].lat
                )
                self.accept_time[i, j] = 60 * self.dis[i, j] / self.delivery_person_speed
        # print(self.dis)

        # 定义停机坪到外卖柜的时间参数 (T_j_m)
        for j in self.data.station_dict.keys():
            for m in list(self.data.locker_dict.keys()):
                self.dis[j, m] = self.data.compute_geo_dis(
                    lng1=self.data.station_dict[j].lng,
                    lat1=self.data.station_dict[j].lat,
                    lng2=self.data.locker_dict[m].lng,
                    lat2=self.data.locker_dict[m].lat
                )
                self.delivery_time[j, m] = 60 * self.dis[j, m] / self.drone_speed
        # print(self.delivery_time)

        # 定义第 i 个订单的发起时间 (S_i)
        for i in self.data.order_set_instance.keys():
            self.start_time[i] = self.data.order_set_instance[i].start_time


        """========================= define decision variables ========================="""
        if self.M_flag:
            M = 1440 + max(self.delivery_time.values()) - min(self.start_time.values())
            print(f'经过计算后，M的取值为：{M}')
        else:
            M = 1440

        self.x = {}         # x[i, j, k, m]: 订单分配决策变量
        self.p = {}         # p[i]: 第 i 个订单在停机坪的等待时间
        self.RT = {}        # RT[j, k]: 任一订单从第 j 个停机坪的第 k 个位置的离开时间
        self.a = {}         # a[i]: 第 i 个订单的开始配送时间（停机坪-外卖柜）
        self.b = {}         # b[i]: 第 i 个订单送至外卖柜的时间
        self.dt = {}        # dt[i]: 第 i 个订单从发起到配送至外卖柜的时间


        # 决定第 i 个订单是否被指派到第 j 个停机坪的第 k 个位置，并配送至第 m 个外卖柜的决策变量
        for i in tqdm(self.data.order_set_instance.keys()):
            for j in self.data.station_dict.keys():
                self.occupied_k = self.compute_waiting_for_delivery_order_num(self.data.station_dict[j])
                for k in range(self.occupied_k, self.K):
                    for m in list(self.data.locker_dict.keys()):
                        self.x[i, j, k, m] = self.AP_model.addVar(
                            lb=0, ub=1,
                            vtype=GRB.BINARY,
                            name=f"Decision variables: x_{i}_{j}_{k}_{m}"
                        )

        # 任一订单离开第 j 个停机坪中第 k 个位置的时间
        for j in self.data.station_dict.keys():
            self.occupied_k = self.compute_waiting_for_delivery_order_num(self.data.station_dict[j])
            for k in range(self.occupied_k, self.K):
                self.RT[j, k] = self.AP_model.addVar(
                    lb=0, ub=1440,
                    vtype=GRB.CONTINUOUS,
                    name=f"Decision variables: RT_{j}_{k}"
                )

        # 决策变量的定义
        for i in self.data.order_set_instance.keys():
            self.a[i] = self.AP_model.addVar(
                lb=0, ub=1440,
                vtype=GRB.CONTINUOUS,
                name=f"Decision variables: a_{i}"
            )   # 订单 i 的开始配送时间（停机坪-外卖柜）
            self.b[i] = self.AP_model.addVar(
                lb=self.start_time[i], ub=1440,
                vtype=GRB.CONTINUOUS,
                name=f"Decision variables: b_{i}"
            )   # 订单 i 送达外卖柜的时间

            self.dt[i] = self.AP_model.addVar(
                lb=0, ub=1440,
                vtype=GRB.CONTINUOUS,
                name=f"Decision variables: dt_{i}"
            )   # 订单 i 从商家发起到送达外卖柜的时间戳


        """========================= creating an objective function ========================="""
        # 优化方向：使得所有订单从发起到送达的时间戳总和最小
        obj = LinExpr(0)
        for i in tqdm(self.data.order_set_instance.keys()):
            obj.addTerms(1, self.dt[i])

        self.AP_model.setObjective(obj, GRB.MINIMIZE)


        """========================= describe constraints ========================="""
        # 决策变量总和为 1
        for i in self.data.order_set_instance.keys():
            lhs = LinExpr(0)
            for j in self.data.station_dict.keys():
                self.occupied_k = self.compute_waiting_for_delivery_order_num(self.data.station_dict[j])
                for k in range(self.occupied_k, self.K):
                    for m in list(self.data.locker_dict.keys()):
                        lhs.addTerms(1, self.x[i, j, k, m])

            self.AP_model.addConstr(lhs == 1, name=f"Constraint_1_{i}")



        #计算一下a[i]的时间约束
        for i in self.data.order_set_instance.keys():
            lhs = LinExpr(0)
            for j in self.data.station_dict.keys():
                self.occupied_k = self.compute_waiting_for_delivery_order_num(self.data.station_dict[j])
                for k in range(self.occupied_k, self.K):
                    for m in list(self.data.locker_dict.keys()):
                        lhs.addTerms(self.delivery_time[j, m], self.x[i, j, k, m])
            self.AP_model.addConstr(self.b[i] - lhs == self.a[i], name=f"Constraint_compute_a{i}")

        """ 添加关于动态仿真的时间约束 """
        # 这一块是定义当前时刻，每个停机坪中的最大起飞时间，即: max_launching_time_dict[station_id]
        max_launching_time_dict = {}
        for station_id in self.data.station_dict.keys():
            max_lunching_time = 0
            # 遍历第 station_id个停机坪中被分配的订单（没有分配无人机）
            for order_id, order_object in self.data.station_dict[
                station_id].assigned_but_not_scheduled_order_set.items():
                this_lunching_time = order_object.start_delivery_time
                if (this_lunching_time > max_lunching_time):
                    max_lunching_time = this_lunching_time

            for order_id, order_object in self.data.station_dict[
                station_id].assigned_and_scheduled_order_set.items():
                this_lunching_time = order_object.start_delivery_time
                if (this_lunching_time > max_lunching_time):
                    max_lunching_time = this_lunching_time

            max_launching_time_dict[station_id] = max_lunching_time

            # 计算一下a[i]的时间约束
        """ if sum x[i,j,k,m]=1, then max_launching_time[station] <= a[i] """
        for i in self.data.order_set_instance.keys():
            for j in self.data.station_dict.keys():
                lhs = LinExpr(0)
                self.occupied_k = self.compute_waiting_for_delivery_order_num(self.data.station_dict[j])
                for k in range(self.occupied_k, self.K):
                    for m in list(self.data.locker_dict.keys()):
                        lhs.addTerms(1, self.x[i, j, k, m])
                self.AP_model.addConstr(
                    max_launching_time_dict[j] - self.a[i] <= 1440 * (1 - lhs),
                    name=f"Constraint_min_launching_time_{i}"
                )

        for i in self.data.order_set_instance.keys():
            lhs = LinExpr(0)
            for j in self.data.station_dict.keys():
                self.occupied_k = self.compute_waiting_for_delivery_order_num(self.data.station_dict[j])
                for k in range(self.occupied_k, self.K):
                    for m in self.dis_d_l_true[i]:
                        lhs.addTerms(1, self.x[i, j, k, m])
            self.AP_model.addConstr(lhs == 1, name=f"Constraint_buchong_{i}")


        # 对于任一停机坪 j 中的任一派单顺序位置 k 来说，最多只能被一个订单 i 占用
        for j in self.data.station_dict.keys():
            self.occupied_k = self.compute_waiting_for_delivery_order_num(self.data.station_dict[j])
            for k in range(self.occupied_k, self.K):
                lhs = LinExpr(0)
                for i in self.data.order_set_instance.keys():
                    for m in list(self.data.locker_dict.keys()):
                        lhs.addTerms(1, self.x[i, j, k, m])
                self.AP_model.addConstr(lhs <= 1, name=f"Constraint_2_{j}_{k}")


       # # 对于每个订单 i 的等待时间约束刻画
       #  for i in self.data.order_set_instance.keys():
       #      lhs = LinExpr(0)
       #      for j in self.data.station_dict.keys():
       #          for k in range(self.K):
       #              for m in range(len(self.data.locker_dict.keys())):
       #                  # lhs.addTerms(1, k * x[i, j, k, m])
       #                  lhs.addTerms(k, self.x[i, j, k, m])
       #      self.AP_model.addConstr(self.p[i] == lhs, name="Constraint_5")

        # 对任一停机坪的第 k 个位置: 订单离开时间 >= 订单发起时间 + 停机坪接受订单时间
        for j in self.data.station_dict.keys():
            self.occupied_k = self.compute_waiting_for_delivery_order_num(self.data.station_dict[j])
            for k in range(self.occupied_k, self.K):
                lhs = LinExpr(0)
                for i in self.data.order_set_instance.keys():
                    for m in list(self.data.locker_dict.keys()):
                        lhs.addTerms(self.start_time[i] + self.accept_time[i, j], self.x[i, j, k, m])
                self.AP_model.addConstr(
                    self.RT[j, k] >= lhs,
                    name=f"Constraint_4_{j}_{k}"
                )

        # 对于任一停机坪的第 k 个位置，需满足: 第 k-1 个位置订单的离开时间 + 时间窗 <= 第 k 个位置订单的离开时间
        for j in self.data.station_dict.keys():
            self.occupied_k = self.compute_waiting_for_delivery_order_num(self.data.station_dict[j])
            #排除k=0的情况
            if self.occupied_k == 0:
                self.occupied_k = 1
            for k in range(self.occupied_k, self.K-1):
                lhs = LinExpr(0)
                for i in self.data.order_set_instance.keys():
                    for m in list(self.data.locker_dict.keys()):
                        lhs.addTerms(1, self.x[i, j, k, m])
                self.AP_model.addConstr(
                    self.RT[j, k] + lhs * self.wait_time <= self.RT[j, k+1],
                    name=f"Constraint_5_{j}_{k}"
                )


        # 对于每个订单 i: 从 j 停机坪的 k 位置离开时间 + 停机坪往外卖柜配送时间 <= 订单送达时间
        for i in self.data.order_set_instance.keys():
            for j in self.data.station_dict.keys():
                self.occupied_k = self.compute_waiting_for_delivery_order_num(self.data.station_dict[j])
                for k in range(self.occupied_k, self.K):
                    for m in list(self.data.locker_dict.keys()):
                        # if x[i,j,k,m] = 1, then RT[j, k] + delivery_time <= b[i]
                        self.AP_model.addConstr(
                            self.RT[j, k] + self.delivery_time[j, m] - self.b[i] <= M * (1 - self.x[i, j, k, m]),
                            name=f"Constraint_7_{i}_{j}_{k}_{m}"
                        )

        # 对于每个订单 i: 全程时间戳 >= 送达时间 - 发起时间
        for i in self.data.order_set_instance.keys():
            self.AP_model.addConstr(
                self.dt[i] >= self.b[i] - self.start_time[i],
                name=f"Constraint_8_{i}"
            )
            
        self.cuts_num = {'cut1_num': 0, 'cut2_num': 0, 'cut3_num': 0, 'obj_lb_num': 0}
        """========================= obj_lb ========================="""
        min_order_time = {}
        if self.obj_lb:
            for i in self.data.order_set_instance.keys():
                obj_i_list = []
                for j in self.data.station_dict.keys():
                    for m in list(self.data.locker_dict.keys()):
                        obj_i_list.append(self.accept_time[i, j] + self.delivery_time[j, m])
                min_order_time[i] = min(obj_i_list)
                self.AP_model.addConstr(self.dt[i] >= min(obj_i_list), name=f"theta_lb_{i}")
                self.cuts_num['obj_lb_num'] += 1

            # self.AP_model.addConstr(obj >= sum(min_order_time.values()),name="obj_lb")  这种形式不太好其实
            # print(sum(min_order_time.values()))


        """========================= solving the model ========================="""
        if(numerical_dict == None):
            self.AP_model.setParam("TimeLimit", 120)
        else:
            self.AP_model.setParam("TimeLimit", self.TimeLimit)
            
        self.AP_model.update()
        
        self.AP_model.optimize()

        # 检查是否不可行
        if self.AP_model.status == GRB.INFEASIBLE:
            print("Model is infeasible")
            self.AP_model.computeIIS()
            iis_file = "model.ilp"
            self.AP_model.write(iis_file)
            print(f"IIS written to file {iis_file}")


        """========================= visual of solution results ========================="""
        print(f"ObjVal: {self.AP_model.ObjVal}")
        for j in self.data.station_dict.keys():
            print(f"-------- 停机坪 {j} ---------")
            print("     x     |", end='')
            print("   Value  |", end='')
            print(" k   |", end='')
            print(" Start Time | ", end = '')
            print(" Delivery Time |", end='')
            print(" Arrive Time |")
            self.occupied_k = self.compute_waiting_for_delivery_order_num(self.data.station_dict[j])
            for k in range(self.occupied_k, self.K):
                for i in self.data.order_set_instance.keys():
                    for m in list(self.data.locker_dict.keys()):
                        if (self.x[i, j, k, m].x >= 0.5):
                            print(f'x_{i}_{j}_{k}_{m}:    {self.x[i, j, k, m].x}', end='  ')
                            print(f'  {k} ', end='     ')
                            print(f'  {round(self.RT[j, k].x)} ', end='        ')
                            print(f'  {round(self.delivery_time[j, m], 3)} ', end='       ')
                            print(f'  {round(self.b[i].x, 3)}  ')



    def build_AP_model_old_version_2(self, TimeLimit=None):

        """========================= creating model ========================="""
        self.AP_model = Model("Assignment_Problem")


        """========================= define parameter ========================="""
        self.K = 50                          # 每个停机坪最多可容纳订单量
        self.accept_time = {}                # accept_time[i, j]: 订单 i 外卖商家到停机坪 j 的时间
        self.delivery_time = {}              # delivery_time[j, m]: 订单 i 从停机坪 j 到外卖柜 m 的时间
        self.dis = {}                        # dis[x, y]: 用于计算 x, y 两地点之间的距离
        self.start_time = {}                 # start_time[i]:每个订单 i 的 产生时间
        self.drone_speed = 83                # 无人机飞行速度
        self.delivery_person_speed = 20      # 外卖小哥配送速度
        self.wait_time = 1                   # 同一个停机坪每个订单开始配送的时间窗


        # 定义外卖商家到停机坪的时间参数 (T_i_j)
        for i in self.data.order_set_instance.keys():
            for j in self.data.station_dict.keys():
                self.dis[i, j] = self.data.compute_geo_dis(
                    lng1=self.data.order_set_instance[i].org_lng,
                    lat1=self.data.order_set_instance[i].org_lat,
                    lng2=self.data.station_dict[j].lng,
                    lat2=self.data.station_dict[j].lat
                )
                self.accept_time[i, j] = 60 * self.dis[i, j] / self.delivery_person_speed

        # 定义停机坪到外卖柜的时间参数 (T_j_m)
        for j in self.data.station_dict.keys():
            for m in range(len(self.data.locker_dict.keys())):
                self.dis[j, m] = self.data.compute_geo_dis(
                    lng1=self.data.station_dict[j].lng,
                    lat1=self.data.station_dict[j].lat,
                    lng2=self.data.locker_dict[m].lng,
                    lat2=self.data.locker_dict[m].lat
                )
                self.delivery_time[j, m] = 60 * self.dis[j, m] / self.drone_speed

        # 定义第 i 个订单的发起时间 (S_i)
        for i in self.data.order_set_instance.keys():
            self.start_time[i] = self.data.order_set_instance[i].start_time


        """========================= define decision variables ========================="""
        self.x = {}         # x[i, j, k, m]: 订单分配决策变量

        self.p = {}         # p[i]: 第 i 个订单在停机坪的等待时间
        self.RT = {}        # RT[j, k]: 任一订单从第 j 个停机坪的第 k 个位置的离开时间
        self.a = {}         # a[i]: 第 i 个订单的开始配送时间（停机坪-外卖柜）
        self.b = {}         # b[i]: 第 i 个订单送至外卖柜的时间
        self.dt = {}        # dt[i]: 第 i 个订单从发起到配送至外卖柜的时间

        # 决定第 i 个订单是否被指派到第 j 个停机坪的第 k 个位置，并配送至第 m 个外卖柜的决策变量
        for i in tqdm(self.data.order_set_instance.keys()):
            for j in self.data.station_dict.keys():
                for k in range(self.K):
                    for m in range(len(self.data.locker_dict.keys())):
                        self.x[i, j, k, m] = self.AP_model.addVar(
                            lb=0, ub=1,
                            vtype=GRB.BINARY,
                            name=f"Decision variables: x_{i}_{j}_{k}_{m}"
                        )

        # 任一订单离开第 j 个停机坪中第 k 个位置的时间
        for j in self.data.station_dict.keys():
            for k in range(self.K):
                self.RT[j, k] = self.AP_model.addVar(
                    lb=0, ub=1440,
                    vtype=GRB.CONTINUOUS,
                    name=f"Decision variables: RT_{j}_{k}"
                )

        # 决策变量的定义
        for i in self.data.order_set_instance.keys():
            self.a[i] = self.AP_model.addVar(
                lb=0, ub=1440,
                vtype=GRB.CONTINUOUS,
                name=f"Decision variables: a_{i}"
            )   # 订单 i 的开始配送时间（停机坪-外卖柜）
            self.b[i] = self.AP_model.addVar(
                lb=self.start_time[i], ub=1440,
                vtype=GRB.CONTINUOUS,
                name=f"Decision variables: b_{i}"
            )   # 订单 i 送达外卖柜的时间

            self.dt[i] = self.AP_model.addVar(
                lb=0, ub=1440,
                vtype=GRB.CONTINUOUS,
                name=f"Decision variables: b_{i}"
            )   # 订单 i 从商家发起到送达外卖柜的时间戳


        """========================= creating an objective function ========================="""
        # 优化方向：使得所有订单从发起到送达的时间戳总和最小
        obj = LinExpr(0)
        for i in tqdm(self.data.order_set_instance.keys()):
            obj.addTerms(1, self.dt[i])

        self.AP_model.setObjective(obj, GRB.MINIMIZE)


        """========================= describe constraints ========================="""
        # 决策变量总和为 1
        for i in self.data.order_set_instance.keys():
            lhs = LinExpr(0)
            for j in self.data.station_dict.keys():
                for k in range(self.K):
                    for m in range(len(self.data.locker_dict.keys())):
                        lhs.addTerms(1, self.x[i, j, k, m])

            self.AP_model.addConstr(lhs == 1, name=f"Constraint_1_{i}")

        # 对于任一停机坪 j 中的任一派单顺序位置 k 来说，最多只能被一个订单 i 占用
        for j in self.data.station_dict.keys():
            for k in range(self.K):
                lhs = LinExpr(0)
                for i in self.data.order_set_instance.keys():
                    for m in range(len(self.data.locker_dict.keys())):
                        lhs.addTerms(1, self.x[i, j, k, m])
                self.AP_model.addConstr(lhs <= 1, name=f"Constraint_2_{j}_{k}")

        # 对于任一停机坪 j 中的任一派单顺序位置 k 来说，只有当第 j 个停机坪的第 k-1 个位置使用后才有使用权
        for j in self.data.station_dict.keys():
            for k in range(self.K - 1):
                lhs1 = LinExpr(0)
                lhs2 = LinExpr(0)
                for i in self.data.order_set_instance.keys():
                    for m in range(len(self.data.locker_dict.keys())):
                        lhs1.addTerms(1, self.x[i, j, k, m])
                        lhs2.addTerms(1, self.x[i, j, k + 1, m])
                self.AP_model.addConstr(lhs1 >= lhs2, name=f"Constraint_3_{j}_{k}")

        # 对任一停机坪的第 k 个位置: 订单离开时间 >= 订单发起时间 + 停机坪接受订单时间
        for j in self.data.station_dict.keys():
            for k in range(self.K):
                lhs = LinExpr(0)
                for i in self.data.order_set_instance.keys():
                    for m in range(len(self.data.locker_dict.keys())):
                        lhs.addTerms(self.start_time[i] + self.accept_time[i, j], self.x[i, j, k, m])
                self.AP_model.addConstr(
                    self.RT[j, k] >= lhs,
                    name=f"Constraint_4_{j}_{k}"
                )

        # 对于任一停机坪的第 k 个位置，需满足: 第 k-1 个位置订单的离开时间 + 时间窗 <= 第 k 个位置订单的离开时间
        for j in self.data.station_dict.keys():
            for k in range(1, self.K):
                lhs = LinExpr(0)
                for i in self.data.order_set_instance.keys():
                    for m in range(len(self.data.locker_dict.keys())):
                        lhs.addTerms(1, self.x[i, j, k, m])
                self.AP_model.addConstr(
                    self.RT[j, k - 1] + lhs * self.wait_time <= self.RT[j, k],
                    name=f"Constraint_5_{j}_{k}"
                )

        # 对于任一订单，他从停机坪中某个位置的离开时间 <= 该订单的开始配送时间
        for i in self.data.order_set_instance.keys():
            for j in self.data.station_dict.keys():
                for k in range(self.K):
                    lhs = LinExpr(0)
                    for m in range(len(self.data.locker_dict.keys())):
                        lhs.addTerms(1, self.x[i, j, k, m])

                    # if lhs = 1, then RT[j, k] <= a[i]
                    self.AP_model.addConstr(
                        self.a[i] - self.RT[j, k] <= 1440 * (1 - lhs),
                        name=f"Constraint_6_{i}_{j}_{k}"
                    )

        # 对于每个订单 i: 从 j 停机坪的 k 位置离开时间 + 停机坪往外卖柜配送时间 <= 订单送达时间
        for i in self.data.order_set_instance.keys():
            for j in self.data.station_dict.keys():
                for k in range(self.K):
                    for m in range(len(self.data.locker_dict.keys())):

                        # if x[i,j,k,m] = 1, then RT[j, k] + delivery_time <= b[i]
                        self.AP_model.addConstr(
                            self.RT[j, k] + self.delivery_time[j, m] - self.b[i] <= 1440 * (1 - self.x[i, j, k, m]),
                            name=f"Constraint_7_{i}_{j}_{k}_{m}"
                        )

        # 对于每个订单 i: 全程时间戳 >= 送达时间 - 发起时间
        for i in self.data.order_set_instance.keys():
            self.AP_model.addConstr(
                self.dt[i] >= self.b[i] - self.start_time[i],
                name=f"Constraint_8_{i}"
            )

        """========================= solving the model ========================="""
        if(TimeLimit == None):
            self.AP_model.setParam("TimeLimit", 120)
        else:
            self.AP_model.setParam("TimeLimit", TimeLimit)
        # self.AP_model.write('model.lp')

        self.AP_model.setParam('Cuts', 1)
        self.AP_model.optimize()


        """========================= visual of solution results ========================="""
        print(f"ObjVal: {self.AP_model.ObjVal}")
        for j in self.data.station_dict.keys():
            print(f"-------- 停机坪 {j} ---------")
            print("     x     |", end='')
            print("   Value  |", end='')
            print(" k   |", end='')
            print(" Start Time | ", end = '')
            print(" Delivery Time |", end='')
            print(" Arrive Time |")
            for k in range(self.K):
                for i in self.data.order_set_instance.keys():
                    for m in range(len(self.data.locker_dict.keys())):
                        if (self.x[i, j, k, m].x >= 0.5):
                            print(f'x_{i}_{j}_{k}_{m}:    {self.x[i, j, k, m].x}', end='  ')
                            print(f'  {k} ', end='     ')
                            print(f'  {self.start_time[i]} ', end='        ')
                            print(f'  {round(self.delivery_time[j, m], 3)} ', end='       ')
                            print(f'  {round(self.b[i].x, 3)}  ')




    # 得到每个停机坪中被指派的订单列表解
    def get_solution(self):

        self.occupied_k_dict = {}
        for station_id in self.data.station_dict.keys():
            self.occupied_k = self.compute_waiting_for_delivery_order_num(self.data.station_dict[station_id])
            self.occupied_k_dict[station_id] = self.occupied_k

        """ 将 station_id 及对应指派后的 order_list 存入 self.assigned_order_set 字典中 """
        for station_id in self.data.station_dict.keys():
            occupied_k = self.occupied_k_dict[station_id]
            for k in range(occupied_k, self.K):
                """注意！！！！这边这个k和order_id的顺序不能反，不然会导致停机坪队列的顺序乱掉"""
                for order_id in self.data.order_set_instance.keys():
                    for locker_id in list(self.data.locker_dict.keys()):
                        if (self.x[order_id, station_id, k, locker_id].x >= 0.5):
                            self.data.station_dict[station_id].assigned_but_not_scheduled_order_set[order_id] = self.data.order_set_instance[order_id]
                            self.data.station_dict[station_id].assigned_but_not_scheduled_order_queue.append(order_id)
                            self.data.order_set_instance[order_id].start_delivery_time = copy.deepcopy(self.a[order_id].x)
                            self.data.order_set_instance[order_id].estimated_arrival_time = copy.deepcopy(self.b[order_id].x)
                            self.data.order_set_instance[order_id].delivery_time = self.delivery_time[station_id, locker_id]
                            self.data.order_set_instance[order_id].assigned_locker_ID = locker_id
                            self.data.order_set_instance[order_id].delivery_distance = self.dis[station_id, locker_id]
                            self.data.order_set_instance[order_id].status = 2
        print("提取解结束 ！！！")




    # 订单在配送过程（停机坪-外卖柜）的轨迹生成
    def delivery_orders_and_generate_trajectory(
            self,
            trajectory_time_interval=5,
            year_ID=2025,
            month_ID=3,
            day_ID=None,
            start_hour=None
    ):

        for station_id in self.data.station_dict.keys():
            idle_drone_id_list = list(self.data.station_dict[station_id].idle_drone_set.keys())
            copy_of_assigned_but_not_scheduled_order_set = copy.deepcopy(self.data.station_dict[station_id].assigned_but_not_scheduled_order_set)
            for order_id, order_object in copy_of_assigned_but_not_scheduled_order_set.items():

                self.data.station_dict[station_id].assigned_and_scheduled_order_set[order_id] = order_object
                del self.data.station_dict[station_id].assigned_but_not_scheduled_order_set[order_id]

                # 要给这个 order 指派一个无人机
                assigned_drone_id = idle_drone_id_list.pop()

                """ 更新无人机的状态 """
                self.data.drone_dict[assigned_drone_id].status = 'delivery'

                # 更新 idle drone set 与 occupied drone set
                self.data.station_dict[station_id].occupied_drone_set[assigned_drone_id] = \
                self.data.station_dict[station_id].idle_drone_set[assigned_drone_id]

                del self.data.station_dict[station_id].idle_drone_set[assigned_drone_id]

                # 更新 order 和 drone 的信息
                self.data.order_set_instance[order_id].assigned_drone_ID = assigned_drone_id
                self.data.drone_dict[assigned_drone_id].assigned_order_ID = order_id

                origin = [self.data.station_dict[station_id].lng, self.data.station_dict[station_id].lat]

                assigned_locker_id = self.data.order_set_instance[order_id].assigned_locker_ID
                destination = [self.data.locker_dict[assigned_locker_id].lng,
                               self.data.locker_dict[assigned_locker_id].lat]
                # 生成轨迹
                trajectory = {
                    'origin': origin,
                    'destination': destination,
                    'trajectory_time_interval': trajectory_time_interval,
                    'route_name': [],
                    'x_coor_list': [],
                    'y_coor_list': [],
                    'date_time_list': [],
                    'current_timestamp_idx_list': [],  # 这个是为了记录现在在这个边上的哪个位置，是为了方便计算轨迹的距离的
                    'drone_status_list': [],  # 这个是为了记录现在在这个边上的哪个位置，是为了方便计算轨迹的距离的
                    'cumulative_dis_list': [],  # 这个是为了记录到目前为止的距离
                    'order_name_list': [],
                    'drone_name_list': []
                }

                route_name = f'drone_{assigned_drone_id}'

                # 无人机在 停机坪-外卖柜 一个来回的距离
                total_delivery_distance = self.data.order_set_instance[order_id].delivery_distance

                delivery_time = 60 * self.data.order_set_instance[order_id].delivery_time
                timestamp_num = math.ceil(delivery_time / trajectory_time_interval)

                # 订单当前经纬度，即该订单所在停机坪经纬度
                current_lng = origin[0]
                current_lat = origin[1]

                # 订单在每个时间戳下的经度差值和纬度差值
                delta_lng = (destination[0] - origin[0]) / timestamp_num
                delta_lat = (destination[1] - origin[1]) / timestamp_num

                # 订单已经配送过的里程
                traveled_distance = 0

                # 订单当前时间：订单 i 在停机坪的开始配送时间 a[i]
                current_time = 60 * self.data.order_set_instance[order_id].start_delivery_time

                """ 停机坪--->外卖柜的轨迹生成 """
                # 对第 order_id 个订单的每个时间戳的 order_info 进行更新
                for cnt in range(0, timestamp_num + 1):
                    # 更新当前订单经纬度
                    current_lng += cnt * delta_lng
                    current_lat += cnt * delta_lat

                    # 使当前订单时间按照给定时间戳自增
                    current_time += trajectory_time_interval

                    # 更新当前订单时间
                    current_hour_ID = (int)((current_time) // (60 * 60))
                    current_minute_ID = (int)(((current_time) % (60 * 60)) // 60)
                    current_second_ID = (int)((((current_time) % (60 * 60))) % 60)

                    datetime_data = datetime(year_ID, month_ID, day_ID, current_hour_ID, current_minute_ID,
                                             current_second_ID)

                    datetime_str = datetime_data.strftime("%Y-%m-%d %H:%M:%S")

                    trajectory['date_time_list'].append(datetime_str)

                    trajectory['route_name'].append(route_name)
                    trajectory['x_coor_list'].append(current_lng)
                    trajectory['y_coor_list'].append(current_lat)
                    trajectory['current_timestamp_idx_list'].append(cnt)
                    trajectory['drone_status_list'].append('delivery')

                    traveled_distance = cnt * total_delivery_distance / timestamp_num

                    trajectory['cumulative_dis_list'].append(traveled_distance)

                    trajectory['order_name_list'].append(order_id)

                    trajectory['drone_name_list'].append(assigned_drone_id)

                """ 外卖柜--->停机坪的轨迹生成 """
                temp = origin
                origin = destination
                destination = temp
                # 订单在每个时间戳下的经度差值和纬度差值
                delta_lng = (destination[0] - origin[0]) / timestamp_num
                delta_lat = (destination[1] - origin[1]) / timestamp_num

                # 对第 order_id 个订单的每个时间戳的 order_info 进行更新
                for cnt in range(timestamp_num, 2*timestamp_num + 1):
                    # 更新当前订单经纬度
                    current_lng += (cnt-timestamp_num) * delta_lng
                    current_lat += (cnt-timestamp_num) * delta_lat

                    # 使当前订单时间按照给定时间戳自增
                    current_time += trajectory_time_interval

                    # 更新当前订单时间
                    current_hour_ID = (int)((current_time) // (60 * 60))
                    current_minute_ID = (int)(((current_time) % (60 * 60)) // 60)
                    current_second_ID = (int)((((current_time) % (60 * 60))) % 60)

                    datetime_data = datetime(year_ID, month_ID, day_ID, current_hour_ID, current_minute_ID,
                                             current_second_ID)

                    datetime_str = datetime_data.strftime("%Y-%m-%d %H:%M:%S")

                    trajectory['date_time_list'].append(datetime_str)

                    trajectory['route_name'].append(route_name)
                    trajectory['x_coor_list'].append(current_lng)
                    trajectory['y_coor_list'].append(current_lat)
                    trajectory['current_timestamp_idx_list'].append(cnt)
                    trajectory['drone_status_list'].append('back')

                    traveled_distance = cnt * total_delivery_distance / timestamp_num

                    trajectory['cumulative_dis_list'].append(traveled_distance)

                    trajectory['order_name_list'].append(order_id)

                    trajectory['drone_name_list'].append(assigned_drone_id)


                self.data.drone_dict[assigned_drone_id].trajectory = trajectory
                self.data.order_set_instance[order_id].trajectory = trajectory

        # 导出成 csv 文件
        trajectory_file_name = f'trajectory_results.csv'
        all_trajectory_results = {
            'route_name': [],
            'x_coor_list': [],
            'y_coor_list': [],
            'date_time_list': [],
            'current_timestamp_idx_list': [],  # 这个是为了记录现在在这个边上的哪个位置，是为了方便计算轨迹的距离的
            'cumulative_dis_list': [],  # 这个是为了记录到目前为止的距离
            'order_name_list': [],
            'drone_name_list': [],
            'drone_status_list': []
        }

        for station_id in self.data.station_dict.keys():
            for drone_id in self.data.station_dict[station_id].occupied_drone_set.keys():
                all_trajectory_results['route_name'] += self.data.drone_dict[drone_id].trajectory['route_name']
                all_trajectory_results['x_coor_list'] += self.data.drone_dict[drone_id].trajectory['x_coor_list']
                all_trajectory_results['y_coor_list'] += self.data.drone_dict[drone_id].trajectory['y_coor_list']
                all_trajectory_results['current_timestamp_idx_list'] += self.data.drone_dict[drone_id].trajectory[
                    'current_timestamp_idx_list']
                all_trajectory_results['cumulative_dis_list'] += self.data.drone_dict[drone_id].trajectory[
                    'cumulative_dis_list']
                all_trajectory_results['order_name_list'] += self.data.drone_dict[drone_id].trajectory[
                    'order_name_list']
                all_trajectory_results['drone_name_list'] += self.data.drone_dict[drone_id].trajectory[
                    'drone_name_list']
                all_trajectory_results['date_time_list'] += self.data.drone_dict[drone_id].trajectory['date_time_list']
                all_trajectory_results['drone_status_list'] += self.data.drone_dict[drone_id].trajectory['drone_status_list']

        all_trajectory_results_df = pd.DataFrame(all_trajectory_results)
        all_trajectory_results_df.T

        all_trajectory_results_df.to_csv(trajectory_file_name, index=None)
