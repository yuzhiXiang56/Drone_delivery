"""

"""
from tqdm import tqdm
# from goto import *
from collections import defaultdict
import pandas as pd
import numpy as np
import random
import math
import ast
import csv
import copy

import h3
# import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

from Drone import *
from Store import *
from Order import *
from Home import *
from Locker import *
from Station import *
from Order_instance import *


# 定义数据类
class Data():

    def __init__(self):

        self.station_list = []              # 停机坪列表
        self.drone_list = []                # 无人机列表
        self.station_dict_list = []         # 停机坪字典列表
        self.order_num = None               # 订单数量
        self.locker_num = None              # 外卖柜数量

        self.drone_dict = {}                # 无人机字典
        self.station_dict = {}              # 停机坪字典
        self.store_dict = {}                # 商家字典
        self.home_dict = {}                 # 商务住宅字典
        self.order_dict = {}                # 历史总订单字典
        self.locker_dict = {}               # 外卖柜字典
        self.time_interval_dict = {}        # 用于存储仿真系统时间戳的数据

        self.h3index_to_area_ID = {}
        self.area_ID_to_h3index = {}
        self.h3index_data = None
        self.h3_data_dict = {}              # h3 划分地图数据字典
        self.h3_id_list = []

        self.order_data = None

        self.order_set_instance = {}        # 指派订单字典（直接生成所需订单数额）
        self.store_reachable_home_df = None
        self.station_per_interval_data = {}



    # 读取指派算例参数数据
    def read_parameter_data(self, file_path):
        df = pd.read_excel(file_path, dtype=int)
        numerical_dict = {}
        for iter, row in df.iterrows():
            if row['Expr_id'] not in numerical_dict.keys():
                numerical_dict[row['Expr_id']] = {'setting_id': int(row['setting_id']),
                                              'station_num': int(row['station_num']),
                                             'locker_num': int(row['locker_num']),
                                             'start_hour': int(row['start_hour']),
                                             'end_hour': int(row['end_hour']),
                                             'order_num': int(row['order_num']),
                                             'drone_num': int(row['drone_num']),
                                             'Time_limit': int(row['Time_limit']),
                                             'seed': int(row['seed']),
                                             'locker_cut_flag': int(row['locker_cut_flag']),
                                             'M_flag': int(row['M_flag']),
                                             'output_bound_flag': int(row['output_bound_flag']),
                                              'cut_1_flag':int(row['cut_1_flag']),
                                              'cut_2_flag': int(row['cut_2_flag']),
                                              'cut_3_flag': int(row['cut_3_flag']),
                                              'obj_lb_flag': int(row['obj_lb_flag']),
                                             }
            else:
                numerical_dict[row['Expr_id']] = {'setting_id': int(row['setting_id']),
                                                  'station_num': int(row['station_num']),
                                             'locker_num': int(row['locker_num']),
                                             'start_hour': int(row['start_hour']),
                                             'end_hour': int(row['end_hour']),
                                             'order_num': int(row['order_num']),
                                             'drone_num': int(row['drone_num']),
                                             'Time_limit': int(row['Time_limit']),
                                             'seed': int(row['seed']),
                                             'locker_cut_flag': int(row['locker_cut_flag']),
                                             'M_flag': int(row['M_flag']),
                                             'output_bound_flag': int(row['output_bound_flag']),
                                              'cut_1_flag':int(row['cut_1_flag']),
                                              'cut_2_flag': int(row['cut_2_flag']),
                                              'cut_3_flag': int(row['cut_3_flag']),
                                              'obj_lb_flag': int(row['obj_lb_flag']),
                                             }
        return numerical_dict

    """ 关于H3地图和数据的处理、"""
    # 读取 h3 数据
    def read_h3data(self, path=None):
        self.h3index_data = pd.read_csv(path)

        """ 给 h3index 进行编号 """
        for i in range(len(self.h3index_data)):
            h3_index = self.h3index_data.iloc[i, 0]
            self.h3index_to_area_ID[h3_index] = i
            self.area_ID_to_h3index[i] = h3_index

        """ 为了后面提取方便，这里转化一下坐标的形式 """
        for i in range(len(self.h3index_data)):
            this_area = {
                'ID': i,
                'h3index': self.h3index_data.iloc[i, 0],
                'coor_list':[],
                'center_coor':[self.h3index_data.iloc[i, 13], self.h3index_data.iloc[i, 14]],
                'order_num':0
            }

            for k in range(6):
                this_area['coor_list'].append(
                    (self.h3index_data.iloc[i, 1+2*k], self.h3index_data.iloc[i, 1+2*k+1])
                )

            self.h3_data_dict[i] = this_area


    # 读取 h3data 中每个 area 的 ID 并放入列表 h3_id_list 中
    def read_h3data_ID(self, path=None):
        self.h3_id_list = pd.read_csv(path)
        self.h3index_data = pd.read_csv(path)

        for i in range(len(self.h3index_data)):
            h3_ID = self.h3index_data.iloc[i, 0]
            self.h3_id_list[i] = h3_ID
            self.h3_id_list.append


    # 利用 matplotlib 绘制 h3 图像
    def plot_h3(self, h3_level=None):

        import matplotlib.patches as patches

        """ 下面是画图的 """
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 10))
        # 设置纵横比为相等
        ax.set_aspect(aspect='equal')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        # ax.axis("off")
        # ax.set_xlim([min(x_list) - 0.05, max(x_list) + 0.05])
        # ax.set_ylim([min(y_list) - 0.05, max(y_list) + 0.05])

        x_list = []
        y_list = []
        for h3_area_id in self.h3_data_dict.keys():
            for x in self.h3_data_dict[h3_area_id]['coor_list']:
                x_list.append(x[1])
                y_list.append(x[0])
        x_max = max(x_list)
        y_max = max(y_list)

        ax.set_xlim([min(x_list) - 0.05, max(x_list) + 0.05])
        ax.set_ylim([min(y_list) - 0.05, max(y_list) + 0.05])

        """ 注意传进来的 current_zone_coor 横纵坐标反了，这里调整一下 """

        for h3_area_id in tqdm(self.h3_data_dict.keys()):
            hexagon_coor = [(coor[1], coor[0]) for coor in self.h3_data_dict[h3_area_id]['coor_list']]

            # hexagon_coor = self.h3_data_dict[h3_area_id]['coor_list']
            polygon = patches.Polygon(
                xy=hexagon_coor, fill=True, facecolor='#A0D6D8',
                edgecolor='#1E91C0', linewidth=0.01, alpha=0.8
            )
            ax.add_patch(polygon)

            # for i in range(len(current_zone_coor)):
            #     x = current_zone_coor[i][0]
            #     y = current_zone_coor[i][1]
            #
            #     ax.plot(x, y, linewidth=0.01,
            #             marker='o', markersize=10,
            #             color='blue', alpha=0.7
            #             )

            # x = self.h3_data_dict[h3_area_id]['center_coor'][1]
            # y = self.h3_data_dict[h3_area_id]['center_coor'][0]
            # ax.plot(x, y, linewidth=0.01,
            #         marker='o', markersize=2,
            #         color='black', alpha=0.7
            #         )

            # font_dict_occupy = dict(
            #     fontsize=3.5,
            #     color='k',
            #     family='Arial',  # 'Times New Roman',
            #     weight='light',
            #     # style='italic'
            # )

        fig_name = f'Shenzhen_h3_figure_level_{h3_level}.pdf'
        plt.savefig(fig_name)


    # 把20个 order_results_update_(0 - 19) 拼接成单个数据
    def read_order_data(self, order_data_dir=None):
        """

        要把20个数据文件拼接成单个数据

        :param order_data_dir:
        :return:
        """
        import os

        file_list = os.listdir(order_data_dir)

        cnt = 0
        for filename in file_list:
            filepath = order_data_dir + filename
            data = pd.read_csv(filepath)

            if(cnt <= 0):
                self.order_data = data
            else:
                self.order_data = pd.concat([self.order_data,data])

            cnt += 1


    # 判断一个点向左的射线跟一个多边形的交叉点有几个，如果结果为奇数的话那么说明这个点落在多边形中，反之则不在。
    # https://blog.csdn.net/enweitech/article/details/80654420
    # reference <一种判断点与多边形关系的快速算法>
    def IsPtInPoly(self, aLon, aLat, pointList) -> bool:
        ''''
        这个函数的作用：
            判断一个点（已知经纬度）是否在一个由一个点集pointList围成的多边形内。

        :param aLon: double 经度 -- 想要判断的点的精度
        :param aLat: double 纬度 -- 想要判断的点的纬度
        :param pointList: list [(lat, lon)...] 多边形点的顺序需根据顺时针或逆时针，不能乱
        :return: False 如果不在多边形内； True： 如果在多边形内
        '''
        iSum = 0
        iCount = len(pointList)

        if (iCount < 3):
            return False

        for i in range(iCount):

            pLon1 = pointList[i][1]
            pLat1 = pointList[i][0]

            if (i == iCount - 1):

                pLon2 = pointList[0][1]
                pLat2 = pointList[0][0]
            else:
                pLon2 = pointList[i + 1][1]
                pLat2 = pointList[i + 1][0]

            if ((aLat >= pLat1) and (aLat < pLat2)) or ((aLat >= pLat2) and (aLat < pLat1)):

                if (abs(pLat1 - pLat2) > 0):

                    pLon = pLon1 - ((pLon1 - pLon2) * (pLat1 - aLat)) / (pLat1 - pLat2);

                    if (pLon < aLon):
                        iSum += 1

        if (iSum % 2 != 0):
            return True
        else:
            return False


    # 计算每个 h3_index(area) 中包含的订单数量
    def count_order_num_per_h3_area(self):
        """
        根据订单数据和h3index的数据，统计每个h3 index所包含的订单数量

        :return:
        """

        # cnt = 0
        for order_id in tqdm(range(len(self.order_data))):
            """ 循环每个 h3index，查看该订单，是否落在这个 h3index中 """
            order_des_coor = [0, 0]
            order_des_coor_list = self.order_data.iloc[order_id, 4][1:-1].split(',')
            order_des_coor[0] = float(order_des_coor_list[1].strip())
            order_des_coor[1] = float(order_des_coor_list[0].strip())

            for h3_area_id in self.h3_data_dict.keys():
                """ 首先提取出 h3_area_id 的 coor_list"""

                coor_list = self.h3_data_dict[h3_area_id]['coor_list']

                # 引用 IsPtInPoly 来判断每个 order_des 是否在 h3_area 中
                is_in_this_area = self.IsPtInPoly(
                    aLon=order_des_coor[1],
                    aLat=order_des_coor[0],
                    pointList=coor_list
                )

                if(is_in_this_area == True):
                    self.h3_data_dict[h3_area_id]['order_num'] += 1

            # cnt += 1
            # if(cnt >= 10000):
            #     break

        """ 将结果导出成为 csv """
        order_num_by_h3index = pd.DataFrame(
            self.h3_data_dict
        )
        order_num_by_h3index = order_num_by_h3index.T

        order_num_by_h3index.to_csv(
            'order_num_by_h3index_8.csv',
            index=None
        )







    """ test """
    def count_order_num_per_h3_area_single_process(self, h3_index_list=None):
        """
        根据订单数据和h3index的数据，统计每个h3 index所包含的订单数量

        :return:
        """

        temp_h3_data_dict = copy.deepcopy(self.h3_data_dict)
        # cnt = 0
        for order_id in tqdm(range(len(self.order_data))):
            """ 循环每个 h3index，查看该订单，是否落在这个 h3index中 """
            order_des_coor = [0, 0]
            order_des_coor_list = self.order_data.iloc[order_id, 4][1:-1].split(',')
            order_des_coor[0] = float(order_des_coor_list[1].strip())
            order_des_coor[1] = float(order_des_coor_list[0].strip())

            for h3_area_id in h3_index_list:
                """ 首先提取出 h3_area_id 的 coor_list"""

                coor_list = temp_h3_data_dict[h3_area_id]['coor_list']

                # 引用 IsPtInPoly 来判断每个 order_des 是否在 h3_area 中
                is_in_this_area = self.IsPtInPoly(
                    aLon=order_des_coor[1],
                    aLat=order_des_coor[0],
                    pointList=coor_list
                )

                if(is_in_this_area == True):
                    temp_h3_data_dict[h3_area_id]['order_num'] += 1

        return temp_h3_data_dict

    def split_list(self, lst, n):
        k, m = divmod(len(lst), n)
        return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

    def count_order_num_per_h3_area_multiprocess(self, process_num=10, h3_level=None, file_id=None):
        """ 使用多进程进行统计 """
        import time
        from concurrent.futures import ProcessPoolExecutor  # 多进程模块

        """ 首先对 H3 进行分割 """
        h3_id_list_set = list(self.h3_data_dict.keys())
        h3_id_list_splitted_parts = self.split_list(h3_id_list_set, n=process_num)

        t1 = time.perf_counter()

        # 构造多进程池
        print(""" 构造多进程池 """)
        pool = ProcessPoolExecutor(process_num)
        fur_list = []
        for i in range(process_num):
            fur_list.append(pool.submit(self.count_order_num_per_h3_area_single_process
                                        , h3_id_list_splitted_parts[i]
                                        )
                            )
        pool.shutdown()

        """
        进行结果集成
        """
        for i in range(process_num):
            fur_object = fur_list[i]
            result = fur_object.result()
            for h3_area_id in self.h3_data_dict.keys():
                """ 首先提取出 h3_area_id 的 coor_list"""
                print(result[h3_area_id]['order_num'])
                self.h3_data_dict[h3_area_id]['order_num'] += result[h3_area_id]['order_num']


        # 计算用时
        t2 = time.perf_counter()
        print(f'  结束多进程，完成 订单统计，用时：{round(t2 - t1, 2)}')

        print(f'===== 正在将结果导出成为 csv ')
        """ 将结果导出成为 csv """
        order_num_by_h3index = pd.DataFrame(
            self.h3_data_dict
        )
        order_num_by_h3index = order_num_by_h3index.T

        order_num_by_h3index.to_csv(
            f'order_num_by_h3index_{h3_level}_new_{file_id}.csv',
            index=None
        )






    # 将 h3_data 中的 order_num 读入到 h3_data_dict中,用于外卖柜的优化选址
    def read_h3_data_for_FLP(
            self,
            h3_data_for_FLP_filepath=None
    ):
        h3_data = pd.read_csv(h3_data_for_FLP_filepath)

        # h3_id = 0
        for i in range(len(h3_data)):
            self.h3_data_dict[i]['order_num'] =  h3_data.iloc[i, 4]




    """ 通过停机坪库中数据和无人机对象生成: 停机坪-无人机算例 """
    # 生成停机坪算例
    def generate_station(
            self,
            station_num=None,
            shoppingcenter_filepath=None
    ):
        df = pd.read_excel(shoppingcenter_filepath)
        columns_to_extract = ['shopping_center_name', 'lng', 'lat']
        self.station_dict_list = df[columns_to_extract].to_dict(orient='records')

        self.station_list = [
            {
                'shopping_center_name': station_id['shopping_center_name'],
                'lng': station_id['lng'],
                'lat': station_id['lat'],
                'assigned_drone_set': {}
            }
            for station_id in random.sample(self.station_dict_list, station_num)
        ]

        # 将生成的停机坪数据存入到 station_dict 中，用于后续派单建模的解
        for station_id in range(len(self.station_list)):
            new_station = Station()
            new_station.ID = station_id
            new_station.name = self.station_list[station_id]['shopping_center_name']
            new_station.lng = self.station_list[station_id]['lng']
            new_station.lat = self.station_list[station_id]['lat']

            self.station_dict[station_id] = new_station

    def generate_station_by_dis(
            self,
            station_num=None,
            shoppingcenter_filepath=None,
            seed=None

    ):

        # 读取购物中心数据
        df = pd.read_excel(shoppingcenter_filepath)
        columns_to_extract = ['shopping_center_name', 'lng', 'lat']
        self.station_dict_list = df[columns_to_extract].to_dict(orient='records')
        seed = int(seed)
        # 循环直到选到一个满足条件的基准停机坪
        while True:
            random.seed(seed)
            base_station = random.choice(self.station_dict_list)
            base_lng = base_station['lng']
            base_lat = base_station['lat']

            # 计算每个站点与基准站点的距离
            distances = []
            for station in self.station_dict_list:
                if station != base_station:
                    distance = self.compute_geo_dis(
                        lng1=base_lng,
                        lat1=base_lat,
                        lng2=station['lng'],
                        lat2=station['lat']
                    )
                    distances.append((station, distance))

            # 按照距离排序
            distances.sort(key=lambda x: x[1])
            seed += 10
            # print(seed)

            # 如果最近的距离大于等于3km，则跳出循环
            if distances[0][1] <= 2:
                # print(distances)
                break

        # 选择最近的 station_num 个停机坪
        nearest_stations = [station for station, _ in distances[:station_num]]

        # 从这些站点中随机选取 station_num - 1 个
        random_stations = random.sample(nearest_stations, station_num - 1)

        # 将基准停机坪加入到最终的停机坪列表中
        selected_stations = [base_station] + random_stations

        # 构建 station_list
        self.station_list = [
            {
                'shopping_center_name': station['shopping_center_name'],
                'lng': station['lng'],
                'lat': station['lat'],
                'assigned_drone_set': {}
            }
            for station in selected_stations
        ]

        # 创建 Station 实例并存入 station_dict
        for station_id in range(len(self.station_list)):
            new_station = Station()
            new_station.ID = station_id
            new_station.name = self.station_list[station_id]['shopping_center_name']
            new_station.lng = self.station_list[station_id]['lng']
            new_station.lat = self.station_list[station_id]['lat']
            self.station_dict[station_id] = new_station

    # 生成无人机算例
    def generate_drone(self, drone_num=None):
        self.drone_list = [
            {
                'drone_id': f"drone_{str(i + 1).zfill(3)}",
                'drone_load': 2.5,  # 载重 2.5kg
                'drone_speed': 40,  # 速度 83km/h
                'drone_range': 10  # 里程 10km
            }
            for i in range(drone_num)
        ]

        cnt = 0
        for drone in self.drone_list:
            new_drone = Drone()
            new_drone.ID = cnt
            new_drone.load = drone['drone_load']
            new_drone.speed = drone['drone_speed']
            new_drone.range = drone['drone_range']

            self.drone_dict[cnt] = new_drone

            cnt += 1


    # 生成停机坪-无人机分配算例
    def distribute_drones_to_stations(self, seed=None):
        # 分配思路：先平均分配；若有剩余的话，其余随机分配
        drones_per_station = len(self.drone_list) // len(self.station_list)     # drones_per_station: 均匀分配数量
        remaining_drones = len(self.drone_list) % len(self.station_list)        # remaining_drones: 随机分配数量

        # drone_idx = 0
        # for station in self.station_list:
        #     # 初始化该站点的无人机分配字典
        #     station['assigned_drone_set'] = {}
        #     for i in range(drones_per_station):
        #         if drone_idx < len(self.drone_list):
        #             drone = self.drone_list[drone_idx]
        #             station['assigned_drone_set'][drone['drone_id']] = drone
        #             drone_idx += 1

        # 均匀分配
        drone_idx = 0
        new_station = Station()
        for station in self.station_dict.keys():
            for drone in range(drones_per_station):
                if drone_idx < len(self.drone_list):
                    self.station_dict[station].assigned_drone_set[drone_idx] = self.drone_dict[drone_idx]
                    self.station_dict[station].idle_drone_set[drone_idx] = self.drone_dict[drone_idx]
                    self.drone_dict[drone_idx]

                    drone_idx += 1

        # 随机分配
        random_stations = random.sample(list(self.station_dict.keys()), remaining_drones)
        for station in random_stations:
            random.seed(seed)
            # station['assigned_drone_set'][self.drone_list[drone_idx]['drone_id']] = self.drone_list[drone_idx]
            # new_station.assigned_drone_set[self.drone_list[drone_idx]['drone_id']] = self.drone_list[drone_idx]

            self.station_dict[station].assigned_drone_set[drone_idx] = self.drone_dict[drone_idx]
            self.station_dict[station].idle_drone_set[drone_idx] = self.drone_dict[drone_idx]
            seed += 1
            drone_idx += 1


    # 获得停机坪-无人机分配算例数据
    def get_station_distribution_instance(self):
        return self.station_list


    # 输出停机坪-无人机分配算例
    def output_station_distribution_instance(self):
        for station in self.get_station_distribution_instance():
            print(f"Station: {station['shopping_center_name']}, Drones: {len(station['assigned_drone_set'])}")
            for drone_id, drone in station['assigned_drone_set'].items():
                print(f"  {drone_id}: {drone}")




    """ 通过两地点经纬度的差值计算出实际地理位置的 distance """
    # 将两地点经纬度差值转换为 distance(km)
    def compute_geo_dis(
            self,
            lng1=None,
            lat1=None,
            lng2=None,
            lat2=None
    ):
        """
        这一步是将两地点经纬度差值计算为 distance (km)

        :param
        lng1, lat1: location1
        lng2, lat2: location2
        """
        if(lng1 == None):
            raise ValueError("lng1 no assignment!")
        if(lat1 == None):
            raise ValueError("lat1 no assignment!")
        if(lng2 == None):
            raise ValueError("lng2 no assignment!")
        if(lat2 == None):
            raise ValueError("lat2 no assignment!")

        # 将角度转换为弧度
        lng1, lat1, lng2, lat2 = map(math.radians, [lng1, lat1, lng2, lat2])

        # 地球半径，单位为公里
        R = 6371.0

        # 计算经纬度差值
        lat_difference = lat2 - lat1
        lng_difference = lng2 - lng1

        # Haversine公式
        a = math.sin(lat_difference / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(lng_difference / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        # 计算距离
        distance = R * c
        return distance




    """ 读取数据功能模块 """
    # 读取 stroe_data 和 home_data 并分别读入到 store_dict{} 和 home_dict{} 中
    def read_all_data(
            self,
            store_data_filepath=None,
            home_data_filepath=None
    ):
        """
        这一步是将 stroe_data 和 home_data 分别读入到 store_dict{} 和 home_dict{} 中

        store_data_filepath: "D:/Pycharm/pythonProject1/Drone_Delivery/Data/深圳市/提取后外卖数据（深圳）.xlsx"
        home_data_filepath: "D:/Pycharm/pythonProject1/Drone_Delivery/Data/深圳市/深圳市商务住宅数据.xlsx"
        """
        # 读取Excel文件(stroe_data、home_data)
        store_data = pd.read_excel(store_data_filepath)
        home_data = pd.read_excel(home_data_filepath)

        # 先读取 store 的数据
        store_id = 0
        for i in range(len(store_data)):
            # 处理new_store.month_sale的读入异常
            try:
                new_store = Store()
                new_store.ID = store_id
                new_store.name = store_data.iloc[i, 2]
                new_store.month_sale = (int)(store_data.iloc[i, 3])
                new_store.lat = (float)(store_data.iloc[i, 5])
                new_store.lng = (float)(store_data.iloc[i, 6])

                self.store_dict[i] = new_store

                store_id += 1
            except:
                new_store = Store()
                new_store.ID = store_id
                new_store.name = store_data.iloc[i, 2]
                new_store.month_sale = 0
                new_store.lat = (float)(store_data.iloc[i, 5])
                new_store.lng = (float)(store_data.iloc[i, 6])

                self.store_dict[i] = new_store

                store_id += 1

                print(f"Generate_store_data {i}: {new_store}")

        # 再读取 home 的数据
        home_id = 0
        for i in range(len(home_data)):
            new_home = Home()
            new_home.ID = home_id
            new_home.name = home_data.iloc[i, 0]
            new_home.lng = (float)(home_data.iloc[i, 1])
            new_home.lat = (float)(home_data.iloc[i, 2])

            self.home_dict[i] = new_home

            home_id += 1

            # print(f"Generate_home_data {i}: {new_home.name}, {new_home.lat}, {new_home.lng}")

        pass


    # 识别每个 store 可及的 home 生成一个 list
    def identify_store_reachable_home_list(
            self,
            store_data_filepath=None,
            home_data_filepath=None,
            dis_range=None
    ):
        """
        这一步是生成订单前的准备工作。
        主要的目的是，针对每个商家，识别其可以到达的住宅list，并导出成csv文件.
        :return:

        :param
        dis_range: 对于每个订单的最大配送范围
        """

        # 读取 store 和 home的数据，并将其整理到 store_dict 和 home_dict中。
        self.read_all_data(
            store_data_filepath="深圳市/提取后外卖数据（深圳）.xlsx",
            home_data_filepath="深圳市/深圳市商务住宅数据.xlsx"
        )

        # 对每个外卖商户来遍历可及的商务住宅
        for i in tqdm(self.store_dict.keys()):
            store = self.store_dict[i]
            store_lng = store.lng
            store_lat = store.lat
            for j in self.home_dict.keys():
                home = self.home_dict[j]
                home_lng = home.lng
                home_lat = home.lat

                distance = self.compute_geo_dis(
                    lng1=store_lng,
                    lat1=store_lat,
                    lng2=home_lng,
                    lat2=home_lat
                )
                if (distance <= dis_range):
                    self.store_dict[i].reachable_home_list.append(j)


        # # 根据每个外卖商户遍历可及的商务住宅的情况，生成CSV文件
        # reachable_result_df = pd.DataFrame(
        #     np.zeros([27000, 2]),
        #     columns=['store_id', 'reachable_home_list']
        # )
        #
        # # for i in tqdm(range(len(store_data_list))):
        # for i in tqdm(self.store_dict.keys()):
        #     reachable_result_df.iloc[i, 0] = i
        #     reachable_result_df.iloc[i, 1] = str(self.store_dict[i].reachable_home_list)
        #
        # # 导入到文件'reachable_results.csv'中
        # reachable_result_df.to_csv('reachable_results.csv')

        pass


    # 读取每个 store 可及的 home_list
    def read_store_reachable_home_list(
            self,
            reachable_home_list_path=None
    ):
        """
        从之前导出的 reachable_home_list_path 的文件中，读入每个 store可以到达的 home_list，方便之后生成订单。

        注意到之前已经生成了 store_dict，但是每个 store 可到达的商务住宅的 list 还是空的，该步骤就是为了讲这些数据填充完毕
        :param reachable_home_list_path:
        :return:
        """

        reachable_home_list_data = pd.read_csv(reachable_home_list_path)

        for store_id in tqdm(range(len(reachable_home_list_data))):
            reachable_home_list_str = reachable_home_list_data.iloc[store_id, 1]
            reachable_home_list_str = reachable_home_list_str[1:-1]
            reachable_home_list_str_list = reachable_home_list_str.split(',')

            home_id_list = []
            cnt = 0
            for home_id_str in reachable_home_list_str_list:
                try:
                    home_id_str = home_id_str.strip()
                    home_id_list.append((int)(home_id_str))

                    cnt += 1
                    # if(cnt > 50):
                    #     break
                except:
                    pass

            self.store_dict[store_id].reachable_home_list = home_id_list

        pass


    # 准备生成订单所需数据中的 (商家数据、 商务住宅数据)
    def prepare_data(
            self,
            store_data_file=None,   # "D:/Pycharm/pythonProject1/Drone_Delivery/Data/深圳市/提取后外卖数据（深圳）.xlsx"
            home_data_file=None     # "D:/Pycharm/pythonProject1/Drone_Delivery/Data/深圳市/深圳市商务住宅数据.xlsx"
    ):

        """
        读取 home_data 和 store data，并存储为对象
        """
        self.read_all_data(
            store_data_filepath=store_data_file,
            home_data_filepath=home_data_file
        )

        """
        读取 每个 store 的 reachable_home_list  
        """
        path = "深圳市/reachable_results_update.csv"
        self.read_store_reachable_home_list(
            reachable_home_list_path=path
        )


    # 读取 locker_results 数据
    def read_locker_results(
            self,
            locker_num=None,      # total_locker_num
            locker_results_filepath=None    # "D:/Pycharm/pythonProject1/Drone_Delivery/Data/深圳市/locker_results/locker_results_200.csv"
    ):
        # 从 csv 中读取 locker_data, 并将 locker_data 转换为 dateframe 类型
        df = pd.read_csv(locker_results_filepath)

        # 建立 Locker 对象，将数据存入到 locker_dict 中
        for locker_id in range(locker_num):
            new_locker = Locker()
            new_locker.ID = df.iloc[locker_id][0]
            new_locker.name = df.iloc[locker_id][1]
            new_locker.lng = df.iloc[locker_id][2]
            new_locker.lat = df.iloc[locker_id][3]

            self.locker_dict[locker_id] = new_locker

    def read_locker_results_cut(
            self,
            locker_num=None,      # total_locker_num
            locker_results_filepath=None    # "D:/Pycharm/pythonProject1/Drone_Delivery/Data/深圳市/locker_results/locker_results_200.csv"
    ):
        self.locker_dict = {}
        # 从 csv 中读取 locker_data, 并将 locker_data 转换为 dateframe 类型
        df = pd.read_csv(locker_results_filepath)

        # 建立 Locker 对象，将数据存入到 locker_dict 中
        for locker_id in range(locker_num):
            dis_list = []
            for order_id, order in self.order_set_instance.items():
                dis = self.compute_geo_dis(
                        lng1=order.des_lng,
                        lat1=order.des_lat,
                        lng2=df.iloc[locker_id][2],
                        lat2=df.iloc[locker_id][3]
                    )
                dis_list.append(dis)
                if min(dis_list) <= 1:
                    new_locker = Locker()
                    new_locker.ID = df.iloc[locker_id][0]
                    new_locker.name = df.iloc[locker_id][1]
                    new_locker.lng = df.iloc[locker_id][2]
                    new_locker.lat = df.iloc[locker_id][3]

                    self.locker_dict[locker_id] = new_locker

    def read_locker_results_cut_in_simulation(
            self,
            locker_num=None,      # total_locker_num
            locker_results_filepath=None    # "D:/Pycharm/pythonProject1/Drone_Delivery/Data/深圳市/locker_results/locker_results_200.csv"
    ):
        self.locker_dict = {}
        # 从 csv 中读取 locker_data, 并将 locker_data 转换为 dateframe 类型
        df = pd.read_csv(locker_results_filepath)

        # 建立 Locker 对象，将数据存入到 locker_dict 中
        for locker_id in range(locker_num):
            dis_list = []
            for order_id, order in self.order_set_instance.items():
                dis = self.compute_geo_dis(
                        lng1=order.des_lng,
                        lat1=order.des_lat,
                        lng2=df.iloc[locker_id][2],
                        lat2=df.iloc[locker_id][3]
                    )
                dis_list.append(dis)
                if min(dis_list) <= 2:
                    new_locker = Locker()
                    new_locker.ID = df.iloc[locker_id][0]
                    new_locker.name = df.iloc[locker_id][1]
                    new_locker.lng = df.iloc[locker_id][2]
                    new_locker.lat = df.iloc[locker_id][3]

                    self.locker_dict[locker_id] = new_locker





    """ 这一块用于订单数据的生成 （根据美团历史数据得出）"""
    # 将商户数据分为 num_parts 份，从而方便计算
    def split_list_into_parts(
            self,
            input_list=None,
            num_parts=10
    ):
        # 使用 numpy.array_split 将列表分成 num_parts 份
        return np.array_split(input_list, num_parts)


    # 根据历史数据生成一个月的订单数据并导出到 csv 文件
    def generate_one_month_order(self, split_group_num=20):
        """
        生成订单数据。
        :return:
        """

        """
        对每个store，按照其 月销量生成订单。
        """

        """
        首先将商户分成 split_group_num 份数
        """
        splitted_groups = self.split_list_into_parts(
            input_list=list(self.store_dict.keys()),
            num_parts=split_group_num
        )

        for i in range(len(splitted_groups)):
            splitted_groups[i] = list(splitted_groups[i])

        print("============================================================")
        print("                 正在生成订单数据...                           ")
        print("============================================================")
        order_id = 0

        for group_id in tqdm(range(len(splitted_groups))):

            print("============================================================")
            print(f"         正在生成订单数据,第{group_id}_update组                           ")
            print("============================================================")
            store_id_list = splitted_groups[group_id]

            order_results = []  # 为了导出文件方便
            # for store_id in tqdm(self.store_dict.keys()):
            for store_id in tqdm(store_id_list):

                # 如果这个 store的reachable homelist是空的，则跳过
                if (len(self.store_dict[store_id].reachable_home_list) <= 0):
                    continue

                month_sale = self.store_dict[store_id].month_sale

                # 生成 month_sale 个订单
                for this_order_id in range(month_sale):

                    # 生成一个新的 order 对象
                    new_order = Order()
                    new_order.ID = order_id
                    new_order.org_store_id = store_id
                    new_order.org_coor = [self.store_dict[store_id].lng, self.store_dict[store_id].lat]

                    # temp_num用于存储 rechable_results.csv 中 reachable_home_list(i) 中的 home_id
                    random.seed(order_id)   # 固定随机数种子，使其可以被复现
                    des_home_id = random.choice(self.store_dict[store_id].reachable_home_list)


                    new_order.des_home_id = des_home_id
                    new_order.des_coor = [self.home_dict[des_home_id].lng, self.home_dict[des_home_id].lat]

                    new_order.distance = self.compute_geo_dis(
                        self.store_dict[store_id].lng,
                        self.store_dict[store_id].lat,
                        self.home_dict[des_home_id].lng,
                        self.home_dict[des_home_id].lat
                    )
                    new_order.delivery_time = new_order.distance / self.drone_speed
                    new_order.start_time = random.randint(1, 720)
                    new_order.estimated_arrival_time = new_order.start_time + new_order.delivery_time

                    self.order_dict[order_id] = new_order

                    # Append the order information to the results list
                    order_results.append({
                        'order_id ID': order_id,
                        'org_store_id': new_order.org_store_id,
                        'org_coor': new_order.org_coor,
                        'des_home_id': new_order.des_home_id,
                        'des_coor': new_order.des_coor,
                        'start_time': new_order.start_time,
                        'estimated_arrival_time': new_order.estimated_arrival_time,
                        'distance': new_order.distance,
                        'delivery_time': new_order.delivery_time
                    })

                    order_id += 1

            print("============================================================")
            print("                 正在打印订单数据...                           ")
            print("============================================================")
            # 订单输出测试
            for order_id, order in self.order_dict.items():
                print(
                    f"Order ID: {order_id},"
                    f"Origin: {order.org_store_id},"
                    f" - : {order.org_coor},"
                    f"Destination: {order.des_home_id},"
                    f" - : {order.des_coor},"
                    f"Start Time: {order.start_time},"
                    f"Estimated_arrival Time: {order.estimated_arrival_time}"
                )

            print("============================================================")
            print("                 正在导出订单数据...                           ")
            print("============================================================")

            # 创建一个新的 csv 文件并将订单数据存入其中
            order_results_file = pd.DataFrame(order_results)

            result_file_name = f'order_results_group_{group_id}_update.csv'
            order_results_file.to_csv(result_file_name, index=False)


    # 根据一个月的订单数据生成一个时段(hour)内的订单
    def generate_hour_order(
            self,
            hour_num=None,
            order_num=None,
            reachable_home_list_path=None       # "D:/Pycharm/pythonProject1/Drone_Delivery/Data/深圳市/reachable_results_update.csv"
    ):
        """ hour_num: 用户输入参数规定的时间段长度，单位：小时 """
        hour_num = hour_num

        # 读取每个 store 可及的 home_list
        self.read_store_reachable_home_list(reachable_home_list_path=reachable_home_list_path)

        # 读取 reachable_home_list 数据
        df = pd.read_csv(reachable_home_list_path)

        # 为了方便导出 order_instace 数据到 csv 文件中
        order_instance_results = []

        temp_order_id_list = []
        order_instance_id = 0
        for store_id in tqdm(range(len(self.store_dict.keys()))):
        # for store_id in tqdm(range(3000)):        # 做测试用
            new_order_instance = Order_instance()

            # cnt:第 store_id 个商家的 order_num
            cnt = 0
            cnt = int(self.store_dict[store_id].month_sale * hour_num / 720)
            for j in range(cnt):

                new_order_instance.ID = order_instance_id

                new_order_instance.org_store_id = self.store_dict[store_id].ID
                new_order_instance.org_lng = self.store_dict[store_id].lng
                new_order_instance.org_lat = self.store_dict[store_id].lat
                # 随机选取一个订单起始时间（不包含在停机坪等待时间）
                new_order_instance.start_time = random.randint(0, 720)

                # 随机选取一个订单目的地
                reachable_home_list = ast.literal_eval(df.loc[store_id, 'reachable_home_list'])
                if reachable_home_list:
                    random_choice_home = random.choice(reachable_home_list)
                else:
                    break

                new_order_instance.des_home_id = random_choice_home
                new_order_instance.des_lng = self.home_dict[new_order_instance.des_home_id].lng
                new_order_instance.des_lat = self.home_dict[new_order_instance.des_home_id].lat

                self.order_set_instance[order_instance_id] = new_order_instance

                # order_instance_results.append({
                #     'order_ID': order_instance_id,
                #     'store_id': new_order_instance.org_store_id,
                #     'des_home_id': new_order_instance.des_home_id,
                #     'org_lng': new_order_instance.org_lng,
                #     'org_lat': new_order_instance.org_lat,
                #     'des_lng': new_order_instance.des_lng,
                #     'des_lat': new_order_instance.des_lat,
                #     'start_time': new_order_instance.start_time
                # })

                order_instance_id += 1


        all_order_id_list = list(self.order_set_instance.keys())
        random_order_id_list = random.sample(all_order_id_list, order_num)

        temp_order_id = 0
        self.order_set_instance = {}
        selected_order_instance_results = {}
        for order_id in random_order_id_list:
            self.order_set_instance[temp_order_id] = self.order_set_instance[order_id]

            selected_order_instance_results[order_id] = {
                'order_ID': self.order_set_instance[order_id].order_ID,
                'store_id': self.order_set_instance[order_id].org_store_id,
                'home_id': self.order_set_instance[order_id].des_home_id,
                'org_lng': self.order_set_instance[order_id].org_lng,
                'org_lat': self.order_set_instance[order_id].org_lat,
                'des_lng': self.order_set_instance[order_id].des_lng,
                'des_lat': self.order_set_instance[order_id].des_lat
            }

            temp_order_id += 1


        # 创建一个新的 csv 文件并将订单数据存入其中
        order_instance_results_df = pd.DataFrame(selected_order_instance_results).T

        order_result_name = f'order_instance_for_AP({hour_num}_hour)_order_num_{order_num}.csv'
        order_instance_results_df.to_csv(order_result_name, index=False)


    # 生成指派所需的订单算例
    def generate_order_of_instance(
            self,
            start_hour=None,
            end_hour=None,
            order_num=None,
            reachable_home_list_path=None       # "D:/Pycharm/pythonProject1/Drone_Delivery/Data/深圳市/reachable_results_update.csv"
    ):

        # 读取每个 store 可及的 home_list
        # self.read_store_reachable_home_list(reachable_home_list_path=reachable_home_list_path)

        # 读取 reachable_home_list 数据
        df = pd.read_csv(reachable_home_list_path)

        # 为了方便导出 order_instace 数据到 csv 文件中
        order_instance_results = []

        temp_order_id_list = []
        order_instance_id = 0

        order_cnt = 0
        order_id_list = list(self.store_dict.keys())
        while(order_cnt < order_num):
            # 选择一个商户
            # 固定随机数种子
            random.seed(order_cnt)
            selected_store_id = random.choice(order_id_list)
            new_order_instance = Order_instance()

            # 生成订单的起点
            new_order_instance.ID = order_cnt

            new_order_instance.org_store_id = self.store_dict[selected_store_id].ID
            new_order_instance.org_lng = self.store_dict[selected_store_id].lng
            new_order_instance.org_lat = self.store_dict[selected_store_id].lat
            # 随机选取一个订单起始时间（不包含在停机坪等待时间）

            start_min_num = start_hour * 60
            end_min_num = end_hour * 60
            # 固定随机数种子
            random.seed(order_cnt)
            new_order_instance.start_time = random.randint(start_min_num, end_min_num)

            # 随机选取一个订单目的地
            reachable_home_list = ast.literal_eval(df.loc[selected_store_id, 'reachable_home_list'])
            if reachable_home_list:
                # 固定随机数种子
                random.seed(order_cnt)
                random_choice_home = random.choice(reachable_home_list)
            else:
                continue

            new_order_instance.des_home_id = random_choice_home
            new_order_instance.des_lng = self.home_dict[new_order_instance.des_home_id].lng
            new_order_instance.des_lat = self.home_dict[new_order_instance.des_home_id].lat

            self.order_set_instance[order_cnt] = new_order_instance


            order_cnt += 1


        # all_order_id_list = list(self.order_set_instance.keys())
        # random_order_id_list = random.sample(all_order_id_list, order_num)

        # temp_order_id = 0
        # self.selected_orders = {}
        selected_order_instance_results = {}
        for order_id, order in self.order_set_instance.items():
            selected_order_instance_results[order_id] = {
                'order_ID': order.order_ID,
                'store_id': order.org_store_id,
                'home_id': order.des_home_id,
                'org_lng': order.org_lng,
                'org_lat': order.org_lat,
                'des_lng': order.des_lng,
                'des_lat': order.des_lat
            }


        # 创建一个新的 csv 文件并将订单数据存入其中
        order_instance_results_df = pd.DataFrame(selected_order_instance_results).T

        order_result_name = f'order_instance_for_AP_({start_hour}_{end_hour})_order_num_{order_num}.csv'
        order_instance_results_df.to_csv(order_result_name, index=False)


    # 用于仿真系统动态优化中所需的生成订单算例操作
    # 生成指派所需的订单算例
    def generate_order_of_instance_by_station_and_locker(
            self,
            Expr_id=None,
            numerical_dict=None,
            start_hour=None,
            end_hour=None,
            order_num=None,
            reachable_home_list_path=None,
            seed_flag=True,
            seed=None
            # "D:/Pycharm/pythonProject1/Drone_Delivery/Data/深圳市/reachable_results_update.csv"
    ):
        this_seed = None
        if seed_flag:
            this_seed = copy.deepcopy(numerical_dict[Expr_id]['seed'])
        else:
            this_seed = random.randint(1,1000000000)

        # 读取每个 store 可及的 home_list
        # self.read_store_reachable_home_list(reachable_home_list_path=reachable_home_list_path)

        # 读取 reachable_home_list 数据
        df = pd.read_csv(reachable_home_list_path)

        # 为了方便导出 order_instace 数据到 csv 文件中
        order_instance_results = []

        temp_order_id_list = []
        order_instance_id = 0

        order_cnt = 0
        order_id_list = list(self.store_dict.keys())
        while (order_cnt < order_num):
            print(f"Order cnt: {order_cnt}")
            # 选择一个商户
            # 固定随机数种子
            random.seed(this_seed)

            selected_store_id = random.choice(order_id_list)
            new_order_instance = Order_instance()

            # random.seed(order_cnt)
            selected_station_id = random.choice(list(self.station_dict.keys()))
            # 生成订单的起点
            new_order_instance.ID = order_cnt

            new_order_instance.org_store_id = self.store_dict[selected_store_id].ID
            # random.seed(order_cnt)
            new_order_instance.org_lng = self.station_dict[selected_station_id].lng + random.uniform(-0.005, 0.005)
            # random.seed(seed+12)
            new_order_instance.org_lat = self.station_dict[selected_station_id].lat + random.uniform(-0.005, 0.005)

            # 随机选取一个订单起始时间（不包含在停机坪等待时间）

            start_min_num = start_hour * 60
            end_min_num = end_hour * 60
            # 固定随机数种子
            # random.seed(seed+13)
            new_order_instance.start_time = random.randint(start_min_num, end_min_num)

            # 随机选取一个订单目的地
            reachable_home_list = ast.literal_eval(df.loc[selected_store_id, 'reachable_home_list'])
            if reachable_home_list:
                # 固定随机数种子
                # random.seed(seed+14)
                random_choice_home = random.choice(reachable_home_list)
            else:
                this_seed += 10
                continue

            new_order_instance.des_home_id = random_choice_home


            # selected_locker_id = random.choice(list(self.locker_dict.keys()))


            new_order_instance.des_home_id = random_choice_home

            while True:
                # random.seed(seed+1)
                new_order_instance.des_lng = self.station_dict[selected_station_id].lng + random.uniform(-0.08, 0.08)
                # print(random.uniform(-0.01, 0.01), end=' ')
                # random.seed(seed+2)
                new_order_instance.des_lat = self.station_dict[selected_station_id].lat + random.uniform(-0.08, 0.08)
                # print(random.uniform(-0.01, 0.01))
                dis_1 = self.compute_geo_dis(
                    lng1=self.station_dict[selected_station_id].lng,
                    lat1=self.station_dict[selected_station_id].lat,
                    lng2=new_order_instance.des_lng,
                    lat2=new_order_instance.des_lat
                )
                dis_d_l = {}
                for m in range(len(self.locker_dict.keys())):
                    dis_d_l[m] = self.compute_geo_dis(
                        lng1=new_order_instance.des_lng,
                        lat1=new_order_instance.des_lat,
                        lng2=self.locker_dict[m].lng,
                        lat2=self.locker_dict[m].lat
                    )
                this_seed += 10
                # print(f"current seed: {seed}")
                if dis_1 <= 4 and dis_1 >=2 and min(list(dis_d_l.values())) < 0.5:
                    self.order_set_instance[order_cnt] = new_order_instance
                    # print(dis_1, min(list(dis_d_l.values())))
                    break

            this_seed += 10
            order_cnt += 1

        # all_order_id_list = list(self.order_set_instance.keys())
        # random_order_id_list = random.sample(all_order_id_list, order_num)

        # temp_order_id = 0
        # self.selected_orders = {}
        selected_order_instance_results = {}
        for order_id, order in self.order_set_instance.items():
            selected_order_instance_results[order_id] = {
                'order_ID': order.order_ID,
                'store_id': order.org_store_id,
                'home_id': order.des_home_id,
                'org_lng': order.org_lng,
                'org_lat': order.org_lat,
                'des_lng': order.des_lng,
                'des_lat': order.des_lat
            }

        # 创建一个新的 csv 文件并将订单数据存入其中
        order_instance_results_df = pd.DataFrame(selected_order_instance_results).T

        station_num = numerical_dict[Expr_id]['station_num']
        locker_num = numerical_dict[Expr_id]['locker_num']
        drone_num = numerical_dict[Expr_id]['drone_num']
        seed = numerical_dict[Expr_id]['seed']
        order_result_name = f'Expr{Expr_id}_order_instance_for_AP_({start_hour}_{end_hour})_orderNum_{order_num}_stationNum_{station_num}_lockerNum_{locker_num}_droneNum_{drone_num}_seed_{seed}.csv'
        order_instance_results_df.to_csv(order_result_name, index=False)


    def generate_order_of_instance_by_station_and_locker_in_simulation_first(
            self,
            Expr_id=None,
            numerical_dict=None,
            start_hour=None,
            end_hour=None,
            order_num=None,
            reachable_home_list_path=None,
            seed_flag=True,
            seed=None
            # "D:/Pycharm/pythonProject1/Drone_Delivery/Data/深圳市/reachable_results_update.csv"
    ):
        this_seed = None
        if seed_flag:
            this_seed = copy.deepcopy(numerical_dict['seed'])
        else:
            this_seed = random.randint(1,1000000000)

        # # 读取每个 store 可及的 home_list
        # self.read_store_reachable_home_list(reachable_home_list_path=reachable_home_list_path)

        # 读取 reachable_home_list 数据
        df = pd.read_csv(reachable_home_list_path)

        # 为了方便导出 order_instace 数据到 csv 文件中
        order_instance_results = []

        temp_order_id_list = []
        order_instance_id = 0

        order_cnt = 0
        order_id_list = list(self.store_dict.keys())
        while (order_cnt < order_num):
            print(f"Order cnt: {order_cnt}")
            # 选择一个商户
            # 固定随机数种子
            random.seed(this_seed)

            selected_store_id = random.choice(order_id_list)
            new_order_instance = Order_instance()

            # random.seed(order_cnt)
            selected_station_id = random.choice(list(self.station_dict.keys()))
            # 生成订单的起点
            new_order_instance.ID = order_cnt

            new_order_instance.org_store_id = self.store_dict[selected_store_id].ID
            # random.seed(order_cnt)
            new_order_instance.org_lng = self.station_dict[selected_station_id].lng + random.uniform(-0.005, 0.005)
            # random.seed(seed+12)
            new_order_instance.org_lat = self.station_dict[selected_station_id].lat + random.uniform(-0.005, 0.005)

            # 随机选取一个订单起始时间（不包含在停机坪等待时间）

            start_min_num = start_hour * 60
            end_min_num = end_hour * 60
            # 固定随机数种子
            # random.seed(seed+13)
            new_order_instance.start_time = random.randint(start_min_num, end_min_num)

            # 随机选取一个订单目的地
            reachable_home_list = ast.literal_eval(df.loc[selected_store_id, 'reachable_home_list'])
            if reachable_home_list:
                # 固定随机数种子
                # random.seed(seed+14)
                random_choice_home = random.choice(reachable_home_list)
            else:
                this_seed += 10
                continue

            new_order_instance.des_home_id = random_choice_home


            # selected_locker_id = random.choice(list(self.locker_dict.keys()))


            new_order_instance.des_home_id = random_choice_home

            while True:
                # random.seed(seed+1)
                new_order_instance.des_lng = self.station_dict[selected_station_id].lng + random.uniform(-0.08, 0.08)
                # print(random.uniform(-0.01, 0.01), end=' ')
                # random.seed(seed+2)
                new_order_instance.des_lat = self.station_dict[selected_station_id].lat + random.uniform(-0.08, 0.08)
                # print(random.uniform(-0.01, 0.01))
                dis_1 = self.compute_geo_dis(
                    lng1=self.station_dict[selected_station_id].lng,
                    lat1=self.station_dict[selected_station_id].lat,
                    lng2=new_order_instance.des_lng,
                    lat2=new_order_instance.des_lat
                )
                dis_d_l = {}
                for m in range(len(self.locker_dict.keys())):
                    dis_d_l[m] = self.compute_geo_dis(
                        lng1=new_order_instance.des_lng,
                        lat1=new_order_instance.des_lat,
                        lng2=self.locker_dict[m].lng,
                        lat2=self.locker_dict[m].lat
                    )
                this_seed += 10
                # print(f"current seed: {seed}")
                if dis_1 <= 4 and dis_1 >=2 and min(list(dis_d_l.values())) < 0.5:
                    self.order_set_instance[order_cnt] = new_order_instance
                    # print(dis_1, min(list(dis_d_l.values())))
                    break

            this_seed += 10
            order_cnt += 1

        # all_order_id_list = list(self.order_set_instance.keys())
        # random_order_id_list = random.sample(all_order_id_list, order_num)

        # temp_order_id = 0
        # self.selected_orders = {}
        selected_order_instance_results = {}
        for order_id, order in self.order_set_instance.items():
            selected_order_instance_results[order_id] = {
                'order_ID': order.order_ID,
                'store_id': order.org_store_id,
                'home_id': order.des_home_id,
                'org_lng': order.org_lng,
                'org_lat': order.org_lat,
                'des_lng': order.des_lng,
                'des_lat': order.des_lat
            }



    def generate_order_of_instance_by_station_and_locker_in_simulation(
            self,
            time_interval_id=None,
            start_minute=None,
            end_minute=None,
            time_interval=None,
            order_num_lb=None,
            order_num_ub=None,
            reachable_home_list_path=None
            # "D:/Pycharm/pythonProject1/Drone_Delivery/Data/深圳市/reachable_results_update.csv"
    ):

        # # 读取每个 store 可及的 home_list
        # self.read_store_reachable_home_list(reachable_home_list_path=reachable_home_list_path)

        # 读取 reachable_home_list 数据
        df = pd.read_csv(reachable_home_list_path)
        this_seed = time_interval_id


        # 为了方便导出 order_instace 数据到 csv 文件中
        order_instance_results = []

        temp_order_id_list = []
        order_instance_id = 0
        random.seed(this_seed)
        order_num = random.randint(order_num_lb,order_num_ub)
        order_cnt = 0
        order_id_list = list(self.store_dict.keys())
        while (order_cnt < order_num):
            print(f"Order cnt: {order_cnt}")
            # 选择一个商户
            # 固定随机数种子
            random.seed(this_seed)

            selected_store_id = random.choice(order_id_list)
            new_order_instance = Order_instance()

            # random.seed(order_cnt)
            selected_station_id = random.choice(list(self.station_dict.keys()))

            order_ID = f'order_{time_interval_id}-{order_cnt}'

            # 生成订单的起点
            new_order_instance.ID = order_ID

            new_order_instance.org_store_id = self.store_dict[selected_store_id].ID
            # random.seed(order_cnt)
            new_order_instance.org_lng = self.station_dict[selected_station_id].lng + random.uniform(-0.005, 0.005)
            # random.seed(seed+12)
            new_order_instance.org_lat = self.station_dict[selected_station_id].lat + random.uniform(-0.005, 0.005)

            # 随机选取一个订单起始时间（不包含在停机坪等待时间）

            # 随机选取一个订单起始时间（不包含在停机坪等待时间）
            start_min_num = start_minute
            end_min_num = start_minute + time_interval
            # 固定随机数种子
            # random.seed(seed+13)
            new_order_instance.start_time = random.randint(start_min_num, end_min_num)

            # 随机选取一个订单目的地
            reachable_home_list = ast.literal_eval(df.loc[selected_store_id, 'reachable_home_list'])
            if reachable_home_list:
                # 固定随机数种子
                # random.seed(seed+14)
                random_choice_home = random.choice(reachable_home_list)
            else:
                this_seed += 10
                continue

            new_order_instance.des_home_id = random_choice_home

            # selected_locker_id = random.choice(list(self.locker_dict.keys()))

            new_order_instance.des_home_id = random_choice_home

            while True:
                # random.seed(seed+1)
                new_order_instance.des_lng = self.station_dict[selected_station_id].lng + random.uniform(-0.08, 0.08)
                # print(random.uniform(-0.01, 0.01), end=' ')
                # random.seed(seed+2)
                new_order_instance.des_lat = self.station_dict[selected_station_id].lat + random.uniform(-0.08, 0.08)
                # print(random.uniform(-0.01, 0.01))
                dis_1 = self.compute_geo_dis(
                    lng1=self.station_dict[selected_station_id].lng,
                    lat1=self.station_dict[selected_station_id].lat,
                    lng2=new_order_instance.des_lng,
                    lat2=new_order_instance.des_lat
                )
                dis_d_l = {}
                for m in list(self.locker_dict.keys()):
                    dis_d_l[m] = self.compute_geo_dis(
                        lng1=new_order_instance.des_lng,
                        lat1=new_order_instance.des_lat,
                        lng2=self.locker_dict[m].lng,
                        lat2=self.locker_dict[m].lat
                    )
                this_seed += 10
                # print(f"current seed: {seed}")
                if dis_1 <= 4 and dis_1 >= 2 and min(list(dis_d_l.values())) < 0.5:
                    self.order_set_instance[order_ID] = new_order_instance
                    # print(dis_1, min(list(dis_d_l.values())))
                    break

            this_seed += 10
            order_cnt += 1

        # all_order_id_list = list(self.order_set_instance.keys())
        # random_order_id_list = random.sample(all_order_id_list, order_num)

        # temp_order_id = 0
        # self.selected_orders = {}
        selected_order_instance_results = {}
        for order_id, order in self.order_set_instance.items():
            selected_order_instance_results[order_id] = {
                'order_ID': order.order_ID,
                'store_id': order.org_store_id,
                'home_id': order.des_home_id,
                'org_lng': order.org_lng,
                'org_lat': order.org_lat,
                'des_lng': order.des_lng,
                'des_lat': order.des_lat
            }

        # # 创建一个新的 csv 文件并将订单数据存入其中
        # order_instance_results_df = pd.DataFrame(selected_order_instance_results).T
        #
        # station_num = numerical_dict[Expr_id]['station_num']
        # locker_num = numerical_dict[Expr_id]['locker_num']
        # drone_num = numerical_dict[Expr_id]['drone_num']
        # seed = numerical_dict[Expr_id]['seed']
        # order_result_name = f'Expr{Expr_id}_order_instance_for_AP_({start_hour}_{end_hour})_orderNum_{order_num}_stationNum_{station_num}_lockerNum_{locker_num}_droneNum_{drone_num}_seed_{seed}.csv'
        # order_instance_results_df.to_csv(order_result_name, index=False)

    # 用于仿真系统动态优化中所需的生成订单算例操作

    def generate_order_of_instance_for_simulation(
            self,
            time_interval_id=None,
            start_minute=None,
            end_minute=None,
            time_interval=None,
            order_num=None,
            reachable_home_list_path=None       # "D:/Pycharm/pythonProject1/Drone_Delivery/Data/深圳市/reachable_results_update.csv"
    ):

        # 读取每个 store 可及的 home_list
        # self.read_store_reachable_home_list(reachable_home_list_path=reachable_home_list_path)

        # 读取 reachable_home_list 数据
        df = pd.read_csv(reachable_home_list_path)

        # 为了方便导出 order_instace 数据到 csv 文件中
        order_instance_results = []

        temp_order_id_list = []
        order_instance_id = 0

        order_cnt = 0
        order_id_list = list(self.store_dict.keys())
        while(order_cnt < order_num):
            # 选择一个商户
            # 固定随机数种子
            random.seed(time_interval_id * order_num + order_cnt)

            selected_store_id = random.choice(order_id_list)
            new_order_instance = Order_instance()

            order_ID = f'order_{time_interval_id}-{order_cnt}'

            # 生成订单的终点
            new_order_instance.ID = order_ID

            new_order_instance.org_store_id = self.store_dict[selected_store_id].ID
            new_order_instance.org_lng = self.store_dict[selected_store_id].lng
            new_order_instance.org_lat = self.store_dict[selected_store_id].lat

            # 随机选取一个订单起始时间（不包含在停机坪等待时间）
            start_min_num = start_minute
            end_min_num = start_minute + time_interval

            # 固定随机数种子
            random.seed(time_interval_id * order_num + order_cnt)
            new_order_instance.start_time = random.randint(start_min_num, end_min_num)

            new_order_instance.status = 1

            # 随机选取一个订单目的地
            reachable_home_list = ast.literal_eval(df.loc[selected_store_id, 'reachable_home_list'])
            if reachable_home_list:
                # 固定随机数种子
                random.seed(time_interval_id * order_num + order_cnt)
                random_choice_home = random.choice(reachable_home_list)
            else:
                continue

            new_order_instance.des_home_id = random_choice_home
            new_order_instance.des_lng = self.home_dict[new_order_instance.des_home_id].lng
            new_order_instance.des_lat = self.home_dict[new_order_instance.des_home_id].lat

            self.order_set_instance[order_ID] = new_order_instance


            order_cnt += 1


        # all_order_id_list = list(self.order_set_instance.keys())
        # random_order_id_list = random.sample(all_order_id_list, order_num)

        # temp_order_id = 0
        # self.selected_orders = {}
        selected_order_instance_results = {}
        for order_id, order in self.order_set_instance.items():
            selected_order_instance_results[order_id] = {
                'order_ID': order.order_ID,
                'store_id': order.org_store_id,
                'home_id': order.des_home_id,
                'org_lng': order.org_lng,
                'org_lat': order.org_lat,
                'des_lng': order.des_lng,
                'des_lat': order.des_lat
            }


        # # 创建一个新的 csv 文件并将订单数据存入其中
        # order_instance_results_df = pd.DataFrame(selected_order_instance_results).T
        #
        # order_result_name = f'order_instance_for_AP_({start_hour}_{end_hour})_order_num_{order_num}.csv'
        # order_instance_results_df.to_csv(order_result_name, index=False)