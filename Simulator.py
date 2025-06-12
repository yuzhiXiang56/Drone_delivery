import pandas as pd
import Data
from Data import *
from datetime import *
import ModelBuilder
import copy
import time
import os
import math
from collections import defaultdict
import random

class Simulator():

    def __init__(self):
        self.data = None
        self.unassigned_order_set = None
        self.assigned_order_set = {}
        self.unfinished_order_set = {}
        self.finished_order_set = {}
        self.idle_drone_set = {}

        self.system_minute = 0
        self.start_minute = None
        self.end_minute = None
        self.time_interval_dict = {}
        self.time_interval = None
        self.trajectory_time_interval = None

        self.start_datetime = None
        self.end_datetime = None

        self.year_ID = None
        self.month_ID = None
        self.day_ID = None

        self.order_num = None
        self.drone_num = None

        self.reachable_home_list_path = None

        self.model_solving_timelimit = None

        self.simulation_start_time = time.time()
        self.simulation_end_time = None
        self.simulation_CPU_time = None


        self.order_cnt_assigned_but_wait_drone_dispatch = 0
        self.order_cnt_assigned_and_wait_drone_delivery = 0
        self.order_cnt_delivery = 0

        self.result_file_dir = None



    def generate_time_interval_dict(self):
        time_interval_num = math.ceil((self.end_minute - self.start_minute) / self.time_interval)

        self.time_interval_dict = defaultdict(lambda : [0, 0])

        for i in range(time_interval_num):
            self.time_interval_dict[i][0] = self.start_minute + i * self.time_interval
            self.time_interval_dict[i][1] = self.time_interval_dict[i][0] + self.time_interval



    def distribute_drones_to_stations_for_simulation(self, seed=None):
        drones_per_station = len(self.data.drone_list) // len(self.data.station_list)
        remaining_drones = len(self.data.drone_list) % len(self.data.station_list)

        drone_idx = 0
        for station in self.data.station_dict.keys():
            for drone in range(drones_per_station):

                if drone_idx < len(self.data.drone_list):
                    self.data.station_dict[station].assigned_drone_set[drone_idx] = self.data.drone_dict[drone_idx]
                    self.data.station_dict[station].idle_drone_set[drone_idx] = self.data.drone_dict[drone_idx]

                    self.data.drone_dict[drone_idx].current_station_ID = station
                    drone_idx += 1

        random.seed(seed)
        random_stations = random.sample(list(self.data.station_dict.keys()), remaining_drones)
        for station in random_stations:
            # station['assigned_drone_set'][self.drone_list[drone_idx]['drone_id']] = self.drone_list[drone_idx]
            # new_station.assigned_drone_set[self.drone_list[drone_idx]['drone_id']] = self.drone_list[drone_idx]

            self.data.station_dict[station].assigned_drone_set[drone_idx] = self.data.drone_dict[drone_idx]
            self.data.station_dict[station].idle_drone_set[drone_idx] = self.data.drone_dict[drone_idx]

            self.data.drone_dict[drone_idx].current_station_ID = station
            drone_idx += 1



    def initialize_simulator(
            self,
            data=None,
            reachable_home_list_path="Data/深圳市/reachable_results_update.csv",
            result_file_dir=None,
            numerical_dict=None
    ):
        self.system_minute = numerical_dict['start_minute']

        self.start_minute = numerical_dict['start_minute']
        self.end_minute = numerical_dict['end_minute']

        # self.order_num = numerical_dict['start_minute']
        self.drone_num = numerical_dict['drone_num']

        self.time_interval = numerical_dict['time_interval']
        self.trajectory_time_interval = numerical_dict['trajectory_time_interval']

        self.reachable_home_list_path = reachable_home_list_path

        self.year_ID = numerical_dict['year_ID']
        self.month_ID = numerical_dict['month_ID']
        self.day_ID = numerical_dict['day_ID']

        self.model_solving_timelimit = numerical_dict['model_solving_timelimit']

        self.result_file_dir = result_file_dir
        self.numerical_dict = numerical_dict
        self.order_num_lb = numerical_dict['order_num_lb']
        self.order_num_ub = numerical_dict['order_num_ub']

        if not os.path.exists(self.result_file_dir):
            os.makedirs(self.result_file_dir)

        self.data = data

        self.generate_time_interval_dict()

        for drone_id, drone in self.data.drone_dict.items():
            drone.status = 'idle'

        self.distribute_drones_to_stations_for_simulation(seed=numerical_dict['seed'])

        self.start_datetime = datetime(
            self.year_ID,
            self.month_ID,
            self.day_ID,
            self.start_minute // 60,
            self.start_minute % 60,
            0
        )
        self.end_datetime = datetime(
            self.year_ID,
            self.month_ID,
            self.day_ID,
            self.end_minute // 60,
            self.end_minute % 60,
            0
        )



    def deploy_Simulation_log(self, time_interval_id):

        print("\n\n\n======================================================")
        print('%30s:  %-10d  ' % ('time_interval_id', time_interval_id))
        print('%30s:  %-10d  ' % ('unfinished order cnt', len(self.unfinished_order_set)))
        print("***************************************")
        print('%30s:  %-10d  ' % (' wait_drone_dispatch', self.order_cnt_assigned_but_wait_drone_dispatch))
        print('%30s:  %-10d  ' % (' wait_delivery', self.order_cnt_assigned_and_wait_drone_delivery))
        print('%30s:  %-10d  ' % (' delivering', self.order_cnt_delivery))
        print("***************************************")
        print('%30s:  %-10d  ' % ('finished order cnt', len(self.finished_order_set)))
        print('%30s:  %-10.1f  ' % ('system_minute', self.system_minute))
        print('%30s:  %-10.1f %2s ' % ('CPU time', self.simulation_CPU_time, 's'))
        print("======================================================\n\n\n")




    def generate_trajectory_for_simulation(
            self,
            trajectory_time_interval=None,
            year_ID=2025,
            month_ID=3,
            day_ID=None,
            start_hour=None
    ):
        trajectory_time_interval = self.trajectory_time_interval

        for station_id in self.data.station_dict.keys():
            idle_drone_id_list = list(self.data.station_dict[station_id].idle_drone_set.keys())

            copy_of_assigned_but_not_scheduled_order_set = copy.deepcopy \
                (self.data.station_dict[station_id].assigned_but_not_scheduled_order_set)
            for order_id, order_object in copy_of_assigned_but_not_scheduled_order_set.items():


                if (len(idle_drone_id_list) == 0):
                    break

                assigned_drone_id = idle_drone_id_list.pop()

                self.data.station_dict[station_id].assigned_but_not_scheduled_order_set[order_id].status = 3
                order_object.status = 3

                self.data.station_dict[station_id].assigned_and_scheduled_order_set[order_id] = order_object
                del self.data.station_dict[station_id].assigned_but_not_scheduled_order_set[order_id]

                self.unfinished_order_set[order_id] = order_object

                self.data.drone_dict[assigned_drone_id].assigned_order_ID = order_id
                assigned_locker_id = self.unfinished_order_set[order_id].assigned_locker_ID
                self.data.drone_dict[assigned_drone_id].assigned_locker_ID = assigned_locker_id
                self.data.drone_dict[assigned_drone_id].status = 'wait'

                self.unfinished_order_set[order_id].assigned_drone_ID = assigned_drone_id

                self.data.station_dict[station_id].occupied_drone_set[assigned_drone_id] = self.data.station_dict[station_id].idle_drone_set[assigned_drone_id]

                del self.data.station_dict[station_id].idle_drone_set[assigned_drone_id]

                self.unfinished_order_set[order_id].assigned_drone_ID = assigned_drone_id
                self.data.drone_dict[assigned_drone_id].assigned_order_ID = order_id

                origin = [self.data.station_dict[station_id].lng, self.data.station_dict[station_id].lat]

                self.data.drone_dict[assigned_drone_id].org_lng = self.data.station_dict[station_id].lng
                self.data.drone_dict[assigned_drone_id].org_lat = self.data.station_dict[station_id].lat

                self.data.drone_dict[assigned_drone_id].current_location = origin
                self.data.drone_dict[assigned_drone_id].current_trajectory_idx = 0

                assigned_locker_id = self.unfinished_order_set[order_id].assigned_locker_ID
                destination = [self.data.locker_dict[assigned_locker_id].lng,
                               self.data.locker_dict[assigned_locker_id].lat]

                self.data.drone_dict[assigned_drone_id].des_lng = self.data.locker_dict[assigned_locker_id].lng
                self.data.drone_dict[assigned_drone_id].des_lat = self.data.locker_dict[assigned_locker_id].lat

                trajectory = {
                    'origin': origin,
                    'destination': destination,
                    'trajectory_time_interval': trajectory_time_interval,
                    'route_name': [],
                    'x_coor_list': [],
                    'y_coor_list': [],
                    'date_time_list': [],
                    'current_minute_list': [],
                    'current_timestamp_idx_list': [],
                    'drone_status_list': [],
                    'cumulative_dis_list': [],
                    'order_name_list': [],
                    'drone_name_list': []
                }
                route_name = f'drone_{assigned_drone_id}_{self.data.drone_dict[assigned_drone_id].task_cnt}'

                self.data.drone_dict[assigned_drone_id].task_cnt += 1

                total_delivery_distance = self.unfinished_order_set[order_id].delivery_distance

                delivery_time = 60 * self.unfinished_order_set[order_id].delivery_time
                timestamp_num = math.ceil(delivery_time / trajectory_time_interval)

                current_lng = origin[0]
                current_lat = origin[1]

                delta_lng = (destination[0] - origin[0]) / timestamp_num
                delta_lat = (destination[1] - origin[1]) / timestamp_num

                traveled_distance = 0

                current_time = 60 * self.unfinished_order_set[order_id].start_delivery_time

                for cnt in range(0, timestamp_num + 1):

                    current_lng += cnt * delta_lng
                    current_lat += cnt * delta_lat

                    current_time += trajectory_time_interval

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
                    trajectory['current_minute_list'].append(current_time / 60)
                    trajectory['drone_status_list'].append('delivery')

                    traveled_distance = cnt * total_delivery_distance / timestamp_num

                    trajectory['cumulative_dis_list'].append(traveled_distance)

                    trajectory['order_name_list'].append(order_id)

                    trajectory['drone_name_list'].append(assigned_drone_id)

                temp = origin
                origin = destination
                destination = temp

                delta_lng = (destination[0] - origin[0]) / timestamp_num
                delta_lat = (destination[1] - origin[1]) / timestamp_num

                for cnt in range(timestamp_num, 2* timestamp_num + 1):

                    current_lng += (cnt - timestamp_num) * delta_lng
                    current_lat += (cnt - timestamp_num) * delta_lat

                    current_time += trajectory_time_interval

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
                    trajectory['current_minute_list'].append(current_time / 60)
                    trajectory['drone_status_list'].append('back')

                    traveled_distance = cnt * total_delivery_distance / timestamp_num

                    trajectory['cumulative_dis_list'].append(traveled_distance)

                    trajectory['order_name_list'].append(order_id)

                    trajectory['drone_name_list'].append(assigned_drone_id)

                self.data.drone_dict[assigned_drone_id].trajectory = copy.deepcopy(trajectory)
                self.data.drone_dict[assigned_drone_id].remaining_trajectory = copy.deepcopy(trajectory)
                self.data.drone_dict[assigned_drone_id].current_trajectory_timestamp_idx = -1

                self.unfinished_order_set[order_id].trajectory = trajectory

        trajectory_file_name = f'trajectory_results.csv'
        all_trajectory_results = {
            'route_name': [],
            'x_coor_list': [],
            'y_coor_list': [],
            'date_time_list': [],
            'current_minute_list': [],
            'current_timestamp_idx_list': [],
            'cumulative_dis_list': [],
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
                all_trajectory_results['current_minute_list'] += self.data.drone_dict[drone_id].trajectory[
                    'current_minute_list']
                all_trajectory_results['cumulative_dis_list'] += self.data.drone_dict[drone_id].trajectory[
                    'cumulative_dis_list']
                all_trajectory_results['order_name_list'] += self.data.drone_dict[drone_id].trajectory[
                    'order_name_list']
                all_trajectory_results['drone_name_list'] += self.data.drone_dict[drone_id].trajectory[
                    'drone_name_list']
                all_trajectory_results['date_time_list'] += self.data.drone_dict[drone_id].trajectory['date_time_list']
                all_trajectory_results['drone_status_list'] += self.data.drone_dict[drone_id].trajectory[
                    'drone_status_list']

        # all_trajectory_results_df = pd.DataFrame(all_trajectory_results)
        # all_trajectory_results_df.T
        #
        # all_trajectory_results_df.to_csv(trajectory_file_name, index=None)




    def check_whether_drone_starts_delivery(self, drone_id):
        is_started = False
        current_timestamp_idx = None

        traj_start_time = self.data.drone_dict[drone_id].trajectory['current_minute_list'][0]
        if (self.system_minute >= traj_start_time):
            is_started = True

            for i in range(len(self.data.drone_dict[drone_id].trajectory['current_minute_list'])):
                if (self.system_minute >= self.data.drone_dict[drone_id].trajectory['current_minute_list'][i]):
                    current_timestamp_idx = i
                    break

        return is_started, current_timestamp_idx




    def update_system_status(self, time_interval_id=None):
        self.system_minute = self.start_minute + time_interval_id * self.time_interval

        print(f"当前系统时间：{self.system_minute}")

        if (self.system_minute == self.start_minute):
            return

        print("====================================================")
        print(f"     interval:{time_interval_id},  更新前的状态    ")
        for station_id in self.data.station_dict.keys():
            print(f"停机坪 {station_id} ：{self.data.station_dict[station_id].assigned_but_not_scheduled_order_queue}")
        print("====================================================")

        elapsed_timestamp_cnt = math.ceil(self.time_interval * 60 // self.trajectory_time_interval)
        for drone_id, drone in self.data.drone_dict.items():

            if (drone.status == 'idle'):

                is_in_one_station_occupied_drone_list = False
                for station_id in self.data.station_dict.keys():
                    for each_drone in self.data.station_dict[station_id].occupied_drone_set.values():
                        if each_drone.ID == drone.ID:
                            is_in_one_station_occupied_drone_list = True
                            break

                if(is_in_one_station_occupied_drone_list == False):
                    continue

            previous_timestamp_idx = 0
            if (drone.status not in ['idle', 'wait']):
                previous_timestamp_idx = drone.current_trajectory_timestamp_idx
                drone.current_trajectory_timestamp_idx += elapsed_timestamp_cnt

                if (drone.current_trajectory_timestamp_idx > len(drone.trajectory['x_coor_list'])):
                    drone.current_trajectory_timestamp_idx = len(drone.trajectory['x_coor_list']) - 1

            if (drone.status == 'wait'):

                is_started_delivery, current_timestamp_idx = self.check_whether_drone_starts_delivery(drone_id=drone_id)

                if (is_started_delivery == False):
                    continue
                else:
                    drone.current_trajectory_timestamp_idx = current_timestamp_idx

                    order_id = drone.assigned_order_ID

                    self.unfinished_order_set[order_id].status = 4

            traveled_trajectory = {
                'origin': drone.trajectory['origin'],
                'destination': drone.trajectory['destination'],
                'trajectory_time_interval': drone.trajectory['trajectory_time_interval'],
                'route_name': [],
                'x_coor_list': [],
                'y_coor_list': [],
                'date_time_list': [],
                'current_minute_list': [],
                'current_timestamp_idx_list': [],
                'drone_status_list': [],
                'cumulative_dis_list': [],
                'order_name_list': [],
                'drone_name_list': []
            }

            traveled_trajectory['route_name'] = drone.trajectory['route_name'][
                                                previous_timestamp_idx:drone.current_trajectory_timestamp_idx]
            traveled_trajectory['x_coor_list'] = drone.trajectory['x_coor_list'][
                                                 previous_timestamp_idx:drone.current_trajectory_timestamp_idx]
            traveled_trajectory['y_coor_list'] = drone.trajectory['y_coor_list'][
                                                 previous_timestamp_idx:drone.current_trajectory_timestamp_idx]
            traveled_trajectory['date_time_list'] = drone.trajectory['date_time_list'][
                                                    previous_timestamp_idx:drone.current_trajectory_timestamp_idx]
            traveled_trajectory['current_timestamp_idx_list'] = drone.trajectory['current_timestamp_idx_list'][
                                                                previous_timestamp_idx:drone.current_trajectory_timestamp_idx]
            traveled_trajectory['current_minute_list'] = drone.trajectory['current_minute_list'][
                                                         previous_timestamp_idx:drone.current_trajectory_timestamp_idx]
            traveled_trajectory['drone_status_list'] = drone.trajectory['drone_status_list'][
                                                       previous_timestamp_idx:drone.current_trajectory_timestamp_idx]

            traveled_trajectory['cumulative_dis_list'] = drone.trajectory['cumulative_dis_list'][
                                                         previous_timestamp_idx:drone.current_trajectory_timestamp_idx]

            traveled_trajectory['order_name_list'] = drone.trajectory['order_name_list'][
                                                     previous_timestamp_idx:drone.current_trajectory_timestamp_idx]

            traveled_trajectory['drone_name_list'] = drone.trajectory['drone_name_list'][
                                                     previous_timestamp_idx:drone.current_trajectory_timestamp_idx]

            if (drone.finished_trajectory == None):
                drone.finished_trajectory = {
                    'origin': drone.trajectory['origin'],
                    'destination': drone.trajectory['destination'],
                    'trajectory_time_interval': drone.trajectory['trajectory_time_interval'],
                    'route_name': [],
                    'x_coor_list': [],
                    'y_coor_list': [],
                    'date_time_list': [],
                    'current_minute_list': [],
                    'current_timestamp_idx_list': [],
                    'drone_status_list': [],
                    'cumulative_dis_list': [],
                    'order_name_list': [],
                    'drone_name_list': []
                }

            """ 更新 finished轨迹 """
            drone.finished_trajectory['route_name'] += traveled_trajectory['route_name']
            drone.finished_trajectory['x_coor_list'] += traveled_trajectory['x_coor_list']
            drone.finished_trajectory['y_coor_list'] += traveled_trajectory['y_coor_list']
            drone.finished_trajectory['date_time_list'] += traveled_trajectory['date_time_list']
            drone.finished_trajectory['current_minute_list'] += traveled_trajectory['current_minute_list']
            drone.finished_trajectory['current_timestamp_idx_list'] += traveled_trajectory['current_timestamp_idx_list']
            drone.finished_trajectory['drone_status_list'] += traveled_trajectory['drone_status_list']
            drone.finished_trajectory['cumulative_dis_list'] += traveled_trajectory['cumulative_dis_list']
            drone.finished_trajectory['order_name_list'] += traveled_trajectory['order_name_list']
            drone.finished_trajectory['drone_name_list'] += traveled_trajectory['drone_name_list']

            if (drone.current_trajectory_timestamp_idx >= len(drone.trajectory['x_coor_list'])):
                drone.remaining_trajectory = None
            else:

                drone.remaining_trajectory['route_name'] = drone.trajectory['route_name'][
                                                           drone.current_trajectory_timestamp_idx:]
                drone.remaining_trajectory['x_coor_list'] = drone.trajectory['x_coor_list'][
                                                            drone.current_trajectory_timestamp_idx:]
                drone.remaining_trajectory['y_coor_list'] = drone.trajectory['y_coor_list'][
                                                            drone.current_trajectory_timestamp_idx:]
                drone.remaining_trajectory['date_time_list'] = drone.trajectory['date_time_list'][
                                                               drone.current_trajectory_timestamp_idx:]
                drone.remaining_trajectory['current_minute_list'] = drone.trajectory['current_minute_list'][
                                                                    drone.current_trajectory_timestamp_idx:]
                drone.remaining_trajectory['current_timestamp_idx_list'] = drone.trajectory[
                                                                               'current_timestamp_idx_list'][
                                                                           drone.current_trajectory_timestamp_idx:]
                drone.remaining_trajectory['drone_status_list'] = drone.trajectory['drone_status_list'][
                                                                  drone.current_trajectory_timestamp_idx:]
                drone.remaining_trajectory['cumulative_dis_list'] = drone.trajectory['cumulative_dis_list'][
                                                                    drone.current_trajectory_timestamp_idx:]
                drone.remaining_trajectory['order_name_list'] = drone.trajectory['order_name_list'][
                                                                drone.current_trajectory_timestamp_idx:]
                drone.remaining_trajectory['drone_name_list'] = drone.trajectory['drone_name_list'][
                                                                drone.current_trajectory_timestamp_idx:]

            previous_drone_status = drone.status

            if (drone.trajectory == None):
                drone.status = 'idle'

            if (drone.current_trajectory_timestamp_idx + 1 >= len(drone.trajectory['x_coor_list'])):
                drone.status = 'idle'


            else:
                timestamp_idx = drone.current_trajectory_timestamp_idx
                drone.status = drone.trajectory['drone_status_list'][timestamp_idx]

                if (drone.status == 'back' or (previous_drone_status in ['delivery'] and drone.status == 'idle')):
                    order_id = drone.assigned_order_ID
                    station_id = drone.current_station_ID

                    try:
                        self.data.station_dict[station_id].assigned_and_scheduled_order_set[order_id].status = 5    # drone_status: 'finished_delivery'
                        copy_order = copy.deepcopy(
                            self.data.station_dict[station_id].assigned_and_scheduled_order_set[order_id])

                        self.finished_order_set[order_id] = copy_order
                        del self.data.station_dict[station_id].assigned_and_scheduled_order_set[order_id]
                        try:
                            self.data.station_dict[station_id].assigned_but_not_scheduled_order_queue.remove(order_id)
                        except:
                            pass
                        del self.unfinished_order_set[order_id]

                    except:
                        print('*****************************************')
                        print('该订单之前已经被移动到完成的订单中，目前为空')
                        print('*****************************************')

                    drone.org_lng = drone.trajectory['destination'][0]
                    drone.org_lat = drone.trajectory['destination'][1]

                    drone.des_lng = drone.trajectory['origin'][0]
                    drone.des_lat = drone.trajectory['origin'][1]

                else:

                    current_lng = drone.trajectory['x_coor_list'][drone.current_trajectory_timestamp_idx]
                    current_lat = drone.trajectory['y_coor_list'][drone.current_trajectory_timestamp_idx]
                    drone.current_location = [current_lng, current_lat]
                    # self.data.station_dict[station_id].occupied_drone_set[drone.ID] = drone

            if (drone.status == 'idle'):

                station_id = drone.current_station_ID
                try:
                    self.data.station_dict[station_id].idle_drone_set[drone.ID] = drone
                    del self.data.station_dict[station_id].occupied_drone_set[drone.ID]
                except:
                    pass

                # try:
                #     del self.data.station_dict[station_id].occupied_drone_set[drone.ID]
                # except:
                #     pass

                drone.trajectory = None

                drone.reset()

        order_set_copy = copy.deepcopy(self.data.order_set_instance)
        for order_id, order in order_set_copy.items():
            self.assigned_order_set[order_id] = self.data.order_set_instance[order_id]
            del self.data.order_set_instance[order_id]

        print("\n\n\n\n\n**********************************************")
        print(f'删完后data的订单数量：{len(self.data.order_set_instance)}')
        print("**********************************************\n\n\n\n\n")
        self.data.order_set_instance = {}

        self.order_cnt_assigned_but_wait_drone_dispatch = 0
        self.order_cnt_assigned_and_wait_drone_delivery = 0
        self.order_cnt_delivery = 0
        for order_id, order_object in self.unfinished_order_set.items():
            if (order_object.status == 2):
                self.order_cnt_assigned_but_wait_drone_dispatch += 1
            if (order_object.status == 3):
                self.order_cnt_assigned_and_wait_drone_delivery += 1
            if (order_object.status == 4):
                self.order_cnt_delivery += 1

        # """ 更新完成 """
        # print("======================================")
        # print("    系统更新完毕 ！")
        # for station_id in self.data.station_dict.keys():
        #     print(f"停机坪： {station_id}, 订单队列为： {self.data.station_dict[station_id].assigned_but_not_scheduled_order_queue}")
        # print("======================================")

        print("====================================================")
        print(f"     interval:{time_interval_id},  更新后的状态    ")
        for station_id in self.data.station_dict.keys():
            print(f"停机坪 {station_id} ：{self.data.station_dict[station_id].assigned_but_not_scheduled_order_queue}")
        print("====================================================")

        print("====================================================")
        print(f"     interval:{time_interval_id},  储存更新后的停机坪的订单的状态    ")
        for station_id in self.data.station_dict.keys():
            if station_id not in self.data.station_per_interval_data.keys():
                self.data.station_per_interval_data[station_id] = {}

            idle_drone_id_set = {}
            occupied_drone_set = {}
            row_dict = {'start_time': self.system_minute,
                        'end_time': self.system_minute+self.time_interval,
                        'not_scheduled_order_queue': str(self.data.station_dict[station_id].assigned_but_not_scheduled_order_queue),
                        'idle_drone_set': str(list(self.data.station_dict[station_id].idle_drone_set)),
                        'occupied_drone_set': str(list(self.data.station_dict[station_id].occupied_drone_set.keys())),
                        'num_not_scheduled_order_queue': len(self.data.station_dict[station_id].assigned_but_not_scheduled_order_queue),
                        'num_idle_drone_set': len(self.data.station_dict[station_id].idle_drone_set),
                        'num_occupied_drone_set': len(self.data.station_dict[station_id].occupied_drone_set)
                        }
            self.data.station_per_interval_data[station_id][time_interval_id] = row_dict
            print(f"停机坪 {station_id} ：{self.data.station_dict[station_id].assigned_but_not_scheduled_order_queue}")
        print("====================================================")



    def convert_minute_to_datetime(
            self,
            minute=None,
            is_in_file_name=False
    ):

        current_hour_ID = (int)((minute * 60) // (60 * 60))
        current_minute_ID = (int)(((minute * 60) % (60 * 60)) // 60)
        current_second_ID = (int)((((minute * 60) % (60 * 60))) % 60)

        datetime_data = datetime(self.year_ID, self.month_ID, self.day_ID, current_hour_ID, current_minute_ID,
                                 current_second_ID)

        datetime_str = None
        if(is_in_file_name == True):
            datetime_str = datetime_data.strftime("%Y-%m-%d %H_%M_%S")
        else:
            datetime_str = datetime_data.strftime("%Y-%m-%d %H:%M:%S")


        return datetime_str




    def export_results(
            self,
            time_interval_id=None
    ):
        datetime_str = self.convert_minute_to_datetime(
            minute=self.system_minute,
            is_in_file_name=True
        )

        order_file_name = f'shenzhen_timeslot_{time_interval_id}_{datetime_str}_order.csv'
        order_file_path = os.path.join(self.result_file_dir, order_file_name)

        order_data_for_single_interval = {
            'order_ID': [],
            'org_store_id': [],
            'des_home_id': [],
            'org_lng': [],
            'org_lat': [],
            'des_lng': [],
            'des_lat': [],
            'delivery_distance': [],
            'delivery_time': [],
            'start_time': [],
            'start_datetime': [],
            'estimated_arrival_time': [],
            'estimated_arrival_datetime': [],
            'assigned_locker_ID': []
        }
        for order_num_cnt in self.data.order_set_instance.keys():
            order_data_for_single_interval['order_ID'].append(self.data.order_set_instance[order_num_cnt].ID)

            order_data_for_single_interval['org_store_id'].append(self.data.order_set_instance[order_num_cnt].org_store_id)
            order_data_for_single_interval['des_home_id'].append(self.data.order_set_instance[order_num_cnt].des_home_id)

            order_data_for_single_interval['org_lng'].append(self.data.order_set_instance[order_num_cnt].org_lng)
            order_data_for_single_interval['org_lat'].append(self.data.order_set_instance[order_num_cnt].org_lat)

            order_data_for_single_interval['des_lng'].append(self.data.order_set_instance[order_num_cnt].des_lng)
            order_data_for_single_interval['des_lat'].append(self.data.order_set_instance[order_num_cnt].des_lat)

            order_data_for_single_interval['delivery_distance'].append(self.data.order_set_instance[order_num_cnt].delivery_distance)
            order_data_for_single_interval['delivery_time'].append(self.data.order_set_instance[order_num_cnt].delivery_time)

            order_data_for_single_interval['start_time'].append(self.data.order_set_instance[order_num_cnt].start_time)
            start_datetime = self.convert_minute_to_datetime(self.data.order_set_instance[order_num_cnt].start_time)
            order_data_for_single_interval['start_datetime'].append(start_datetime)

            order_data_for_single_interval['estimated_arrival_time'].append(self.data.order_set_instance[order_num_cnt].estimated_arrival_time)
            estimated_arrival_datetime = self.convert_minute_to_datetime(int(self.data.order_set_instance[order_num_cnt].estimated_arrival_time))
            order_data_for_single_interval['estimated_arrival_datetime'].append(estimated_arrival_datetime)

            order_data_for_single_interval['assigned_locker_ID'].append(self.data.order_set_instance[order_num_cnt].assigned_locker_ID)

        order_data_for_single_interval_df = pd.DataFrame(order_data_for_single_interval)
        order_data_for_single_interval_df.to_csv(order_file_path, index=False)

        # trajectory_file_name = 'shenzhen_trajectory_total_timeslot.csv'
        trajectory_file_name = f'shenzhen_timeslot_{time_interval_id}_{datetime_str}_trajectory.csv'
        trajectory_file_path = os.path.join(self.result_file_dir, trajectory_file_name)

        drone_trajectory_data_for_single_interval = {
            'route_name': [],
            'x_coor_list': [],
            'y_coor_list': [],
            'date_time_list': [],
            'current_minute_list': [],
            'current_timestamp_idx_list': [],
            'cumulative_dis_list': [],
            'order_name_list': [],
            'drone_name_list': [],
            'drone_status_list': []
        }

        for station_id in self.data.station_dict.keys():
            for drone_id in self.data.station_dict[station_id].occupied_drone_set.keys():
                for trajectory_id in range(len(self.data.drone_dict[drone_id].trajectory['date_time_list'])):

                    trajectory_date_time = self.data.drone_dict[drone_id].trajectory['date_time_list'][trajectory_id]
                    dt = datetime.strptime(trajectory_date_time, '%Y-%m-%d %H:%M:%S')

                    trajectory_minutes = dt.hour * 60 + dt.minute + dt.second / 60

                    if (
                            trajectory_minutes >= self.system_minute and
                            trajectory_minutes <= self.system_minute + self.time_interval
                    ):
                        route_name = self.data.drone_dict[drone_id].trajectory['route_name'][trajectory_id]
                        drone_trajectory_data_for_single_interval['route_name'].append(route_name)

                        x_coor_list = self.data.drone_dict[drone_id].trajectory['x_coor_list'][trajectory_id]
                        drone_trajectory_data_for_single_interval['x_coor_list'].append(x_coor_list)

                        y_coor_list = self.data.drone_dict[drone_id].trajectory['y_coor_list'][trajectory_id]
                        drone_trajectory_data_for_single_interval['y_coor_list'].append(y_coor_list)

                        date_time_list = self.data.drone_dict[drone_id].trajectory['date_time_list'][trajectory_id]
                        drone_trajectory_data_for_single_interval['date_time_list'].append(date_time_list)

                        current_minute_list = self.data.drone_dict[drone_id].trajectory['current_minute_list'][trajectory_id]
                        drone_trajectory_data_for_single_interval['current_minute_list'].append(current_minute_list)

                        current_timestamp_idx_list = self.data.drone_dict[drone_id].trajectory['current_timestamp_idx_list'][trajectory_id]
                        drone_trajectory_data_for_single_interval['current_timestamp_idx_list'].append(current_timestamp_idx_list)

                        cumulative_dis_list = self.data.drone_dict[drone_id].trajectory['cumulative_dis_list'][trajectory_id]
                        drone_trajectory_data_for_single_interval['cumulative_dis_list'].append(cumulative_dis_list)

                        order_name_list = self.data.drone_dict[drone_id].trajectory['order_name_list'][trajectory_id]
                        drone_trajectory_data_for_single_interval['order_name_list'].append(order_name_list)

                        drone_name_list = self.data.drone_dict[drone_id].trajectory['drone_name_list'][trajectory_id]
                        drone_trajectory_data_for_single_interval['drone_name_list'].append(drone_name_list)

                        drone_status_list = self.data.drone_dict[drone_id].trajectory['drone_status_list'][trajectory_id]
                        drone_trajectory_data_for_single_interval['drone_status_list'].append(drone_status_list)

                drone_trajectory_data_for_single_interval_df = pd.DataFrame(drone_trajectory_data_for_single_interval)
                drone_trajectory_data_for_single_interval_df.to_csv(trajectory_file_path, index=False)

        statistic_file_name = f'shenzhen_timeslot_{time_interval_id}_{datetime_str}_statistic.csv'
        statistic_file_path = os.path.join(self.result_file_dir, statistic_file_name)

        statistic_data_for_single_interval = {
            'Attribute': [
                'time_interval_id',
                'unfinished_order_cnt',
                'wait_drone_dispatch',
                'wait_delivery',
                'delivering',
                'finished_order_cnt',
                'system_minute',
                'CPU_time'
            ],

            'Value': [
                time_interval_id,
                len(self.unfinished_order_set),
                self.order_cnt_assigned_but_wait_drone_dispatch,
                self.order_cnt_assigned_and_wait_drone_delivery,
                self.order_cnt_delivery,
                len(self.finished_order_set),
                self.system_minute,
                self.simulation_CPU_time
            ]
        }

        statistic_data_for_single_interval_df = pd.DataFrame(statistic_data_for_single_interval)
        statistic_data_for_single_interval_df.to_csv(statistic_file_path, index=False)




    def run(self):
        self.simulation_start_time = time.time()
        self.data.order_set_instance = {}
        for time_interval_id in self.time_interval_dict.keys():
            self.simulation_end_time = time.time()
            self.simulation_CPU_time = self.simulation_end_time - self.simulation_start_time
            print("=================================================")
            print("                更新系统                     ")
            print("=================================================")
            self.update_system_status(time_interval_id=time_interval_id)
            self.deploy_Simulation_log(time_interval_id=time_interval_id)

            print("=================================================")
            print("                生成订单                     ")
            print("=================================================")
            self.data.generate_order_of_instance_by_station_and_locker_in_simulation(
                time_interval_id=time_interval_id,
                start_minute=self.time_interval_dict[time_interval_id][0],
                end_minute=self.time_interval_dict[time_interval_id][1],
                time_interval=self.time_interval,
                order_num_lb=self.order_num_lb,
                order_num_ub=self.order_num_ub,
                reachable_home_list_path=self.reachable_home_list_path
            )

            print("=================================================")
            print("                开始建模                     ")
            print("=================================================")
            model_handler = ModelBuilder.ModelBuilder(data=self.data)
            model_handler.build_AP_model_with_cut_simulation(numerical_dict=self.numerical_dict)
            model_handler.get_solution()

            print("=================================================")
            print("                生成轨迹                     ")
            print("=================================================")
            self.generate_trajectory_for_simulation(
                year_ID=self.year_ID,
                month_ID=self.month_ID,
                day_ID=self.day_ID,
                trajectory_time_interval=self.time_interval,
                start_hour=self.start_minute // 60
            )


            print("=================================================")
            print("                导出每个时间窗下的详细文件                     ")
            print("=================================================")

            self.export_results(time_interval_id=time_interval_id)
            print("=================================================")
            print("                导出所有时间窗下的统计文件                   ")
            print("=================================================")
        df_Statistics = pd.DataFrame(columns=['station_id', 'time_interval_id', 'start_time',
                                              'end_time', 'start_time_datetime', 'end_time_datetime',
                                              'not_scheduled_order_queue', 'idle_drone_set',
                                              'occupied_drone_set', 'num_not_scheduled_order_queue',
                                              'num_idle_drone_set', 'num_occupied_drone_set'])
        for station_id in self.data.station_per_interval_data.keys():
            for time_interval_id in self.data.station_per_interval_data[station_id].keys():
                cur_row = self.data.station_per_interval_data[station_id][time_interval_id]
                row_dict = {'station_id': station_id,
                            'time_interval_id': time_interval_id,
                            'start_time_datetime': self.convert_minute_to_datetime(cur_row['start_time']),
                            'end_time_datetime': self.convert_minute_to_datetime(cur_row['end_time']),
                            'start_time': cur_row['start_time'],
                            'end_time': cur_row['end_time'],
                            'not_scheduled_order_queue': cur_row['not_scheduled_order_queue'],
                            'idle_drone_set': cur_row['idle_drone_set'],
                            'occupied_drone_set': cur_row['occupied_drone_set'],
                            'num_not_scheduled_order_queue': cur_row['num_not_scheduled_order_queue'],
                            'num_idle_drone_set': cur_row['num_idle_drone_set'],
                            'num_occupied_drone_set': cur_row['num_occupied_drone_set']
                            }
                df_Statistics = pd.concat([df_Statistics, pd.DataFrame([row_dict])], ignore_index=True)
        df_Statistics.to_csv(self.result_file_dir+'/Simulation_Statistics_Data.csv', index=False)

