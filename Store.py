


# 定义外卖商户类
class Store:

    """
        self.ID                         # 商户ID
        self.name                       # 商户名
        self.month_sale                 # 商户月销量
        self.lat                        # 商户地址(纬度)
        self.lng                        # 商户地址(经度)
        self.reachable_home_list        # 商户可配送商务住宅列表 [1, 4, 5, 7, 9, ...]
    """

    def __init__(self):
        self.ID = 0
        self.name = None
        self.month_sale = 0
        self.lat = None
        self.lng = None
        self.reachable_home_list = []

    def __str__(self):
        return (
            f"Store("
            f"ID: {self.ID},"
            f"name: {self.name},"
            f"month_sale: {self.month_sale},"
            f"lat: {self.lat}, lng: {self.lng})"
        )