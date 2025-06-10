

# 定义商务住宅类
class Home:

    """
        self.ID         # 商务住宅ID
        self.lng        # 商务住宅地址(经度)
        self.lat        # 商务住宅地址(纬度)
        self.name       # 商务住宅名
    """

    def __init__(self):
        self.ID = None
        self.lng = None
        self.lat = None
        self.name = None


    def __str__(self):
        return (
            f"Home(ID: {self.ID},"
            f"lng: {self.lng},"
            f"lat: {self.lat},"
            f"name: {self.name})"
        )
