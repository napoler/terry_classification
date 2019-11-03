import Terry_toolkit as tkit
import os
from  tqdm import tqdm
import sqlite3
DB='data/baidu.db'

class BaiduDb:
    def __init__(self):
        self.DB ='baidu.db'
    def connect(self):
        """
        连接数据库
        """
        self.conn = sqlite3.connect(DB)
        self.connect = self.conn.cursor()
        # self.connect =
    def close(self):
        """
        关闭数据库
        """
        self.conn.close()
    def get_keywords(self):
        """
        随机获取一千个关键词
        """
        sql="SELECT * FROM keywords ORDER BY RANDOM() limit 1000"
        self.connect.execute(sql)
        return self.connect.fetchall()

    def get_nodes(self):
        """
        随机获取一千个关键词
        """
        sql="SELECT * FROM nodes ORDER BY id"
        self.connect.execute(sql)
        return self.connect.fetchall()


path='data/'

baiduDb = BaiduDb()
baiduDb.connect()

nodes=baiduDb.get_nodes()
print(nodes[:10])


nodes=nodes[:1000]
data=[]
for node in tqdm(nodes):
    item ={
        "label":node[2],
        'sentence':node[1]
    }
    data.append(item)


tkit.File().mkdir(path)

# len(data)
train_data=data[:int(len(data)*0.8)]
dev_data=data[int(len(data)*0.8):]
#创建train数据集
train_path=path+"train.json"
tjson=tkit.Json(file_path=train_path)
tjson.save(train_data)

#创建dev数据集
dev_path=path+"dev.json"
tjson=tkit.Json(file_path=dev_path)
tjson.save(dev_data)
# return   tjson.load()