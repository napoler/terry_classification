
import numpy as np # linear algebra


# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
import Terry_toolkit as tkit
logging.basicConfig(level=logging.INFO)
from collections import Counter

 

import torch
from transformers import *


# '/home/terry/pan/github/terry_classification/terry_classification/
class Classifier:
    """
    加载模型　进行预测
    """
    def __init__(self,model_path='data/terry_output/'):
        self.model_path = model_path
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.data={}
    def new():
        self.data = {}
    def get_label(self,sequence):
        input_ids = torch.tensor(self.tokenizer.encode(sequence)).unsqueeze(0)  # Batch size 1
        outputs = self.model(input_ids)
        # print(seq_relationship_scores)
        self.seq_relationship_scores = outputs[0] #对应的概率信息
        return torch.argmax(self.seq_relationship_scores).item()

    # def prediction(self,text):
        
    #     logits = self.c(text)
    #     y = self.classifier_label(logits)
    #     return y
    def prediction_list(self,text_list):
#         text_list=[
#     "其实铲屎官们常常有种错觉，养了喵跟没养差不多，平时基本都很难看到它们，唯一例外的是饭点。",
#     '北交大原校长宁滨遇车祸去世 其座驾变道与旁车接触后失控翻滚',
#     '20楼玻璃窗坠落砸伤6岁男童，涉事租户：不会逃避责任',
#     ' 监管部门：上海已经有99家P2P网贷机构失联 易互贷在列',
#     '喵星人的食物以什么为主才是最好的？没有最好的，只有适合的'
    
# ]
        data=[]
        self.data['items']=[]
        for item in text_list:

            pred= self.get_label(item)
            #这里转化成为十分制
            rank=10-int(pred)
            data.append(rank)
            one={
                'text':item,'rank':rank
            }
            self.data['items'].append(one)
        return data
    def article_prediction(self,article):
        tx = tkit.Text()
        text_list= tx.sentence_segmentation(article)
        return self.prediction_list(text_list)

    def proportion(self,full,element):
        """
        计算元素所占比例"""
        # full.self.article_prediction(article)
        return full.count(element)/len(full)

    def proportion_article(self,article,element):
        """
        预测文章中某个分类所占比例"""
        full =self.article_prediction(article)
        
        self.top_three(full)
        return self.proportion(full,element)
    def proportion_article_auto(self,article):
        """
        预测文章各种类别所占比重"""
        full =self.article_prediction(article)
        self.article_rank=full
        l = self.top_three(full)
        data = []
        for label,num in l:
            
            it ={
                'label':label,
                'proportion':self.proportion(full,label)

            }
            data.append(it)
        self.data['proportion'] = data
        return data
    def top_three(self,full):
        """计算数组中出现次数最多的元素"""
        lab_counts = Counter(full)
        # 出现频率最高的3个单词
        top_three = lab_counts.most_common(3)
        self.data['top_three']=top_three
        return top_three

