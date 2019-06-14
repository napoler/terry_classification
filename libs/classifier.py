
import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM,BertForSequenceClassification

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

class Classifier:
    """预测分类"""
    def __init__(self,model="",num_labels =2):
        print('kaishi')
        self.num_labels =num_labels
        self.tokenizer = BertTokenizer.from_pretrained(model)
        self.model = BertForSequenceClassification.from_pretrained(model,num_labels = num_labels)
        # model = BertForSequenceClassification.from_pretrained('bert-base-chinese')
        # model = BertForNextSentencePrediction.from_pretrained('bert-base-chinese')
        self.model.eval()
        # num_labels =2
    def c(self,text):
        num_labels =self.num_labels
        max_seq_length =20
        # text="天气真是太好了"
        tokens_a = self.tokenizer.tokenize(text)
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        
        # if tokens_b:
        #     tokens += tokens_b + ["[SEP]"]
        #     segment_ids += [1] * (len(tokens_b) + 1)
        
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        
        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        
        labels = self.tokenizer.convert_tokens_to_ids(['0','1'])
        # labels
        
        
        input_ids_tensor = torch.tensor([input_ids])
        input_mask_tensor = torch.tensor([input_mask])
        segment_ids_tensor = torch.tensor([segment_ids])
        labels_tensor = torch.tensor([labels])
        
        device='cpu'
        
        input_ids_tensor = input_ids_tensor.to(device)
        input_mask_tensor = input_mask_tensor.to(device)
        segment_ids_tensor = segment_ids_tensor.to(device)
        labels_tensor = labels_tensor.to(device)
        # print(input_ids_tensor)
        # print(input_mask_tensor)
        # print(segment_ids_tensor)
        # print(labels_tensor)
        with torch.no_grad():
            logits = self.model(input_ids_tensor,segment_ids_tensor,input_mask_tensor )
            print(logits)
            return logits
    def classifier_label(self,logits):
    #     k = logits.view(-1, 2)
        logits.detach().cpu().numpy()
        n = logits.detach().cpu().numpy()
        preds = np.argmax(n, axis=1)
        return preds

    def yuce(self,text):
        
        logits = self.c(text)
        y = self.classifier_label(logits)
        return y
    def yuce_list(self,text_list):
#         text_list=[
#     "其实铲屎官们常常有种错觉，养了喵跟没养差不多，平时基本都很难看到它们，唯一例外的是饭点。",
#     '北交大原校长宁滨遇车祸去世 其座驾变道与旁车接触后失控翻滚',
#     '20楼玻璃窗坠落砸伤6岁男童，涉事租户：不会逃避责任',
#     ' 监管部门：上海已经有99家P2P网贷机构失联 易互贷在列',
#     '喵星人的食物以什么为主才是最好的？没有最好的，只有适合的'
    
# ]

        for item in text_list:
            print(item)
            print(self.yuce(item))
            print("##"*50)