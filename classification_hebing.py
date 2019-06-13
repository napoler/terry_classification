# from libs import TerrySearch
# from .bert_run as run_cmd
# import bert_run
import configparser
import json,time,re
import Terry_toolkit as tkit
from tqdm import tqdm
#宠物数据
corpus_path_1 = '/home/terry/pan/github/ai_writer/ai_writer/data/kw2text_mini/' #
#其他数据
corpus_path_2 = '/home/terry/github/ai_writer/ai_writer/data/kw2text_other/'
last = '../data/corpus_classification.json'
# item = 'data/article_30e0446c6d5915a10c4a939d70dff353.txt'

tfile=tkit.File()
ttext=tkit.Text()

#打开文件
def openf(file):
    tfile=tkit.File()
    try:

        text = tfile.open_file(file)
        return  text
    except:
        pass


def text2sentences(paragraph):
    """分句函数
    """
    # pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）'
    # pattern ='(。|！|\!|\.|？|\?|\,|，|\;|；|\:|)'
    # pattern ='(。|！|\!|\.|？|\?|\;|；)'
    pattern ='(。|！|\!|？|\?|\;|；)'
    sentences = re.split(pattern,paragraph)         # 保留分割符
    

    new_sents = []
    for i in range(int(len(sentences)/2)):
        sent = sentences[2*i] + sentences[2*i+1]
        new_sents.append(sent)
    return new_sents

# new_sents = text2sentences(paragraph)

def c_inputfile(inputfile,data):
    """创建预测文件

    """
    with open(inputfile,"w") as f:
        json.dump(data,f)
        print("创建训练资料完成..."+inputfile)
        return True

# text= ''
def creat_corpus(corpus_path,lei):
    juzis_list =[]
    for item in tqdm(tfile.file_List(corpus_path,'txt')):
        paragraph= openf(item)
        # 去除回车
        # paragraph= paragraph.strip('\n') 
        paragraph = tfile.clear(paragraph)
        new_sents = text2sentences(paragraph)
        if len(new_sents)>1:
            try:
                juzis = ''
                for j in new_sents:
                    # juzis = str(lei)+"\t"+j +'\n'
                    juzis={
                        'label':'1',
                        'sentence':j

                    }

                    

                juzis_list.append(juzis)

                # text = text +'\n\n'+ '\n'.join(new_sents)
            except:
                pass
        
    # text = ''.join(juzis_list)
    return juzis_list


# 开始运行
lei='1'
text1= creat_corpus(corpus_path_1,lei)
lei='0'
text2= creat_corpus(corpus_path_2,lei)
text = text1+text2


c_inputfile(last,text)
# my_open = open(last, 'a')
# my_open.write(text)
# my_open.close()



