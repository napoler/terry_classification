import libs
model ="/home/terry/pan/github/bert/model/last_xiangguan/"
cf= libs.Classifier(model)



text_list=[
"其实铲屎官们常常有种错觉，养了喵跟没养差不多，平时基本都很难看到它们，唯一例外的是饭点。",
'北交大原校长宁滨遇车祸去世 其座驾变道与旁车接触后失控翻滚',
'20楼玻璃窗坠落砸伤6岁男童，涉事租户：不会逃避责任',
' 监管部门：上海已经有99家P2P网贷机构失联 易互贷在列',
'喵星人的食物以什么为主才是最好的？没有最好的，只有适合的'

]

cf.yuce_list(text_list)