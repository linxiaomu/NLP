import jieba

# 1.精确分词模式
content = '工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作'
jieba.cut(content,cut_all=False)
result1 = jieba.lcut(content,cut_all=False)
print(result1)

# 2.全模式分词
result2 = jieba.lcut(content,cut_all=True)
print(result2)

# 3.搜索引擎模式分词
result3 = jieba.lcut_for_search(content)
print(result3)

# 使用用户自定义的词典：
#       