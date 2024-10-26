# math_CB.py
import random
import pandas as pd
import math
#from nltk.tokenize import word_tokenize
#from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
#from nltk.stem.lancaster import LancasterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from operator import itemgetter
import itertools
import time
import numpy as np
import json
#导入数据训练数据和测试数据
def load_trainset():
    trainpaper_df = pd.read_csv("train.csv", low_memory=False)
    for i, row in trainpaper_df.iterrows():
        yield row  #yield函数可以减少内存消耗，不知道好用不，试试看
def load_testset():
    testpaper_df = pd.read_csv("test.csv", low_memory=False)
    for i,row in testpaper_df.iterrows():
        yield row


def get_train_test_set():
    global trainSet, testSet, trainSet_unit_abstract, trainSet_author_abstract
    trainSet_len = 0
    testSet_len = 0
    trainSet = {}
    testSet = {}
    trainSet_unit_paperID_abstract = dict()
    trainSet_author_paperID_abstract = dict()
    trainSet_unit_abstract = dict()
    trainSet_author_abstract = dict()
    #得到作者-单位字典
    for row in load_trainset():
        author_id, unit_id, abstract = row['author_id'],row["unit_id"],row["abstract"]
        trainSet.setdefault(author_id, {})
        trainSet[author_id][unit_id] = abstract
        trainSet_len += 1
    #找到每个作者发表的所有文章
    for row in load_trainset():
        author_id, paper_id, abstract = row['author_id'], row['paper_id'], row['abstract']
        trainSet_author_paperID_abstract.setdefault(author_id, {})
        trainSet_author_paperID_abstract[author_id][paper_id] = abstract
    for a, pa in trainSet_author_paperID_abstract.items():
        abstract_list1 = []
        for p,ab in pa.items():
            abstract_list1.append(ab)
        trainSet_author_abstract[a] = abstract_list1
    # 将每个单位发表的所有（不重复）论文摘要找到并放入到trainSet_unit字典中
    for row in load_trainset():
        unit_id, paper_id, abstract = row['unit_id'], row['paper_id'], row['abstract']
        trainSet_unit_paperID_abstract.setdefault(unit_id, {})
        trainSet_unit_paperID_abstract[unit_id][paper_id] = abstract
    for u, pa in trainSet_unit_paperID_abstract.items():
        abstract_list2 = []
        for paperid, ab in pa.items():
            abstract_list2.append(ab)
        trainSet_unit_abstract[u] = abstract_list2
    ##构建测试集数据，主要使用作者——单位数据
    for row in load_testset():
        author_id, unit_id, abstract = row['author_id'], row["unit_id"], row["abstract"]
        testSet.setdefault(author_id, {})
        testSet[author_id][unit_id] = abstract
        testSet_len += 1
    print('划分训练集与测试集成功！')
    print('训练集长 = %s' % trainSet_len)
    print('测试集长 = %s' % testSet_len)
def cal_cosine_similarity(abstract1,abstract2):
    corpus = [str(abstract1),str(abstract2)]
    tfidf = TfidfVectorizer(decode_error='ignore',lowercase=True,stop_words='english',min_df=1)
    tfidf_matrix = tfidf.fit_transform(corpus)
    similarity = linear_kernel(tfidf_matrix)
    return similarity[0][1]

def cal_author_unit_similarity():
    author_unit_simmatrix = {}
    for authorid,author_abstract in trainSet_author_abstract.items():
        for unitid, unit_abstract in trainSet_unit_abstract.items():
            author_unit_simmatrix.setdefault(authorid, {})
            author_unit_simmatrix[authorid].setdefault(unitid, )
            sim = cal_cosine_similarity(trainSet_author_abstract[authorid], trainSet_unit_abstract[unitid])
            author_unit_simmatrix[authorid][unitid] = sim
            print(author_unit_simmatrix[authorid][unitid])
    return author_unit_simmatrix

def recommend(user,author_sim_matrix,n_sim_unit,n_rec_unit):
    global trainSet, testSet, trainSet_unit_abstract, trainSet_author_abstract
    K = int(n_sim_unit)  # 用于推荐参考的单位数K
    N = int(n_rec_unit)  # 推荐单位数量
    rank = {}  # 推荐度字典
    watched_units = trainSet[user]  # 获取作者曾经的发表单位
    #authorids = trainSet_author_abstract.keys() #获取作者id
    for units, _ in watched_units.items():
        # 对目标作者每一篇论文所在单位，从相似作者单位矩阵中取与这个作者单位关联值最大的前K个单位，若这K个单位作者之前没有去过，则把它加入rank字典中，其键为unit_id名，其值（即推荐度）为w（相似单位矩阵的值）
        for related_unit, s in sorted(author_sim_matrix[user].items(), key=itemgetter(1), reverse=True)[:K]:
            if related_unit in watched_units:
                continue  # 如果与user发表的论文所在单位重复了，则直接跳过
            rank.setdefault(related_unit, 0)
                # 计算推荐度
            rank[related_unit] = s
    return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]  ## 根据被推荐单位的相似度逆序排列，然后推荐前N个单位给到用户（作者）
def precommend(rec_m):
    df = pd.read_csv("D:/论文工作/job recommendation/recommendation task1/data/math_data/数学数据第二步.csv", low_memory = False)
    print ('=='*40)
    for unid,w in rec_m:#展示推荐单位名称以及推荐度
        print('单位名称:',df.loc[df['unit_id']==int(unid),'unit'].values[0],'推荐度:',w)
        print ('--'*40)
#产生推荐并通过精确率、召回率和命中率率进行评估
def evaluate(sim_matrix,n_rec_unit,n_sim_unit):
    global trainSet, testSet, trainSet_unit_abstract, trainSet_author_abstract
    print('评估并推荐-----')
    N = int(n_rec_unit)
    hit = 0
    mrr_hit = 0
    MRR = 0
    ndgg_i = 0
    rec_count = 0
    test_count = 0
    hr_count = len(testSet.keys())
    all_rec_units = set()  # 推荐单位集合
    for user, m in list(testSet.items()):  ## 先获取user的喜爱物品列表（作者发表论文所在单位列表）
        test_units = testSet.get(user, {})
        rec_units = recommend(user, sim_matrix, n_sim_unit, n_rec_unit)
        print("作者 %s 的单位推荐列表为：" % user)
        precommend(rec_units)
        # 注意，这里的w与上面recommend的w不要一样，上面的w是指计算出的相似单位矩阵的权值，而这里是这推荐字典rank对应的推荐度
        for i in range(len(rec_units)):
            if rec_units[i][0] in test_units:
                mrr_hit += 1
                MRR += 1 / (i + 1)
                ndgg_i += 1 / (math.log2(1 + i + 1))
                break
        for u, w in rec_units:
            if u in test_units:
                hit += 1
            all_rec_units.add(u)
        rec_count += N
        test_count += len(test_units)
    precision = hit / (1.0 * rec_count)  # 精确率
    recall = hit / (1.0 * test_count)  # 召回率
    hr = hit / (1.0 * hr_count)
    MRR = MRR / hr_count
    NDGG = ndgg_i / hr_count
    f1 = (2 * precision * recall) / (recall + precision)
    # coverage = len(all_rec_units) / (1.0 * unit_count)  # 覆盖率
    print('--' * 40)
    print('MRR=%.4f\t命中率=%.4f\t精确率=%.4f\t召回率=%.4f\tNDGG=%.4f\tf1=%.4f' % (MRR, hr, precision, recall, NDGG, f1))
    print('--' * 40)


start = time.perf_counter()
get_train_test_set()
n_sim_unit = 60#相似单位数量
n_rec_unit = 50#推荐单位数量
#anuthor_num =2000  #参与评估的作者数量
author_unit_sim_matrix = np.load('D:/论文工作/job recommendation/recommendation task1/data/math_data/math_author-unit_simmatrix.npy', allow_pickle=True).item()
#print(author_unit_sim_matrix)
evaluate(author_unit_sim_matrix,n_rec_unit,n_sim_unit)#评估
end = time.perf_counter()
print('结束！')
print('Running time: %s Seconds'%(end-start))