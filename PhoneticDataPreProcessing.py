
# coding: utf-8

# In[617]:


import multiprocessing
import re

import numpy as np
import pandas as pd
# cmudict的entries方法找出所有音素
import nltk
from nltk.corpus import stopwords  # 这个stopwords.words("english")
from nltk.corpus import cmudict
# import scikit-learn里面的两个计算tf-idf必要的类
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# In[618]:


# 用cmudcit的hashmap版本，查询会更快
phonetic_check_dict = cmudict.dict()


# In[619]:


# cmudict.entries是一个list，里面的每个elements是一个tuple，tuple[0]是单词或字母，tuple[1]是对应的音素


# In[620]:


def get_data(filename):
    data_origin = pd.read_csv(filename, encoding='utf-8')
    return data_origin


# In[621]:


data_origin = pd.read_csv("./df.csv")


# In[622]:


def preprocessing_the_sents(sents_string):
    sents_string = re.sub("\?", "", sents_string)
    sents_string = re.sub("\.", "", sents_string)
    sents_string = re.sub("!", "", sents_string)
    sents_string = re.sub(";", "", sents_string)
    sents_string = re.sub(":", "", sents_string)
    sents_string = re.sub("-", " ", sents_string)  # 如果是破折号有必要替换成空格符
    sents_string = re.sub(",", "", sents_string)
    # 如果是外层引号也应该去掉，但是不要去掉
    sents_string = re.sub(r"(?<!.)'(?=.)", "", sents_string)
    sents_string = re.sub(r"(?<=.)'(?!.)", "", sents_string)
    # 双引号，中文引号
    sents_string = re.sub(r'"', '', sents_string)
    sents_string = re.sub(r'‘', '', sents_string)
    # 还有各种奇葩符号
    sents_string = re.sub("\*", "", sents_string)
    sents_string = re.sub("&", "", sents_string)
    sents_string = re.sub("#", "", sents_string)
    # 最后处理掉多余的空格符
    sents_string = re.sub(" +", " ", sents_string)
    # 所有字母小写化，适应cmudict的key的需求
    sents_string = sents_string.lower()  
    # 打散所有字符串，变成list
    sents_string_list = sents_string.split(" ")
    
    return sents_string_list


# In[623]:


# 应用这个函数清洗句子
data_processed_list = data_origin["text"].apply(preprocessing_the_sents)


# In[624]:


# 拼回原表
data_origin = pd.concat([data_origin, data_processed_list], axis = 1)
data_origin.columns = ['key', 'text', 'text_list']


# In[625]:


# test_passed
# 作出两个版本的，一个有strip_stopwords的，一个没有的
def strip_the_stopwords(words_list):
    # 如果不是停止词就允许添加，否则就直接return false
    words_list = list(
        map(lambda x:x if x not in stopwords.words("english") else None, words_list)
    )
    for item_index in range(len(words_list) - 1, -1, -1):  # 倒序遍历，确保不会出现正序删除之后的错误
        if words_list[item_index] == None:
            del words_list[item_index]
    
    return words_list


# In[626]:


# count单词数
def count_words(words_list):
    return len(words_list)


# In[627]:


# 计算每个列表的音素数


# In[628]:


# 返回两个数据，进入一个tuple，这个tuple再做两个apply切分成两列
def count_phonetics(words_list):
    phonetic_length = 0
    exception_list = []
    for words in words_list:
        # 如果不是这个列表里面的，放进exception_list
        try:
            phonetic_check_dict[words]
            phonetic_length += len(phonetic_check_dict[words][0])
        except:
            exception_list.append(words)
    
    return phonetic_length, exception_list


# In[629]:


arrays_with_words_count_SW = data_origin["text_list"].apply(count_words)


# In[630]:


arrays_with_words_count_SW.name = "series_with_words_count_SW"


# In[631]:


data_origin = pd.concat([data_origin, arrays_with_words_count_SW], axis=1)


# In[632]:


arrays_with_count_and_exception = data_origin["text_list"].apply(count_phonetics)


# In[633]:


series_with_phonetic_counts = arrays_with_count_and_exception.apply(lambda x:x[0])


# In[634]:


series_with_phonetic_counts.name = "series_with_phonetic_counts_with_SW"


# In[635]:


series_with_exception_list = arrays_with_count_and_exception.apply(lambda x:x[1])


# In[636]:


series_with_exception_list.name = "series_with_exception_list_with_SW"


# In[637]:


data_origin = pd.concat([data_origin, series_with_phonetic_counts], axis=1)
data_origin = pd.concat([data_origin, series_with_exception_list], axis=1)


# In[638]:


text_list_without_stopwords = data_origin["text_list"].apply(strip_the_stopwords)


# In[639]:


text_list_without_stopwords.name = "text_list_without_stopwords"


# In[640]:


data_origin = pd.concat([data_origin, text_list_without_stopwords], axis = 1)


# In[641]:


arrays_with_words_count_no_SW = data_origin["text_list_without_stopwords"].apply(count_words)


# In[642]:


arrays_with_words_count_no_SW.name = "series_with_words_count_no_SW"


# In[643]:


data_origin = pd.concat([data_origin, arrays_with_words_count_no_SW], axis=1)


# In[644]:


arrays_with_count_and_exception = data_origin["text_list_without_stopwords"].apply(count_phonetics)


# In[645]:


series_with_phonetic_counts_no_SW = arrays_with_count_and_exception.apply(lambda x:x[0])


# In[646]:


series_with_phonetic_counts_no_SW.name = "series_with_phonetic_counts"


# In[647]:


series_with_exception_list_no_SW = arrays_with_count_and_exception.apply(lambda x:x[1])


# In[648]:


series_with_exception_list_no_SW.name = "series_with_exception_list"


# In[649]:


data_origin = pd.concat([data_origin, series_with_phonetic_counts_no_SW], axis=1)
data_origin = pd.concat([data_origin, series_with_exception_list_no_SW], axis=1)


# In[650]:


def get_the_corpus(df_origin):
    # 获取corpus，用处理好的数据重新获取
    corpus_with_SW = [sents for sents in data_origin["text_list"].apply(lambda x:" ".join(x))]
    corpus_without_SW = [sents for sents in data_origin["text_list_without_stopwords"].apply(lambda x:" ".join(x))]
    
    return corpus_with_SW, corpus_without_SW


# In[651]:


# 这个是被多进程双开的函数
def cal_tf_idf(corpus):
    # 向量化，算出这个文本的词的向量表达情况
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    # 这个并不直接计算tf_idf,这个只是显示出高频词汇
    word = vectorizer.get_feature_names()
    
    # 定义transformer
    transformer = TfidfTransformer()
    tf_idf = transformer.fit_transform(X)
    # 然后得到相应的数组，转换成series之后重新拼接
    # 如果想保留信息，要转换成str之后再存储为Series
    array_checked = tf_idf.toarray()
    # 保存原始信息
    check_list = pd.Series(map(str, array_checked))
    
    # 算出max，min和mean
    max_of_texts = pd.Series(map(max, array_checked))
    min_of_texts = pd.Series(map(min, array_checked))
    mean_of_texts = pd.Series(map(np.mean, array_checked))
    
    # 添加一个功能，计算特殊的tf_idf，只要有这个corpus就足够了
    splited_words_length_list = list(map(lambda x:x.split(" "), corpus_full))
    each_sents_words_count = pd.Series(map(lambda x:len(x), splited_words_length_list))
    # 完成了count的series，求出sum，然后series相除，得到需要的值
    tf_idf_sum_Series = pd.Series(map(sum, array_checked))
    # 相除之后得到一个Series，就是需要的Series
    customized_tf_idf_result_series = tf_idf_sum_Series/each_sents_words_count
    
    return max_of_texts, min_of_texts, mean_of_texts, customized_tf_idf_result_series


# In[652]:


def mp_cal_td_idf(target_func, corpus_with_SW, corpus_without_SW):
    # tasks组建的时候，key是进程name，value是进程的args，因为target统一是cal_tf_idf
    tasks = {"td_idf_SW":corpus_with_SW, "td_idf_no_SW":corpus_without_SW}
    # 没有数据分块切割，就是两个进程就足够了
    processes_pool = multiprocessing.Pool(processes=2)
    processes_result = list()
    for task_name, task_args in tasks.items():
        # 动态生成两个任务, 顺序将结果存储，导出，apply_async非阻塞，直接apply是阻塞
        # 传入的两个位置的参数需要是args=(task_args, )以完成需求
        result = processes_pool.apply_async(target_func, args=(task_args, ))
        processes_result.append(result)
    processes_pool.close()  # 关闭进程池，通知不再加入新进程
    processes_pool.join()  # 阻塞主进程等待pool里面所有进程结束返回值
    
    # 对在进程池的两个结果进行返回，一共两个结果，返回的是三个的集合tuple
    td_idf_with_SW = processes_result[0]
    td_idf_no_SW = processes_result[1]
    
    # return的结果需要在外面用get的方法获取数值
    return td_idf_with_SW, td_idf_no_SW


# In[653]:


def get_td_idf(df_origin):
    # 首先获取corpus
    corpus_sets = get_the_corpus(df_origin)
    corpus_SW = corpus_sets[0]
    corpus_no_SW = corpus_sets[1]
    
    # 然后使用多进程产生结果
    # 这两个结果是max, min, mean的集合tuple
    result_set = mp_cal_td_idf(cal_tf_idf, corpus_SW, corpus_no_SW)
    
    return result_set


# In[654]:


# 算出了六列tf_idf相关的数据
# 六列的顺序是：第一个大tuple里面是有SW的版本
# 第二个大tuple里面是没有SW的版本
# 两个tuple里面都是max，min, mean的顺序
# 最后一个是特殊的顺序的Series
tf_idf_set = get_td_idf(data_origin)


# In[667]:


tf_idf_SW = tf_idf_set[0].get()
tf_idf_max_SW = tf_idf_SW[0]
tf_idf_min_SW = tf_idf_SW[1]
tf_idf_mean_SW = tf_idf_SW[2]
tf_idf_customized_SW = tf_idf_SW[3]
tf_idf_no_SW = tf_idf_set[1].get()
tf_idf_max_no_SW = tf_idf_no_SW[0]
tf_idf_min_no_SW = tf_idf_no_SW[1]
tf_idf_mean_no_SW = tf_idf_no_SW[2]
tf_idf_customized_no_SW = tf_idf_no_SW[3]


# In[669]:


tf_idf_max_SW.name = "tf_idf_max_SW"
tf_idf_min_SW.name = "tf_idf_min_SW"
tf_idf_mean_SW.name = "tf_idf_mean_SW"
tf_idf_max_no_SW.name = "tf_idf_max_no_SW"
tf_idf_min_no_SW.name = "tf_idf_min_no_SW"
tf_idf_mean_no_SW.name = "tf_idf_mean_no_SW"
tf_idf_customized_SW.name = "tf_idf_customized_SW"
tf_idf_customized_no_SW.name = "tf_idf_customized_no_SW"


# In[671]:


data_origin = pd.concat([data_origin, tf_idf_max_SW], axis = 1)
data_origin = pd.concat([data_origin, tf_idf_min_SW], axis = 1)
data_origin = pd.concat([data_origin, tf_idf_mean_SW], axis = 1)
data_origin = pd.concat([data_origin, tf_idf_max_no_SW], axis = 1)
data_origin = pd.concat([data_origin, tf_idf_min_no_SW], axis = 1)
data_origin = pd.concat([data_origin, tf_idf_mean_no_SW], axis = 1)
data_origin = pd.concat([data_origin, tf_idf_customized_SW], axis = 1)
data_origin = pd.concat([data_origin, tf_idf_customized_no_SW], axis = 1)


# In[672]:


data_final = data_origin.applymap(lambda x:x if x != [] else None)


# In[674]:


data_final.to_csv("dsr_preprocessed.csv", encoding="utf-8")

