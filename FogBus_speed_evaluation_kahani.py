# # # # # # # # # # # # # # # # #
# By: Ali Gandomi               #
# Created at: 1400/02/26        #
# # # # # # # # # # # # # # # # #

import gensim.downloader
import numpy as np
import pandas as pd
import pickle
import requests
from prettytable import PrettyTable
from sklearn.metrics import precision_recall_fscore_support as score
import time
import grequests

decoding = ['building_collapse', 'crash', 'earthquake', 'explosion', 'flood', 'haze', 'meteor', 'shoot', 'typhoon',
            'wildfire']


test_dataset = pd.read_csv('test.csv', encoding='utf-8')
X_test = test_dataset.iloc[:, 13]
y_test = test_dataset.iloc[:, 10]
y_test = y_test.values.ravel()

predicted = np.empty(shape=len(y_test))
urls = ['172.21.51.166', '172.21.50.44', '172.21.50.85', '172.21.50.49', '172.21.50.98']
elapsed_times = np.zeros(1000)
cnt_url = 0
all_requests = []
for index, tweet in enumerate(X_test):
    url = 'http://' + urls[cnt_url] + '/Healthfog/myThesis.php'
    data = {'data': tweet}
    all_requests += [grequests.post(url, data=data)]
    print(str(index), '[' + urls[cnt_url] + '] :', tweet)
    cnt_url = (cnt_url + 1) % 5

    if index == 999:
        break

start_time = time.time()
grequests.map(all_requests)
elapsed_times = (time.time() - start_time)
print(elapsed_times, 'seconds')
