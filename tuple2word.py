# # # # # # # # # # # # # # # # #
# By: Ali Gandomi               #
# Created at: 1399/10/20        #
# # # # # # # # # # # # # # # # #

import pandas as pd
from termcolor import colored

allDatasets = 'Datasets/CrisisLexT26/'
event_vector = pd.read_csv(allDatasets + 'TF-IDF.csv', encoding='utf-8')
event_vector = event_vector.loc[:29, :]

for event in list(event_vector):
    for idx, event_word in enumerate(event_vector[event]):
        event_vector[event][idx] = eval(event_word)[0]

event_vector.to_csv(allDatasets + 'TF-IDF_Top20.csv', index=False, encoding='utf-8')
print(colored('Done!', 'green'))
