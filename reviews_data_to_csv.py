'''
This macro reads IMDB movie reviews into a Pandas dataframe and writes the dataframe
as .csv file for further processing.

The source data needed for this macro was downloaded from

http://ai.stanford.edu/~amaas/data/sentiment/

See:
Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher
Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).

Download and extract the data into the directory containing this macro
before running this macro.
'''

# import libraries
import pyprind
import pandas as pd
import os

# change 'basepath' to the directory containing the unzipped movie dataset
basepath = 'aclImdb'

# define labels 'pos' and 'neg' as numeric values 0 and 1
labels = {'pos': 1, 'neg': 0}

# create progress bar
pbar = pyprind.ProgBar(50000)

# create empty dataframe to fill
df = pd.DataFrame()

# loop over all input files to add to dataframe
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)
        for file in os.listdir(path):
            with open(os.path.join(path, file),
                      'r', encoding='utf-8') as infile:
                txt = infile.read()
                df = df.append([[txt, labels[l]]],
                               ignore_index=True)
                pbar.update()

# rename dataframe columns
df.columns = ['review', 'sentiment']

# more libraries
import numpy as np

# randomize order of entries in dataframe
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))

# save dataframe as csv file
df.to_csv('movie_reviews_labeled.csv', index=False, encoding='utf-8')
