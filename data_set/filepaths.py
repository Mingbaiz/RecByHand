import os

ROOT = os.path.split(os.path.realpath(__file__))[0]

# class FB15K_237():
#     # 下载地址: https://www.microsoft.com/en-us/download/details.aspx?id=52312
#     __BASE = os.path.join(ROOT, './FB15K-237')
#     O_TRAIN=os.path.join(__BASE,'train.txt')
#     O_VALID=os.path.join(__BASE,'valid.txt')
#     O_TEST=os.path.join(__BASE,'test.txt')
#
#     EID2INDEX=os.path.join(__BASE,'eid2index.tsv')
#     RID2INDEX=os.path.join(__BASE,'rid2index.tsv')
#
#     TRAIN=os.path.join(__BASE,'train.tsv')
#     VALID=os.path.join(__BASE,'valid.tsv')
#     TEST=os.path.join(__BASE,'test.tsv')

# class Kg_Co_occurrenceMatrix():
#     BASE = os.path.join(ROOT, 'kg_Co_occurrenceMatrix')
#
# class CRITEO():
#     BASE = os.path.join(ROOT,'criteo')
#     SAMPLE = os.path.join(BASE,'criteo_sample.txt')
#

class Ml_100K():
    #下载地址：https://github.com/rexrex9/kb4recMovielensDataProcess
    __BASE = os.path.join(ROOT, 'ml-100k')
    ORGINAL_DIR = os.path.join(ROOT,'ml-100k-orginal')
    USER_DF = os.path.join(ORGINAL_DIR,'user_df.csv')
    ITEM_DF = os.path.join(ORGINAL_DIR,'item_df.csv')
    ITEM_DF_0 = os.path.join(ORGINAL_DIR, 'item_df_0.csv')


    KG=os.path.join(__BASE,'kg_index.tsv')
    RATING = os.path.join(__BASE,'rating_index.tsv')
    RATING5 = os.path.join(__BASE, 'rating_index_5.tsv')

class Ml_latest_small():
    __BASE = os.path.join(ROOT,'ml-latest-small')
    RATING_TS = os.path.join(__BASE,'rating_index_ts.tsv')
    SEQS = os.path.join(__BASE, 'seqs.npy')

    SEQS_NEG = os.path.join(__BASE, 'seqsWithNeg.npy')


# class Ml_1M():
#     #下载地址：https://github.com/rexrex9/kb4recMovielensDataProcess
#     __BASE = os.path.join(ROOT, 'ml-1m')
#     KG=os.path.join(__BASE,'kg_index.tsv')
#     RATING = os.path.join(__BASE,'rating_index.tsv')
#     RATING5 = os.path.join(__BASE, 'rating_index_5.tsv')
#

#
# class Model():
#     __BASE = os.path.join(ROOT, 'model')
#     C5_RIPPLE_NET = os.path.join(__BASE,'c5_ripple_net.model')
#
# class WIKI_VOTE():
#     __BASE = os.path.join(ROOT, 'wiki-vote')
#     WIKI_VOTE = os.path.join(__BASE,'Wiki-Vote.txt')