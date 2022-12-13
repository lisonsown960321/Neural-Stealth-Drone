"""


Breast Cancer
================



This example generates a Mapper built from the `Wisconsin Breast Cancer Dataset <https://www.kaggle.com/uciml/breast-cancer-wisconsin-data>`_.

`Visualization of the breast cancer mapper <../../_static/breast-cancer.html>`_


The reasoning behind the choice of lenses in the demonstration above is:

- **For lens1:** Lenses that make biological sense; in other words, lenses that highlight special features in the data, that I know about. 
- **For lens2:** Lenses that disperse the data, as opposed to clustering many points together.

In the case of this particualr data, using an anomaly score (in this case calculated using the IsolationForest from sklearn) makes biological sense since cancer cells are anomalous. For the second lens, we use the :math:`l^2` norm.

For an interactive exploration of lens for the breast cancer, see the `Choosing a lens notebook <../../notebooks/Cancer-demo.html>`_.



.. image:: ../../../examples/images/breast-cancer.png


"""

import sys
try:
    import pandas as pd
except ImportError as e:
    print("pandas is required for this example. Please install with `pip install pandas` and then try again.")
    sys.exit()

import numpy as np
import kmapper as km
import sklearn
from sklearn import ensemble
from kmapper.plotlyviz import *
import csv

# For data we use the Wisconsin Breast Cancer Dataset
# Via:

path_real='/Users/lichen/Desktop/test/WGAN_generated/real_feature_maps.csv'
path_design='/Users/lichen/Desktop/test/WGAN_generated/fake_feature_maps.csv'
# path_design='/Users/lichen/Desktop/test/WGAN_generated/design_feature_maps.csv'
all_fm='/Users/lichen/Desktop/test/WGAN_generated/all_feature_maps.csv'
tem_list = []
# for i in [path_fake,path_real,path_design]:

import csv

for i in [path_real,path_design]:
    csv_reader = csv.reader(open(i))
    for line in csv_reader:
        tem_list.append(line)

with open(all_fm,'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(tem_list)

df = pd.read_csv(all_fm,header = None,sep=',',low_memory=False)


X = np.array(df.iloc[:,:-1])  # quick and dirty imputation
y = np.array(df.iloc[:,-1])

# We create a custom 1-D lens with Isolation Forest
model = ensemble.IsolationForest(random_state=1729)
model.fit(X)
lens1 = model.decision_function(X).reshape((X.shape[0], 1))

# We create another 1-D lens with L2-norm
mapper = km.KeplerMapper(verbose=3)
lens2 = mapper.fit_transform(X, projection="l2norm")

# Combine both lenses to create a 2-D [Isolation Forest, L^2-Norm] lens
lens = np.c_[lens1, lens2]

# Create the simplicial complex
scomplex = mapper.map(lens,
                   X,
                   cover=km.Cover(n_cubes=15, perc_overlap=0.05),#15,0.2
                   clusterer=sklearn.cluster.KMeans(n_clusters=2,
                                                    random_state=1618033))

# Visualization
mapper.visualize(scomplex,
                 path_html="output/drones.html",
                 title="Wisconsin Breast Cancer Dataset",
                 custom_tooltips=y)

kmgraph,mapper_summary,_ = get_mapper_graph(scomplex)
# ------------------------------ design drone ------------------------------
# node_list_sum = {}
# for j,node in enumerate(kmgraph['nodes']):
#
#     node['custom_tooltips'] = y[scomplex['nodes'][node['name']]]
#     kk = np.unique(node['custom_tooltips'],return_counts=True)
#     dis = node['distribution']
#     if kk[0][-1].startswith('design'):
#         tem = kk[0].copy()
#         for z in tem:
#             if z.startswith('design'):
#                 np.delete(tem,-1)
#             else:pass
#         if len(tem) >= 7:
#             pass
#         else:
#             for id in kk[0]:
#                 temm = list(kk[0].copy())
#                 if id not in node_list_sum:
#                     node_list_sum[id] = kk[1][temm.index(id)]
#                 else:
#                     node_list_sum[id]+=kk[1][temm.index(id)]
#     else:pass
# zs = sorted(node_list_sum.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)
# for i in zs:
#     print(i)

# ------------------------------ 14 drone ------------------------------

node_list_sum = {}
for j,node in enumerate(kmgraph['nodes']):

    node['custom_tooltips'] = y[scomplex['nodes'][node['name']]]
    kk = np.unique(node['custom_tooltips'],return_counts=True)
    dis = node['distribution']

    if kk[0][0] == '1.11':
        fake = kk[1][0]
        all = kk[1].sum()
        if fake <= 2:
            temm = list(kk[0].copy())
            for id in kk[0]:
                if id not in node_list_sum:
                        node_list_sum[id] = kk[1][temm.index(id)]
                else:
                    node_list_sum[id]+=kk[1][temm.index(id)]
        else:pass
    else:
        temm = list(kk[0].copy())
        for id in kk[0]:
            if id not in node_list_sum:
                node_list_sum[id] = kk[1][temm.index(id)]
            else:
                node_list_sum[id]+=kk[1][temm.index(id)]

    #     if len(tem) >= 7:
    #         pass
    #     else:
    #         for id in kk[0]:
    #             temm = list(kk[0].copy())
    #
    #             if id not in node_list_sum:
    #                 node_list_sum[id] = kk[1][temm.index(id)]
    #             else:
    #                 node_list_sum[id]+=kk[1][temm.index(id)]

zs = sorted(node_list_sum.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)
for i in zs:
    print(i)

import matplotlib.pyplot as plt
km.draw_matplotlib(scomplex)
plt.show()