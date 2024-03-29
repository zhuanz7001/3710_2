# -*- coding: utf-8 -*-
"""data.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1n2PIVP_9FYy137EVBsWXWlG0AetAFnMJ
"""

import dataset
train_mask, val_mask, test_mask, data = dataset.get_data()

gcn_m = torch.load('model.pt')
out = gcn_m(data)
test_acc = int((out[test_mask].max(1)[1] == data.y[test_mask]).sum())/int(test_mask.sum())

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
embeddings = model(data).detach().numpy()
embeddings_2d = tsne.fit_transform(embeddings)

import matplotlib.pyplot as plt
colors = ['r', 'g', 'b','y'] # Use different colors for different classes
plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1],s=20,alpha = 0.05,c=[colors[label] for label in data.y])
plt.show()

