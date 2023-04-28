# Semi-Supervised Multi-Class Node Classification on the Facebook Large Page Network Dataset using GCN

This readme file provides an overview of our project which aims to perform semi-supervised multi-class node classification on the Facebook Large Page Network dataset using the GCN module. The dataset used in this project, which contains 22,470 nodes that represent official Facebook pages, and 171,002 edges that represent mutual likes between pages. Each page has a category that is represented by its node features, which can be classified into one of four classes: "company," "government," "politician," and "tvshow."

##  Libraries
- **Numpy**: This library was used to load and process the data.
- **PyG**: This library provides multiple tools to construct and manipulate graph data.
- **Pytorch**: Pytorch was used to build the GCN module.
- **Sklearn**: This library was used to perform the t-SNE to draw ground truth plot.
- **Matplotlib**: This library was used to visualize the results

##  Algorithm Description
GCN is designed to operate on graph-structured data.[2]Its input and output are all graph structures, which are different from the traditional neural network structure, and its hidden layer is directly activated in the graph structure.
## ![image](https://github.com/zhuanz7001/3710_2/blob/main/GCN.png)
### Calculate formula

$$
\widehat{A}  =\widetilde{D}^{-\frac{1}{2}}\widehat{A} \widetilde{D}^{-\frac{1}{2}}
$$

$$
Z = f(X, A) = Softmax(\widehat{A} ReLU(\widehat{A}XW^{(0)}) W^{(1)})
$$

$X$ is data and $A$ is adjacency matrix of graph.
$W^{(0)}$ is an input-to-hidden weight matrix for a hidden layer and $W^{(1)}$ is a hidden-to-output weight matrix.

### Building module
|Layer           |Shape                          |param                         |
|----------------|-------------------------------|-----------------------------|
|Dropout         | (128, 64)          |0.5      |
|GCNConv         |(128, 64)            |        |
|Relu         |      (128, 64)    |          |
|Dropout         |     (128, 64)      |0.5      |
|GCNConv         |(64,4)            |          |

**Loss function** : cross_entropy
**Optimizer** : Adam
### Training Data

Train set, Validation set, Test set ratio is  **0.8, 0.1, 0.1** and **0.2, 0.4, 0.4**. 
In this project, we trained the entire graph structure using the GCN module. For the training set, the labels were masked using a mask. This approach achieved excellent results because the model was able to master the structure of the whole graph and compute the labels of the mask nodes with ease. However, this way of splitting the training set may not be optimal, and it may be necessary to reduce the number of nodes in the training set to increase the robustness of the model.

Some people split the dataset by destroying part of the graph structure, but this method can lead to missing information about the nodes. Therefore, we chose to use the mask method to split the dataset for our project.
### Training loss
I tested two different data set segmentation ratios: (0.8, 0.1, 0.1) and (0.2, 0.4, 0.4). Both of these ratios performed well, with accuracy rates of over 90%. We can find that GCN can be trained on a graphical network even with only a small number of node labels, and still produce valid results.
##### Train set, Validation set, Test set ratio is  **0.8, 0.1, 0.1**.

## ![image](https://github.com/zhuanz7001/3710_2/blob/main/ACC0.8.png)
##### Train set, Validation set, Test set ratio is  **0.2, 0.4, 0.4**. 

## ![image](https://github.com/zhuanz7001/3710_2/blob/main/ACC0.2.png)
### Test accuracy and val accuracy plot
##### Train set, Validation set, Test set ratio is  **0.8, 0.1, 0.1**.

## ![image](https://github.com/zhuanz7001/3710_2/blob/main/loss0.8.png)
##### Train set, Validation set, Test set ratio is  **0.2, 0.4, 0.4**. 

## ![image](https://github.com/zhuanz7001/3710_2/blob/main/loss0.2.png)
### T-SNE Plot
## ![image](https://github.com/zhuanz7001/3710_2/blob/main/TSNE.png)
red dots represented 'tvshow', green dots represented 'company', blue dots represented 'government', and yellow dots represented 'politician'. As we can see from the figure, some of the green dots and red dots overlap, which may indicate that nodes of 'company' and 'tvshow' are difficult to distinguish from each other.
## Reference
[1][SNAP: Network datasets: Wikipedia Article Networks (stanford.edu)](https://snap.stanford.edu/data/facebook-large-page-page-network.html)

[2]https://arxiv.org/pdf/1609.02907.pdf

