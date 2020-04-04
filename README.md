# Exploratory document analysis with TF-IDF, PCA and t-SNE

I analyzed the papers in my Mendeley folder with these scripts. They are 
based on Andrej Karparthy's arxiv sanity preserver:

https://github.com/karpathy/arxiv-sanity-preserver

## Instructions

Mostly follow the readme in Karpathy's repo. I removed the functionalities 
related to serving a webpage and similarity search with SVM.

## What I found

The training data is not available in this repo, only the trained model. With
the dimensionality reduction techniques I found that my papers more or less
fall into to major categories: quantum physics and machine learning. Between
these two classes, there are roughly 20 papers, which can be considered
interdisciplinary. These are the one which show up both for the "machine learning"
and "quantum" vocabulary terms. 

Since the papers do not have labeling on their own, for the visualization I
hand-crafted labels based on the frequency of a given term. If the transformed
frequency is above a manually-set threshold, I count the paper to be in the
class corresponding to the term (1-gram or 2-gram).


## Examples

![quantum](plots/quantum.png)

![machine learning](plots/machine learning.png)

Interestingly, noise and physics is everywhere:

![Alt Text](plots/noise.png)

![Alt Text](plots/physics.png)

Josephson, qubit and microwave largely overlap (no surprise):


![Alt Text](plots/josephson.png)

![Alt Text](plots/qubit.png)


![Alt Text](plots/microwave.png)

Material keywords:

![Alt Text](plots/epitaxial.png)

![Alt Text](plots/graphene.png)

![Alt Text](plots/nanowire.png)

![Alt Text](plots/semiconductor.png)

![Alt Text](plots/superconductor.png)

More plots:

![Alt Text](plots/condensed matter.png)

![Alt Text](plots/cryogenic.png)

![Alt Text](plots/electronics.png)

![Alt Text](plots/engineering.png)

![Alt Text](plots/quality factor.png)

![Alt Text](plots/quantum dot.png)

![Alt Text](plots/topological.png)
