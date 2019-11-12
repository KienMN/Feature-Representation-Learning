# Feature Representation Learning
This repository provides implementation of some feature representation learning such as word2vec, node2vec.

## Introduction
Feature Engineering is the secret sauce to creating superior and better performing machine learning models. Many supervised machine learning algorithm requires a set of informative, discriminating, and independent features. In prediction problems on networks or texts, this means that one has to construct a feature vector representation for the nodes, edges and words.  
Recent advancements in representational learning allows new ways for feature learning of discrete objects.  
In this repository, I try to implement different approaches for feature learning which can be referenced for future study.

## Notebooks
`karate_class_cluster.ipynb`: Using Node2Vec to learn presentation and cluster nodes in Zachary karate class network.  
`les_miserables_network.ipynb`: Using Node2Vec to learn presentation and cluster nodes in the Les Miserables network.

`cbow_model.ipynb`: Learning vector representation of words using Continuous Bag Of Words (CBOW) model.  

`skip_gram_model.ipynb`: Learning vector representation of words using Skip-Gram model.  

## Reference
1. Node2Vec paper: Aditya Grover and Jure Leskovec. node2vec: Scalable Feature Learning for Networks. Knowledge Discovery and Data Mining, 2016.
2. Node2Vec implementation: https://github.com/aditya-grover/node2vec
3. A hands-on intuitive approach to Deep Learning Methods for Text Data — Word2Vec, GloVe and FastText: https://towardsdatascience.com/understanding-feature-engineering-part-4-deep-learning-methods-for-text-data-96c44370bbfa
4. Word2Vec paper: T. Mikolov, K. Chen, G. Corrado, and J. Dean. Efficient estimation of word representations in vector space. In ICLR, 2013.
5. Alias sampling: https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/