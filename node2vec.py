import numpy as np
import networkx as nx
import random

class Graph():
  def __init__(self, nx_G, is_directed, p, q):
    self._G = nx_G
    self._is_directed = is_directed
    self._p = p
    self._q = q

  def preprocess_transition_probs(self):
    '''
    Preprocessing of transition probabilities for guiding random walks.
    '''
    G = self._G
    is_directed = self._is_directed
    alias_nodes = {}

    for node in G.nodes():
      unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
      norm_const = sum(unnormalized_probs)
      normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
      alias_nodes[node] = alias_setup(normalized_probs)

    alias_edges = {}
    # triads = {}
    
    if is_directed:
      for edge in G.edges():
        alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
    else:
      for edge in G.edges():
        alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])
    
    self.alias_nodes = alias_nodes
    self.alias_edges = alias_edges
    
    return

  def get_alias_edge(self, src, dst):
    '''
    Get the alias edge setup lists for a given edge.
    '''
    G = self._G
    p = self._p
    q = self._q

    unnormalized_probs = []
    for dst_nbr in sorted(G.neighbors(dst)):
      if dst_nbr == src:
        unnormalized_probs.append(G[dst][dst_nbr]['weight'] / p)
      elif G.has_edge(dst_nbr, src):
        unnormalized_probs.append(G[dst][dst_nbr]['weight'])
      else:
        unnormalized_probs.append(G[dst][dst_nbr]['weight'] / q)
    norm_const = sum(unnormalized_probs)
    normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

    return alias_setup(normalized_probs)

  def simulate_walks(self, num_walks, walk_length):
    '''
    Repeatedly simulate random walks from each node.
    '''
    G = self._G
    walks = []
    nodes = list(G.nodes())
    print('Walk iteration:')
    for walk_iter in range (num_walks):
      print(str(walk_iter + 1) + '/' + str(num_walks))
      random.shuffle(nodes)
      for node in nodes:
        walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

    return walks

  def node2vec_walk(self, walk_length, start_node):
    '''
    Simulate a random walk starting from start node.
    '''
    G = self._G
    alias_nodes = self.alias_nodes
    alias_edges = self.alias_edges

    walk = [start_node]

    while len(walk) < walk_length:
      cur = walk[-1]
      cur_nbrs = sorted(G.neighbors(cur))
      if len(cur_nbrs) > 0:
        if len(walk) == 1:
          walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
        else:
          prev = walk[-2]
          next_v = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1])]
          walk.append(next_v)
      else:
        break

    return walk

def alias_setup(probs):
  '''
  Compute utility lists for non-uniform sampling from discrete distribution.
  '''
  K = len(probs)
  q = np.zeros(K)
  J = np.zeros(K, dtype=np.int)

  # Sort the data into the outcomes with probabilities that are larger and smaller than 1/K.
  smaller = []
  larger = []

  for kk, prob in enumerate(probs):
    q[kk] = K*prob
    if q[kk] < 1.0:
      smaller.append(kk)
    else:
      larger.append(kk)

  # Loop though and create little binary mixtures that appropriately
  # allocate the larger outcomes over the overall uniform mixture.
  while len(smaller) > 0 and len(larger) > 0:
    small = smaller.pop()
    large = larger.pop()

    J[small] = large
    q[large] = q[large] + q[small] - 1.0
    if q[large] < 1:
      smaller.append(large)
    else:
      larger.append(large)

  return J, q

def alias_draw(J, q):
  '''
  Draw sample from a non-uniform discrete distribution using alias sampling.
  '''
  K = len(J)

  kk = int(np.floor(np.random.rand() * K))
  if np.random.rand() < q[kk]:
    return kk
  else:
    return J[kk]