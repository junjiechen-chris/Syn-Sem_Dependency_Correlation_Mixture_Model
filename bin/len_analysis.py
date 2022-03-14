import os
import sys

import networkx as nx
# import yaml
from operator import add
import pandas as pd


def pack_to_sent_pkg(lines):
  container = []
  sents = []
  for line in lines:
    if line != '\n':
      container += [line.strip().split('\t')]
    else:
      sents += [container]
      container = []
  return sents

class Predicate_Argument_Frame:
  def __init__(self, predicate, pred_id, pos, sentence):
    self.pred_id = pred_id
    self.word = predicate
    self.sentence = sentence
    self.pos = pos
    self.arguments = []
  def add_argument(self, argument, relation, arg_idx, is_direct_de_or_ancestor="NO-REL"):
    self.arguments.append((relation, argument, arg_idx, is_direct_de_or_ancestor))
  def __str__(self):
    return '\n'.join(['({}, {}, {})'.format(self.word, relation, argument) for relation, argument, _ in self.arguments]) + '\n'


class CoNLL09Sent:
  def __init__(self, gold_pkg, pred_pkg):
    self.graph = nx.DiGraph()
    self.words = [token_pkg[1] for token_pkg in gold_pkg]
    self.pos = [token_pkg[4] for token_pkg in gold_pkg]
    # Assuming gold and pred has the same predicate prediction & they're all gold
    self.predicates_gold = [Predicate_Argument_Frame(self.words[idx], idx, token_pkg[4], ' '.join(self.words)) for
                            idx, token_pkg in enumerate(gold_pkg) if token_pkg[12] == "Y"]
    self.predicates_pred = [Predicate_Argument_Frame(self.words[idx], idx, token_pkg[4], ' '.join(self.words)) for
                            idx, token_pkg in enumerate(pred_pkg) if token_pkg[12] == "Y"]

    # self.predicate_count = len(self.predicate_gold)
    # print(self.predicates)
    def add_arguments(sent_pkg, predicate_set):
      for col_idx in range(14, len(sent_pkg[0]), 1):
        this_predicate = predicate_set[col_idx - 14]
        # cnt = col_idx-36
        for row_idx, (token_pkg, pred_token_pkg) in enumerate(zip(sent_pkg, pred_pkg)):
          if token_pkg[col_idx] != '_':
            this_predicate.add_argument(self.words[row_idx], token_pkg[col_idx], row_idx, pred_token_pkg[10] if int(
              pred_token_pkg[8]) == this_predicate.pred_id else "NO-REL")

    add_arguments(gold_pkg, self.predicates_gold)
    add_arguments(pred_pkg, self.predicates_pred)

    self.graph.add_nodes_from([(id, {'word': word}) for id, word in enumerate(self.words)])
    for id, token_pkg in enumerate(gold_pkg):
      if token_pkg[8] != token_pkg[0]:
        # print(token_pkg)
        # print("adding {} {}".format(int(token_pkg[7])-1, id))
        self.graph.add_edge(int(token_pkg[8]), id, type=token_pkg[10])
      else:
        # self.root = token_pkg[2]
        # self.root_word = token_pkg[3]
        continue
        # this_predicate.add_argument(self.words[row_idx], token_pkg[col_idx])
    # print(self.graph.nodes)

  def __str__(self):
    return '\n---start of predicate---\n' + '\n---start of predicate---\n'.join(
      [predicate.__str__() for predicate in self.predicates])

  def get_lca(self, l, r):
    return nx.algorithms.lowest_common_ancestor(self.graph, l, r)

  def get_dep_path(self, l, r, lca):
    path_n_l = nx.algorithms.shortest_path(self.graph, lca, l)
    path_e_l = [self.graph.get_edge_data(path_n_l[idx], path_n_l[idx + 1])['type'] for idx in range(len(path_n_l) - 1)]

    path_n_r = nx.algorithms.shortest_path(self.graph, lca, r)
    path_e_r = [self.graph.get_edge_data(path_n_r[idx], path_n_r[idx + 1])['type'] for idx in range(len(path_n_r) - 1)]
    link_l = [(self.graph.nodes[node_id]['word'], '<--', edge_type, '<--') for node_id, edge_type in
              zip(reversed(path_n_l[1:]), reversed(path_e_l))]
    link_l_rel = ' ↑ '.join([item[2] for item in link_l]) + ' ↑ ' if len(link_l) > 0 else ''
    link_l = [_ for item in link_l for _ in item]
    l_count = int(len(link_l) / 4)
    link_r = [('-->', edge_type, '-->', self.graph.nodes[node_id]['word']) for node_id, edge_type in
              zip(path_n_r[1:], path_e_r)]
    link_r_rel = ' ↓ '.join([item[1] for item in link_r]) + ' ↓ ' if len(link_r) > 0 else ''
    link_r = [_ for item in link_r for _ in item]
    r_count = int(len(link_r) / 4)
    return (l_count, r_count), link_l + [self.graph.nodes[lca]['word']] + link_r

  def tobin(self, predicate, accumulator, all):
    def add_rel_to_bin(bin, triple):
      if bin not in accumulator.keys():
        accumulator[bin] = [triple]
      else:
        accumulator[bin].append(triple)

    pred_id = predicate.pred_id
    rel_track = {}
    for rel in predicate.arguments:
      arg_id = rel[2]
      l = pred_id
      r = arg_id
      dep_length = abs(l-r)
      # lca = self.get_lca(l, r)
      # bin, link = self.get_dep_path(l, r, lca)
      # dep_length = bin[0] + bin[1]
      new_triple = (l, r, rel[0], ' '.join(link))
      if rel[0] not in rel_track.keys():
        add_rel_to_bin(bin, new_triple)
        rel_track[rel[0]] = (new_triple, bin, dep_length)
      elif rel_track[rel[0]][2] > dep_length:
        old_triple, old_bin, _ = rel_track[rel[0]]
        accumulator[old_bin].remove(old_triple)
        add_rel_to_bin(bin, new_triple)
        rel_track[rel[0]] = (new_triple, bin, dep_length)
      else:
        continue
      all.append((l, r, rel[0]))

  def tobin_all(self, predicate, accumulator, all):
    pred_id = predicate.pred_id
    for rel in predicate.arguments:
      arg_id = rel[2]
      l = pred_id
      r = arg_id
      # lca = self.get_lca(l, r)
      # bin, link = self.get_dep_path(l, r, lca)
      bin = abs(l-r)
      if bin not in accumulator.keys():
        accumulator[bin] = [(l, r, rel[0])]
      else:
        accumulator[bin].append((l, r, rel[0]))
      all.append((l, r, rel[0]))

  def run(self):
    gold_bin = {}
    pred_bin = {}
    gold_all = []
    pred_all = []
    for gold_predicate, pred_predicates in zip(self.predicates_gold, self.predicates_pred):
      self.tobin_all(gold_predicate, gold_bin, gold_all)
      self.tobin_all(pred_predicates, pred_bin, pred_all)
    self.gold_bin = gold_bin
    self.pred_bin = pred_bin
    self.gold_all = gold_all
    self.pred_all = pred_all

  def f1_by_bin(self, accumulator):
    pattern_list = set(self.gold_bin.keys()).union(set(self.pred_bin.keys()))

    # print(pattern_list)
    for p in pattern_list:
      have_rel = 0
      no_rel = 0
      if p in self.gold_bin.keys():
        ref = set(self.gold_bin[p])
      else:
        ref = set()
      if p in self.pred_bin.keys():
        pred = set(self.pred_bin[p])
      else:
        pred = set()
      tp = ref.intersection(pred)
      # tp_uniq_pred = set([item[0] for item in tp])
      fp = pred.difference(tp)
      # fp_uniq_pred = set([item[0] for item in fp])
      fn = ref.difference(tp)
      # fn_uniq_pred = set([item[0] for item in fn])
      # for item in fp:
      #   if item[3] != 'NO-REL':
      #     have_rel += 1
      #   else:
      #     no_rel += 1
      if p in accumulator.keys():
        accumulator[p][0] += len(tp)
        accumulator[p][1] += len(fp)
        accumulator[p][2] += len(fn)
        # accumulator[p][3] += have_rel
        # accumulator[p][4] += no_rel
        # accumulator[p][5]+=len(fn_uniq_pred)
      else:
        accumulator[p] = [len(tp), len(fp), len(fn), have_rel, no_rel]
    ref = set(self.gold_all)
    pred = set(self.pred_all)
    tp = ref.intersection(pred)
    # tp_uniq_pred = set([item[0] for item in tp])
    fp = pred.difference(tp)
    # fp_uniq_pred = set([item[0] for item in fp])
    fn = ref.difference(tp)
    # fn_uniq_pred = set([item[0] for item in fn])
    have_rel = 0
    no_rel = 0
    # for item in fp:
    #   if item[3] != 'NO-REL':
    #     have_rel += 1
    #   else:
    #     no_rel += 1
      # print(accumulator)
    # print(tp, fp, fn)
    # print(self.gold_bin)
    # print(self.pred_bin)
    # accumulator[(-1, -1)][0] += len(tp)
    # accumulator[(-1, -1)][1] += len(fp)
    # accumulator[(-1, -1)][2] += len(fn)
    # accumulator[(-1, -1)][3] += have_rel
    # accumulator[(-1, -1)][4] += no_rel

def sum_tuples_per_3(accumulator):
  cnt = 0
  left = 0
  tmp_triple = [0, 0, 0]
  tmp_accumulator = {}
  for key, value in sorted(accumulator.items()):
    tmp_triple = list(map(add, value, tmp_triple))
    cnt+=1
    if cnt>2 and left<9:
      tmp_accumulator[(left, left+cnt-1)] = tmp_triple
      tmp_triple=[0, 0, 0]
      left = left+cnt
      cnt=0
  tmp_accumulator[(left, left + cnt - 1)] = tmp_triple

  return tmp_accumulator



def get_f1_from_triple(accumulator):
  result = {}
  for key, triple in accumulator.items():
    if isinstance(triple, int):
      continue
    p = triple[0]/(triple[0]+triple[1]) if triple[0]+triple[1] >0 else "NaN"
    r = triple[0]/(triple[0]+triple[2]) if triple[0]+triple[2] >0 else "NaN"
    f1 = 2*p*r/(p+r) if  isinstance(p, float) and isinstance(r, float) and p+r>0 else "NaN"
    # fp_direct_decendent_precentage = triple[3]/(triple[3]+triple[4]+1)
    result[key] = (p, r, f1)
  return result
path = sys.argv[1]
section = sys.argv[2]
f_gold = os.path.join(path, "srl_gold.txt.{}".format(section))
f_pred = os.path.join(path, "srl_preds.txt.{}".format(section))
with open(f_gold) as f:
  gold = f.readlines()
with open(f_pred) as f:
  preds = f.readlines()
sents_gold = pack_to_sent_pkg(gold)
sents_preds = pack_to_sent_pkg(preds)
c09sents = [CoNLL09Sent(gold, pred) for gold, pred in zip(sents_gold, sents_preds)]

for item in c09sents:
  item.run()
accumulator = {}
for item in c09sents:
  item.f1_by_bin(accumulator)
accumulator = sum_tuples_per_3(accumulator)
result = get_f1_from_triple(accumulator)
# print(result)
with open(os.path.join(path, "srl_len_analysis.{}.accumulator".format(section)), "w") as f:
  for key, value in sorted(accumulator.items()):
    f.write("{}: {}".format(key, value))
    f.write('\n')
  # yaml.dump(accumulator, stream=f, default_flow_style=False)
# with open(os.path.join(path, "srl_len_analysis.{}.metric".format(section)), "w") as f:
#   for key, value in sorted(result.items()):
#     f.write("{}: {}".format(key, value))
#     f.write('\n')

df = pd.DataFrame.from_dict(result)
df = df.sort_index(axis=1)
df.to_csv(os.path.join(path, "srl_len_analysis.csv").format(path), index=result.keys())
  # yaml.dump(result, stream=f, default_flow_style=False)

