import pandas as pd
import pm4py
import matplotlib.pyplot as plt
from IPython.display import HTML, display
import warnings
warnings.filterwarnings("ignore")

def align_plots(width = "80%"):
  display(HTML("<style>#output-body {width:" + str(width) + ";display:flex; align-items:center; justify-content:center;}</style>"))

def resize_plot(width = "80%"):
  display(HTML("<style>#output-area {width:" + str(width) + ";}</style>"))

def convert_list_to_event_log(list, log):
  from pm4py.objects.log.obj import EventLog
  ret = EventLog(list, attributes=log.attributes, extensions=log.extensions,
                 classifiers=log.classifiers, omni_present=log.omni_present,
                 properties=log.properties)
  return ret

def draw_dfg(log):
  dfg, ia, fa = pm4py.discover_dfg(log)
  pm4py.view_dfg(dfg, ia, fa)
  return dfg

def draw_proc_tree(log):
  tree = pm4py.discovery.discover_process_tree_inductive(log)
  pm4py.vis.view_process_tree(tree)
  return tree

def draw_heur_net(log):
  net = pm4py.discover_heuristics_net(log)
  pm4py.view_heuristics_net(net)
  return net

def draw_perf_dfg(log, aggr="mean"):
  dfg, ia, fa = pm4py.discovery.discover_performance_dfg(log)
  pm4py.vis.view_performance_dfg(dfg, ia, fa, aggregation_measure=aggr)
  return dfg
  
