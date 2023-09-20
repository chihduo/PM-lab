from IPython.display import HTML, display
import pm4py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import humanize
import warnings
warnings.filterwarnings("ignore")
"""
Directly-follows performance heatmap
Note: only applicable to events with a single timestamp attribute
"""
def draw_dfg_perf_matrix(
  event_log: pd.DataFrame, 
  case_var: str = "case:concept:name", 
  activity_var: str = "concept:name", 
  timestamp_var: str = "time:timestamp",
  # Use 'h' for hours, 's' for seconds, 'D' for days, and 'W' for weeks
  time_unit: str ='h'):

  # event log
  log = event_log[[case_var, activity_var, timestamp_var]]
  
  # initiate matrix
  events = log[activity_var].unique()
  matrix = pd.DataFrame(columns=events, index=events)
  
  # groupby case_var
  groups = log.groupby(case_var)
  
  # loop through case groups
  for group in groups:
    event = group[1].sort_values(timestamp_var)\
    .rename(columns = {activity_var:"event_from", timestamp_var:"time_begin"})
    event["event_to"] = event["event_from"].shift(-1)
    event["time_end"] = event["time_begin"].shift(-1)
    event["duration"] = (event["time_end"] - event["time_begin"]) / np.timedelta64(1, time_unit)
    event.dropna(inplace = True)
    # loop through traces
    for row in event.itertuples(index = False):
      matrix.at[row.event_from, row.event_to] = \
      np.nansum([matrix.at[row.event_from, row.event_to], row.duration])
  
  matrix = matrix.astype(float)
  sns.set(rc={"figure.figsize":(8, 6)})
  sns.heatmap(matrix, annot=True, fmt=".0f", cmap="BuPu", square=True)
  
  # Rows are source events and columns are target events
  plt.show()

"""
Directly-follows frequency heatmap
"""
def draw_dfg_freq_matrix(
  event_log: pd.DataFrame, 
  case_var: str = "case:concept:name", 
  activity_var: str = "concept:name", 
  timestamp_var: str = "time:timestamp"):
  # event log
  log = event_log[[case_var, activity_var, timestamp_var]]
  
  # initiate matrix
  events = log[activity_var].unique()
  matrix = pd.DataFrame(columns=events, index=events).fillna(0)
  
  # groupby case_var
  groups = log.groupby(case_var)
  
  # loop through case groups
  for group in groups:
      event = group[1].sort_values(timestamp_var)\
      .drop([case_var, timestamp_var], axis = 1)\
      .rename(columns = {activity_var:"event_from"})
      event["event_to"] = event["event_from"].shift(-1)
      event.dropna(inplace = True)
      
      # loop through traces
      for trace in event.itertuples(index = False):
          matrix.at[trace.event_from, trace.event_to] += 1
              
  matrix = matrix.replace(0, np.nan)
  sns.set(rc={"figure.figsize":(8, 6)})
  sns.heatmap(matrix, annot=True, fmt=".0f", cmap="Reds", square=True)

  # Rows are source events and columns are target events
  plt.show()

"""
Activity duration Boxplot
"""
def draw_duration_boxplot(
  event_log: pd.DataFrame, 
  case_var: str = "case:concept:name", 
  event_var: str = "concept:name", 
  timestamp_var: str = "time:timestamp",
  # Use 'h' for hours, 's' for seconds, 'D' for days, and 'W' for weeks
  time_unit: str ='h'):    
  data = dict()
  duration_var = f"duration ({time_unit})"
  groups = event_log.groupby(case_var)
  for group in groups:
      arc = group[1].sort_values(timestamp_var)\
      .rename(columns = {event_var:"event_from", timestamp_var:"time_from"})
      arc["event_to"] = arc["event_from"].shift(-1)
      arc["time_to"] = arc["time_from"].shift(-1)
      arc.dropna(inplace = True)
      arc[duration_var] = (arc["time_to"] - arc["time_from"]) / np.timedelta64(1, time_unit)
      data[group[0]] = arc[[case_var, "event_from", "event_to", "time_from", "time_to", duration_var]]
  boxes = pd.concat(data.values()).set_index(case_var)\
  .loc[:, ["event_from", "event_to", duration_var]]\
  .reset_index()\
  .pivot(columns="event_from", values=duration_var)\
  .plot(kind="box", vert=False, title="Activity duration (" + time_unit + ")")
  plt.show()

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

def draw_proc_tree(log, noise_threshold = 0.0):
  tree = pm4py.discover_process_tree_inductive(top_k, noise_threshold)
  pm4py.view_process_tree(tree)

def draw_heur_net(log):
  net = pm4py.discover_heuristics_net(log)
  pm4py.view_heuristics_net(net)

def draw_perf_dfg(log, aggr="mean"):
  dfg, ia, fa = pm4py.discovery.discover_performance_dfg(log)
  pm4py.vis.view_performance_dfg(dfg, ia, fa, aggregation_measure=aggr)
  
