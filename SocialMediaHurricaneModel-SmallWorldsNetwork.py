from mesa import Agent, Model
import random
import numpy as np
from mesa.datacollection import DataCollector
import mesa.space
from mesa.time import RandomActivation
import networkx as nx
from enum import Enum
from mesa.visualization.modules import NetworkModule, ChartModule, TextElement
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
#import matplotlib.pyplot as plt

#artefact of visualization tool
class State(Enum):
   neutral = "neutral"
   help_seeker = "help_seeker"
   help_responder = "help_responder"
   helped = "helped"
   damaged = "damaged"
def number_state(model, state):
   return sum([1 for a in model.grid.get_all_cell_contents() if a.state is state])



# Model class- used as the observer and controller
class HurricaneModel(Model):
   """
   Model of hurricane issuing damages to a network of agents which post and respond to help requests
   rnum=seed
   """

   def __init__(
           self,
           rnum,
           num_nodes,
           k,
           prob,
           damage_rate,
           urgency_high,
           urgency_medium,
           urgency_low,
           attitude,
           social_norm,
           pbc,
           freq_post,
           help_seeking_threshold,
           low_response_behavior_threshold,
           high_response_behavior_threshold,
   ):
       """Create a new model.
       Args: number of nodes"""
       # Initialize model parameters
       super().__init__()
       self.reset_randomizer(rnum)
       self.num_nodes = num_nodes
       self.k=k
       self.prob = prob
       self.G = nx.watts_strogatz_graph(n=self.num_nodes, p=self.prob, k=self.k,seed=None)  # nx.convert_node_labels_to_integers
       # Set up model objects
       self.schedule = RandomActivation(self)
       self.grid = mesa.space.NetworkGrid(self.G)
       self.damage_rate = damage_rate/100
       self.urgency_high = np.random.normal(urgency_high, 0)
       self.urgency_medium = np.random.normal(urgency_medium, 0)
       self.urgency_low = np.random.normal(urgency_low, 0)
       self.datacollector = DataCollector(
           model_reporters={"Helped": lambda m: self.count_helped(m),
                            "High RB": lambda m: self.count_fetchers(m),
                            "Damaged": lambda m: self.count_damaged(m),
                            "Low RB": lambda m: self.count_reposters(m),
                            "Sought Help": lambda m: self.count_helpseekers(m)
                            },
           agent_reporters={"state": lambda a: a.state,
                            "attitude": lambda a: a.attitude,
                            "pbc": lambda a: a.pbc,
                            "spatial_value": lambda a: a.spatial_value,
                            "urgency": lambda a: a.urgency,
                            "help_seeking_threshold": lambda a: a.help_seeking_threshold,
                            "low_response_behavior_threshold": lambda a: a.low_response_behavior_threshold,
                            "high_response_behavior_threshold": lambda a: a.high_response_behavior_threshold,
                            "seek_behavior": lambda a: a.seek_behavior,
                            "damaged": lambda a: a.damaged,
                            "can_help": lambda a: a.can_help,
                            "media_post": lambda a: a.media_post,
                            "help_behavior": lambda a: a.help_behavior,
                            "i_fetch": lambda a: a.i_fetch,
                            "i_repost": lambda a: a.i_repost,
                            "helper_state": lambda a: a.helper_state,
                            "helped": lambda a: a.helped,
                            "sought_behavior": lambda a: a.sought_behavior,
                            "media_help_posts": lambda a: a.media_help_posts
                            }
       )
       # Create agents
       for ag, node in enumerate(self.G.nodes()):
           a = Person(
               ag,
               State.neutral,
               attitude,
               social_norm,
               pbc,
               freq_post,
               help_seeking_threshold,
               low_response_behavior_threshold,
               high_response_behavior_threshold,
               self
           )
           self.schedule.add(a)
           self.grid.place_agent(a, node)
       self.running = True
       self.datacollector.collect(self)

   def step(self):
       """
       Advance the model by one step.
       """
       self.schedule.step()
       self.datacollector.collect(self)

#counter function: Number of agents who are helped
   @staticmethod
   def count_helped(HurricaneModel):
       count = 0
       for ag in HurricaneModel.schedule.agents:
           if ag.helped:
               count += 1
       return count

# counter function: Number of agents reposting
   @staticmethod
   def count_reposters(HurricaneModel):
       count = 0
       for ag in HurricaneModel.schedule.agents:
           if ag.i_repost == True:
               count += 1
       return count

# counter function: Number of agents fetching
   @staticmethod
   def count_fetchers(HurricaneModel):
       count = 0
       for ag in HurricaneModel.schedule.agents:
           if ag.i_fetch == True:
               count += 1
       return count
#counter function  # agents damaged
   @staticmethod
   def count_damaged(HurricaneModel):
       count = 0
       for ag in HurricaneModel.schedule.agents:
           if ag.damaged:
               count += 1
       return count
#number of agents sought help
   @staticmethod
   def count_helpseekers(model):
       count = 0
       for ag in model.schedule.agents:
           if ag.i_need_help:
               count += 1
       return count


class Person(Agent):
   """
   A victim agent
   Attributes:
       x, y: Grid coordinates
       unique_id: (x,y) tuple.
   """

   def __init__(
           self,
           unique_id,
           state,
           attitude,
           social_norm,
           pbc,
           freq_post,
           help_seeking_threshold,
           low_response_behavior_threshold,
           high_response_behavior_threshold,
           model
   ):
       """
       Create a new agent.
       Args:
           model: standard model reference for agent.
       """
       super().__init__(unique_id, model)
       self.state = state
       self.spatial_value = np.random.choice([1, 2, 3])
       self.attitude = np.random.normal(attitude, 2.36)
       self.social_norm = np.random.normal(social_norm, 0.95)
       self.pbc = np.random.normal(pbc, 3.09)
       self.freq_post = int(np.random.normal(freq_post, 1.33))
       self.help_seeking_threshold = help_seeking_threshold/100 # no threshold variation
       self.low_response_behavior_threshold = low_response_behavior_threshold
       self.high_response_behavior_threshold = high_response_behavior_threshold#np.random.normal(high_response_behavior_threshold/100, 0.0)
       self.intent = None
       self.urgency = 0
       self.seek_behavior = 0
       self.help_count = 0
       #self.help_energy = 4  #energy decrements with every high response behavior
       self.damaged = False
       self.can_help = True  # if true, state is 'ok'
       self.sought_behavior = False
       self.media_post = (False, None)
       self.help_behavior = None
       self.i_need_help=False
       self.i_repost=False
       self.i_fetch=False
       self.helper_state = None
       self.helped = False
       self.media_help_posts = [None]
       self.calculate_intention()

   # method 1: calculate intention
   def calculate_intention(self):
       self.intent = (
               0.124
               + (0.037 * self.attitude)
               + (0.020 * self.pbc)
               + (0.038 * self.freq_post)
       )

   # method 2: get damaged
   def receive_damages(self):
       if self.model.damage_rate > random.random():
           self.damaged = True
           self.can_help = False
           if self.spatial_value == 1:
               self.urgency = self.model.urgency_high
           elif self.spatial_value == 2:
               self.urgency = self.model.urgency_medium
           else:
               self.urgency = self.model.urgency_low
   # method 4: calculate help-seeking behavior (whether to post)make post, if applicable
   def calculate_seek_behavior(self):
       if self.damaged:
           self.can_help=False
           self.helper_state=None
           self.seek_behavior = 0.106 * self.intent + 0.08 * self.urgency - 0.062
           #print(self.help_seeking_threshold)
   #method 5: make posts
   def make_post(self):
       if self.seek_behavior > self.help_seeking_threshold:
           self.i_need_help=True
           self.media_post = (True, self.unique_id)
           self.sought_behavior=True

   # method 6: check social media (Interaction! with other people)
   def check_neighbors(self):
       neighbors_nodes = self.model.grid.get_neighbors(self.unique_id, include_center=False)
       self.media_help_posts = [
           agent
           for agent in self.model.grid.get_cell_list_contents(neighbors_nodes)
           if agent.media_post[0] and not agent.helped and agent.media_post[1] == agent.unique_id
       ]
       # create new list= all neighbors who help-posted to SM (includes reposts)
       #in there is a post and the agent has not been helped, add to list
       self.help_count = len(self.media_help_posts)
       #if self.unique_id==2:
          #print("For", self.unique_id, "help list len:", self.help_count)

   #mehtod 7- giving help at different levels -low response: repost, high response: go fetch
   def calculate_response_score(self):
       if not self.damaged:
           self.help_behavior = 0.091 * self.freq_post + 0.598 * self.help_count - 0.201

   def make_repost(self):
       if  not self.damaged:
           self.helper_state = "repost"
           if len(self.media_help_posts) > 0:
               share = random.choice(self.media_help_posts)
               self.media_post = share.media_post
               self.i_repost=True
       else:
           pass
   def give_help(self):
       if not self.damaged and len(self.media_help_posts) > 0:# and self.help_energy > 0:
           #print(self.help_behavior)
           #print(self.unique_id)
           self.helper_state = "go fetch"
           #self.help_energy = self.help_energy - 1
           self.i_fetch=True
           fetch = random.choice(self.media_help_posts).media_post
           for ag in self.model.schedule.agents:
               if fetch[1] == ag.unique_id:
                   ag.helped=True
                   #chose a post, then find the agent whose unique ID matches the selected post and help them
               else:
                   pass
       else:
           pass

# artefact of visualization tool
   def visualize_state(self):
       if self.damaged:
           self.state = State.damaged
       if self.media_post[0] == True and not self.helper_state == "repost":
           self.state = State.help_seeker
       if self.helper_state == "go fetch":
           self.state = State.help_responder
       if self.helped:
           self.state = State.helped
   #sets schedule and order of methods
   def step(self):
       """
       check damages; if damaged, decide how to seek help with calculate tpb
       """
       if not self.helped:
           self.receive_damages()
           self.calculate_seek_behavior()
           self.make_post()
           if not self.damaged:
               self.check_neighbors()
               self.calculate_response_score()
               if self.help_behavior>self.low_response_behavior_threshold:
                   self.make_repost()
               if self.help_behavior>self.high_response_behavior_threshold:
                   self.give_help()
       self.visualize_state()




# ------------------------
#TEST1
# model1 = HurricaneModel(1, 10000, 100, 5, 4, 3.5, 3, 11, 6, 17, 3, 30, 8, 40)
# for i in range(16):
#     model1.step()
#     value1 = model1.datacollector.get_agent_vars_dataframe().to_csv("RUN1DR5_long.csv")
#     print(value1)
# model1 = HurricaneModel(1, 10000, 100, 5, 4, 3.5, 3, 11, 6, 17, 3, 30, 8, 40)
# for i in range(16):
#     model1.step()
#     value1 = model1.datacollector.get_agent_vars_dataframe().to_csv("RUN2DR5_long.csv")
#     print(value1)
# model1 = HurricaneModel(1, 10000, 100, 5, 4, 3.5, 3, 11, 6, 17, 3, 30, 8, 40)
# for i in range(16):
#     model1.step()
#     value1 = model1.datacollector.get_agent_vars_dataframe().to_csv("RUN3DR5_long.csv")
#     print(value1)
# model1 = HurricaneModel(1, 10000, 100, 5, 4, 3.5, 3, 11, 6, 17, 3, 30, 8, 40)
# for i in range(16):
#     model1.step()
#     value1 = model1.datacollector.get_agent_vars_dataframe().to_csv("RUN4DR5_long.csv")
#     print(value1)
# model1 = HurricaneModel(1, 10000, 100, 5, 4, 3.5, 3, 11, 6, 17, 3, 30, 8, 40)
# for i in range(16):
#     model1.step()
#     value1 = model1.datacollector.get_agent_vars_dataframe().to_csv("RUN5DR5_long.csv")
#     print(value1)
# # TEST2
# model1 = HurricaneModel(1, 10000, 100, 1, 4, 3.5, 3, 11, 6, 17, 3, 30, 8, 40)
# for i in range(12):
#     model1.step()
#     value1 = model1.datacollector.get_agent_vars_dataframe().to_csv("agentprintoutDR1HRB40.csv")
#     print(value1)
# # TEST3
# model1 = HurricaneModel(1, 10000, 100, 1, 4, 3.5, 3, 11, 6, 17, 3, 30, 8, 10)
# for i in range(12):
#     model1.step()
#     value1 = model1.datacollector.get_agent_vars_dataframe().to_csv("graph9.csv")
#     print(value1)
# # TEST4
# model1 = HurricaneModel(1, 10000, 100, 1, 4, 3.5, 3, 11, 6, 17, 3, 30, 8, 10)
# for i in range(12):
#     model1.step()
#     value1 = model1.datacollector.get_agent_vars_dataframe().to_csv("graph9.csv")
#     print(value1)

###BatchRunner Set-Up
from mesa.batchrunner import BatchRunner
fixed_params = {
    "rnum": 1,
    "num_nodes": 20000,
    "damage_rate": 5,
    "urgency_high": 4,
    "urgency_medium": 3.5,
    "urgency_low": 3,
    "attitude": 11.08,
    "social_norm": 6.25,
    "pbc": 17.2,
    "freq_post": 3
}

variable_params = {"high_response_behavior_threshold": range(25,55,10),
                   "low_response_behavior_threshold": range(5,35,10),
                   "help_seeking_threshold": range(20,40,10),
                   "prob":range(0,50,100)
                   "k": range(10,100,10)
                   }
##runs with varying thresholds, node degree 300 run again 20000************



#rerun at: Nodes 10000 SH 30, Nodes 200000, SH 20, and nodes 20000 SH 30***  (4 runs)
#repeat those same reruns for   threshold settings20 (4 runs)

#repeat those same reruns for avg node 300, threshold settings 5 and 15 and 25 (4 runs)

# this prints thresholds settings 10, includes new 30

#


# # The variables parameters will be invoke along with the fixed parameters allowing for either or both to be honored.
batch_run = BatchRunner(
    HurricaneModel,
    variable_params,
    fixed_params,
    iterations=10,
    max_steps=12,
    model_reporters={"Helped": lambda m: m.count_helped(m),
                            "High RB": lambda m: m.count_fetchers(m),
                            "Damaged": lambda m: m.count_damaged(m),
                     "Sought Help": lambda m: m.count_helpseekers(m),
                            "Low RB": lambda m: m.count_reposters(m)}
    #  agent_reporters={"attitude": "attitude",
    #                           "pbc": "pbc",
    #                           "intent":"intent",
    #                           "spatial_value": "spatial_value",
    #                           "urgency": "urgency",
    #                           "seek_behavior":  "seek_behavior",
    #                           "damaged": "damaged",
    #                           "can_help": "can_help",
    #                           "media_post": "media_post", "i_fetch": "i_fetch", "i_repost": "i_repost",
    #                           "help_behavior": "help_behavior",
    #                           "helper_state": "helper_state",
    #                           "helped": "helped",
    #                           "sought_behavior": "sought_behavior",
    #                   }
)

batch_run.run_all()
# #lowlow
run_data = batch_run.get_model_vars_dataframe().to_csv("testnetworksmallworldrangeLARGEWELL.csv")


#

##Visualization Server launch

# import datetime
#
# now = datetime.datetime.now()
# num = int((now - datetime.datetime(1970, 1, 1)).total_seconds())
#
#
# def network_portrayal(G):
#    # The model ensures there is always 1 agent per node
#    def node_color(agent):
#        colors = {
#            State.neutral: "gray",
#            State.damaged: "red",
#            State.help_seeker: "purple",
#            State.help_responder: "blue",
#            State.helped: "green"
#        }.get(agent.state, "#808080")
#        return colors
#
#    def edge_color(agent1, agent2):
#        return "#000000"
#
#    def get_agents(source, target):
#        return G.node[source]["agent"][0], G.node[target]["agent"][0]
#
#    portrayal = dict()
#    portrayal["nodes"] = [
#        {
#            "size": 6,
#            "color": node_color(agents[0]),
#            "tooltip": "id: {}<br>state: {}".format(
#                agents[0].unique_id, agents[0].state
#            ),
#        }
#        for (_, agents) in G.nodes.data("agent")
#    ]
#    portrayal["edges"] = [
#        {
#            "source": source,
#            "target": target,
#            "color": edge_color(*get_agents(source, target)),
#        }
#        for (source, target) in G.edges
#    ]
#    return portrayal
#
#
# network = NetworkModule(network_portrayal, 500, 500, library="d3")
# chart = ChartModule(
#    [
#        {"Label": "Number who Received Help", "Color": "green"},
#        {"Label": "Number Sought Help", "Color": "purple"},
#        {"Label": "Number who Went to Help", "Color": "blue"},
#    ]
# )
# random_number = UserSettableParameter("number", "Random Seed", value=num)
# urgency_low_slider = UserSettableParameter("slider", "Avg Low Urgency of Damages", 3, 0, 6, 1)
# urgency_medium_slider = UserSettableParameter("slider", "Avg Medium Urgency of Damages", 3.5, 0, 6, 1)
# urgency_high_slider = UserSettableParameter("slider", "Avg High Urgency of Damages", 4, 0, 6, 1)
# attitude_slider = UserSettableParameter("slider", "Attitude", 11.08, 0, 14, 0.01)
# freq_post_slider = UserSettableParameter("slider", "Frequency Posting", 2, 0, 6, 0.01)
# social_norm_slider = UserSettableParameter("slider", "Social Norms", 6.25, 0, 7, 0.01)
# damage_rate_slider = UserSettableParameter("slider", "% Damaged", 30, 0, 100, 1)
# pbc_slider = UserSettableParameter("slider", "PBC", 17.2, 0, 21, 0.01)
# help_seeking_threshold_slider = UserSettableParameter("slider", "Threshold to Seek Help", 15, 0, 100, 0.01)
# low_response_behavior_threshold_slider = UserSettableParameter("slider", "Low Help Response Threshold", 25, 0, 100, 1)
# high_response_behavior_threshold_slider = UserSettableParameter("slider", "High Help Response Threshold", 50, 0, 100,
#                                                                1)
# num_agents_slider = UserSettableParameter("slider", "Num of Agents", 100, 2, 100, 1)
# num_connections = UserSettableParameter("slider", "Average # of Connections", 4, 0, 50, 1)
# server = ModularServer(
#    HurricaneModel,
#    [network, chart],
#    "Hurricane Model",
#    {
#        "rnum": random_number,
#        "urgency_low": urgency_low_slider,
#        "urgency_medium": urgency_medium_slider,
#        "urgency_high": urgency_high_slider,
#        "attitude": attitude_slider,
#        "freq_post": freq_post_slider,
#        "help_seeking_threshold": help_seeking_threshold_slider,
#        "low_response_behavior_threshold": low_response_behavior_threshold_slider,
#        "high_response_behavior_threshold": high_response_behavior_threshold_slider,
#        "damage_rate": damage_rate_slider,
#        "social_norm": social_norm_slider,
#        "pbc": pbc_slider,
#        "num_nodes": num_agents_slider,
#        "avg_node_degree": num_connections,
#    },
# )
# server.port = 9011  # The default
# server.launch()
