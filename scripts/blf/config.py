import numpy as np
import networkx as nx
import configparser
import blf.transition_matrices as tx
import re
import sys
from blf import utilities
import sklearn
from scipy.stats import truncnorm
import math
import random


def cast_param(val):
    if re.fullmatch(r"[-0-9.]+",val):
        if "." in val:
            return float(val)
        else:
            return int(val)

    else:
        return val


class Config:

    _required_sections = ['FIXED','EXPERIMENT','EXECUTION']

    def __init__(self):
        self._round_settings = {}
        self._params = {}

        self._initialized = False
        self._replication_counter = 0
        self._replications = 1
        self._nesting = []

    @property
    def alpha(self):
        return self._round_settings["alpha"]["_object"]

    @property
    def number_of_bits(self):
        return self._round_settings["number_of_bits"]

    @property
    def number_of_agents(self):
        return self._round_settings["number_of_agents"]

    @property
    def number_of_steps(self):
        return self._round_settings["number_of_steps"]

    @property
    def seed_val(self):
        return self._round_settings["seed_val"]


    @property
    def graph(self):
        return self._round_settings["graph"]["_object"]

    @property
    def tx_matrix(self):
        return self._round_settings["tx_matrix"]["_object"]

    @property
    def tau(self):
        return self._round_settings["tau"]["_object"]

    def get_round_settings(self,key):
        return self._round_settings[key]

    class Parameter:

        internal = {}

        def __init__(self,p,val,experimental = False):
            self.experimental = experimental
            self.mapname = None
            self.type = None
            if "." in p:
                parts = p.split(".")
                self.mapname = parts[0]
                self.name = parts[1]
            else:
                self.name = p

            if p.endswith("$code") | p.endswith("$iter"):
                self.type = self.name[-4:]
                self.name = self.name[:-5]


            if self.is_map():
                if not self.mapname in self.internal:
                    self.internal[self.mapname] = {}
                target = self.internal[self.mapname]
            else:
                target = self.internal

            if self.get_type() == "code":
                target[self.name] = eval(val)
            elif self.get_type() == "iter":
                vals = [float(x) for x in val.split(",")]
                target[self.name] = list(np.linspace(vals[0],vals[1],int(vals[2])))
            else:
                target[self.name] = cast_param(val)

            if experimental:
                if type(target[self.name]) is list:
                    self.numelts = len(target[self.name])
                else:
                    self.numelts = 1
                self.curr_idx = 0

        def increment(self):
            """
                Increments this parameter. If the parameter goes beyond the end of its array, resets counter to zero
                and return a False
            """
            if not self.experimental:
                raise Exception(f"Can't increment a non-experimental parameter: ${self.name}")
            else:
                self.curr_idx+=1
                result = True
                if self.curr_idx == self.numelts:
                    result = False
                    self.curr_idx = 0
            return result

        def get_current_val(self):
            if self.is_map():
                target = self.internal[self.mapname]
            else:
                target = self.internal

            if self.experimental:
                return target[self.name][self.curr_idx]
            else:
                return target[self.name]

        def export(self,target):
            if self.is_map():
                if self.mapname not in target:
                    target[self.mapname]={}
                target = target[self.mapname]
            target[self.name]=self.get_current_val()

        def reset(self):
            self.curr_idx = 0

        def __str__(self):
            s = f"{self.fullname}:{self.type}"
            if self.experimental:
                s = f"{s} [{self.curr_idx} of {self.numelts}]"
            return f"{s}->{self.get_current_val()}"

        @property
        def fullname(self):
            return f"{self.mapname}.{self.name}" if self.is_map() else self.name

        def is_map(self):
            return not self.mapname is None

        def get_type(self):
            return self.type

    def process_config_file(self,fname):
        config = configparser.ConfigParser()
        config.read(fname)
        missing_sections = set(Config._required_sections) - set(config.sections())
        if len(missing_sections):
            print("Configuration file {} missing required sections {}".format(fname,missing_sections))
            sys.exit(-1)

        for p in config["FIXED"]:
            param = Config.Parameter(p,config.get("FIXED",p),False)
            self._params[param.fullname] = param


        for p in config["EXPERIMENT"]:
            param = Config.Parameter(p,config.get("EXPERIMENT",p),True)
            self._params[param.fullname] = param

        self._replications = config.getint('EXECUTION','replications')

        for p in [s.strip() for s in config.get('EXECUTION','nesting').split(",")]:
            if "|" in p:
                p = [self._params[s.strip()] for s in p.split("|")]
            else:
                p = self._params[p.strip()]
            self._nesting.append(p)

        self.prepare_static()

    def prepare_graph(self,param_map):
        if param_map["type"] == "newman-watts-strogatz":
            return nx.newman_watts_strogatz_graph(int(self.number_of_agents), param_map["k"], param_map["p"])
        else:
            print("Graph type {} is unsupported",param_map["type"])
            sys.exit(-1)

    def prepare_tx_matrix(self,param_map):

        if param_map["type"] == "random-ising":
            return tx.buildRandomIsingBasedTransitionMatrix(self.number_of_bits, param_map["pneg"], param_map['width'])
        elif param_map["type"] == "ising-trial":
            return tx.trial1IsingMatrix(self.number_of_bits, param_map["pzero"])
        elif param_map["type"] == "manual":
            attractors = param_map["attractors"][:int(param_map['num_attractors'])]
            aw = int(param_map['attractor_width'])
            lad = int(param_map['local_depth'])
            ga = (attractors[0][0],attractors[0][1],aw)
            la = [(x[0],lad,aw) for x in attractors[1:]]
            ap = tx.build_attractor_profile(self.number_of_bits,[ga]+la)
            return tx.build_manual_transition_matrix(self.number_of_bits,ap,search_width = param_map["search_width"])
        elif param_map['type'] == "loadable":
            m = np.load(param_map['filename'])
            return tx.amplify(m,param_map["amplification"])
        elif param_map['type'] == "hamming_sweep":
            attractors = param_map['attractors']
            h = tx.build_scaled_hamming_distance_matrix(8, 4)
            m = tx.generate_tuned_surface(self.number_of_bits,attractors)
            tmp = h*m
            for i,x in enumerate(np.sum(tmp,axis=1)):
                if x == 0:
                    tmp[i] = h[i]
            result = tx.amplify(tmp,1)
            return result
        else:
            print("Transition matrix type {} is unsupported",param_map["type"])
            sys.exit(-1)

    def prepare_alpha(self,param_map):
        if param_map['type'] == "direct":
            return [param_map['value']]*self.number_of_agents
        elif param_map['type'] == 'distribution':
            lower, upper = 0, 1
            mu, sigma = param_map['mu'], param_map['sigma']
            X = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
            x = X.rvs(self.number_of_agents)
            delta = x - mu
            x = mu+(delta**3)
            return x
        elif param_map['type'] == 'power_distribution':
            rho = param_map["r"]
            x = np.random.uniform(size = self.number_of_agents) ** math.exp(rho)
            return x
        else:
            print(f"Alpha type {param_map['type']} is unsupported")
            sys.exit(-1)

    def prepare_tau(self,param_map):
        if param_map['type'] == "sigmoid":
            return utilities.get_tau_distr(param_map['upper'], param_map['lower'], param_map["mu"], param_map['sigma'], self.number_of_agents)
        elif param_map['type'] == "direct-proportion":
            return np.ones(self.number_of_agents)
        else:
            print("Tau distribution type {} is unsupported",param_map["type"])
            sys.exit(-1)



    def prepare_static(self):
        for x in self._params.values():
            if not x.experimental:
                x.export(self._round_settings)

    def prepare_round(self):
        #print("Start prepare round")
        for x in self._params.values():
            x.export(self._round_settings)

        self._round_settings['graph']['_object']= self.prepare_graph(self._round_settings["graph"])
        self._round_settings['tx_matrix']['_object'] = self.prepare_tx_matrix(self._round_settings["tx_matrix"])
        self._round_settings['tau']['_object'] = self.prepare_tau(self._round_settings["tau"])
        self._round_settings['alpha']['_object'] = self.prepare_alpha(self._round_settings["alpha"])
        #print("Stop prepare round")

    def init(self):
        for p in self._params.values():
            p.reset()
        self._initialized = True

    def reset(self):
        self._initialized = False




    def step(self):
        if not self._initialized:
            self.init()
            self._replication_counter+=1
        else:
            if self._replication_counter >= self._replications:
                self._replication_counter = 0
                for i,p in enumerate(reversed(self._nesting)):
                    if type(p) is list:
                        p_list = p.copy()
                        p_max = max(p_list,key=lambda x: x.numelts)
                        for p_sub in p_list:
                            if p_sub != p_max:
                                p_sub.increment()
                        p=p_max
                    if p.increment():
                        break
                    elif i == len(self._nesting)-1:
                        print("Finished!")
                        self.reset()
                        return False

            self._replication_counter+=1
        self.prepare_round()
        return True

    def _collect_parameters(self,map,prefix = ""):
        result = {}
        for key,val in map.items():
            if key == "_object":
                continue
            elif type(val) == dict:
                result.update(self._collect_parameters(val,key))
            else:
                result[f"{prefix}{'.' if len(prefix)>0 else ''}{key}"] = val
        return result

    def collect_parameters(self):
        return self._collect_parameters(self._round_settings)


    def get_run_id(self):
        vals = [str(i.curr_idx) for i in self._nesting] + [str(self._replication_counter)]
        return ".".join(vals)

    def inspect(self):
        vals = [str(i) for i in self._nesting]
        print(f"Replication: {self._replication_counter} -> {vals}")



def test(file):
    c = Config()
    c.process_config_file(file)
    while c.step():
       print(f"{c.collect_parameters()}")








