import networkx as nx
import matplotlib.pyplot as plt

class CognitiveMapNetwork:
    def __init__(self, factors, relations):
        self.g = nx.DiGraph()
        self.rels = relations
        self.factors = factors
        mapping = {factors.iloc[i]["Variables"]:i for i in range(len(factors))}

        for index, rel in relations.iterrows():
            self.g.add_edge(rel["From"], rel["To"], weight = rel["Effect"])


    def action_effect(self, action, goal):
        paths = list(nx.all_simple_paths(self.g, action, goal))
        res = 0
        for p in paths:
            path_effect = 1
            for i in range(len(p)-1):
                effect = self.rels[(self.rels["From"] == p[i]) & (self.rels["To"] == p[i + 1])].iloc[0]["Effect"]
                path_effect = path_effect * effect
            if(path_effect == self.factors[self.factors["Variables"] == goal].iloc[0]["Direction"]):
                res += 1
            else:
                res += -1
        return res

    def drawNetwork(self):
        nx.draw_networkx(G = self.g, pos=nx.kamada_kawai_layout(self.g, scale=2))
        plt.axis("off")
        plt.show()
