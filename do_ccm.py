import pandas as pd
import numpy as np
from cm_network import CognitiveMapNetwork

def most_occurring_factors(factors, type=None):
    if(type):
        return factors[factors["Type"] == type]["Variables"].value_counts()
    return factors["Variables"].value_counts()

def factor_similarity(factors, relations, a, b):
    f_a = factors[factors["Person"] == a]["Variables"]
    f_b = factors[factors["Person"] == b]["Variables"]
    total = len(pd.merge(f_a, f_b, on='Variables', how='outer'))
    count_a = len(f_a)
    count_b = len(f_b)

    return f"{round((count_a + count_b - total)/total*100, 2)}%"

def goal_similarity(factors, relations, a, b):
    f_a = factors[(factors["Person"] == a) & (factors["Type"] == "Goal")]["Variables"]
    f_b = factors[(factors["Person"] == b) & (factors["Type"] == "Goal")]["Variables"]
    total = len(pd.merge(f_a, f_b, on='Variables', how='outer'))
    count_a = len(f_a)
    count_b = len(f_b)

    return f"{round((count_a + count_b - total)/total*100, 2)}%"

def action_similarity(factors, relations, a, b):
    f_a = factors[(factors["Person"] == a) & (factors["Type"] == "Action")]["Variables"]
    f_b = factors[(factors["Person"] == b) & (factors["Type"] == "Action")]["Variables"]
    total = len(pd.merge(f_a, f_b, on='Variables', how='outer'))
    count_a = len(f_a)
    count_b = len(f_b)

    return f"{round((count_a + count_b - total)/total*100, 2)}%"

def getRelationSign(rel, fro, to):
    res = rel[(rel["From"] == fro) & (rel["To"] == to)]
    sign = 0
    if len(res) > 0:
        sign = res.iloc[0]["Effect"]
    return sign

# For every pair of shared factors, we check if the person agree on the relation in both directions
def causality_similarity(factors, relations, a, b):
    f_a = factors[factors["Person"] == a]
    f_b = factors[factors["Person"] == b]
    intersect = pd.merge(f_a, f_b, on='Variables', how='inner')["Variables"]
    rel_a = relations[(relations["Person"] == a)]
    rel_b = relations[(relations["Person"] == b)]

    total = 0
    sim = 0
    for fro in intersect:
        for to in intersect:
            if fro != to:
                res_a = getRelationSign(rel_a, fro, to)
                res_b = getRelationSign(rel_b, fro, to)
                total += 1
                if res_a == res_b:
                    sim += 1

    return sim/total

def goal_conflicts(factors, relations, a, b):
    f_a = factors[(factors["Person"] == a) & (factors["Type"] == "Goal")]
    f_b = factors[(factors["Person"] == b) & (factors["Type"] == "Goal")]
    goals = pd.merge(f_a, f_b, on='Variables', how='inner')["Variables"]

    conflicts = []
    for goal in goals:
        sign_a = f_a[f_a["Variables"] == goal].iloc[0]["Direction"]
        sign_b = f_b[f_b["Variables"] == goal].iloc[0]["Direction"]
        if(sign_a != sign_b):
            conflicts.append(goal)
    return conflicts

def total_action_effect(cmn, action, goals):
    effects = 0
    for goal in goals:
        effect = cmn.action_effect(action, goal)
        effects += effect

    return effects


def action_conflicts(factors, relations):
    persons = list(factors["Person"].unique())
    actions = list(factors[factors["Type"] == "Action"]["Variables"].unique())

    res = pd.DataFrame(np.empty((len(actions), len(persons))), columns=list(persons), index=list(actions))

    for p in persons:
        p_fac = factors[factors["Person"] == p]
        p_rel = relations[relations["Person"] == p]
        cmn = CognitiveMapNetwork(p_fac, p_rel)
        cmn.drawNetwork()
        for a in actions:
            res.at[a, p] = np.NaN
            if(len(p_fac[p_fac["Variables"] == a]) > 0):
                effect = total_action_effect(cmn, a, p_fac[p_fac["Type"] == "Goal"]["Variables"])
                res.at[a, p] = effect
    return res

def create_table(factors, relations, metric):
    persons = factors["Person"].unique()
    N = len(persons)
    table = pd.DataFrame(np.zeros(shape=(N, N)), columns=persons, index=persons)
    for p1 in persons:
        for p2 in persons:
            table.loc[p1,p2] = metric(factors, relations, p1, p2)
    return table


if __name__ == "__main__":
    factors = pd.read_excel("data/ASM project input.xlsx")
    relations = pd.read_excel("data/ASM project input.xlsx", "Relations")
    print(factors)
    # print(create_table(factors, relations, causality_similarity))
    # print(goal_conflicts(factors, relations, 2, 5))
    # print(most_occurring_factors(factors, "Goal"))
    print(action_conflicts(factors, relations))

