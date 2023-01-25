import pandas as pd
import numpy as np
from cm_network import CognitiveMapNetwork
import re

def most_occurring_factors(factors, type=None):
    res = factors["Variables"].value_counts()
    if(type):
        res = factors[factors["Type"] == type]["Variables"].value_counts()
    res.index.rename(type, inplace=True)
    return res

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
    if(total == 0):
        return 0
    return sim/total

def goal_sign(cmn, goal, p_fac):
    return p_fac[p_fac["Variables"] == goal].iloc[0]["Direction"]

def goal_conflicts(factors, relations):
    return create_conflict_table(factors, relations, "Goal", goal_sign)

def inter_goal_conflicts(factors, relations, a, b):
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

def total_action_effect(cmn, action, p_fac):
    goals = p_fac[p_fac["Type"] == "Goal"]["Variables"]
    effects = 0
    for goal in goals:
        effect = cmn.action_effect(action, goal)
        effects += effect

    return effects

def action_conflicts(factors, relations):
    res = create_conflict_table(factors, relations, "Action", total_action_effect)
    return res



def create_conflict_table(factors, relations, type, conflict):
    persons = list(factors["Person"].unique())
    conflict_items = list(factors[factors["Type"] == type]["Variables"].unique())

    res = pd.DataFrame(np.empty((len(conflict_items), len(persons))), columns=list(persons), index=list(conflict_items))
    # res = res.astype({p:'int' for p in persons})
    res.index.rename(type, inplace=True)

    for p in persons:
        p_fac = factors[factors["Person"] == p]
        p_rel = relations[relations["Person"] == p]
        cmn = CognitiveMapNetwork(p_fac, p_rel)
        # cmn.drawNetwork()
        for item in conflict_items:
            res.at[item, p] = np.NaN
            if (len(p_fac[p_fac["Variables"] == item]) > 0):
                res.at[item, p] = conflict(cmn, item, p_fac)
    return res

def create_table(factors, relations, metric):
    persons = factors["Person"].unique()
    N = len(persons)
    table = pd.DataFrame(np.zeros(shape=(N, N)), columns=persons, index=persons)
    for p1 in persons:
        for p2 in persons:
            table.loc[p1,p2] = metric(factors, relations, p1, p2)

    table.index.rename("Actor", inplace=True)
    return table

def table_to_latex(df, caption="Add caption", sort_by=None, ascending=True, gradient=False):
    df = df.reset_index()

    if(not sort_by):
        df = df.sort_values(df.columns[0], ascending=ascending)
    else:
        df = df.sort_values(sort_by, ascending=ascending)
    styler = df.style
    styler.hide(axis='index')
    styler.format(precision=0, na_rep="")
    if (gradient):
        styler.background_gradient(cmap='PuBu', low=0, high=100, axis=0, subset=None, text_color_threshold=0.408,
                                   vmin=None, vmax=None, gmap=None)

    col_str = "|" + "|".join(
        ["l" if t == object else "r" for t in df.dtypes]) + "|"
    text = styler.to_latex(convert_css=True, column_format=col_str, caption=caption)
    text = re.sub(r"(\\begin{tabular}{.*})\n(.*)", r"\1\n\\hline\n\2\n\\hline", text)
    text = re.sub(r"(\\end{tabular})", r"\\hline\n\1", text)
    text = text.replace("%", "\%")
    return text

def checkPaths(p, action, factors, relations):
    p_fac = factors[factors["Person"] == p]
    p_rel = relations[relations["Person"] == p]
    cmn = CognitiveMapNetwork(p_fac, p_rel)
    total_action_effect(cmn, action, p_fac)


def write_to_file(data, file_name):
    with open(file_name, "w") as f:
        for d in data:
            f.write("\n")
            f.write(d)


if __name__ == "__main__":
    factors = pd.read_excel("data/ASM project input.xlsx")
    relations = pd.read_excel("data/ASM project input.xlsx", "Relations")
    tables = []
    types = factors["Type"].unique()
    metrics = [{"caption": "Factor similarity",
                "metric": factor_similarity},
               {"caption": "Goal similarity",
                "metric": goal_similarity},
               {"caption": "Action similarity",
                "metric": action_similarity},
               {"caption": "Causality similarity",
                "metric": causality_similarity}]

    for type in types:
        tables.append(table_to_latex(
            most_occurring_factors(factors, type),
            f"Most occurring {type} factors", "Variables", ascending = False))

    for m in metrics:
        tables.append(table_to_latex(create_table(factors, relations, m["metric"]), m["caption"]))

    tables.append(table_to_latex(
        goal_conflicts(factors, relations), "Goal conflicts"))

    tables.append(table_to_latex(
        action_conflicts(factors, relations), "Action conflicts"))

    write_to_file(tables, "tables.tex")
