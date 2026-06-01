from sklearn.ensemble import RandomForestRegressor
import numpy as np

def get_tree_paths(tree,feature_names):
    paths = []
    stack = [(0, [])]
    while stack:
        node_id, path = stack.pop()
        feature_index = tree.tree_.feature[node_id]
        threshold = tree.tree_.threshold[node_id]

        if feature_index != -2:  
            feature_name = feature_names[feature_index]
            left_path = path + [(feature_name, threshold, 'L')]
            stack.append((tree.tree_.children_left[node_id], left_path))
            right_path = path + [(feature_name, threshold, 'R')]
            stack.append((tree.tree_.children_right[node_id], right_path))
        else:
            paths.append(path)

    return paths

def has_negative_value(lst):
    for item in lst:
        if item < 0:
            return True
        
    return False

def in_rule(x,rule,feature_names):
    for single in rule:
        k = feature_names.index(single[0])
        if single[2] == 'L':
            if float(x[k]) <= float(single[1]):
                continue
            else:
                return False
        else:
            if float(x[k]) > float(single[1]):
                continue
            else:
                return False
            
    return True

def Model(train_independent, train_dependent):
    model = RandomForestRegressor()
    model.fit(train_independent, train_dependent)

    return model

def merge_constraints(constraints_list):
    merged = {}
    counts = {}
    for feat, val, side in constraints_list:
        op = '<' if side == 'L' else '>'
        counts[feat] = counts.get(feat, 0) + 1
        if feat not in merged:
            merged[feat] = (op, val)
        else:
            cur_op, cur_val = merged[feat]
            if op == cur_op:
                merged[feat] = (op, min(cur_val, val)) if op == '<' else (op, max(cur_val, val))
    def fmt(val):
        return int(val) if val == int(val) else val
    return ','.join(f"{feat}{op}{fmt(val)}" for feat, (op, val) in merged.items())

def Latin_sample(file, num_samples):
    def round_num(num, discrete_list):
        max_num = float('inf')
        return_num = 0
        for i in discrete_list:
            if abs(num-i) < max_num:
                max_num = abs(num-i)
                return_num = i
        return return_num
    variables = {}
    bounds = file.independent_set
    for i in range(len(bounds)):
        if len(bounds[i]) == 1:
            bounds[i].append(1) 
    for i in range(len(file.independent_set)):
        variables[file.features[i]] = bounds[i]
    from doepy import build
    sample = build.space_filling_lhs(variables, num_samples)
    data_array = np.array(sample)
    data_list = data_array.tolist()
    for i in range(len(data_list)):
        for j in range(len(data_list[i])):
            data_list[i][j] = round_num(data_list[i][j], file.independent_set[j])

    return data_list

