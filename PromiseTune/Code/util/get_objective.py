import random
import numpy as np
from scipy import spatial
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------
# SPEED FIX: Cache the scaler and KD-Tree so we don't 
# rebuild them 10,000 times inside the optimization loop.
# ---------------------------------------------------------
_CACHE = {}

def get_objective_direct(dict_search, best_solution):
    return dict_search.get(tuple(best_solution)) 

def get_objective(dict_search, best_solution):
    # 1. Exact match (Fastest)
    tmp = dict_search.get(tuple(best_solution))
    if tmp:
        return tmp, list(best_solution)
        
    # 2. Check if we have already built the KDTree for this dataset
    dict_id = id(dict_search)
    
    if dict_id not in _CACHE:
        # Build it ONCE
        keys_list = [list(k) for k in dict_search.keys()]
        scaler = MinMaxScaler()
        scaler.fit(keys_list)
        keys_vectors = scaler.transform(keys_list)
        
        # Build KDTree once
        kdtree = spatial.KDTree(keys_vectors)
        
        # Pre-transform all keys for the fallback loop to avoid calling 
        # scaler.transform() inside a for-loop (which is incredibly slow)
        key_vect_map = {tuple(k): v for k, v in zip(keys_list, keys_vectors)}
        
        _CACHE[dict_id] = {
            'scaler': scaler,
            'keys_vectors': keys_vectors,
            'kdtree': kdtree,
            'key_vect_map': key_vect_map
        }
        
    # 3. Retrieve from cache
    c = _CACHE[dict_id]
    scaler = c['scaler']
    keys_vectors = c['keys_vectors']
    kdtree = c['kdtree']
    key_vect_map = c['key_vect_map']

    # Transform the single query vector
    query_vect = scaler.transform([best_solution])[0]
    
    # 4. KD-Tree Query
    _, idx = kdtree.query(query_vect, k=1)
    result = list(scaler.inverse_transform([keys_vectors[idx]])[0])
    result = [round(x) for x in result]
    
    tmp_value = dict_search.get(tuple(result))
    if tmp_value:
        return tmp_value, result
        
    # 5. The Fallback Loop (Optimized)
    # The original code called scaler.transform() on every single row here!
    min_dist = float("inf")
    final_result = None
    final_value = None
    
    for key, key_vect in key_vect_map.items():
        dist = np.linalg.norm(key_vect - query_vect)
        if dist < min_dist:
            min_dist = dist
            final_value = dict_search.get(key)
            final_result = list(key)
            
    return final_value, final_result

def distance(p1,p2):
	p1 = np.array(p1)
	p2 = np.array(p2)
	dist_orig = np.sum(np.square(p1-p2))
	return dist_orig