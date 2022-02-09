#Filename:	nondominant_ratio.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Kam 22 Apr 2021 04:10:35 

import numpy as np

def count_diffs(a, b, to_min, to_max):
    n_better = 0
    n_worse = 0

    for f in to_min:
        n_better += a[f] < b[f]
        n_worse += a[f] > b[f]

    for f in to_max:
        n_better += a[f] > b[f]
        n_worse += a[f] < b[f]

    return n_better, n_worse

def nondominant_ratio(data, to_min, to_max):
    """
    Case 1: The point is dominated by one of the elements in the skyline
    Case 2: The point dominate one or more points in the skyline
    Case 3: The point is same to one of the elements in the skyline
    Case 4: The point is neither better nor worse than all of the points in the skyline
    """
    skyline_index = {0}
    nr = np.ones(10)

    for i in range(1, len(data)):

        to_drop = set()
        is_dominated = False

        for j in skyline_index:
            n_better, n_worse = count_diffs(data[i], data[j], to_min, to_max)
            # Case 1
            if n_worse > 0 and n_better == 0:
                is_dominated = True
                break

            # Case 2
            if n_better > 0 and n_worse == 0:
                to_drop.add(j)

            # Case 3
            if n_better == 0 and n_worse == 0:
                to_drop.add(j)

        if is_dominated:
            nr[i] = nr[i-1]
            continue

        skyline_index = skyline_index.difference(to_drop)
        skyline_index.add(i)
        nr[i] = len(skyline_index)
    
    nr_final = nr / np.arange(1, 11)
    return nr_final

def nondominant_ratio_v1(data, to_min, to_max):
    """
    Case 1: The point is dominated by one of the elements in the skyline
    Case 2: The point dominate one or more points in the skyline
    Case 3: The point is same to one of the elements in the skyline
    Case 4: The point is neither better nor worse than all of the points in the skyline
    """
    skyline_index = {0}

    for i in range(1, len(data)):

        to_drop = set()
        is_dominated = False

        for j in skyline_index:
            n_better, n_worse = count_diffs(data[i], data[j], to_min, to_max)
            # Case 1
            if n_worse > 0 and n_better == 0:
                is_dominated = True
                break

            # Case 2
            if n_better > 0 and n_worse == 0:
                to_drop.add(j)

            # Case 3
            if n_better == 0 and n_worse == 0:
                to_drop.add(j)

        if is_dominated:
            continue

        skyline_index = skyline_index.difference(to_drop)
        skyline_index.add(i)
    
    return len(skyline_index) / len(data)
