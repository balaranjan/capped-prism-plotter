import os
import traceback
import numpy as np
import pandas as pd
from cifkit import Cif
from cifkit.utils import unit
from collections import defaultdict
from scipy.spatial import ConvexHull
from matplotlib import pyplot as plt
from cif_parser import _parse_formula, cif_to_dict


def rearrange_coordinates(points):
    """
    Rearrange a set of 2D coordinates for proper polygon plotting with pyplot.fill.
    The function sorts points in counterclockwise order around the centroid.
    """
    if not len(points):
        return []

    centroid = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    sorted_points = points[np.argsort(angles)]
    sorted_points = [tuple([float(point[0]), float(point[1])]) for point in sorted_points]
    sorted_points.append(sorted_points[0])

    return [p[0] for p in sorted_points], [p[1] for p in sorted_points]



def is_close(v1, v2):
    if abs(v1 - v2) < 0.1:
        return True
    return False



def sround(val):
    
    val = round(val, 4)
    val = round(val, 3)
    val = round(val, 2)
    # val = round(val, 1)
    
    return val


def get_sg_symbol(cif_path):

    sgs = {'P121/m1': 'P2_1/m', 'P-62m': 'P\\bar{6}2m', 'I41md': 'I4_1md', 
           'P-6m2': 'P\\bar{6}m2', 'P-6': 'P\\bar{6}', 'C12/m1': 'C2/m', 
           'P42/mnm': 'P4_2mnm', 'Pmn21': 'Pmn2_1', 'Pmc21': 'Pmc2_1', 
           'P1m1': 'Pm', 'P63/m': 'P6_3/m', 'Cmc21': 'Cmc2_1', 
           'C1m1': 'Cm', 'C2221': 'C222_1', 'P21212': 'P2_12_12'}
    
    with open('sgs.csv', 'r') as f:
        for line in f.readlines()[1:]:
            k, v = line.split(',')
            if k not in sgs:
                sgs[k] = v[:-1].replace("$", "")
    
    with open(cif_path, 'r') as f:

        lines = f.readlines()
        i_start = [i for i in range(len(lines)) if "_space_group_name_H-M_alt" in lines[i]][0]
        i_end = [i for i in range(i_start, len(lines)) if "loop_" in lines[i]][0]
        
        sg_symbol = "".join(lines[i_start:i_end]).replace("_space_group_name_H-M_alt", "").strip().replace(" ", "").replace("'", "")
        formatted_sg_symbol = ""
        
        for s in sg_symbol:
            if s.isalpha():
                formatted_sg_symbol += f"{s}"
            else:
                formatted_sg_symbol += f"{s}"
        formatted_sg_symbol = formatted_sg_symbol.replace("**", "")
        formatted_sg_symbol = sgs.get(formatted_sg_symbol.strip(), formatted_sg_symbol)
        formatted_sg_symbol = "$" + formatted_sg_symbol + "$"
        if "originchoice2" in formatted_sg_symbol:
            formatted_sg_symbol = formatted_sg_symbol.replace("(originchoice2)", "")
            formatted_sg_symbol += " (O2)"
        
        return formatted_sg_symbol


def format_formula(formula):
    
    formula = _parse_formula(formula)
    new_formula = ""
    for k, v in formula.items():
        if v == 1.0:
            new_formula += k
        else:
            if abs(int(v) - v) < 1e-3:
                new_formula += k + "$_{%s}$" % int(v)
            else:
                new_formula += k + "$_{%s}$" % v
    
    return new_formula


def get_capped_prism_data(site, layer_axis, points_wd, CN):
    
    """
    Check for the presence of capped prisms and return the 
    coordinates.
    """

    num_capping_atoms = int(len(points_wd) / 3)
    num_prism_atoms = num_capping_atoms * 2
    
    center = np.array(points_wd[0][2])
    capped_prism_data = {'center_site': site, 'cp_present': False, 
                        'layer_height': round(float(points_wd[0][2][layer_axis]), 2),
                         'capped_prism_present': False}

    # drop center from points_wd 
    points_wd = [[p[0], p[1], np.array(p[3])] for p in points_wd]
    non_layer_axes = np.array([0, 1, 2]) != layer_axis
    capped_prism_data['center_coordinate'] = center[non_layer_axes]
    
    capping_inds = [
        i for i in range(len(points_wd)) \
            if round(points_wd[i][2][layer_axis], 2) == round(center[layer_axis], 2)]
    
    capping_atoms = [
        p for p in points_wd if abs(float(p[2][layer_axis]) - center[layer_axis]) <= 0.2]

    # if len(capping_atoms) != num_capping_atoms:
    #     capped_prism_data['error'] = f"Num capping atoms are {len(capping_atoms)}"
    #     return capped_prism_data
    
    prism_atoms = [points_wd[c] for c in range(len(points_wd)) if c not in capping_inds]
    unique_heights = set([round(float(p[-1][layer_axis]), 1) for p in prism_atoms])
    m_unique_heights = []
    for unique_height in unique_heights:
        if np.any(abs(np.array(m_unique_heights) - unique_height) <= 0.5):
            continue
        m_unique_heights.append(unique_height)

    unique_heights = m_unique_heights
    
    # if len(unique_heights) != 2:
    #     capped_prism_data["error"] = f"Num unique heights for triangle prism are {len(unique_heights)}, {unique_heights}"
    #     return capped_prism_data

    # split square based on height
    prism_sites_by_heights = {}
    for k in unique_heights:
        val = []
        for atom in prism_atoms:
            if abs(round(float(atom[-1][layer_axis]), 1) -k) <= 0.5:
                val.append(atom)
        prism_sites_by_heights[k] = val
        
    for k, v in prism_sites_by_heights.items():
        v = np.vstack([p[-1] for p in v])[:, non_layer_axes]
        assert len(np.unique(v, axis=1) == 3), f"{site}, {len(v)}"
    
    # pair capping atoms to square edges
    layer_1 = sorted([k for k in prism_sites_by_heights.keys() if len(prism_sites_by_heights[k])==int(num_prism_atoms/2)])

    if len(layer_1):
        layer_1 = layer_1[0]
    else:
        capped_prism_data["error"] = "No prism found"
        return capped_prism_data
    

    prism = prism_sites_by_heights[layer_1]
    
    for i in range(len(prism)):
        point = prism[i][-1][non_layer_axes]
        poly = [prism[j][-1][non_layer_axes] for j in range(len(prism)) if j!=i]
        
        if point_in_hull(poly, point):
            capped_prism_data["error"] = "Prism point inside prism"
            return capped_prism_data
        
    for i in range(len(capping_atoms)):
        point = capping_atoms[i][-1][non_layer_axes]
        poly = [prism[j][-1][non_layer_axes] for j in range(len(prism))]
        
        if point_in_hull(poly, point):
            capped_prism_data["error"] = "Capping point inside prism"
            return capped_prism_data
    
    if len(prism) != num_prism_atoms / 2:
        capped_prism_data["error"] = f"Number of atoms for prism is {len(prism)}. Expected {int(num_prism_atoms/2)}"
        return capped_prism_data
    try:
        hull = ConvexHull([p[-1][non_layer_axes] for p in prism])
    except:
        capped_prism_data["error"] = f"Error while constructing hull"
        return capped_prism_data
    edges = []
    for simplex in hull.simplices:
        edges.append([int(simplex[0]), int(simplex[1])])
    
    faces = []
    for edge in edges:
        edge = list(edge)[:2]
        for i in range(int(CN/3)):
            if i not in edge:
                edge.append(i)
        faces.append(edge)
        
    face_points = np.array([p[-1][non_layer_axes] for p in prism])

    if not point_in_hull(poly=[p[2][non_layer_axes] for p in prism],
                         point=center[non_layer_axes]):
        capped_prism_data["error"] = f"Center not inside hull"
        return capped_prism_data
    
    # check for capping atoms for all faces
    # cap_atom_present = []
    # for i, face in enumerate(faces):

    #     _has_capped_atom = has_capped_atom(face_points=face_points, 
    #                                        face_indices=face,
    #                                        capping_atoms=capping_atoms,
    #                                        non_layer_axes=non_layer_axes)
    #     cap_atom_present.append(_has_capped_atom)
      
    # if all(cap_atom_present):
    capped_prism_data['capped_prism_present'] = True
    capped_prism_data['prism'] = face_points
    capped_prism_data['edges'] = edges
    capped_prism_data['caps'] = capping_atoms
    capped_prism_data['prism_full'] = prism

    # capped_prism_data['cap_for_faces'] = cap_atom_present

    return capped_prism_data


def get_point_inside_cell(v):
    if v > 1:
        v -= 1
    elif v < 0:
        v += 1

    return v


def get_first_n_neighbors(point, supercell_points, n):

    points_wd = []
    
    sites = [p[-1] for p in supercell_points]
    supercell_points = np.vstack([p[0] for p in supercell_points])
    d = np.linalg.norm(supercell_points - point[0].reshape(1, 3), axis=1)

    ind9 = np.argsort(d)[1:n+1]

    for i in ind9:
        points_wd.append((sites[i], 
                          round(float(d[i]), 3), 
                          [round(float(v), 3) for v in point[0].tolist()], 
                          [round(float(v), 3) for v in supercell_points[i].tolist()]))
    return points_wd


def get_bbox_y(supercell_points, non_layer_axes, fix_min_length=None):
    
    points = np.vstack([p[0][non_layer_axes] for p in supercell_points])
    height = points[:, 1].max() - points[:, 1].min()
    
    if fix_min_length:
        percent = 0.2
        return -percent, points[:, 1].max()+0.5, points[:, 1].min()-0.5, points[:, 0].max()+0.5, points[:, 0].min()-0.5
    
    percent = 1 / height
    return -percent, points[:, 1].max()+1, points[:, 1].min()-1, points[:, 0].max()+1, points[:, 0].min()-1


def adjust_xyz(x, y, z):
    return get_point_inside_cell(x), get_point_inside_cell(y), get_point_inside_cell(z)

def point_in_hull(poly, point):
    try:
        hull = ConvexHull(poly)
        new_hull = ConvexHull(np.concatenate((poly, [point])))
        return np.array_equal(new_hull.vertices, hull.vertices)
    except:
        # print("err ch_hull", len(poly), poly)
        # print(traceback.format_exc())
        return False
    

def has_prism(neighbors, non_layer_axes, n):
    neighbors =sorted(neighbors, key=lambda x: x[1])[:n]
    center = np.array(neighbors[0][2])
    points = [np.array(p[3]) for p in neighbors]
    
    stdev = -1
    if len(points) < 6:
        return False, -1
    
    inside = point_in_hull(points, center)
    
    if inside:
        lengths = []
        for i in range(n):
            lengths.append(np.linalg.norm(points[i][non_layer_axes]-center[non_layer_axes]))
            # for j in range(i+1, n):
            #     lengths.append(np.linalg.norm(points[i]-points[j]))

        lengths = sorted(lengths)
        # print(lengths)
        stdev = np.std(lengths[:n])

        avg = np.vstack([point for point in points]).mean(axis=0)
        # print(avg.shape)
        stdev = np.linalg.norm(avg-center)

    return inside, stdev


def find_most_suitable_prism(neighbors, non_layer_axes):
    prism_metrics = []
    n_max = int(len(neighbors)/2) + 1

    for n in range(3, n_max):
        prism, stdev = has_prism(neighbors, non_layer_axes, n*2)
        if prism:
            prism_metrics.append([n, round(float(stdev), 3)])

    if len(prism_metrics):
        prism_metrics = sorted(prism_metrics, key=lambda x: x[0], reverse=True)
        prism_metrics = sorted(prism_metrics, key=lambda x: x[1])
        print("\nPM", prism_metrics)
        return prism_metrics[0][0]
    return None


def CN_numbers_of_site(v):
    
    # Finds the coordination numbers using the d/d_min method.
    
    points_wd =[[p[3], p[1], p[0]] for p in v]
    
    # sort
    points_wd = sorted(points_wd, key=lambda x: x[1])[:30]
    distances = np.array([p[1] for p in points_wd])
    distances /= distances.min()

    gaps = np.array([distances[i] - distances[i-1] for i in range(1, len(distances))])
    ind_gaps = np.argsort(gaps)
    
    CN_values = np.array(ind_gaps[::-1]) + 1
    CN_values = CN_values[CN_values >= 4]
    
    return CN_values


def has_capped_atom(face_points, face_indices, capping_atoms, non_layer_axes):
    
    """
    This checks for the presence of capping atoms by checking for atoms outside of 
    the plane created by extending the face of the prisms.  When more than one 
    capping atoms are found, only atoms forming an angle greater than 45 degrees
    are considered as capping atoms.
    
    If all faces of the prism has capping, returns true.
    """

    face_points = np.array([face_points[i] for i in face_indices])
    
    t1, t2 = face_points[:2]
    t3 = face_points[2:].mean(axis=0)
    assert t3.shape[0] == 2, f"T3 Shape not correct, {t3.shape}"
    
    points = [p[-1][non_layer_axes] for p in capping_atoms]
    
    x1, y1 = t1
    x2, y2 = t2
    m = (y2 - y1) / (x2 - x1)  

    m = np.nan_to_num(m, nan=0.0, posinf=10., neginf=-10.0)
    b = y1 - m * x1
    b = np.nan_to_num(b, nan=0.0, posinf=10.0, neginf=-10.0)
    
    A = m
    B = -1
    C = b
    p_hat = lambda p: A*p[0] + B*p[1] + C
    
    third_point_side = p_hat(t3)
    num_points_opposite_to_third_vertex = 0
    points_opposite = []
    for i, point in enumerate(points):
        atom_side = p_hat(point)
        
        # third point and atom has to be opposite to each other
        if third_point_side < 0 and atom_side > 0:
            num_points_opposite_to_third_vertex += 1
            points_opposite.append(point)
        elif third_point_side > 0 and atom_side < 0:
            num_points_opposite_to_third_vertex += 1
            points_opposite.append(point)
    
    if num_points_opposite_to_third_vertex == 1:
        return True
    elif num_points_opposite_to_third_vertex > 1:
        critical_angle = 45
        num_points_above_crit_angle = 0
        
        for point in points_opposite:
            v1 = point - t1
            v2 = point - t2
            
            angle = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)), -1.0, 1.0)
            angle = (180/(22/7)) * np.arccos(angle)
            if angle >= critical_angle:
                num_points_above_crit_angle += 1
        return num_points_above_crit_angle == 1
    return False


def get_colors(site_symbol_map):
    
    colors = pd.read_csv('colors.csv')
    color_labels = dict(zip(colors['Element'].tolist(), colors['Color'].tolist()))
    
    reds = ['red', 'firebrick', 'darkred', 'orangered', 'crimson']
    blues = ['blue', 'darkblue', 'dodgerblue', 'royalblue', 'skyblue']
    greys = ['dimgrey', 'darkgrey', 'grey', 'silver', 'slategrey']
    pink = ['violet', 'darkviolet', 'magenta', 'purple', 'orchid']
    
    cpd_elements = list(set(list(site_symbol_map.values())))
    ir, ib, ig, iv = 0, 0, 0, 0
    cpd_colors = {}
    
    for el in cpd_elements:
        cl = color_labels[el]

        if cl == 'red':
            cpd_colors[el] = reds[ir]
            ir += 1
        elif cl == "blue":
            cpd_colors[el] = blues[ib]
            ib += 1
        elif cl == "grey":
            cpd_colors[el] = greys[ig]
            ig += 1
        elif cl == "pink":
            cpd_colors[el] = pink[ig]
            iv += 1
        else:
            print("COLOR ERROR", el, cl)
            
    return cpd_colors


def get_points(points_wd, center, layer_axis, N):
    points = []
    center_height = center[layer_axis]

    for p in points_wd:
        if abs(p[2][layer_axis] - center_height) <= 0.2:
            points.append(p)
            if len(points) == int(N/3):
                break

    for p in points_wd:
        if abs(p[2][layer_axis] - center_height) > 0.2:
            points.append(p)
            if len(points) == N:
                break

    return points


def get_data(cif_path, CN):
    
    # get capped prisms data for the cif.
    
    cif = Cif(cif_path)
    cif_d = cif_to_dict(cif_path)
    unitcell_points = cif.unitcell_points

    loop_vals = cif._loop_values
    site_symbol_map = dict(zip([l for l in loop_vals[0]], [s for s in loop_vals[1]]))
    unitcell_coordinates = np.array([c[:-1] for c in unitcell_points if site_symbol_map[c[-1]] != "H"])
    
    layer_vals, layer_index = None, None
    for i in range(3):
        if len(np.unique(np.abs(unitcell_coordinates[:, i]))) == 2:
            layer_vals = np.unique(unitcell_coordinates[:, i])
            layer_index = i
            
    ncnp_cif = {'layer_axis': layer_index, 'layer_vals': layer_vals, 'Formula': cif_d.get('_chemical_formula_sum', 'NA'), 
                'Structure Type': cif_d.get('_chemical_name_structure_type', 'NA')}

    # remove third layer
    if layer_index is not None:
        if len(layer_vals) == 3:
            unitcell_coordinates = unitcell_coordinates[unitcell_coordinates[:, layer_index] != layer_vals[0]]
    
        cif.compute_connections()
        conns = cif.connections
        
        ncnp_cif['non_layer_axes'] = np.array([0, 1, 2]) != layer_index
        
        site_data = {}
        i = 0
        for site, points_wd in conns.items():
            CN_vals = CN_numbers_of_site(points_wd)
            ncnp_cif["CNs"] = CN_vals
            # if not CN in CN_vals[:10] and not int(CN*2/3) in CN_vals[:10]:
            #     continue
            
            points_wd = sorted(points_wd, key=lambda x: x[0])  # make sorting consistent
            points_wd = sorted(points_wd, key=lambda x: x[1])[:CN]

            # points_wd = get_points(points_wd, center=)

            site_capped_prims_data = get_capped_prism_data(site=site,
                                        layer_axis=layer_index,
                                        points_wd=points_wd.copy(),
                                        CN=CN)
            if site_capped_prims_data['capped_prism_present']:
                site_capped_prims_data['center_element'] = site_symbol_map[site]
                site_capped_prims_data['coordination_formula'] = get_formula(points_wd, site_symbol_map)
                site_data[site] = site_capped_prims_data
    
        if site_data:
            ncnp_cif['site_data'] = site_data

    else:
        print("layer index is None")
    return ncnp_cif


def get_formula(points_wd, site_symbol_map):
    
    # Create formula for the prism and capping sites.
    
    symbols = [s[0] for s in points_wd]
    site_elements = defaultdict(int)
    for s in symbols:
        site_elements[site_symbol_map[s]] += 1
    site_elements = [[k, v] for k, v in site_elements.items()]
    site_elements = sorted(site_elements, key=lambda x: x[0], reverse=False)
    site_elements = sorted(site_elements, key=lambda x: x[1], reverse=True)
        
    formula = ""
    for e, c in site_elements:
        formula += f"{e}{'' if c==1 else c}"
    return formula