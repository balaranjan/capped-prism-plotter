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


def point_in_hull(poly, point):
    try:
        hull = ConvexHull(poly)
        new_hull = ConvexHull(np.concatenate((poly, [point])))
        return np.array_equal(new_hull.vertices, hull.vertices)
    except:
        # print("ConvexHull Error", traceback.format_exc())
        return False


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

    if len(capping_atoms) != num_capping_atoms:
        capped_prism_data['error'] = f"Num capping atoms are {len(capping_atoms)}"
        return capped_prism_data
    
    prism_atoms = [points_wd[c] for c in range(len(points_wd)) if c not in capping_inds]
    unique_heights = set([round(float(p[-1][layer_axis]), 1) for p in prism_atoms])
    m_unique_heights = []
    for unique_height in unique_heights:
        if np.any(abs(np.array(m_unique_heights) - unique_height) <= 0.5):
            continue
        m_unique_heights.append(unique_height)

    unique_heights = m_unique_heights
    
    if len(unique_heights) != 2:
        capped_prism_data["error"] = f"Num unique heights for triangle prism are {len(unique_heights)}, {unique_heights}"
        return capped_prism_data

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
    layer_1 = sorted(list(prism_sites_by_heights.keys()))[0]

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
        capped_prism_data["error"] = f"Number of atoms for prism is {len(prism)}"
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
    cap_atom_present = []
    for i, face in enumerate(faces):

        _has_capped_atom = has_capped_atom(face_points=face_points, 
                                           face_indices=face,
                                           capping_atoms=capping_atoms,
                                           non_layer_axes=non_layer_axes)
        cap_atom_present.append(_has_capped_atom)
      
    if all(cap_atom_present):
        capped_prism_data['capped_prism_present'] = True
        capped_prism_data['prism'] = face_points
        capped_prism_data['edges'] = edges
        capped_prism_data['caps'] = capping_atoms

    capped_prism_data['cap_for_faces'] = cap_atom_present
    if CN == 7:
        print(capped_prism_data)
    return capped_prism_data


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
            if not CN in CN_vals[:10]:
                continue
            
            points_wd = sorted(points_wd, key=lambda x: x[0])  # make sorting consistent
            points_wd = sorted(points_wd, key=lambda x: x[1])[:CN]

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
    return ncnp_cif


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


def get_sg_symbol(cif_path):
    # print(cif_path)
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
        # print(cif_path)
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


def plot_supercell(cif_path, CNs, ms=None, lw=None, label=False, 
                   fix_min_length=None, flip_layers=False, 
                   sites_to_plot=None, multiple=False,
                   plot_individual=False):
    
    # plot the capped prisms in a 3x3 super cell
    if lw is None:
        lw = 1
        
    if sites_to_plot is not None:
        sites_to_plot = sites_to_plot.split(',')
    
    any_CN_plotted = False
    cn_color = {9: 'tab:red', 12: 'tab:green', 15: 'tab:blue', 18: 'tab:purple', 21: 'tab:orange'}
    
    cif_data = get_data(cif_path, CN=CNs[0])
    title = cif_data['Structure Type']
    layer_axis = cif_data['layer_axis']
    layer_vals = cif_data['layer_vals']
    non_layer_axes = cif_data['non_layer_axes']
        
    cif = Cif(cif_path)
    unitcell_lengths = cif.unitcell_lengths
    num_supercell_atoms = int(len(cif.supercell_points)/3)
    
    non_layer_lengths = [unitcell_lengths[l] for l in range(3) if l != layer_axis]
    
    plt.close()
    
    if fix_min_length:
        non_layer_lengths = np.array(non_layer_lengths)
        non_layer_lengths /= non_layer_lengths.min()
        non_layer_lengths *= fix_min_length

        if ms is None:
            ms = max(5, int(316.27*num_supercell_atoms**(-0.596)))
    else:
        ms = 20
    
    fig = plt.figure()
    fig.set_size_inches(non_layer_lengths)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    
    for CN in CNs:
        cif_data = get_data(cif_path, CN=CN)
        if cif_data is None:
            continue
        
        cif = Cif(cif_path)
        unitcell_lengths = cif.unitcell_lengths
        unitcell_angles = cif.unitcell_angles
        supercell_points = cif.supercell_points
        unitcell_points = cif.unitcell_points
        
        for i in range(len(supercell_points)):
            s = supercell_points[i][-1]
            cart = unit.fractional_to_cartesian(supercell_points[i][:3],
                                                unitcell_lengths,
                                                unitcell_angles)
            supercell_points[i] = [np.array(cart), s]

        for i in range(len(unitcell_points)):
            s = unitcell_points[i][-1]
            cart = unit.fractional_to_cartesian(unitcell_points[i][:3],
                                                unitcell_lengths,
                                                unitcell_angles)
            unitcell_points[i] = [np.array(cart), s]

        supercell_layer_heights = list(set([sround(float(p[0][layer_axis])) for p in supercell_points]))
        supercell_layer_heights = sorted(list(set(supercell_layer_heights)))
        # print(set([p[0][2] for p in supercell_points]))
        # assert len(supercell_layer_heights) == 11, f"Number of super cell heights != 6, {supercell_layer_heights}, layer axis is {layer_axis}"
        
        supercell_layer_height_map = {supercell_layer_heights[i]: i % 2 for i in range(len(supercell_layer_heights))}
        
        loop_vals = cif._loop_values
        site_symbol_map = dict(zip([l for l in loop_vals[0]], [s for s in loop_vals[1]]))
        
        tctp_cif = get_data(cif_path=cif_path, CN=CN)
        layer_axis = tctp_cif['layer_axis']
        non_layer_axes = tctp_cif['non_layer_axes']
        
        # gather data for plot
        capped_prism_data_by_layers = {0: [], 1: []}
        if 'site_data' not in cif_data:
            continue
        
        for k, v in cif_data['site_data'].items():
            if sites_to_plot and k not in sites_to_plot:
                continue
            atoms_with_this_site = [p for p in supercell_points if p[-1]==k]  #Change
            site_formula = v['coordination_formula']
            center_coordinate = np.array(v["center_coordinate"])

            for atom in atoms_with_this_site:
                points_wd = get_first_n_neighbors(atom, supercell_points, n=CN)
                formula = get_formula(points_wd, site_symbol_map)

                if formula != site_formula:
                    continue

                atom_capped_prims_data = get_capped_prism_data(site=atom[1], 
                                            layer_axis=layer_axis,
                                            points_wd=points_wd,
                                            CN=CN)
                
                if atom_capped_prims_data['capped_prism_present']:
                    center_coordinate = atom_capped_prims_data["center_coordinate"]
                    prism = atom_capped_prims_data['prism']
                    caps = atom_capped_prims_data['caps']
                    edges = atom_capped_prims_data['edges']
                    
                    key = sround(float(atom[0][layer_axis]))
                    key = supercell_layer_height_map[key]
                    capped_prism_data_by_layers[key].append([center_coordinate, prism, edges, caps])

        capped_prism_data_by_layers = [[k, v] for k, v in capped_prism_data_by_layers.items()]
        capped_prism_data_by_layers = sorted(capped_prism_data_by_layers, key=lambda x: float(x[0]))
        
        assert len(capped_prism_data_by_layers) >= 2, "Less than two layers in the supercell!"

        tctp_counts = []
        for i in range(1, len(capped_prism_data_by_layers), 2):
            tctp_counts.append([i, len(capped_prism_data_by_layers[i-1][1]) + len(capped_prism_data_by_layers[i][1])])
        
        tctp_counts = sorted(tctp_counts, key=lambda x: x[1], reverse=True)

        if not len(tctp_counts):
            print(f"Less than two layers, {tctp_counts} {len(capped_prism_data_by_layers)}")
            return False
        selected_layers = [capped_prism_data_by_layers[tctp_counts[0][0]-1], 
                        capped_prism_data_by_layers[tctp_counts[0][0]]]
        
        selected_layers = sorted(selected_layers, key=lambda x: float(x[0]), reverse=True)
        element_colors = get_colors(site_symbol_map)

        plotted_prisms = []
        cn_col = cn_color.get(CN)
        any_CN_plotted = True
        
        for i, layer in capped_prism_data_by_layers:
            if i == 0:
                for center_coordinate, prism, edges, caps in layer:
                    prism = prism[np.argsort(prism[:, 0])]
                    prism = prism[np.argsort(prism[:, 1])]
                    if any([np.all(np.allclose(prism, t, rtol=5e-02)) for t in plotted_prisms]):
                        continue
                    
                    hull = ConvexHull(prism)
                    edges = []
                    for simplex in hull.simplices:
                        edges.append([int(simplex[0]), int(simplex[1])])
                        
                    if flip_layers:
                        for inds in edges:
                            plt.plot([prism[inds[0]][0], prism[inds[1]][0]], [prism[inds[0]][1], prism[inds[1]][1]], c=cn_col, lw=lw, alpha=0.7)
                            plotted_prisms.append(prism)
                    else:
                        # order edges
                        ordered_edges = [edges[0]]
                        added = [0]

                        for _ in range(int(CN/3) - 1):
                            last_i = ordered_edges[-1][-1]

                            sel_i = [i for i in range(int(CN/3)) if i not in added and last_i in edges[i]][0]
                            if edges[sel_i][0] == last_i:
                                ordered_edges.append(edges[sel_i])
                            else:
                                ordered_edges.append([edges[sel_i][1], edges[sel_i][0]])
                            added.append(sel_i)
                        
                        edges = ordered_edges
                        
                        tx, ty = [], []
                        for inds in edges:
                            tx.extend([prism[inds[0]][0], prism[inds[1]][0]])
                            ty.extend([prism[inds[0]][1], prism[inds[1]][1]])

                        plt.fill(tx, ty, color=cn_col, alpha=0.5, edgecolor=cn_col)
                        plotted_prisms.append(prism)
                    
            if i == 1:
                for center_coordinate, prism, edges, caps in layer:
                    prism = prism[np.argsort(prism[:, 0])]
                    prism = prism[np.argsort(prism[:, 1])]
                    if any([np.all(np.allclose(prism, t, rtol=5e-02)) for t in plotted_prisms]):
                        continue
                    
                    hull = ConvexHull(prism)
                    edges = []
                    for simplex in hull.simplices:
                        edges.append([int(simplex[0]), int(simplex[1])])
                    
                    if not flip_layers:
                        for inds in edges:
                            plt.plot([prism[inds[0]][0], prism[inds[1]][0]], [prism[inds[0]][1], prism[inds[1]][1]], c=cn_col, lw=lw, alpha=0.7)
                        plotted_prisms.append(prism)
                    else:
                        # order edges
                        ordered_edges = [edges[0]]
                        added = [0]

                        for _ in range(int(CN/3) - 1):
                            last_i = ordered_edges[-1][-1]

                            sel_i = [i for i in range(int(CN/3)) if i not in added and last_i in edges[i]][0]
                            if edges[sel_i][0] == last_i:
                                ordered_edges.append(edges[sel_i])
                            else:
                                ordered_edges.append([edges[sel_i][1], edges[sel_i][0]])
                            added.append(sel_i)
                        
                        edges = ordered_edges
                        
                        tx, ty = [], []
                        for inds in edges:
                            tx.extend([prism[inds[0]][0], prism[inds[1]][0]])
                            ty.extend([prism[inds[0]][1], prism[inds[1]][1]])

                        plt.fill(tx, ty, color=cn_col, alpha=0.5, edgecolor=cn_col)
                        plotted_prisms.append(prism)
        
    # cell
    if any_CN_plotted:
        unitcell_lengths = np.array(unitcell_lengths)[non_layer_axes]
        x1 = unitcell_lengths[1] * np.cos(unitcell_angles[layer_axis])
        y1 = unitcell_lengths[1] * np.sin(unitcell_angles[layer_axis])
        
        plt.plot([0., unitcell_lengths[0]], [0., 0.], c='k', alpha=0.5, lw=lw)
        plt.plot([0., x1], [0., y1], c='k', alpha=0.5, lw=lw)
        plt.plot([x1, x1+unitcell_lengths[0]], [y1, y1], c='k', alpha=0.5, lw=lw)
        plt.plot([unitcell_lengths[0], x1+unitcell_lengths[0]], [0., y1], c='k', alpha=0.5, lw=lw)

        legends = []
        layer_heights = sorted(supercell_layer_heights[2:4])
        markers = ['.', 'o']
        if flip_layers:
            markers = ['o', '.']
        
        supercell_points_tl = [p for p in supercell_points if sround(float(p[0][layer_axis])) in supercell_layer_heights[2:4]]  # CHANGE
        print(layer_heights)
        o_marker_offset = 25
        if fix_min_length:
            o_marker_offset = int(o_marker_offset/(num_supercell_atoms/10))
        for p, s in supercell_points_tl:
            sy = site_symbol_map[s]
            lh = p[layer_axis]
            m = markers[int(np.argmin(np.abs(layer_heights - lh)))]
            
            size = ms if m != "." else int(ms*2)

            p = p[non_layer_axes]

            if sy not in legends:
                legends.append(sy)
                plt.scatter(p[0], None, c=element_colors[sy], label=sy, s=ms+25, marker='.')
            
            if m == 'o':
                plt.scatter(p[0], p[1], edgecolor=element_colors[sy], s=size, marker=m, facecolor='none')
            else:
                plt.scatter(p[0], p[1], c=element_colors[sy], s=size, marker=m)
                
            if label:
                plt.text(p[0]-(p[0]*0.025), p[1]-(p[1]*0.035), s, size=6)
        
        assert set(legends) == set(site_symbol_map.values()), f"{set(legends)}"
        bby, ymax, ymin, xmax, xmin = get_bbox_y(supercell_points_tl, non_layer_axes, fix_min_length)
        plt.ylim(ymin, ymax)
        plt.xlim(xmin, xmax)

        plt.legend(ncol=len(legends),  bbox_to_anchor=(0.5, bby), framealpha=0, loc="lower center", fontsize=16, columnspacing=0.1)
        plt.title(format_formula(title.split(',')[0].replace("~", "")) + "-type " + get_sg_symbol(cif_path), y=1.01, size=16)
        plt.axis('off')
        plt.tight_layout()
        
        if plot_individual:
            dirname = f"output_images{os.sep}{title.replace(',', '-')} {cif_path.split(os.sep)[-1][:-4]}"
            if not os.path.isdir(dirname):
                os.mkdir(dirname)
            print(dirname)
            plt.savefig(f"{dirname}{os.sep}{CN}.png", dpi=300, transparent=True, bbox_inches='tight')
        else:
            print(f"Saving output_images/{title.replace(',', '-')} {cif_path.split(os.sep)[-1][:-4]}.png")
            plt.savefig(f"output_images/{title.replace(',', '-')} {cif_path.split(os.sep)[-1][:-4]}.png", dpi=300, transparent=True, bbox_inches='tight')

        return True
    return False


def sround(val):
    
    val = round(val, 3)
    val = round(val, 2)
    val = round(val, 1)
    
    return val


def get_bbox_y(supercell_points, non_layer_axes, fix_min_length=None):
    
    points = np.vstack([p[0][non_layer_axes] for p in supercell_points])
    height = points[:, 1].max() - points[:, 1].min()
    
    if fix_min_length:
        percent = 0.2
        return -percent, points[:, 1].max()+0.5, points[:, 1].min()-0.5, points[:, 0].max()+0.5, points[:, 0].min()-0.5
    
    percent = 1 / height
    return -percent, points[:, 1].max()+1, points[:, 1].min()-1, points[:, 0].max()+1, points[:, 0].min()-1


if __name__ == "__main__":
    import multiprocessing as mp
    import argparse
    
    parser = argparse.ArgumentParser(
        prog=__file__.split(os.sep)[-1],
        description="Script to plot capped-prisms in bi-layered compounds.",
        epilog="Output images will be (over)written inside images folder in the current directory. \
                CIFs encountering errors will be written to errors.txt")
    
    parser.add_argument('path', help="path to CIF or folder containing multiple CIFs")
    parser.add_argument('-p', '--prisms', help="set the numbers for prisms \
                        (3 for trigonal, 4 for square, 5 for pentagonal, etc.) eg. 3,4 (default)", 
                        type=str, metavar='')
    parser.add_argument('-f', '--fix_length', help="fix the length of shortest side", type=float, metavar='')
    parser.add_argument('-m', '--marker_size', help="set marker size eg. 1", type=int, metavar='')
    parser.add_argument('-w', '--line_width', help="set line width eg. 3.0", type=int, metavar='')
    parser.add_argument('-s', '--sites', help="select sites to plot eg. Si1 Al1", type=str, metavar='')
    parser.add_argument('-i', '--invert', action='store_true', help="invert the layers")
    parser.add_argument('-l', '--label', action='store_true', help="label the sites")
    parser.add_argument('-n', '--no_overlay', action='store_true', help="create individual plots", default=False)
    
    args = parser.parse_args()
    print(f"file: {args.path}, invert {args.invert}, side_length {args.fix_length}, marker_size {args.marker_size} \
        line_width {args.line_width} sites {args.sites} cns {args.prisms}")  
    
    if not os.path.isdir("output_images"):
        os.mkdir("output_images")
    
    if args.prisms is not None:
        try:
            cns = [int(c)*3 for c in args.prisms.split(',')]
        except:
            print(f"Invalid value for prisms. \nEnter intergers from 3-7, separated by comma. e.g. 3,5")
            exit(0)
    else:
        cns = [9, 12]
    cns = sorted(cns, reverse=True)
    if os.path.isfile(args.path) and args.path.endswith('cif'):
        if args.no_overlay:
            for cn in cns:
                plot_supercell(cif_path=args.path, 
                                CNs=[cn],
                                flip_layers=args.invert,
                                label=args.label,
                                fix_min_length=args.fix_length,
                                ms=args.marker_size,
                                lw=args.line_width,
                                sites_to_plot=args.sites,
                                multiple=False, plot_individual=True)
        else:
            plot_supercell(cif_path=args.path, 
                            CNs=cns,
                            flip_layers=args.invert,
                            label=args.label,
                            fix_min_length=args.fix_length,
                            ms=args.marker_size,
                            lw=args.line_width,
                            sites_to_plot=args.sites,
                            multiple=False, plot_individual=False)
    
    elif os.path.isdir(args.path):
        cifs = [os.path.join(args.path, f) for f in os.listdir(args.path) if f.endswith('.cif')]
        print(cifs)
        total = len(cifs)
        errors = []
        
        for i, cif in enumerate(cifs, 1):
            print(f"{i} of {total}:  {cif}")
            try:
                success = plot_supercell(cif_path=cif, 
                                CNs=cns,
                                flip_layers=args.invert,
                                label=args.label,
                                fix_min_length=args.fix_length,
                                ms=args.marker_size,
                                lw=args.line_width,
                                sites_to_plot=args.sites,
                                multiple=True, plot_individual=True)
                if not success:
                    errors.append(cif)
            except:
                errors.append(cif)
                print(traceback.format_exc())
                
        with open('output_images/errors.txt', 'w') as f:
            for e in errors:
                f.write(f"{e}\n")
    else:
        print("Unknown option for path", args.path)
    exit(0)
    
    # root = "/home/bala/Documents/data/not_prototype_CIFs/"
    # porder = {'complete structure determined': 1,
    #         'cell parameters determined and structure type assigned': 2,
    #         'cell parameters determined and type with fixed coordinates assigned': 3,
    #         'part of atom coordinates determined': 4,
    #         'commensurate approximant determined' 'average structure determined': 5}
    
    # df = pd.read_csv('layered_im_same.csv')
    # print(len(df['Structure Type'].unique()))
    # # exit(0)
    # df = df[df['CNs'] != "[]"]
    # df['sgs'] = df['cif'].map(lambda x: get_sg_symbol(f"{root}{x}"))
    # df['porder'] = df['level'].map(lambda x: porder.get(x, 6))
    # sgd = pd.read_csv('sgs.csv')
    # sgd = dict(zip(sgd['sgnsy'].tolist(), [s.replace("$", "") for s in sgd['latex'].tolist()]))
    
    
    # df['sgs'] = df['sgs'].map(lambda x: sgd.get(x.replace('$', ""), x))
    
    # c = 0
    # errors = []
    # t = len(df)
    # skip_untill = 0
    
    # stypes = sorted(df['Structure Type'].unique().tolist())
    # t = len(stypes)
    # cwd = os.getcwd()
    # cns = [9, 12, 15, 18, 21]
    # # cns = cns[::-1]
    
    # working = pd.read_csv("working_cifs.csv")['filename'].to_list()
    # working = [f"{fn[:-4]}.cif" for fn in working]
    
    # tasks = []
    
    # def worker(error, **kwargs):
    #     # try:
    #     plot_supercell(**kwargs)
    #     # except:
    #     #     error.append(f"Error processing {r['cif']} {r['Structure Type']}\n")
    #     # try:
    #     #     for cn in [9, 12, 15, 18, 21]:
    #     #         kwargs['CNs'] = [cn]
    #     #         # multiple=False, plot_individual=True
    #     #         kwargs['multiple'] = True
    #     #         kwargs['plot_individual'] = False
    #     #         plot_supercell(**kwargs)
    #     # except:
    #     #     error.append(f"Error processing {r['cif']} {r['Structure Type']}\n")
            
    # def mp_aux(*args):
    #     for kwarg in args:
    #         error = kwarg.pop('error')
    #         worker(error, **kwarg)

    # manager = mp.Manager()
    # error = manager.list()
    
    # for i, st in enumerate(stypes, 1):
    #     print(f"{i} of {t} {st}")
        
    #     if i <= skip_untill:
    #         continue
    
    #     tdf = df[df['Structure Type'] == st]
    #     tdf.sort_values('porder', inplace=True, ascending=True)

    #     CNs = str(tdf.iloc[0]['CNs']).replace("\n", "")
    #     CNs = [int(c) for c in CNs[1:-1].split()]
        
    #     if not len(set(cns).intersection(set(CNs))):
    #         continue
        
    #     str_plotted = False
    #     for _, r in tdf.iterrows():
    #         # if r['cif'] not in working:
    #         #     continue
    #         tasks.append({'error': error, 'cif_path': f"{root}{r['cif']}",
    #                       "CNs": cns, "multiple": True, "fix_min_length": 5})
    # print(len(tasks))
    # with mp.Pool(16) as p:
    #     p.map(mp_aux, tasks)
        
    # p.close()
    # p.join()
                
    # errors = list(errors)
    # with open('err.txt', 'w') as f:
    #     for e in errors:
    #         f.write(e)
    # exit(0)
    
    