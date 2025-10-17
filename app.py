import streamlit as st
from tempfile import NamedTemporaryFile
from cifkit import Cif
from utils import get_data, get_colors, adjust_xyz, get_first_n_neighbors, get_formula
from utils import sround, format_formula, get_sg_symbol
from utils import get_bbox_y, rearrange_coordinates, is_close, has_prism, find_most_suitable_prism
from utils import get_capped_prism_data, point_in_hull, get_prism_metrics
import os
import traceback
import numpy as np
import pandas as pd
from cifkit import Cif
from cifkit.utils import unit
from collections import defaultdict
from scipy.spatial import ConvexHull
from matplotlib import pyplot as plt
# from cif_parser import _parse_formula, cif_to_dict
from matplotlib.colors import to_hex
import io
# from shapely.geometry import Polygon
from functools import cmp_to_key

st.set_page_config(layout="wide")

REs = ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Eu', 'Gd', 'Tb', 
       'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Sc', 'Y',

        'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 
        'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',

       'Ti', 'Hf', 'Zr',
       'Li', 'Na', 'K', 'Rb', 'Cs',
       'Be', 'Mg', 'Ca', 'Sr', 'Ba',
       ]
REs = set(REs)

fig = None
# left_col, right_col = st.columns([1, 3])

st.set_page_config(page_title="Prism Plotter")
st.markdown(
    "<h1 style='text-align: center;'>Prism Plotter</h1>",
    unsafe_allow_html=True
)
top_cols = st.columns([1, 1])

# with top_cols[0]:
#     ce = st.number_input(
#                 label="Cell edge width:",
#                 min_value=1,
#                 max_value=10,
#                 value=3,   # Default value
#                 step=1
#             )

#     lw = st.number_input(
#         label="Line width:",
#         min_value=1,
#         max_value=10,
#         value=3,   # Default value
#         step=1
#     )

#     ms = st.number_input(
#             label="Marker size:",
#             min_value=1,
#             max_value=1000,
#             value=300,   # Default value
#             step=1
#         )

# plot_options = top_cols[0].columns(2)
# separate_atoms_by_layer = plot_options[0].toggle("Separate atoms by layer", value=False)
# shade_prisms = plot_options[1].toggle("Shade prisms", value=True)
# label_options = top_cols[0].columns(2)
# show_labels = label_options[0].toggle("Show labels", value=False)
# swap_layers = label_options[1].toggle("Swap layers", value=False)

# top_cols[2].write("Prisms")
# env_cols_1 = top_cols[2].columns(6)
# env_cols_1[0].write("All")

# env_3 = env_cols_1[1].toggle("3", value=True, key='all3')
# env_4 = env_cols_1[2].toggle("4", value=True, key='all4')
# env_5 = env_cols_1[3].toggle("5", value=True, key='all5')
# env_6 = env_cols_1[4].toggle("6", value=True, key='all6')
# env_7 = env_cols_1[5].toggle("7", value=True, key='all7')

# allowed_envs = []
# if env_3:
#     allowed_envs.append(3*2)
# if env_4:
#     allowed_envs.append(4*2)
# if env_5:
#     allowed_envs.append(5*2)
# if env_6:
#     allowed_envs.append(6*2)
# if env_7:
#     allowed_envs.append(7*2)


def overlap_present(site_neighbors_ud, prism, non_layer_axes, plot=False):
    prism = sorted(prism, key=lambda x: x[1])

    site_neighbors_ud = sorted(site_neighbors_ud, key=lambda x: x[1])
    site_neighbors_ud = np.vstack([p[-1][non_layer_axes] for p in site_neighbors_ud])

    commom_vertices = [int((np.linalg.norm(site_neighbors_ud - p[-1][non_layer_axes],axis=1) < 0.1).astype(int).sum()) for p in prism]

    return sum(commom_vertices) > 8

def get_non_overlapping_prism(site_neighbors_ud, site_prism_metrics, selected_prisms):

    def custom_cmp(a, b):
        if abs(b[1] - a[1]) <= 0.03:
            res = -1 if (a[0] - b[0]) > 0 else 1
        else:
            res = a[1] - b[1]
        return res

    
    site_prism_metrics = sorted(site_prism_metrics, key=cmp_to_key(custom_cmp))
    if not len(selected_prisms):
        return site_prism_metrics[0][0]
    
    selected = None
    for n, _ in site_prism_metrics:
        overlap = []
        for k, prism in selected_prisms.items():
            overlap.append(overlap_present(site_neighbors_ud[:n*2], prism, non_layer_axes, plot=k=='Ir1'))
        if not any(overlap):
            return n
    return selected

with top_cols[0]:
    uploaded_file = st.file_uploader("Choose a file")

top_cols[0].write("Plot Options")
plot_options_container = top_cols[0].container(border=True)
plot_options = plot_options_container.columns(2)
separate_atoms_by_layer = plot_options[0].toggle("Separate atoms by layer", value=False)
shade_prisms = plot_options[1].toggle("Shade prisms", value=True)
label_options = plot_options_container.columns(2)
show_labels = label_options[0].toggle("Show labels", value=False)
swap_layers = label_options[1].toggle("Swap layers", value=False)

with plot_options_container:
    ce = st.number_input(
                label="Cell edge width:",
                min_value=1,
                max_value=10,
                value=3,   # Default value
                step=1
            )

    lw = st.number_input(
        label="Line width:",
        min_value=1,
        max_value=10,
        value=3,   # Default value
        step=1
    )

    ms = st.number_input(
            label="Marker size:",
            min_value=1,
            max_value=1000,
            value=300,   # Default value
            step=1
        )

if uploaded_file is not None:
    file_name = uploaded_file.name
    with NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

        cif_name = uploaded_file.name[:-4]
        print("\n\n", "NEW", cif_name)
        
        cif = Cif(temp_file_path)
        cif.compute_connections()
        cif.compute_CN()
        unitcell_lengths = cif.unitcell_lengths
        unitcell_angles = cif.unitcell_angles
        unitcell_points = cif.unitcell_points
        supercell_points = cif.supercell_points

        loop_vals = cif._loop_values
        site_symbol_map = dict(zip([l for l in loop_vals[0]], [s for s in loop_vals[1]]))
        element_colors = get_colors(site_symbol_map)
        elements = list(set(list(site_symbol_map.values())))

        top_cols[1].write("Colors ")
        color_container = top_cols[1].container(border=True)
        cols = color_container.columns(len(elements)+1)
        color_cols = color_container.columns(len(elements)+1)
        
        for i, e in enumerate(elements, 0):
            if i > 0 and i % 3 == 0:
                color_cols = top_cols[1].columns(3)
            current_color = to_hex(element_colors[e])
            color = color_cols[i % 3].color_picker(f'{e}', current_color)
            element_colors[e] = color

        top_cols[1].write("Prisms")
        container = top_cols[1].container(border=True)
        env_cols_1 = container.columns(6)
        env_cols_1[0].write("All sites")

        env_3 = env_cols_1[1].toggle("3", value=True, key='all3')
        env_4 = env_cols_1[2].toggle("4", value=True, key='all4')
        env_5 = env_cols_1[3].toggle("5", value=True, key='all5')
        env_6 = env_cols_1[4].toggle("6", value=True, key='all6')
        env_7 = env_cols_1[5].toggle("7", value=True, key='all7')

        allowed_envs = []
        if env_3:
            allowed_envs.append(3*2)
        if env_4:
            allowed_envs.append(4*2)
        if env_5:
            allowed_envs.append(5*2)
        if env_6:
            allowed_envs.append(6*2)
        if env_7:
            allowed_envs.append(7*2)

        cif_data_1 = get_data(temp_file_path, CN=9)
        title = cif_data_1['Structure Type']
        layer_axis = cif_data_1['layer_axis']
        # layer_vals = cif_data_1['layer_vals']

        non_layer_axes = cif_data_1['non_layer_axes']
        non_layer_lengths = [unitcell_lengths[l] for l in range(3) if l != layer_axis]


        layer_vals = set()
        for i in range(len(unitcell_points)):
            x, y, z, l = unitcell_points[i]
            x, y, z = adjust_xyz(x, y, z)

            cart = unit.fractional_to_cartesian([x, y, z],
                                                unitcell_lengths,
                                                unitcell_angles)
            unitcell_points[i] = (*cart, l)
            layer_vals.add(sround(float(cart[layer_axis])))

        for i in range(len(supercell_points)):
            s = supercell_points[i][-1]
            cart = unit.fractional_to_cartesian(supercell_points[i][:3],
                                                unitcell_lengths,
                                                unitcell_angles)
            supercell_points[i] = [np.array(cart), s]

        # Plots
        cell_height = unitcell_lengths[layer_axis]
        unitcell_lengths = np.array(unitcell_lengths)[non_layer_axes]
        x1 = unitcell_lengths[1] * np.cos(unitcell_angles[layer_axis])
        y1 = unitcell_lengths[1] * np.sin(unitcell_angles[layer_axis])

        unitcell_layer_heights = set([sround(p[layer_axis]) for p in unitcell_points])
        layer_heights = np.array(sorted(unitcell_layer_heights)[:2], dtype=float)
        markers = ['.', 'o']

        capped_prism_data_by_layers = defaultdict(list)
        tol = 0.05
        cell = [
            [0.-unitcell_lengths[0]*tol, 0.-unitcell_lengths[1]*tol],
            [unitcell_lengths[0]+unitcell_lengths[0]*tol, 0.-unitcell_lengths[1]*tol],
            [x1-unitcell_lengths[0]*tol, y1+unitcell_lengths[1]*tol],
            [x1+unitcell_lengths[0]+unitcell_lengths[0]*tol, y1+unitcell_lengths[1]*tol],
        ]


        # TCTPs
        prism_and_cap_xys = []
        sites_with_envs = set()
        conns = cif.connections


            
        prism_info_by_site = {}
        prisms_in_unitcell = {}
        prism_metrics = []
        atoms_in_middle = [atom for atom in supercell_points if -0.1 <= atom[0][layer_axis] <= cell_height+0.1]

        for site, neighbors in conns.items():
            neighbors = sorted(neighbors, key=lambda x: x[1])[:22]
            neighbors_ud = [neighbor for neighbor in neighbors if cell_height-0.1 > abs(neighbor[3][layer_axis]-neighbor[2][layer_axis]) >= 0.2]
            neighbors_c_2d = [np.array(neighbor[3])[non_layer_axes] for neighbor in neighbors if abs(neighbor[3][layer_axis])-abs(neighbor[2][layer_axis]) < 0.2]
            neighbors_c = [neighbor for neighbor in neighbors if abs(neighbor[3][layer_axis])-abs(neighbor[2][layer_axis]) < 0.2]
            
            center = np.array(neighbors[0][2])[non_layer_axes]
            middle_layer_neighbor_dist_2D = [np.linalg.norm(center - np.array(p[-1])[non_layer_axes]) for p in neighbors_c]
            middle_layer_neighbor_dist_2D = [d for d in middle_layer_neighbor_dist_2D if d > 0.1]
            min_middle_layer_neighbor_dist_2D = -1
            if len(middle_layer_neighbor_dist_2D):
                min_middle_layer_neighbor_dist_2D = min(middle_layer_neighbor_dist_2D)

            neighbors_with_one_center = None
            
            for n in range(6, len(neighbors_ud)+1, 2):
                poly_2d = []
                for neighbor in neighbors_ud[:n]:
                    neighbor = np.array(neighbor[3])[non_layer_axes]
                    
                    if not len(poly_2d):
                        poly_2d.append(neighbor)

                    elif not any([np.allclose(p, neighbor, atol=1e-2, rtol=1e-2) for p in poly_2d]):
                        poly_2d.append(neighbor)

                poly_2d = poly_2d[:int(n/2)]
                center_layer = [c for c in neighbors_c_2d if point_in_hull(poly_2d, c)]
                ud_layers = [p for p in poly_2d if point_in_hull(poly_2d, p)]

                if len(poly_2d) == len(ConvexHull(poly_2d).vertices) and not len(center_layer):
                    neighbors_with_one_center = n

            neighbors_ud = neighbors_ud[:neighbors_with_one_center]

            pm = get_prism_metrics(neighbors_ud, non_layer_axes)
            # print(site, pm, [np.array(p[-1])[non_layer_axes] for p in neighbors_ud])
    
            unique_2Ds = []
            neighbors_ud_sorted = []

            for neighbor in neighbors_ud:
                c2d = np.array(neighbor[-1])[non_layer_axes]

                if len(unique_2Ds):
                    if any(np.linalg.norm(np.vstack(unique_2Ds)-c2d, axis=1) <= 1e-3):
                        continue

                neighbors_ud_sorted.append(neighbor)
                neighbors_ud_sorted.append(neighbor)
                unique_2Ds.append(c2d)

            prism_metrics.append([site, pm, neighbors_ud_sorted])
        prism_metrics = sorted(prism_metrics, key=lambda x: min([v[1] for v in x[1]]))
            

        selected_prisms = {}
        c = 0

        top_cols[1].markdown("Site Selection  \nn of prism (distance between center site coordinate and the center of prism)")
        sites_container = top_cols[1].container(border=True)
        for site, pm, neighbors_ud in prism_metrics:

            info = {v[0]: f"{v[0]} ({v[1]})" for v in pm}
            def inline_label(opt):
                return f"{info[opt]}"
            
            site_neighbors_ud = []
            for i in range(0, len(neighbors_ud)):
                site_neighbors_ud.append([*neighbors_ud[i][:2], np.array(neighbors_ud[i][3])])

            selected = get_non_overlapping_prism(site_neighbors_ud, pm, selected_prisms)
            
            if len(pm) > 1:
                container = sites_container.container(border=True)
                env_cols_2 = container.columns([1, 5])
                env_cols_2[0].write(f"Site {site}")
                values = sorted([v[0] for v in pm])
                selected = env_cols_2[1].radio(
                    f"Site {site}",
                    values,
                    index=values.index(selected),
                    horizontal=True,
                    format_func=inline_label,
                    key=f"{site}-{c}-{cif_name}",
                    label_visibility="collapsed"
                )
                c += 1
            

            if selected:
                n_prism = selected
                prism = []
                # for i in range(0, len(neighbors_ud)):
                #     prism.append([*neighbors_ud[i][:2], np.array(neighbors_ud[i][3])])
                # prism = prism[:n_prism*2]
                # print(site, selected, [p[-1][non_layer_axes] for p in prism])
                selected_prisms[site] = prism
            
                prism = []
                for i in range(0, len(neighbors_ud)):
                    prism.append([np.array(neighbors_ud[i][3]), neighbors_ud[i][:2]])
                prism = prism[:n_prism*2]

                prism = sorted(prism, key=lambda x: x[0][layer_axis])[:n_prism]
                # print(site, selected, [p[0][non_layer_axes] for p in prism])
                prism = [[p[0][non_layer_axes], p[1]] for p in prism]
                prism_info_by_site[site] = [sround(neighbors[0][2][layer_axis]), prism]

                atoms_of_this_site = [atom for atom in atoms_in_middle if atom[-1]==site]
                prisms = []
                prism_sd = defaultdict(set)

                for vertex in prism:
                    prism_sd[vertex[1][0]].add(sround(vertex[1][1]))

                envs_in_extended_cell = []
                uc_layer_heights = set()
                center_heights = []

                for atom in atoms_of_this_site:
                    
                    a_neighbors = get_first_n_neighbors(atom[:-1], supercell_points, 22)
                    a_neighbors = sorted(a_neighbors, key=lambda x: x[1])
                    a_neighbors_ud = [neighbor for neighbor in a_neighbors if cell_height-0.1 > abs(neighbor[3][layer_axis]-neighbor[2][layer_axis]) >= 0.2]
                    # a_prism = []
                    # asites = []

                    # for an in a_neighbors:
                    #     if an[0] in prism_sd:
                    #         # if sround(an[1]) in prism_sd[an[0]]:
                    #         if any([abs(an[1] - v) < 0.1 for v in prism_sd[an[0]]]):
                    #             a_prism.append(np.array(an[-1]))
                    #             asites.append(an[:2])
                    # print(a_prism)
                    # a_prism = a_prism[:n_prism*2]

                    unique_2Ds = []
                    neighbors_ud_sorted = []

                    for neighbor in a_neighbors_ud:
                        c2d = np.array(neighbor[-1])[non_layer_axes]

                        if len(unique_2Ds):
                            if any(np.linalg.norm(np.vstack(unique_2Ds)-c2d, axis=1) <= 1e-3):
                                continue

                        neighbors_ud_sorted.append(neighbor)
                        neighbors_ud_sorted.append(neighbor)
                        unique_2Ds.append(c2d)

                    a_neighbors_ud = neighbors_ud_sorted

                    a_prism = [np.array(a[-1]) for a in a_neighbors_ud[:n_prism*2]]

                    if len(a_prism) == n_prism*2:                        
                        if point_in_hull(cell, np.array(atom[:-1]).reshape(-1)[non_layer_axes]) or \
                            any([point_in_hull(cell, point[non_layer_axes]) for point in a_prism]):

                            envs_in_extended_cell.append([p[non_layer_axes] for p in a_prism])
                            ah = float(atom[0][layer_axis])
                            if abs(cell_height-ah) < 0.1:
                                ah -= cell_height
                            if ah < -0.1:
                                ah += cell_height
                            center_heights.append(sround(ah))
                            prism_and_cap_xys.extend([p[non_layer_axes] for p in a_prism])
                            prism_and_cap_xys.append(atom[0][non_layer_axes])
                
                            lh = a_neighbors[0][2][layer_axis]

                            uc_layer_heights.add(lh)

                prisms_in_unitcell[site] = [uc_layer_heights, envs_in_extended_cell, center_heights]

        prism_metrics = sorted(prism_metrics, key=lambda x: x[1][0][1])

        plot_containers = st.container(border=True)
        left_plot, right_plot = plot_containers.columns(2)
        atoms_to_plot = []
        layer_elements = defaultdict(set)

        for p in supercell_points:
            h = float(p[0][layer_axis])
            if not -0.1 < h < cell_height+0.1:
                continue

            c = p[0][non_layer_axes]
            if any([np.all(np.allclose(c, t, rtol=5e-2)) for t in prism_and_cap_xys]) or \
                point_in_hull(cell, c):
                atoms_to_plot.append(p)

                symbol = site_symbol_map[p[1]]
                layer_elements[sround(h)].add(symbol)

        atom_heights = [sround(float(p[0][layer_axis])) for p in atoms_to_plot]
        atom_heights = list(set(atom_heights))
        if sround(cell_height) in atom_heights:
            atom_heights.remove(sround(cell_height))
        
        if swap_layers:
            atom_heights = sorted(atom_heights, reverse=True)
        else:
            atom_heights = sorted(atom_heights)

        atom_heights = np.array(atom_heights)
        prism_colors = {6: 'tab:red', 8: 'tab:green', 10: 'tab:blue', 12: 'tab:orange', 14: 'tab:purple'}

        with left_plot:
            height = atom_heights[0]
            plt.close()
            fig = plt.figure()
            fig.set_size_inches(non_layer_lengths)
            ax = plt.gca()
            ax.set_aspect('equal', adjustable='box')

            shade = True
            
            plotted_prisms = defaultdict(list)
            for site, (layer_heights, prisms, center_heights) in prisms_in_unitcell.items():
                for layer_height in layer_heights:
                    if abs(sround(layer_height) - sround(height)) > 0.2:
                        continue
                    if not len(prisms):
                        continue
                    
                    if len(prisms[0]) not in allowed_envs:
                        continue
                    
                    for prism, center_height in zip(prisms, center_heights):
                        if abs(center_height - height) > 0.2:
                            continue
                        prism = np.array(prism)
                        prism = prism[np.argsort(prism[:, 0])]
                        prism = prism[np.argsort(prism[:, 1])]
                        if any([np.all(np.allclose(prism, t, atol=1e-2, rtol=1e-2)) for t in plotted_prisms[len(prism)]]):
                            continue

                        hull = ConvexHull(prism)
                        edges = []
                        for simplex in hull.simplices:
                            edges.append([int(simplex[0]), int(simplex[1])])
                        
                        if shade:

                            for inds in edges:
                                plt.plot([prism[inds[0]][0], prism[inds[1]][0]], 
                                [prism[inds[0]][1], prism[inds[1]][1]],
                                c='y', # 'r' if len(edges)==3 else 'b',
                                alpha=0.7,
                                lw=lw)

                        if shade_prisms:
                            # order edges
                            ordered_edges = [edges[0]]
                            added = [0]
                            for _ in range(int(len(prism)/2)):
                                last_i = ordered_edges[-1][-1]
                                sel_i = [i for i in range(len(edges)) if i not in added and last_i in edges[i]]
                                if len(sel_i):
                                    sel_i = sel_i[0]
                                    if edges[sel_i][0] == last_i:
                                        ordered_edges.append(edges[sel_i])
                                    else:
                                        ordered_edges.append([edges[sel_i][1], edges[sel_i][0]])
                                    added.append(sel_i)
                            
                            edges = ordered_edges

                            tx, ty = [], []
                            for inds in edges:
                                tx.extend([float(prism[inds[0]][0]), float(prism[inds[1]][0])])
                                ty.extend([float(prism[inds[0]][1]), float(prism[inds[1]][1])])
                            
                            cn_col = prism_colors.get(len(prism), 'tab:gray')

                            plt.fill(tx, ty, color=cn_col, alpha=0.5, edgecolor=cn_col)
                        plotted_prisms[len(prism)].append(prism)

        legends = []

        for p in atoms_to_plot:
            sy = site_symbol_map[p[1]]
            s = p[1]
            lh = p[0][layer_axis]
            if abs(lh - cell_height) < 0.1:
                lh -= cell_height

            if separate_atoms_by_layer and not is_close(height, sround(lh)):
                continue

            m = markers[int(np.argmin(np.abs(atom_heights - lh)))]
            
            size = ms if m != "." else int(ms*2)

            p = p[0][non_layer_axes]

            if sy not in legends:
                legends.append(sy)
                plt.scatter(p[0], None, c=element_colors[sy], label=sy, s=ms+25, marker='.')
            
            if m == 'o':
                plt.scatter(p[0], p[1], edgecolor=element_colors[sy], s=size+25, marker=m, facecolor='none')
            else:
                plt.scatter(p[0], p[1], c=element_colors[sy], s=size, marker=m)
                
            if show_labels:
                plt.text(p[0]-(p[0]*0.025), p[1]-(p[1]*0.035), s, size=20)

        bby, ymax, ymin, xmax, xmin = get_bbox_y(atoms_to_plot, non_layer_axes)

        bby = plot_options_container.number_input(
            label="Legend position:",
            value=bby,   # Default value
        )

        plt.plot([0., unitcell_lengths[0]], [0., 0.], c='k', alpha=0.5, lw=ce)
        plt.plot([0., x1], [0., y1], c='k', alpha=0.5, lw=ce)
        plt.plot([x1, x1+unitcell_lengths[0]], [y1, y1], c='k', alpha=0.5, lw=ce)
        plt.plot([unitcell_lengths[0], x1+unitcell_lengths[0]], [0., y1], c='k', alpha=0.5, lw=ce)

        plt.legend(ncol=len(legends),  bbox_to_anchor=(0.5, bby), framealpha=0, loc="lower center", fontsize=16, columnspacing=0.1)
        plt.title(title, y=1.01, size=16)
        plt.axis('off')
        plt.tight_layout()
    
        if fig:
            left_plot.pyplot(fig)

            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format="png")
            img_buffer.seek(0)

            left_plot.download_button(
                label="Download Plot as PNG",
                data=img_buffer,
                file_name=f"{title.split(',')[0].replace('~', '')}_{file_name[:-4]}.png",
                mime="image/png",
                key='left'
            )

        with right_plot:
            height = atom_heights[1]
            plt.close()
            fig = plt.figure()
            fig.set_size_inches(non_layer_lengths)
            ax = plt.gca()
            ax.set_aspect('equal', adjustable='box')

            shade = True

            plotted_prisms = defaultdict(list)

            for site, (layer_heights, prisms, center_heights) in prisms_in_unitcell.items():
                for layer_height in layer_heights:
                    if abs(sround(layer_height) - sround(height)) > 0.2:
                        continue
                    if not len(prisms):
                        continue

                    if len(prisms[0]) not in allowed_envs:
                        continue
                    
                    for prism, center_height in zip(prisms, center_heights):

                        if abs(center_height - height) > 0.2:
                            continue

                        prism = np.array(prism)
                        prism = prism[np.argsort(prism[:, 0])]
                        prism = prism[np.argsort(prism[:, 1])]
                        if any([np.all(np.allclose(prism, t, atol=1e-2, rtol=1e-2)) for t in plotted_prisms[len(prism)]]):
                            continue

                        hull = ConvexHull(prism)
                        edges = []
                        for simplex in hull.simplices:
                            edges.append([int(simplex[0]), int(simplex[1])])
                        
                        if shade:

                            for inds in edges:
                                plt.plot([prism[inds[0]][0], prism[inds[1]][0]], 
                                [prism[inds[0]][1], prism[inds[1]][1]],
                                c='y', # 'r' if len(edges)==3 else 'b',
                                alpha=0.7,
                                lw=lw)

                        if shade_prisms:
                            # order edges
                            ordered_edges = [edges[0]]
                            added = [0]

                            for _ in range(int(len(prism)/2)):
                                last_i = ordered_edges[-1][-1]
                                
                                sel_i = [i for i in range(len(edges)) if i not in added and last_i in edges[i]]

                                if len(sel_i):
                                    sel_i = sel_i[0]
                                    if edges[sel_i][0] == last_i:
                                        ordered_edges.append(edges[sel_i])
                                    else:
                                        ordered_edges.append([edges[sel_i][1], edges[sel_i][0]])
                                    added.append(sel_i)
                            edges = ordered_edges

                            tx, ty = [], []
                            for inds in edges:
                                tx.extend([float(prism[inds[0]][0]), float(prism[inds[1]][0])])
                                ty.extend([float(prism[inds[0]][1]), float(prism[inds[1]][1])])
                            
                            cn_col = prism_colors.get(len(prism), 'tab:gray')

                            plt.fill(tx, ty, color=cn_col, alpha=0.5, edgecolor=cn_col)
                        plotted_prisms[len(prism)].append(prism)

        legends = []

        for p in atoms_to_plot:
            sy = site_symbol_map[p[1]]
            s = p[1]
            lh = p[0][layer_axis]
            if abs(lh - cell_height) < 0.1:
                lh -= cell_height

            if separate_atoms_by_layer and not is_close(height, sround(lh)):
                continue

            m = markers[int(np.argmin(np.abs(atom_heights - lh)))]
            
            size = ms if m != "." else int(ms*2)
            h = p[0][layer_axis]
            p = p[0][non_layer_axes]

            if sy not in legends:
                legends.append(sy)
                plt.scatter(p[0], None, c=element_colors[sy], label=sy, s=ms+25, marker='.')
            
            if m == 'o':
                plt.scatter(p[0], p[1], edgecolor=element_colors[sy], s=size+25, marker=m, facecolor='none')
            else:
                plt.scatter(p[0], p[1], c=element_colors[sy], s=size, marker=m)
                
            if show_labels:
                plt.text(p[0]-(p[0]*0.025), p[1]-(p[1]*0.035), s, size=20)

        plt.plot([0., unitcell_lengths[0]], [0., 0.], c='k', alpha=0.5, lw=ce)
        plt.plot([0., x1], [0., y1], c='k', alpha=0.5, lw=ce)
        plt.plot([x1, x1+unitcell_lengths[0]], [y1, y1], c='k', alpha=0.5, lw=ce)
        plt.plot([unitcell_lengths[0], x1+unitcell_lengths[0]], [0., y1], c='k', alpha=0.5, lw=ce)

        plt.legend(ncol=len(legends),  bbox_to_anchor=(0.5, bby), framealpha=0, loc="lower center", fontsize=16, columnspacing=0.1)
        plt.title(title, y=1.01, size=16)
        plt.axis('off')
        plt.tight_layout()
    
        if fig:
            right_plot.pyplot(fig)

            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format="png")
            img_buffer.seek(0)

            right_plot.download_button(
                label="Download Plot as PNG",
                data=img_buffer,
                file_name=f"{title.split(',')[0].replace('~', '')}_{file_name[:-4]}.png",
                mime="image/png",
                key='right'
            )




        # for CN in [12, 9]:
        #     cif_data = get_data(temp_file_path, CN=CN)
        #     if 'site_data' not in cif_data:
        #         print("No site data")
        #         continue
        #     for k, v in cif_data['site_data'].items():

        #         if k in sites_with_envs:
        #             continue

        #         atoms_with_this_site = [p for p in supercell_points if p[-1]==k]  #Change
        #         site_formula = v['coordination_formula']
        #         center_coordinate = np.array(v["center_coordinate"])

        #         for atom in atoms_with_this_site:
        #             if not 0-cell_height*tol < atom[0][layer_axis] < cell_height+cell_height*tol:
        #                 #  or \
        #                 # not point_in_hull(cell, atom[0][non_layer_axes])
        #                 continue
        #             points_wd = get_first_n_neighbors(atom, supercell_points, n=CN)
        #             formula = get_formula(points_wd, site_symbol_map)

        #             if formula != site_formula:
        #                 continue

        #             atom_capped_prims_data = get_capped_prism_data(site=atom[1], 
        #                                         layer_axis=layer_axis,
        #                                         points_wd=points_wd,
        #                                         CN=CN)

        #             if atom_capped_prims_data['capped_prism_present']:
        #                 center_coordinate = atom_capped_prims_data["center_coordinate"]
        #                 prism = atom_capped_prims_data['prism']

        #                 prism_atom_inside_cell = [point_in_hull(cell, atom[0][non_layer_axes])]
        #                 for point in prism:
        #                     prism_atom_inside_cell.append(point_in_hull(cell, point))

        #                 if not any(prism_atom_inside_cell):
        #                     continue

        #                 caps = atom_capped_prims_data['caps']
        #                 edges = atom_capped_prims_data['edges']
        #                 prism_full = atom_capped_prims_data['prism_full']

        #                 prism_and_cap_xys.extend(prism)
        #                 prism_and_cap_xys.extend([cap[2][non_layer_axes] for cap in caps])
        #                 prism_and_cap_xys.append(atom[0][non_layer_axes])

        #                 key = max([float(p[-1][layer_axis]) for p in prism_full])
        #                 if key < 0:
        #                     key += cell_height
        #                 if key > cell_height:
        #                     key -= cell_height
        #                 key = sround(key)
        #                 capped_prism_data_by_layers[key].append(
        #                     [center_coordinate, prism, edges, caps]
        #                 )
        #                 sites_with_envs.add(k)

        # # Plot prisms
        # layer_heights = sorted(list(capped_prism_data_by_layers.keys()))

        # atoms_to_plot = []
        # layer_elements = defaultdict(set)
        # for p in supercell_points:
        #     h = float(p[0][layer_axis])
        #     if not 0-cell_height*0.03 < h < cell_height*1.03:
        #         continue
        #     c = p[0][non_layer_axes]
        #     if any([np.all(np.allclose(c, t, rtol=5e-2)) for t in prism_and_cap_xys]) or \
        #         point_in_hull(cell, c):
        #         atoms_to_plot.append(p)

        #         symbol = site_symbol_map[p[1]]
        #         layer_elements[sround(h)].add(symbol)

        # if len(capped_prism_data_by_layers) == 0:
        #     st.write("No trigonal or square prisms found!")
        #     # exit(0)
        # elif len(capped_prism_data_by_layers) == 1:
        #     for val in layer_vals:
        #         if val not in capped_prism_data_by_layers:
        #             capped_prism_data_by_layers[val] = []
        # elif len(capped_prism_data_by_layers) == 3:
        #     top_h = sround(cell_height)
        #     if top_h in capped_prism_data_by_layers:
        #         # capped_prism_data_by_layers.pop(top_h)
        #         # del layer_heights[layer_heights.index(top_h)]

        #         # merge
        #         if 0.0 in capped_prism_data_by_layers:
        #             v0 = capped_prism_data_by_layers[0.0]
        #             v0.extend(capped_prism_data_by_layers[top_h])
        #             capped_prism_data_by_layers[0.0] = v0

        #             capped_prism_data_by_layers.pop(top_h)
        #             del layer_heights[layer_heights.index(top_h)]

        # layer_heights = np.array(layer_heights)
        
        # assert len(capped_prism_data_by_layers) == 2, f"Num layers: {len(capped_prism_data_by_layers)}, {capped_prism_data_by_layers.keys()}, {cell_height}"
        # title = format_formula(title.split(',')[0].replace("~", "")) + "-type " + get_sg_symbol(temp_file_path)
        # latex_code = top_cols[3].text_area("Edit title (LaTeX):", value=title, height=100)

        # # if latex_code.strip():
        # #     top_cols[3].latex(latex_code)
        # #     title = latex_code.strip()
        # # else:
        # #     top_cols[3].info("Type some LaTeX code above to see the preview.")

        # sorted_layer_heights = sorted(list(capped_prism_data_by_layers.keys()))
        # # for k, v in capped_prism_data_by_layers.items():
        # #     print(k, [_v[1].shape for _v in v])
        # # sorted_layer_heights = sorted(list(capped_prism_data_by_layers.keys()), 
        # #                               key=lambda h: int(layer_elements[h] <= REs),
        # #                               reverse=True)

        # left_plot, right_plot = st.columns(2)

        # with left_plot:
        #     plt.close()
        #     fig = plt.figure()
        #     fig.set_size_inches(non_layer_lengths)
        #     ax = plt.gca()
        #     ax.set_aspect('equal', adjustable='box')

        #     # add points
        #     height = sorted_layer_heights[0]
        #     layer_contents = capped_prism_data_by_layers[height]
        #     # for height, layer_contents in capped_prism_data_by_layers.items():
        #     shade = layer_elements[height] <= REs
        #     shade = True

        #     for center_coordinate, prism, edges, caps in layer_contents:
        #         hull = ConvexHull(prism)
        #         edges = []
        #         for simplex in hull.simplices:
        #             edges.append([int(simplex[0]), int(simplex[1])])
                
        #         if shade:

        #             for inds in edges:
        #                 plt.plot([prism[inds[0]][0], prism[inds[1]][0]], 
        #                 [prism[inds[0]][1], prism[inds[1]][1]],
        #                 c='y', # 'r' if len(edges)==3 else 'b',
        #                 alpha=0.7,
        #                 lw=lw)

        #             if shade_prisms:
        #                 # order edges
        #                 ordered_edges = [edges[0]]
        #                 added = [0]

        #                 for _ in range(len(prism) - 1):
        #                     last_i = ordered_edges[-1][-1]

        #                     sel_i = [i for i in range(len(prism)) if i not in added and last_i in edges[i]][0]
        #                     if edges[sel_i][0] == last_i:
        #                         ordered_edges.append(edges[sel_i])
        #                     else:
        #                         ordered_edges.append([edges[sel_i][1], edges[sel_i][0]])
        #                     added.append(sel_i)
                        
        #                 edges = ordered_edges

        #                 tx, ty = [], []
        #                 for inds in edges:
        #                     tx.extend([float(prism[inds[0]][0]), float(prism[inds[1]][0])])
        #                     ty.extend([float(prism[inds[0]][1]), float(prism[inds[1]][1])])
                        
        #                 if len(prism) == 3:
        #                     cn_col = 'r' 
        #                 else:
        #                     cn_col = 'g'

        #                 plt.fill(tx, ty, color=cn_col, alpha=0.5, edgecolor=cn_col)

        #     legends = []
        #     atom_heights = [sround(float(p[0][layer_axis])) for p in atoms_to_plot]
        #     atom_heights = list(set(atom_heights))
        #     atom_heights.remove(sround(cell_height))
        #     atom_heights = np.array(atom_heights)


        #     for p in atoms_to_plot:
        #         sy = site_symbol_map[p[1]]
        #         s = p[1]
        #         lh = p[0][layer_axis]
        #         if abs(lh - cell_height) < 0.1:
        #             lh -= cell_height

        #         if separate_atoms_by_layer and not is_close(height, sround(lh)):
        #             continue

        #         m = markers[int(np.argmin(np.abs(atom_heights - lh)))]
                
        #         size = ms if m != "." else int(ms*2)

        #         p = p[0][non_layer_axes]

        #         if sy not in legends:
        #             legends.append(sy)
        #             plt.scatter(p[0], None, c=element_colors[sy], label=sy, s=ms+25, marker='.')
                
        #         if m == 'o':
        #             plt.scatter(p[0], p[1], edgecolor=element_colors[sy], s=size+25, marker=m, facecolor='none')
        #         else:
        #             plt.scatter(p[0], p[1], c=element_colors[sy], s=size, marker=m)
                    
        #         if show_labels:
        #             plt.text(p[0]-(p[0]*0.025), p[1]-(p[1]*0.035), s, size=20)

        #     bby, ymax, ymin, xmax, xmin = get_bbox_y(atoms_to_plot, non_layer_axes)

        #     bby = top_cols[3].number_input(
        #         label="Legend position:",
        #         value=bby,   # Default value
        #     )

        #     plt.plot([0., unitcell_lengths[0]], [0., 0.], c='k', alpha=0.5, lw=ce)
        #     plt.plot([0., x1], [0., y1], c='k', alpha=0.5, lw=ce)
        #     plt.plot([x1, x1+unitcell_lengths[0]], [y1, y1], c='k', alpha=0.5, lw=ce)
        #     plt.plot([unitcell_lengths[0], x1+unitcell_lengths[0]], [0., y1], c='k', alpha=0.5, lw=ce)

        #     plt.legend(ncol=len(legends),  bbox_to_anchor=(0.5, bby), framealpha=0, loc="lower center", fontsize=16, columnspacing=0.1)
        #     plt.title(title, y=1.01, size=16)
        #     plt.axis('off')
        #     plt.tight_layout()
        
        #     if fig:
        #         st.pyplot(fig)

        #         img_buffer = io.BytesIO()
        #         fig.savefig(img_buffer, format="png")
        #         img_buffer.seek(0)

        #         st.download_button(
        #             label="Download Plot as PNG",
        #             data=img_buffer,
        #             file_name=f"{title.split(',')[0].replace('~', '')}_{file_name[:-4]}.png",
        #             mime="image/png",
        #             key='left'
        #         )


        # with right_plot:
        #     plt.close()
        #     fig = plt.figure()
        #     fig.set_size_inches(non_layer_lengths)
        #     ax = plt.gca()
        #     ax.set_aspect('equal', adjustable='box')
        

        #     # add points
        #     height = sorted_layer_heights[1]
        #     layer_contents = capped_prism_data_by_layers[height]
        #     # for height, layer_contents in capped_prism_data_by_layers.items():
        #     shade = layer_elements[height] <= REs

        #     shade = True
        #     for center_coordinate, prism, edges, caps in layer_contents:
        #         # print("475", prism)
        #         hull = ConvexHull(prism)
        #         edges = []
        #         for simplex in hull.simplices:
        #             edges.append([int(simplex[0]), int(simplex[1])])
                
        #         if shade:

        #             for inds in edges:
        #                 plt.plot([prism[inds[0]][0], prism[inds[1]][0]], 
        #                 [prism[inds[0]][1], prism[inds[1]][1]],
        #                 c='y', # 'r' if len(edges)==3 else 'b',
        #                 alpha=0.7,
        #                 lw=lw)

        #         if shade_prisms:
        #             # order edges
        #             ordered_edges = [edges[0]]
        #             added = [0]

        #             for _ in range(len(prism) - 1):
        #                 last_i = ordered_edges[-1][-1]

        #                 sel_i = [i for i in range(len(prism)) if i not in added and last_i in edges[i]][0]
        #                 if edges[sel_i][0] == last_i:
        #                     ordered_edges.append(edges[sel_i])
        #                 else:
        #                     ordered_edges.append([edges[sel_i][1], edges[sel_i][0]])
        #                 added.append(sel_i)
                    
        #             edges = ordered_edges

        #             tx, ty = [], []
        #             for inds in edges:
        #                 tx.extend([prism[inds[0]][0], prism[inds[1]][0]])
        #                 ty.extend([prism[inds[0]][1], prism[inds[1]][1]])
                    
        #             cn_col = 'g'
        #             if len(prism) == 3:
        #                 cn_col = 'r' 
                        
        #             plt.fill(tx, ty, color=cn_col, alpha=0.5, edgecolor=cn_col)

        #     legends = []
        #     for p in atoms_to_plot:
        #         sy = site_symbol_map[p[1]]
        #         s = p[1]
        #         lh = p[0][layer_axis]
        #         if separate_atoms_by_layer and not is_close(height, sround(lh)):
        #             continue
        #         m = markers[int(np.argmin(np.abs(layer_heights - lh)))]
                
        #         size = ms if m != "." else int(ms*2)

        #         p = p[0][non_layer_axes]

        #         if sy not in legends:
        #             legends.append(sy)
        #             plt.scatter(p[0], None, c=element_colors[sy], label=sy, s=ms+25, marker='.')
                
        #         if m == 'o':
        #             plt.scatter(p[0], p[1], edgecolor=element_colors[sy], s=size+25, marker=m, facecolor='none')
        #         else:
        #             plt.scatter(p[0], p[1], c=element_colors[sy], s=size, marker=m)
                    
        #         if show_labels:
        #             plt.text(p[0]-(p[0]*0.025), p[1]-(p[1]*0.035), s, size=20)

        #     plt.plot([0., unitcell_lengths[0]], [0., 0.], c='k', alpha=0.5, lw=ce)
        #     plt.plot([0., x1], [0., y1], c='k', alpha=0.5, lw=ce)
        #     plt.plot([x1, x1+unitcell_lengths[0]], [y1, y1], c='k', alpha=0.5, lw=ce)
        #     plt.plot([unitcell_lengths[0], x1+unitcell_lengths[0]], [0., y1], c='k', alpha=0.5, lw=ce)

        #     # bby, ymax, ymin, xmax, xmin = get_bbox_y(atoms_to_plot, non_layer_axes)
        #     plt.legend(ncol=len(legends),  bbox_to_anchor=(0.5, bby), framealpha=0, loc="lower center", fontsize=16, columnspacing=0.1)
        #     plt.title(title, y=1.01, size=16)
        #     plt.axis('off')
        #     plt.tight_layout()
        
        #     if fig:
        #         st.pyplot(fig)

        #         img_buffer = io.BytesIO()
        #         fig.savefig(img_buffer, format="png")
        #         img_buffer.seek(0)

        #         st.download_button(
        #             label="Download Plot as PNG",
        #             data=img_buffer,
        #             file_name=f"{title.split(',')[0].replace('~', '')}_{file_name[:-4]}.png",
        #             mime="image/png",
        #             key='right'
        #         )
