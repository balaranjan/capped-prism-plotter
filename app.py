import streamlit as st
from tempfile import NamedTemporaryFile
from cifkit import Cif
from utils import get_data, get_colors, adjust_xyz, get_first_n_neighbors, get_formula
from utils import get_capped_prism_data, point_in_hull, sround, format_formula, get_sg_symbol
from utils import get_bbox_y, rearrange_coordinates
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
from matplotlib.colors import to_hex
import io

st.set_page_config(layout="wide")

REs = ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Eu', 'Gd', 'Tb', 
       'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Sc', 'Y',
       'Ti', 'Hf', 'Zr',
       'Li', 'Na', 'K', 'Rb', 'Cs',
       'Be', 'Mg', 'Ca', 'Sr', 'Ba',
       'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
       ]
REs = set(REs)

fig = None
left_col, right_col = st.columns([1, 3])

with left_col:
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        file_name = uploaded_file.name
        with NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
            
            cif = Cif(temp_file_path)
            unitcell_lengths = cif.unitcell_lengths
            unitcell_angles = cif.unitcell_angles
            unitcell_points = cif.unitcell_points
            supercell_points = cif.supercell_points

            for i in range(len(unitcell_points)):
                x, y, z, l = unitcell_points[i]
                x, y, z = adjust_xyz(x, y, z)

                cart = unit.fractional_to_cartesian([x, y, z],
                                                    unitcell_lengths,
                                                    unitcell_angles)
                unitcell_points[i] = (*cart, l)

            for i in range(len(supercell_points)):
                s = supercell_points[i][-1]
                cart = unit.fractional_to_cartesian(supercell_points[i][:3],
                                                    unitcell_lengths,
                                                    unitcell_angles)
                supercell_points[i] = [np.array(cart), s]

            cif_data_1 = get_data(temp_file_path, CN=9)
            title = cif_data_1['Structure Type']
            layer_axis = cif_data_1['layer_axis']
            layer_vals = cif_data_1['layer_vals']
            non_layer_axes = cif_data_1['non_layer_axes']
            non_layer_lengths = [unitcell_lengths[l] for l in range(3) if l != layer_axis]

            loop_vals = cif._loop_values
            site_symbol_map = dict(zip([l for l in loop_vals[0]], [s for s in loop_vals[1]]))

            element_colors = get_colors(site_symbol_map)

            # Plot
            plt.close()
            fig = plt.figure()
            fig.set_size_inches(non_layer_lengths)
            ax = plt.gca()
            ax.set_aspect('equal', adjustable='box')
            
            cell_height = unitcell_lengths[layer_axis]
            unitcell_lengths = np.array(unitcell_lengths)[non_layer_axes]
            x1 = unitcell_lengths[1] * np.cos(unitcell_angles[layer_axis])
            y1 = unitcell_lengths[1] * np.sin(unitcell_angles[layer_axis])

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
            
            plt.plot([0., unitcell_lengths[0]], [0., 0.], c='k', alpha=0.5, lw=ce)
            plt.plot([0., x1], [0., y1], c='k', alpha=0.5, lw=ce)
            plt.plot([x1, x1+unitcell_lengths[0]], [y1, y1], c='k', alpha=0.5, lw=ce)
            plt.plot([unitcell_lengths[0], x1+unitcell_lengths[0]], [0., y1], c='k', alpha=0.5, lw=ce)

            # add points
            legends = []
            unitcell_layer_heights = set([p[layer_axis] for p in unitcell_points])
            layer_heights = np.array(sorted(unitcell_layer_heights)[:2], dtype=float)
            markers = ['.', '.']

            # flip_layers = st.toggle("Flip layers", value=False)
            # if flip_layers:
            #     markers = ['.', '.']

            ms = st.number_input(
                label="Marker size:",
                min_value=1,
                max_value=1000,
                value=300,   # Default value
                step=1
            )
            
            # o_marker_offset = 25
            # if fix_min_length:
            #     o_marker_offset = int(o_marker_offset/(num_supercell_atoms/10))

            capped_prism_data_by_layers = defaultdict(list)
            tol = 0.05
            cell = [
                [0.-unitcell_lengths[0]*tol, 0.-unitcell_lengths[1]*tol],
                [unitcell_lengths[0]+unitcell_lengths[0]*tol, 0.-unitcell_lengths[1]*tol],
                [x1-unitcell_lengths[0]*tol, y1+unitcell_lengths[1]*tol],
                [x1+unitcell_lengths[0]+unitcell_lengths[0]*tol, y1+unitcell_lengths[1]*tol],
            ]

            show_labels = st.toggle("Show labels", value=False)

            # TCTPs
            prism_and_cap_xys = []
            for CN in [9, 12]:
                cif_data = get_data(temp_file_path, CN=CN)
                if 'site_data' not in cif_data:
                    continue
                for k, v in cif_data['site_data'].items():
                    # print(k, v)
                    atoms_with_this_site = [p for p in supercell_points if p[-1]==k]  #Change
                    site_formula = v['coordination_formula']
                    center_coordinate = np.array(v["center_coordinate"])

                    for atom in atoms_with_this_site:
                        if not 0 < atom[0][layer_axis] < cell_height or \
                            not point_in_hull(cell, atom[0][non_layer_axes]):
                            continue
                        points_wd = get_first_n_neighbors(atom, supercell_points, n=CN)
                        formula = get_formula(points_wd, site_symbol_map)

                        if formula != site_formula:
                            continue

                        atom_capped_prims_data = get_capped_prism_data(site=atom[1], 
                                                    layer_axis=layer_axis,
                                                    points_wd=points_wd,
                                                    CN=9)
                        
                        if atom_capped_prims_data['capped_prism_present']:
                            center_coordinate = atom_capped_prims_data["center_coordinate"]
                            prism = atom_capped_prims_data['prism']
                            caps = atom_capped_prims_data['caps']
                            edges = atom_capped_prims_data['edges']
                            prism_full = atom_capped_prims_data['prism_full']

                            prism_and_cap_xys.extend(prism)
                            prism_and_cap_xys.extend([cap[2][non_layer_axes] for cap in caps])
                            prism_and_cap_xys.append(atom[0][non_layer_axes])

                            key = sround(max([float(p[-1][layer_axis]) for p in prism_full]))
                            capped_prism_data_by_layers[key].append(
                                [center_coordinate, prism, edges, caps]
                            )

            # print(capped_prism_data_by_layers)

            # Plot prisms
            layer_heights = sorted(list(capped_prism_data_by_layers.keys()))
            # if flip_layers:
            #     layer_heights = sorted(layer_heights, reverse=True)
            layer_heights = np.array(layer_heights)

            atoms_to_plot = []
            layer_elements = defaultdict(set)
            for p in supercell_points:
                h = float(p[0][layer_axis])
                if not 0-cell_height*0.03 < h < cell_height*1.03:
                    continue
                c = p[0][non_layer_axes]
                if any([np.all(np.allclose(c, t, rtol=5e-2)) for t in prism_and_cap_xys]) or \
                    point_in_hull(cell, c):
                    atoms_to_plot.append(p)

                    symbol = site_symbol_map[p[1]]
                    layer_elements[sround(h)].add(symbol)

            # print(capped_prism_data_by_layers[1.84])
            for height, layer_contents in capped_prism_data_by_layers.items():
                shade = layer_elements[height] <= REs

                for center_coordinate, prism, edges, caps in layer_contents:
                    # shade = np.argmin(np.abs(layer_heights - caps[0][2][layer_axis])) == 0
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

                    # else:                    
                    #     for inds in edges:
                    #         plt.plot([prism[inds[0]][0], prism[inds[1]][0]], 
                    #         [prism[inds[0]][1], prism[inds[1]][1]],
                    #         c='y', # 'r' if len(edges)==3 else 'b', alpha=0.7, 
                    #         )

            elements = list(set(list(site_symbol_map.values())))
            cols = st.columns(len(elements)+1)
            cols[0].write("Colors ")
            for i, e in enumerate(elements, 1):
                current_color = to_hex(element_colors[e])
                color = cols[i].color_picker(f'{e}', current_color)
                element_colors[e] = color

            for p in atoms_to_plot:
                sy = site_symbol_map[p[1]]
                s = p[1]
                lh = p[0][layer_axis]
                m = markers[int(np.argmin(np.abs(layer_heights - lh)))]
                
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
            # plt.ylim(ymin, ymax)
            # plt.xlim(xmin, xmax)
            plt.legend(ncol=len(legends),  bbox_to_anchor=(0.5, bby), framealpha=0, loc="lower center", fontsize=16, columnspacing=0.1)

            title = format_formula(title.split(',')[0].replace("~", "")) + "-type " + get_sg_symbol(temp_file_path)
            latex_code = st.text_area("Edit title (LaTeX):", value=title, height=100)

            if latex_code.strip():
                st.latex(latex_code)
                title = latex_code.strip()
            else:
                st.info("Type some LaTeX code above to see the preview.")

            plt.title(title, y=1.01, size=16)
            plt.axis('off')
            plt.tight_layout()
        
        # for p in cell:
        #     plt.scatter(p[0], p[1], marker='^', s=100)

        # print(cif_data_1)
with right_col:
    if fig:
        st.pyplot(fig)

        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format="png")
        img_buffer.seek(0)

        st.download_button(
            label="Download Plot as PNG",
            data=img_buffer,
            file_name=f"{title.split(',')[0].replace('~', '')}_{file_name[:-4]}.png",
            mime="image/png"
        )
