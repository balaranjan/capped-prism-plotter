import os
import re
import numpy as np
import pandas as pd
from typing import Any
from collections import defaultdict

def cif_to_dict(path: str) -> dict:
    if not os.path.isfile(path):
        print(f"File {path} not found!")
        return
    
    # now this will only read formula and coordinates
    # TODO: write a full parser
    
    attributes_to_read = [
        '_chemical_formula_structural',
        '_chemical_formula_sum',
        '_chemical_name_structure_type',
        '_chemical_formula_weight',
        '_cell_length_a',
        '_cell_length_b',
        '_cell_length_c',
        '_cell_angle_alpha',
        '_cell_angle_beta',
        '_cell_angle_gamma',
        '_cell_volume',
        '_cell_formula_units_Z',
        '_space_group_IT_number',
        '_space_group_name_H-M_alt',
        '#_database_code_PCD'
    ]
    
    data = defaultdict(Any)
    with open(path, 'r') as f: # encoding="latin-1"
        lines = f.readlines()
        ln = 0
        while ln < len(lines):
            line = lines[ln].lstrip()
            if not len(line.split()):
                ln += 1
                continue
            
            if line.split()[0] in attributes_to_read:
                next_line = lines[ln+1].lstrip()
                if next_line.startswith('_') or next_line.startswith('#') or next_line.startswith('loop_'):
                    data[line.split()[0]] = line.split()[1:]
                else:
                    line_data = ' '.join(line.split()[1:])
                    while not (next_line.startswith('_') or next_line.startswith('#')):
                        line_data += next_line.strip().replace(';', '').replace(' ', '')
                        ln += 1
                        next_line = lines[ln+1].lstrip()
                    # print(line.split()[0], line_data)
                    data[line.split()[0]] = line_data.strip().replace(';', '').replace(' ', '')
                
            if line.startswith('loop_') and lines[ln+1].lstrip().startswith('_atom_site'):
                site_data = []
                keys = []
                ln += 1
                while lines[ln].lstrip().startswith('_atom_site'):
                    keys.append(lines[ln].lstrip().replace('\n', ''))
                    ln += 1
                
                while not lines[ln].lstrip().startswith('_'):
                    if len(lines[ln].strip()):
                        site_data.append(dict(zip(keys, lines[ln].lstrip().split())))
                    ln += 1
                data['atom_site_data'] = site_data
                
            ln += 1
    # print(data)
    data = format_cif_data(data)
    # print(data)
    return dict(data)
    
    
def format_cif_data(cif_data: dict) -> dict:
    numeric_arribs = [
    '_chemical_formula_weight',
    '_cell_length_a',
    '_cell_length_b',
    '_cell_length_c',
    '_cell_angle_alpha',
    '_cell_angle_beta',
    '_cell_angle_gamma',
    '_cell_volume',
    '_cell_formula_units_Z',
    '_space_group_IT_number',
    '#_database_code_PCD'
    ]
    
    for k in numeric_arribs:
        if k in cif_data:
            if k in ['_cell_formula_units_Z', '_space_group_IT_number', '#_database_code_PCD']:
                try:
                    cif_data[k] = int(cif_data[k][0])
                except:
                    cif_data[k] = -1
            else:
                cif_data[k] = float(cif_data[k][0])
    
    string_arribs = [
    '_chemical_formula_sum',
    '_chemical_name_structure_type',
    '_space_group_name_H-M_alt'
    ]
    
    for k in string_arribs:
        if k in cif_data:
            if isinstance(cif_data[k], list):
                cif_data[k] = ''.join(cif_data[k]).replace("'", '')
                
    site_data = []
    if cif_data.get('atom_site_data', None) is not None:
        for site in cif_data['atom_site_data']:
            sdict = {}
            for k, v in site.items():
                if k not in ['_atom_site_label', '_atom_site_type_symbol', '_atom_site_Wyckoff_symbol']:
                    sdict[k] = float(v.split('(')[0])
                else:
                    sdict[k] = v
            sdict['coordinates'] = [float(c) for c in [site['_atom_site_fract_x'], site['_atom_site_fract_y'], site['_atom_site_fract_z']]]
            site_data.append(sdict)
        cif_data['atom_site_data'] = site_data
        cif_data['wyckoff_sequence'] = get_Wyckoff_sequence(site_data)
    else:
        cif_data['atom_site_data'] = []
    

    cif_data['formula'] = _parse_formula(cif_data['_chemical_formula_sum'])
    cif_data['_chemical_name_structure_type'] = cif_data['_chemical_name_structure_type'].replace('~', "").replace('-', ',')
    return cif_data


def get_Wyckoff_sequence(atom_site_data):

    tdf = pd.DataFrame(atom_site_data)
    tdf.drop_duplicates(['_atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z'], inplace=True)
    
    wyckoff_letters = tdf['_atom_site_Wyckoff_symbol'].tolist()
    wyckoff_letters = {wl: wyckoff_letters.count(wl) for wl in set(wyckoff_letters)}
    wyckoff_letters = [[k, v] for k, v in wyckoff_letters.items()]
    wyckoff_letters = sorted(wyckoff_letters, key=lambda x: x[0], reverse=True)
    
    wyckoff_sequence = ""
    for wl, c in wyckoff_letters:
        if c == 1:
            wyckoff_sequence += f"{wl}"
        else:
            wyckoff_sequence += f"{wl}{c}"

    return wyckoff_sequence
            


def _parse_formula(formula: str, strict: bool = True) -> dict[str, float]:
    """
    copied from pymatgen
    
    Args:
        formula (str): A string formula, e.g. Fe2O3, Li3Fe2(PO4)3.
        strict (bool): Whether to throw an error if formula string is invalid (e.g. empty).
            Defaults to True.

    Returns:
        Composition with that formula.

    Notes:
        In the case of Metallofullerene formula (e.g. Y3N@C80),
        the @ mark will be dropped and passed to parser.
    """
    # Raise error if formula contains special characters or only spaces and/or numbers
    if "'" in formula:
        formula = formula.replace("'", "")

    if strict and re.match(r"[\s\d.*/]*$", formula):
        print(formula)
        raise ValueError(f"Invalid {formula=}")

    # For Metallofullerene like "Y3N@C80"
    formula = formula.replace("@", "")
    # Square brackets are used in formulas to denote coordination complexes (gh-3583)
    formula = formula.replace("[", "(")
    formula = formula.replace("]", ")")
    
    def get_sym_dict(form: str, factor: float) -> dict[str, float]:
        sym_dict: dict[str, float] = defaultdict(float)
        for match in re.finditer(r"([A-Z][a-z]*)\s*([-*\.e\d]*)", form):
            el = match[1]
            amt = 1.0
            if match[2].strip() != "":
                amt = float(match[2])
            sym_dict[el] += amt * factor
            form = form.replace(match.group(), "", 1)
        if form.strip():
            raise ValueError(f"{form} is an invalid formula!")
        return sym_dict

    match = re.search(r"\(([^\(\)]+)\)\s*([\.e\d]*)", formula)
    while match:
        factor = 1.0
        if match[2] != "":
            factor = float(match[2])
        unit_sym_dict = get_sym_dict(match[1], factor)
        expanded_sym = "".join(f"{el}{amt}" for el, amt in unit_sym_dict.items())
        expanded_formula = formula.replace(match.group(), expanded_sym, 1)
        formula = expanded_formula
        match = re.search(r"\(([^\(\)]+)\)\s*([\.e\d]*)", formula)
    return get_sym_dict(formula, 1)

        
                
# cif_to_dict("/Users/bala/research/1_substituition_analysis/data/538170.cif")
# cif_to_dict("/Users/bala/research/1_substituition_analysis/data/539102.cif")
# cif_to_dict("/Users/bala/research/1_substituition_analysis/data/555511.cif")
# c = cif_to_dict("/home/bala/Documents/1_CN4_filtering/cif-cleaner/im_full_occupancy/1251296.cif")
c = cif_to_dict("/home/bala/Documents/data/not_prototype_CIFs/1903768.cif")
# print(c)
