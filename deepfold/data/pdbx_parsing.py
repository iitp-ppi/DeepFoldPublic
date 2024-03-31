# Copyright 2024 DeepFold Team


import collections
import gzip
import logging
import os
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from io import BytesIO, StringIO
from itertools import product
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import requests
import tqdm
from Bio.Data.PDBData import protein_letters_3to1
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from deepfold.common import residue_constants as rc
from deepfold.data.errors import PDBxError, PDBxWarning
from deepfold.data.monomer import get_atom_map
from deepfold.utils.file_utils import read_text

logger = logging.getLogger(__name__)


# _entity_poly_seq
@dataclass(frozen=True)
class Monomer:
    entity_id: int
    num: int
    mon_id: str


# _pdbx_struct_oper_list
@dataclass(frozen=True)
class StructOp:
    op_id: str
    is_identity: bool = True
    rot: np.ndarray = field(default=np.identity(3))
    trans: np.ndarray = field(default=np.zeros(3))

    def to_dict(self):
        return {
            "is_identity": self.is_identity,
            "rot": self.rot.tolist(),
            "trans": self.trans.tolist(),
        }


OpSeq = List[StructOp]


# _pdbx_struct_assembly_gen
@dataclass(frozen=True)
class AssemblyGenerator:
    oper_list: List[OpSeq] = field(default_factory=list)
    asym_id_list: List[str] = field(default_factory=list)


# _atom_site
@dataclass
class Atom:
    element: str
    atom_id: str
    alt_id: str
    # comp_id: str
    # asym_id: str
    # entity_id: int
    # seq_id: int
    # insertion_code: str
    occupancy: float
    b_factor: float
    # model_num: int
    coord: np.ndarray = field(default=np.zeros(3))


AtomList = List[Atom]


# Used to map SEQRES index to a residue in the structure.
@dataclass
class Residue:
    name: str
    residue_number: int
    atom_list: AtomList = field(default_factory=list)


# _pdbx_struct_mod_residue
@dataclass(frozen=True)
class ModRes:
    asym_id: str
    seq_id: int
    insertion_code: str
    comp_id: str
    parent_comp_id: str


@dataclass(frozen=True)
class MMCIFHeader:
    entry_id: str
    name: str
    deposition_date: str
    method: str
    resolution: float | None = None


@dataclass(frozen=True)
class ResidueAtPositon:
    name: str
    is_missing: bool
    hetflag: str
    chain_id: str | None = None
    residue_number: int | None = None
    insertion_code: str | None = None


@dataclass(frozen=True)
class MMCIFObject:
    header: MMCIFHeader

    entity_poly: Dict[int, List[Monomer]] = field(default_factory=dict)
    entity_map: Dict[int, List[str]] = field(default_factory=dict)
    valid_chains: List[str] = field(default_factory=list)

    chain_to_seqres: Dict[str, str] = field(default_factory=dict)
    seqres_to_structure: Dict[int, Dict[str, Dict[int, ResidueAtPositon]]] = field(default_factory=dict)
    modres: Dict[str, Dict[int, ModRes]] = field(default_factory=dict)

    models: Dict[int, Dict[str, Dict[Tuple[str, int, str], Residue]]] = field(default_factory=dict)
    assemblies: Dict[int, List[AssemblyGenerator]] = field(default_factory=dict)
    operations: Dict[str, StructOp] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsingResult:
    mmcif_object: MMCIFObject | None
    errors: Dict[Tuple[str, str], Any]


class MMCIFParser:
    """Parse a mmCIF file."""

    def __init__(self):
        self._header: MMCIFHeader = None
        self._polymers = None
        self._entity_to_chains = None
        self._chain_to_entity = None
        self._valid_chains = None

        self._chem_comps = None

        self._models: Dict[int, dict] = {}
        self._current_residue: Residue = None
        self._current_chain: Dict[Tuple[str, int, str], Residue] = None
        self._current_model: Dict[str, dict] = None

        self._seqres_to_structure: Dict[int, Dict[str, Dict[int, ResidueAtPositon]]] = None
        self._chain_to_seqres: Dict[str, str] = None

        self._modres: Dict[str, Dict[int, ModRes]] = None
        self._assemblies: Dict[int, List[AssemblyGenerator]] = None
        self._operations: Dict[str, StructOp] = None

        self._tqdm_kwargs = {"leave": False}

    def parse(
        self,
        mmcif_string: str,
        entry_id: str | None = None,
        catch_all_errors: bool = False,
    ) -> ParsingResult:
        entry_id = "" if entry_id is None else entry_id
        errors = {}
        try:
            self._mmcif_dict = MMCIF2Dict(StringIO(mmcif_string))
            self._get_header()
            self._build_chem_comps()
            self._build_modres()
            self._build_entity_poly()
            self._build_structure()
            self._get_assemblies()

            mmcif_object = MMCIFObject(
                header=self._header,
                models=self._models,
                entity_poly=self._polymers,
                entity_map=self._entity_to_chains,
                valid_chains=self._valid_chains,
                chain_to_seqres=self._chain_to_seqres,
                seqres_to_structure=self._seqres_to_structure,
                modres=self._modres,
                assemblies=self._assemblies,
                operations=self._operations,
            )

            return ParsingResult(mmcif_object=mmcif_object, errors=errors)
        except Exception as e:
            errors[(entry_id, "")] = e
            if not catch_all_errors:
                raise
            return ParsingResult(mmcif_object=None, errors=errors)

    def _get_header_entry(self, keys: Iterable[str]) -> str | None:
        for key in keys:
            val = self._mmcif_dict.get(key)
            try:
                item = val[0]
            except (TypeError, IndexError):
                continue
            if item != "?" and item != ".":
                return item
        return None

    def _get_header(self) -> None:
        entry_id = self._get_header_entry(["_entry_id", "_exptl.entry_id", "_struct.entry_id"])
        name = self._get_header_entry(["_struct.title"])
        deposition_date = self._get_header_entry(["_pdbx_database_status.recvd_initial_deposition_date"])
        method = self._get_header_entry(["_exptl.method"])
        resolution = self._get_header_entry(["_refine.ls_d_res_high", "_refine_hist.d_res_high", "_em_3d_reconstruction.resolution"])
        if resolution is not None:
            try:
                resolution = float(resolution)
            except ValueError:
                resolution = None

        self._header = MMCIFHeader(
            entry_id=entry_id,
            name=name,
            deposition_date=deposition_date,
            method=method,
            resolution=resolution,
        )

    def _build_modres(self) -> None:
        try:
            modres_asym_id = self._mmcif_dict["_pdbx_struct_mod_residue.label_asym_id"]
            modres_comp_id = self._mmcif_dict["_pdbx_struct_mod_residue.label_comp_id"]
            modres_seq_id = [int(x) for x in self._mmcif_dict["_pdbx_struct_mod_residue.label_seq_id"]]
            modres_icode = self._mmcif_dict["_pdbx_struct_mod_residue.PDB_ins_code"]
            modres_parent_comp_id = self._mmcif_dict["_pdbx_struct_mod_residue.parent_comp_id"]
        except KeyError:
            self._modres = {}
            return

        mod_residues: Dict[str, List[ModRes]] = {}
        for asym_id, comp_id, seq_id, icode, parent in tqdm.tqdm(
            zip(
                modres_asym_id,
                modres_comp_id,
                modres_seq_id,
                modres_icode,
                modres_parent_comp_id,
            ),
            desc="_mod_residue",
            **self._tqdm_kwargs,
        ):
            # if asym_id not in self._valid_chains:
            #     continue  # Skip for non-valid chains.
            current = mod_residues.get(asym_id, {})
            current[seq_id] = ModRes(
                asym_id=asym_id,
                seq_id=seq_id,
                insertion_code=" " if icode in (".", "?") else icode,
                comp_id=comp_id,
                parent_comp_id=parent,
            )
            mod_residues[asym_id] = current
        self._modres = mod_residues

    def _build_chem_comps(self) -> None:
        # Get chemical compositions which allow us to identify which of these polymers are polypeptides.
        chem_comps_id_list = self._mmcif_dict["_chem_comp.id"]
        chem_comps_type_list = self._mmcif_dict["_chem_comp.type"]

        chem_comps = {}
        for id, type in zip(chem_comps_id_list, chem_comps_type_list):
            chem_comps[id] = type

        self._chem_comps = chem_comps

    def _build_entity_poly(self) -> None:
        # Get chain information for each entity.
        struct_asym_asym_id_list = self._mmcif_dict["_struct_asym.id"]
        struct_asym_entity_id_list = self._mmcif_dict["_struct_asym.entity_id"]

        entity_to_chains = collections.defaultdict(list)
        for asym_id, entity_id in tqdm.tqdm(
            zip(struct_asym_asym_id_list, struct_asym_entity_id_list),
            desc="_struct_asym",
            **self._tqdm_kwargs,
        ):
            entity_id = int(entity_id)
            entity_to_chains[entity_id].append(asym_id)
        self._entity_to_chains = entity_to_chains

        chain_to_entity = {}
        for entity_id, chain_ids in entity_to_chains.items():
            for chain_id in chain_ids:
                chain_to_entity[chain_id] = entity_id
        self._chain_to_entity = chain_to_entity

        # Generate modified residue table
        entity_modres_table = {}
        for chain_id, mrs in self._modres.items():
            entity_id = self._chain_to_entity[chain_id]
            for v in mrs.values():
                entity_modres_table[(entity_id, v.seq_id)] = (v.comp_id, v.parent_comp_id)

        # Get polymer information for each entity.
        entity_seq_entity_id_list = self._mmcif_dict["_entity_poly_seq.entity_id"]
        entity_seq_num_list = self._mmcif_dict["_entity_poly_seq.num"]
        entity_seq_mon_id_list = self._mmcif_dict["_entity_poly_seq.mon_id"]

        polymers = collections.defaultdict(list)
        for entity_id, num, mon_id in tqdm.tqdm(
            zip(entity_seq_entity_id_list, entity_seq_num_list, entity_seq_mon_id_list),
            desc="_entity_seq",
            **self._tqdm_kwargs,
        ):
            entity_id, num = int(entity_id), int(num)
            # if (entity_id, num) in entity_modres_table:
            #     comp_id, parent_id = entity_modres_table[(entity_id, num)]
            #     if mon_id == comp_id:
            #         mon_id = parent_id
            #     else:
            #         raise PDBxError(f"Conflict ouccurs between entity_poly and mod_residue")
            polymers[entity_id].append(Monomer(entity_id=entity_id, num=num, mon_id=mon_id))
        self._polymers = polymers

        # Identify and return the valid protein chains.
        valid_chains = {}
        for entity_id, seq_info in polymers.items():
            chain_ids = self._entity_to_chains[entity_id]

            # Reject polymers without any peptide-like components, such as nucleic acids.
            valid = all(["peptide linking" in self._chem_comps[monomer.mon_id] for monomer in seq_info])

            if valid:
                for chain_id in chain_ids:
                    valid_chains[chain_id] = seq_info
        self._valid_chains = valid_chains

        # Apply MODRES.
        for entity_id, chain in self._polymers.items():
            for i, mono in enumerate(chain):
                if (entity_id, mono.num) in entity_modres_table:
                    comp_id, parent_id = entity_modres_table[(entity_id, mono.num)]
                    mon_id = mono.mon_id
                    if mon_id == comp_id:
                        mon_id = parent_id
                    else:
                        raise PDBxError(f"Conflict ouccurs between entity_poly and mod_residue")
                    chain[i] = Monomer(entity_id=entity_id, num=mono.num, mon_id=mon_id)

        # Generate SEQRES for chains.
        chain_to_seqres = {}
        for chain_id, seq_info in self._valid_chains.items():
            seq = []
            for monomer in seq_info:
                code = protein_letters_3to1.get(monomer.mon_id, "X")
                seq.append(code if len(code) == 1 else "X")
            seq = "".join(seq)
            chain_to_seqres[chain_id] = seq
        self._chain_to_seqres = chain_to_seqres

    def _init_model(self, model_id: int):
        self._current_model = {}
        self._models[model_id] = self._current_model

    def _init_chain(self, chain_id: str):
        if chain_id in self._current_model:
            self._current_chain = self._current_model[chain_id]
            warnings.warn(f"Chain {chain_id} is discontinuous", PDBxWarning)
        else:
            self._current_chain = {}
            self._current_model[chain_id] = self._current_chain

    def _init_residue(self, resname: str, field: str, resseq: int, icode: str):
        # if field != " ":
        # if field == "H":
        # field = "H_" + resname
        res_id = (field, resseq, icode)

        self._current_residue = Residue(name=resname, residue_number=res_id)
        self._current_chain[res_id] = self._current_residue

    def _build_structure(self) -> None:
        # Two special chars as placeholders in the mmCIF format.
        _unassigned = {".", "?"}

        mmcif_dict = self._mmcif_dict

        atom_serial_list = mmcif_dict["_atom_site.id"]
        atom_id_list = mmcif_dict["_atom_site.label_atom_id"]
        residue_id_list = mmcif_dict["_atom_site.label_comp_id"]
        try:
            element_list = mmcif_dict["_atom_site.type_symbol"]
        except KeyError:
            element_list = None
        chain_id_list = mmcif_dict["_atom_site.label_asym_id"]
        x_list = [float(x) for x in mmcif_dict["_atom_site.Cartn_x"]]
        y_list = [float(x) for x in mmcif_dict["_atom_site.Cartn_y"]]
        z_list = [float(x) for x in mmcif_dict["_atom_site.Cartn_z"]]
        alt_list = mmcif_dict["_atom_site.label_alt_id"]
        icode_list = mmcif_dict["_atom_site.pdbx_PDB_ins_code"]
        b_factor_list = mmcif_dict["_atom_site.B_iso_or_equiv"]
        occupancy_list = mmcif_dict["_atom_site.occupancy"]
        fieldname_list = mmcif_dict["_atom_site.group_PDB"]
        try:
            serial_list = [int(n) for n in mmcif_dict["_atom_site.pdbx_PDB_model_num"]]
        except KeyError:
            serial_list = None
        except ValueError:
            raise PDBxError("Invalid model number") from None
        seq_id_list = mmcif_dict["_atom_site.label_seq_id"]

        # Loop over atoms and build the structure
        current_chain_id = None
        current_residue_id = None
        current_resname = None
        current_model_id = -1  # array index
        current_serial_id = -1  # id in the file

        # self._models: Dict[int, dict] = {}
        # self._current_atoms: AtomList = None
        # self._current_chain: dict = None
        # self._current_model: dict = None

        seq_start_num = {chain_id: min([monomer.num for monomer in seq]) for chain_id, seq in self._valid_chains.items()}
        seq_to_struct_mappings = {}
        valid_chain_ids = list(self._valid_chains.keys())

        for i in tqdm.tqdm(
            range(len(atom_id_list)),
            desc="_atom_site",
            **self._tqdm_kwargs,
        ):
            # Try cast serial to int.
            # But do not quit if it fails.
            # mmCIF format specs allow strings.
            try:
                serial = int(atom_serial_list[i])
            except ValueError:
                serial = atom_serial_list[i]
                warnings.warn(f"Some atom serial numbers are not numerical: {serial}", PDBxWarning)

            x = x_list[i]
            y = y_list[i]
            z = z_list[i]
            resname = residue_id_list[i]
            chainid = chain_id_list[i]  # asym_id
            altloc = alt_list[i]
            if altloc in _unassigned:
                altloc = " "
            resseq = seq_id_list[i]
            if resseq == ".":
                # Non-existing residue ID
                try:
                    msg_resseq = mmcif_dict["_atom_site.auth_seq_id"][i]
                    msg = f"Non-existing residue ID in chain '{chainid}', residue '{msg_resseq}'"
                except (KeyError, IndexError):
                    msg = f"Non-existing residue ID in chain '{chainid}'"
                warnings.warn(msg, PDBxWarning)
                continue
            int_resseq = int(resseq)
            icode = icode_list[i]
            for icode in _unassigned:
                icode = " "
            name = atom_id_list[i]
            # Occupancy and B-factor
            try:
                tempfactor = float(b_factor_list[i])
            except ValueError:
                raise PDBxError("Invalid or missing B-factor") from None
            try:
                occupancy = float(occupancy_list[i])
            except ValueError:
                raise PDBxError("Invalid or missing occupancy") from None
            fieldname = fieldname_list[i]
            if fieldname == "HETATM":
                if resname == "HOH" or resname == "WAT":
                    hetatm_flag = "W"
                else:
                    hetatm_flag = "H"
            else:
                hetatm_flag = " "

            resseq = (hetatm_flag, int_resseq, icode)  # Residue

            if serial_list is not None:
                serial_id = serial_list[i]
                if current_serial_id != serial_id:
                    # If serial changes, update it and start new model.
                    current_serial_id = serial_id
                    current_model_id += 1

                    # Initialize model(current_model_id, current_serial_id)
                    self._init_model(current_model_id)
                    seq_to_struct_mappings[current_model_id] = {}

                    current_chain_id = None
                    current_residue_id = None
                    current_resname = None
            else:
                # No explicit model column. Initializie single model
                # Initialize model(current_model_id)
                self._init_model(current_model_id)
                seq_to_struct_mappings[current_model_id] = {}

            if current_chain_id != chainid:
                current_chain_id = chainid
                # Initalize chain(current_chain_id)
                self._init_chain(current_chain_id)

                current_residue_id = None
                current_resname = None

            if current_residue_id != resseq or current_resname != resname:
                current_residue_id = resseq
                current_resname = resname
                # Initialize residue(resname, hetatm_flag, int_resseq, icode)
                self._init_residue(resname, hetatm_flag, int_resseq, icode)

            if chainid in valid_chain_ids:
                seq_idx = int_resseq - seq_start_num[chainid]
                current = seq_to_struct_mappings[current_model_id].get(chainid, {})
                current[seq_idx] = ResidueAtPositon(
                    chain_id=chainid,
                    residue_number=int_resseq,
                    insertion_code=icode,
                    name=resname,
                    is_missing=False,
                    hetflag=hetatm_flag,
                )
                seq_to_struct_mappings[current_model_id][chainid] = current

            coord = np.array((x, y, z), "float")
            element = element_list[i].upper() if element_list else None
            # Initialize atom
            atom = Atom(
                element=element,
                atom_id=name,
                alt_id=altloc,
                occupancy=occupancy,
                b_factor=tempfactor,
                coord=coord,
            )
            self._current_residue.atom_list.append(atom)

        # Add missing residue information to seq_to_structure_mappings.
        for n in self._models:
            for chain_id, seq_info in self._valid_chains.items():
                current_mapping = seq_to_struct_mappings[n][chain_id]
                for idx, monomer in enumerate(seq_info):
                    if idx not in current_mapping:
                        current_mapping[idx] = ResidueAtPositon(
                            is_missing=True,
                            name=monomer.mon_id,
                            hetflag=" ",
                        )

        self._seqres_to_structure = seq_to_struct_mappings

        # Remove unvalid chains. TODO: Move to _atom_site loop.
        for n in self._models:
            to_be_removed = []
            for chain_id in self._models[n]:
                if chain_id not in self._valid_chains.keys():
                    to_be_removed.append(chain_id)
            for chain_id in to_be_removed:
                del self._models[n][chain_id]

    # _pdbx_struct_assembly
    def _get_assemblies(self):

        oper_list_id_list = self._mmcif_dict["_pdbx_struct_oper_list.id"]
        oper_list_type_list = self._mmcif_dict["_pdbx_struct_oper_list.type"]
        oper_list_r11_list = self._mmcif_dict["_pdbx_struct_oper_list.matrix[1][1]"]
        oper_list_r12_list = self._mmcif_dict["_pdbx_struct_oper_list.matrix[1][2]"]
        oper_list_r13_list = self._mmcif_dict["_pdbx_struct_oper_list.matrix[1][3]"]
        oper_list_r21_list = self._mmcif_dict["_pdbx_struct_oper_list.matrix[2][1]"]
        oper_list_r22_list = self._mmcif_dict["_pdbx_struct_oper_list.matrix[2][2]"]
        oper_list_r23_list = self._mmcif_dict["_pdbx_struct_oper_list.matrix[2][3]"]
        oper_list_r31_list = self._mmcif_dict["_pdbx_struct_oper_list.matrix[3][1]"]
        oper_list_r32_list = self._mmcif_dict["_pdbx_struct_oper_list.matrix[3][2]"]
        oper_list_r33_list = self._mmcif_dict["_pdbx_struct_oper_list.matrix[3][3]"]
        oper_list_t1_list = self._mmcif_dict["_pdbx_struct_oper_list.vector[1]"]
        oper_list_t2_list = self._mmcif_dict["_pdbx_struct_oper_list.vector[2]"]
        oper_list_t3_list = self._mmcif_dict["_pdbx_struct_oper_list.vector[3]"]

        oper_list = {}
        for op_id, type, r11, r12, r13, r21, r22, r23, r31, r32, r33, x, y, z in zip(
            oper_list_id_list,
            oper_list_type_list,
            oper_list_r11_list,
            oper_list_r12_list,
            oper_list_r13_list,
            oper_list_r21_list,
            oper_list_r22_list,
            oper_list_r23_list,
            oper_list_r31_list,
            oper_list_r32_list,
            oper_list_r33_list,
            oper_list_t1_list,
            oper_list_t2_list,
            oper_list_t3_list,
        ):
            # op_id = int(op_id)  # str
            if type == "identity operation":
                op = StructOp(op_id=op_id, is_identity=True)
            else:
                r11, r12, r13, r21, r22, r23, r31, r32, r33 = map(float, [r11, r12, r13, r21, r22, r23, r31, r32, r33])
                x, y, z = map(float, [x, y, z])
                rot = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])
                vec = np.array([x, y, z])
                op = StructOp(op_id=op_id, is_identity=False, rot=rot, trans=vec)
            oper_list[op_id] = op

        assembly_id_list = self._mmcif_dict["_pdbx_struct_assembly_gen.assembly_id"]
        oper_expression_list = self._mmcif_dict["_pdbx_struct_assembly_gen.oper_expression"]
        asym_id_list_list = self._mmcif_dict["_pdbx_struct_assembly_gen.asym_id_list"]

        generators = collections.defaultdict(list)
        for assem_id, expr, asym_ids in zip(assembly_id_list, oper_expression_list, asym_id_list_list):
            assem_id = int(assem_id)
            asym_ids = asym_ids.split(",")
            asym_ids = [aid for aid in asym_ids if aid in self._valid_chains]
            ops = parse_oper_expr(expr)

            generators[assem_id].append(
                AssemblyGenerator(
                    oper_list=ops,
                    asym_id_list=asym_ids,
                )
            )

        self._assemblies = dict(generators)
        self._operations = oper_list


def parse_oper_expr(expr: str):
    # Remove whitespaces for easier parsing
    expr = expr.replace(" ", "")

    # Helper function to expand ranges like "1-4" into "1,2,3,4"
    def expand_range(r):
        start, end = map(int, r.split("-"))
        return list(map(str, range(start, end + 1)))

    # Helper function to compute cartesian product
    def cartesian_product(groups):
        result = list(map(list, product(*groups[::-1])))
        return [list(r) for r in result]

    # Tokenize and parse the expression
    tokens = expr.strip("()").split(")(")
    parsed_tokens = []

    for token in tokens:
        if "-" in token:  # Range detected
            ranges = token.split(",")
            expanded_ranges = [expand_range(r) if "-" in r else [r] for r in ranges]
            merged_ranges = [item for sublist in expanded_ranges for item in sublist]
            parsed_tokens.append(merged_ranges)
        else:  # Single numbers or lists
            # parsed_tokens.append(list(map(int, token.split(","))))
            parsed_tokens.append(token.split(","))

    # Compute cartesian product if necessary
    return cartesian_product(parsed_tokens)


### Get mmCIF


@lru_cache(maxsize=16, typed=False)
def fetch_mmcif(rcsb_id: str) -> str:
    """Fetch mmCIF from RCSB."""
    rcsb_id = rcsb_id.lower()
    r = requests.get(
        f"https://files.rcsb.org/download/{rcsb_id}.cif.gz",
        headers={"Accept-Encoding": "gzip"},
    )
    if r.status_code == 200:
        data = r.content
        text = gzip.GzipFile(mode="r", fileobj=BytesIO(data)).read()
        return text
    RuntimeError(f"Cannot fetch '{rcsb_id}'")


def read_mmcif(
    rcsb_id: str,
    mmcif_path: os.PathLike | None = None,
) -> str:
    """Fetch a mmCIF file from local directory or from the web."""

    # Is valid ID?
    assert len(rcsb_id) == 4
    # Use lowercase
    rcsb_id = rcsb_id.lower()
    dv = rcsb_id[1:3]

    mmcif_path = "." if mmcif_path is None else mmcif_path
    mmcif_path = os.path.join(mmcif_path, dv, f"{rcsb_id}.cif.gz")
    if mmcif_path is not None and os.path.exists(mmcif_path):
        return read_text(mmcif_path)
    else:
        return fetch_mmcif(rcsb_id=rcsb_id)


def get_chain_features(
    mmcif_object: MMCIFObject,
    model_num: int,
    chain_id: str,
    mask_modres: bool = False,
) -> Dict[str, np.ndarray]:
    """Get feature dictionary.

    Returns:
        - aatype
        - residue_index
        - seq_length
        - seq_mask
        - all_atom_positions
        - all_atom_mask
    """

    # STRUCT
    chain = mmcif_object.models[model_num][chain_id]

    # MODRES
    modres_table = {}
    if chain_id in mmcif_object.modres:
        for seq_id, mr in mmcif_object.modres[chain_id].items():
            modres_table[seq_id, mr.insertion_code] = (mr.comp_id, mr.parent_comp_id)

    num_res = len(mmcif_object.chain_to_seqres[chain_id])
    aatype = np.full(num_res, rc.unk_restype_index, dtype=np.int64)
    residue_index = np.zeros(num_res, dtype=np.int64)
    seq_length = np.array(num_res, dtype=np.int64)
    seq_mask = np.ones(num_res, dtype=np.float32)
    all_atom_positions = np.zeros([num_res, rc.atom_type_num, 3], dtype=np.float32)
    all_atom_mask = np.zeros([num_res, rc.atom_type_num], dtype=np.float32)

    for res_index in range(num_res):
        pos = np.zeros([rc.atom_type_num, 3], dtype=np.float32)
        mask = np.zeros([rc.atom_type_num], dtype=np.float32)
        res_at_pos = mmcif_object.seqres_to_structure[model_num][chain_id][res_index]

        if not res_at_pos.is_missing:
            res = chain[(res_at_pos.hetflag, res_at_pos.residue_number, res_at_pos.insertion_code)]
            atoms = {x.atom_id: x for x in res.atom_list}

            # Modifiy
            key = (res_at_pos.residue_number, res_at_pos.insertion_code)

            if key in modres_table:
                if mask_modres:
                    seq_mask[res_index] = 0.0

                mod, can = modres_table[key]
                res.name = can  # Rename the residue
                # logger.info(f"{model_num}:{chain_id}:{num_res+1} {mod} -> {can}")
                atom_map = get_atom_map(can, mod)  # Get atom map

                # Rename keys
                for before, after in atom_map.mapping.items():
                    if before != after:
                        if before in atoms.keys():
                            atoms[after] = atoms[before]
                            atoms[after].atom_id = after
                            del atoms[before]

                # Remove atoms
                for atom_name in atom_map.removed:
                    if atom_name in atoms:
                        del atoms[atom_name]

            aatype[res_index] = rc.resname_to_idx.get(res.name, rc.unk_restype_index)

            for atom in atoms.values():
                atom_name = atom.atom_id
                x, y, z = atom.coord
                if atom_name in rc.atom_order.keys():
                    pos[rc.atom_order[atom_name]] = [x, y, z]
                    mask[rc.atom_order[atom_name]] = 1.0

            # Fix naming errors in arginine residues where NH2 incorrectly assigned to be closer to CD and NH1.
            cd = rc.atom_order["CD"]
            nh1 = rc.atom_order["NH1"]
            nh2 = rc.atom_order["NH2"]
            if (
                (res.name == "ARG")
                and all(mask[atom_index] for atom_index in (cd, nh1, nh2))
                and (np.linalg.norm(pos[nh1] - pos[cd]) > np.linalg.norm(pos[nh2] - pos[cd]))
            ):
                pos[nh1], pos[nh2] = pos[nh2].copy(), pos[nh1].copy()
                mask[nh1], mask[nh2] = mask[nh2].copy(), mask[nh1].copy()
        else:
            aatype[res_index] = rc.resname_to_idx.get(res_at_pos.name, rc.unk_restype_index)
            seq_mask[res_index] = 0.0

        all_atom_positions[res_index] = pos
        all_atom_mask[res_index] = mask
        residue_index[res_index] = res_index + 1

    return {
        "aatype": aatype,
        "residue_index": residue_index,
        "seq_length": seq_length,
        "seq_mask": seq_mask,
        "all_atom_positions": all_atom_positions,
        "all_atom_mask": all_atom_mask,
    }


def get_fasta(mmcif_object: MMCIFObject) -> str:
    entry_id = mmcif_object.header.entry_id
    seqres = mmcif_object.chain_to_seqres
    fasta_str = ""
    for k in sorted(seqres):
        fasta_str += f">{entry_id}_{k}\n{seqres[k]}\n"
    return fasta_str


def get_assembly_infos(mmcif_object: MMCIFObject):

    header = mmcif_object.header
    common = {}
    entry_id = header.entry_id
    common["entry_id"] = entry_id
    common["release_date"] = header.deposition_date
    common["method"] = header.method
    common["resolution"] = header.resolution

    model_keys = list(mmcif_object.models.keys())

    operations = {k: v.to_dict() for k, v in mmcif_object.operations.items()}
    assemblies = {}

    for assem_id, assem in mmcif_object.assemblies.items():
        name = f"{entry_id}-{assem_id}"
        asym_ids_needed = []  # Asym units to construct a complex
        op_ids_needed = []
        generators = []  # Sequences of operations
        for gen in assem:
            g = {}
            g["asym_ids"] = gen.asym_id_list
            asym_ids_needed.extend(gen.asym_id_list)
            g["oper_list"] = gen.oper_list
            op_ids_needed.extend(sum(gen.oper_list, []))
            generators.append(g)
        # Clean up asym_ids
        assembly_num_chains = len(asym_ids_needed)
        asym_ids_needed = list(set(asym_ids_needed))
        asym_ids_needed.sort()
        # Clean up operations
        op_ids_needed = list(set(op_ids_needed))
        op_ids_needed.sort()

        assemblies[name] = {
            "name": name,
            "assembly_num_chains": assembly_num_chains,
            "generators": generators,
            "asym_ids": asym_ids_needed,
            "oper_list": {k: operations[k] for k in op_ids_needed},
        }

    # NOTE: JSON does not allow numerical keys.
    return assemblies
