# Copyright 2024 DeepFold Team


"""Protein Data Bank."""


import collections
import functools
import gzip
import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from io import BytesIO, StringIO
from itertools import product
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import requests
from Bio.Data.PDBData import protein_letters_3to1_extended as protein_letters_3to1

# from Bio.Data.PDBData import protein_letters_3to1
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.PDBExceptions import PDBConstructionException, PDBConstructionWarning
from Bio.PDB.Structure import Structure
from Bio.PDB.StructureBuilder import StructureBuilder
from tqdm import tqdm

from deepfold.common import residue_constants as rc
from deepfold.data.errors import PDBxConstructionError, PDBxConstructionWarning, PDBxError, PDBxWarning
from deepfold.data.monomer import build_atom_map
from deepfold.utils.file_utils import read_text

logger = logging.getLogger(__name__)


class CategoryNotFoundError(PDBxError):
    pass


UNASSIGNED = {".", "?"}
LATIN = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DEFAULT_ATOM_MAP = {
    ("MSE", "MET"): {"N": "N", "CA": "CA", "C": "C", "O": "O", "OXT": "OXT", "CB": "CB", "CG": "CG", "SE": "SD", "CE": "CE"},
    ("UNK", "UNK"): {"N": "N", "CA": "CA", "C": "C", "O": "O", "OXT": "OXT", "CB": "CB"},
}  # TODO: Implement ATOM_MAP


def loop_to_list(prefix: str, dic: Mapping[str, Sequence[str]]) -> Sequence[Mapping[str, str]]:
    cols = []
    data = []
    for key, value in dic.items():
        if key.startswith(prefix):
            cols.append(key)
            data.append(value)

    if not all([len(xs) == len(data[0]) for xs in data]):
        raise PDBxConstructionError(f"Not all loops are the same length: {cols}")

    return [dict(zip(cols, xs)) for xs in zip(*data)]


def loop_to_dict(prefix: str, index: str, dic: Mapping[str, Sequence[str]]) -> Mapping[str, Mapping[str, str]]:
    entries = loop_to_list(prefix, dic)
    return {entry[index]: entry for entry in entries}


# _entity_poly_seq
@dataclass(frozen=True)
class Monomer:
    entity_id: str
    num: int
    mon_id: str


@dataclass(frozen=True)
class AtomSite:
    residue_name: str
    author_chain_id: str
    label_chain_id: str
    author_seq_num: str
    label_seq_num: int
    insertion_code: str
    hetatm_atom: str
    model_num: int


# Used to map SEQRES index to a residue in the structure.
@dataclass(frozen=True)
class ResiduePosition:
    label_asym_id: str
    auth_asym_id: str
    label_seq_id: int
    auth_seq_id: int
    insertion_code: str


@dataclass(frozen=True)
class ResidueAtPosition:
    position: ResiduePosition | None
    name: str
    is_missing: bool
    hetflag: str
    residue_index: int
    ordinal: int = 0  # For hetero-residues


# _pdbx_struct_mod_residue
# NOTE: PDB_model_num not used
@dataclass(frozen=True)
class ModResidue:
    label_asym_id: str
    label_comp_id: str
    label_seq_id: int
    insertion_code: str
    parent_comp_id: List[str] = field(default_factory=list)


# _pdbx_struct_oper_list
@dataclass(frozen=True)
class Op:
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


# _pdbx_struct_assembly_gen
@dataclass(frozen=True)
class AssemblyGenerator:
    assembly_id: str
    asym_id_list: List[str]
    oper_sequence: List[str]


@dataclass(frozen=True)
class PDBxHeader:
    # Required
    entry_id: str  # _entry.id
    deposition_date: str  # _pdbx_database_status.recvd_initial_deposition_date
    method: str  # _exptl.method
    resolution: float | None = None


@dataclass(frozen=True)
class PDBxObject:
    header: PDBxHeader
    structure: Structure
    chain_ids: List[str]
    modres: Dict[Tuple[str, str, str], ModResidue]
    label_to_auth: Dict[str, str]
    auth_to_label: Dict[str, str]
    chain_to_entity: Dict[str, str]
    # chain_to_seqres: Dict[str, str]
    chain_to_structure: Dict[str, Dict[int, ResidueAtPosition]]
    assemblies: Dict[int, List[AssemblyGenerator]]
    operations: Dict[int, Op]


@dataclass(frozen=True)
class ParsingResult:
    mmcif_object: PDBxObject | None = None
    errors: List[PDBxError] = field(default_factory=list)


class PDBxParser:
    """Parse a PDBx/mmCIF record."""

    def __init__(self) -> None:
        self.mmcif_dict: Dict[str, List[str]]

        # Header
        self.header = None

        # Structure
        self.structure: Structure = None

        # Maps mmCIF chain ids to chain ids used the authors.
        self.label_to_auth = None
        # Maps index into sequence to ResidueAtPosition.
        self.chain_to_structure = None

        #
        self.chain_to_seqres = None

        #
        self.entity_to_chains = None
        self.chain_to_entity = None
        self.valid_chains = None

        #
        self.modified_residues = None

        # Assembly
        self.assemblies: Dict[int, List[AssemblyGenerator]] = None
        self.operations: Dict[int, Op] = None

    def parse(self, mmcif_string: str, catch_all_errors: bool = False) -> ParsingResult:
        errors = []
        try:
            self.mmcif_dict = MMCIF2Dict(StringIO(mmcif_string))
            self._parse()
            self._handle_missing_residues()
            self._process_modres()

            pdbx_object = PDBxObject(
                header=self.header,
                structure=self.structure,
                chain_ids=sorted(list(self.valid_chains.keys())),
                modres=self.modified_residues,
                label_to_auth=self.label_to_auth,
                auth_to_label={y: x for x, y in self.label_to_auth.items()},
                chain_to_entity=self.chain_to_entity,
                chain_to_structure=self.chain_to_structure,
                assemblies=self.assemblies,
                operations=self.operations,
            )
            return ParsingResult(mmcif_object=pdbx_object, errors=errors)
        except Exception as e:
            errors.append(e)
            if not catch_all_errors:
                raise
            return ParsingResult(errors=errors)

    # NOTE: Do before run `parse()`
    def inject(parents_table: Dict[str, Dict[str, Any]] | None = None):
        pass

    def _parse(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=PDBConstructionWarning)

            self.header = self._get_header(self.mmcif_dict)
            self._get_protein_chains()  # valid_chains, entity_to_chains

            self.structure, self.label_to_auth, self.chain_to_structure = self._build_structure(self.mmcif_dict)
            self._get_mod_residue()  # modified_residues

            self.assemblies, self.operations = self._get_assemblies(self.mmcif_dict, self.valid_chains)

    def _handle_missing_residues(self):
        missing_chains = []
        for chain_id, seq_info in self.valid_chains.items():
            try:
                current_mapping = self.chain_to_structure[chain_id]
            except KeyError as key:
                logger.info(f"Chain {chain_id} has no structure. Remove...")
                missing_chains.append(chain_id)
                continue

            # Add missing residue information to seq_to_structure_mappings.
            for monomer in seq_info:
                idx = monomer.num
                if idx not in current_mapping:
                    current_mapping[idx] = ResidueAtPosition(
                        position=None,
                        name=monomer.mon_id,
                        is_missing=True,
                        hetflag=" ",
                        residue_index=idx,
                    )

        for chain_id in missing_chains:
            del self.valid_chains[chain_id]

    # Filter modres records which are used.
    def _process_modres(self):
        modres_processed = {}
        for chain_id, seq_info in self.valid_chains.items():
            seq_info = {m.num: m for m in seq_info}
            for value in self.modified_residues.values():
                if (value.label_asym_id == chain_id) and (value.label_comp_id == seq_info[value.label_seq_id].mon_id):
                    modres_processed[(chain_id, value.label_seq_id)] = value
        self.modified_residues = modres_processed

    @staticmethod
    def _get_header(mmcif_dict):
        header = {"entry_id": "", "deposition_date": "", "method": "", "resolution": 0.0}

        def update_header_entry(target_key, keys):
            md = mmcif_dict
            for key in keys:
                val = md.get(key)
                try:
                    item = val[0]
                except (TypeError, IndexError):
                    continue
                if item != "?" and item != ".":
                    header[target_key] = item
                    break

        update_header_entry("entry_id", ["_entry_id", "_exptl.entry_id", "_struct.entry_id"])
        update_header_entry("deposition_date", ["_pdbx_database_status.recvd_initial_deposition_date"])
        update_header_entry("method", ["_exptl.method"])
        update_header_entry("resolution", ["_refine.ls_d_res_high", "_refine_hist.d_res_high", "_em_3d_reconstruction.resolution"])

        # NOTE: NMR result has empty resolution.
        header["resolution"] = float(header["resolution"])

        return PDBxHeader(**header)

    def _get_protein_chains(self, only_valid: bool = True):
        mmcif_dict = self.mmcif_dict

        # _entity_poly
        try:
            entity_poly_seqs = loop_to_list("_entity_poly_seq.", mmcif_dict)
        except KeyError:
            raise PDBxConstructionError("Cannot find `_entity_poly`")

        polymers = collections.defaultdict(list)
        for entity_poly_seq in entity_poly_seqs:
            polymers[entity_poly_seq["_entity_poly_seq.entity_id"]].append(
                Monomer(
                    mon_id=entity_poly_seq["_entity_poly_seq.mon_id"],
                    num=int(entity_poly_seq["_entity_poly_seq.num"]),
                    entity_id=entity_poly_seq["_entity_poly_seq.entity_id"],
                )
            )

        # Get chemical compositions.
        chem_comps = loop_to_dict("_chem_comp.", "_chem_comp.id", mmcif_dict)

        # Get chains information for each entity.
        struct_asyms = loop_to_list("_struct_asym.", mmcif_dict)

        entity_to_chains = collections.defaultdict(list)
        for struct_asym in struct_asyms:
            chain_id = struct_asym["_struct_asym.id"]
            entity_id = struct_asym["_struct_asym.entity_id"]
            entity_to_chains[entity_id].append(chain_id)

        # Identify and return the valid portein chains.
        valid_chains = {}
        for entity_id, seq_info in polymers.items():
            chain_ids = entity_to_chains[entity_id]

            # Reject polymers without any peptide-like componenets, such as DNA/RNA.
            if any(["peptide" in chem_comps[monomer.mon_id]["_chem_comp.type"].lower() for monomer in seq_info]):
                for chain_id in chain_ids:
                    valid_chains[chain_id] = seq_info

        chain_to_entity = {}
        for k, v in entity_to_chains.items():
            for x in v:
                if x in valid_chains:
                    chain_to_entity[x] = k

        self.entity_to_chains = entity_to_chains
        self.chain_to_entity = chain_to_entity
        self.valid_chains = valid_chains

    def _get_mod_residue(self):
        mod_residues = loop_to_list("_pdbx_struct_mod_residue.", self.mmcif_dict)

        modified_residues = {}
        for modres in mod_residues:
            asym_id = modres["_pdbx_struct_mod_residue.label_asym_id"]
            seq_id = modres["_pdbx_struct_mod_residue.label_seq_id"]
            comp_id = modres["_pdbx_struct_mod_residue.label_comp_id"]
            parent_comp_id = modres["_pdbx_struct_mod_residue.parent_comp_id"]
            insertion_code = modres["_pdbx_struct_mod_residue.PDB_ins_code"]

            if asym_id in {".", "?"}:
                asym_id = " "

            if asym_id not in self.valid_chains:
                logger.debug(f"Skip {asym_id}:{seq_id}{insertion_code} {comp_id}...")
                continue

            seq_id = int(seq_id)

            if insertion_code in {".", "?"}:
                insertion_code = " "

            key = (asym_id, seq_id, insertion_code)

            if key in modified_residues:
                row = dict(vars(modified_residues[key]))
                row["parent_comp_id"].append(parent_comp_id)
                row = ModResidue(**row)
            else:
                row = ModResidue(
                    label_asym_id=asym_id,
                    label_comp_id=comp_id,
                    label_seq_id=seq_id,
                    insertion_code=insertion_code,
                    parent_comp_id=[parent_comp_id],
                )

            modified_residues[key] = row

        self.modified_residues = modified_residues

    @staticmethod
    def _build_structure(mmcif_dict):
        structure_builder = StructureBuilder()

        atom_serial_list = mmcif_dict["_atom_site.id"]
        atom_id_list = mmcif_dict["_atom_site.label_atom_id"]
        residue_id_list = mmcif_dict["_atom_site.label_comp_id"]
        try:
            element_list = mmcif_dict["_atom_site.type_symbol"]
        except KeyError:
            element_list = None
        auth_chain_id_list = mmcif_dict["_atom_site.auth_asym_id"]
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
            # No model number column
            serial_list = None
        except ValueError:
            # Invalid model number (malformed file)
            raise PDBxConstructionError("Invalid model number") from None
        auth_seq_id_list = mmcif_dict["_atom_site.auth_seq_id"]
        seq_id_list = mmcif_dict["_atom_site.label_seq_id"]

        # Now loop over atoms and build the structure
        label_to_author_chain_id = {}
        seq_to_structure_mappings = {}

        current_chain_id = None
        current_residue_id = None
        current_resname = None
        structure_builder.init_structure("")
        structure_builder.init_seg(" ")

        current_model_id = -1
        current_serial_id = -1
        for i in range(len(atom_serial_list)):
            # set the line_counter for 'ATOM' lines only and not
            # as a global line counter found in the PDBParser()
            structure_builder.set_line_counter(i)

            # Try coercing serial to int, for compatibility with PDBParser
            # But do not quit if it fails. mmCIF format specs allow strings.
            try:
                serial = int(atom_serial_list[i])
            except ValueError:
                serial = atom_serial_list[i]
                warnings.warn(f"Some atom serial numbers ({serial}) are not numerical", PDBConstructionWarning)

            x = x_list[i]
            y = y_list[i]
            z = z_list[i]
            resname = residue_id_list[i]
            chain_id = chain_id_list[i]
            auth_chain_id = auth_chain_id_list[i]
            altloc = alt_list[i]
            if altloc in UNASSIGNED:
                altloc = " "
            resseq = seq_id_list[i]
            if resseq == ".":
                # Non-existing residue ID
                try:
                    msg_resseq = mmcif_dict["_atom_site.auth_seq_id"][i]
                    msg = f"Non-existing residue ID in chain '{chain_id}', residue '{msg_resseq}'"
                except (KeyError, IndexError):
                    msg = f"Non-existing residue ID in chain '{chain_id}'"
                warnings.warn(msg, PDBxConstructionWarning)
                continue
            int_resseq = int(resseq)
            icode = icode_list[i]
            if icode in UNASSIGNED:
                icode = " "
            name = atom_id_list[i]
            # occupancy & B factor
            try:
                tempfactor = float(b_factor_list[i])
            except ValueError:
                raise PDBxConstructionError("Invalid or missing B factor") from None
            try:
                occupancy = float(occupancy_list[i])
            except ValueError:
                raise PDBxConstructionError("Invalid or missing occupancy") from None
            fieldname = fieldname_list[i]
            if fieldname == "HETATM":
                if resname == "HOH" or resname == "WAT":
                    hetatm_flag = "W"
                else:
                    hetatm_flag = " "  # "H"
            else:
                hetatm_flag = " "

            #
            position = ResiduePosition(
                label_asym_id=chain_id,
                auth_asym_id=auth_chain_id,
                label_seq_id=int_resseq,
                auth_seq_id=int(auth_seq_id_list[i]),
                insertion_code=icode,
            )
            current = seq_to_structure_mappings.get(chain_id, {})
            current[int_resseq] = ResidueAtPosition(
                position=position,
                name=resname,
                is_missing=False,
                hetflag=hetatm_flag,
                residue_index=int_resseq,
            )
            seq_to_structure_mappings[chain_id] = current

            resseq = (hetatm_flag, int_resseq, icode)

            if serial_list is not None:
                # model column exists; use it
                serial_id = serial_list[i]
                if current_serial_id != serial_id:
                    # if serial changes, update it and start new model
                    current_serial_id = serial_id
                    current_model_id += 1
                    structure_builder.init_model(current_model_id, current_serial_id)
                    current_chain_id = None
                    current_residue_id = None
                    current_resname = None
            else:
                # no explicit model column; initialize single model
                structure_builder.init_model(current_model_id)

            if current_chain_id != chain_id:
                current_chain_id = chain_id
                structure_builder.init_chain(current_chain_id)
                label_to_author_chain_id[chain_id] = auth_chain_id  # Author chain id
                current_residue_id = None
                current_resname = None

            if current_residue_id != resseq or current_resname != resname:
                current_residue_id = resseq
                current_resname = resname
                structure_builder.init_residue(resname, hetatm_flag, int_resseq, icode)

            coord = np.array((x, y, z), "f")
            element = element_list[i].upper() if element_list else None
            structure_builder.init_atom(
                name,
                coord,
                tempfactor,
                occupancy,
                altloc,
                name,
                serial_number=serial,
                element=element,
            )

        return structure_builder.get_structure(), label_to_author_chain_id, seq_to_structure_mappings

    @staticmethod
    def _get_assemblies(mmcif_dict, valid_chains):

        oper_list_id_list = mmcif_dict["_pdbx_struct_oper_list.id"]
        oper_list_type_list = mmcif_dict["_pdbx_struct_oper_list.type"]
        oper_list_r11_list = mmcif_dict["_pdbx_struct_oper_list.matrix[1][1]"]
        oper_list_r12_list = mmcif_dict["_pdbx_struct_oper_list.matrix[1][2]"]
        oper_list_r13_list = mmcif_dict["_pdbx_struct_oper_list.matrix[1][3]"]
        oper_list_r21_list = mmcif_dict["_pdbx_struct_oper_list.matrix[2][1]"]
        oper_list_r22_list = mmcif_dict["_pdbx_struct_oper_list.matrix[2][2]"]
        oper_list_r23_list = mmcif_dict["_pdbx_struct_oper_list.matrix[2][3]"]
        oper_list_r31_list = mmcif_dict["_pdbx_struct_oper_list.matrix[3][1]"]
        oper_list_r32_list = mmcif_dict["_pdbx_struct_oper_list.matrix[3][2]"]
        oper_list_r33_list = mmcif_dict["_pdbx_struct_oper_list.matrix[3][3]"]
        oper_list_t1_list = mmcif_dict["_pdbx_struct_oper_list.vector[1]"]
        oper_list_t2_list = mmcif_dict["_pdbx_struct_oper_list.vector[2]"]
        oper_list_t3_list = mmcif_dict["_pdbx_struct_oper_list.vector[3]"]

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
                op = Op(op_id=op_id, is_identity=True)
            else:
                r11, r12, r13, r21, r22, r23, r31, r32, r33 = map(float, [r11, r12, r13, r21, r22, r23, r31, r32, r33])
                x, y, z = map(float, [x, y, z])
                rot = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])
                vec = np.array([x, y, z])
                op = Op(op_id=op_id, is_identity=False, rot=rot, trans=vec)
            oper_list[op_id] = op

        assembly_id_list = mmcif_dict["_pdbx_struct_assembly_gen.assembly_id"]
        oper_expression_list = mmcif_dict["_pdbx_struct_assembly_gen.oper_expression"]
        asym_id_list_list = mmcif_dict["_pdbx_struct_assembly_gen.asym_id_list"]

        generators = collections.defaultdict(list)
        for assem_id, expr, asym_ids in zip(assembly_id_list, oper_expression_list, asym_id_list_list):
            assem_id = int(assem_id)
            asym_ids = asym_ids.split(",")
            asym_ids = [aid for aid in asym_ids if aid in valid_chains]
            op_seq = _parse_oper_expr(expr)

            generators[assem_id].append(
                AssemblyGenerator(
                    assembly_id=assem_id,
                    asym_id_list=asym_ids,
                    oper_sequence=op_seq,
                )
            )

        return dict(generators), oper_list


def _parse_oper_expr(expr: str):
    # Remove whitespaces for easier parsing
    expr = expr.replace(" ", "")

    # Helper function to expand ranges like "1-4" into "1,2,3,4"
    def expand_range(r):
        start, end = map(int, r.split("-"))
        return list(map(str, range(start, end + 1)))

    # Helper function to compute cartesian product
    def cartesian_product(groups):
        # NOTE: Apply from right to left.
        # if len(groups) == 1:
        # return [groups[0][::-1]]
        return [list(r) for r in product(*groups[::-1])]

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


## Fetch PDB records


@functools.lru_cache(maxsize=16, typed=False)
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
        return text.decode()
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


# Parse PDB records


def get_fasta(mmcif_object: PDBxObject) -> str:
    entry_id = mmcif_object.header.entry_id.lower()
    seqres = mmcif_object.chain_to_structure
    label_to_auth = mmcif_object.label_to_auth
    modres = mmcif_object.modres

    fasta_str = ""
    for asym_id in sorted(seqres):
        seq = []
        for seq_id in sorted(seqres[asym_id]):
            key = (asym_id, seq_id)
            if key in modres:  # Apply MODRES
                seq.extend(modres[key].parent_comp_id)
            else:
                seq.append(seqres[asym_id][seq_id].name)

        for i, s in enumerate(seq):
            code = protein_letters_3to1.get(s, "X")
            seq[i] = code if len(code) == 1 else "X"

        seq = "".join(seq)
        auth_id = label_to_auth[asym_id]
        fasta_str += f">{entry_id}_{asym_id} | {entry_id}_{auth_id}\n"
        fasta_str += f"{seq}\n"

    return fasta_str


def get_assemblies(mmcif_object: PDBxObject):
    header = mmcif_object.header
    common = {}
    common["entry_id"] = header.entry_id
    common["release_date"] = header.deposition_date
    common["method"] = header.method
    common["resolution"] = header.resolution

    assemblies = {}
    oper_list = {k: v.to_dict() for k, v in mmcif_object.operations.items()}

    for aid, assembly in mmcif_object.assemblies.items():
        name = f"{header.entry_id}-{aid}"
        asym_ids_needed = []  # Asym units need to construct a complex.
        op_ids_needed = []
        generators = []  # Sequences of operations.

        for generator in assembly:
            g = {}
            g["asym_id_list"] = generator.asym_id_list
            g["oper_list"] = generator.oper_sequence
            asym_ids_needed.extend(generator.asym_id_list * len(generator.oper_sequence))
            op_ids_needed.extend(sum(generator.oper_sequence, []))
            generators.append(g)

        # Clean up operations
        op_ids_needed = list(set(op_ids_needed))
        op_ids_needed.sort()

        # Oligomeric state
        asym_counter = collections.Counter(asym_ids_needed)
        entity_counter = collections.defaultdict(int)
        chain_to_entity = mmcif_object.chain_to_entity
        for chain_id, num in asym_counter.items():
            entity_id = chain_to_entity[chain_id]
            entity_counter[entity_id] += num

        assemblies[name] = {
            **common,
            "assembly_id": name,
            "assembly_num_chains": len(asym_ids_needed),
            "generators": generators,
            "oper_list": {k: oper_list[k] for k in op_ids_needed},
            "oligomeric_state": dict(sorted((k, v) for k, v in entity_counter.items())),
            "label_to_auth": {x: y for x, y in mmcif_object.label_to_auth.items() if x in asym_ids_needed},
        }

    return assemblies


def get_chain_features(
    mmcif_object: PDBxObject,
    model_num: int,
    chain_id: str,
    out_chain_id: str | None = None,
) -> Tuple[Dict[str, np.ndarray], Structure]:
    """Get atom positions and mask from a list of Biopython Residues."""

    builder = StructureBuilder()
    builder.set_header(f"{mmcif_object.header}")
    builder.init_structure(structure_id=f"{mmcif_object.header.entry_id}_{chain_id}")
    builder.init_model(model_id=model_num)
    builder.init_seg(" ")
    builder.init_chain(chain_id=chain_id if out_chain_id is None else out_chain_id)

    model_iter = mmcif_object.structure.get_models()
    for _ in range(model_num):
        model = next(model_iter)
    relevant_chains = [c for c in model.get_chains() if c.id == chain_id]
    if len(relevant_chains) != 1:
        raise PDBxConstructionError(f"Expected exactly one chain in structure with id {chain_id}")
    chain = relevant_chains[0]

    modres = {m.label_seq_id: m for m in mmcif_object.modres.values() if m.label_asym_id == chain_id}
    max_rid = len(mmcif_object.chain_to_structure[chain_id])
    num_res = max_rid + sum(len(m.parent_comp_id) - 1 for m in modres.values())

    aatype = np.full(num_res, rc.unk_restype_index, dtype=np.int64)
    residue_index = np.zeros(num_res, dtype=np.int64)
    seq_length = np.array(num_res, dtype=np.int64)
    seq_mask = np.ones(num_res, dtype=np.float32)
    all_atom_positions = np.zeros([num_res, rc.atom_type_num, 3])
    all_atom_mask = np.zeros([num_res, rc.atom_type_num], dtype=np.int64)

    rid = min(mmcif_object.chain_to_structure[chain_id].keys())  # First index of chain_to_structure
    aid = 0  # Array index must be 0
    shift = rid

    while rid <= max_rid:
        offset = 1  # Default
        rap = mmcif_object.chain_to_structure[chain_id][rid]

        def fill_features(res, array_index, builder, atom_map=None):
            pos = np.zeros([rc.atom_type_num, 3], dtype=np.float32)
            mask = np.zeros([rc.atom_type_num], dtype=np.float32)

            for atom in res.get_atoms():
                atom_name = atom.get_name()
                if atom_map is not None:
                    if atom_name in atom_map:
                        atom_name = atom_map[atom_name]
                    else:
                        continue  # TODO: Really?

                # Get only canonical atoms
                if atom_name in rc.atom_order.keys():
                    x, y, z = atom.get_coord()
                    pos[rc.atom_order[atom_name]] = [x, y, z]
                    mask[rc.atom_order[atom_name]] = 1.0
                    builder.init_atom(
                        name=atom_name,
                        coord=[x, y, z],
                        b_factor=atom.get_bfactor(),
                        occupancy=atom.get_occupancy(),
                        altloc=atom.get_altloc(),
                        fullname=f"{atom_name:^4s}",
                        element=atom_name[0],
                    )

            # Fixing naming errors in arginine residues where NH2 is incorrectly assigned to be closer to CD than NH1.
            if res.get_resname() == "ARG":
                _fix_arg(pos, mask)

            all_atom_positions[array_index] = pos
            all_atom_mask[array_index] = mask
            residue_index[array_index] = array_index + shift

        if not rap.is_missing:
            assert rap.position is not None
            key = (rap.hetflag, rap.position.label_seq_id, rap.position.insertion_code)
            try:
                res = chain[key]
            except KeyError as e:
                logger.error(f"Model {model_num} does not have residue {key} in chain {chain_id}")
                # Missing residues
                aatype[aid] = rc.resname_to_idx.get(rap.name, rc.unk_restype_index)
                builder.init_residue(rap.name, field=" ", resseq=rid, icode=" ")
                seq_mask[aid] = 0.0  # Initially one.
                residue_index[aid] = aid + shift
                rid += 1
                aid += offset
                continue

            resname = res.get_resname()

            if key[1] in modres:  # Modified residues
                mr = modres[key[1]]
                offset = len(mr.parent_comp_id)
                for i, resname_parent in enumerate(mr.parent_comp_id):
                    aatype[aid + i] = rc.resname_to_idx.get(resname_parent, rc.unk_restype_index)
                    # icode = " " if offset == 1 else LATIN[i]
                    icode = LATIN[i]
                    builder.init_residue(resname=resname_parent, field=" ", resseq=rid, icode=icode)
                    atom_map = DEFAULT_ATOM_MAP.get((resname, resname_parent), None)
                    if atom_map is None:
                        atom_map = DEFAULT_ATOM_MAP[("UNK", "UNK")]
                    fill_features(res, aid + i, builder, atom_map=atom_map)
            else:  # Otherwise
                aatype[aid] = rc.resname_to_idx.get(resname, rc.unk_restype_index)
                builder.init_residue(resname, field=" ", resseq=rid, icode=" ")
                fill_features(res, aid, builder)
        else:  # Missing residues
            aatype[aid] = rc.resname_to_idx.get(rap.name, rc.unk_restype_index)
            builder.init_residue(rap.name, field=" ", resseq=rid, icode=" ")
            seq_mask[aid] = 0.0  # Initially one.
            residue_index[aid] = aid + shift

        rid += 1
        aid += offset

    # Casting for saving storage:
    return {
        "aatype": aatype.astype(np.int8),
        "residue_index": residue_index.astype(np.int32),
        "seq_length": seq_length.astype(np.int32),
        "seq_mask": seq_mask.astype(np.int8),
        "all_atom_positions": all_atom_positions.astype(np.float32),
        "all_atom_mask": all_atom_mask.astype(np.int8),
    }, builder.get_structure()


def _fix_arg(pos, mask):
    cd = rc.atom_order["CD"]
    nh1 = rc.atom_order["NH1"]
    nh2 = rc.atom_order["NH2"]
    if all(mask[atom_index] for atom_index in (cd, nh1, nh2)) and (np.linalg.norm(pos[nh1] - pos[cd]) > np.linalg.norm(pos[nh2] - pos[cd])):
        pos[nh1], pos[nh2] = pos[nh2].copy(), pos[nh1].copy()
        mask[nh1], mask[nh2] = mask[nh2].copy(), mask[nh1].copy()


def print_chain_features(feats, file=sys.stdout):
    first_col_width = 7 + int(math.log10(len(feats["aatype"])))
    header = [f"{x:>3s}" for x in rc.atom_types]
    for i in range(3):
        print(" " * first_col_width, *[f"{s[i]}" for s in header], sep="|", end="|\n", file=file)
    print("-" * first_col_width, *["-" for _ in range(len(header))], sep="+", end="+\n", file=file)
    for resid, aatype, mask, flag in zip(
        feats["residue_index"],
        feats["aatype"],
        feats["all_atom_mask"],
        feats["seq_mask"],
    ):
        std_mask = rc.STANDARD_ATOM_MASK[aatype]
        first_col = f"{resid} {' ' if flag else '!'} {rc.restype_1to3.get(rc.restypes_with_x[aatype], rc.unk_restype)}"
        print(f"{first_col:>{first_col_width}s}", end="|", file=file)
        for x, y in zip(mask, std_mask):
            if x and y:
                b = "O"
            if not x and y:
                b = "X"
            if x and not y:
                b = "?"
            if not x and not y:
                b = " "
            print(f"{b}", end="|", file=file)
        print(file=file)


def print_amap(mmcif_object: PDBxObject, auth_asym_id: str, file=sys.stdout):
    # raise NotImplementedError()

    chain_id = mmcif_object.auth_to_label[auth_asym_id]
    feats, structure = get_chain_features(mmcif_object, model_num=1, chain_id=chain_id, out_chain_id=auth_asym_id)
    chain = next(structure.get_models())[chain_id]
    mask = feats["all_atom_mask"]

    # Backbone mask
    bb_mask = np.zeros(rc.atom_type_num, dtype="bool")
    for i in [rc.atom_order[x] for x in ("N", "CA", "C", "O")]:
        bb_mask[i] = True

    bb = 0
    ca = 0
    b1 = 0

    ca_order = rc.atom_order["CA"]
    # for i, (_, rap) in enumerate(residues):
    for i, res in enumerate(chain.get_residues()):

        _, resnum, ins_code = res.id
        resname = res.get_resname()
        one = protein_letters_3to1.get(resname, "X")

        nbb = sum(mask[i][bb_mask])

        if nbb == 0:  # No backbone
            print(f"{i+1:>5} {one:1}    -    -    - 0       -   -   - - {mask[i]}", file=file)
            continue

        bb_count = 0
        if nbb == 4:  # If the backbone complete
            bb += 1
            bb_str = f"{bb:4}"
            bb_count += 1
        else:
            bb_str = "   -"

        if mask[i][ca_order]:  # If CA exists
            ca += 1
            ca_str = f"{ca:4}"
            bb_count += 1
        else:
            ca_str = "   -"

        if nbb > 0:  # If atom exists
            b1 += 1
            b1_str = f"{b1:4}"
            bb_count += 1
        else:
            b1_str = "   -"

        a = resname  # SEQRES
        b = protein_letters_3to1.get(a, a)
        if len(b) == 1 or len(b) == 4:
            b = a
        c = one

        line = f"{i+1:>5} {one} {bb_str} {ca_str} {b1_str} {bb_count:1}   {resnum:>4}{ins_code} {a:3} {b:>3} {c:1} {mask[i]}"
        print(line, file=file)
