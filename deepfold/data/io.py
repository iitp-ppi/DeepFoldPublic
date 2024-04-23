"""Output of PDB files."""

from typing import Iterable

from Bio.Data.IUPACData import atom_weights
from Bio.PDB.StructureBuilder import StructureBuilder

_ATOM_FORMAT_STRING = "%s%5i %-4s%c%3s %c%4i%c   %8.3f%8.3f%8.3f%s%6.2f      %4s%2s%2s\n"
_TER_FORMAT_STRING = "TER   %5i      %3s %c%4i%c                                                      \n"
_REMARK_FORMAT_STRING = "REMARK   {:<70s}\n"


class IO:
    """Base class."""

    def __init__(self) -> None:
        pass

    def set_structure(self, pdb_object) -> None:
        """Set structures. Precedence Structures are overrideds."""
        if pdb_object.level == "S":
            structure = pdb_object
        else:  # Not a Structure
            sb = StructureBuilder()
            sb.init_structure("")
            sb.init_seg(" ")

            if pdb_object.level == "M":
                sb.structure.add(pdb_object.copy())
                self.structure = sb.structure
            else:  # Not a Model
                sb.init_model(0)

                if pdb_object.level == "C":
                    sb.structure[0].add(pdb_object.copy())
                else:  # Not a Chain
                    raise AttributeError(f"{pdb_object} is not one of Structure, Model, or Chain. ({pdb_object.level})")
            structure = sb.get_structure()
        self.structure = structure

    def set_chains(self, *chains) -> None:
        sb = StructureBuilder()
        sb.init_structure("")
        sb.init_seg(" ")
        sb.init_model(0)
        for chain in chains:
            if chain.level == "C":
                sb.structure[0].add(chain.copy())
            else:
                AttributeError(f"{chain} is not a Chain instance. ({chain.level})")
        self.structure = sb.get_structure()


class PDBIO(IO):
    """Write a Structure object as a PDB file."""

    def __init__(self, use_model_flag: int = 0, remarks: str | Iterable[str] = "") -> None:
        super().__init__()

        if isinstance(remarks, str):
            remarks = [remarks]
        remarks = list(remarks)

        self.use_model_flag = use_model_flag
        self.remarks = remarks

    def _get_atom_line(
        self,
        atom,
        hetfield,
        segid,
        atom_number,
        resname,
        resseq,
        icode,
        chain_id,
        charge="  ",
    ) -> str:
        """Return an ATOM PDB string."""
        if hetfield != " ":
            record_type = "HETATM"
        else:
            record_type = "ATOM  "

        # Atom properties

        # Check if the atom serial number is an integer
        # Not always the case for structures built from
        # mmCIF files.
        try:
            atom_number = int(atom_number)
        except ValueError:
            raise ValueError(
                f"{atom_number!r} is not a number."
                "Atom serial numbers must be numerical"
                " If you are converting from an mmCIF"
                " structure, try using"
                " preserve_atom_numbering=False"
            )

        if atom_number > 99999:
            raise ValueError(f"Atom serial number ('{atom_number}') exceeds PDB format limit.")

        # Check if the element is valid, unknown (X), or blank
        if atom.element:
            element = atom.element.strip().upper()
            if element.capitalize() not in atom_weights and element != "X":
                raise ValueError(f"Unrecognised element {atom.element}")
            element = element.rjust(2)
        else:
            element = "  "

        # Format atom name
        # Pad if:
        #     - smaller than 4 characters
        # AND - is not C, N, O, S, H, F, P, ..., one letter elements
        # AND - first character is NOT numeric (funky hydrogen naming rules)
        name = atom.fullname.strip()
        if len(name) < 4 and name[:1].isalpha() and len(element.strip()) < 2:
            name = " " + name

        altloc = atom.altloc
        x, y, z = atom.coord

        # Write PDB format line
        bfactor = atom.bfactor
        try:
            occupancy = f"{atom.occupancy:6.2f}"
        except (TypeError, ValueError):
            if atom.occupancy is None:
                occupancy = " " * 6  # Missing occupancy is written as blank.
            else:
                raise ValueError(f"Invalid occupancy value: {atom.occupancy!r}") from None

        args = (
            record_type,
            atom_number,
            name,
            altloc,
            resname,
            chain_id,
            resseq,
            icode,
            x,
            y,
            z,
            occupancy,
            bfactor,
            segid,
            element,
            charge,
        )
        return _ATOM_FORMAT_STRING % args

    def save(self, file, write_end: bool = True, preserve_atom_numbering: bool = False):
        """Save structure to a file."""

        if isinstance(file, str):
            fhandle = open(file, "w")
        else:
            fhandle = file  # TODO: Check the type

        get_atom_line = self._get_atom_line

        # Remarks
        for remark in self.remarks:
            fhandle.write(_REMARK_FORMAT_STRING.format(remark))

        # multiple models?
        if len(self.structure) > 1 or self.use_model_flag:
            model_flag = 1
        else:
            model_flag = 0

        for model in self.structure.get_list():
            # Necessary for ENDMDL
            # Do not write ENDMDL if no residues were written for this model
            model_residues_written = 0
            if not preserve_atom_numbering:
                atom_number = 1
            if model_flag:
                fhandle.write(f"MODEL      {model.serial_num}\n")

            for chain in model.get_list():
                chain_id = chain.id
                if len(chain_id) > 1:
                    e = f"Chain id ('{chain_id}') exceeds PDB format limit."
                    raise RuntimeError(e)

                # Necessary for TER
                # Do not write TER if no residues were written for this chain
                chain_residues_written = 0

                for residue in chain.get_unpacked_list():
                    hetfield, resseq, icode = residue.id
                    resname = residue.resname
                    segid = residue.segid
                    resid = residue.id[1]
                    if resid > 9999:
                        e = f"Residue number ('{resid}') exceeds PDB format limit."
                        raise RuntimeError(e)

                    for atom in residue.get_unpacked_list():
                        chain_residues_written = 1
                        model_residues_written = 1
                        if preserve_atom_numbering:
                            atom_number = atom.serial_number

                        try:
                            s = get_atom_line(
                                atom,
                                hetfield,
                                segid,
                                atom_number,
                                resname,
                                resseq,
                                icode,
                                chain_id,
                            )
                        except Exception as err:
                            # catch and re-raise with more information
                            raise RuntimeError(f"Error when writing atom {atom.full_id}") from err
                        else:
                            fhandle.write(s)
                            # inconsequential if preserve_atom_numbering is True
                            atom_number += 1

                if chain_residues_written:
                    fhandle.write(_TER_FORMAT_STRING % (atom_number, resname, chain_id, resseq, icode))

            if model_flag and model_residues_written:
                fhandle.write("ENDMDL\n")
        if write_end:
            fhandle.write("END   \n")

        if isinstance(file, str):
            fhandle.close()
