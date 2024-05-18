import logging
from pathlib import Path
from typing import List

from deepfold.data.search.templates import TemplateHit

logger = logging.getLogger(__name__)


def parse_crf(crf_string: str, query_id: str, alignment_dir: Path) -> List[TemplateHit]:
    lst_crf = crf_string.strip().splitlines()

    crf_hits = []
    for i, chain_id in enumerate(lst_crf, start=1):
        # Handle comments:
        if chain_id.startswith("#"):
            continue
        chain_id, _, _ = chain_id.partition("#")
        chain_id = chain_id.strip()

        # Parse template hits:
        with open(alignment_dir / f"{query_id}-{chain_id}.pir", "r") as fp:
            lines = fp.read()
        hit = parse_pir(lines, index=i)
        crf_hits.append(hit)

    return crf_hits


def parse_pir(pir_string: str, index: int = 0) -> TemplateHit:
    query_sequence = []
    hit_sequence = []
    query_name = None
    hit_name = None
    reading_query = False
    reading_hit = False
    sum_probs = 0.0

    lines = pir_string.strip().splitlines()

    for line in lines:
        if line.startswith(">P1;"):
            if not query_name:
                query_name = line.split(";")[1].strip()
                reading_query = True
                reading_hit = False
            else:
                hit_name = line.split(";")[1].strip()
                reading_query = False
                reading_hit = True

        elif line.startswith("structureX"):
            continue
        elif line.startswith("C;"):
            if "probs_sum" in line:
                sum_probs = float(line.split("=")[1].strip())
        else:
            if reading_query:
                query_sequence.extend(line.strip().strip("*"))
            elif reading_hit:
                hit_sequence.extend(line.strip().strip("*"))

    # Create the `TemplateHit` object for the single hit:
    hit_sequence_str = "".join(hit_sequence)  # .replace("-", "")
    query_sequence_str = "".join(query_sequence)  # .replace("-", "")

    indices_query, indices_hit = [], []
    qi, hi = 0, 0
    for q, h in zip(query_sequence_str, hit_sequence_str):
        if h == "-":
            indices_hit.append(-1)
        else:
            indices_hit.append(hi)
            hi += 1
        if q == "-":
            indices_query.append(-1)
        else:
            indices_query.append(qi)
            qi += 1

    hit = TemplateHit(
        index=index,
        name=hit_name,
        aligned_cols=len(hit_sequence_str),
        sum_probs=sum_probs,
        query=query_sequence_str,
        hit_sequence=hit_sequence_str,
        indices_query=indices_query,
        indices_hit=indices_hit,
    )

    return hit
