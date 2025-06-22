# app.py – FastAPI wrapper around REINVENT4 CLI
# (supports De-Novo, Scaffold Decoration, Fragment Linking)

import logging
import subprocess
import tempfile
import csv
import toml
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uuid, shutil                                        # <-- NEW
from fastapi import FastAPI, HTTPException                 # <-- NEW
from fastapi.responses import FileResponse                 # <-- NEW



# ─────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

REINVENT_PRIOR   = WORKSPACE_ROOT / "priors" / "reinvent.prior"
LIBINVENT_PRIOR  = WORKSPACE_ROOT / "priors" / "libinvent_transformer_pubchem.prior"
LINKINVENT_PRIOR = WORKSPACE_ROOT / "priors" / "linkinvent_transformer_pubchem.prior"
# Mol2Mol (three similarity levels)
M2M_HIGH     = WORKSPACE_ROOT / "priors" / "mol2mol_similarity.prior"
M2M_HIGHER   = WORKSPACE_ROOT / "priors" / "mol2mol_medium_similarity.prior"
M2M_HIGHEST  = WORKSPACE_ROOT / "priors" / "mol2mol_high_similarity.prior"

if not REINVENT_PRIOR.is_file():
    raise FileNotFoundError(f"Reinvent prior not found - De Novo Generation will fail.")
if not LIBINVENT_PRIOR.is_file():
    logger.warning("LibInvent prior not found - Scaffold Decoration will fail.")
if not LINKINVENT_PRIOR.is_file():
    logger.warning("LinkInvent prior not found - Fragment Linking will fail.")
# ─── Verify Mol2Mol priors exist ───────────────────────────────
if not M2M_HIGH.is_file():
    logger.warning("Mol2Mol-High prior not found - Molecular Transformation (High) will fail.")
if not M2M_HIGHER.is_file():
    logger.warning("Mol2Mol-Higher prior not found - Molecular Transformation (Higher) will fail.")
if not M2M_HIGHEST.is_file():
    logger.warning("Mol2Mol-Highest prior not found - Molecular Transformation (Highest) will fail.")

# ─────────────────────────────────────────────────────────
# FastAPI setup
# ─────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────
# Request schema
# ─────────────────────────────────────────────────────────
class UserInput(BaseModel):
    samples:     int
    temperature: float
    mode:        str

    # Scaffold Decoration
    scaffold:            Optional[str] = ""
    scaffold_file_text:  Optional[str] = None

    # Fragment Linking (new field names)
    fragment_a:          Optional[str] = ""
    fragment_b:          Optional[str] = ""
    fragments_file_text: Optional[str] = None
    fragment_file_text:  Optional[str] = None  # alt name used by some UIs

    # ── legacy field names (for old front-ends) ──────────
    fragment1:           Optional[str] = None
    fragment2:           Optional[str] = None
    
    # Molecular Transformation
    starting_smiles:            str = ""
    starting_molecule:          Optional[str] = None   # ← legacy / UI alias
    transformation_file_text:   Optional[str] = None
    similarity:                 str = "Tanimoto-High"   # dropdown value


# ─────────────────────────────────────────────────────────
# Helper builders
# ─────────────────────────────────────────────────────────
def build_reinvent_cfg(samples: int, T: float, csv_path: Path) -> dict:
    return {
        "run_type": "sampling",
        "use_cuda": False,
        "parameters": {
            "model_file": str(REINVENT_PRIOR),
            "output_file": str(csv_path),
            "num_smiles": samples,
            "temperature": T,
            "unique_molecules": False, #change it to True if you want unique molecules
            "randomize_smiles": True,
        },
    }


def build_libinvent_cfg(samples: int, T: float,
                        smi_file: Path, csv_path: Path) -> dict:
    return {
        "run_type": "sampling",
        "use_cuda": False,
        "parameters": {
            "model_file": str(LIBINVENT_PRIOR),
            "smiles_file": str(smi_file),
            "output_file": str(csv_path),
            "num_smiles": samples,
            "temperature": T,
            "unique_molecules": False, #change it to True if you want unique molecules
            "randomize_smiles": True,
        },
    }


def build_linkinvent_cfg(samples: int, T: float,
                         frg_file: Path, csv_path: Path) -> dict:
    return {
        "run_type": "sampling",
        "use_cuda": False,
        "parameters": {
            "model_file": str(LINKINVENT_PRIOR),
            "smiles_file": str(frg_file),
            "output_file": str(csv_path),
            "num_smiles": samples,
            "temperature": T,
            # LinkInvent does not support unique_molecules
            # because it generates pairs of fragments
            # which may not be unique by design
            "unique_molecules": False,
            "randomize_smiles": True,
        },
    }

def build_mol2mol_cfg(samples:int, T:float, in_smi:Path, prior:Path, csv_out:Path)->dict:
    return {
        "run_type": "sampling",
        "use_cuda": False,
        "parameters": {
            "model_file": str(prior),
            "smiles_file": str(in_smi),
            "output_file": str(csv_out),
            "num_smiles": samples,
            "temperature": T,
            "unique_molecules": False,  # change it to True if you want unique molecules
            "randomize_smiles": False,
        },
    }


def write_lines(lines: List[str], path: Path) -> None:
    with open(path, "w") as fh:
        for ln in lines:
            ln = ln.strip()
            if ln:
                fh.write(ln + "\n")


def write_pairs(pairs: List[tuple[str, str]], path: Path) -> None:
    with open(path, "w") as fh:
        for a, b in pairs:
            if a and b:
                fh.write(f"{a}\t{b}\n")


# ─────────────────────────────────────────────────────────
# Endpoint
# ─────────────────────────────────────────────────────────
@app.post("/generate-molecules")
async def generate_molecules(u: UserInput):
    logger.info("mode=%s | n=%s | T=%.2f", u.mode, u.samples, u.temperature)

    if u.samples < 1:
        return {"error": "samples must be ≥ 1"}
    if u.mode not in {"De Novo Generation",
                      "Scaffold Decoration",
                      "Fragment Linking",
                      "Molecular Transformation"}:
        return {"error": f"Unsupported mode: {u.mode}"}

    with tempfile.TemporaryDirectory() as td:
        tmp      = Path(td)
        csv_path = tmp / "sampling.csv"
        cfg_path = tmp / "cfg.toml"

        # ───────── Scaffold Decoration ─────────
        if u.mode == "Scaffold Decoration":
            scaffolds = []
            if u.scaffold_file_text:
                scaffolds.extend(u.scaffold_file_text.splitlines())
            if u.scaffold:
                scaffolds.append(u.scaffold)

            scaffolds = [s.strip() for s in scaffolds if s.strip()]
            if not scaffolds:
                return {"error": "No scaffold provided."}

            smi_path = tmp / "scaffolds.smi"
            write_lines(scaffolds, smi_path)
            cfg = build_libinvent_cfg(u.samples, u.temperature, smi_path,
                                      csv_path)

        # ───────── Fragment Linking ────────────
        elif u.mode == "Fragment Linking":
            pairs: List[tuple[str, str]] = []

            # file upload (pipe |, tab or space separated)
            content = u.fragments_file_text or u.fragment_file_text
            if content:
                for ln in content.splitlines():
                    ln = ln.strip()
                    if not ln or ln.startswith("#"):
                        continue
                    if "|" in ln:
                        a, b, *_ = [p.strip() for p in ln.split("|")]
                        pairs.append((a, b))
                    else:
                        toks = [t for t in ln.replace(",", " ").split() if t]
                        if len(toks) >= 2:
                            pairs.append((toks[0], toks[1]))

            # text fields (accept both new and legacy names)
            fragA = u.fragment_a or u.fragment1 or ""
            fragB = u.fragment_b or u.fragment2 or ""
            if fragA and fragB:
                pairs.append((fragA, fragB))

            if not pairs:
                return {"error": "No fragment pairs supplied."}

            frg_path = tmp / "fragments.smi"
            write_pairs(pairs, frg_path)
            cfg = build_linkinvent_cfg(u.samples, u.temperature, frg_path,
                                       csv_path)
            
        # ───────── Molecular Transformation ────────
        elif u.mode == "Molecular Transformation":
            # gather input SMILES
            smi_lines: list[str] = []
            if u.transformation_file_text:
                smi_lines.extend([ln.strip() for ln in u.transformation_file_text.splitlines() if ln.strip()])
            # if u.starting_smiles.strip():
                # smi_lines.append(u.starting_smiles.strip())
            first = (u.starting_smiles or u.starting_molecule or "").strip()
            if first:
                smi_lines.append(first)
            if not smi_lines:
                return {"error": "No starting molecules supplied."}

            in_path = tmp / "mol2mol_input.smi"
            write_lines(smi_lines, in_path)

            # choose prior by dropdown
            if u.similarity == "Tanimoto-Highest":
                prior = M2M_HIGHEST
            elif u.similarity == "Tanimoto-Higher":
                prior = M2M_HIGHER
            else:                       
                prior = M2M_HIGH       # default = High
            cfg = build_mol2mol_cfg(u.samples, u.temperature,
                                    in_path, prior, csv_path)

        # ───────── De-Novo ─────────────────────
        else:
            cfg = build_reinvent_cfg(u.samples, u.temperature, csv_path)

        # write TOML and launch CLI
        with open(cfg_path, "w") as fh:
            toml.dump(cfg, fh)
        logger.info("TOML written → %s", cfg_path)

        try:
            subprocess.run(["reinvent", str(cfg_path)], check=True)
        except subprocess.CalledProcessError as exc:
            logger.error("CLI crashed: %s", exc)
            return {"error": "REINVENT execution failed", "detail": str(exc)}


        # -----------------------------------------------------------
        # 1️⃣  Persist the CSV so the user can download it later
        # -----------------------------------------------------------
        file_id  = f"{uuid.uuid4()}.csv"          # random handle
        dl_path  = Path("/tmp") / file_id         # short-lived store
        shutil.copy(csv_path, dl_path)

        # read CSV
        try:
            rows = list(csv.DictReader(open(csv_path)))
        except FileNotFoundError:
            return {"error": "sampling.csv not produced"}

        if not rows:
            return {"error": "No molecules generated"}

        key = "smiles" if "smiles" in rows[0] else "SMILES"
        molecules = [{"smiles": r[key],
                      "valid": r.get("state", "True") == "True"} for r in rows]
        valid_n = sum(m["valid"] for m in molecules)

    return {
        "message": f"Success via {u.mode}",
        "generated_molecules": molecules,
        "requested_samples": u.samples,
        "total_generated": len(molecules),
        "valid_molecules": valid_n,
        "csv_url": f"/download/{file_id}"      # 2️⃣  give FE the link
        # "filtered_count": u.samples - len(molecules),
        # "filtering_reason": "Duplicates or invalid SMILES were removed",
    }

# -----------------------------------------------------------
# 3️⃣  Single generic endpoint to stream any saved CSV
# -----------------------------------------------------------
@app.get("/download/{file_id}")
def download_csv(file_id: str):
    path = Path("/tmp") / f"{file_id}"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File expired.")
    return FileResponse(path,
                        media_type="text/csv",
                        filename="sampling.csv")