from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from docx import Document

BIDS_VERSION = "1.11.1"
DCM2NIIX_LATEST_URL = (
    "https://github.com/rordenlab/dcm2niix/releases/latest/download/dcm2niix_lnx.zip"
)
DENO_LATEST_URL = (
    "https://github.com/denoland/deno/releases/latest/download/"
    "deno-x86_64-unknown-linux-gnu.zip"
)


class ZipToBIDSError(RuntimeError):
    """Base error for problems that should stop the pipeline cleanly."""


class UnknownSeriesError(ZipToBIDSError):
    """Raised when a series cannot be named safely without guessing."""

    def __init__(self, rows: pd.DataFrame):
        self.rows = rows
        super().__init__("One or more imaging series could not be classified safely.")


@dataclass(frozen=True)
class ToolInstall:
    executable: Path
    version: str


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "zip-to-bids"})
    with urllib.request.urlopen(request) as response, open(destination, "wb") as out:
        shutil.copyfileobj(response, out)


def _first_version_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else "unknown"


def install_latest_dcm2niix(tool_root: Path) -> ToolInstall:
    """Install the latest stable Linux dcm2niix release for the current session."""
    tool_root = Path(tool_root)
    executable = tool_root / "dcm2niix"
    marker = tool_root / ".installed_from_latest"

    if not executable.exists() or not marker.exists():
        shutil.rmtree(tool_root, ignore_errors=True)
        tool_root.mkdir(parents=True, exist_ok=True)
        archive_path = tool_root / "dcm2niix_lnx.zip"
        _download(DCM2NIIX_LATEST_URL, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(tool_root)
        archive_path.unlink(missing_ok=True)

        candidates = [
            p for p in tool_root.rglob("dcm2niix") if p.is_file() and not p.is_symlink()
        ]
        if not candidates:
            raise ZipToBIDSError("dcm2niix was downloaded but the executable was not found.")
        source = candidates[0]
        if source != executable:
            shutil.copy2(source, executable)
        os.chmod(executable, 0o755)
        marker.write_text(DCM2NIIX_LATEST_URL + "\n", encoding="utf-8")

    proc = subprocess.run(
        [str(executable), "--version"], capture_output=True, text=True, check=False
    )
    # dcm2niix intentionally returns exit status 3 after printing version info.
    if proc.returncode not in {0, 3}:
        detail = _first_version_line(proc.stderr or proc.stdout)
        raise ZipToBIDSError(
            "dcm2niix is present but could not be executed"
            + (f": {detail}" if detail != "unknown" else ".")
        )
    version = _first_version_line((proc.stdout or "") + "\n" + (proc.stderr or ""))
    return ToolInstall(executable=executable, version=version)


def install_latest_deno(tool_root: Path) -> ToolInstall:
    """Install the latest stable Linux Deno release for BIDS validation."""
    tool_root = Path(tool_root)
    executable = tool_root / "deno"
    marker = tool_root / ".installed_from_latest"

    if not executable.exists() or not marker.exists():
        shutil.rmtree(tool_root, ignore_errors=True)
        tool_root.mkdir(parents=True, exist_ok=True)
        archive_path = tool_root / "deno.zip"
        _download(DENO_LATEST_URL, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(tool_root)
        archive_path.unlink(missing_ok=True)
        if not executable.exists():
            raise ZipToBIDSError("Deno was downloaded but the executable was not found.")
        os.chmod(executable, 0o755)
        marker.write_text(DENO_LATEST_URL + "\n", encoding="utf-8")

    proc = subprocess.run(
        [str(executable), "--version"], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise ZipToBIDSError("Deno is present but could not be executed.")
    version = _first_version_line(proc.stdout or proc.stderr)
    return ToolInstall(executable=executable, version=version)


def run_bids_validator(
    bids_root: Path,
    deno_executable: Path,
    report_path: Path,
) -> tuple[str, int]:
    """Run the current BIDS Validator through Deno and save the full console report."""
    command = [
        str(deno_executable),
        "run",
        "-ERWN",
        "jsr:@bids/validator",
        str(bids_root),
    ]
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    output = (proc.stdout or "") + (proc.stderr or "")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "$ " + " ".join(command) + "\n\n" + output,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        status = "FAIL"
    elif "[WARNING]" in output:
        status = "PASS_WITH_WARNINGS"
    else:
        status = "PASS"
    return status, proc.returncode


def find_column(
    df: pd.DataFrame,
    exact_names: Iterable[str],
    contains_names: Iterable[str] = (),
):
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for name in exact_names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    for keyword in contains_names:
        keyword = keyword.lower()
        for column in df.columns:
            if keyword in str(column).lower():
                return column
    return None


def detect_columns(df: pd.DataFrame) -> dict[str, object | None]:
    image = find_column(
        df,
        ["Image Data ID", "ImageID", "Image ID"],
        ["image data id", "imageid"],
    )
    subject = find_column(
        df,
        ["Subject", "PTID", "Subject ID"],
        ["subject", "ptid"],
    )
    visit = find_column(
        df,
        ["Visit", "VISCODE", "Timepoint"],
        ["visit", "viscode", "timepoint"],
    )
    description = find_column(
        df,
        ["Description", "SeriesDescription", "Series Description"],
        ["description", "seriesdescription"],
    )
    modality = find_column(df, ["Modality"], ["modality"])
    task = find_column(
        df,
        ["Task", "TaskName", "Task Name", "Paradigm"],
        ["task name", "paradigm"],
    )

    missing = []
    if image is None:
        missing.append("Image ID")
    if subject is None:
        missing.append("Subject ID")
    if visit is None:
        missing.append("Visit")
    if description is None and modality is None:
        missing.append("Description or Modality")
    if missing:
        raise ZipToBIDSError("Could not detect required column(s): " + ", ".join(missing))

    return {
        "image": image,
        "subject": subject,
        "visit": visit,
        "description": description,
        "modality": modality,
        "task": task,
    }


def _clean_table(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(c).strip() for c in cleaned.columns]
    cleaned = cleaned.dropna(how="all").reset_index(drop=True)
    return cleaned


def table_score(df: pd.DataFrame) -> int:
    try:
        cols = detect_columns(_clean_table(df))
    except ZipToBIDSError:
        return 0
    score = 4 + 3 + 2
    if cols["description"] or cols["modality"]:
        score += 2
    if cols["task"]:
        score += 1
    return score


def read_docx_tables(path: Path) -> list[pd.DataFrame]:
    doc = Document(str(path))
    frames: list[pd.DataFrame] = []
    for table in doc.tables:
        rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
        if len(rows) < 2:
            continue
        try:
            frame = pd.DataFrame(rows[1:], columns=rows[0])
        except ValueError:
            continue
        frames.append(_clean_table(frame))
    return frames


def load_best_table(path: Path) -> tuple[pd.DataFrame, str]:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return _clean_table(pd.read_csv(path)), "CSV"

    if suffix in {".xlsx", ".xls"}:
        book = pd.ExcelFile(path)
        candidates: list[tuple[int, str, pd.DataFrame]] = []
        for sheet in book.sheet_names:
            try:
                frame = _clean_table(pd.read_excel(path, sheet_name=sheet))
            except Exception:
                continue
            candidates.append((table_score(frame), sheet, frame))
        if not candidates:
            raise ZipToBIDSError("No readable Excel sheet was found.")
        score, sheet, frame = max(candidates, key=lambda item: item[0])
        if score == 0:
            raise ZipToBIDSError("No suitable Excel sheet was detected.")
        return frame, sheet

    if suffix == ".docx":
        tables = read_docx_tables(path)
        if not tables:
            raise ZipToBIDSError("No usable Word table was found.")
        score, index, frame = max(
            ((table_score(frame), i, frame) for i, frame in enumerate(tables)),
            key=lambda item: item[0],
        )
        if score == 0:
            raise ZipToBIDSError("No suitable Word table was detected.")
        return frame, f"WordTable{index + 1}"

    raise ZipToBIDSError(f"Unsupported table format: {suffix}")


def _text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def sanitize_bids_label(value, *, field_name: str) -> str:
    raw = _text(value)
    clean = re.sub(r"[^A-Za-z0-9]", "", raw)
    if not clean:
        raise ZipToBIDSError(f"{field_name} is empty or cannot form a BIDS label: {raw!r}")
    return clean


def bids_subject(value) -> str:
    return "sub-" + sanitize_bids_label(value, field_name="Subject ID")


def session_from_visit(value) -> str:
    raw = _text(value)
    if not raw:
        raise ZipToBIDSError("Visit is empty.")
    lowered = raw.lower()
    mapping = {
        "sc": "T0",
        "init": "T0",
        "initial": "T0",
        "bl": "T0",
        "baseline": "T0",
        "t0": "T0",
        "y1": "Y1",
        "m12": "Y1",
        "y2": "Y2",
        "m24": "Y2",
    }
    if lowered in mapping:
        return "ses-" + mapping[lowered]
    if lowered.startswith("ses-"):
        raw = raw[4:]
    return "ses-" + sanitize_bids_label(raw, field_name="Visit")


def _series_text(row: Mapping, cols: Mapping[str, object | None]) -> str:
    parts = []
    for key in ("description", "modality"):
        column = cols.get(key)
        if column:
            parts.append(_text(row.get(column)))
    return " ".join(part for part in parts if part).lower()


def _is_rest_text(text: str) -> bool:
    rest_markers = (
        "rsfmri",
        "rs-fmri",
        "resting state",
        "resting-state",
        "rest fMRI".lower(),
        "rest bold",
    )
    return any(marker in text for marker in rest_markers)


def infer_task_label(row: Mapping, cols: Mapping[str, object | None]) -> str | None:
    task_col = cols.get("task")
    if task_col:
        task_value = _text(row.get(task_col))
        if task_value:
            return sanitize_bids_label(task_value, field_name="Task").lower()
    text = _series_text(row, cols)
    if _is_rest_text(text):
        return "rest"
    return None


def classify_series(row: Mapping, cols: Mapping[str, object | None]) -> str:
    text = _series_text(row, cols)

    if any(term in text for term in ("field mapping", "fieldmap", "field map")):
        return "FMAP"
    if any(term in text for term in ("dti", "dwi", "diffusion")):
        return "DWI"
    if any(term in text for term in ("mprage", "t1w", "t1-weight", "t1 weighted")):
        return "T1"

    fmri_like = any(term in text for term in ("fmri", "bold"))
    if fmri_like and infer_task_label(row, cols):
        return "BOLD"

    return "UNKNOWN"


def validate_required_values(df: pd.DataFrame, cols: Mapping[str, object | None]) -> None:
    problems: list[str] = []
    for key, label in (("image", "Image ID"), ("subject", "Subject ID"), ("visit", "Visit")):
        column = cols[key]
        missing_rows = [
            i + 2 for i, value in enumerate(df[column].tolist()) if not _text(value)
        ]
        if missing_rows:
            preview = ", ".join(map(str, missing_rows[:10]))
            problems.append(f"{label} is missing on table row(s): {preview}")

    image_col = cols["image"]
    normalized = df[image_col].map(_text)
    duplicates = normalized[normalized.duplicated(keep=False) & normalized.ne("")]
    if not duplicates.empty:
        values = ", ".join(sorted(set(duplicates.tolist()))[:10])
        problems.append(f"Duplicate Image ID(s): {values}")

    if problems:
        raise ZipToBIDSError("\n".join(problems))


def prepare_plan(df: pd.DataFrame, cols: Mapping[str, object | None]) -> pd.DataFrame:
    plan = _clean_table(df)
    validate_required_values(plan, cols)

    for key in ("image", "subject"):
        column = cols[key]
        plan[column] = plan[column].map(_text)

    plan["session_id"] = plan[cols["visit"]].map(session_from_visit)
    plan["SeriesType"] = plan.apply(lambda row: classify_series(row, cols), axis=1)
    plan["task_label"] = plan.apply(lambda row: infer_task_label(row, cols), axis=1)

    unknown = plan[plan["SeriesType"] == "UNKNOWN"].copy()
    if not unknown.empty:
        raise UnknownSeriesError(unknown)

    plan["BIDS_Subject"] = plan[cols["subject"]].map(bids_subject)
    plan["_run_group"] = plan.apply(
        lambda row: (
            f"BOLD:{row['task_label']}" if row["SeriesType"] == "BOLD" else row["SeriesType"]
        ),
        axis=1,
    )

    group_cols = ["BIDS_Subject", "session_id", "_run_group"]
    plan["_run_index"] = plan.groupby(group_cols, sort=False).cumcount() + 1
    plan["_run_count"] = plan.groupby(group_cols, sort=False)[cols["image"]].transform("size")
    plan["run_label"] = np.where(
        plan["_run_count"] > 1,
        plan["_run_index"].map(lambda x: f"run-{int(x)}"),
        "",
    )
    return plan


def build_zip_index(
    zip_paths: Iterable[Path], image_ids: Iterable[str]
) -> dict[str, list[tuple[Path, str]]]:
    wanted = {_text(x) for x in image_ids if _text(x)}
    index: dict[str, list[tuple[Path, str]]] = {iid: [] for iid in wanted}
    for zip_path in map(Path, zip_paths):
        if not zipfile.is_zipfile(zip_path):
            raise ZipToBIDSError(f"Not a readable ZIP file: {zip_path}")
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.namelist():
                found = set(re.findall(r"I\d+", member))
                for iid in found & wanted:
                    pair = (zip_path, member)
                    if pair not in index[iid]:
                        index[iid].append(pair)
    return index


def safe_project_name(value: str) -> str:
    name = value.strip()
    if not name or name in {".", ".."}:
        raise ZipToBIDSError("Project name cannot be empty.")
    if "/" in name or "\\" in name:
        raise ZipToBIDSError("Project name cannot contain path separators.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._ -]*", name):
        raise ZipToBIDSError(
            "Project name may contain letters, numbers, spaces, dots, underscores, and hyphens."
        )
    return name


def bids_target(row: Mapping, bids_root: Path) -> tuple[Path | None, str | None]:
    sub = row["BIDS_Subject"]
    ses = row["session_id"]
    run = row["run_label"]
    stype = row["SeriesType"]
    run_part = f"_{run}" if run else ""

    if stype == "T1":
        return bids_root / sub / ses / "anat", f"{sub}_{ses}{run_part}_T1w"
    if stype == "DWI":
        return bids_root / sub / ses / "dwi", f"{sub}_{ses}{run_part}_dwi"
    if stype == "BOLD":
        task = row.get("task_label")
        if not task:
            raise ZipToBIDSError("BOLD series is missing a task label.")
        return (
            bids_root / sub / ses / "func",
            f"{sub}_{ses}_task-{task}{run_part}_bold",
        )
    return None, None


def stage_image(image_id: str, destination: Path, zip_index) -> int:
    shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True, exist_ok=True)

    grouped: dict[Path, list[str]] = defaultdict(list)
    for zip_path, member in zip_index.get(image_id, []):
        if not member.endswith("/"):
            grouped[Path(zip_path)].append(member)

    count = 0
    for zip_path, members in grouped.items():
        with zipfile.ZipFile(zip_path) as archive:
            for member in members:
                out = destination / f"{count:06d}_{Path(member).name}"
                with archive.open(member) as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                count += 1
    return count


def convert_dicom(source: Path, destination: Path, dcm2niix: Path) -> subprocess.CompletedProcess:
    shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        [
            str(dcm2niix),
            "-z",
            "y",
            "-b",
            "y",
            "-ba",
            "y",
            "-f",
            "converted",
            "-o",
            str(destination),
            str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def qc_conversion(stype: str, folder: Path) -> dict[str, object]:
    niis = sorted(folder.glob("*.nii.gz"))
    jsons = sorted(folder.glob("*.json"))
    bvals = sorted(folder.glob("*.bval"))
    bvecs = sorted(folder.glob("*.bvec"))

    result: dict[str, object] = {
        "Install": False,
        "QC": "FAIL",
        "Reason": "",
        "Shape": "",
        "VoxelSizes": "",
        "NVolumes": np.nan,
        "TR": np.nan,
        "PED": "",
        "SliceTimingCount": np.nan,
        "Shells": "",
    }

    if stype == "FMAP":
        result["Install"] = len(niis) >= 1 and len(jsons) >= 1
        result["QC"] = "PASS" if result["Install"] else "FAIL"
        result["Reason"] = "CACHE_ONLY" if result["Install"] else "FMAP_CONVERSION_FAILED"
        return result

    if len(niis) != 1:
        result["Reason"] = f"EXPECTED_1_NIFTI_GOT_{len(niis)}"
        return result
    if len(jsons) != 1:
        result["Reason"] = f"EXPECTED_1_JSON_GOT_{len(jsons)}"
        return result

    try:
        import nibabel as nib

        img = nib.load(str(niis[0]))
        shape = tuple(img.shape)
        result["Shape"] = str(shape)
        result["VoxelSizes"] = str(tuple(round(float(x), 4) for x in img.header.get_zooms()[:3]))

        with open(jsons[0], encoding="utf-8") as handle:
            meta = json.load(handle)
        result["PED"] = str(meta.get("PhaseEncodingDirection", ""))
        if isinstance(meta.get("SliceTiming"), list):
            result["SliceTimingCount"] = len(meta["SliceTiming"])

        if stype == "T1":
            if not (len(shape) == 3 or (len(shape) == 4 and shape[3] == 1)):
                result["Reason"] = "INVALID_T1_DIMENSION"
                return result

        elif stype == "BOLD":
            if len(shape) != 4 or shape[3] < 2:
                result["Reason"] = "INVALID_BOLD_DIMENSION"
                return result
            result["NVolumes"] = int(shape[3])
            tr = meta.get("RepetitionTime")
            if tr is None:
                result["Reason"] = "MISSING_TR"
                return result
            result["TR"] = float(tr)
            review = []
            if not result["PED"]:
                review.append("MISSING_PED")
            if np.isnan(result["SliceTimingCount"]):
                review.append("MISSING_SLICETIMING")
            if review:
                result["QC"] = "REVIEW"
                result["Reason"] = ";".join(review)

        elif stype == "DWI":
            if len(shape) != 4:
                result["Reason"] = "INVALID_DWI_DIMENSION"
                return result
            if len(bvals) != 1 or len(bvecs) != 1:
                result["Reason"] = "MISSING_BVAL_BVEC"
                return result

            nvol = int(shape[3])
            bv = np.loadtxt(bvals[0]).reshape(-1)
            vec = np.loadtxt(bvecs[0])
            if vec.shape == (nvol, 3):
                vec = vec.T
            if len(bv) != nvol or vec.shape != (3, nvol):
                result["Reason"] = "GRADIENT_COUNT_MISMATCH"
                return result
            if (bv > 50).sum() == 0:
                result["Reason"] = "ALL_B0_DWI"
                return result

            shells = Counter(int(round(float(x) / 100) * 100) for x in bv)
            result["NVolumes"] = nvol
            result["Shells"] = ";".join(f"{k}:{v}" for k, v in sorted(shells.items()))
            if not result["PED"]:
                result["QC"] = "REVIEW"
                result["Reason"] = "MISSING_PED"

        result["Install"] = True
        if result["QC"] != "REVIEW":
            result["QC"] = "PASS"
            result["Reason"] = "OK"
        return result

    except Exception as exc:
        result["Reason"] = f"QC_EXCEPTION:{exc}"
        return result


def install_core(row: Mapping, conversion_folder: Path, bids_root: Path) -> str:
    folder, stem = bids_target(row, bids_root)
    if folder is None or stem is None:
        raise ZipToBIDSError(f"No BIDS target is defined for {row['SeriesType']}.")
    folder.mkdir(parents=True, exist_ok=True)

    for src in conversion_folder.iterdir():
        if not src.is_file():
            continue
        suffix = ".nii.gz" if src.name.endswith(".nii.gz") else src.suffix
        if suffix not in {".nii.gz", ".json", ".bval", ".bvec"}:
            continue
        shutil.copy2(src, folder / f"{stem}{suffix}")

    if row["SeriesType"] == "BOLD":
        sidecar = folder / f"{stem}.json"
        if sidecar.exists():
            with open(sidecar, encoding="utf-8") as handle:
                meta = json.load(handle)
            meta["TaskName"] = row["task_label"]
            with open(sidecar, "w", encoding="utf-8") as handle:
                json.dump(meta, handle, indent=2)
                handle.write("\n")
    return stem


def cache_fmap(image_id: str, conversion_folder: Path, fmap_cache: Path) -> None:
    destination = fmap_cache / image_id
    shutil.rmtree(destination, ignore_errors=True)
    shutil.copytree(conversion_folder, destination)


def write_dataset_metadata(
    bids_root: Path,
    dataset_name: str,
    dcm2niix_version: str,
    installed_subjects: Iterable[str],
    *,
    zip_to_bids_version: str | None = None,
    authors: Iterable[str] = (),
    dataset_license: str = "",
) -> None:
    bids_root.mkdir(parents=True, exist_ok=True)

    generated_by = []
    if zip_to_bids_version:
        generated_by.append(
            {
                "Name": "ZIP-to-BIDS",
                "Version": zip_to_bids_version,
                "CodeURL": "https://github.com/sadeghghaderi93/zip-to-bids",
            }
        )
    generated_by.append(
        {
            "Name": "dcm2niix",
            "Version": dcm2niix_version,
            "CodeURL": "https://github.com/rordenlab/dcm2niix",
        }
    )

    description: dict[str, object] = {
        "Name": dataset_name,
        "BIDSVersion": BIDS_VERSION,
        "DatasetType": "raw",
        "GeneratedBy": generated_by,
    }
    author_list = [str(author).strip() for author in authors if str(author).strip()]
    if author_list:
        description["Authors"] = author_list
    if dataset_license.strip():
        description["License"] = dataset_license.strip()

    with open(bids_root / "dataset_description.json", "w", encoding="utf-8") as handle:
        json.dump(description, handle, indent=2)
        handle.write("\n")

    (bids_root / "README").write_text(
        f"{dataset_name}\n"
        + "=" * len(dataset_name)
        + "\n\n"
        + "This dataset was assembled with ZIP-to-BIDS from selected DICOM "
        + "acquisitions stored in ZIP archives. Conversion provenance and QC "
        + "reports are kept outside the BIDS dataset in the sibling BIDS_WORK "
        + "directory. Field maps that could not be assigned safely are not "
        + "included in this BIDS directory.\n",
        encoding="utf-8",
    )

    subjects = sorted(set(installed_subjects))
    if subjects:
        pd.DataFrame({"participant_id": subjects}).to_csv(
            bids_root / "participants.tsv", sep="\t", index=False
        )


def converter_warnings(proc: subprocess.CompletedProcess) -> list[str]:
    """Return warning lines emitted by dcm2niix, preserving their text for review."""
    warnings: list[str] = []
    for stream in (proc.stdout or "", proc.stderr or ""):
        for line in stream.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("warning:") and stripped not in warnings:
                warnings.append(stripped)
    return warnings


def warning_reason(warnings: Iterable[str]) -> str:
    """Create a compact manifest reason while keeping full warnings in their own column."""
    codes: list[str] = []
    for warning in warnings:
        for code in re.findall(r"\bIssue\d+\b", warning):
            if code not in codes:
                codes.append(code)
    return "DCM2NIIX_WARNING" + (":" + ",".join(codes) if codes else "")


def append_reason(existing: object, reason: str) -> str:
    current = str(existing or "").strip()
    if not current or current == "OK":
        return reason
    parts = [part for part in current.split(";") if part]
    if reason not in parts:
        parts.append(reason)
    return ";".join(parts)


def apply_converter_warnings(
    qc: dict[str, object], proc: subprocess.CompletedProcess
) -> list[str]:
    """Mark an otherwise installable conversion for review when dcm2niix warns."""
    warnings = converter_warnings(proc)
    if warnings and qc.get("Install"):
        qc["QC"] = "REVIEW"
        qc["Reason"] = append_reason(qc.get("Reason"), warning_reason(warnings))
    return warnings


def write_conversion_log(log_path: Path, proc: subprocess.CompletedProcess) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        (proc.stdout or "") + ("\n" if proc.stdout else "") + (proc.stderr or ""),
        encoding="utf-8",
    )
