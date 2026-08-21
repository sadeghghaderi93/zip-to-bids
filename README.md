# ZIP-to-BIDS

ZIP-to-BIDS is a Google Colab workflow for turning selected DICOM acquisitions stored inside ZIP archives into a BIDS dataset. It is aimed at longitudinal neuroimaging downloads where a cohort table links an image ID to a subject, visit, modality, and series description.

The practical workflow is straightforward: keep the large ZIP files in Google Drive, select the ZIP archive(s) and cohort table in Colab, choose an output folder, and run either an index-only check or the full conversion. The archive is indexed first and only the requested acquisition is staged to temporary Colab storage during conversion.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sadeghghaderi93/zip-to-bids/blob/main/notebooks/ZIP_to_BIDS_Colab.ipynb)

## What it does

- Reads CSV, XLSX, XLS, or DOCX cohort tables.
- Detects common Image ID, Subject, Visit, Description, Modality, and optional Task columns.
- Indexes ZIP members before extraction, so the whole archive is not unpacked to disk.
- Downloads the current stable Linux build of [dcm2niix](https://github.com/rordenlab/dcm2niix/releases) from the official `releases/latest` endpoint when conversion starts.
- Writes T1w, DWI, and BOLD outputs with BIDS-compatible names.
- Recognizes resting-state fMRI only from explicit resting-state wording such as `rsfMRI` or `resting-state`.
- Uses `Task`, `TaskName`, or `Paradigm` for task-fMRI instead of labeling generic BOLD scans as `rest`.
- Checks NIfTI dimensions, BOLD TR, diffusion gradients, phase-encoding metadata, and slice-timing presence.
- Saves a conversion manifest after every acquisition and keeps the full dcm2niix log for each Image ID.
- Preserves dcm2niix warning lines in the manifest and marks successful conversions with warnings as `REVIEW`.
- Converts unresolved field maps to a separate review cache instead of guessing their BIDS role.
- Runs the current BIDS Validator through Deno at the end when validation is enabled and distinguishes a clean `PASS` from `PASS_WITH_WARNINGS`.

The generated dataset declares BIDS 1.11.1. Each fresh Colab runtime downloads whatever release the official dcm2niix `releases/latest` endpoint currently serves and records the version reported by the binary.

## Cohort table

A typical table looks like this:

| Image Data ID | Subject | Visit | Modality | Description | Task (optional) |
| --- | --- | --- | --- | --- | --- |
| I123456 | 001_S_0001 | bl | MRI | Accelerated Sagittal MPRAGE | |
| I123457 | 001_S_0001 | bl | fMRI | Axial MB rsfMRI (Eyes Open) | |
| I123458 | 001_S_0001 | bl | fMRI | Motor BOLD | motor |
| I123459 | 001_S_0001 | bl | DTI | Axial MB DTI | |

Common alternatives such as `PTID`, `VISCODE`, `SeriesDescription`, `TaskName`, and `Paradigm` are detected too. ZIP member paths are expected to contain image identifiers in the `I123456` style.

For task-based fMRI, ZIP-to-BIDS can name the BOLD file from the cohort table, but it does not create experimental timing files. Those runs are therefore marked for review so that the appropriate `*_events.tsv` can be added from the experiment logs. Resting-state scans are not treated as task runs requiring events.

## Field maps

Field-map naming depends on what was actually acquired: phasediff/magnitude pairs, phase1/phase2, direct field maps, or EPI-based field maps. ZIP-to-BIDS does not try to infer that from a generic `Field Mapping` description. By default it converts those acquisitions into:

```text
BIDS_WORK/FIELDMAP_CONVERTED_CACHE/
```

This leaves the core BIDS dataset incomplete rather than silently giving field maps the wrong semantics.

## Running in Google Colab

Open `notebooks/ZIP_to_BIDS_Colab.ipynb`, run the setup cell, and then launch the interface. The setup cell installs the repository package and also adds the local `src` directory to Python's path as a fallback, which avoids import failures in Colab editable-install edge cases.

While the GitHub repository is private, an unauthenticated Colab session cannot clone it directly. In that case the setup cell asks for a ZIP download of the repository and continues from the uploaded archive. Once the repository is public, the same cell clones GitHub automatically. No GitHub token needs to be pasted into the notebook.

In the interface:

1. Add one or more ZIP archives.
2. Select the cohort table.
3. Choose the output parent folder and a project name. Optional dataset authors and license can also be entered; they are added to `dataset_description.json` only when supplied.
4. Start with `INDEX_ONLY` for a new dataset.
5. If every expected Image ID is found, switch to `CONVERT`.

During conversion, every acquisition prints its Image ID, series type, staged DICOM count, dcm2niix step, warning count when applicable, and final status. The start button is disabled while a run is active to avoid accidental duplicate callbacks.

## Output

```text
PROJECT_NAME/
├── BIDS/
│   ├── dataset_description.json
│   ├── participants.tsv
│   └── sub-.../
└── BIDS_WORK/
    ├── BIDS_BUILD_MANIFEST.csv
    ├── BIDS_PROBLEMS.csv
    ├── BIDS_REVIEW.csv
    ├── BIDS_VALIDATION.txt
    ├── SELECTED_PLAN.csv
    ├── ZIP_TO_BIDS_SETTINGS.json
    ├── FIELDMAP_CONVERTED_CACHE/
    └── logs/
```

`BIDS_WORK` is deliberately outside the BIDS dataset. It contains provenance, QC information, conversion logs, and material that needs manual review. A small root `README` is created automatically for the BIDS dataset; dataset authors and license remain optional because the converter should not invent study-specific metadata.

The manifest is updated after each acquisition, so a partial run still leaves a record of what finished. Automatic resume is not implemented; rerunning an existing output requires an explicit overwrite or a new project name.

## QC and review states

`PASS` means the structural checks implemented here passed and dcm2niix did not emit a warning. It is not a substitute for visual image QC or protocol-specific review.

`REVIEW` is used when the conversion is usable enough to install but needs attention, for example:

- dcm2niix emitted a warning such as an `Issue...` message;
- BOLD is missing phase-encoding or slice-timing metadata;
- a task-fMRI run was named but event timing files were not generated.

Hard failures such as inconsistent DWI gradient counts are not installed into the BIDS dataset.

## Data privacy

Do not commit source DICOM files, participant tables, generated BIDS datasets, or sensitive conversion logs to this repository. The `.gitignore` blocks common imaging files and ZIP archives, but it cannot protect data that are deliberately added.

The dcm2niix command uses `-ba y` to anonymize BIDS sidecars. The original DICOM archive is still source research data and should be handled according to the rules that apply to the study.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check src tests
```

GitHub Actions runs linting and tests on Python 3.10, 3.11, and 3.12.

## Scope

ZIP-to-BIDS is a conversion helper, not a substitute for protocol knowledge. It intentionally stops or asks for review when the table and DICOM metadata are not enough to make a safe naming decision.

## License

MIT.
