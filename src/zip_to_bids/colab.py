from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import clear_output, display
from ipyfilechooser import FileChooser

from . import __version__
from .core import (
    BIDS_VERSION,
    UnknownSeriesError,
    append_reason,
    apply_converter_warnings,
    ZipToBIDSError,
    build_zip_index,
    cache_fmap,
    convert_dicom,
    detect_columns,
    install_core,
    install_latest_dcm2niix,
    install_latest_deno,
    load_best_table,
    prepare_plan,
    qc_conversion,
    run_bids_validator,
    safe_project_name,
    stage_image,
    write_conversion_log,
    write_dataset_metadata,
)

TMP_ROOT = Path("/content/ZIP_TO_BIDS_TMP")
TOOL_ROOT = Path("/tmp/zip_to_bids_tools")


def _progress(message: str = "") -> None:
    print(message, flush=True)


def _status(label: str, value) -> str:
    return f"<b>{html.escape(label)}:</b><br>{html.escape(str(value))}"


def _show_unknown(unknown: pd.DataFrame, cols) -> None:
    show_cols = [cols["image"], cols["subject"]]
    if cols.get("description"):
        show_cols.append(cols["description"])
    if cols.get("modality"):
        show_cols.append(cols["modality"])
    if cols.get("task"):
        show_cols.append(cols["task"])
    print("\nThese series could not be named safely:")
    print(unknown[show_cols].drop_duplicates().to_string(index=False))
    print(
        "\nFor task-fMRI, add a Task/TaskName/Paradigm column. "
        "Resting-state scans are detected from their description."
    )


def launch() -> None:
    """Open the Colab interface and run the ZIP-to-BIDS pipeline from a button."""
    try:
        from google.colab import drive
    except ImportError as exc:
        raise RuntimeError("This interface is intended for Google Colab.") from exc

    drive.mount("/content/drive", force_remount=False)
    mydrive = Path("/content/drive/MyDrive")
    state = {"zip_paths": [], "table_path": None, "output_parent": None}

    zip_picker = FileChooser(str(mydrive))
    zip_picker.title = "Select source ZIP"
    zip_picker.filter_pattern = "*.zip"
    add_zip = widgets.Button(description="Add ZIP", button_style="info", icon="plus")
    clear_zips = widgets.Button(description="Clear ZIPs", icon="trash")
    zip_status = widgets.HTML("<b>Selected ZIPs:</b> None")

    table_picker = FileChooser(str(mydrive))
    table_picker.title = "Select cohort table"
    table_picker.filter_pattern = ["*.csv", "*.xlsx", "*.xls", "*.docx"]
    use_table = widgets.Button(description="Use Table", button_style="info", icon="table")
    table_status = widgets.HTML("<b>Selected table:</b> None")

    output_picker = FileChooser(str(mydrive))
    output_picker.title = "Browse to output parent folder"
    output_picker.show_only_dirs = True
    use_output = widgets.Button(
        description="Use Current Folder", button_style="info", icon="folder"
    )
    output_status = widgets.HTML("<b>Output parent:</b> None")

    project_name = widgets.Text(
        value="ZIP_TO_BIDS_OUTPUT",
        description="Project:",
        layout=widgets.Layout(width="95%"),
    )
    dataset_authors = widgets.Text(
        value="",
        description="Authors:",
        placeholder="Optional; separate names with semicolons",
        layout=widgets.Layout(width="95%"),
    )
    dataset_license = widgets.Text(
        value="",
        description="License:",
        placeholder="Optional; e.g. CC0",
        layout=widgets.Layout(width="95%"),
    )
    run_mode = widgets.Dropdown(
        options=["CONVERT", "INDEX_ONLY"], value="CONVERT", description="Mode:"
    )
    fmap_policy = widgets.Dropdown(
        options=["CACHE_ONLY", "SKIP"], value="CACHE_ONLY", description="Field maps:"
    )
    missing_policy = widgets.Dropdown(
        options=["STOP", "CONTINUE"], value="STOP", description="Missing IDs:"
    )
    validate_bids = widgets.Checkbox(value=True, description="Run BIDS Validator")
    overwrite = widgets.Checkbox(value=False, description="Overwrite existing output")
    start_button = widgets.Button(
        description="Confirm & Start",
        button_style="success",
        icon="play",
        layout=widgets.Layout(width="240px", height="45px"),
    )
    pipeline_output = widgets.Output()

    def refresh_zip_status() -> None:
        if state["zip_paths"]:
            lines = "<br>".join(f"✓ {html.escape(str(p))}" for p in state["zip_paths"])
        else:
            lines = "None"
        zip_status.value = "<b>Selected ZIPs:</b><br>" + lines

    def add_zip_clicked(_):
        if not zip_picker.selected:
            return
        path = Path(zip_picker.selected)
        if path.suffix.lower() == ".zip" and path not in state["zip_paths"]:
            state["zip_paths"].append(path)
        refresh_zip_status()

    def clear_zips_clicked(_):
        state["zip_paths"] = []
        refresh_zip_status()

    def use_table_clicked(_):
        if not table_picker.selected:
            return
        path = Path(table_picker.selected)
        if path.suffix.lower() in {".csv", ".xlsx", ".xls", ".docx"}:
            state["table_path"] = path
            table_status.value = _status("Selected table", path)

    def use_output_clicked(_):
        folder = output_picker.selected or output_picker.selected_path
        if not folder:
            return
        path = Path(folder)
        if path.is_dir():
            state["output_parent"] = path
            output_status.value = _status("Output parent", path)

    add_zip.on_click(add_zip_clicked)
    clear_zips.on_click(clear_zips_clicked)
    use_table.on_click(use_table_clicked)
    use_output.on_click(use_output_clicked)

    def run_pipeline(_):
        if start_button.disabled:
            return
        start_button.disabled = True
        start_button.description = "Running..."
        with pipeline_output:
            clear_output()
            try:
                if not state["zip_paths"]:
                    raise ZipToBIDSError("Add at least one ZIP file.")
                if state["table_path"] is None:
                    raise ZipToBIDSError("Select the cohort table.")
                if state["output_parent"] is None:
                    raise ZipToBIDSError("Select the output folder.")

                name = safe_project_name(project_name.value)
                output_root = state["output_parent"] / name
                bids_root = output_root / "BIDS"
                work_root = output_root / "BIDS_WORK"
                fmap_cache = work_root / "FIELDMAP_CONVERTED_CACHE"
                manifest_path = work_root / "BIDS_BUILD_MANIFEST.csv"
                log_root = work_root / "logs"

                print("1. Reading cohort table")
                table, source_sheet = load_best_table(state["table_path"])
                cols = detect_columns(table)
                try:
                    plan = prepare_plan(table, cols)
                except UnknownSeriesError as exc:
                    _show_unknown(exc.rows, cols)
                    return

                print(f"   Source: {state['table_path']}")
                print(f"   Sheet/table: {source_sheet}")
                print(f"   Rows: {len(plan)}")
                print("   Series:")
                for kind, count in plan["SeriesType"].value_counts().items():
                    print(f"     {kind}: {count}")

                print("\n2. Indexing ZIP archives")
                zip_index = build_zip_index(state["zip_paths"], plan[cols["image"]])
                missing = [iid for iid, members in zip_index.items() if not members]
                print(f"   Image IDs: {len(zip_index)}")
                print(f"   Found: {len(zip_index) - len(missing)}")
                print(f"   Missing: {len(missing)}")
                if missing:
                    print("   Missing IDs: " + ", ".join(sorted(missing)))
                    if missing_policy.value == "STOP":
                        print("\nStopped before conversion because source data are missing.")
                        return

                if run_mode.value == "INDEX_ONLY":
                    work_root.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(
                        {
                            "ImageID": list(zip_index),
                            "FoundInZIP": [bool(zip_index[iid]) for iid in zip_index],
                        }
                    ).to_csv(work_root / "ZIP_INDEX_QC.csv", index=False)
                    print("\nINDEX_ONLY complete.")
                    return

                if bids_root.exists() and any(bids_root.iterdir()) and not overwrite.value:
                    raise ZipToBIDSError(
                        "Existing BIDS output found. Enable overwrite or use another project name."
                    )
                if overwrite.value:
                    shutil.rmtree(bids_root, ignore_errors=True)
                    shutil.rmtree(work_root, ignore_errors=True)

                bids_root.mkdir(parents=True, exist_ok=True)
                work_root.mkdir(parents=True, exist_ok=True)
                fmap_cache.mkdir(parents=True, exist_ok=True)
                TMP_ROOT.mkdir(parents=True, exist_ok=True)

                print("\n3. Preparing tools")
                dcm2niix = install_latest_dcm2niix(TOOL_ROOT / "dcm2niix")
                print(f"   dcm2niix: {dcm2niix.version}")

                settings = {
                    "ZIP_PATHS": [str(path) for path in state["zip_paths"]],
                    "TABLE_PATH": str(state["table_path"]),
                    "SOURCE_SHEET": source_sheet,
                    "OUTPUT_ROOT": str(output_root),
                    "DETECTED_COLUMNS": cols,
                    "FMAP_POLICY": fmap_policy.value,
                    "ZIP_TO_BIDS_VERSION": __version__,
                    "BIDS_VERSION": BIDS_VERSION,
                    "DCM2NIIX_VERSION": dcm2niix.version,
                    "DCM2NIIX_SOURCE": "official GitHub releases/latest Linux build",
                    "DATASET_AUTHORS": [
                        name.strip()
                        for name in dataset_authors.value.split(";")
                        if name.strip()
                    ],
                    "DATASET_LICENSE": dataset_license.value.strip(),
                }
                with open(work_root / "ZIP_TO_BIDS_SETTINGS.json", "w", encoding="utf-8") as handle:
                    json.dump(settings, handle, indent=2, default=str)
                    handle.write("\n")
                plan.to_csv(work_root / "SELECTED_PLAN.csv", index=False)

                _progress("\n4. Converting acquisitions")
                records = []
                total = len(plan)
                for i, (_, row) in enumerate(plan.iterrows(), start=1):
                    image_id = str(row[cols["image"]])
                    stype = row["SeriesType"]
                    task = row.get("task_label") or ""
                    label = f"{stype}" + (f" task-{task}" if stype == "BOLD" and task else "")
                    _progress(f"\n   [{i}/{total}] {image_id} — {label}")

                    local_root = TMP_ROOT / image_id
                    dicom_dir = local_root / "dicom"
                    conv_dir = local_root / "converted"
                    rec = {
                        "ImageID": image_id,
                        "Subject": row[cols["subject"]],
                        "Session": row["session_id"],
                        "SeriesType": stype,
                        "Task": task,
                        "DICOM_Count": 0,
                        "Status": "",
                        "QC": "",
                        "Reason": "",
                        "Shape": "",
                        "VoxelSizes": "",
                        "NVolumes": np.nan,
                        "TR": np.nan,
                        "PED": "",
                        "SliceTimingCount": np.nan,
                        "Shells": "",
                        "DCM2NIIXWarnings": "",
                        "LogFile": f"logs/{image_id}.log",
                        "BIDSStem": "",
                    }

                    if not zip_index.get(image_id):
                        rec["Status"] = "MISSING_SOURCE"
                        rec["Reason"] = "IMAGE_ID_NOT_FOUND"
                        records.append(rec)
                        pd.DataFrame(records).to_csv(manifest_path, index=False)
                        _progress("      source missing — recorded and skipped")
                        continue

                    if stype == "FMAP" and fmap_policy.value == "SKIP":
                        rec["Status"] = "SKIPPED_BY_USER"
                        rec["Reason"] = "FMAP_SKIPPED_BY_USER"
                        records.append(rec)
                        pd.DataFrame(records).to_csv(manifest_path, index=False)
                        _progress("      field map skipped by user")
                        continue

                    try:
                        rec["DICOM_Count"] = stage_image(image_id, dicom_dir, zip_index)
                        _progress(f"      staged {rec['DICOM_Count']} DICOM file(s)")
                        if rec["DICOM_Count"] == 0:
                            rec["Status"] = "FAILED"
                            rec["Reason"] = "NO_DICOM_FILES_STAGED"
                        else:
                            _progress("      running dcm2niix...")
                            conversion = convert_dicom(dicom_dir, conv_dir, dcm2niix.executable)
                            write_conversion_log(log_root / f"{image_id}.log", conversion)
                            if conversion.returncode != 0:
                                rec["Status"] = "CONVERSION_FAILED"
                                rec["Reason"] = "DCM2NIIX_FAILED"
                            else:
                                qc = qc_conversion(stype, conv_dir)
                                warnings = apply_converter_warnings(qc, conversion)
                                rec["DCM2NIIXWarnings"] = " | ".join(warnings)

                                if (
                                    stype == "BOLD"
                                    and task
                                    and task != "rest"
                                    and qc["Install"]
                                ):
                                    qc["QC"] = "REVIEW"
                                    qc["Reason"] = append_reason(
                                        qc.get("Reason"), "TASK_EVENTS_NOT_GENERATED"
                                    )

                                for key in (
                                    "QC",
                                    "Reason",
                                    "Shape",
                                    "VoxelSizes",
                                    "NVolumes",
                                    "TR",
                                    "PED",
                                    "SliceTimingCount",
                                    "Shells",
                                ):
                                    rec[key] = qc[key]

                                if stype == "FMAP":
                                    if qc["Install"]:
                                        cache_fmap(image_id, conv_dir, fmap_cache)
                                        rec["Status"] = (
                                            "CACHE_REVIEW"
                                            if qc["QC"] == "REVIEW"
                                            else "CACHE_OK"
                                        )
                                    else:
                                        rec["Status"] = "FAILED_QC"
                                elif qc["Install"]:
                                    rec["BIDSStem"] = install_core(row, conv_dir, bids_root)
                                    rec["Status"] = (
                                        "BIDS_OK" if qc["QC"] == "PASS" else "BIDS_REVIEW"
                                    )
                                else:
                                    rec["Status"] = "FAILED_QC"

                                if warnings:
                                    _progress(
                                        "      dcm2niix emitted "
                                        f"{len(warnings)} warning(s); marked for review"
                                    )
                    except Exception as exc:
                        rec["Status"] = "FAILED"
                        rec["Reason"] = f"EXCEPTION:{exc}"
                    finally:
                        shutil.rmtree(local_root, ignore_errors=True)

                    records.append(rec)
                    pd.DataFrame(records).to_csv(manifest_path, index=False)
                    qc_label = rec["QC"] or "-"
                    _progress(f"      {rec['Status']} | QC: {qc_label}")

                report = pd.DataFrame(records)
                installed_subjects = report.loc[
                    report["Status"].isin(["BIDS_OK", "BIDS_REVIEW"]), "Subject"
                ]
                subject_map = dict(
                    zip(plan[cols["subject"]].astype(str), plan["BIDS_Subject"].astype(str))
                )
                installed_bids_subjects = [
                    subject_map[str(subject)]
                    for subject in installed_subjects
                    if str(subject) in subject_map
                ]
                write_dataset_metadata(
                    bids_root,
                    name,
                    dcm2niix.version,
                    installed_bids_subjects,
                    zip_to_bids_version=__version__,
                    authors=[
                        author.strip()
                        for author in dataset_authors.value.split(";")
                        if author.strip()
                    ],
                    dataset_license=dataset_license.value,
                )

                report[report["QC"] == "REVIEW"].to_csv(
                    work_root / "BIDS_REVIEW.csv", index=False
                )
                report[
                    report["Status"].isin(
                        ["FAILED", "FAILED_QC", "CONVERSION_FAILED", "MISSING_SOURCE"]
                    )
                ].to_csv(work_root / "BIDS_PROBLEMS.csv", index=False)

                print("\n5. BIDS validation")
                validation_status = "NOT_RUN"
                if validate_bids.value:
                    try:
                        deno = install_latest_deno(TOOL_ROOT / "deno")
                        print(f"   Deno: {deno.version}")
                        validation_status, _ = run_bids_validator(
                            bids_root,
                            deno.executable,
                            work_root / "BIDS_VALIDATION.txt",
                        )
                    except Exception as exc:
                        validation_status = "NOT_RUN"
                        (work_root / "BIDS_VALIDATION.txt").write_text(
                            f"Validator could not be run: {exc}\n", encoding="utf-8"
                        )
                print(f"   Validator: {validation_status}")

                _progress("\nFinal summary")
                print(f"   Rows: {len(plan)}")
                print(f"   Subjects: {plan[cols['subject']].nunique()}")
                print(
                    "   Subject-sessions: "
                    + str(plan[[cols["subject"], "session_id"]].drop_duplicates().shape[0])
                )
                print("   Status:")
                print(report["Status"].value_counts(dropna=False).to_string())
                print(f"\nBIDS: {bids_root}")
                print(f"Work/QC: {work_root}")
                print(f"Manifest: {manifest_path}")
                print("\nZIP-to-BIDS finished.")

            except ZipToBIDSError as exc:
                _progress(f"\nStopped: {exc}")
            except Exception as exc:
                _progress(f"\nPipeline error: {type(exc).__name__}: {exc}")
            finally:
                start_button.disabled = False
                start_button.description = "Confirm & Start"

    start_button.on_click(run_pipeline)

    display(widgets.HTML("<h2>1 — Select ZIP file(s)</h2>"))
    display(zip_picker, widgets.HBox([add_zip, clear_zips]), zip_status)
    display(widgets.HTML("<hr><h2>2 — Select cohort table</h2><p>CSV / XLSX / XLS / DOCX</p>"))
    display(table_picker, use_table, table_status)
    display(widgets.HTML("<hr><h2>3 — Select BIDS output location</h2>"))
    display(output_picker, use_output, output_status, project_name)
    display(
        widgets.HTML(
            "<p><b>Optional dataset metadata</b> — authors and license are written "
            "to dataset_description.json when provided.</p>"
        ),
        dataset_authors,
        dataset_license,
    )
    display(widgets.HTML("<hr><h2>4 — Processing options</h2>"))
    display(run_mode, fmap_policy, missing_policy, validate_bids, overwrite)
    display(widgets.HTML("<hr><h2>5 — Start</h2>"))
    display(start_button, pipeline_output)
