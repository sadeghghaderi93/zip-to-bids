import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from zip_to_bids.core import (
    UnknownSeriesError,
    ZipToBIDSError,
    bids_subject,
    bids_target,
    build_zip_index,
    classify_series,
    apply_converter_warnings,
    converter_warnings,
    detect_columns,
    infer_task_label,
    install_latest_dcm2niix,
    prepare_plan,
    qc_conversion,
    run_bids_validator,
    safe_project_name,
    session_from_visit,
    write_dataset_metadata,
)


def sample_df():
    return pd.DataFrame(
        [
            {
                "Image Data ID": "I100",
                "Subject": "999_S_0001",
                "Visit": "sc",
                "Modality": "MRI",
                "Description": "Accelerated Sagittal MPRAGE",
            },
            {
                "Image Data ID": "I101",
                "Subject": "999_S_0001",
                "Visit": "sc",
                "Modality": "fMRI",
                "Description": "Axial MB rsfMRI (Eyes Open)",
            },
            {
                "Image Data ID": "I102",
                "Subject": "999_S_0001",
                "Visit": "sc",
                "Modality": "DTI",
                "Description": "Axial MB DTI",
            },
            {
                "Image Data ID": "I103",
                "Subject": "999_S_0001",
                "Visit": "sc",
                "Modality": "MRI",
                "Description": "Field Mapping",
            },
        ]
    )


def test_detect_columns_and_prepare_plan():
    df = sample_df()
    cols = detect_columns(df)
    plan = prepare_plan(df, cols)
    assert plan["SeriesType"].tolist() == ["T1", "BOLD", "DWI", "FMAP"]
    assert plan.loc[1, "task_label"] == "rest"
    assert plan.loc[0, "BIDS_Subject"] == "sub-999S0001"
    assert plan.loc[0, "session_id"] == "ses-T0"


def test_rest_is_not_inferred_from_generic_bold():
    df = pd.DataFrame(
        [
            {
                "Image Data ID": "I200",
                "Subject": "001",
                "Visit": "bl",
                "Modality": "fMRI",
                "Description": "Motor BOLD",
            }
        ]
    )
    cols = detect_columns(df)
    assert classify_series(df.iloc[0], cols) == "UNKNOWN"
    assert infer_task_label(df.iloc[0], cols) is None
    with pytest.raises(UnknownSeriesError):
        prepare_plan(df, cols)


def test_explicit_task_column_allows_task_fmri():
    df = pd.DataFrame(
        [
            {
                "Image Data ID": "I200",
                "Subject": "001",
                "Visit": "bl",
                "Modality": "fMRI",
                "Description": "Motor BOLD",
                "Task": "Finger Tapping",
            }
        ]
    )
    cols = detect_columns(df)
    plan = prepare_plan(df, cols)
    assert plan.loc[0, "SeriesType"] == "BOLD"
    assert plan.loc[0, "task_label"] == "fingertapping"


def test_multiple_bold_tasks_do_not_share_run_numbers():
    df = pd.DataFrame(
        [
            {
                "Image Data ID": "I1",
                "Subject": "001",
                "Visit": "bl",
                "Modality": "fMRI",
                "Description": "Task BOLD",
                "Task": "motor",
            },
            {
                "Image Data ID": "I2",
                "Subject": "001",
                "Visit": "bl",
                "Modality": "fMRI",
                "Description": "Task BOLD",
                "Task": "memory",
            },
        ]
    )
    cols = detect_columns(df)
    plan = prepare_plan(df, cols)
    assert plan["run_label"].tolist() == ["", ""]


def test_duplicate_same_task_gets_run_labels():
    df = pd.DataFrame(
        [
            {
                "Image Data ID": "I1",
                "Subject": "001",
                "Visit": "bl",
                "Modality": "fMRI",
                "Description": "Task BOLD",
                "Task": "motor",
            },
            {
                "Image Data ID": "I2",
                "Subject": "001",
                "Visit": "bl",
                "Modality": "fMRI",
                "Description": "Task BOLD",
                "Task": "motor",
            },
        ]
    )
    plan = prepare_plan(df, detect_columns(df))
    assert plan["run_label"].tolist() == ["run-1", "run-2"]


@pytest.mark.parametrize(
    ("visit", "expected"),
    [
        ("sc", "ses-T0"),
        ("init", "ses-T0"),
        ("baseline", "ses-T0"),
        ("y1", "ses-Y1"),
        ("m12", "ses-Y1"),
        ("y2", "ses-Y2"),
        ("m24", "ses-Y2"),
        ("ses-follow_up", "ses-followup"),
    ],
)
def test_session_mapping(visit, expected):
    assert session_from_visit(visit) == expected


def test_missing_subject_is_rejected():
    df = sample_df()
    df.loc[0, "Subject"] = np.nan
    with pytest.raises(ZipToBIDSError, match="Subject ID is missing"):
        prepare_plan(df, detect_columns(df))


def test_duplicate_image_id_is_rejected():
    df = sample_df()
    df.loc[1, "Image Data ID"] = "I100"
    with pytest.raises(ZipToBIDSError, match="Duplicate Image ID"):
        prepare_plan(df, detect_columns(df))


def test_bids_subject_sanitizes_adni_id():
    assert bids_subject("999_S_0001") == "sub-999S0001"


def test_project_name_rejects_path_traversal():
    with pytest.raises(ZipToBIDSError):
        safe_project_name("../outside")


def test_build_zip_index(tmp_path: Path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("ADNI/001/I100/file1.dcm", b"a")
        zf.writestr("ADNI/001/I101/file2.dcm", b"b")
        zf.writestr("ADNI/001/I999/unselected.dcm", b"c")
    index = build_zip_index([archive], ["I100", "I101", "I102"])
    assert len(index["I100"]) == 1
    assert len(index["I101"]) == 1
    assert index["I102"] == []



def test_dcm2niix_version_exit_code_three_is_success(tmp_path: Path, monkeypatch):
    tool_root = tmp_path / "dcm2niix"
    tool_root.mkdir()
    executable = tool_root / "dcm2niix"
    executable.write_bytes(b"fake")
    (tool_root / ".installed_from_latest").write_text("latest\n")

    class Result:
        returncode = 3
        stdout = "Chris Rorden's dcm2niiX version v1.0.20260724 (64-bit Linux)\n24-July-2026\n"
        stderr = ""

    def fake_run(command, **kwargs):
        assert command == [str(executable), "--version"]
        return Result()

    monkeypatch.setattr("zip_to_bids.core.subprocess.run", fake_run)
    install = install_latest_dcm2niix(tool_root)
    assert install.executable == executable
    assert "v1.0.20260724" in install.version

def test_bids_target_uses_dynamic_task(tmp_path: Path):
    row = {
        "BIDS_Subject": "sub-001",
        "session_id": "ses-T0",
        "run_label": "",
        "SeriesType": "BOLD",
        "task_label": "motor",
    }
    folder, stem = bids_target(row, tmp_path)
    assert folder == tmp_path / "sub-001" / "ses-T0" / "func"
    assert stem == "sub-001_ses-T0_task-motor_bold"


def _save_nifti(path: Path, shape):
    nib = pytest.importorskip("nibabel")
    image = nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine=np.eye(4))
    nib.save(image, path)


def test_qc_bold_passes_with_required_metadata(tmp_path: Path):
    _save_nifti(tmp_path / "converted.nii.gz", (4, 4, 4, 10))
    (tmp_path / "converted.json").write_text(
        json.dumps(
            {
                "RepetitionTime": 2.0,
                "PhaseEncodingDirection": "j-",
                "SliceTiming": [0.0, 0.5, 1.0, 1.5],
            }
        )
    )
    qc = qc_conversion("BOLD", tmp_path)
    assert qc["Install"] is True
    assert qc["QC"] == "PASS"
    assert qc["NVolumes"] == 10


def test_qc_dwi_checks_gradient_count(tmp_path: Path):
    _save_nifti(tmp_path / "converted.nii.gz", (4, 4, 4, 3))
    (tmp_path / "converted.json").write_text(
        json.dumps({"PhaseEncodingDirection": "j"})
    )
    (tmp_path / "converted.bval").write_text("0 1000 1000\n")
    (tmp_path / "converted.bvec").write_text("1 0 0\n0 1 0\n0 0 1\n")
    qc = qc_conversion("DWI", tmp_path)
    assert qc["Install"] is True
    assert qc["QC"] == "PASS"
    assert qc["NVolumes"] == 3


def test_converter_warnings_extracts_warning_lines():
    proc = subprocess.CompletedProcess(
        ["dcm2niix"],
        0,
        stdout="Found 10 DICOM file(s)\nWarning: Issue870: check slice timing\nConvert done\n",
        stderr="Warning: secondary warning\n",
    )
    assert converter_warnings(proc) == [
        "Warning: Issue870: check slice timing",
        "Warning: secondary warning",
    ]


def test_dcm2niix_version_exit_code_zero_is_success(tmp_path: Path, monkeypatch):
    tool_root = tmp_path / "dcm2niix"
    tool_root.mkdir()
    executable = tool_root / "dcm2niix"
    executable.write_bytes(b"fake")
    (tool_root / ".installed_from_latest").write_text("latest\n")

    class Result:
        returncode = 0
        stdout = "Chris Rorden's dcm2niiX version v1.0.20260724 (64-bit Linux)\n"
        stderr = ""

    monkeypatch.setattr("zip_to_bids.core.subprocess.run", lambda *a, **k: Result())
    install = install_latest_dcm2niix(tool_root)
    assert "v1.0.20260724" in install.version


def test_converter_warning_marks_installable_qc_for_review():
    proc = subprocess.CompletedProcess(
        ["dcm2niix"],
        0,
        stdout="Warning: Issue870: check slice timing\n",
        stderr="",
    )
    qc = {"Install": True, "QC": "PASS", "Reason": "OK"}
    warnings = apply_converter_warnings(qc, proc)
    assert warnings == ["Warning: Issue870: check slice timing"]
    assert qc["QC"] == "REVIEW"
    assert qc["Reason"] == "DCM2NIIX_WARNING:Issue870"


def test_validator_reports_pass_with_warnings(tmp_path: Path, monkeypatch):
    class Result:
        returncode = 0
        stdout = "[WARNING] README_FILE_MISSING something\n"
        stderr = ""

    monkeypatch.setattr("zip_to_bids.core.subprocess.run", lambda *a, **k: Result())
    status, code = run_bids_validator(
        tmp_path / "BIDS", tmp_path / "deno", tmp_path / "report.txt"
    )
    assert code == 0
    assert status == "PASS_WITH_WARNINGS"


def test_dataset_metadata_writes_readme_and_optional_fields(tmp_path: Path):
    write_dataset_metadata(
        tmp_path,
        "Example Dataset",
        "v1.0.20260724",
        ["sub-001"],
        zip_to_bids_version="0.1.0",
        authors=["Jane Doe", "John Smith"],
        dataset_license="CC0",
    )
    description = json.loads((tmp_path / "dataset_description.json").read_text())
    assert description["Authors"] == ["Jane Doe", "John Smith"]
    assert description["License"] == "CC0"
    assert description["GeneratedBy"][0]["Name"] == "ZIP-to-BIDS"
    assert (tmp_path / "README").exists()
    assert (tmp_path / "participants.tsv").exists()
