# Changelog

## Unreleased

- First private testing candidate for ZIP-to-BIDS.
- Converts selected DICOM acquisitions stored in ZIP archives without unpacking the full archive first.
- Reads CSV, Excel, and Word cohort tables and detects common subject, visit, image, modality, description, and optional task columns.
- Handles T1w, DWI, and BOLD naming; unresolved field maps are converted to a review cache rather than guessed into BIDS.
- Resting-state BOLD is identified only from explicit resting-state terms. Task-fMRI requires a task label and is flagged for review because event timing files are not generated.
- Downloads the current stable Linux dcm2niix release from the official `releases/latest` endpoint at runtime and records the reported version.
- Accepts dcm2niix exit status 3 when reporting its version.
- Shows acquisition-by-acquisition progress in Colab so long conversions do not look stalled.
- Saves dcm2niix logs and warning text; successful conversions with dcm2niix warnings are marked `REVIEW`.
- Saves the manifest after each acquisition and runs the BIDS Validator through Deno at the end when enabled.
- Adds tests, GitHub Actions, Dependabot, issue templates, privacy guidance, citation metadata, and an MIT license.
