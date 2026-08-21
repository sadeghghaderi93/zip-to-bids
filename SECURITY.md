# Security and data privacy

ZIP-to-BIDS is intended for neuroimaging data, which may contain sensitive participant information.

Please do not upload DICOM files, cohort spreadsheets with participant information, generated BIDS datasets, or conversion logs containing sensitive paths to a public GitHub issue. If a bug depends on a particular dataset, reduce it to a synthetic example before sharing it.

The conversion command asks dcm2niix to anonymize BIDS sidecars with `-ba y`, but the original DICOM archive remains source data and should be handled according to the rules that apply to your study.

For software security problems, report the smallest reproducible example that does not include participant data.
