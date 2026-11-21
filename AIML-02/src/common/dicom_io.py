"""DICOM read/write helpers (pydicom).

The brief asks us to "learn the DICOM format" and run an Orthanc DICOM server.
These helpers wrap a grayscale image into a minimal valid DICOM (so synthetic
data can populate Orthanc) and read pixel data back from real or synthetic DICOM
files, honouring Photometric Interpretation and Rescale slope/intercept.
"""
from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import (ExplicitVRLittleEndian, SecondaryCaptureImageStorage,
                         generate_uid)


def is_dicom(path: str | Path) -> bool:
    path = Path(path)
    if path.suffix.lower() in (".dcm", ".dicom"):
        return True
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except OSError:
        return False


def array_to_dicom(arr: np.ndarray, out_path: str | Path, *,
                   patient_id: str = "SYN0001",
                   patient_name: str = "Synthetic^Phantom",
                   modality: str = "DX") -> str:
    """Wrap an 8-bit grayscale array into a Secondary Capture DICOM file."""
    arr = np.ascontiguousarray(arr.astype(np.uint8))
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(out_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    now = datetime.datetime.now()
    ds.PatientName = patient_name
    ds.PatientID = patient_id
    ds.Modality = modality
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.StudyDate = now.strftime("%Y%m%d")
    ds.ContentDate = now.strftime("%Y%m%d")
    ds.ContentTime = now.strftime("%H%M%S")
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows, ds.Columns = arr.shape
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = arr.tobytes()
    ds.save_as(str(out_path), enforce_file_format=True)
    return str(out_path)


def read_dicom(path: str | Path) -> np.ndarray:
    """Return pixel data as a float32 array in [0, 1] (1 channel)."""
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    arr = arr * slope + intercept
    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        arr = arr.max() - arr                       # invert so bone is bright
    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return arr


def dicom_tags(path: str | Path) -> dict:
    """A few human-readable header tags for the viewer / audit log."""
    ds = pydicom.dcmread(str(path), stop_before_pixels=True)
    return {
        "patient_id": str(getattr(ds, "PatientID", "")),
        "modality": str(getattr(ds, "Modality", "")),
        "study_uid": str(getattr(ds, "StudyInstanceUID", "")),
        "rows": int(getattr(ds, "Rows", 0)),
        "columns": int(getattr(ds, "Columns", 0)),
    }
