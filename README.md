# Film-Dosimetry_App

## Overview
This repository contains a Python-based Graphical User Interface (GUI) application designed for the calibration and analysis of Gafchromic EBT4 film dosimetry. 

The software automates the process of:
1.  **Film Calibration:** Extracting pixel values from scanned films and fitting them to a rational function dose-response curve.
2.  **Dose Measurement:** Converting pixel readings from patient films into absorbed dose using the generated calibration parameters.

A key feature of this tool is the implementation of **orientation averaging** (averaging scans from "original" and "rotated" orientations) to mitigate the film orientation dependency.

## Features
* **Interactive GUI:** Built with `tkinter` for easy file selection and ROI drawing.
* **Red Channel Extraction:** Automatically extracts and processes the red color channel for maximum sensitivity.
* **Dual-Orientation Support:** Workflows for handling both landscape (original) and portrait (rotated) scans.
* **Curve Fitting:** Implements a rational function fit with automated parameter optimization:
  > Dose = a + b / (PixelValue - c)
* **Data Persistence:** Saves calibration parameters to Excel for consistent reuse during patient measurement.

## Package Requirements
* `tkinter` (Standard with Python)
* `numpy`
* `pandas`
* `matplotlib`
* `scipy`
* `Pillow` (PIL)
* `openpyxl`
* `tifffile`


## Usage
To launch the application, run the main script from your terminal:
```bash
python film_dosimetry_app.py
```

## Workflow

### 1. Calibration Mode
Use this mode to establish your film's dose-response curve.
* **Step 1:** Click "Film Calibration".
* **Step 2:** Load TIFF images for each dose level (e.g., 0, 50, 100... 800 cGy).
* **Step 3:** Draw an ROI on the canvas for each image.
* **Step 4:** The software calculates the mean pixel value (automatically removing outliers).
* **Step 5:** Click "Process Calibration and Save Result" to generate the calibration curve. The coefficients (a, b, c) are saved to an Excel file.

### 2. Measurement Mode
Use this mode to measure patient dose.
* **Step 1:** Click "Dose Measurement".
* **Step 2:** Load the Calibration Excel file generated in the previous step.
* **Step 3:** Load the patient film images (both Original and Rotated scans).
* **Step 4:** Draw ROIs on the target measurement areas.
* **Step 5:** The software computes the final dose (cGy) based on the averaged pixel value of the two orientations.

## AI Disclosure & Development
This software was developed as a custom solution for clinical film dosimetry.
* **Methodology:** The logical framework, GUI workflow, rational function model, and orientation-averaging protocol were conceptualized and validated by the authors.
* **Implementation:** The Python code generation was facilitated by a Large Language Model (Gemini 2.5 Pro, Google LLC).
* **Validation:** All algorithms and code outputs were rigorously validated against known datasets to ensure dosimetric accuracy.

## License
This project is licensed under the MIT License - see the [LICENSE] file for details.
