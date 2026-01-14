"""
Film Dosimetry Analysis Tool (GUI)
----------------------------------
Description: A Tkinter-based application for calibrating EBT4 film and performing 
             in-vivo dosimetry analysis using scanning orientation averaging.

AI Disclosure: 
    This software was developed with the assistance of a Large Language Model 
    (Gemini 2.5 Pro, Google LLC) for code generation and syntax optimization. 
    The logic, GUI design, and dosimetric algorithms were defined and validated 
    by the authors.

License: MIT License
"""

from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Toplevel, Label, Button, messagebox, Canvas, Text, Scrollbar, Entry
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from datetime import date
from scipy.optimize import curve_fit
from scipy import stats
import tifffile


class FilmDosimetryCalibrationApp(tk.Frame):
    def __init__(self, master=None, main_app=None):
        super().__init__(master=master)
        self.main_app = main_app
        self.title = "Film Dosimetry Calibration"

        self.dose_levels = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        self.orientations = ["original", "rotated"]
        self.current_dose_index = 0
        self.current_orientation_index = 0
        self.calibration_data = {}
        self.loaded_image = None
        self.image_tk = None
        self.canvas = None
        self.image_on_canvas = None
        self.start_x = None
        self.start_y = None
        self.rect_id = None
        self.roi_coords = None
        self.checkbox_vars = {}
        self.checkboxes = {}
        self.select_image_button = None
        self.confirm_button = None
        self.redraw_button = None
        self.save_result_button = None
        self.skip_button = None
        self.scatter = None # To store the scatter plot object
        self.plot_data_map = {} # To map dose to the index in plot_data
        self.plot_data_x = [] # Pixel values
        self.plot_data_y = [] # Doses
        self.pil_image = None # To store the PIL Image object
        self.first_save_click = True # Flag to track the first click of save button

        # Matplotlib plot setup
        self.fig, self.ax = plt.subplots(figsize=(4, 3))
        self.canvas_plot_tk = None

        self.create_widgets()
        self.update_message("Welcome to Film Dosimetry Calibration!")

    def create_widgets(self):
        self.start_button = Button(self, text="Start Calibration", command=self.start_calibration)
        self.start_button.grid(row=0, column=0, columnspan=3, pady=10)

        # Left: Image display (occupy all left side)
        self.canvas_frame = tk.Frame(self)
        self.canvas_frame.grid(row=1, column=0, rowspan=3, padx=10, pady=10, sticky="nsew")

        self.canvas = Canvas(self.canvas_frame, width=600, height=600, bg='lightgray')
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<ButtonPress-1>", self.start_roi)
        self.canvas.bind("<B1-Motion>", self.draw_roi)
        self.canvas.bind("<ButtonRelease-1>", self.end_roi)

        # Middle Top: Progress Checkbox List (Two Columns)
        self.checkbox_frame = tk.Frame(self)
        self.checkbox_frame.grid(row=1, column=1, padx=10, pady=10, sticky="n")
        status_label = Label(self.checkbox_frame, text="Film Processing Status:")
        status_label.grid(row=0, column=0, columnspan=2, sticky="w")
        row_num = 1
        for i, dose in enumerate(self.dose_levels):
            Label(self.checkbox_frame, text=f"{dose} Gy:").grid(row=row_num + i, column=0, sticky="w", padx=5)
            for j, orientation in enumerate(self.orientations):
                label_text = orientation.capitalize()
                var = tk.BooleanVar()
                cb = tk.Checkbutton(self.checkbox_frame, text=label_text, variable=var,
                                    command=lambda d=dose, o=orientation: self.checkbox_changed(d, o),
                                    state=tk.DISABLED) # Initially disabled
                cb.grid(row=row_num + i, column=j + 1, sticky="w")
                self.checkbox_vars[(dose, orientation)] = var
                self.checkboxes[(dose, orientation)] = cb

        # Middle Bottom: Buttons
        self.button_frame = tk.Frame(self)
        self.button_frame.grid(row=2, column=1, padx=10, pady=10, sticky="s")
        self.select_image_button = Button(self.button_frame, text="Select Image", command=self.select_image, state=tk.DISABLED)
        self.select_image_button.pack(pady=10) # Increased pady
        self.confirm_button = Button(self.button_frame, text="Confirm ROI", command=self.confirm_roi, state=tk.DISABLED)
        self.confirm_button.pack(pady=10) # Increased pady
        self.redraw_button = Button(self.button_frame, text="Redraw ROI", command=self.redraw_roi, state=tk.DISABLED)
        self.redraw_button.pack(pady=5)
        self.skip_button = Button(self.button_frame, text="Skip", command=self.skip_current, state=tk.DISABLED) # Initially disabled
        self.skip_button.pack(pady=10) # Increased pady
        self.save_result_button = Button(self.button_frame, text="Process Calibration and Save Result", command=self.save_results, state=tk.DISABLED)
        self.save_result_button.pack(pady=10) # Increased pady
        self.exit_button = Button(self.button_frame, text="Exit Calibration", command=self.main_app.show_main_menu)
        self.exit_button.pack(pady=10)

        # Top Right: Calibration Curve Plot
        self.plot_frame = tk.Frame(self)
        self.plot_frame.grid(row=0, column=2, rowspan=2, sticky="ne", padx=10, pady=10)

        self.canvas_plot_tk = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_plot_widget = self.canvas_plot_tk.get_tk_widget()
        self.canvas_plot_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.ax.set_xlabel("Average Pixel Value")
        self.ax.set_ylabel("Dose (x100)") # Updated y-axis label
        self.ax.set_title("Calibration Curve")
        self.ax.grid(True)
        self.fig.tight_layout()

        # Bottom Right: Output Text
        self.message_frame = tk.Frame(self)
        self.message_frame.grid(row=2, column=2, rowspan=2, sticky="sew", padx=10, pady=10)
        self.message_label = Label(self.message_frame, text="Output Messages:")
        self.message_label.pack()
        self.message_scrollbar = Scrollbar(self.message_frame)
        self.message_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.message_text = Text(self.message_frame, height=30, width=50, yscrollcommand=self.message_scrollbar.set)
        self.message_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.message_scrollbar.config(command=self.message_text.yview)

    def update_message(self, message):
        self.message_text.insert(tk.END, message + "\n")
        self.message_text.see(tk.END)

    def start_calibration(self):
        self.current_dose_index = 0
        self.current_orientation_index = 0
        self.calibration_data = {}
        self.plot_data_x = []
        self.plot_data_y = []
        self.plot_data_map = {}
        self.ax.clear()
        self.ax.set_xlabel("Average Pixel Value")
        self.ax.set_ylabel("Dose (x100)") # Updated y-axis label
        self.ax.set_title("Calibration Curve")
        self.ax.grid(True)
        self.fig.tight_layout()
        self.canvas_plot_tk.draw()
        self.update_message("Calibration started. \nPlease select the image for 0.0 Gy (original).")
        for dose, orientation in self.checkbox_vars:
            self.checkbox_vars[(dose, orientation)].set(False)
            self.checkboxes[(dose, orientation)].config(state=tk.DISABLED) # Initially disabled
        self.select_image_button.config(state=tk.NORMAL) # Enable Select Image button at the start
        self.start_button.config(state=tk.DISABLED)
        self.confirm_button.config(state=tk.DISABLED) # Confirm ROI should be disabled initially
        self.redraw_button.config(state=tk.DISABLED) # Redraw ROI should also be disabled initially
        self.save_result_button.config(state=tk.DISABLED) # Save Result button disabled at start
        self.skip_button.config(state=tk.NORMAL) # Enable Skip button at start
        self.scatter = None
        self.pil_image = None # Reset pil_image at the start of calibration
        self.first_save_click = True

    def select_image(self):
        dose = self.dose_levels[self.current_dose_index]
        orientation = self.orientations[self.current_orientation_index]
        file_path = filedialog.askopenfilename(
            title=f"Select the image for {dose} Gy ({orientation})",
            filetypes=[("TIFF files", "*.tif"), ("All files", "*.*")]
        )
        if file_path:
            img, warnings = self.load_tiff_image(file_path)
            if img is not None:  # Changed the if condition
                if warnings:
                    if not self.show_warning_popup(warnings):
                        self.update_message(f"Processing of {dose} Gy ({orientation}) aborted due to unacknowledged warnings.")
                        return
                self.loaded_image = img
                self.display_image()
                self.update_message(f"Image loaded for {dose} Gy ({orientation}). Draw an ROI.")
                self.confirm_button.config(state=tk.NORMAL) # Enable confirm button after image is loaded
        else:
            self.update_message(f"No image selected for {dose} Gy ({orientation}).")

    def load_tiff_image(self, image_path):
        warnings = []
        try:
            img = Image.open(image_path)
            img_array = np.array(img)
            return img_array, warnings
        except Exception as e:
            return None, [f"Error loading image using PIL: {e}"]

    def display_image(self):
        if self.loaded_image is not None:
            img_array = self.loaded_image
            print(f"Data type of loaded image array: {img_array.dtype}, Shape: {img_array.shape}") # Debugging

            if img_array.ndim == 3 and img_array.shape[-1] in [3, 4]: # Color image
                red_channel = img_array[:, :, 0]
            elif img_array.ndim == 2: # Grayscale image
                red_channel = img_array
            else:
                print("Unsupported image format for display.")
                return

            print(f"Data type of red_channel for display: {red_channel.dtype}, Shape: {red_channel.shape}") # Debugging
            print(f"Min/Max of red_channel for display: {np.min(red_channel)}, {np.max(red_channel)}") # Debugging

            # Remove contrast enhancement
            image_pil = None
            if red_channel.dtype == np.uint16:
                # Directly convert 16-bit to 8-bit by right-shifting 8 bits
                image_pil = Image.fromarray((red_channel >> 8).astype(np.uint8)).convert("L")
            else:
                image_pil = Image.fromarray(red_channel.astype(np.uint8)).convert("L")

            self.pil_image = image_pil # Store the PIL Image object

            max_size = 600
            width, height = image_pil.size
            if width > max_size or height > max_size:
                image_pil.thumbnail((max_size, max_size))
            self.image_tk = ImageTk.PhotoImage(image_pil)
            self.canvas.image = self.image_tk # Keep a reference to the image object
            self.canvas.config(width=self.image_tk.width(), height=self.image_tk.height())
            self.canvas.delete("all")
            self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)
            self.rect_id = None
            self.roi_coords = None
            self.redraw_button.config(state=tk.DISABLED)
            print("Image displayed successfully using tifffile.") # Debugging
        else:
            print("No image loaded for display.") # Debugging

    def show_warning_popup(self, warnings):
        if not warnings:
            return True
        popup = Toplevel(self)
        popup.title("Image Loading Warnings")
        warning_text = "\n".join(warnings)
        Label(popup, text=warning_text, justify='left').pack(padx=10, pady=10)
        proceed = tk.BooleanVar(value=False)
        Button(popup, text="Acknowledge and Proceed", command=lambda: proceed.set(True)).pack(pady=10)
        popup.wait_variable(proceed)
        return proceed.get()


    def start_roi(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
            self.roi_coords = None
        self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x + 1, self.start_y + 1, outline='red')
        self.redraw_button.config(state=tk.DISABLED) # Disable redraw when starting a new ROI

    def draw_roi(self, event):
        if self.start_x is not None and self.start_y is not None and self.rect_id:
            current_x = self.canvas.canvasx(event.x)
            current_y = self.canvas.canvasy(event.y)
            x1 = min(self.start_x, current_x)
            y1 = min(self.start_y, current_y)
            x2 = max(self.start_x, current_x)
            y2 = max(self.start_y, current_y)
            self.canvas.coords(self.rect_id, x1, y1, x2, y2)

    def end_roi(self, event):
        if self.start_x is not None and self.start_y is not None and self.rect_id:
            current_x = self.canvas.canvasx(event.x)
            current_y = self.canvas.canvasy(event.y)
            x1, y1 = min(self.start_x, current_x), min(self.start_y, current_y)
            x2, y2 = max(self.start_x, current_x), max(self.start_y, current_y)
            self.roi_coords = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            # self.update_message(f"ROI Coordinates (x, y, width, height): {self.roi_coords}")
            # print(f"ROI Coordinates in end_roi: {self.roi_coords}") # Debugging
            self.start_x = None
            self.start_y = None
            self.redraw_button.config(state=tk.NORMAL) # Enable redraw after ROI is drawn


    def confirm_roi(self):
        if self.roi_coords and self.pil_image is not None:
            x1_canvas, y1_canvas, w_canvas, h_canvas = self.roi_coords
            x2_canvas = x1_canvas + w_canvas
            y2_canvas = y1_canvas + h_canvas

            # print(f"ROI Coordinates (Canvas): (x1={x1_canvas}, y1={y1_canvas}, x2={x2_canvas}, y2={y2_canvas})")

            try:
                roi_pil = self.pil_image.crop((x1_canvas, y1_canvas, x2_canvas, y2_canvas))
                img_array_roi = np.array(roi_pil)

                # print(f"Shape of ROI array: {img_array_roi.shape}")
                # print(f"Data type of ROI array: {img_array_roi.dtype}")

                if img_array_roi.ndim == 3 and img_array_roi.shape[-1] in [3, 4]:
                    roi_data = img_array_roi[:, :, 0].flatten() # Red channel
                elif img_array_roi.ndim == 2:
                    roi_data = img_array_roi.flatten() # Grayscale
                else:
                    messagebox.showerror("Error", "Unsupported image format for ROI selection.")
                    return

                # print(f"Min/Max of roi_data: {np.min(roi_data)}, {np.max(roi_data)}")
                # print(f"Average of roi_data: {np.mean(roi_data)}")

                if roi_data.size > 0:
                    # Outlier removal using IQR
                    Q1 = np.percentile(roi_data, 25)
                    Q3 = np.percentile(roi_data, 75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    filtered_data = roi_data[(roi_data >= lower_bound) & (roi_data <= upper_bound)]

                    if filtered_data.size > 0:
                        avg_value = np.mean(filtered_data)
                        dose = self.dose_levels[self.current_dose_index]
                        orientation = self.orientations[self.current_orientation_index]
                        self.calibration_data.setdefault(dose, {})[orientation] = avg_value
                        self.update_message(f"Average red channel value (outliers removed): {avg_value:.2f} for {dose} Gy ({orientation})")
                        self.checkbox_vars[(dose, orientation)].set(True)
                        self.checkboxes[(dose, orientation)].config(state=tk.NORMAL)
                        self.canvas.delete(self.rect_id)
                        self.rect_id = None
                        self.roi_coords = None
                        self.advance_calibration()
                        self.select_image_button.config(state=tk.NORMAL)
                        self.redraw_button.config(state=tk.DISABLED)
                        self.update_plot(dose)
                    else:
                        messagebox.showerror("Error", "No valid data points left after outlier removal.")
                else:
                    messagebox.showerror("Error", "Selected ROI has zero area.")
            except Exception as e:
                messagebox.showerror("Error during ROI cropping or processing", str(e))
        else:
            messagebox.showerror("Error", "No ROI selected or image loaded.")

    def skip_current(self):
        dose = self.dose_levels[self.current_dose_index]
        orientation = self.orientations[self.current_orientation_index]
        self.update_message(f"Skipped {dose} Gy ({orientation}).")
        self.advance_calibration()
        self.select_image_button.config(state=tk.NORMAL)
        self.redraw_button.config(state=tk.DISABLED)
        if self.loaded_image:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
            self.roi_coords = None

    def advance_calibration(self):
        if self.current_orientation_index == 0: # Processing original
            if self.current_dose_index < len(self.dose_levels) - 1:
                self.current_dose_index += 1
                self.update_message(f"Ready for {self.dose_levels[self.current_dose_index]} Gy (original).")
            else:
                self.current_orientation_index = 1
                self.current_dose_index = 0
                self.update_message(f"Ready for {self.dose_levels[self.current_dose_index]} Gy (rotated).")
        elif self.current_orientation_index == 1: # Processing rotated
            if self.current_dose_index < len(self.dose_levels) - 1:
                self.current_dose_index += 1
                self.update_message(f"Ready for {self.dose_levels[self.current_dose_index]} Gy (rotated).")
            else:
                self.update_message("Calibration data collection complete. You can now review the processing status and click 'Process Calibration and Save Result'.")
                self.save_result_button.config(state=tk.NORMAL)
                return

    def checkbox_changed(self, dose, orientation):
        if not self.checkbox_vars[(dose, orientation)].get():
            if messagebox.askyesno("Confirmation", f"Are you sure you want to remove the data for {dose} Gy ({orientation})?"):
                if dose in self.calibration_data:
                    if orientation in self.calibration_data[dose]:
                        del self.calibration_data[dose][orientation]
                        self.update_message(f"Removed data for {dose} Gy ({orientation}).")
                        self.update_plot(dose)
                        self.checkboxes[(dose, orientation)].config(state=tk.DISABLED) # Gray-out the checkbox
                        if not self.calibration_data[dose]:
                            del self.calibration_data[dose]
                    else:
                        self.update_message(f"No data found for {dose} Gy ({orientation}) to remove.")
            else:
                # If user cancels, re-check the box
                self.checkbox_vars[(dose, orientation)].set(True)

    def update_plot(self, dose):
        if dose in self.calibration_data:
            original_value = self.calibration_data[dose].get("original")
            rotated_value = self.calibration_data[dose].get("rotated")

            plot_point = None
            if original_value is not None and rotated_value is not None:
                plot_point = (original_value + rotated_value) / 2
            elif original_value is not None:
                plot_point = original_value
            elif rotated_value is not None:
                plot_point = rotated_value

            dose_for_plot = dose * 100

            if plot_point is not None:
                if dose not in self.plot_data_map:
                    self.plot_data_x.append(plot_point)
                    self.plot_data_y.append(dose_for_plot)
                    self.plot_data_map[dose] = len(self.plot_data_x) - 1
                else:
                    index = self.plot_data_map[dose]
                    self.plot_data_x[index] = plot_point
                    self.plot_data_y[index] = dose_for_plot
            elif dose in self.plot_data_map:
                # If no valid point to plot, remove if it exists
                index_to_remove = self.plot_data_map[dose]
                del self.plot_data_x[index_to_remove]
                del self.plot_data_y[index_to_remove]
                del self.plot_data_map[dose]
                # Rebuild plot_data_map
                self.plot_data_map = {}
                for i, d in enumerate(self.plot_data_y):
                    self.plot_data_map[d / 100] = i

            self.ax.clear()
            if self.plot_data_x and self.plot_data_y:
                self.scatter = self.ax.scatter(self.plot_data_x, self.plot_data_y)
            self.ax.set_xlabel("Average Pixel Value")
            self.ax.set_ylabel("Dose (x100)") # Updated y-axis label
            self.ax.set_title("Calibration Curve")
            self.ax.grid(True)
            self.fig.tight_layout()
            self.canvas_plot_tk.draw()
        elif dose in self.plot_data_map:
            # If dose is in plot_data_map but not in calibration_data, remove
            index_to_remove = self.plot_data_map[dose]
            del self.plot_data_x[index_to_remove]
            del self.plot_data_y[index_to_remove]
            del self.plot_data_map[dose]
            # Rebuild plot_data_map
            self.plot_data_map = {}
            for i, d in enumerate(self.plot_data_y):
                self.plot_data_map[d / 100] = i
            self.ax.clear()
            if self.plot_data_x and self.plot_data_y:
                self.scatter = self.ax.scatter(self.plot_data_x, self.plot_data_y)
            self.ax.set_xlabel("Average Pixel Value")
            self.ax.set_ylabel("Dose (x100)") # Updated y-axis label
            self.ax.set_title("Calibration Curve")
            self.ax.grid(True)
            self.fig.tight_layout()
            self.canvas_plot_tk.draw()

    def update_plot_after_removal(self, dose_to_remove):
        self.update_plot(dose_to_remove)

    def redraw_roi(self):
        if self.loaded_image:
            if self.rect_id:
                self.canvas.delete(self.rect_id)
                self.rect_id = None
                self.roi_coords = None
            self.update_message("Please redraw the ROI.")
            self.confirm_button.config(state=tk.NORMAL) # Enable confirm button after requesting redraw
            self.redraw_button.config(state=tk.DISABLED) # Disable redraw while redrawing
            self.start_x = None
            self.start_y = None

    def show_results(self): # This method will now be called from save_results
        results_text = "Calibration Data:\n"
        for dose, values in self.calibration_data.items():
            results_text += f"{dose} Gy: {values}\n"
        results_window = Toplevel(self)
        results_window.title("Calibration Results")
        results_label = Label(results_window, text=results_text)
        results_label.pack(padx=20, pady=20)

    def save_results(self):
        if not self.calibration_data:
            messagebox.showerror("Error", "No calibration data available. Please collect data first.")
            return

        if self.first_save_click:
            self.first_save_click = False
            self._perform_curve_fitting()
        else:
            self.retry_curve_fit_prompt()

        # Show calibration data always
        self.show_results()

    def _perform_curve_fitting(self, initial_guess=None):
        self.update_message("Performing curve fitting...")
        filename = f"Film_Calibration_{date.today().strftime('%Y%m%d')}.xlsx"
        try:
            # Display dataset for curve fitting
            fit_data_window = Toplevel(self)
            fit_data_window.title("Dataset for Curve Fitting")
            dataset_text = "Dataset for Curve Fitting:\n"
            for i, dose_times_100 in enumerate(self.plot_data_y):
                pixel_value = self.plot_data_x[i]
                dataset_text += f"Dose: {dose_times_100}, Average Pixel Value: {pixel_value:.2f}\n"

            dataset_label = Label(fit_data_window, text=dataset_text, justify='left')
            dataset_label.pack(padx=10, pady=10)

            with pd.ExcelWriter(filename) as writer:
                # Prepare calibration data for Excel
                excel_data = []
                for dose in self.dose_levels:
                    original = self.calibration_data.get(dose, {}).get("original", "")
                    rotated = self.calibration_data.get(dose, {}).get("rotated", "")
                    avg = (original + rotated) / 2 if original != "" and rotated != "" else (original if original != "" else (rotated if rotated != "" else ""))
                    excel_data.append({"Dose (Gy)": dose, "Original Pixel Value": original, "Rotated Pixel Value": rotated, "Average Pixel Value": avg})
                df_calibration = pd.DataFrame(excel_data)
                df_calibration.to_excel(writer, sheet_name='Calibration Data', index=False)

                # Perform curve fitting using plot data
                averaged_pixel_values = self.plot_data_x
                doses_for_fit = self.plot_data_y

                if len(averaged_pixel_values) > 2: # Need at least 3 points for fitting
                    try:
                        # Define bounds for the parameters: a < 0 and b > 0
                        bounds = ((-np.inf, 0, -np.inf), (0, np.inf, np.inf))
                        p0 = initial_guess if initial_guess else [-3, 5, 3]
                        popt, pcov = curve_fit(self.fit_function, averaged_pixel_values, doses_for_fit, p0=p0, bounds=bounds)
                        a, b, c = popt
                        df_fit_params = pd.DataFrame([{"Parameter": "a", "Value": a}, {"Parameter": "b", "Value": b}, {"Parameter": "c", "Value": c}])
                        df_fit_params.to_excel(writer, sheet_name='Fit Parameters', index=False)

                        # Plot fitted curve - the y-axis label should reflect the multiplication
                        pixel_range = np.linspace(min(averaged_pixel_values), max(averaged_pixel_values), 100)
                        fitted_dose = self.fit_function(pixel_range, a, b, c)
                        self.ax.plot(pixel_range, fitted_dose, 'g--', label='Fitted Curve')
                        self.ax.set_ylabel("Dose (x100)") # Update y-axis label
                        self.ax.legend()
                        self.fig.tight_layout()
                        self.canvas_plot_tk.draw()

                        self.update_message(f"Calibration data and fit parameters saved to {filename}")
                        self.update_message(f"Fitted parameters: a={100 * a:.2f}, b={10000 * b:.2f}, c={10 * c:.2f}")

                    except RuntimeError:
                        self.update_message("Error: Optimal parameters not found - curve fitting failed (check data or bounds).")
                else:
                    self.update_message("Warning: Not enough data points to perform curve fitting.")

        except Exception as e:
            self.update_message(f"Error saving results: {e}")

    def retry_curve_fit_prompt(self):
        retry_window = Toplevel(self)
        retry_window.title("Retry Curve Fitting")

        Label(retry_window, text="Enter initial guess for parameter 'a' (negative):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        entry_a = Entry(retry_window)
        entry_a.grid(row=0, column=1, padx=5, pady=5)

        Label(retry_window, text="Enter initial guess for parameter 'b' (positive):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        entry_b = Entry(retry_window)
        entry_b.grid(row=1, column=1, padx=5, pady=5)

        Label(retry_window, text="Enter initial guess for parameter 'c':").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        entry_c = Entry(retry_window)
        entry_c.grid(row=2, column=1, padx=5, pady=5)

        def perform_retry_fit():
            try:
                a_guess = float(entry_a.get()) / 100
                b_guess = float(entry_b.get()) / 10000
                c_guess = float(entry_c.get()) / 10
                self._perform_curve_fitting(initial_guess=[a_guess, b_guess, c_guess])
                retry_window.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numeric values for a, b, and c.")

        retry_button = Button(retry_window, text="Fit with New Guess", command=perform_retry_fit)
        retry_button.grid(row=3, column=0, columnspan=2, pady=10)

    def fit_function(self, x, a, b, c):
        return 100 * a + 10000 * b / (x - 10 * c)



class FilmDosimetryMeasurementApp(tk.Frame):
    def __init__(self, master=None, main_app=None):
        super().__init__(master=master)
        self.main_app = main_app
        self.title = "Dose Measurement"
        self.calibration_file_path = tk.StringVar()
        self.calibration_params = {}
        self.measurement_image_paths = {"original": None, "rotated": None}
        self.loaded_images = {"original": None, "rotated": None}
        self.image_tks = {"original": None, "rotated": None}
        self.pil_images = {"original": None, "rotated": None}
        self.canvases = {"original": None, "rotated": None}
        self.image_on_canvases = {"original": None, "rotated": None}
        self.start_coords = {"original": None, "rotated": None}
        self.rect_ids = {"original": None, "rotated": None}
        self.roi_coords = {"original": None, "rotated": None}
        self.avg_pixel_values = {"original": None, "rotated": None}
        self.calculated_dose = tk.StringVar()
        self.avg_pixel_value_original_str = tk.StringVar()
        self.avg_pixel_value_rotated_str = tk.StringVar()
        self.avg_pixel_value_combined_str = tk.StringVar()
        self.current_roi_orientation = None

        self.create_widgets()

    def create_widgets(self):
        # Calibration File Selection
        calib_frame = tk.Frame(self)
        calib_frame.pack(pady=10)
        calib_label = Label(calib_frame, text="Calibration File:")
        calib_label.pack(side=tk.LEFT)
        calib_entry = Entry(calib_frame, textvariable=self.calibration_file_path, width=50, state="readonly")
        calib_entry.pack(side=tk.LEFT, padx=5)
        calib_button = Button(calib_frame, text="Select File", command=self.select_calibration_file)
        calib_button.pack(side=tk.LEFT)

        # Measurement Image Selection
        measure_frame = tk.Frame(self)
        measure_frame.pack(pady=10)
        self.select_original_button = Button(measure_frame, text="Select Original Image", command=lambda: self.select_measurement_image("original"))
        self.select_original_button.pack(side=tk.LEFT, padx=5)
        self.select_rotated_button = Button(measure_frame, text="Select Rotated Image", command=lambda: self.select_measurement_image("rotated"))
        self.select_rotated_button.pack(side=tk.LEFT, padx=5)

        # Image Display Frame
        image_display_frame = tk.Frame(self)
        image_display_frame.pack(pady=10, padx=10)

        # Original Image Canvas
        original_frame = tk.Frame(image_display_frame)
        original_frame.pack(side=tk.LEFT, padx=5)
        Label(original_frame, text="Original Image").pack()
        self.canvases["original"] = Canvas(original_frame, width=300, height=200, bg='lightgray')
        self.canvases["original"].pack(fill="both", expand=True)
        self.canvases["original"].bind("<ButtonPress-1>", lambda event: self.start_roi(event, "original"))
        self.canvases["original"].bind("<B1-Motion>", lambda event: self.draw_roi(event, "original"))
        self.canvases["original"].bind("<ButtonRelease-1>", lambda event: self.end_roi(event, "original"))

        # Rotated Image Canvas
        rotated_frame = tk.Frame(image_display_frame)
        rotated_frame.pack(side=tk.LEFT, padx=5)
        Label(rotated_frame, text="Rotated Image").pack()
        self.canvases["rotated"] = Canvas(rotated_frame, width=300, height=200, bg='lightgray')
        self.canvases["rotated"].pack(fill="both", expand=True)
        self.canvases["rotated"].bind("<ButtonPress-1>", lambda event: self.start_roi(event, "rotated"))
        self.canvases["rotated"].bind("<B1-Motion>", lambda event: self.draw_roi(event, "rotated"))
        self.canvases["rotated"].bind("<ButtonRelease-1>", lambda event: self.end_roi(event, "rotated"))

        # Display Average Pixel Values
        avg_pixel_frame = tk.Frame(self)
        avg_pixel_frame.pack(pady=5)
        Label(avg_pixel_frame, text="Avg Pixel Value (Original):").pack(side=tk.LEFT)
        Label(avg_pixel_frame, textvariable=self.avg_pixel_value_original_str, font=("Arial", 12)).pack(side=tk.LEFT, padx=5)
        Label(avg_pixel_frame, text="Avg Pixel Value (Rotated):").pack(side=tk.LEFT, padx=10)
        Label(avg_pixel_frame, textvariable=self.avg_pixel_value_rotated_str, font=("Arial", 12)).pack(side=tk.LEFT, padx=5)
        Label(avg_pixel_frame, text="Combined Average:").pack(side=tk.LEFT, padx=10)
        Label(avg_pixel_frame, textvariable=self.avg_pixel_value_combined_str, font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=5)
        self.avg_pixel_value_original_str.set("N/A")
        self.avg_pixel_value_rotated_str.set("N/A")
        self.avg_pixel_value_combined_str.set("N/A")

        # Display Calculated Dose
        dose_frame = tk.Frame(self)
        dose_frame.pack(pady=5)
        Label(dose_frame, text="Calculated Dose (Gy):").pack(side=tk.LEFT)
        Label(dose_frame, textvariable=self.calculated_dose, font=("Arial", 14, "bold")).pack(side=tk.LEFT)
        self.calculated_dose.set("N/A")

        # Back to Main Menu
        back_button = Button(self, text="Back to Main Menu", command=self.main_app.show_main_menu)
        back_button.pack(pady=20)

    def select_calibration_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Calibration Excel File",
            filetypes=[("Excel files", "*.xlsx")]
        )
        if file_path:
            self.calibration_file_path.set(file_path)
            try:
                df_fit = pd.read_excel(file_path, sheet_name='Fit Parameters')
                self.calibration_params = df_fit.set_index('Parameter')['Value'].to_dict()
                if 'a' in self.calibration_params and 'b' in self.calibration_params and 'c' in self.calibration_params:
                    messagebox.showinfo("Calibration Loaded", "Calibration parameters loaded successfully.")
                else:
                    messagebox.showerror("Error", "Could not find all required parameters (a, b, c) in the calibration file.")
                    self.calibration_params = {}
            except Exception as e:
                messagebox.showerror("Error", f"Error loading calibration file: {e}")
                self.calibration_params = {}

    def select_measurement_image(self, orientation):
        file_path = filedialog.askopenfilename(
            title=f"Select {orientation.capitalize()} Measurement Image",
            filetypes=[("TIFF files", "*.tif"), ("All files", "*.*")]
        )
        if file_path:
            self.measurement_image_paths[orientation] = file_path
            img, warnings = self.load_tiff_image(file_path)
            if img is not None:
                self.loaded_images[orientation] = img
                self.pil_images[orientation] = self.convert_to_pil(img)
                self.display_image(orientation)
                if self.loaded_images["original"] is not None and self.loaded_images["rotated"] is not None:
                    pass # Both images loaded
            else:
                messagebox.showerror("Error", f"Could not load {orientation} image: {warnings[0] if warnings else 'Unknown error'}")

    def load_tiff_image(self, image_path):
        warnings = []
        try:
            img = Image.open(image_path)
            img_array = np.array(img)
            return img_array, warnings
        except Exception as e:
            return None, [f"Error loading image using PIL: {e}"]

    def convert_to_pil(self, img_array):
        if img_array.ndim == 3 and img_array.shape[-1] in [3, 4]: # Color image
            red_channel = img_array[:, :, 0]
        elif img_array.ndim == 2: # Grayscale image
            red_channel = img_array
        else:
            messagebox.showerror("Error", "Unsupported image format for display.")
            return None

        if red_channel.dtype == np.uint16:
            return Image.fromarray((red_channel >> 8).astype(np.uint8)).convert("L")
        else:
            return Image.fromarray(red_channel.astype(np.uint8)).convert("L")

    def display_image(self, orientation):
        pil_image = self.pil_images.get(orientation)
        canvas = self.canvases.get(orientation)
        if pil_image and canvas:
            max_canvas_width = 300
            max_canvas_height = 200
            img_width, img_height = pil_image.size

            if img_width > max_canvas_width or img_height > max_canvas_height:
                pil_image.thumbnail((max_canvas_width, max_canvas_height))

            self.image_tks[orientation] = ImageTk.PhotoImage(pil_image)
            canvas.image = self.image_tks[orientation] # Keep a reference
            canvas.config(width=self.image_tks[orientation].width(), height=self.image_tks[orientation].height())
            canvas.delete("all")
            self.image_on_canvases[orientation] = canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tks[orientation])
            if self.roi_coords[orientation]:
                x1, y1, w, h = self.roi_coords[orientation]
                self.rect_ids[orientation] = canvas.create_rectangle(x1, y1, x1 + w, y1 + h, outline='red')
            else:
                self.rect_ids[orientation] = None

    def start_roi(self, event, orientation):
        canvas = self.canvases.get(orientation)
        if canvas:
            self.current_roi_orientation = orientation
            self.start_coords[orientation] = (canvas.canvasx(event.x), canvas.canvasy(event.y))
            if self.rect_ids[orientation]:
                canvas.delete(self.rect_ids[orientation])
                self.rect_ids[orientation] = None
                self.roi_coords[orientation] = None
            start_x, start_y = self.start_coords[orientation]
            self.rect_ids[orientation] = canvas.create_rectangle(start_x, start_y, start_x + 1, start_y + 1, outline='red')
            self.avg_pixel_values[orientation] = None
            if orientation == "original":
                self.avg_pixel_value_original_str.set("N/A")
            elif orientation == "rotated":
                self.avg_pixel_value_rotated_str.set("N/A")
            self.avg_pixel_value_combined_str.set("N/A")
            self.calculated_dose.set("N/A")

    def draw_roi(self, event, orientation):
        canvas = self.canvases.get(orientation)
        start_coord = self.start_coords.get(orientation)
        rect_id = self.rect_ids.get(orientation)
        if canvas and start_coord and rect_id:
            start_x, start_y = start_coord
            current_x = canvas.canvasx(event.x)
            current_y = canvas.canvasy(event.y)
            x1 = min(start_x, current_x)
            y1 = min(start_y, current_y)
            x2 = max(start_x, current_x)
            y2 = max(start_y, current_y)
            canvas.coords(rect_id, x1, y1, x2, y2)

    def end_roi(self, event, orientation):
        canvas = self.canvases.get(orientation)
        start_coord = self.start_coords.get(orientation)
        if canvas and start_coord:
            start_x, start_y = start_coord
            current_x = canvas.canvasx(event.x)
            current_y = canvas.canvasy(event.y)
            x1, y1 = min(start_x, current_x), min(start_y, current_y)
            x2, y2 = max(start_x, current_x), max(start_y, current_y)
            self.roi_coords[orientation] = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            self.start_coords[orientation] = None
            self.process_roi(orientation)
            if self.roi_coords["original"] is not None and self.roi_coords["rotated"] is not None:
                self.calculate_combined_dose()

    def process_roi(self, orientation):
        roi_coords = self.roi_coords.get(orientation)
        pil_image = self.pil_images.get(orientation)
        if roi_coords and pil_image:
            x1_canvas, y1_canvas, w_canvas, h_canvas = roi_coords
            x2_canvas = x1_canvas + w_canvas
            y2_canvas = y1_canvas + h_canvas

            try:
                roi_pil = pil_image.crop((x1_canvas, y1_canvas, x2_canvas, y2_canvas))
                img_array_roi = np.array(roi_pil)

                if img_array_roi.ndim == 3 and img_array_roi.shape[-1] in [3, 4]:
                    roi_data = img_array_roi[:, :, 0].flatten() # Red channel
                elif img_array_roi.ndim == 2:
                    roi_data = img_array_roi.flatten() # Grayscale
                else:
                    messagebox.showerror("Error", f"Unsupported image format for {orientation} ROI selection.")
                    return

                if roi_data.size > 0:
                    # Outlier removal using IQR
                    Q1 = np.percentile(roi_data, 25)
                    Q3 = np.percentile(roi_data, 75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    filtered_data = roi_data[(roi_data >= lower_bound) & (roi_data <= upper_bound)]

                    if filtered_data.size > 0:
                        avg_pixel = np.mean(filtered_data)
                        self.avg_pixel_values[orientation] = avg_pixel
                        if orientation == "original":
                            self.avg_pixel_value_original_str.set(f"{avg_pixel:.2f}")
                        elif orientation == "rotated":
                            self.avg_pixel_value_rotated_str.set(f"{avg_pixel:.2f}")
                        return
                    else:
                        messagebox.showerror("Error", f"No valid data points left after outlier removal in {orientation} ROI.")
                        self.avg_pixel_values[orientation] = None
                        if orientation == "original":
                            self.avg_pixel_value_original_str.set("N/A")
                        elif orientation == "rotated":
                            self.avg_pixel_value_rotated_str.set("N/A")
                        return
                else:
                    messagebox.showerror("Error", f"Selected ROI has zero area in {orientation} image.")
                    self.avg_pixel_values[orientation] = None
                    if orientation == "original":
                        self.avg_pixel_value_original_str.set("N/A")
                    elif orientation == "rotated":
                        self.avg_pixel_value_rotated_str.set("N/A")
                    return
            except Exception as e:
                messagebox.showerror(f"Error during {orientation} ROI processing", str(e))
                self.avg_pixel_values[orientation] = None
                if orientation == "original":
                    self.avg_pixel_value_original_str.set("N/A")
                elif orientation == "rotated":
                    self.avg_pixel_value_rotated_str.set("N/A")
                return
        else:
            self.avg_pixel_values[orientation] = None
            if orientation == "original":
                self.avg_pixel_value_original_str.set("N/A")
            elif orientation == "rotated":
                self.avg_pixel_value_rotated_str.set("N/A")

    def calculate_combined_dose(self):
        avg_original = self.avg_pixel_values.get("original")
        avg_rotated = self.avg_pixel_values.get("rotated")

        if avg_original is not None and avg_rotated is not None:
            combined_avg = (avg_original + avg_rotated) / 2
            self.avg_pixel_value_combined_str.set(f"{combined_avg:.2f}")
            self.calculate_dose(combined_avg)
        else:
            self.avg_pixel_value_combined_str.set("N/A")
            self.calculated_dose.set("N/A")

    def calculate_dose(self, avg_pixel_value):
        if not self.calibration_params:
            self.calculated_dose.set("Error: No calibration loaded")
            return
        if avg_pixel_value is None:
            self.calculated_dose.set("N/A")
            return

        try:
            a = self.calibration_params.get('a', None)
            b = self.calibration_params.get('b', None)
            c = self.calibration_params.get('c', None)

            if a is not None and b is not None and c is not None:
                # Calibration function: Dose = 100 * a + 10000 * b / (PixelValue - 10 * c)
                pixel_value = avg_pixel_value
                if (pixel_value - 10 * c) == 0:
                    self.calculated_dose.set("Error: Division by zero")
                    return
                dose_gy = (100 * a + 10000 * b / (pixel_value - 10 * c)) / 100 # Divide by 100
                self.calculated_dose.set(f"{dose_gy:.3f}")
            else:
                self.calculated_dose.set("Error: Missing calibration params")
        except Exception as e:
            self.calculated_dose.set(f"Error: {e}")

class FilmDosimetryApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Film Dosimetry")
        self.geometry("500x100")  # Increased window size
        self.create_main_menu()
        self.current_window = None

    def create_main_menu(self):
        self.menu_frame = tk.Frame(self)
        self.menu_frame.pack(side=tk.TOP, fill=tk.X)

        calibration_button = Button(self.menu_frame, text="Film Calibration", command=self.open_calibration_app, padx=20, pady=10)
        calibration_button.pack(side=tk.LEFT, padx=25)

        measurement_button = Button(self.menu_frame, text="Dose Measurement", command=self.open_measurement_app, padx=20, pady=10)
        measurement_button.pack(side=tk.LEFT, padx=10)

        about_button = Button(self.menu_frame, text="About", command=self.show_about_info, padx=20, pady=10)
        about_button.pack(side=tk.LEFT, padx=10)

    def open_calibration_app(self):
        self.clear_current_window()
        self.title("Film Dosimetry - Calibration")
        self.calibration_app = FilmDosimetryCalibrationApp(master=self, main_app=self)
        self.calibration_app.pack(fill="both", expand=True)

        self.update()  # Force the window to update and realize the size of the new content
        self.geometry(f"{self.winfo_reqwidth()}x{self.winfo_reqheight()}")

        self.current_window = self.calibration_app

    def open_measurement_app(self):
        self.clear_current_window()
        self.title("Film Dosimetry - Dose Measurement")
        self.measurement_app = FilmDosimetryMeasurementApp(master=self, main_app=self)
        self.measurement_app.pack(fill="both", expand=True)

        self.update()  # Force the window to update
        self.geometry(f"{self.winfo_reqwidth()}x{self.winfo_reqheight()}")

        self.current_window = self.measurement_app

    def show_about_info(self):
        about_window = Toplevel(self)
        about_window.title("About")
        about_window.geometry("600x400")  # Set the desired width and height

        instructions = (
            "Film Dosimetry Application Instructions:\n\n"
            "1. Film Calibration:\n"
            "   a. Click the 'Film Calibration' button.\n"
            "   b. Click 'Select Folder' and choose the folder containing your calibration images.\n"
            "   c. On the displayed image, draw a rectangle to select a Region Of Interest (ROI).\n"
            "   d. Click 'Confirm ROI' to record the average pixel value for the selected dose.\n"
            "   e. Repeat steps c-d for all your calibration dose points.\n"
            "   f. Click 'Process Calibration and Save Result'. Review the plot and fit parameters.\n\n"
            "2. Dose Measurement:\n"
            "   a. Click the 'Dose Measurement' button.\n"
            "   b. Click 'Select File' next to 'Calibration File' and choose the Excel file saved during calibration.\n"
            "   c. Click 'Select Original Image' and choose the original measurement film image.\n"
            "   d. On the left canvas, draw a rectangle to select an ROI on the original image.\n"
            "   e. Click 'Select Rotated Image' and choose the rotated measurement film image.\n"
            "   f. On the right canvas, draw a rectangle to select a corresponding ROI on the rotated image.\n"
            "   g. The average pixel values for both images, the combined average, and the calculated dose (Gy) will be displayed below the images.\n\n"
        )

        text_area = tk.Text(about_window, wrap=tk.WORD)
        text_area.insert(tk.END, instructions)
        text_area.config(state=tk.DISABLED)  # Make it read-only
        text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(about_window, command=text_area.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_area.config(yscrollcommand=scrollbar.set)

        about_window.transient(self)  # Make it a dependent window
        about_window.grab_set()       # Make it modal
        self.wait_window(about_window) # Wait until the window is closed

    def show_main_menu(self):
        self.clear_current_window()
        self.title("Film Dosimetry")
        self.create_main_menu()
        self.geometry("700x350")

    def clear_current_window(self):
        if self.current_window:
            self.current_window.destroy()
            self.current_window = None
        if hasattr(self, 'menu_frame') and self.menu_frame is not None:
            self.menu_frame.destroy()
            self.menu_frame = None

if __name__ == "__main__":
    main_app = FilmDosimetryApp()
    main_app.mainloop()