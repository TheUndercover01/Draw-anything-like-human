import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time
import os
import sys
import argparse
from util import util
from math import sqrt, ceil
from collections import defaultdict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time
import os
import sys


def process_single_image(image_path, output_dir='output'):
    """Process a single image using the pix2pix model"""
    # Override the default arguments
    sys.argv = [
        sys.argv[0],
        '--dataroot', os.path.dirname(image_path),
        '--results_dir', output_dir,
        '--checkpoints_dir', 'output',  # Corrected checkpoints directory
        '--name', 'pretrained',
        '--model', 'pix2pix',
        '--which_direction', 'AtoB',
        '--norm', 'batch',
        '--input_nc', '3',
        '--output_nc', '1',
        '--which_model_netG', 'resnet_9blocks',
        '--no_dropout',
        '--dataset_mode', 'test_dir',  # Correct dataset mode for single image
        '--phase', 'test'
    ]

    opt = TestOptions().parse()
    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    if not os.path.isdir(opt.results_dir):
        os.makedirs(opt.results_dir)

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)

    image_name = os.path.basename(image_path).replace('.jpg', '.png')
    #output_subdir = os.path.join(opt.results_dir, opt.name, 'test_latest', 'images')
    output_path = os.path.join(output_dir, image_name)

    # test
    for i, data in enumerate(dataset):
        model.set_input(data)
        img_path = model.get_image_paths()
        print('Processing %04d (%s)' % (i + 1, img_path[0]))
        model.test()
        model.write_image(opt.results_dir) # Save image to the correct subdirectory

    print(f"Output saved to: {output_path}")
    return output_path



def get_connected_strokes(image, min_length=10, fixed_thickness=2):
    """Extract connected strokes from the image with consistent thickness"""
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    stroke_groups = []

    for label in range(1, num_labels):
        mask = np.zeros_like(binary)
        mask[labels == label] = 255

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        if len(contours) > 0:
            main_contour = max(contours, key=cv2.contourArea)
            thickness = fixed_thickness
            bottom_point = tuple(main_contour[main_contour[:, :, 1].argmax()][0])

            stroke_groups.append({
                'contour': main_contour,
                'thickness': thickness,
                'start_point': bottom_point
            })

    stroke_groups.sort(key=lambda x: (-x['start_point'][1], x['start_point'][0]))
    return stroke_groups

class DrawingAnimator:
    def __init__(self, canvas, stroke_groups, original_width, original_height, delay=20, step_size=4):
        self.canvas = canvas
        self.stroke_groups = stroke_groups
        self.original_width = original_width
        self.original_height = original_height
        self.delay = delay
        self.step_size = step_size
        self.drawn_elements = []
        self.current_stroke = None
        self.animation_running = False
        self.animation_tasks = []
        self.tracing_dot = None
        self.dot_size = 16  # Increased dot size

    def get_scale_factors(self):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width < 10 or canvas_height < 10:
            canvas_width = max(self.canvas.winfo_reqwidth(), 800)
            canvas_height = max(self.canvas.winfo_reqheight(), 600)

        scale_x = canvas_width / self.original_width
        scale_y = canvas_height / self.original_height
        scale = min(scale_x, scale_y)
        return scale, scale

    def transform_point(self, point):
        scale_x, scale_y = self.get_scale_factors()
        x, y = point
        return x * scale_x, y * scale_y

    def create_tracing_dot(self, x, y, color="red"):
        dot_size = self.dot_size
        if self.tracing_dot is None:
            self.tracing_dot = self.canvas.create_oval(
                x - dot_size / 2, y - dot_size / 2,
                x + dot_size / 2, y + dot_size / 2,
                fill=color, outline="white", width=1
            )
        else:
            self.canvas.coords(
                self.tracing_dot,
                x - dot_size / 2, y - dot_size / 2,
                x + dot_size / 2, y + dot_size / 2
            )
        self.canvas.tag_raise(self.tracing_dot)
        return self.tracing_dot

    def draw_strokes_fast(self):
        if self.animation_running:
            self.cancel_animation()

        self.animation_running = True
        scale_x, scale_y = self.get_scale_factors()

        for stroke_group in self.stroke_groups:
            contour = stroke_group['contour']
            thickness = stroke_group['thickness'] * min(scale_x, scale_y)

            points = [self.transform_point(tuple(point[0])) for point in contour]
            if len(points) < 2:
                continue

            if len(points) > 0:
                self.create_tracing_dot(points[0][0], points[0][1])
                self.canvas.update()

            line = self.canvas.create_line(
                *[coord for point in points for coord in point],
                fill="black",
                width=thickness,
                capstyle=tk.ROUND,
                joinstyle=tk.ROUND,
                smooth=True
            )
            self.drawn_elements.append(line)

            if len(points) > 0:
                self.create_tracing_dot(points[-1][0], points[-1][1])

            self.canvas.update()
            time.sleep(0.005)

            if not self.animation_running:
                break

        self.animation_running = False

    def draw_strokes_animated(self):
        if self.animation_running:
            self.cancel_animation()

        self.animation_running = True
        scale_x, scale_y = self.get_scale_factors()

        def draw_next_stroke_group(group_index=0):
            if group_index >= len(self.stroke_groups) or not self.animation_running:
                self.animation_running = False
                return

            stroke_group = self.stroke_groups[group_index]
            contour = stroke_group['contour']
            thickness = stroke_group['thickness'] * min(scale_x, scale_y)

            points = [self.transform_point(tuple(point[0])) for point in contour]
            if len(points) < 2:
                task_id = self.canvas.after(self.delay,
                                            lambda: draw_next_stroke_group(group_index + 1))
                self.animation_tasks.append(task_id)
                return

            start_idx = np.argmax([p[1] for p in points])
            points = np.roll(points, -start_idx, axis=0).tolist()

            current_line = []
            self.current_stroke = []

            def draw_next_point(point_index=0):
                if point_index >= len(points) or not self.animation_running:
                    if current_line and self.animation_running:
                        self.drawn_elements.extend(current_line)
                    if self.animation_running:
                        task_id = self.canvas.after(self.delay,
                                                    lambda: draw_next_stroke_group(group_index + 1))
                        self.animation_tasks.append(task_id)
                    return

                current_point = points[point_index]
                self.current_stroke.append(current_point)
                self.create_tracing_dot(current_point[0], current_point[1])

                if len(self.current_stroke) > 1:
                    line = self.canvas.create_line(
                        self.current_stroke[-2][0], self.current_stroke[-2][1],
                        current_point[0], current_point[1],
                        fill="black",
                        width=thickness,
                        capstyle=tk.ROUND,
                        joinstyle=tk.ROUND
                    )
                    current_line.append(line)

                next_index = point_index + self.step_size
                if next_index >= len(points):
                    next_index = len(points)

                task_id = self.canvas.after(self.delay,
                                            lambda: draw_next_point(next_index))
                self.animation_tasks.append(task_id)

            draw_next_point()

        draw_next_stroke_group()

    def cancel_animation(self):
        self.animation_running = False
        for task_id in self.animation_tasks:
            self.canvas.after_cancel(task_id)
        self.animation_tasks = []

    def clear(self):
        self.cancel_animation()
        self.canvas.delete("all")
        self.drawn_elements = []
        self.tracing_dot = None
        self.canvas.config(bg="white")

#
# class ProfessionalImageSelector:
#     def __init__(self, parent_frame, examples_dir, on_image_selected):
#         self.parent_frame = parent_frame
#         self.examples_dir = examples_dir
#         self.on_image_selected = on_image_selected
#         self.current_index = 0
#
#         # Get list of valid image files
#         self.image_files = [f for f in os.listdir(examples_dir)
#                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
#
#         self.setup_ui()
#
#     def setup_ui(self):
#         # Create main container with padding
#         self.container = ttk.Frame(self.parent_frame, padding="10")
#         self.container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
#
#         # Style configuration
#         style = ttk.Style()
#         style.configure("Gallery.TFrame", background="white")
#
#         # Image display area
#         self.display_frame = ttk.Frame(self.container, style="Gallery.TFrame")
#         self.display_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
#
#         # Canvas for image display
#         self.canvas = tk.Canvas(self.display_frame, bg="white", highlightthickness=0)
#         self.canvas.pack(fill=tk.BOTH, expand=True)
#
#         # Navigation controls
#         self.controls_frame = ttk.Frame(self.container)
#         self.controls_frame.pack(fill=tk.X, pady=(0, 10))
#
#         # Navigation buttons with modern styling
#         btn_style = {"width": 10, "padding": 5}
#         self.prev_btn = ttk.Button(self.controls_frame, text="← Previous",
#                                    command=self.prev_image, **btn_style)
#         self.prev_btn.pack(side=tk.LEFT, padx=5)
#
#         self.select_btn = ttk.Button(self.controls_frame, text="Select",
#                                      command=self.select_image, **btn_style)
#         self.select_btn.pack(side=tk.LEFT, padx=5)
#
#         self.next_btn = ttk.Button(self.controls_frame, text="Next →",
#                                    command=self.next_image, **btn_style)
#         self.next_btn.pack(side=tk.LEFT, padx=5)
#
#         # Image counter label
#         self.counter_label = ttk.Label(self.controls_frame,
#                                        text="Image 1 of {}".format(len(self.image_files)))
#         self.counter_label.pack(side=tk.RIGHT, padx=5)
#
#         self.show_current_image()
#
#     def show_current_image(self):
#         if not self.image_files:
#             return
#
#         image_path = os.path.join(self.examples_dir, self.image_files[self.current_index])
#         image = Image.open(image_path)
#
#         # Calculate scaled dimensions while maintaining aspect ratio
#         canvas_width = self.canvas.winfo_width()
#         canvas_height = self.canvas.winfo_height()
#
#         if canvas_width < 10:  # Initial load
#             canvas_width = 400
#             canvas_height = 400
#
#         img_width, img_height = image.size
#         scale = min(canvas_width / img_width, canvas_height / img_height)
#         new_width = int(img_width * scale)
#         new_height = int(img_height * scale)
#
#         image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
#         self.photo = ImageTk.PhotoImage(image)
#
#         self.canvas.delete("all")
#         x = (canvas_width - new_width) // 2
#         y = (canvas_height - new_height) // 2
#         self.canvas.create_image(x + new_width // 2, y + new_height // 2, image=self.photo)
#
#         # Update counter
#         self.counter_label.config(
#             text=f"Image {self.current_index + 1} of {len(self.image_files)}")
#
#     def prev_image(self):
#         if self.image_files:
#             self.current_index = (self.current_index - 1) % len(self.image_files)
#             self.show_current_image()
#
#     def next_image(self):
#         if self.image_files:
#             self.current_index = (self.current_index + 1) % len(self.image_files)
#             self.show_current_image()
#
#     def select_image(self):
#         if self.image_files:
#             selected_path = os.path.join(self.examples_dir,
#                                          self.image_files[self.current_index])
#             self.on_image_selected(selected_path)


class ProfessionalImageSelector:
    def __init__(self, parent_frame, examples_dir, on_image_selected):
        self.parent_frame = parent_frame
        self.examples_dir = examples_dir
        self.on_image_selected = on_image_selected
        self.current_index = 0

        # Get list of valid image files
        self.image_files = [f for f in os.listdir(examples_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        self.setup_ui()

    def setup_ui(self):
        # Create main container with padding
        self.container = ttk.Frame(self.parent_frame, padding="10")
        self.container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Style configuration
        style = ttk.Style()
        style.configure("Gallery.TFrame", background="white")

        # Image display area
        self.display_frame = ttk.Frame(self.container, style="Gallery.TFrame")
        self.display_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Canvas for image display
        self.canvas = tk.Canvas(self.display_frame, bg="white", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Navigation controls
        self.controls_frame = ttk.Frame(self.container)
        self.controls_frame.pack(fill=tk.X, pady=(0, 10))

        # Navigation buttons with modern styling
        btn_style = {"width": 10, "padding": 5}
        self.prev_btn = ttk.Button(self.controls_frame, text="← Previous",
                                   command=self.prev_image, **btn_style)
        self.prev_btn.pack(side=tk.LEFT, padx=5)

        self.select_btn = ttk.Button(self.controls_frame, text="Select",
                                     command=self.select_image, **btn_style)
        self.select_btn.pack(side=tk.LEFT, padx=5)

        self.next_btn = ttk.Button(self.controls_frame, text="Next →",
                                   command=self.next_image, **btn_style)
        self.next_btn.pack(side=tk.LEFT, padx=5)

        # Image counter label
        self.counter_label = ttk.Label(self.controls_frame,
                                       text="Image 1 of {}".format(len(self.image_files)))
        self.counter_label.pack(side=tk.RIGHT, padx=5)

        self.show_current_image()

    def show_current_image(self):
        if not self.image_files:
            return

        image_path = os.path.join(self.examples_dir, self.image_files[self.current_index])
        image = Image.open(image_path)

        # Calculate scaled dimensions while maintaining aspect ratio
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width < 10:  # Initial load
            canvas_width = 400
            canvas_height = 400

        img_width, img_height = image.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)

        self.canvas.delete("all")
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        self.canvas.create_image(x + new_width // 2, y + new_height // 2, image=self.photo)

        # Update counter
        self.counter_label.config(
            text=f"Image {self.current_index + 1} of {len(self.image_files)}")

    def prev_image(self):
        if self.image_files:
            self.current_index = (self.current_index - 1) % len(self.image_files)
            self.show_current_image()

    def next_image(self):
        if self.image_files:
            self.current_index = (self.current_index + 1) % len(self.image_files)
            self.show_current_image()

    def select_image(self):
        if self.image_files:
            selected_path = os.path.join(self.examples_dir,
                                         self.image_files[self.current_index])
            self.on_image_selected(selected_path)


class ProfessionalAnimatorApp:
    def __init__(self, root, examples_dir):
        self.root = root
        self.root.title("Image Animation Studio")
        self.examples_dir = examples_dir

        # Configure root window
        self.root.geometry("1400x800")
        self.root.minsize(1200, 700)

        # Configure styles
        self.setup_styles()

        # Create main container
        self.main_container = ttk.Frame(root, padding="10")
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # Create left panel for image selection/display
        self.left_panel = ttk.Frame(self.main_container)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create right panel for controls and animation
        self.right_panel = ttk.Frame(self.main_container)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        # Initialize components
        self.setup_left_panel()
        self.setup_right_panel()
        self.setup_status_bar()

        # Initialize variables
        self.animator = None
        self.processed_image = None
        self.stroke_groups = None

    def setup_styles(self):
        style = ttk.Style()
        style.configure("Action.TButton", padding=10, font=("Helvetica", 10, "bold"))
        style.configure("Status.TLabel", padding=5, background="#f0f0f0")

    def setup_left_panel(self):
        # Create image selector
        self.image_selector = ProfessionalImageSelector(
            self.left_panel,
            self.examples_dir,
            self.on_image_selected
        )

    def setup_right_panel(self):
        # Animation canvas
        self.canvas_frame = ttk.Frame(self.right_panel)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.canvas = tk.Canvas(self.canvas_frame, bg="white", highlightthickness=1,
                                highlightbackground="#cccccc")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Control buttons
        self.controls_frame = ttk.Frame(self.right_panel)
        self.controls_frame.pack(fill=tk.X, pady=(0, 10))

        # Process button
        self.process_btn = ttk.Button(self.controls_frame, text="Process Image",
                                      style="Action.TButton", state=tk.DISABLED,
                                      command=self.process_image)
        self.process_btn.pack(side=tk.LEFT, padx=5)

        # Animation controls
        ttk.Button(self.controls_frame, text="Fast Draw",
                   style="Action.TButton", command=self.draw_fast).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.controls_frame, text="Animated Draw",
                   style="Action.TButton", command=self.draw_animated).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.controls_frame, text="Clear",
                   style="Action.TButton", command=self.clear_canvas).pack(side=tk.LEFT, padx=5)

    def setup_status_bar(self):
        self.status_var = tk.StringVar(value="Select an image to begin")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var,
                                    style="Status.TLabel", anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def on_image_selected(self, image_path):
        # Clear the canvas and reset animator when a new image is selected
        self.clear_canvas()

        # Reset the animator and processed image
        self.animator = None
        self.processed_image = None
        self.stroke_groups = None

        # Update the image path and enable processing
        self.image_path = image_path
        self.process_btn.config(state=tk.NORMAL)
        self.update_status(f"Selected image: {os.path.basename(image_path)}")

    def process_image(self):
        self.update_status("Processing image...")
        self.root.update()

        try:
            output_path = process_single_image(self.image_path)
            self.update_status(f"Processing complete. Output at: {output_path}")

            self.processed_image = cv2.imread(output_path, 0)
            if self.processed_image is None:
                self.update_status(f"Error: Could not load processed image from {output_path}")
                return

            processed = cv2.medianBlur(self.processed_image, 3)
            processed = cv2.Laplacian(processed, cv2.CV_8U)
            processed = cv2.bitwise_not(processed)

            self.stroke_groups = get_connected_strokes(processed, fixed_thickness=1.5)
            num_strokes = len(self.stroke_groups)
            self.update_status(f"Detected {num_strokes} stroke groups")

            original_height, original_width = self.processed_image.shape
            self.animator = DrawingAnimator(
                self.canvas,
                self.stroke_groups,
                original_width=original_width,
                original_height=original_height,
                delay=15
            )

            self.update_status(f"Ready to animate {num_strokes} strokes. Choose drawing mode.")

        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

    def draw_fast(self):
        if self.animator:
            self.update_status("Drawing in fast mode...")
            self.animator.draw_strokes_fast()
            self.update_status("Fast drawing complete")
        else:
            self.update_status("No image processed yet. Press 'Process Image' first.")

    def draw_animated(self):
        if self.animator:
            self.update_status("Drawing in animated mode...")
            self.animator.draw_strokes_animated()
        else:
            self.update_status("No image processed yet. Press 'Process Image' first.")

    def clear_canvas(self):
        if self.animator:
            self.animator.clear()
            self.update_status("Canvas cleared")
        else:
            self.canvas.delete("all")
            self.update_status("Canvas cleared")

    def update_status(self, message):
        self.status_var.set(message)
        self.root.update()


def main(examples_dir):
    root = tk.Tk()
    app = ProfessionalAnimatorApp(root, examples_dir)
    root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('examples_dir', help='Directory containing example images')
    parser.add_argument('--output_dir', default='output', help='Output directory for processed images')
    parser.add_argument('--checkpoints_dir', default='checkpoints', help='Directory for model checkpoints')
    parser.add_argument('--results_dir', default='output', help='Directory for results')

    args = parser.parse_args()

    if not os.path.exists(args.examples_dir):
        print(f"Error: Examples directory not found: {args.examples_dir}")
        sys.exit(1)

    # Ensure output directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # Set required environment variables for the TestOptions parser
    os.environ['TEST_DATAROOT'] = args.examples_dir
    os.environ['TEST_RESULTS_DIR'] = args.results_dir
    os.environ['TEST_CHECKPOINTS_DIR'] = args.checkpoints_dir

    print(f"Examples directory: {args.examples_dir}")
    print(f"Output directory: {args.output_dir}")

    main(args.examples_dir)