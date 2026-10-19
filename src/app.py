"""
Graphical User Interface for Spider-Man AR Studio.
Built with CustomTkinter for a sleek, modern dark-themed studio experience.
Features:
- Live high-framerate camera preview
- Visual filter gallery with generated thumbnails & category filtering
- Keyboard shortcuts (Left/Right arrow, Space, R, 0, F5)
- Real-time photo capture with visual shutter feedback
- Thread-safe video recording with animated REC indicator
- Adjustable filter opacity and landmark tracking visualization
- Custom output folder selector
"""

import os
import time
import math
import logging
from pathlib import Path
from typing import Optional, Dict, List
import cv2
import numpy as np
from PIL import Image, ImageTk

import customtkinter as ctk
from tkinter import filedialog, messagebox

from .camera import Camera
from .face_tracker import FaceTracker, FaceData
from .filter_manager import FilterManager, FilterItem
from .recorder import PhotoSaver, VideoRecorder

logger = logging.getLogger(__name__)

# Configure CustomTkinter theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class ARStudioApp:
    """Main desktop application window and event controller."""

    def __init__(self, camera_index: int = 0, output_dir: str = "output") -> None:
        self.output_dir = str(Path(output_dir).resolve())

        # Initialize core computer vision engines
        self.camera = Camera(device_index=camera_index, width=1280, height=720)
        self.tracker = FaceTracker(smoothing_alpha=0.72, enable_smoothing=True)
        self.filter_manager = FilterManager(
            filters_dir="assets/filters",
            thumbnails_dir="assets/thumbnails",
            config_path="config/filters.json"
        )
        self.recorder = VideoRecorder(output_dir=self.output_dir)

        # UI state
        self.show_landmarks = False
        self.filter_opacity = 1.0
        self.is_camera_running = False
        self.shutter_flash_frames = 0
        self.thumbnail_images: Dict[str, ctk.CTkImage] = {}
        self.filter_buttons: Dict[str, ctk.CTkButton] = {}

        # Setup GUI Window
        self.root = ctk.CTk()
        self.root.title("Spider-Man AR Studio - Real-Time Face Filter Engine")
        self.root.geometry("1340x880")
        self.root.minsize(1080, 720)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Build interface layout
        self._build_ui()
        self._bind_shortcuts()

        # Start Camera
        self.start_camera()

        # Start video update loop
        self.update_video_loop()

    def _build_ui(self) -> None:
        """Constructs the desktop layout."""
        # Main Grid Layout: Left preview (weight=3), Right control sidebar (weight=1)
        self.root.grid_columnconfigure(0, weight=3)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # --- LEFT PANEL: VIDEO PREVIEW CANVAS & STATUS BAR ---
        self.left_panel = ctk.CTkFrame(self.root, corner_radius=12, fg_color="#10121a")
        self.left_panel.grid(row=0, column=0, padx=(16, 8), pady=16, sticky="nsew")
        self.left_panel.grid_rowconfigure(0, weight=0)  # Top HUD
        self.left_panel.grid_rowconfigure(1, weight=1)  # Video
        self.left_panel.grid_rowconfigure(2, weight=0)  # Status footer
        self.left_panel.grid_columnconfigure(0, weight=1)

        # Top HUD Banner (Title & Badges)
        self.top_hud = ctk.CTkFrame(self.left_panel, fg_color="#181b26", corner_radius=8, height=48)
        self.top_hud.grid(row=0, column=0, padx=12, pady=(12, 6), sticky="ew")

        self.title_label = ctk.CTkLabel(
            self.top_hud,
            text="🕷️ SPIDER-MAN AR STUDIO",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
            text_color="#e62429"
        )
        self.title_label.pack(side="left", padx=16, pady=8)

        self.fps_badge = ctk.CTkLabel(
            self.top_hud,
            text="0.0 FPS",
            font=ctk.CTkFont(family="Consolas", size=13, weight="bold"),
            text_color="#00e5ff",
            fg_color="#0e2333",
            corner_radius=6,
            width=80
        )
        self.fps_badge.pack(side="right", padx=(6, 16), pady=8)

        self.tracking_badge = ctk.CTkLabel(
            self.top_hud,
            text="WAITING FOR FACE",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color="#ffb300",
            fg_color="#332400",
            corner_radius=6,
            width=160
        )
        self.tracking_badge.pack(side="right", padx=6, pady=8)

        self.rec_badge = ctk.CTkLabel(
            self.top_hud,
            text="● REC 00:00",
            font=ctk.CTkFont(family="Consolas", size=13, weight="bold"),
            text_color="#ffffff",
            fg_color="#d32f2f",
            corner_radius=6,
            width=110
        )
        # Not packed initially, packed when recording

        # Video Canvas Container
        self.video_frame = ctk.CTkFrame(self.left_panel, fg_color="#08090d", corner_radius=10)
        self.video_frame.grid(row=1, column=0, padx=12, pady=6, sticky="nsew")
        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)

        self.video_canvas = ctk.CTkLabel(self.video_frame, text="", fg_color="transparent")
        self.video_canvas.grid(row=0, column=0, sticky="nsew")

        # Bottom Status Notification Footer
        self.status_footer = ctk.CTkFrame(self.left_panel, fg_color="#181b26", corner_radius=8, height=36)
        self.status_footer.grid(row=2, column=0, padx=12, pady=(6, 12), sticky="ew")

        self.status_label = ctk.CTkLabel(
            self.status_footer,
            text="Ready. Select a filter or use Left/Right arrows to explore.",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color="#a0aab8"
        )
        self.status_label.pack(side="left", padx=14, pady=6)

        # --- RIGHT PANEL: FILTER GALLERY & STUDIO CONTROLS ---
        self.right_panel = ctk.CTkFrame(self.root, corner_radius=12, fg_color="#161822")
        self.right_panel.grid(row=0, column=1, padx=(8, 16), pady=16, sticky="nsew")
        self.right_panel.grid_rowconfigure(2, weight=1)  # Filter gallery takes remaining height
        self.right_panel.grid_columnconfigure(0, weight=1)

        # 1. Action Controls Section (Capture, Record, Navigation)
        self.actions_group = ctk.CTkFrame(self.right_panel, fg_color="#1f2230", corner_radius=10)
        self.actions_group.grid(row=0, column=0, padx=12, pady=(12, 8), sticky="ew")

        act_title = ctk.CTkLabel(
            self.actions_group,
            text="CAPTURE & CONTROLS",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color="#7b889b"
        )
        act_title.pack(anchor="w", padx=12, pady=(8, 4))

        # Photo & Record Buttons Row
        self.btn_row_1 = ctk.CTkFrame(self.actions_group, fg_color="transparent")
        self.btn_row_1.pack(fill="x", padx=10, pady=(2, 6))

        self.photo_btn = ctk.CTkButton(
            self.btn_row_1,
            text="📷 Take Photo",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            fg_color="#0288d1",
            hover_color="#0277bd",
            height=38,
            command=self.capture_photo
        )
        self.photo_btn.pack(side="left", expand=True, fill="x", padx=(0, 4))

        self.record_btn = ctk.CTkButton(
            self.btn_row_1,
            text="⏺ Record Video",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            fg_color="#c62828",
            hover_color="#b71c1c",
            height=38,
            command=self.toggle_recording
        )
        self.record_btn.pack(side="right", expand=True, fill="x", padx=(4, 0))

        # Prev / Next Navigation Row
        self.btn_row_2 = ctk.CTkFrame(self.actions_group, fg_color="transparent")
        self.btn_row_2.pack(fill="x", padx=10, pady=(0, 8))

        self.prev_btn = ctk.CTkButton(
            self.btn_row_2,
            text="◀ Previous",
            fg_color="#2b3044",
            hover_color="#39405a",
            height=32,
            command=self.previous_filter
        )
        self.prev_btn.pack(side="left", expand=True, fill="x", padx=(0, 4))

        self.reset_btn = ctk.CTkButton(
            self.btn_row_2,
            text="Reset",
            fg_color="#3d445c",
            hover_color="#4d5675",
            width=65,
            height=32,
            command=self.reset_filter
        )
        self.reset_btn.pack(side="left", padx=2)

        self.next_btn = ctk.CTkButton(
            self.btn_row_2,
            text="Next ▶",
            fg_color="#2b3044",
            hover_color="#39405a",
            height=32,
            command=self.next_filter
        )
        self.next_btn.pack(side="right", expand=True, fill="x", padx=(4, 0))

        # 2. Filter Adjustments (Opacity, Smoothing, Wireframe)
        self.adjust_group = ctk.CTkFrame(self.right_panel, fg_color="#1f2230", corner_radius=10)
        self.adjust_group.grid(row=1, column=0, padx=12, pady=6, sticky="ew")

        # Opacity Slider
        opacity_header = ctk.CTkFrame(self.adjust_group, fg_color="transparent")
        opacity_header.pack(fill="x", padx=12, pady=(8, 0))

        ctk.CTkLabel(
            opacity_header,
            text="Effect Opacity:",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color="#b0bccd"
        ).pack(side="left")

        self.opacity_val_label = ctk.CTkLabel(
            opacity_header,
            text="100%",
            font=ctk.CTkFont(family="Consolas", size=12, weight="bold"),
            text_color="#00e5ff"
        )
        self.opacity_val_label.pack(side="right")

        self.opacity_slider = ctk.CTkSlider(
            self.adjust_group,
            from_=0.1,
            to=1.0,
            number_of_steps=90,
            command=self._on_opacity_change
        )
        self.opacity_slider.set(1.0)
        self.opacity_slider.pack(fill="x", padx=12, pady=(2, 6))

        # Switches Row (Smoothing & Wireframe)
        switches_row = ctk.CTkFrame(self.adjust_group, fg_color="transparent")
        switches_row.pack(fill="x", padx=12, pady=(2, 8))

        self.smooth_switch = ctk.CTkSwitch(
            switches_row,
            text="Smooth Jitter",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            command=self._on_smooth_toggle
        )
        self.smooth_switch.select()
        self.smooth_switch.pack(side="left")

        self.mesh_switch = ctk.CTkSwitch(
            switches_row,
            text="Landmarks",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            command=self._on_mesh_toggle
        )
        self.mesh_switch.pack(side="right")

        # 3. Filter Gallery Section (Header + Scrollable Gallery)
        gallery_header = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        gallery_header.grid(row=2, column=0, padx=14, pady=(10, 4), sticky="new")

        ctk.CTkLabel(
            gallery_header,
            text="FILTER GALLERY",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color="#7b889b"
        ).pack(side="left")

        self.refresh_btn = ctk.CTkButton(
            gallery_header,
            text="🔄 Refresh",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            fg_color="#2b3044",
            hover_color="#3c4360",
            width=80,
            height=26,
            command=self.refresh_filters
        )
        self.refresh_btn.pack(side="right")

        # Scrollable Cards List
        self.gallery_scroll = ctk.CTkScrollableFrame(
            self.right_panel,
            fg_color="#10121a",
            corner_radius=8
        )
        self.gallery_scroll.grid(row=2, column=0, padx=12, pady=(40, 10), sticky="nsew")
        self.gallery_scroll.grid_columnconfigure(0, weight=1)

        # 4. Settings Footer (Output Directory & Exit)
        self.settings_footer = ctk.CTkFrame(self.right_panel, fg_color="#1f2230", corner_radius=10)
        self.settings_footer.grid(row=3, column=0, padx=12, pady=(0, 12), sticky="ew")

        out_header = ctk.CTkFrame(self.settings_footer, fg_color="transparent")
        out_header.pack(fill="x", padx=10, pady=(6, 2))

        ctk.CTkLabel(
            out_header,
            text="Output Folder:",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color="#8997a9"
        ).pack(side="left")

        self.browse_btn = ctk.CTkButton(
            out_header,
            text="Browse...",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            fg_color="#2b3044",
            hover_color="#39405a",
            width=65,
            height=22,
            command=self.browse_output_dir
        )
        self.browse_btn.pack(side="right")

        self.out_path_label = ctk.CTkLabel(
            self.settings_footer,
            text=f".../{Path(self.output_dir).name}",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color="#5f6d80",
            anchor="w"
        )
        self.out_path_label.pack(fill="x", padx=10, pady=(0, 6))

        # Exit Button
        self.exit_btn = ctk.CTkButton(
            self.settings_footer,
            text="Exit Studio",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            fg_color="#2a2228",
            hover_color="#5c1d27",
            height=30,
            command=self.on_closing
        )
        self.exit_btn.pack(fill="x", padx=10, pady=(0, 8))

        # Populate filter cards into the gallery
        self.populate_gallery()

    def _bind_shortcuts(self) -> None:
        """Configures keyboard shortcuts."""
        self.root.bind("<Left>", lambda e: self.previous_filter())
        self.root.bind("<Right>", lambda e: self.next_filter())
        self.root.bind("<space>", lambda e: self.capture_photo())
        self.root.bind("<r>", lambda e: self.toggle_recording())
        self.root.bind("<R>", lambda e: self.toggle_recording())
        self.root.bind("0", lambda e: self.reset_filter())
        self.root.bind("<F5>", lambda e: self.refresh_filters())
        self.root.bind("<Escape>", lambda e: self.on_closing())

    def populate_gallery(self) -> None:
        """Builds interactive filter cards inside the scrollable gallery."""
        # Clear existing card widgets
        for widget in self.gallery_scroll.winfo_children():
            widget.destroy()

        self.thumbnail_images.clear()
        self.filter_buttons.clear()

        current_item = self.filter_manager.get_current_filter()

        for idx, item in enumerate(self.filter_manager.filters):
            # Load thumbnail into CTkImage
            ctk_thumb = None
            if item.thumbnail_path and os.path.exists(item.thumbnail_path):
                try:
                    pil_img = Image.open(item.thumbnail_path)
                    ctk_thumb = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(48, 48))
                    self.thumbnail_images[item.filter_id] = ctk_thumb
                except Exception as e:
                    logger.warning(f"Error loading thumb {item.thumbnail_path}: {e}")

            # Highlight if currently active
            is_active = (current_item and current_item.filter_id == item.filter_id)
            bg_col = "#20304a" if is_active else "#161922"
            border_col = "#00d2ff" if is_active else "#282d3d"

            card = ctk.CTkFrame(
                self.gallery_scroll,
                fg_color=bg_col,
                border_color=border_col,
                border_width=2 if is_active else 1,
                corner_radius=8,
                height=56
            )
            card.pack(fill="x", padx=4, pady=3)

            # Clickable button covering the card
            btn = ctk.CTkButton(
                card,
                text=f"  {item.display_name}",
                image=ctk_thumb,
                compound="left",
                anchor="w",
                font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold" if is_active else "normal"),
                fg_color="transparent",
                hover_color="#242b3b",
                height=52,
                command=lambda f_idx=idx: self.select_filter_index(f_idx)
            )
            btn.pack(fill="both", expand=True, padx=4, pady=2)
            self.filter_buttons[item.filter_id] = btn

    def select_filter_index(self, index: int) -> None:
        """Selects a filter by index and updates GUI highlights."""
        if self.filter_manager.select_filter(index):
            current = self.filter_manager.get_current_filter()
            if current:
                self.status_label.configure(
                    text=f"Active Filter: {current.display_name} ({current.category})"
                )
            self.populate_gallery()

    def next_filter(self) -> None:
        """Switches to the next filter."""
        item = self.filter_manager.next_filter()
        if item:
            self.status_label.configure(text=f"Active Filter: {item.display_name}")
            self.populate_gallery()

    def previous_filter(self) -> None:
        """Switches to the previous filter."""
        item = self.filter_manager.prev_filter()
        if item:
            self.status_label.configure(text=f"Active Filter: {item.display_name}")
            self.populate_gallery()

    def reset_filter(self) -> None:
        """Resets to 'No Filter'."""
        self.filter_manager.select_no_filter()
        self.status_label.configure(text="Filter reset to original webcam feed.")
        self.populate_gallery()

    def refresh_filters(self) -> None:
        """Dynamically scans assets/filters/ for newly added images."""
        self.filter_manager.reload()
        self.populate_gallery()
        self.status_label.configure(
            text=f"Filters refreshed! Total available: {len(self.filter_manager.filters)}"
        )

    def _on_opacity_change(self, value: float) -> None:
        """Updates filter opacity from slider."""
        self.filter_opacity = float(value)
        self.opacity_val_label.configure(text=f"{int(round(value * 100))}%")

    def _on_smooth_toggle(self) -> None:
        """Toggles EMA landmark jitter reduction."""
        self.tracker.enable_smoothing = bool(self.smooth_switch.get())

    def _on_mesh_toggle(self) -> None:
        """Toggles landmark wireframe overlay."""
        self.show_landmarks = bool(self.mesh_switch.get())

    def browse_output_dir(self) -> None:
        """Opens directory browser dialog."""
        folder = filedialog.askdirectory(initialdir=self.output_dir)
        if folder:
            self.output_dir = folder
            self.out_path_label.configure(text=f".../{Path(folder).name}")
            self.status_label.configure(text=f"Output folder updated: {folder}")

    def capture_photo(self) -> None:
        """Saves current processed frame to output folder."""
        if not hasattr(self, "latest_processed_frame") or self.latest_processed_frame is None:
            self.status_label.configure(text="Capture failed: no frame available.")
            return

        success, path_or_err = PhotoSaver.save(self.latest_processed_frame, self.output_dir)
        if success:
            self.shutter_flash_frames = 3  # Trigger white shutter flash feedback
            fname = Path(path_or_err).name
            self.status_label.configure(text=f"📸 Saved snapshot: {fname}")
        else:
            self.status_label.configure(text=f"Photo save error: {path_or_err}")

    def toggle_recording(self) -> None:
        """Starts or stops video recording."""
        if not self.recorder.is_recording:
            # Start Recording
            if not hasattr(self, "latest_processed_frame") or self.latest_processed_frame is None:
                self.status_label.configure(text="Cannot record: no camera feed.")
                return

            h, w = self.latest_processed_frame.shape[:2]
            fps = self.camera.get_fps()
            if fps < 10:
                fps = 30.0

            success, path_or_err = self.recorder.start(w, h, fps=fps, output_dir=self.output_dir)
            if success:
                self.record_btn.configure(
                    text="⏹ Stop Recording",
                    fg_color="#d32f2f",
                    hover_color="#b71c1c"
                )
                self.rec_badge.pack(side="right", padx=6, pady=8)
                self.status_label.configure(text=f"Recording video to {Path(path_or_err).name}...")
            else:
                self.status_label.configure(text=f"Recording error: {path_or_err}")
        else:
            # Stop Recording
            success, path_or_err, frames, duration = self.recorder.stop()
            self.record_btn.configure(
                text="⏺ Record Video",
                fg_color="#c62828",
                hover_color="#b71c1c"
            )
            self.rec_badge.pack_forget()

            if success:
                fname = Path(path_or_err).name
                self.status_label.configure(
                    text=f"🎬 Video saved: {fname} ({frames} frames, {duration:.1f}s)"
                )
            else:
                self.status_label.configure(text=f"Stop recording error: {path_or_err}")

    def start_camera(self) -> None:
        """Opens camera hardware."""
        self.is_camera_running = self.camera.start()
        if not self.is_camera_running:
            self.status_label.configure(
                text="CAMERA ERROR: Could not open webcam. Ensure camera is plugged in and accessible."
            )
            messagebox.showerror(
                "Camera Error",
                "Could not connect to webcam.\nPlease check camera privacy permissions in Windows Settings."
            )

    def update_video_loop(self) -> None:
        """Main rendering loop executed on Tkinter timer."""
        if self.is_camera_running:
            ret, frame = self.camera.read_frame()
            if ret and frame is not None:
                # Mirror webcam horizontally for natural mirror behavior
                frame = cv2.flip(frame, 1)

                # Process face tracking
                face_data: FaceData = self.tracker.process(frame)

                # Update HUD Tracking Badge
                if face_data.detected:
                    self.tracking_badge.configure(
                        text=f"TRACKING FACE ({face_data.roll_deg:.0f}°)",
                        text_color="#00e676",
                        fg_color="#0e331c"
                    )
                else:
                    self.tracking_badge.configure(
                        text="SEARCHING FOR FACE",
                        text_color="#ffb300",
                        fg_color="#332400"
                    )

                # Apply active filter
                processed = self.filter_manager.apply_current_filter(
                    frame.copy(),
                    face_data,
                    opacity_override=self.filter_opacity
                )

                # Optional landmark debug overlay
                if self.show_landmarks and face_data.detected:
                    processed = self.tracker.draw_debug_landmarks(processed, face_data)

                # Shutter flash effect
                if self.shutter_flash_frames > 0:
                    white_layer = np.full_like(processed, 255)
                    processed = cv2.addWeighted(processed, 0.4, white_layer, 0.6, 0)
                    self.shutter_flash_frames -= 1

                # Save current processed frame for photo capture
                self.latest_processed_frame = processed.copy()

                # Feed frame to video writer if recording
                if self.recorder.is_recording:
                    self.recorder.write_frame(processed)
                    rec_sec = int(self.recorder.get_duration())
                    mins = rec_sec // 60
                    secs = rec_sec % 60
                    self.rec_badge.configure(text=f"● REC {mins:02d}:{secs:02d}")

                # Update FPS badge
                fps = self.camera.get_fps()
                self.fps_badge.configure(text=f"{fps:.1f} FPS")

                # Convert BGR frame to PIL Image for Tkinter
                self._render_to_canvas(processed)

        # Reschedule next frame (targeting ~30-60 FPS)
        self.root.after(16, self.update_video_loop)

    def _render_to_canvas(self, frame_bgr: np.ndarray) -> None:
        """Scales and draws BGR frame onto the CTkLabel canvas with aspect-ratio preservation."""
        try:
            canvas_w = max(100, self.video_frame.winfo_width())
            canvas_h = max(100, self.video_frame.winfo_height())

            img_h, img_w = frame_bgr.shape[:2]
            scale = min(canvas_w / float(img_w), canvas_h / float(img_h))
            disp_w = max(1, int(img_w * scale))
            disp_h = max(1, int(img_h * scale))

            # Resize frame for canvas
            resized = cv2.resize(frame_bgr, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            pil_img = Image.fromarray(rgb)
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(disp_w, disp_h))

            self.video_canvas.configure(image=ctk_img)
            self.video_canvas.image = ctk_img  # Keep reference to avoid GC
        except Exception as e:
            logger.warning(f"Error rendering frame to canvas: {e}")

    def on_closing(self) -> None:
        """Gracefully releases resources and exits."""
        logger.info("Closing application...")
        if self.recorder.is_recording:
            self.recorder.stop()

        self.camera.release()
        self.tracker.release()
        self.root.destroy()

    def run(self) -> None:
        """Starts the Tkinter event loop."""
        self.root.mainloop()
