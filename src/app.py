"""
Immersive Full-Screen AR Studio Application.
Provides a clean, edge-to-edge camera experience where hands directly
control the visual transformation without dashboard clutter.

Modes:
- MODE A: Local Effect (Only pixels inside hand polygon transform)
- MODE B: Scene Portal (Virtual scene appears perspective-warped inside hand polygon)
- MODE C: Character Transformation (Virtual background + segmented user + face mask / costume overlay)
- MODE D: Combined AR (Full background + character transformation + hand-controlled portal/energy zone)
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
from tkinter import messagebox

from .camera import Camera
from .hand_tracker import HandTracker, DualHandData
from .gesture_recognizer import GestureRecognizer, GestureResult
from .region_controller import RegionController
from .mask_renderer import MaskRenderer
from .effect_manager import EffectManager
from .person_segmenter import PersonSegmenter
from .background_compositor import BackgroundCompositor
from .character_compositor import CharacterCompositor
from .recorder import PhotoSaver, VideoRecorder

logger = logging.getLogger(__name__)


class ARMode:
    MODE_A = "MODE A: LOCAL EFFECT"
    MODE_B = "MODE B: SCENE PORTAL"
    MODE_C = "MODE C: CHARACTER TRANSFORMATION"
    MODE_D = "MODE D: COMBINED AR"

    ALL = [MODE_A, MODE_B, MODE_C, MODE_D]


class ImmersiveARStudioApp:
    """Full-screen immersive AR studio window with gesture and hand control."""

    def __init__(self, camera_index: int = 0, output_dir: str = "output") -> None:
        self.output_dir = str(Path(output_dir).resolve())

        # Core Computer Vision Pipeline (initialized once)
        self.camera = Camera(device_index=camera_index, width=640, height=480, fps=30)
        self.hand_tracker = HandTracker(max_num_hands=2, base_smoothing_alpha=0.78)
        self.gesture_recognizer = GestureRecognizer()
        self.region_controller = RegionController(feather_radius=21, smoothing_alpha=0.82)
        self.effect_manager = EffectManager(effects_dir="assets/effects", textures_dir="assets/textures")
        self.person_segmenter = PersonSegmenter(model_selection=0)
        self.background_compositor = BackgroundCompositor(backgrounds_dir="assets/backgrounds")
        self.character_compositor = CharacterCompositor(characters_dir="assets/characters", filters_dir="assets/filters")
        self.recorder = VideoRecorder(output_dir=self.output_dir)

        # State
        self.current_mode_idx = 0
        self.is_debug_mode = False
        self.is_fullscreen = False
        self.shutter_flash_frames = 0
        self.is_camera_running = False
        self.latest_frame: Optional[np.ndarray] = None

        # Floating toast notification
        self.toast_message: str = "Show both hands to begin"
        self.toast_expires_at: float = time.time() + 3.5

        # Build clean full-screen compatible window
        self.root = ctk.CTk()
        self.root.title("Spider-Man AR Studio - Gesture-Controlled Augmented Reality")
        self.root.geometry("1280x720")
        self.root.minsize(800, 600)
        self.root.configure(fg_color="#000000")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Full-window video canvas
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.canvas_label = ctk.CTkLabel(self.root, text="", fg_color="#000000")
        self.canvas_label.grid(row=0, column=0, sticky="nsew")

        self._bind_shortcuts()
        self.start_camera()
        self.update_loop()

    def _bind_shortcuts(self) -> None:
        """Configures clean keyboard shortcuts."""
        self.root.bind("<Escape>", lambda e: self._on_escape())
        self.root.bind("<F11>", lambda e: self.toggle_fullscreen())
        self.root.bind("<f>", lambda e: self.toggle_fullscreen())
        self.root.bind("<F>", lambda e: self.toggle_fullscreen())
        self.root.bind("<d>", lambda e: self.toggle_debug())
        self.root.bind("<D>", lambda e: self.toggle_debug())
        self.root.bind("<m>", lambda e: self.cycle_mode())
        self.root.bind("<M>", lambda e: self.cycle_mode())
        self.root.bind("1", lambda e: self.set_mode(0))
        self.root.bind("2", lambda e: self.set_mode(1))
        self.root.bind("3", lambda e: self.set_mode(2))
        self.root.bind("4", lambda e: self.set_mode(3))
        self.root.bind("<Right>", lambda e: self.next_effect())
        self.root.bind("<Left>", lambda e: self.prev_effect())
        self.root.bind("<n>", lambda e: self.next_effect())
        self.root.bind("<N>", lambda e: self.next_effect())
        self.root.bind("<p>", lambda e: self.prev_effect())
        self.root.bind("<P>", lambda e: self.prev_effect())
        self.root.bind("<b>", lambda e: self.next_background())
        self.root.bind("<B>", lambda e: self.next_background())
        self.root.bind("<c>", lambda e: self.next_character())
        self.root.bind("<C>", lambda e: self.next_character())
        self.root.bind("<space>", lambda e: self.capture_photo())
        self.root.bind("<r>", lambda e: self.toggle_recording())
        self.root.bind("<R>", lambda e: self.toggle_recording())
        self.root.bind("<l>", lambda e: self.toggle_lock())
        self.root.bind("<L>", lambda e: self.toggle_lock())
        self.root.bind("<F5>", lambda e: self.reload_assets())

    def start_camera(self) -> None:
        self.is_camera_running = self.camera.start()
        if not self.is_camera_running:
            self.set_toast("CAMERA ERROR: Cannot connect to webcam", duration=8.0)
            messagebox.showerror(
                "Camera Error",
                "Could not connect to webcam. Please verify camera permissions in Windows Settings."
            )

    def set_toast(self, message: str, duration: float = 2.0) -> None:
        """Sets a temporary floating toast message."""
        self.toast_message = message
        self.toast_expires_at = time.time() + duration

    def set_mode(self, mode_idx: int) -> None:
        if 0 <= mode_idx < len(ARMode.ALL):
            self.current_mode_idx = mode_idx
            mode = ARMode.ALL[self.current_mode_idx]
            self.set_toast(f"Switched: {mode}", duration=2.2)

    def cycle_mode(self) -> None:
        self.current_mode_idx = (self.current_mode_idx + 1) % len(ARMode.ALL)
        mode = ARMode.ALL[self.current_mode_idx]
        self.set_toast(f"Switched: {mode}", duration=2.2)

    def next_effect(self) -> None:
        eff = self.effect_manager.next_effect()
        if eff:
            self.set_toast(f"Effect: {eff.display_name}")

    def prev_effect(self) -> None:
        eff = self.effect_manager.prev_effect()
        if eff:
            self.set_toast(f"Effect: {eff.display_name}")

    def next_background(self) -> None:
        bg = self.background_compositor.next_background()
        self.set_toast(f"Scene: {bg}")

    def next_character(self) -> None:
        char = self.character_compositor.next_character()
        self.set_toast(f"Character Overlay: {char}")

    def toggle_lock(self) -> None:
        locked = self.region_controller.toggle_freeze()
        self.set_toast("🔒 Region LOCKED in place" if locked else "🔓 Region LIVE: Following hands")

    def toggle_debug(self) -> None:
        self.is_debug_mode = not self.is_debug_mode
        self.set_toast("Debug Mode ON (Skeletons & Telemetry)" if self.is_debug_mode else "Debug Mode OFF (Clean AR)")

    def toggle_fullscreen(self) -> None:
        self.is_fullscreen = not self.is_fullscreen
        self.root.attributes("-fullscreen", self.is_fullscreen)
        self.set_toast("Fullscreen Enabled (F11/F or Esc to exit)" if self.is_fullscreen else "Windowed Mode")

    def _on_escape(self) -> None:
        if self.is_fullscreen:
            self.toggle_fullscreen()
        else:
            self.on_closing()

    def update_loop(self) -> None:
        """Core real-time frame processing and compositing loop."""
        if self.is_camera_running:
            ret, frame = self.camera.read_frame()
            if ret and frame is not None:
                # Mirror webcam horizontally for intuitive mirror view
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                # 1. Track Hands continuously (in mirrored camera coordinates)
                dual: DualHandData = self.hand_tracker.process(frame)

                # 2. Recognize Gestures & Dispatch Hands-Free Actions
                g_res: GestureResult = self.gesture_recognizer.process(dual)
                if g_res.action_triggered:
                    self._dispatch_gesture_action(g_res.action_triggered)

                # 3. Update Hand-Controlled Region Polygon continuously from fresh landmarks
                poly_pts = self.region_controller.update(dual, (h, w))

                # 4. Multi-Layer AR Compositing
                active_mode = ARMode.ALL[self.current_mode_idx]
                composited = self._render_mode_pipeline(frame, poly_pts, active_mode)

                # 5. Visual Overlays: ONLY in Debug Mode!
                # (Normal mode presents clean AR output without polygon lines or skeletons)
                if self.is_debug_mode:
                    composited = self.hand_tracker.draw_debug_hands(composited, dual)
                    if poly_pts is not None and self.region_controller.is_active():
                        composited = MaskRenderer.draw_region_outline(
                            composited, poly_pts, state="LOCKED" if self.region_controller.is_frozen else "PREVIEW"
                        )

                # Shutter Flash Feedback
                if self.shutter_flash_frames > 0:
                    white = np.full_like(composited, 255)
                    composited = cv2.addWeighted(composited, 0.4, white, 0.6, 0)
                    self.shutter_flash_frames -= 1

                # Record frame if video writer active
                self.latest_frame = composited.copy()
                if self.recorder.is_recording:
                    self.recorder.write_frame(composited)

                # 6. Render Minimalist Heads-Up Display onto Frame
                display_frame = self._render_minimal_hud(composited, dual, g_res, active_mode)

                # 7. Push to GUI Canvas
                self._draw_to_canvas(display_frame)

        # Target ~30-60 FPS
        self.root.after(16, self.update_loop)

    def _render_mode_pipeline(
        self,
        frame_bgr: np.ndarray,
        poly_pts: Optional[np.ndarray],
        mode: str
    ) -> np.ndarray:
        """Executes the specific compositing logic for Modes A, B, C, D."""
        h, w = frame_bgr.shape[:2]

        if mode == ARMode.MODE_A:
            # MODE A: Local Effect (Only pixels inside hand polygon transform; outside is 100% original webcam)
            if poly_pts is not None and self.region_controller.is_active():
                effect_frame = self.effect_manager.render_effect_frame(frame_bgr, poly_pts)
                mask = self.region_controller.generate_mask((h, w))
                return MaskRenderer.apply_masked_effect(frame_bgr, effect_frame, mask, opacity=0.96)
            return frame_bgr

        elif mode == ARMode.MODE_B:
            # MODE B: Scene Portal (Virtual scene appears warped inside hand polygon; outside is 100% original webcam)
            if poly_pts is not None and self.region_controller.is_active():
                portal_frame = self.background_compositor.render_portal_frame(frame_bgr, poly_pts)
                mask = self.region_controller.generate_mask((h, w))
                return MaskRenderer.apply_masked_effect(frame_bgr, portal_frame, mask, opacity=0.98)
            return frame_bgr

        elif mode == ARMode.MODE_C:
            # MODE C: Character Transformation (Virtual background + live person + face/costume overlay)
            person_mask = self.person_segmenter.segment(frame_bgr)
            scene = self.background_compositor.composite_full(frame_bgr, person_mask)
            self.character_compositor.is_enabled = True
            # Pass original frame for pristine face landmark detection
            return self.character_compositor.composite(scene, raw_frame_for_detection=frame_bgr)

        elif mode == ARMode.MODE_D:
            # MODE D: Combined AR (Background replaced + character overlay + hand-controlled portal/energy zone)
            person_mask = self.person_segmenter.segment(frame_bgr)
            scene = self.background_compositor.composite_full(frame_bgr, person_mask)
            self.character_compositor.is_enabled = True
            scene = self.character_compositor.composite(scene, raw_frame_for_detection=frame_bgr)

            if poly_pts is not None and self.region_controller.is_active():
                effect_frame = self.effect_manager.render_effect_frame(scene, poly_pts)
                mask = self.region_controller.generate_mask((h, w))
                return MaskRenderer.apply_masked_effect(scene, effect_frame, mask, opacity=0.95)
            return scene

        return frame_bgr

    def _render_minimal_hud(
        self,
        frame_bgr: np.ndarray,
        dual: DualHandData,
        g_res: GestureResult,
        mode: str
    ) -> np.ndarray:
        """Renders subtle, non-intrusive on-screen text and status indicator."""
        hud = frame_bgr.copy()
        h, w = frame_bgr.shape[:2]
        now = time.time()

        # 1. Top-Right: Tiny Unobtrusive Tracking Dot
        dot_x = w - 24
        dot_y = 24
        if dual.both_hands_visible:
            dot_col = (0, 230, 100)  # Green: 2 hands active
        elif dual.hands_count == 1:
            dot_col = (0, 200, 255)  # Yellow: 1 hand active
        else:
            dot_col = (70, 70, 100)  # Dim: looking for hands

        cv2.circle(hud, (dot_x, dot_y), 6, dot_col, -1, cv2.LINE_AA)
        cv2.circle(hud, (dot_x, dot_y), 8, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Recording Indicator (when recording active)
        if self.recorder.is_recording:
            dur = int(self.recorder.get_duration())
            rec_str = f"● REC {dur // 60:02d}:{dur % 60:02d}"
            cv2.rectangle(hud, (w - 145, 12), (w - 40, 36), (15, 15, 20), -1)
            cv2.putText(hud, rec_str, (w - 138, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)

        # 3. Minimal Instruction Pill (shown ONLY when hands are not in view)
        if not dual.both_hands_visible:
            if dual.hands_count == 0:
                prompt_text = "👋 Show both hands to control AR region"
            else:
                prompt_text = "👋 Show other hand to frame active region"

            tw = len(prompt_text) * 9 + 30
            tx1 = (w - tw) // 2
            ty1 = h - 46
            # Translucent dark pill
            overlay_box = hud.copy()
            cv2.rectangle(overlay_box, (tx1, ty1), (tx1 + tw, ty1 + 28), (15, 18, 24), -1)
            cv2.rectangle(overlay_box, (tx1, ty1), (tx1 + tw, ty1 + 28), (60, 70, 90), 1)
            hud = cv2.addWeighted(overlay_box, 0.75, hud, 0.25, 0)
            cv2.putText(hud, prompt_text, (tx1 + 15, ty1 + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 225, 235), 1, cv2.LINE_AA)

        # 4. Floating Toast Notification (fades out automatically)
        if now < self.toast_expires_at:
            alpha = min(1.0, (self.toast_expires_at - now) / 0.4)
            tw = len(self.toast_message) * 10 + 36
            tx1 = (w - tw) // 2
            ty1 = 20
            toast_overlay = hud.copy()
            cv2.rectangle(toast_overlay, (tx1, ty1), (tx1 + tw, ty1 + 32), (12, 16, 24), -1)
            cv2.rectangle(toast_overlay, (tx1, ty1), (tx1 + tw, ty1 + 32), (0, 215, 255), 1)
            hud = cv2.addWeighted(toast_overlay, 0.85 * alpha, hud, 1.0 - (0.85 * alpha), 0)
            cv2.putText(hud, self.toast_message, (tx1 + 18, ty1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

        # 5. Debug Mode Overlay (ONLY when debug mode is enabled)
        if self.is_debug_mode:
            fps = self.camera.get_fps()
            debug_info = f"DEBUG | FPS: {fps:.1f} | Mode: {mode.split(':')[0]} | Gesture: {g_res.primary_gesture}"
            if self.region_controller.is_frozen:
                debug_info += " | [LOCKED]"
            cv2.rectangle(hud, (0, 0), (w, 24), (20, 25, 35), -1)
            cv2.putText(hud, debug_info, (12, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 200), 1, cv2.LINE_AA)

        return hud

    def _draw_to_canvas(self, frame_bgr: np.ndarray) -> None:
        """Scales image to fill the application window smoothly with correct aspect ratio."""
        try:
            cw = max(100, self.root.winfo_width())
            ch = max(100, self.root.winfo_height())
            fh, fw = frame_bgr.shape[:2]

            # Aspect-ratio preserving fit to entire window
            scale = min(cw / float(fw), ch / float(fh))
            disp_w = max(1, int(round(fw * scale)))
            disp_h = max(1, int(round(fh * scale)))

            resized = cv2.resize(frame_bgr, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(disp_w, disp_h))

            self.canvas_label.configure(image=ctk_img)
            self.canvas_label.image = ctk_img
        except Exception as e:
            logger.warning(f"Canvas draw warning: {e}")

    # --- GESTURE ACTION DISPATCHER ---

    def _dispatch_gesture_action(self, action: str) -> None:
        """Hands-free gesture action execution."""
        if action == "next_effect":
            self.next_effect()
        elif action == "prev_effect":
            self.prev_effect()
        elif action == "cycle_scene_mode":
            self.cycle_mode()

    # --- ACTIONS & CAPTURE ---

    def capture_photo(self) -> None:
        if self.latest_frame is not None:
            success, path_or_err = PhotoSaver.save(self.latest_frame, self.output_dir)
            if success:
                self.shutter_flash_frames = 3
                self.set_toast(f"📸 Snapshot Saved: {Path(path_or_err).name}")
            else:
                self.set_toast(f"Capture error: {path_or_err}")

    def toggle_recording(self) -> None:
        if not self.recorder.is_recording:
            if self.latest_frame is not None:
                h, w = self.latest_frame.shape[:2]
                fps = max(15.0, self.camera.get_fps())
                success, path_or_err = self.recorder.start(w, h, fps=fps, output_dir=self.output_dir)
                if success:
                    self.set_toast(f"⏺ Recording Video to {Path(path_or_err).name}...")
        else:
            success, path_or_err, frames, dur = self.recorder.stop()
            if success:
                self.set_toast(f"🎬 Video Saved: {Path(path_or_err).name} ({dur:.1f}s)")

    def reload_assets(self) -> None:
        self.effect_manager.reload()
        self.background_compositor.reload()
        self.character_compositor.reload()
        self.set_toast("All Assets Reloaded from Disk!")

    def on_closing(self) -> None:
        logger.info("Releasing AR Studio resources...")
        if self.recorder.is_recording:
            self.recorder.stop()
        self.camera.release()
        self.hand_tracker.release()
        self.person_segmenter.release()
        self.character_compositor.release()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
