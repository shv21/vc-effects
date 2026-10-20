"""
Immersive Full-Screen AR Studio Application.
Replaces clutter with an edge-to-edge camera experience where hands directly
control the visual transformation.
Modes:
- MODE A: Local Effect (Only pixels inside hand polygon transform)
- MODE B: Scene Portal (Virtual scene appears warped inside hand polygon)
- MODE C: Character Transformation (Virtual background + face mask / costume overlay)
- MODE D: Combined AR (Full background + character transformation + hand-controlled portal/effect)
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
    """Full-screen immersive AR studio window with gesture control."""

    def __init__(self, camera_index: int = 0, output_dir: str = "output") -> None:
        self.output_dir = str(Path(output_dir).resolve())

        # Core Computer Vision Pipeline
        self.camera = Camera(device_index=camera_index, width=640, height=480, fps=30)
        self.hand_tracker = HandTracker(max_num_hands=2, base_smoothing_alpha=0.75)
        self.gesture_recognizer = GestureRecognizer()
        self.region_controller = RegionController(feather_radius=21, smoothing_alpha=0.80)
        self.effect_manager = EffectManager(effects_dir="assets/effects", textures_dir="assets/textures")
        self.person_segmenter = PersonSegmenter(model_selection=0)
        self.background_compositor = BackgroundCompositor(backgrounds_dir="assets/backgrounds")
        self.character_compositor = CharacterCompositor(characters_dir="assets/characters")
        self.recorder = VideoRecorder(output_dir=self.output_dir)

        # State
        self.current_mode_idx = 0
        self.is_debug_mode = False
        self.is_fullscreen = False
        self.shutter_flash_frames = 0
        self.is_camera_running = False
        self.latest_frame: Optional[np.ndarray] = None

        # Auto-fading toast notification
        self.toast_message: str = "Show both hands to begin"
        self.toast_expires_at: float = time.time() + 4.0

        # Build borderless / immersive window
        self.root = ctk.CTk()
        self.root.title("Gesture-Controlled AR Experience")
        self.root.geometry("1280x720")
        self.root.minsize(800, 600)
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
        self.root.bind("<d>", lambda e: self.toggle_debug())
        self.root.bind("<D>", lambda e: self.toggle_debug())
        self.root.bind("<m>", lambda e: self.cycle_mode())
        self.root.bind("<M>", lambda e: self.cycle_mode())
        self.root.bind("<Right>", lambda e: self.next_effect())
        self.root.bind("<Left>", lambda e: self.prev_effect())
        self.root.bind("<b>", lambda e: self.next_background())
        self.root.bind("<B>", lambda e: self.next_background())
        self.root.bind("<c>", lambda e: self.next_character())
        self.root.bind("<C>", lambda e: self.next_character())
        self.root.bind("<space>", lambda e: self.capture_photo())
        self.root.bind("<r>", lambda e: self.toggle_recording())
        self.root.bind("<R>", lambda e: self.toggle_recording())
        self.root.bind("<f>", lambda e: self.toggle_freeze())
        self.root.bind("<F>", lambda e: self.toggle_freeze())
        self.root.bind("<i>", lambda e: self.toggle_invert())
        self.root.bind("<I>", lambda e: self.toggle_invert())
        self.root.bind("<F5>", lambda e: self.reload_assets())

    def start_camera(self) -> None:
        self.is_camera_running = self.camera.start()
        if not self.is_camera_running:
            self.set_toast("CAMERA ERROR: Cannot connect to webcam", duration=8.0)
            messagebox.showerror("Camera Error", "Could not connect to webcam. Verify permissions in Windows Settings.")

    def set_toast(self, message: str, duration: float = 2.0) -> None:
        """Sets a temporary floating toast message."""
        self.toast_message = message
        self.toast_expires_at = time.time() + duration

    def update_loop(self) -> None:
        """Core real-time frame processing and compositing loop."""
        if self.is_camera_running:
            ret, frame = self.camera.read_frame()
            if ret and frame is not None:
                # Mirror webcam horizontally for intuitive mirror view
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                # 1. Track Hands continuously
                dual: DualHandData = self.hand_tracker.process(frame)

                # 2. Recognize Gestures & Dispatch Hands-Free Actions
                g_res: GestureResult = self.gesture_recognizer.process(dual)
                if g_res.action_triggered:
                    self._dispatch_gesture_action(g_res.action_triggered)

                # 3. Update Hand-Controlled Region Polygon continuously
                poly_pts = self.region_controller.update(dual, (h, w))

                # 4. Multi-Layer AR Compositing
                active_mode = ARMode.ALL[self.current_mode_idx]
                composited = self._render_mode_pipeline(frame, poly_pts, active_mode)

                # 5. Visual Overlays: Outline & Debug Info
                if self.is_debug_mode:
                    # Draw diagnostic skeleton & boundary in debug mode only
                    composited = self.hand_tracker.draw_debug_hands(composited, dual)
                    if poly_pts is not None and self.region_controller.is_active():
                        composited = MaskRenderer.draw_region_outline(
                            composited, poly_pts, state="LOCKED" if self.region_controller.is_frozen else "PREVIEW"
                        )
                elif poly_pts is not None and self.region_controller.is_active():
                    # In normal presentation, draw subtle, elegant perimeter glow
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
            # MODE A: Local Effect (Only pixels inside hand polygon transform)
            if poly_pts is not None and self.region_controller.is_active():
                effect_frame = self.effect_manager.render_effect_frame(frame_bgr, poly_pts)
                mask = self.region_controller.generate_mask((h, w))
                return MaskRenderer.apply_masked_effect(frame_bgr, effect_frame, mask, opacity=0.95)
            return frame_bgr

        elif mode == ARMode.MODE_B:
            # MODE B: Scene Portal (Background image appears warped inside hand polygon)
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
            return self.character_compositor.composite(scene)

        elif mode == ARMode.MODE_D:
            # MODE D: Combined AR (Background + person + character overlay + hand polygon effect)
            person_mask = self.person_segmenter.segment(frame_bgr)
            scene = self.background_compositor.composite_full(frame_bgr, person_mask)
            self.character_compositor.is_enabled = True
            scene = self.character_compositor.composite(scene)

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

        # Top-Left: Sleek Mode & Active Item Badge
        active_eff = self.effect_manager.get_active_effect().display_name
        active_bg = self.background_compositor.get_current_background_name()
        active_char = self.character_compositor.get_current_character_name()

        if mode == ARMode.MODE_A:
            item_str = f"Effect: {active_eff}"
        elif mode == ARMode.MODE_B:
            item_str = f"Portal Scene: {active_bg}"
        elif mode == ARMode.MODE_C:
            item_str = f"Character: {active_char} | Scene: {active_bg}"
        else:
            item_str = f"{active_char} • {active_bg} • {active_eff}"

        # Translucent top bar
        cv2.rectangle(hud, (0, 0), (w, 36), (12, 14, 20), -1)

        # Mode title
        cv2.putText(hud, f"{mode}", (14, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 225, 255), 1, cv2.LINE_AA)
        cv2.putText(hud, f"|  {item_str}", (260, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 215, 230), 1, cv2.LINE_AA)

        # Top-Right: Tracking Status Dot
        dot_x = w - 90
        dot_col = (0, 230, 100) if dual.both_hands_visible else ((0, 200, 255) if dual.hands_count == 1 else (80, 80, 220))
        cv2.circle(hud, (dot_x, 18), 6, dot_col, -1, cv2.LINE_AA)
        status_txt = "2 HANDS" if dual.both_hands_visible else ("1 HAND" if dual.hands_count == 1 else "LOOKING")
        cv2.putText(hud, status_txt, (dot_x + 12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 190, 200), 1, cv2.LINE_AA)

        # Recording Badge
        if self.recorder.is_recording:
            dur = int(self.recorder.get_duration())
            rec_str = f"REC {dur // 60:02d}:{dur % 60:02d}"
            cv2.circle(hud, (w - 200, 18), 6, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(hud, rec_str, (w - 188, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Center Toast Notification (Fades out automatically)
        if now < self.toast_expires_at:
            alpha = min(1.0, (self.toast_expires_at - now) / 0.5)
            tw = len(self.toast_message) * 11 + 30
            tx1 = (w - tw) // 2
            ty1 = h - 55
            cv2.rectangle(hud, (tx1, ty1), (tx1 + tw, ty1 + 34), (16, 20, 28), -1)
            cv2.rectangle(hud, (tx1, ty1), (tx1 + tw, ty1 + 34), (0, 215, 255), 1)
            cv2.putText(hud, self.toast_message, (tx1 + 15, ty1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # Debug Information Overlay
        if self.is_debug_mode:
            fps = self.camera.get_fps()
            cv2.putText(hud, f"DEBUG MODE | FPS: {fps:.1f}", (14, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(hud, f"Gesture: {g_res.primary_gesture}", (220, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            if self.region_controller.is_frozen:
                cv2.putText(hud, "[REGION FROZEN]", (420, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 255), 1, cv2.LINE_AA)

        return cv2.addWeighted(hud, 0.92, frame_bgr, 0.08, 0)

    def _draw_to_canvas(self, frame_bgr: np.ndarray) -> None:
        """Scales image to fill the application window smoothly."""
        try:
            cw = max(100, self.root.winfo_width())
            ch = max(100, self.root.winfo_height())
            fh, fw = frame_bgr.shape[:2]

            scale = min(cw / float(fw), ch / float(ch))
            disp_w = max(1, int(fw * scale))
            disp_h = max(1, int(fh * scale))

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
        elif action == "toggle_freeze":
            self.toggle_freeze()
        elif action == "ensure_unlocked":
            if self.region_controller.is_frozen:
                self.region_controller.unfreeze()
                self.set_toast("Region Unfrozen: Following Hands")

    # --- CONTROLS & SHORTCUTS ---

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

    def toggle_freeze(self) -> None:
        frozen = self.region_controller.toggle_freeze()
        self.set_toast("Region FROZEN in place" if frozen else "Region LIVE: Following hands")

    def toggle_invert(self) -> None:
        inv = self.region_controller.toggle_invert()
        self.set_toast("Mask Inverted (Outside Effect)" if inv else "Mask Normal (Inside Effect)")

    def toggle_debug(self) -> None:
        self.is_debug_mode = not self.is_debug_mode
        self.set_toast("Debug Telemetry ON" if self.is_debug_mode else "Debug Telemetry OFF")

    def toggle_fullscreen(self) -> None:
        self.is_fullscreen = not self.is_fullscreen
        self.root.attributes("-fullscreen", self.is_fullscreen)
        self.set_toast("Fullscreen Enabled (F11 or Esc to exit)" if self.is_fullscreen else "Windowed Mode")

    def _on_escape(self) -> None:
        if self.is_fullscreen:
            self.toggle_fullscreen()
        else:
            self.on_closing()

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
