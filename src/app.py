"""
Desktop Application Controller for Gesture-Controlled AR Experience.
Provides real-time hands-free AR interaction:
- Two-hand dynamic region definition & locking
- Gesture-based effect switching (swipes, dual pinch, V-sign)
- Layered compositing: Background Replacement -> Live Person -> Character Overlay -> Hand Effect Polygon
- Futuristic non-intrusive HUD with gesture telemetry, FPS, and hint banner
- Photo snapshots and asynchronous MP4 video recording
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
from .hand_tracker import HandTracker, DualHandData
from .gesture_recognizer import GestureRecognizer, GestureResult
from .region_controller import RegionController, RegionState
from .mask_renderer import MaskRenderer
from .effect_manager import EffectManager
from .person_segmenter import PersonSegmenter
from .background_compositor import BackgroundCompositor
from .character_compositor import CharacterCompositor
from .recorder import PhotoSaver, VideoRecorder

logger = logging.getLogger(__name__)

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class SceneMode:
    MODE_A = "Mode A: Hand-Controlled Region"
    MODE_B = "Mode B: Background Replacement"
    MODE_C = "Mode C: Background + Hand Effect"
    MODE_D = "Mode D: Character Overlay"
    MODE_E = "Mode E: Full Scene Interaction"

    ALL_MODES = [MODE_A, MODE_B, MODE_C, MODE_D, MODE_E]


class GestureARStudioApp:
    """Main desktop application window and interactive event controller."""

    def __init__(self, camera_index: int = 0, output_dir: str = "output") -> None:
        self.output_dir = str(Path(output_dir).resolve())

        # Initialize computer vision pipeline engines
        self.camera = Camera(device_index=camera_index, width=640, height=480, fps=30)
        self.hand_tracker = HandTracker(max_num_hands=2, smoothing_alpha=0.72)
        self.gesture_recognizer = GestureRecognizer()
        self.region_controller = RegionController(feather_radius=21, smoothing_alpha=0.70)
        self.effect_manager = EffectManager(effects_dir="assets/effects", textures_dir="assets/textures")
        self.person_segmenter = PersonSegmenter(model_selection=0)
        self.background_compositor = BackgroundCompositor(backgrounds_dir="assets/backgrounds")
        self.character_compositor = CharacterCompositor(characters_dir="assets/characters")
        self.recorder = VideoRecorder(output_dir=self.output_dir)

        # State
        self.current_mode_idx = 0
        self.show_debug_skeletons = False
        self.effect_opacity = 0.95
        self.shutter_flash_frames = 0
        self.is_camera_running = False
        self.latest_processed_frame: Optional[np.ndarray] = None

        # Build GUI
        self.root = ctk.CTk()
        self.root.title("Gesture-Controlled AR Experience - Hand-Defined Effects & Scene Studio")
        self.root.geometry("1280x820")
        self.root.minsize(1000, 680)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self._build_ui()
        self._bind_shortcuts()

        # Start Camera & Processing Loop
        self.start_camera()
        self.update_loop()

    def _build_ui(self) -> None:
        """Constructs layout: Left large video canvas, Right compact studio sidebar."""
        self.root.grid_columnconfigure(0, weight=4)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # --- LEFT PANEL: VIDEO PREVIEW CANVAS & HUD ---
        self.left_panel = ctk.CTkFrame(self.root, corner_radius=12, fg_color="#0e1017")
        self.left_panel.grid(row=0, column=0, padx=(16, 8), pady=16, sticky="nsew")
        self.left_panel.grid_rowconfigure(0, weight=0)  # Top HUD Telemetry
        self.left_panel.grid_rowconfigure(1, weight=1)  # Video
        self.left_panel.grid_rowconfigure(2, weight=0)  # Bottom Hint Banner
        self.left_panel.grid_columnconfigure(0, weight=1)

        # 1. Top HUD Telemetry Bar
        self.hud_top = ctk.CTkFrame(self.left_panel, fg_color="#151824", corner_radius=8, height=44)
        self.hud_top.grid(row=0, column=0, padx=12, pady=(10, 6), sticky="ew")

        self.title_badge = ctk.CTkLabel(
            self.hud_top,
            text="⚡ GESTURE AR STUDIO",
            font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
            text_color="#00e5ff"
        )
        self.title_badge.pack(side="left", padx=14, pady=6)

        self.mode_badge = ctk.CTkLabel(
            self.hud_top,
            text="MODE A: HAND REGION",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color="#ffffff",
            fg_color="#1e3250",
            corner_radius=6,
            width=180
        )
        self.mode_badge.pack(side="left", padx=8, pady=6)

        self.gesture_badge = ctk.CTkLabel(
            self.hud_top,
            text="SEARCHING HANDS",
            font=ctk.CTkFont(family="Consolas", size=12, weight="bold"),
            text_color="#ffb300",
            fg_color="#332400",
            corner_radius=6,
            width=170
        )
        self.gesture_badge.pack(side="left", padx=8, pady=6)

        self.fps_badge = ctk.CTkLabel(
            self.hud_top,
            text="0.0 FPS",
            font=ctk.CTkFont(family="Consolas", size=12, weight="bold"),
            text_color="#00e5ff",
            fg_color="#0e2333",
            corner_radius=6,
            width=70
        )
        self.fps_badge.pack(side="right", padx=(6, 14), pady=6)

        self.rec_badge = ctk.CTkLabel(
            self.hud_top,
            text="● REC 00:00",
            font=ctk.CTkFont(family="Consolas", size=12, weight="bold"),
            text_color="#ffffff",
            fg_color="#d32f2f",
            corner_radius=6,
            width=100
        )
        # Hidden by default

        # 2. Main Video Canvas
        self.video_container = ctk.CTkFrame(self.left_panel, fg_color="#050609", corner_radius=10)
        self.video_container.grid(row=1, column=0, padx=12, pady=6, sticky="nsew")
        self.video_container.grid_rowconfigure(0, weight=1)
        self.video_container.grid_columnconfigure(0, weight=1)

        self.video_canvas = ctk.CTkLabel(self.video_container, text="", fg_color="transparent")
        self.video_canvas.grid(row=0, column=0, sticky="nsew")

        # 3. Bottom Intuitive Hint / Gesture Ticker Banner
        self.hud_bottom = ctk.CTkFrame(self.left_panel, fg_color="#151824", corner_radius=8, height=36)
        self.hud_bottom.grid(row=2, column=0, padx=12, pady=(6, 12), sticky="ew")

        self.hint_label = ctk.CTkLabel(
            self.hud_bottom,
            text="💡 Bring both hands into view to define region • Pinch to lock • Swipe right for next effect • Peace sign for modes",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color="#8ea0b5"
        )
        self.hint_label.pack(side="left", padx=14, pady=6)

        # --- RIGHT PANEL: STUDIO TELEMETRY & CONTROLS ---
        self.right_panel = ctk.CTkFrame(self.root, corner_radius=12, fg_color="#131620")
        self.right_panel.grid(row=0, column=1, padx=(8, 16), pady=16, sticky="nsew")
        self.right_panel.grid_rowconfigure(5, weight=1)
        self.right_panel.grid_columnconfigure(0, weight=1)

        # Section 1: Active Telemetry Card
        self.telemetry_card = ctk.CTkFrame(self.right_panel, fg_color="#1a1e2c", corner_radius=10)
        self.telemetry_card.grid(row=0, column=0, padx=12, pady=(12, 6), sticky="ew")

        ctk.CTkLabel(
            self.telemetry_card,
            text="AR PIPELINE TELEMETRY",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#6e7d91"
        ).pack(anchor="w", padx=12, pady=(8, 4))

        self.active_effect_label = ctk.CTkLabel(
            self.telemetry_card,
            text="Effect: Magic Energy Portal",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color="#00d2ff",
            anchor="w"
        )
        self.active_effect_label.pack(fill="x", padx=12, pady=2)

        self.active_bg_label = ctk.CTkLabel(
            self.telemetry_card,
            text="Background: Cyberpunk City",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color="#c0cad6",
            anchor="w"
        )
        self.active_bg_label.pack(fill="x", padx=12, pady=2)

        self.hands_state_label = ctk.CTkLabel(
            self.telemetry_card,
            text="Hands: Left [LOST] | Right [LOST]",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color="#ffb300",
            anchor="w"
        )
        self.hands_state_label.pack(fill="x", padx=12, pady=(2, 8))

        # Section 2: Mode Selector & Hand Region Lock
        self.modes_card = ctk.CTkFrame(self.right_panel, fg_color="#1a1e2c", corner_radius=10)
        self.modes_card.grid(row=1, column=0, padx=12, pady=6, sticky="ew")

        ctk.CTkLabel(
            self.modes_card,
            text="SCENE MODE & LOCK",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#6e7d91"
        ).pack(anchor="w", padx=12, pady=(8, 4))

        self.mode_menu = ctk.CTkOptionMenu(
            self.modes_card,
            values=SceneMode.ALL_MODES,
            command=self._on_mode_dropdown
        )
        self.mode_menu.pack(fill="x", padx=10, pady=4)

        # Quick Lock & Invert row
        lock_row = ctk.CTkFrame(self.modes_card, fg_color="transparent")
        lock_row.pack(fill="x", padx=10, pady=(2, 8))

        self.lock_btn = ctk.CTkButton(
            lock_row,
            text="🔒 Lock Region",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            fg_color="#26354a",
            hover_color="#364966",
            height=30,
            command=self.toggle_lock
        )
        self.lock_btn.pack(side="left", expand=True, fill="x", padx=(0, 4))

        self.invert_btn = ctk.CTkButton(
            lock_row,
            text="Invert Mask",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            fg_color="#2b3044",
            hover_color="#39405a",
            width=80,
            height=30,
            command=self.toggle_invert
        )
        self.invert_btn.pack(side="right", padx=(4, 0))

        # Section 3: Fine Tuning & Opacity
        self.tune_card = ctk.CTkFrame(self.right_panel, fg_color="#1a1e2c", corner_radius=10)
        self.tune_card.grid(row=2, column=0, padx=12, pady=6, sticky="ew")

        op_header = ctk.CTkFrame(self.tune_card, fg_color="transparent")
        op_header.pack(fill="x", padx=12, pady=(8, 0))
        ctk.CTkLabel(op_header, text="Effect Opacity:", font=ctk.CTkFont(size=12), text_color="#a0b0c2").pack(side="left")
        self.op_val_label = ctk.CTkLabel(op_header, text="95%", font=ctk.CTkFont(family="Consolas", size=12, weight="bold"), text_color="#00e5ff")
        self.op_val_label.pack(side="right")

        self.op_slider = ctk.CTkSlider(self.tune_card, from_=0.1, to=1.0, number_of_steps=90, command=self._on_opacity_change)
        self.op_slider.set(0.95)
        self.op_slider.pack(fill="x", padx=12, pady=(2, 6))

        sw_row = ctk.CTkFrame(self.tune_card, fg_color="transparent")
        sw_row.pack(fill="x", padx=12, pady=(2, 8))

        self.debug_switch = ctk.CTkSwitch(sw_row, text="Hand Skeletons", font=ctk.CTkFont(size=12), command=self._on_debug_toggle)
        self.debug_switch.pack(side="left")

        # Section 4: Capture & Recording
        self.media_card = ctk.CTkFrame(self.right_panel, fg_color="#1a1e2c", corner_radius=10)
        self.media_card.grid(row=3, column=0, padx=12, pady=6, sticky="ew")

        ctk.CTkLabel(self.media_card, text="SNAPSHOT & RECORDING", font=ctk.CTkFont(size=11, weight="bold"), text_color="#6e7d91").pack(anchor="w", padx=12, pady=(8, 4))

        m_row = ctk.CTkFrame(self.media_card, fg_color="transparent")
        m_row.pack(fill="x", padx=10, pady=(2, 8))

        self.photo_btn = ctk.CTkButton(m_row, text="📷 Photo (Space)", font=ctk.CTkFont(size=12, weight="bold"), fg_color="#0288d1", hover_color="#0277bd", height=34, command=self.capture_photo)
        self.photo_btn.pack(side="left", expand=True, fill="x", padx=(0, 4))

        self.rec_btn = ctk.CTkButton(m_row, text="⏺ Record (R)", font=ctk.CTkFont(size=12, weight="bold"), fg_color="#c62828", hover_color="#b71c1c", height=34, command=self.toggle_recording)
        self.rec_btn.pack(side="right", expand=True, fill="x", padx=(4, 0))

        # Section 5: Gesture Cheat Sheet
        self.cheat_card = ctk.CTkFrame(self.right_panel, fg_color="#1a1e2c", corner_radius=10)
        self.cheat_card.grid(row=4, column=0, padx=12, pady=6, sticky="ew")

        ctk.CTkLabel(self.cheat_card, text="HANDS-FREE GESTURE GUIDE", font=ctk.CTkFont(size=11, weight="bold"), text_color="#6e7d91").pack(anchor="w", padx=12, pady=(8, 4))

        cheats = [
            ("👋 Swipe Right/Left", "Next / Previous Effect"),
            ("🤏 Dual Pinch", "Lock / Unlock Region"),
            ("✌️ Peace / V-Sign", "Cycle Scene Mode (A→E)"),
            ("✋ Open Palms", "Dynamic Region Follow"),
            ("✊ Fist", "Freeze Region Position")
        ]
        for g_name, g_desc in cheats:
            r = ctk.CTkFrame(self.cheat_card, fg_color="transparent")
            r.pack(fill="x", padx=10, pady=1)
            ctk.CTkLabel(r, text=g_name, font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), text_color="#00e5ff", width=120, anchor="w").pack(side="left")
            ctk.CTkLabel(r, text=g_desc, font=ctk.CTkFont(size=11), text_color="#8ca0b5", anchor="w").pack(side="left")

        # Spacer row 5

        # Footer
        self.footer = ctk.CTkFrame(self.right_panel, fg_color="#1a1e2c", corner_radius=10)
        self.footer.grid(row=6, column=0, padx=12, pady=(6, 12), sticky="ew")

        f_row = ctk.CTkFrame(self.footer, fg_color="transparent")
        f_row.pack(fill="x", padx=10, pady=(6, 6))

        self.reload_btn = ctk.CTkButton(f_row, text="🔄 Reload Assets (F5)", font=ctk.CTkFont(size=11), fg_color="#2b3044", hover_color="#39405a", height=26, command=self.reload_assets)
        self.reload_btn.pack(side="left", expand=True, fill="x", padx=(0, 4))

        self.exit_btn = ctk.CTkButton(f_row, text="Exit", font=ctk.CTkFont(size=11), fg_color="#2e2226", hover_color="#541a22", width=60, height=26, command=self.on_closing)
        self.exit_btn.pack(side="right")

    def _bind_shortcuts(self) -> None:
        """Accessibility and debugging keyboard shortcuts."""
        self.root.bind("<Right>", lambda e: self.next_effect())
        self.root.bind("<Left>", lambda e: self.prev_effect())
        self.root.bind("<space>", lambda e: self.capture_photo())
        self.root.bind("<r>", lambda e: self.toggle_recording())
        self.root.bind("<R>", lambda e: self.toggle_recording())
        self.root.bind("<m>", lambda e: self.cycle_mode())
        self.root.bind("<M>", lambda e: self.cycle_mode())
        self.root.bind("<b>", lambda e: self.next_background())
        self.root.bind("<B>", lambda e: self.next_background())
        self.root.bind("<c>", lambda e: self.next_character())
        self.root.bind("<C>", lambda e: self.next_character())
        self.root.bind("<l>", lambda e: self.toggle_lock())
        self.root.bind("<L>", lambda e: self.toggle_lock())
        self.root.bind("<i>", lambda e: self.toggle_invert())
        self.root.bind("<I>", lambda e: self.toggle_invert())
        self.root.bind("<d>", lambda e: self._toggle_debug_shortcut())
        self.root.bind("<D>", lambda e: self._toggle_debug_shortcut())
        self.root.bind("<F5>", lambda e: self.reload_assets())
        self.root.bind("<Escape>", lambda e: self.on_closing())

    def start_camera(self) -> None:
        self.is_camera_running = self.camera.start()
        if not self.is_camera_running:
            self.hint_label.configure(text="CAMERA ERROR: Check webcam connection and Windows privacy permissions.")
            messagebox.showerror("Webcam Error", "Could not connect to webcam. Verify permissions in Windows Settings.")

    def update_loop(self) -> None:
        """Core real-time frame processing loop executed on Tkinter timer."""
        if self.is_camera_running:
            ret, frame = self.camera.read_frame()
            if ret and frame is not None:
                # Mirror webcam horizontally for natural intuitive interaction
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                # 1. Track Hands
                dual_data: DualHandData = self.hand_tracker.process(frame)

                # 2. Recognize Gestures
                gesture_res: GestureResult = self.gesture_recognizer.process(dual_data)

                # Handle gesture-triggered actions (Hands-Free!)
                if gesture_res.action_triggered:
                    self._handle_gesture_action(gesture_res.action_triggered)

                # Update HUD Telemetry
                self._update_hud_telemetry(dual_data, gesture_res)

                # 3. Update Hand-Controlled Region Polygon
                poly_pts = self.region_controller.update(dual_data, (h, w))

                # 4. Multi-Layer AR Compositing Pipeline
                active_mode = SceneMode.ALL_MODES[self.current_mode_idx]

                # LAYER 1 & 2: Background Replacement + Segmented Person
                if active_mode in [SceneMode.MODE_B, SceneMode.MODE_C, SceneMode.MODE_D, SceneMode.MODE_E]:
                    self.background_compositor.is_enabled = True
                    person_mask = self.person_segmenter.segment(frame)
                    base_scene = self.background_compositor.composite(frame, person_mask)
                else:
                    self.background_compositor.is_enabled = False
                    base_scene = frame.copy()

                # LAYER 3: Character / Costume Overlay
                if active_mode in [SceneMode.MODE_D, SceneMode.MODE_E]:
                    self.character_compositor.is_enabled = True
                    base_scene = self.character_compositor.composite(base_scene)
                else:
                    self.character_compositor.is_enabled = False

                # LAYER 4: Hand-Controlled Region Polygon Effect
                if active_mode in [SceneMode.MODE_A, SceneMode.MODE_C, SceneMode.MODE_E]:
                    if poly_pts is not None and self.region_controller.is_active():
                        # Render effect transformed into polygon space
                        effect_frame = self.effect_manager.render_effect_frame(base_scene, poly_pts)
                        # Generate smooth feathered polygon mask
                        feathered_mask = self.region_controller.generate_mask((h, w))
                        # Apply effect strictly to mask
                        base_scene = MaskRenderer.apply_masked_effect(
                            base_scene, effect_frame, feathered_mask, opacity=self.effect_opacity
                        )
                        # Draw glowing animated polygon outline
                        base_scene = MaskRenderer.draw_region_outline(
                            base_scene, poly_pts, state=self.region_controller.state
                        )

                # Optional Hand Landmark Skeleton Rendering
                if self.show_debug_skeletons:
                    base_scene = self.hand_tracker.draw_debug_hands(base_scene, dual_data)

                # Visual Shutter Flash on Snapshot Capture
                if self.shutter_flash_frames > 0:
                    white = np.full_like(base_scene, 255)
                    base_scene = cv2.addWeighted(base_scene, 0.4, white, 0.6, 0)
                    self.shutter_flash_frames -= 1

                # Save frame reference for snapshot & video recording
                self.latest_processed_frame = base_scene.copy()
                if self.recorder.is_recording:
                    self.recorder.write_frame(base_scene)
                    dur = int(self.recorder.get_duration())
                    self.rec_badge.configure(text=f"● REC {dur // 60:02d}:{dur % 60:02d}")

                # Display to canvas
                self._render_canvas(base_scene)

        # Reschedule next frame (~30-60 FPS)
        self.root.after(16, self.update_loop)

    def _handle_gesture_action(self, action: str) -> None:
        """Executes hands-free actions triggered by recognized gestures."""
        if action == "next_effect":
            self.next_effect()
        elif action == "prev_effect":
            self.prev_effect()
        elif action == "toggle_lock_region":
            self.toggle_lock()
        elif action == "cycle_scene_mode":
            self.cycle_mode()
        elif action == "preview_mode":
            if self.region_controller.state == RegionState.LOCKED:
                self.region_controller.set_lock(False)
                self.lock_btn.configure(text="🔒 Lock Region", fg_color="#26354a")
                self.hint_label.configure(text="Region unlocked: Follows hand movement")
        elif action == "freeze_region":
            self.region_controller.set_lock(True)
            self.lock_btn.configure(text="🔓 Unlock Region", fg_color="#d32f2f")
            self.hint_label.configure(text="Region locked in place")

    def _update_hud_telemetry(self, dual: DualHandData, g_res: GestureResult) -> None:
        """Updates telemetry labels and badges."""
        fps = self.camera.get_fps()
        self.fps_badge.configure(text=f"{fps:.1f} FPS")

        # Gesture Badge
        if g_res.primary_gesture in ["DUAL_PINCH", "SWIPE_RIGHT", "SWIPE_LEFT", "V_SIGN", "FIST"]:
            self.gesture_badge.configure(text=f"★ {g_res.primary_gesture}", text_color="#00e676", fg_color="#0e331c")
        elif "TWO_HAND" in g_res.primary_gesture or "OPEN_PALM" in g_res.primary_gesture:
            self.gesture_badge.configure(text=g_res.primary_gesture, text_color="#00e5ff", fg_color="#0e2333")
        else:
            self.gesture_badge.configure(text=g_res.primary_gesture, text_color="#ffb300", fg_color="#332400")

        # Hands status
        l_status = "L: OK" if dual.left_hand else "L: --"
        r_status = "R: OK" if dual.right_hand else "R: --"
        col = "#00e676" if dual.both_hands_visible else ("#ffb300" if (dual.left_hand or dual.right_hand) else "#ef5350")
        self.hands_state_label.configure(text=f"Hands: {l_status} | {r_status}", text_color=col)

        # Dynamic Instruction Banner
        if g_res.description and g_res.description != "Waiting for hands...":
            self.hint_label.configure(text=f"💡 {g_res.description}")
        elif dual.both_hands_visible:
            if self.region_controller.state == RegionState.LOCKED:
                self.hint_label.configure(text="💡 Region is LOCKED • Pinch with both hands to unlock • Swipe hand to change effect")
            else:
                self.hint_label.configure(text="💡 Region is TRACKING • Pinch both hands to lock • Move hands apart/together to scale")
        else:
            self.hint_label.configure(text="💡 Show both hands to webcam to define interactive region")

    def _render_canvas(self, frame_bgr: np.ndarray) -> None:
        """Scales and draws BGR frame onto the CTkLabel canvas."""
        try:
            cw = max(100, self.video_container.winfo_width())
            ch = max(100, self.video_container.winfo_height())
            fh, fw = frame_bgr.shape[:2]
            scale = min(cw / float(fw), ch / float(fh))
            disp_w = max(1, int(fw * scale))
            disp_h = max(1, int(fh * scale))

            resized = cv2.resize(frame_bgr, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(disp_w, disp_h))

            self.video_canvas.configure(image=ctk_img)
            self.video_canvas.image = ctk_img
        except Exception as e:
            logger.warning(f"Error drawing canvas: {e}")

    # --- ACTIONS & CONTROLS ---

    def next_effect(self) -> None:
        eff = self.effect_manager.next_effect()
        if eff:
            self.active_effect_label.configure(text=f"Effect: {eff.display_name}")
            self.hint_label.configure(text=f"Switched Effect: {eff.display_name} ({eff.category})")

    def prev_effect(self) -> None:
        eff = self.effect_manager.prev_effect()
        if eff:
            self.active_effect_label.configure(text=f"Effect: {eff.display_name}")
            self.hint_label.configure(text=f"Switched Effect: {eff.display_name} ({eff.category})")

    def cycle_mode(self) -> None:
        self.current_mode_idx = (self.current_mode_idx + 1) % len(SceneMode.ALL_MODES)
        mode_str = SceneMode.ALL_MODES[self.current_mode_idx]
        self.mode_menu.set(mode_str)
        self.mode_badge.configure(text=mode_str.split(":")[0].upper())
        self.hint_label.configure(text=f"Active Scene Mode: {mode_str}")

    def _on_mode_dropdown(self, value: str) -> None:
        if value in SceneMode.ALL_MODES:
            self.current_mode_idx = SceneMode.ALL_MODES.index(value)
            self.mode_badge.configure(text=value.split(":")[0].upper())

    def next_background(self) -> None:
        name = self.background_compositor.next_background()
        self.active_bg_label.configure(text=f"Background: {name}")
        self.hint_label.configure(text=f"Virtual Background: {name}")

    def next_character(self) -> None:
        name = self.character_compositor.next_character()
        self.hint_label.configure(text=f"Character Overlay: {name}")

    def toggle_lock(self) -> None:
        res = self.region_controller.toggle_lock()
        if res == "LOCKED":
            self.lock_btn.configure(text="🔓 Unlock Region", fg_color="#d32f2f")
            self.hint_label.configure(text="Region LOCKED in space (Pinch both hands to unlock)")
        else:
            self.lock_btn.configure(text="🔒 Lock Region", fg_color="#26354a")
            self.hint_label.configure(text="Region UNLOCKED: Follows hands dynamically")

    def toggle_invert(self) -> None:
        inv = self.region_controller.toggle_invert()
        self.invert_btn.configure(fg_color="#0288d1" if inv else "#2b3044")
        self.hint_label.configure(text="Mask Inverted (Outside Effect)" if inv else "Mask Normal (Inside Effect)")

    def _on_opacity_change(self, value: float) -> None:
        self.effect_opacity = float(value)
        self.op_val_label.configure(text=f"{int(round(value * 100))}%")

    def _on_debug_toggle(self) -> None:
        self.show_debug_skeletons = bool(self.debug_switch.get())

    def _toggle_debug_shortcut(self) -> None:
        self.show_debug_skeletons = not self.show_debug_skeletons
        if self.show_debug_skeletons:
            self.debug_switch.select()
        else:
            self.debug_switch.deselect()

    def reload_assets(self) -> None:
        self.effect_manager.reload()
        self.background_compositor.reload()
        self.character_compositor.reload()
        self.hint_label.configure(text="All assets reloaded from disk!")

    def capture_photo(self) -> None:
        if self.latest_processed_frame is not None:
            success, path_or_err = PhotoSaver.save(self.latest_processed_frame, self.output_dir)
            if success:
                self.shutter_flash_frames = 3
                self.hint_label.configure(text=f"📸 Saved Photo: {Path(path_or_err).name}")
            else:
                self.hint_label.configure(text=f"Photo error: {path_or_err}")

    def toggle_recording(self) -> None:
        if not self.recorder.is_recording:
            if self.latest_processed_frame is not None:
                h, w = self.latest_processed_frame.shape[:2]
                fps = max(15.0, self.camera.get_fps())
                success, path_or_err = self.recorder.start(w, h, fps=fps, output_dir=self.output_dir)
                if success:
                    self.rec_btn.configure(text="⏹ Stop (R)", fg_color="#d32f2f")
                    self.rec_badge.pack(side="right", padx=6, pady=6)
                    self.hint_label.configure(text=f"Recording to {Path(path_or_err).name}...")
        else:
            success, path_or_err, frames, dur = self.recorder.stop()
            self.rec_btn.configure(text="⏺ Record (R)", fg_color="#c62828")
            self.rec_badge.pack_forget()
            if success:
                self.hint_label.configure(text=f"🎬 Video Saved: {Path(path_or_err).name} ({frames} frames, {dur:.1f}s)")

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
