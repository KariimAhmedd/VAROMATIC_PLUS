import sys
import os
from pathlib import Path
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                          QHBoxLayout, QLabel, QComboBox, QPushButton,
                          QFileDialog, QMessageBox, QFrame, QGroupBox,
                          QSlider, QProgressBar, QLineEdit, QCheckBox, QSizePolicy,
                          QTabWidget, QSpinBox, QDoubleSpinBox, QGridLayout)
from PyQt5.QtGui import QPixmap, QIcon, QImage, QFont, QPalette, QColor
from PyQt5.QtCore import Qt, QSize, QTimer, QThread, pyqtSignal, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
import cv2
import numpy as np
import traceback
import time
import yt_dlp
import tempfile
import json
import queue
import copy
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any, Union

from lib.offside_modules.player_detection import PlayerDetector
from lib.offside_modules.color_analysis import ColorAnalyzer
from lib.offside_modules.offside_detection import OffsideDetector, Player
from lib.offside_modules.goal_detection import GoalDetector

# Import analytics modules
from lib.analytics.offside_analytics import OffsideAnalytics, OffsideEvent
from lib.analytics.visualization import AnalyticsVisualizer
from lib.analytics.ai_insights import InsightGenerator

# Utility functions
from lib.utils import create_logger, FPS_Counter

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VAROMATIC+ Pro")
        self.resize(1600, 1000)  # Larger window for better visibility
        
        # Initialize components without optimizations
        self.player_detector = PlayerDetector(confidence_threshold=0.15)  # Lower threshold for better detection
        self.color_analyzer = ColorAnalyzer()
        self.offside_detector = OffsideDetector()
        self.goal_detector = GoalDetector()
        
        # Video state
        self.cap = None
        self.video_path = None
        self.is_running = False
        self.is_paused = False
        self.current_frame = None
        self.frame_count = 0
        self.current_frame_number = 0
        self.fps = 30
        self.temp_video_file = None
        
        # Analysis results
        self.analysis_results = []
        self.current_frame_results = {}  # Store current frame analysis data
        self.output_video_writer = None
        self.save_directory = None
        self.last_save_time = 0  # Track when the last save occurred
        
        # Performance monitoring
        self.frame_times = []
        self.detection_count = 0
        
        # Fixed video quality settings (no UI selection)
        self.display_resolution = (1920, 1080)  # Full HD for display
        self.processing_resolution = (1280, 720)  # HD for processing
        self.video_interpolation = cv2.INTER_LANCZOS4  # High quality interpolation
        
        # Create UI
        self.init_ui()
        
        # Timer for stats update
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_stats)
        self.stats_timer.start(1000)  # Update every second
        
        self.last_offside_state = False  # Track previous offside state
        
        # Goal and offside indicators
        self.goal_indicator = QLabel(self)
        self.goal_indicator.setStyleSheet("background-color: transparent; color: red; font-weight: bold; font-size: 42px;")
        self.goal_indicator.setAlignment(Qt.AlignCenter)
        self.goal_indicator.setVisible(False)
        
        self.offside_indicator = QLabel(self)
        self.offside_indicator.setStyleSheet("background-color: transparent; color: red; font-weight: bold; font-size: 42px;")
        self.offside_indicator.setAlignment(Qt.AlignCenter)
        self.offside_indicator.setVisible(False)
        
        # Initialize offside detection
        self.offside_detector = OffsideDetector()
        
        # Initialize goal detector
        self.goal_detector = GoalDetector()
        
        # Initialize analytics components
        self.analytics = OffsideAnalytics()
        self.visualizer = AnalyticsVisualizer(self.analytics)
        self.insight_generator = InsightGenerator(self.analytics)
        
        # Analytics tab
        self.analytics_tab = QWidget()
        self.tabs.addTab(self.analytics_tab, "Analytics")
        self.setup_analytics_tab()
        
        # Analytics state
        self.last_offside_state = False
        self.last_goal_state = False
        self.last_frame_number = 0
        
    def init_ui(self):
        """Initialize the modern and enhanced user interface."""
        # Set professional dark theme for the entire application
        self.setStyleSheet("""
            QMainWindow, QDialog, QWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
            }
            QLabel, QCheckBox {
                color: #cdd6f4;
            }
            QPushButton {
                background-color: #45475a;
                color: #cdd6f4;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #585b70;
            }
            QPushButton:pressed {
                background-color: #313244;
            }
            QPushButton:disabled {
                background-color: #313244;
                color: #6c7086;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #313244;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #89b4fa;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background: #74c7ec;
                border-radius: 4px;
            }
            QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
                background-color: #313244;
                color: #cdd6f4;
                border-radius: 4px;
                padding: 6px;
                selection-background-color: #89b4fa;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QGroupBox {
                border: 1px solid #313244;
                border-radius: 6px;
                margin-top: 12px;
                padding: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                top: -7px;
                padding: 0 5px;
                background-color: #1e1e2e;
            }
            QTabWidget::pane {
                border: 1px solid #313244;
                border-radius: 6px;
            }
            QTabBar::tab {
                background-color: #313244;
                color: #cdd6f4;
                padding: 10px 15px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #89b4fa;
                color: #1e1e2e;
                font-weight: bold;
            }
            QTabBar::tab:hover:!selected {
                background-color: #45475a;
            }
            QProgressBar {
                border: none;
                border-radius: 4px;
                background-color: #313244;
                text-align: center;
                color: #cdd6f4;
            }
            QProgressBar::chunk {
                background-color: #a6e3a1;
                border-radius: 4px;
            }
            QScrollBar:vertical {
                border: none;
                background: #313244;
                width: 12px;
                margin: 0px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #45475a;
                border-radius: 6px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: #585b70;
            }
            QScrollBar:horizontal {
                border: none;
                background: #313244;
                height: 12px;
                margin: 0px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background: #45475a;
                border-radius: 6px;
                min-width: 30px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #585b70;
            }
        """)
        
        # Create main layout
        main_layout = QVBoxLayout()
        
        # Create header with logo and title
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background-color: #11111b;
                border-radius: 8px;
                margin: 0px 0px 10px 0px;
            }
        """)
        header.setFixedHeight(70)
        header_layout = QHBoxLayout(header)
        
        # App title
        title_label = QLabel("VAROMATIC+ Pro")
        title_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #cdd6f4;
        """)
        
        header_layout.addWidget(title_label)
        
        # Status indicator
        self.status_indicator = QLabel("Ready")
        self.status_indicator.setStyleSheet("""
            color: #a6e3a1;
            font-weight: bold;
            font-size: 16px;
            padding: 5px 10px;
            background-color: #313244;
            border-radius: 5px;
        """)
        header_layout.addWidget(self.status_indicator, alignment=Qt.AlignRight)
        
        main_layout.addWidget(header)
        
        # Create main tab widget with custom styling
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                background-color: #1e1e2e;
                border: 1px solid #313244;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        
        # Create tabs
        self.main_tab = QWidget()
        self.tabs.addTab(self.main_tab, "Video Analysis")
        
        self.settings_tab = QWidget()
        self.tabs.addTab(self.settings_tab, "Settings")
        
        self.live_tab = QWidget()
        self.tabs.addTab(self.live_tab, "Camera Mode")
        
        # Set up Video Analysis tab
        self.setup_video_analysis_tab()
        
        # Set up Settings tab
        self.setup_settings_tab()
        
        # Set up Live Mode tab
        self.setup_live_mode_tab()
        
        main_layout.addWidget(self.tabs)
        
        # Footer with app info
        footer = QFrame()
        footer.setStyleSheet("""
            QFrame {
                background-color: #11111b;
                border-radius: 8px;
                margin: 10px 0px 0px 0px;
            }
        """)
        footer.setFixedHeight(30)
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(10, 0, 10, 0)
        
        footer_label = QLabel("© VAROMATIC+ Pro 2025")
        footer_label.setStyleSheet("color: #6c7086; font-size: 12px;")
        footer_layout.addWidget(footer_label)
        
        fps_label = QLabel("FPS: 0")
        fps_label.setStyleSheet("color: #6c7086; font-size: 12px;")
        self.fps_label = fps_label
        footer_layout.addWidget(fps_label, alignment=Qt.AlignRight)
        
        main_layout.addWidget(footer)
        
        # Set central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
    def setup_video_analysis_tab(self):
        """Set up the Video Analysis tab with a more professional and organized layout."""
        layout = QVBoxLayout(self.main_tab)
        
        # Main content split into video display and controls
        content_layout = QHBoxLayout()
        
        # Video display area - left side
        video_panel = QFrame()
        video_panel.setStyleSheet("""
            QFrame {
                background-color: #11111b;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        video_layout = QVBoxLayout(video_panel)
        
        # Video display
        self.video_frame = QLabel()
        self.video_frame.setMinimumSize(800, 450)
        self.video_frame.setAlignment(Qt.AlignCenter)
        self.video_frame.setStyleSheet("""
            background-color: #000000; 
            border-radius: 8px; 
            border: 1px solid #313244;
        """)
        video_layout.addWidget(self.video_frame)
        
        # Video timeline control
        timeline_layout = QHBoxLayout()
        
        self.video_slider = QSlider(Qt.Horizontal)
        self.video_slider.setMinimum(0)
        self.video_slider.setMaximum(100)
        self.video_slider.valueChanged.connect(self.slider_changed)
        timeline_layout.addWidget(self.video_slider)
        
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet("font-family: monospace;")
        timeline_layout.addWidget(self.time_label)
        
        video_layout.addLayout(timeline_layout)
        
        # Offside and Goal indicators
        indicators_layout = QHBoxLayout()
        
        self.offside_label = QLabel("NO OFFSIDE")
        self.offside_label.setAlignment(Qt.AlignCenter)
        self.offside_label.setStyleSheet("""
            color: #cdd6f4;
            font-size: 18px;
            font-weight: bold;
            padding: 8px;
            background-color: #313244;
            border-radius: 6px;
            min-width: 150px;
        """)
        indicators_layout.addWidget(self.offside_label)
        
        indicators_layout.addStretch()
        
        self.goal_status = QLabel("NO GOAL")
        self.goal_status.setAlignment(Qt.AlignCenter)
        self.goal_status.setStyleSheet("""
            color: #cdd6f4;
            font-size: 18px;
            font-weight: bold;
            padding: 8px;
            background-color: #313244;
            border-radius: 6px;
            min-width: 150px;
        """)
        indicators_layout.addWidget(self.goal_status)
        
        video_layout.addLayout(indicators_layout)
        
        # Add playback controls
        playback_layout = QHBoxLayout()
        
        self.start_button = QPushButton("▶ Start")
        self.start_button.clicked.connect(self.start_analysis)
        self.start_button.setStyleSheet("""
            background-color: #a6e3a1;
            color: #11111b;
        """)
        playback_layout.addWidget(self.start_button)
        
        self.pause_button = QPushButton("⏸ Pause")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.pause_button.setEnabled(False)
        playback_layout.addWidget(self.pause_button)
        
        self.stop_button = QPushButton("⏹ Stop")
        self.stop_button.clicked.connect(self.stop_analysis)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("""
            background-color: #f38ba8;
            color: #11111b;
        """)
        playback_layout.addWidget(self.stop_button)
        
        video_layout.addLayout(playback_layout)
        
        content_layout.addWidget(video_panel, 7)
        
        # Controls panel - right side
        controls_panel = QFrame()
        controls_panel.setStyleSheet("""
            QFrame {
                background-color: #11111b;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        controls_panel.setMaximumWidth(350)
        controls_layout = QVBoxLayout(controls_panel)
        
        # Video source selection
        source_group = QGroupBox("Video Source")
        source_layout = QVBoxLayout()
        
        self.video_combo = QComboBox()
        self.video_combo.addItems(["Video File", "YouTube"])  # Removed Camera option
        self.video_combo.currentTextChanged.connect(self.on_source_changed)
        source_layout.addWidget(self.video_combo)
        
        # Add a note about Camera Mode
        live_mode_note = QLabel("<i>For camera input, use the 'Camera Mode' tab</i>")
        live_mode_note.setStyleSheet("color: #cba6f7; font-size: 12px;")
        source_layout.addWidget(live_mode_note)
        
        self.youtube_link = QLineEdit()
        self.youtube_link.setPlaceholderText("Enter YouTube URL")
        self.youtube_link.hide()
        source_layout.addWidget(self.youtube_link)
        
        self.file_button = QPushButton("Select Video File")
        self.file_button.clicked.connect(self.select_video)
        self.file_button.setStyleSheet("""
            background-color: #89b4fa;
            color: #11111b;
        """)
        source_layout.addWidget(self.file_button)
        
        source_group.setLayout(source_layout)
        controls_layout.addWidget(source_group)
        
        # Detection options
        detection_group = QGroupBox("Detection Options")
        detection_layout = QVBoxLayout()
        
        self.player_detection_cb = QCheckBox("Player Detection")
        self.player_detection_cb.setChecked(True)
        detection_layout.addWidget(self.player_detection_cb)
        
        self.offside_detection_cb = QCheckBox("Offside Detection")
        self.offside_detection_cb.setChecked(True)
        detection_layout.addWidget(self.offside_detection_cb)
        
        self.goal_detection_cb = QCheckBox("Goal Detection")
        self.goal_detection_cb.setChecked(True)
        detection_layout.addWidget(self.goal_detection_cb)
        
        detection_group.setLayout(detection_layout)
        controls_layout.addWidget(detection_group)
        
        # Save options
        save_group = QGroupBox("Save Options")
        save_layout = QVBoxLayout()
        
        self.save_video_cb = QCheckBox("Save Analysis Video")
        self.save_video_cb.setChecked(True)
        save_layout.addWidget(self.save_video_cb)
        
        self.save_results_cb = QCheckBox("Save Analysis Results")
        self.save_results_cb.setChecked(True)
        save_layout.addWidget(self.save_results_cb)
        
        save_group.setLayout(save_layout)
        controls_layout.addWidget(save_group)
        
        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()
        
        self.detection_label = QLabel("Players: 0")
        stats_layout.addWidget(self.detection_label)
        
        cpu_layout = QHBoxLayout()
        cpu_layout.addWidget(QLabel("CPU Usage:"))
        self.cpu_usage_bar = QProgressBar()
        self.cpu_usage_bar.setTextVisible(True)
        self.cpu_usage_bar.setFormat("%p%")
        cpu_layout.addWidget(self.cpu_usage_bar)
        stats_layout.addLayout(cpu_layout)
        
        stats_group.setLayout(stats_layout)
        controls_layout.addWidget(stats_group)
        
        # Spacer to push everything to the top
        controls_layout.addStretch()
        
        content_layout.addWidget(controls_panel, 3)
        
        layout.addLayout(content_layout)
        
    def setup_settings_tab(self):
        """Set up the Settings tab with configuration options."""
        layout = QVBoxLayout(self.settings_tab)
        
        # Create a card-like container for settings
        settings_card = QFrame()
        settings_card.setStyleSheet("""
            QFrame {
                background-color: #11111b;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        card_layout = QVBoxLayout(settings_card)
        
        # Analysis settings
        analysis_group = QGroupBox("Analysis Settings")
        analysis_layout = QVBoxLayout()
        
        # Detection confidence
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("Detection Confidence:"))
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(10)
        self.confidence_slider.setMaximum(90)
        self.confidence_slider.setValue(15)
        confidence_layout.addWidget(self.confidence_slider)
        self.confidence_value = QLabel("15%")
        confidence_layout.addWidget(self.confidence_value)
        analysis_layout.addLayout(confidence_layout)
        
        analysis_group.setLayout(analysis_layout)
        card_layout.addWidget(analysis_group)
        
        # Advanced settings
        advanced_group = QGroupBox("Advanced Settings")
        advanced_layout = QVBoxLayout()
        
        # Data storage location
        storage_layout = QHBoxLayout()
        storage_layout.addWidget(QLabel("Data Storage Location:"))
        self.storage_path = QLineEdit()
        self.storage_path.setText("./analysis_results")
        storage_layout.addWidget(self.storage_path)
        self.browse_btn = QPushButton("Browse")
        storage_layout.addWidget(self.browse_btn)
        advanced_layout.addLayout(storage_layout)
        
        # Auto-save interval
        autosave_layout = QHBoxLayout()
        autosave_layout.addWidget(QLabel("Auto-save Interval (seconds):"))
        self.autosave_spin = QSpinBox()
        self.autosave_spin.setMinimum(5)
        self.autosave_spin.setMaximum(120)
        self.autosave_spin.setValue(30)
        autosave_layout.addWidget(self.autosave_spin)
        advanced_layout.addLayout(autosave_layout)
        
        advanced_group.setLayout(advanced_layout)
        card_layout.addWidget(advanced_group)
        
        # About section
        about_group = QGroupBox("About VAROMATIC+ Pro")
        about_layout = QVBoxLayout()
        
        about_text = QLabel(
            """VAROMATIC+ Pro is an advanced soccer analysis tool featuring:
            
• Real-time player detection and tracking
• Offside detection with high accuracy
• Goal detection technology
• Advanced analytics and visualization
• AI-powered insights
            
For support, contact support@varomatic.com"""
        )
        about_text.setWordWrap(True)
        about_text.setStyleSheet("color: #cdd6f4; font-size: 14px;")
        about_layout.addWidget(about_text)
        
        about_group.setLayout(about_layout)
        card_layout.addWidget(about_group)
        
        # Save settings button
        save_settings_btn = QPushButton("Save Settings")
        save_settings_btn.setStyleSheet("""
            background-color: #89b4fa;
            color: #11111b;
            padding: 10px;
            font-weight: bold;
            font-size: 14px;
        """)
        card_layout.addWidget(save_settings_btn)
        
        layout.addWidget(settings_card)
        
    def setup_live_mode_tab(self):
        """Set up the Live Mode tab for real-time camera analysis."""
        layout = QVBoxLayout(self.live_tab)
        
        # Add a header banner explaining live mode
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background-color: #313244;
                border-radius: 8px;
                padding: 10px;
                margin-bottom: 10px;
            }
        """)
        header_layout = QHBoxLayout(header)
        
        # Add camera icon
        camera_icon = QLabel("📹")
        camera_icon.setStyleSheet("font-size: 24px;")
        header_layout.addWidget(camera_icon)
        
        # Add explanation text
        explanation = QLabel(
            "<b>Live Camera Mode</b><br>"
            "Use this tab to analyze soccer footage directly from your camera in real-time."
        )
        explanation.setWordWrap(True)
        header_layout.addWidget(explanation, 1)
        
        layout.addWidget(header)
        
        # Create split view for live mode
        split_layout = QHBoxLayout()
        
        # Live video feed panel
        live_panel = QFrame()
        live_panel.setStyleSheet("""
            QFrame {
                background-color: #11111b;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        live_layout = QVBoxLayout(live_panel)
        
        # Live feed display
        live_title = QLabel("Live Camera Feed")
        live_title.setAlignment(Qt.AlignCenter)
        live_title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        live_layout.addWidget(live_title)
        
        self.live_frame = QLabel()
        self.live_frame.setMinimumSize(640, 360)
        self.live_frame.setAlignment(Qt.AlignCenter)
        self.live_frame.setStyleSheet("""
            background-color: #000000; 
            border-radius: 8px; 
            border: 1px solid #313244;
        """)
        live_layout.addWidget(self.live_frame)
        
        # Live controls
        live_controls = QHBoxLayout()
        
        self.start_live_btn = QPushButton("▶ Start Live Analysis")
        self.start_live_btn.setStyleSheet("""
            background-color: #a6e3a1;
            color: #11111b;
        """)
        self.start_live_btn.clicked.connect(self.start_live_analysis)
        live_controls.addWidget(self.start_live_btn)
        
        self.stop_live_btn = QPushButton("⏹ Stop")
        self.stop_live_btn.setEnabled(False)
        self.stop_live_btn.setStyleSheet("""
            background-color: #f38ba8;
            color: #11111b;
        """)
        self.stop_live_btn.clicked.connect(self.stop_live_analysis)
        live_controls.addWidget(self.stop_live_btn)
        
        live_layout.addLayout(live_controls)
        
        split_layout.addWidget(live_panel)
        
        # Real-time analytics panel
        analytics_panel = QFrame()
        analytics_panel.setStyleSheet("""
            QFrame {
                background-color: #11111b;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        analytics_layout = QVBoxLayout(analytics_panel)
        
        # Real-time stats title
        rt_title = QLabel("Real-time Analytics")
        rt_title.setAlignment(Qt.AlignCenter)
        rt_title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        analytics_layout.addWidget(rt_title)
        
        # Current stats display
        self.rt_stats = QLabel("No data available")
        self.rt_stats.setWordWrap(True)
        self.rt_stats.setStyleSheet("""
            background-color: #181825;
            border-radius: 6px;
            padding: 10px;
            min-height: 200px;
        """)
        analytics_layout.addWidget(self.rt_stats)
        
        # Current offside stats
        offside_stats_group = QGroupBox("Offside Statistics")
        offside_stats_layout = QVBoxLayout()
        
        self.offside_count = QLabel("Total Offsides: 0")
        offside_stats_layout.addWidget(self.offside_count)
        
        self.team_stats = QLabel("Team 1: 0 | Team 2: 0")
        offside_stats_layout.addWidget(self.team_stats)
        
        self.last_offside = QLabel("Last Offside: None")
        offside_stats_layout.addWidget(self.last_offside)
        
        offside_stats_group.setLayout(offside_stats_layout)
        analytics_layout.addWidget(offside_stats_group)
        
        # Export live session button
        self.export_live_btn = QPushButton("Export Live Session Data")
        self.export_live_btn.setStyleSheet("""
            background-color: #89b4fa;
            color: #11111b;
        """)
        self.export_live_btn.clicked.connect(self.export_live_data)
        analytics_layout.addWidget(self.export_live_btn)
        
        split_layout.addWidget(analytics_panel)
        
        layout.addLayout(split_layout)
        
    def setup_analytics_tab(self):
        """Setup the analytics tab with a modern and professional layout"""
        layout = QVBoxLayout(self.analytics_tab)
        
        # Create top control bar
        controls_frame = QFrame()
        controls_frame.setStyleSheet("""
            QFrame {
                background-color: #11111b;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        controls_layout = QHBoxLayout(controls_frame)
        
        # Generate analytics button
        self.generate_analytics_btn = QPushButton("Generate Analytics")
        self.generate_analytics_btn.clicked.connect(self.generate_analytics)
        controls_layout.addWidget(self.generate_analytics_btn)
        
        # Export report button
        self.export_report_btn = QPushButton("Export PDF Report")
        self.export_report_btn.clicked.connect(self.export_analytics_report)
        controls_layout.addWidget(self.export_report_btn)
        
        # Reset analytics button
        self.reset_analytics_btn = QPushButton("Reset Analytics")
        self.reset_analytics_btn.clicked.connect(self.reset_analytics)
        controls_layout.addWidget(self.reset_analytics_btn)
        
        controls_frame.setLayout(controls_layout)
        layout.addWidget(controls_frame)
        
        # Analytics visualization options
        viz_panel = QFrame()
        viz_panel.setStyleSheet("""
            QFrame {
                background-color: #11111b;
                border-radius: 10px;
                padding: 10px;
                margin-top: 10px;
            }
        """)
        viz_layout = QVBoxLayout(viz_panel)
        
        # Visualization options title
        viz_title = QLabel("Visualization Options")
        viz_title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        viz_layout.addWidget(viz_title)
        
        # Checkboxes grid layout for better organization
        checkboxes_layout = QGridLayout()
        
        # Create all checkboxes with consistent styling
        checkbox_style = """
            QCheckBox {
                font-size: 14px;
                spacing: 10px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
            }
            QCheckBox::indicator:unchecked {
                border: 2px solid #7f849c;
                background: #313244;
            }
            QCheckBox::indicator:checked {
                border: 2px solid #89b4fa;
                background: #89b4fa;
            }
        """
        
        # Player chart checkbox
        self.player_chart_cb = QCheckBox("Player Offside Chart")
        self.player_chart_cb.setChecked(True)
        self.player_chart_cb.setStyleSheet(checkbox_style)
        checkboxes_layout.addWidget(self.player_chart_cb, 0, 0)
        
        # Zone behavior checkbox
        self.zone_behavior_cb = QCheckBox("Zone Behavior")
        self.zone_behavior_cb.setChecked(True)
        self.zone_behavior_cb.setStyleSheet(checkbox_style)
        checkboxes_layout.addWidget(self.zone_behavior_cb, 0, 1)
        
        # Timeline checkbox
        self.timeline_cb = QCheckBox("Event Timeline")
        self.timeline_cb.setChecked(True)
        self.timeline_cb.setStyleSheet(checkbox_style)
        checkboxes_layout.addWidget(self.timeline_cb, 1, 0)
        
        # Team comparison checkbox
        self.team_comparison_cb = QCheckBox("Team Comparison")
        self.team_comparison_cb.setChecked(True)
        self.team_comparison_cb.setStyleSheet(checkbox_style)
        checkboxes_layout.addWidget(self.team_comparison_cb, 1, 1)
        
        # Dashboard checkbox
        self.dashboard_cb = QCheckBox("Complete Dashboard")
        self.dashboard_cb.setChecked(True)
        self.dashboard_cb.setStyleSheet(checkbox_style)
        checkboxes_layout.addWidget(self.dashboard_cb, 1, 2)
        
        viz_layout.addLayout(checkboxes_layout)
        layout.addWidget(viz_panel)
        
        # Create AI Insights panel
        insights_panel = QFrame()
        insights_panel.setStyleSheet("""
            QFrame {
                background-color: #11111b;
                border-radius: 10px;
                padding: 10px;
                margin-top: 10px;
            }
        """)
        insights_layout = QVBoxLayout(insights_panel)
        
        # Insights title and controls
        insights_header = QHBoxLayout()
        
        insights_title = QLabel("AI Insights")
        insights_title.setStyleSheet("font-size: 16px; font-weight: bold;")
        insights_header.addWidget(insights_title)
        
        insights_header.addWidget(QLabel("Number of insights:"))
        
        self.insights_count = QSpinBox()
        self.insights_count.setMinimum(1)
        self.insights_count.setMaximum(10)
        self.insights_count.setValue(5)
        self.insights_count.setFixedWidth(60)
        insights_header.addWidget(self.insights_count)
        
        self.generate_insights_btn = QPushButton("Generate Insights")
        self.generate_insights_btn.clicked.connect(self.show_ai_insights)
        self.generate_insights_btn.setStyleSheet("""
            background-color: #cba6f7;
            color: #11111b;
            padding: 5px 10px;
            font-weight: bold;
        """)
        insights_header.addWidget(self.generate_insights_btn)
        
        insights_layout.addLayout(insights_header)
        
        # Insights display area
        self.insights_display = QLabel("No insights available yet. Process video and generate insights.")
        self.insights_display.setWordWrap(True)
        self.insights_display.setStyleSheet("""
            background-color: #181825;
            color: #cdd6f4;
            border-radius: 6px;
            padding: 15px;
            font-size: 14px;
            min-height: 200px;
        """)
        self.insights_display.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        insights_layout.addWidget(self.insights_display)
        
        layout.addWidget(insights_panel)
        
        # Status bar
        status_frame = QFrame()
        status_frame.setStyleSheet("""
            QFrame {
                background-color: #313244;
                border-radius: 6px;
                padding: 8px;
                margin-top: 10px;
            }
        """)
        status_layout = QHBoxLayout(status_frame)
        
        self.analytics_status = QLabel("No analytics data generated yet")
        self.analytics_status.setStyleSheet("color: #cdd6f4;")
        status_layout.addWidget(self.analytics_status)
        
        layout.addWidget(status_frame)
        
    def generate_analytics(self):
        """Generate all analytics visualizations based on current data"""
        if not self.analytics.offside_events:
            # First try to generate sample events
            if not self.generate_sample_events():
                QMessageBox.warning(self, "No Data", 
                                  "No offside events recorded. Process video first to collect data.")
                return
            
        visualizations = []
        
        # Generate selected visualizations
        try:
            self.visualizer.generate_offside_heatmap()
            visualizations.append("Heatmap")
        except Exception as e:
            print(f"Error generating heatmap: {str(e)}")
        
        if self.player_chart_cb.isChecked():
            self.visualizer.generate_player_offside_chart()
            visualizations.append("Player Chart")
            
        if self.zone_behavior_cb.isChecked():
            self.visualizer.generate_zone_team_behavior()
            visualizations.append("Zone Behavior")
            
        if self.timeline_cb.isChecked():
            self.visualizer.generate_match_timeline()
            visualizations.append("Timeline")
            
        if self.team_comparison_cb.isChecked():
            self.visualizer.generate_team_comparison()
            visualizations.append("Team Comparison")
            
        if self.dashboard_cb.isChecked():
            self.visualizer.generate_dashboard()
            visualizations.append("Dashboard")
            
        # Add efficiency chart
        try:
            self.visualizer.generate_efficiency_chart()
            visualizations.append("Efficiency Analysis")
        except Exception as e:
            print(f"Error generating efficiency chart: {str(e)}")
            
        # Generate AI insights
        insights = self.insight_generator.get_top_insights(max_insights=7)
        
        # Update insights display
        if insights:
            insight_html = "<h3>AI Insights</h3><ul>"
            for insight in insights:
                insight_html += f"<li>{insight}</li>"
            insight_html += "</ul>"
            self.insights_display.setText(insight_html)
        else:
            self.insights_display.setText("<p>No significant insights generated.</p>")
            
        # Save analytics data to JSON
        self.analytics.save_analytics()
        
        # Show success message
        QMessageBox.information(self, "Analytics Generated", 
                             f"Generated visualizations: {', '.join(visualizations)}\n\n"
                             f"All visualizations saved to:\n{self.visualizer.viz_dir}")
        
        # Update status
        total_events = len(self.analytics.offside_events) + len(self.analytics.goal_events) + len(self.analytics.key_pass_events)
        self.analytics_status.setText(
            f"Analytics generated with {len(self.analytics.offside_events)} offside events, "
            f"{len(self.analytics.goal_events)} goals, and {len(self.analytics.key_pass_events)} key passes. "
            f"Total events: {total_events}"
        )
        
    def export_analytics_report(self):
        """Export analytics report as PDF"""
        if not self.analytics.offside_events:
            QMessageBox.warning(self, "No Data", 
                              "No offside events recorded. Process video first to collect data.")
            return
            
        try:
            pdf_path = self.visualizer.export_report_pdf()
            if pdf_path:
                QMessageBox.information(self, "Report Exported", 
                                     f"PDF Report exported successfully to:\n{pdf_path}")
            else:
                QMessageBox.warning(self, "Export Failed", 
                                 "Failed to export PDF. Make sure ReportLab is installed.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", 
                              f"Error exporting PDF report:\n{str(e)}")
            
    def reset_analytics(self):
        """Reset all analytics data"""
        reply = QMessageBox.question(self, "Reset Analytics", 
                                  "Are you sure you want to reset all analytics data?",
                                  QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.analytics.reset_analytics()
            self.analytics_status.setText("Analytics data reset. No data available.")
            self.insights_display.setText("No insights available yet. Process video and generate insights.")
            QMessageBox.information(self, "Reset Complete", "All analytics data has been reset.")
            
    def show_ai_insights(self):
        """Show AI-generated insights from collected data"""
        if not self.analytics.offside_events:
            QMessageBox.warning(self, "No Data", 
                              "No offside events recorded. Process video first to collect data.")
            return
            
        # Get insights
        max_insights = self.insights_count.value()
        insights = self.insight_generator.get_top_insights(max_insights)
        
        if not insights:
            self.insights_display.setText("No significant insights found. Try collecting more data.")
            return
            
        # Format insights as HTML for better display
        insights_html = "<ul>"
        for insight in insights:
            insights_html += f"<li>{insight}</li>"
        insights_html += "</ul>"
        
        # Add recommendation
        recommendation = self.insight_generator.generate_recommendation()
        insights_html += f"<p><b>Key Recommendation:</b> {recommendation}</p>"
        
        # Set formatted text
        self.insights_display.setText(insights_html)
        
        # Update status
        self.analytics_status.setText(
            f"Generated {len(insights)} insights from {len(self.analytics.offside_events)} events"
        )

    def start_live_analysis(self):
        """Start live camera feed analysis"""
        # Open default camera (usually webcam)
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open camera. Please check your camera connection.")
            return
            
        # Setup UI
        self.start_live_btn.setEnabled(False)
        self.stop_live_btn.setEnabled(True)
        self.status_indicator.setText("Live Analysis Running")
        self.status_indicator.setStyleSheet("""
            color: #a6e3a1;
            font-weight: bold;
            font-size: 16px;
            padding: 5px 10px;
            background-color: #313244;
            border-radius: 5px;
        """)
        
        # Reset analytics data for this session
        self.analytics.reset_analytics()
        
        # Start processing frames
        self.is_running = True
        self.is_paused = False
        self.process_live_frame()
        
    def stop_live_analysis(self):
        """Stop live camera feed analysis"""
        self.is_running = False
        
        if self.cap:
            self.cap.release()
            
        # Reset UI
        self.start_live_btn.setEnabled(True)
        self.stop_live_btn.setEnabled(False)
        self.status_indicator.setText("Live Analysis Stopped")
        self.status_indicator.setStyleSheet("""
            color: #f38ba8;
            font-weight: bold;
            font-size: 16px;
            padding: 5px 10px;
            background-color: #313244;
            border-radius: 5px;
        """)
        
    def process_live_frame(self):
        """Process next frame from camera for live analysis"""
        if not self.is_running or not self.cap or not self.cap.isOpened():
            return
            
        # Read frame
        ret, frame = self.cap.read()
        if not ret:
            self.stop_live_analysis()
            QMessageBox.warning(self, "Camera Error", "Failed to capture frame from camera.")
            return
        
        # Keep original resolution for display
        original_frame = frame.copy()
        
        # Start timing frame processing
        start_time = time.time()
        
        # Resize for processing based on quality setting
        proc_width, proc_height = self.processing_resolution
        processing_frame = cv2.resize(frame, (proc_width, proc_height), interpolation=self.video_interpolation)
        
        # Detect players
        boxes, kept_indices, keypoints, player_roles = self.player_detector.detect_players(processing_frame)
        
        # Update detection count
        self.detection_count = len(kept_indices)
        
        # Get team colors
        if boxes and kept_indices:
            # Get actual team colors if players detected
            team1_color, team2_color = self.color_analyzer.get_team_colors(
                processing_frame, boxes, kept_indices)
                
            # Get team visual colors
            visual_team1_color, visual_team2_color = self.color_analyzer.get_team_display_colors()
            
            # Draw detections on processing frame
            self.player_detector.draw_detections(
                processing_frame, boxes, kept_indices, 
                keypoints, [visual_team1_color, visual_team2_color], player_roles)
            
            # Generate team assignments
            team_assignments = [0] * len(boxes)  # Default all to team 0
            try:
                # Create player objects
                player_objects = [
                    Player(
                        position=(boxes[i][0] + boxes[i][2]/2, boxes[i][1] + boxes[i][3]/2),
                        box=boxes[i],
                        team=0,  # temporary
                        keypoints=keypoints[i] if keypoints and i < len(keypoints) else None
                    ) for i in kept_indices
                ]
                
                # Assign teams
                team_assignments = self.offside_detector._assign_teams_with_clustering(player_objects)
                
                # Check for offside
                is_offside, offside_frame = self.offside_detector.detect_offside(
                    processing_frame.copy(), boxes, team_assignments, keypoints)
                
                # Update offside indicator
                self.update_offside_indicator(is_offside)
                
                if is_offside:
                    # Use the frame with offside markings
                    processing_frame = offside_frame
                    
                    # Record offside event
                    offside_timestamp = time.time()
                    offside_player_id = None
                    offside_position = (0, 0)
                    offside_team = 0
                    
                    # Get offside player info
                    if hasattr(self.offside_detector, 'offside_players') and self.offside_detector.offside_players:
                        offside_player = self.offside_detector.offside_players[0]
                        offside_team = offside_player.team if hasattr(offside_player, 'team') else 0
                        if hasattr(offside_player, 'position'):
                            offside_position = offside_player.position
                        
                        # Try to find player ID
                        for i, idx in enumerate(kept_indices):
                            if boxes[idx] == offside_player.box:
                                offside_player_id = i
                                break
                    
                    # Create offside event
                    offside_event = OffsideEvent(
                        timestamp=offside_timestamp,
                        frame_number=0,  # No frame number in live mode
                        player_id=offside_player_id,
                        player_position=offside_position,
                        offside_line_position=0,
                        team=offside_team,
                        match_time=self.format_time(offside_timestamp - self.start_time if hasattr(self, 'start_time') else 0)
                    )
                    
                    # Add to analytics
                    self.analytics.add_offside_event(offside_event)
                    
                    # Update stats UI
                    total_offsides = len(self.analytics.offside_events)
                    self.offside_count.setText(f"Total Offsides: {total_offsides}")
                    
                    team0 = self.analytics.get_team_offsides(0)
                    team1 = self.analytics.get_team_offsides(1)
                    self.team_stats.setText(f"Team 1: {team0} | Team 2: {team1}")
                    
                    # Update last offside info
                    self.last_offside.setText(f"Last Offside: {offside_event.match_time}")
                
            except Exception as e:
                print(f"Error in live offside detection: {str(e)}")
        
        # Update the live frame display
        h, w, ch = processing_frame.shape
        rgb_frame = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.live_frame.setPixmap(pixmap.scaled(
            self.live_frame.width(), 
            self.live_frame.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))
        
        # Update stats
        frame_time = time.time() - start_time
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
            
        # Update real-time stats display
        self.update_live_stats()
        
        # Schedule next frame
        QTimer.singleShot(1, self.process_live_frame)
        
    def update_live_stats(self):
        """Update real-time statistics display"""
        if not hasattr(self, 'analytics') or not hasattr(self.analytics, 'offside_events'):
            return
            
        # Calculate FPS
        fps = 0
        if self.frame_times:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            fps = 1 / avg_time if avg_time > 0 else 0
            
        # Update real-time stats
        stats_text = f"""
        <b>Processing Statistics:</b><br>
        FPS: {fps:.1f}<br>
        Players Detected: {self.detection_count}<br>
        <br>
        <b>Offside Summary:</b><br>
        Total Offsides: {len(self.analytics.offside_events)}<br>
        Team 1 Offsides: {self.analytics.get_team_offsides(0)}<br>
        Team 2 Offsides: {self.analytics.get_team_offsides(1)}<br>
        """
        
        self.rt_stats.setText(stats_text)
        
    def export_live_data(self):
        """Export live session data to file"""
        if not hasattr(self, 'analytics') or not self.analytics.offside_events:
            QMessageBox.warning(self, "No Data", "No offside events recorded yet.")
            return
            
        try:
            # Save analytics data
            self.analytics.save_analytics()
            
            # Generate visualizations
            if hasattr(self, 'visualizer'):
                self.visualizer.generate_match_timeline()
                self.visualizer.generate_team_comparison()
                
            QMessageBox.information(self, "Export Complete", 
                                 f"Live session data exported to:\n{self.analytics.output_dir}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting data: {str(e)}")

    def slider_changed(self):
        """Handle slider value change for video navigation."""
        if self.cap and not self.is_running:
            # Get frame position from slider
            frame_pos = self.video_slider.value()
            
            # Ensure the frame position is valid
            if frame_pos >= 0 and frame_pos < self.frame_count:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                
                # Read frame and display it
                ret, frame = self.cap.read()
                if ret:
                    # Process frame for display
                    # Resize for processing based on quality setting
                    proc_width, proc_height = self.processing_resolution
                    processing_frame = cv2.resize(frame, (proc_width, proc_height), interpolation=self.video_interpolation)
                    self.display_frame(processing_frame)
                
                # Update time display
                current_time = frame_pos / self.fps if self.fps else 0
                total_time = self.frame_count / self.fps if self.fps else 0
                self.time_label.setText(
                    f"{int(current_time//60):02d}:{int(current_time%60):02d} / "
                    f"{int(total_time//60):02d}:{int(total_time%60):02d}"
                )
                
    def format_time(self, seconds: float) -> str:
        """Format seconds into MM:SS format for match time"""
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes:02d}:{remaining_seconds:02d}"
        
    def update_offside_indicator(self, is_offside):
        """Update the offside indicator with enhanced modern visuals."""
        if is_offside:
            self.offside_label.setText("OFFSIDE!")
            self.offside_label.setStyleSheet("""
                color: white;
                font-size: 18px;
                font-weight: bold;
                padding: 8px;
                background-color: #f38ba8;
                border-radius: 6px;
                min-width: 150px;
            """)
        else:
            self.offside_label.setText("NO OFFSIDE")
            self.offside_label.setStyleSheet("""
                color: #cdd6f4;
                font-size: 18px;
                font-weight: bold;
                padding: 8px;
                background-color: #313244;
                border-radius: 6px;
                min-width: 150px;
            """)
            
    def display_frame(self, frame):
        """Display the frame in the UI with proper scaling and high quality."""
        try:
            if frame is None:
                print("Error: Frame is None")
                return
            
            # Convert frame to RGB for Qt
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            
            # Get the actual size of the video frame label
            label_size = self.video_frame.size()
            
            # Calculate scaling while maintaining aspect ratio
            scale = min(label_size.width() / w, label_size.height() / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # High-quality resizing
            rgb_frame = cv2.resize(rgb_frame, (new_w, new_h), interpolation=self.video_interpolation)
            
            # Convert to QImage with no quality loss
            bytes_per_line = ch * new_w
            qt_image = QImage(rgb_frame.data, new_w, new_h, bytes_per_line, QImage.Format.Format_RGB888)
            
            # Create QPixmap and set it - preserve quality
            pixmap = QPixmap.fromImage(qt_image)
            
            # Set the pixmap with proper alignment
            self.video_frame.setPixmap(pixmap)
            self.video_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            # Force update
            self.video_frame.update()
            QApplication.processEvents()
            
        except Exception as e:
            print(f"Error displaying frame: {str(e)}")

    def on_source_changed(self, source):
        """Handle video source change."""
        self.youtube_link.setVisible(source == "YouTube")
        if source == "YouTube":
            self.file_button.setText("Download YouTube Video")
        else:
            self.file_button.setText("Select Video File")
        self.file_button.setVisible(True)
        
    def select_video(self):
        """Open file dialog to select video file or process YouTube URL."""
        if self.video_combo.currentText() == "YouTube":
            url = self.youtube_link.text().strip()
            
            if not url:
                QMessageBox.warning(self, "Error", "Please enter a YouTube URL!")
                return
                
            try:
                # Show immediate feedback
                self.status_indicator.setText("Starting YouTube download...")
                self.status_indicator.setStyleSheet("color: #FFC107; font-weight: bold; font-size: 16px; padding: 5px 10px; background-color: #313244; border-radius: 5px;")
                QApplication.processEvents()
                
                # Create temporary file
                temp_dir = tempfile.gettempdir()
                temp_video = os.path.join(temp_dir, 'temp_video.mp4')
                
                # Configure yt-dlp options
                ydl_opts = {
                    'format': 'best[ext=mp4]',
                    'outtmpl': temp_video,
                    'progress_hooks': [self._download_progress_hook],
                }
                
                # Download video
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                
                # Set video path and get info
                self.video_path = temp_video
                self.temp_video_file = temp_video
                
                # Get video info
                cap = cv2.VideoCapture(self.video_path)
                self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps = int(cap.get(cv2.CAP_PROP_FPS))
                cap.release()
                
                # Update UI
                self.video_slider.setRange(0, self.frame_count - 1)
                duration = self.frame_count / self.fps
                self.time_label.setText(f"00:00 / {int(duration//60):02d}:{int(duration%60):02d}")
                self.status_indicator.setText("YouTube video loaded successfully")
                self.status_indicator.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 16px; padding: 5px 10px; background-color: #313244; border-radius: 5px;")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not load YouTube video:\n{str(e)}")
                self.status_indicator.setText("Error loading YouTube video")
                self.status_indicator.setStyleSheet("color: #F44336; font-weight: bold; font-size: 16px; padding: 5px 10px; background-color: #313244; border-radius: 5px;")
                return
        else:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Select Video File",
                "",
                "Video Files (*.mp4 *.avi *.mkv);;All Files (*)"
            )
            
            if file_name:
                self.video_path = file_name
                    
                # Get video info
                cap = cv2.VideoCapture(self.video_path)
                self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps = int(cap.get(cv2.CAP_PROP_FPS))
                cap.release()
                
                # Update UI
                self.video_slider.setRange(0, self.frame_count - 1)
                duration = self.frame_count / self.fps
                self.time_label.setText(f"00:00 / {int(duration//60):02d}:{int(duration%60):02d}")
                self.status_indicator.setText(f"Loaded: {os.path.basename(file_name)}")
                self.status_indicator.setStyleSheet("color: #a6e3a1; font-weight: bold; font-size: 16px; padding: 5px 10px; background-color: #313244; border-radius: 5px;")
                
    def _download_progress_hook(self, d):
        """Update download progress in UI."""
        if d['status'] == 'downloading':
            try:
                percent = d.get('_percent_str', '0%').strip()
                speed = d.get('_speed_str', 'N/A')
                eta = d.get('_eta_str', 'N/A')
                self.status_indicator.setText(f"Downloading: {percent} (Speed: {speed}, ETA: {eta})")
                QApplication.processEvents()
            except Exception as e:
                self.status_indicator.setText(f"Downloading... (Error updating progress: {str(e)})")
                QApplication.processEvents()
        elif d['status'] == 'finished':
            self.status_indicator.setText("Download finished, processing video...")
            QApplication.processEvents()
    
    def start_analysis(self):
        """Start video analysis."""
        if not self.video_path:
            QMessageBox.warning(self, "Error", "Please select a video file or YouTube URL first!")
            return
            
        # Setup save directory
        if self.save_video_cb.isChecked() or self.save_results_cb.isChecked():
            self.save_directory = self.create_save_directory()
            if not self.save_directory:
                return
            
        # Setup video capture
        self.cap = cv2.VideoCapture(self.video_path)
            
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open video source!")
            return
            
        # Setup video writer if saving is enabled
        if self.save_video_cb.isChecked():
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.output_video_writer = cv2.VideoWriter(
                os.path.join(self.save_directory, 'analyzed_video.mp4'),
                cv2.VideoWriter_fourcc(*'mp4v'),
                self.fps,
                (width, height)
            )
            
        # Clear previous results
        self.analysis_results = []
        self.current_frame_results = {}  # Store current frame analysis data
        self.detection_count = 0
        self.frame_times = []
        self.current_frame_number = 0
        
        # Reset analytics data for this session
        self.analytics.reset_analytics()
            
        # Update UI state
        self.is_running = True
        self.is_paused = False
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.video_slider.setEnabled(True)
        self.status_indicator.setText("Analysis running...")
        self.status_indicator.setStyleSheet("""
            color: #a6e3a1;
            font-weight: bold;
            font-size: 16px;
            padding: 5px 10px;
            background-color: #313244;
            border-radius: 5px;
        """)
        
        # Start processing
        self.process_next_frame()
        
    def create_save_directory(self) -> str:
        """Create a directory to save analysis results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = "camera_feed"
        
        if self.video_path:
            video_name = os.path.splitext(os.path.basename(self.video_path))[0]
            
        # Make sure we create the parent directory first
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_results")
        try:
            os.makedirs(base_dir, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not create analysis_results directory: {str(e)}")
            return None
            
        # Now create the specific save directory
        save_dir = os.path.join(base_dir, f"{video_name}_{timestamp}")
                               
        try:
            os.makedirs(save_dir, exist_ok=True)
            print(f"Created save directory: {save_dir}")  # Debug output
            return save_dir
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not create save directory: {str(e)}")
            return None
            
    def process_next_frame(self):
        """Process the next video frame with enhanced detection."""
        if not self.cap or not self.cap.isOpened() or self.is_paused or not self.is_running:
            return
            
        # Read frame
        ret, frame = self.cap.read()
        if not ret:
            print("End of video reached")
            # Stop playback when video ends
            self.stop_analysis()
            self.status_indicator.setText("Video playback completed")
            return
            
        # Keep original resolution for saving
        original_frame = frame.copy()
        
        # Update progress slider and time display
        self.current_frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.video_slider.setValue(self.current_frame_number)
        
        # Update time display
        current_time = self.current_frame_number / self.fps
        total_time = self.frame_count / self.fps
        self.time_label.setText(
            f"{int(current_time//60):02d}:{int(current_time%60):02d} / "
            f"{int(total_time//60):02d}:{int(total_time%60):02d}"
        )
            
        # Start timing frame processing
        start_time = time.time()
        
        # Resize for processing based on quality setting
        proc_width, proc_height = self.processing_resolution
        processing_frame = cv2.resize(frame, (proc_width, proc_height), interpolation=self.video_interpolation)
        
        # Detect players if enabled
        if self.player_detection_cb.isChecked():
            boxes, kept_indices, keypoints, player_roles = self.player_detector.detect_players(processing_frame)
            
            # Update detection count
            self.detection_count = len(kept_indices)
            
            # Get team colors
            if boxes and kept_indices:
                # Get actual team colors if players detected
                team1_color, team2_color = self.color_analyzer.get_team_colors(
                    processing_frame, boxes, kept_indices)
                    
                # Get team visual colors
                visual_team1_color, visual_team2_color = self.color_analyzer.get_team_display_colors()
                
                # Draw detections on processing frame
                self.player_detector.draw_detections(
                    processing_frame, boxes, kept_indices, 
                    keypoints, [visual_team1_color, visual_team2_color], player_roles)
                
                # Only check for offside if offside detection is enabled
                if self.offside_detection_cb.isChecked() and len(kept_indices) > 1:
                    try:
                        # Create player objects
                        player_objects = [
                            Player(
                                position=(boxes[i][0] + boxes[i][2]/2, boxes[i][1] + boxes[i][3]/2),
                                box=boxes[i],
                                team=0,  # temporary
                                keypoints=keypoints[i] if keypoints and i < len(keypoints) else None
                            ) for i in kept_indices
                        ]
                        
                        # Assign teams
                        team_assignments = self.offside_detector._assign_teams_with_clustering(player_objects)
                        
                        # Check for offside
                        is_offside, offside_frame = self.offside_detector.detect_offside(
                            processing_frame.copy(), boxes, team_assignments, keypoints)
                        
                        # Update offside indicator
                        self.update_offside_indicator(is_offside)
                        
                        if is_offside:
                            # Use the frame with offside markings
                            processing_frame = offside_frame
                            
                            # Get offside player info
                            offside_timestamp = self.current_frame_number / self.fps if self.fps else 0
                            offside_player_id = None
                            offside_position = (0, 0)
                            offside_team = 0
                            
                            if hasattr(self.offside_detector, 'offside_players') and self.offside_detector.offside_players:
                                offside_player = self.offside_detector.offside_players[0]
                                offside_team = offside_player.team if hasattr(offside_player, 'team') else 0
                                if hasattr(offside_player, 'position'):
                                    offside_position = offside_player.position
                                
                                # Try to find player ID
                                for i, idx in enumerate(kept_indices):
                                    if boxes[idx] == offside_player.box:
                                        offside_player_id = i
                                        break
                            
                            # Only record offside events when state transitions to True
                            if not self.last_offside_state:
                                # Create offside event
                                offside_event = OffsideEvent(
                                    timestamp=offside_timestamp,
                                    frame_number=self.current_frame_number,
                                    player_id=offside_player_id,
                                    player_position=offside_position,
                                    offside_line_position=0,
                                    team=offside_team,
                                    match_time=self.format_time(offside_timestamp)
                                )
                                
                                # Add to analytics
                                self.analytics.add_offside_event(offside_event)
                                print(f"Recorded offside event at frame {self.current_frame_number}")
                        
                        # Update last offside state
                        self.last_offside_state = is_offside
                        
                    except Exception as e:
                        print(f"Error in offside detection: {str(e)}")
            
            # Check for goal if goal detection is enabled
            if self.goal_detection_cb.isChecked():
                try:
                    # Detect ball
                    ball = self.goal_detector.detect_ball(processing_frame)
                    
                    if ball is not None and hasattr(ball, 'position'):
                        # Check for goal
                        is_goal, goal_annotated = self.goal_detector.detect_goal(processing_frame, ball)
                        
                        if is_goal:
                            # Use the goal-annotated frame
                            processing_frame = goal_annotated
                            
                            # Update goal indicator
                            self.update_goal_indicator(True)
                            
                            # Add goal event to analytics when goal is detected
                            goal_timestamp = self.current_frame_number / self.fps if self.fps else 0
                            
                            # Only record goal events when state transitions to True
                            if not self.last_goal_state:
                                self.analytics.add_goal_event(
                                    frame_number=self.current_frame_number,
                                    timestamp=goal_timestamp,
                                    team=0,  # Default team
                                    position=ball.position if hasattr(ball, 'position') else (0, 0),
                                    match_time=self.format_time(goal_timestamp)
                                )
                                print(f"Recorded goal event at frame {self.current_frame_number}")
                        else:
                            # Reset goal indicator
                            self.update_goal_indicator(False)
                            
                        # Update last goal state
                        self.last_goal_state = is_goal
                        
                except Exception as e:
                    print(f"Error in goal detection: {str(e)}")
        
        # Display the processed frame
        self.display_frame(processing_frame)
        
        # Save the original frame with overlay if video writer is enabled
        if self.output_video_writer is not None:
            try:
                # Resize processing frame to original size
                overlay = cv2.resize(processing_frame, (original_frame.shape[1], original_frame.shape[0]), 
                                   interpolation=cv2.INTER_LANCZOS4)
                
                # Create a blended overlay
                blended = cv2.addWeighted(original_frame, 0.5, overlay, 0.5, 0)
                
                # Write frame
                self.output_video_writer.write(blended)
            except Exception as e:
                print(f"Error writing video frame: {str(e)}")
        
        # Calculate frame processing time
        frame_time = time.time() - start_time
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        # Update performance stats
        self.update_stats()
        
        # Schedule next frame
        QTimer.singleShot(1, self.process_next_frame)
        
    def stop_analysis(self):
        """Stop video analysis."""
        self.is_running = False
        self.is_paused = False
        
        if self.cap:
            self.cap.release()
            
        if self.output_video_writer:
            self.output_video_writer.release()
            self.output_video_writer = None
            
        # Reset UI state
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.video_slider.setEnabled(False)
        self.video_frame.clear()
        self.status_indicator.setText("Analysis stopped")
        self.status_indicator.setStyleSheet("""
            color: #f38ba8;
            font-weight: bold;
            font-size: 16px;
            padding: 5px 10px;
            background-color: #313244;
            border-radius: 5px;
        """)
        self.update_offside_indicator(False)
        self.update_goal_indicator(False)
        
    def update_goal_indicator(self, is_goal):
        """Update the goal indicator with modern visual effects."""
        if is_goal:
            # Create a more visually exciting goal notification
            self.goal_status.setText("GOAL!!!")
            self.goal_status.setStyleSheet("""
                color: white;
                font-size: 18px;
                font-weight: bold;
                padding: 8px;
                background-color: #fab387;
                border-radius: 6px;
                min-width: 150px;
            """)
        else:
            self.goal_status.setText("NO GOAL")
            self.goal_status.setStyleSheet("""
                color: #cdd6f4;
                font-size: 18px;
                font-weight: bold;
                padding: 8px;
                background-color: #313244;
                border-radius: 6px;
                min-width: 150px;
            """)
            
    def toggle_pause(self):
        """Toggle video pause state."""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_button.setText("⏵ Resume")
            self.status_indicator.setText("Analysis paused")
            self.status_indicator.setStyleSheet("""
                color: #FFC107;
                font-weight: bold;
                font-size: 16px;
                padding: 5px 10px;
                background-color: #313244;
                border-radius: 5px;
            """)
        else:
            self.pause_button.setText("⏸ Pause")
            self.status_indicator.setText("Analysis running...")
            self.status_indicator.setStyleSheet("""
                color: #a6e3a1;
                font-weight: bold;
                font-size: 16px;
                padding: 5px 10px;
                background-color: #313244;
                border-radius: 5px;
            """)
            self.process_next_frame()
        
    def update_stats(self):
        """Update performance statistics."""
        if self.frame_times:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            fps = 1 / avg_time if avg_time > 0 else 0
            self.fps_label.setText(f"FPS: {fps:.1f}")
            
        self.detection_label.setText(f"Players: {self.detection_count}")
        
        # Update CPU usage bar
        self.cpu_usage_bar.setValue(min(100, self.detection_count * 5))

    def generate_sample_events(self):
        """Generate sample offside events for testing analytics features"""
        # Only generate if no events exist yet
        if self.analytics.offside_events:
            return True  # Already have events
            
        # Create sample events with varied data
        sample_events = [
            # Team 1 offsides
            OffsideEvent(
                timestamp=15.2,
                frame_number=456,
                player_id=2,
                player_position=(80, 25),
                offside_line_position=78,
                team=0,
                zone="attacking_right",
                match_time="00:15"
            ),
            OffsideEvent(
                timestamp=45.6,
                frame_number=1368,
                player_id=3,
                player_position=(82, 34),
                offside_line_position=80,
                team=0,
                zone="attacking_center",
                match_time="00:45"
            ),
            OffsideEvent(
                timestamp=68.4,
                frame_number=2052,
                player_id=2,
                player_position=(85, 28),
                offside_line_position=83,
                team=0,
                zone="attacking_center",
                match_time="01:08"
            ),
            OffsideEvent(
                timestamp=92.7,
                frame_number=2781,
                player_id=4,
                player_position=(78, 20),
                offside_line_position=76,
                team=0,
                zone="attacking_right",
                match_time="01:32"
            ),
            OffsideEvent(
                timestamp=110.3,
                frame_number=3309,
                player_id=2,
                player_position=(83, 32),
                offside_line_position=81,
                team=0,
                zone="attacking_center",
                match_time="01:50"
            ),
            OffsideEvent(
                timestamp=150.8,
                frame_number=4524,
                player_id=5,
                player_position=(81, 45),
                offside_line_position=79,
                team=0,
                zone="attacking_right",
                match_time="02:30"
            ),
            
            # Team 2 offsides
            OffsideEvent(
                timestamp=32.5,
                frame_number=975,
                player_id=7,
                player_position=(25, 30),
                offside_line_position=27,
                team=1,
                zone="attacking_left",
                match_time="00:32"
            ),
            OffsideEvent(
                timestamp=58.2,
                frame_number=1746,
                player_id=9,
                player_position=(22, 42),
                offside_line_position=24,
                team=1,
                zone="attacking_center",
                match_time="00:58"
            ),
            OffsideEvent(
                timestamp=75.6,
                frame_number=2268,
                player_id=9,
                player_position=(24, 38),
                offside_line_position=26,
                team=1,
                zone="attacking_center",
                match_time="01:15"
            ),
            OffsideEvent(
                timestamp=125.1,
                frame_number=3753,
                player_id=8,
                player_position=(26, 15),
                offside_line_position=28,
                team=1,
                zone="attacking_left",
                match_time="02:05"
            ),
        ]
        
        # Add goal events
        self.analytics.add_goal_event(
            frame_number=1200,
            timestamp=40.0,
            team=0,
            position=(90, 34),
            match_time="00:40"
        )
        
        self.analytics.add_goal_event(
            frame_number=2400,
            timestamp=80.0,
            team=1,
            position=(15, 34),
            match_time="01:20"
        )
        
        self.analytics.add_goal_event(
            frame_number=3600,
            timestamp=120.0,
            team=0,
            position=(88, 40),
            match_time="02:00"
        )
        
        # Add key pass events
        self.analytics.add_key_pass_event(
            frame_number=900,
            timestamp=30.0,
            team=0,
            position=(70, 30),
            match_time="00:30"
        )
        
        self.analytics.add_key_pass_event(
            frame_number=1170,
            timestamp=39.0,
            team=0,
            position=(75, 35),
            match_time="00:39"
        )
        
        self.analytics.add_key_pass_event(
            frame_number=1800,
            timestamp=60.0,
            team=1,
            position=(35, 40),
            match_time="01:00"
        )
        
        self.analytics.add_key_pass_event(
            frame_number=2370,
            timestamp=79.0,
            team=1,
            position=(30, 30),
            match_time="01:19"
        )
        
        self.analytics.add_key_pass_event(
            frame_number=3570,
            timestamp=119.0,
            team=0,
            position=(65, 45),
            match_time="01:59"
        )
        
        # Add clustered offsides (for testing pattern detection)
        # Team 1 offside cluster
        for i in range(3):
            self.analytics.add_offside_event(
                OffsideEvent(
                    timestamp=180.0 + i * 20,
                    frame_number=5400 + i * 600,
                    player_id=3,
                    player_position=(84, 30 + i * 3),
                    offside_line_position=82,
                    team=0,
                    match_time=f"0{3 + i}:00"
                )
            )
        
        # Team 2 offside cluster after conceding
        for i in range(2):
            self.analytics.add_offside_event(
                OffsideEvent(
                    timestamp=125.0 + i * 15,
                    frame_number=3750 + i * 450,
                    player_id=7,
                    player_position=(23, 25 + i * 5),
                    offside_line_position=25,
                    team=1,
                    match_time=f"0{2 + i}:{10 + i * 5}"
                )
            )
        
        # Add all events to analytics
        for event in sample_events:
            self.analytics.add_offside_event(event)
            
        print(f"Added {len(sample_events) + 5} sample offside events and {len(self.analytics.goal_events)} goals for testing")
        return True

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 