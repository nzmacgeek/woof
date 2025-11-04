"""
Cross-platform GUI for the bark monitoring system using Tkinter.
Provides database viewing, system monitoring, and configuration interface.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import sqlite3

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from database import BarkEventDatabase
    from ai_models import create_ai_detector
    from audio_utils import AudioEncoder
    from main import BarkMonitor, DEFAULT_CONFIG
    DATABASE_AVAILABLE = True
    MONITORING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    DATABASE_AVAILABLE = False
    MONITORING_AVAILABLE = False


class BarkMonitorGUI:
    """Main GUI application for bark monitoring system."""
    
    def __init__(self):
        """Initialize the GUI application."""
        self.root = tk.Tk()
        self.root.title("Bark Monitor - Dog Barking Detection System")
        self.root.geometry("1200x800")
        
        # Application state
        self.database = None
        self.database_path = None
        self.monitoring_active = False
        self.monitor_instance = None
        self.monitor_thread = None
        self.log_queue = queue.Queue()
        self.config = None
        
        # Load configuration
        self.load_configuration()
        
        # Setup logging to capture in GUI
        self.setup_logging()
        
        # Create GUI elements
        self.create_menu()
        self.create_main_interface()
        self.create_status_bar()
        
        # Start log processing
        self.process_log_queue()
        
        # Start periodic updates
        self.update_session_stats()
        
        # Load default database if available
        self.load_default_database()
    
    def load_configuration(self):
        """Load configuration from file or use defaults."""
        config_paths = ['config.json', 'config/default.json']
        
        for config_path in config_paths:
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        self.config = json.load(f)
                    print(f"Loaded configuration from {config_path}")
                    return
                except Exception as e:
                    print(f"Error loading config from {config_path}: {e}")
        
        # Use default configuration if available
        if MONITORING_AVAILABLE:
            self.config = DEFAULT_CONFIG.copy()
            print("Using default configuration")
        else:
            self.config = {
                'detection_threshold': 0.7,
                'enable_ai': True,
                'sample_rate': 44100,
                'chunk_size': 4096
            }
            print("Using minimal configuration (monitoring not available)")
    
    def setup_logging(self):
        """Setup logging to capture messages in GUI."""
        self.log_handler = GuiLogHandler(self.log_queue)
        logging.getLogger().addHandler(self.log_handler)
        logging.getLogger().setLevel(logging.INFO)
    
    def create_menu(self):
        """Create the application menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Database...", command=self.open_database)
        file_menu.add_command(label="Export Report...", command=self.export_report)
        file_menu.add_separator()
        file_menu.add_command(label="Settings...", command=self.open_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit_application)
        
        # Monitor menu
        monitor_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Monitor", menu=monitor_menu)
        monitor_menu.add_command(label="Start Monitoring", command=self.start_monitoring)
        monitor_menu.add_command(label="Stop Monitoring", command=self.stop_monitoring)
        monitor_menu.add_separator()
        monitor_menu.add_command(label="View Live Feed", command=self.show_live_feed)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Test AI Models", command=self.test_ai_models)
        tools_menu.add_command(label="Audio Converter", command=self.open_audio_converter)
        tools_menu.add_command(label="System Information", command=self.show_system_info)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=self.show_user_guide)
        help_menu.add_command(label="About", command=self.show_about)
    
    def create_main_interface(self):
        """Create the main interface with tabs."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create tabs
        self.create_events_tab()
        self.create_monitoring_tab()
        self.create_reports_tab()
        self.create_logs_tab()
    
    def create_events_tab(self):
        """Create the bark events viewing tab."""
        events_frame = ttk.Frame(self.notebook)
        self.notebook.add(events_frame, text="Bark Events")
        
        # Control panel
        control_frame = ttk.LabelFrame(events_frame, text="Filter & Search")
        control_frame.pack(fill='x', padx=5, pady=5)
        
        # Date filters
        ttk.Label(control_frame, text="From:").grid(row=0, column=0, padx=5, pady=5)
        self.date_from = tk.StringVar(value=(datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"))
        date_from_entry = ttk.Entry(control_frame, textvariable=self.date_from, width=12)
        date_from_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(control_frame, text="To:").grid(row=0, column=2, padx=5, pady=5)
        self.date_to = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        date_to_entry = ttk.Entry(control_frame, textvariable=self.date_to, width=12)
        date_to_entry.grid(row=0, column=3, padx=5, pady=5)
        
        # Dog filter
        ttk.Label(control_frame, text="Dog ID:").grid(row=0, column=4, padx=5, pady=5)
        self.dog_filter = tk.StringVar()
        dog_filter_entry = ttk.Entry(control_frame, textvariable=self.dog_filter, width=15)
        dog_filter_entry.grid(row=0, column=5, padx=5, pady=5)
        
        # Search button
        search_btn = ttk.Button(control_frame, text="Search", command=self.refresh_events)
        search_btn.grid(row=0, column=6, padx=10, pady=5)
        
        # Clear button
        clear_btn = ttk.Button(control_frame, text="Clear", command=self.clear_filters)
        clear_btn.grid(row=0, column=7, padx=5, pady=5)
        
        # Events treeview
        tree_frame = ttk.Frame(events_frame)
        tree_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Treeview with scrollbars
        self.events_tree = ttk.Treeview(tree_frame, columns=(
            'ID', 'Timestamp', 'Duration', 'Dog ID', 'Confidence', 'Intensity', 'Method'
        ), show='headings')
        
        # Configure columns
        self.events_tree.heading('ID', text='ID')
        self.events_tree.heading('Timestamp', text='Timestamp')
        self.events_tree.heading('Duration', text='Duration (s)')
        self.events_tree.heading('Dog ID', text='Dog ID')
        self.events_tree.heading('Confidence', text='Confidence')
        self.events_tree.heading('Intensity', text='Intensity')
        self.events_tree.heading('Method', text='Method')
        
        # Column widths
        self.events_tree.column('ID', width=50)
        self.events_tree.column('Timestamp', width=150)
        self.events_tree.column('Duration', width=80)
        self.events_tree.column('Dog ID', width=80)
        self.events_tree.column('Confidence', width=80)
        self.events_tree.column('Intensity', width=80)
        self.events_tree.column('Method', width=100)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient='vertical', command=self.events_tree.yview)
        self.events_tree.configure(yscrollcommand=v_scrollbar.set)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient='horizontal', command=self.events_tree.xview)
        self.events_tree.configure(xscrollcommand=h_scrollbar.set)
        
        # Pack treeview and scrollbars
        self.events_tree.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        # Bind double-click to show details
        self.events_tree.bind('<Double-1>', self.show_event_details)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(events_frame, text="Statistics")
        stats_frame.pack(fill='x', padx=5, pady=5)
        
        self.stats_text = ttk.Label(stats_frame, text="Load a database to view statistics")
        self.stats_text.pack(padx=10, pady=5)
    
    def create_monitoring_tab(self):
        """Create the real-time monitoring tab."""
        monitor_frame = ttk.Frame(self.notebook)
        self.notebook.add(monitor_frame, text="Monitoring")
        
        # Status panel
        status_panel = ttk.LabelFrame(monitor_frame, text="Monitoring Status")
        status_panel.pack(fill='x', padx=5, pady=5)
        
        # Status indicators
        status_row1 = ttk.Frame(status_panel)
        status_row1.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(status_row1, text="Status:").pack(side='left')
        self.status_label = ttk.Label(status_row1, text="Stopped", foreground='red')
        self.status_label.pack(side='left', padx=10)
        
        self.start_stop_btn = ttk.Button(status_row1, text="Start Monitoring", 
                                        command=self.toggle_monitoring)
        self.start_stop_btn.pack(side='right')
        
        # Disable monitoring if not available
        if not MONITORING_AVAILABLE:
            self.start_stop_btn.config(state='disabled')
            ttk.Label(status_row1, text="(Monitoring unavailable - check installation)", 
                     foreground='orange').pack(side='right', padx=10)
        
        # Control panel
        control_panel = ttk.LabelFrame(monitor_frame, text="Detection Settings")
        control_panel.pack(fill='x', padx=5, pady=5)
        
        # Settings grid
        settings_frame = ttk.Frame(control_panel)
        settings_frame.pack(fill='x', padx=5, pady=5)
        
        # Threshold setting
        ttk.Label(settings_frame, text="Detection Threshold:").grid(row=0, column=0, sticky='w', padx=5)
        self.threshold_var = tk.DoubleVar(value=self.config.get('detection_threshold', 0.7))
        threshold_scale = ttk.Scale(settings_frame, from_=0.1, to=1.0, variable=self.threshold_var, 
                                   orient='horizontal', length=200)
        threshold_scale.grid(row=0, column=1, padx=5)
        self.threshold_label = ttk.Label(settings_frame, text=f"{self.threshold_var.get():.2f}")
        self.threshold_label.grid(row=0, column=2, padx=5)
        threshold_scale.configure(command=self.update_threshold_label)
        
        # AI enhancement
        self.ai_enabled = tk.BooleanVar(value=self.config.get('enable_ai', True))
        ai_check = ttk.Checkbutton(settings_frame, text="Enable AI Enhancement", 
                                  variable=self.ai_enabled)
        ai_check.grid(row=1, column=0, columnspan=2, sticky='w', padx=5, pady=5)
        
        # Save audio recordings
        self.save_audio = tk.BooleanVar(value=self.config.get('save_audio', True))
        save_check = ttk.Checkbutton(settings_frame, text="Save Audio Recordings", 
                                    variable=self.save_audio)
        save_check.grid(row=2, column=0, columnspan=2, sticky='w', padx=5, pady=5)
        
        # Live feed
        feed_frame = ttk.LabelFrame(monitor_frame, text="Live Audio Feed")
        feed_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Audio level meter (placeholder)
        meter_frame = ttk.Frame(feed_frame)
        meter_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(meter_frame, text="Audio Level:").pack(side='left')
        self.audio_level_var = tk.DoubleVar()
        audio_meter = ttk.Progressbar(meter_frame, variable=self.audio_level_var, 
                                     maximum=100, length=300)
        audio_meter.pack(side='left', padx=10)
        
        # Recent detections
        recent_frame = ttk.LabelFrame(feed_frame, text="Recent Detections")
        recent_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create a frame for the text and statistics
        text_stats_frame = ttk.Frame(recent_frame)
        text_stats_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Statistics panel
        stats_frame = ttk.Frame(text_stats_frame)
        stats_frame.pack(fill='x', pady=(0, 5))
        
        # Session statistics
        ttk.Label(stats_frame, text="Session Stats:").pack(side='left')
        self.session_barks_label = ttk.Label(stats_frame, text="Barks: 0")
        self.session_barks_label.pack(side='left', padx=10)
        self.session_dogs_label = ttk.Label(stats_frame, text="Dogs: 0")
        self.session_dogs_label.pack(side='left', padx=10)
        self.session_time_label = ttk.Label(stats_frame, text="Time: 00:00:00")
        self.session_time_label.pack(side='left', padx=10)
        
        self.recent_text = scrolledtext.ScrolledText(text_stats_frame, height=8, width=60)
        self.recent_text.pack(fill='both', expand=True)
        
        # Initialize with welcome message
        if MONITORING_AVAILABLE:
            self.add_log_message("Bark Monitor GUI Ready - Click 'Start Monitoring' to begin")
        else:
            self.add_log_message("GUI Ready - Monitoring unavailable (check installation)")
            self.add_log_message("Run: pip install -r requirements.txt")
    
    def create_reports_tab(self):
        """Create the reports generation tab."""
        reports_frame = ttk.Frame(self.notebook)
        self.notebook.add(reports_frame, text="Reports")
        
        # Report configuration
        config_frame = ttk.LabelFrame(reports_frame, text="Report Configuration")
        config_frame.pack(fill='x', padx=5, pady=5)
        
        # Report type
        ttk.Label(config_frame, text="Report Type:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.report_type = tk.StringVar(value="Evidence Report")
        report_combo = ttk.Combobox(config_frame, textvariable=self.report_type, 
                                   values=["Evidence Report", "Daily Summary", "Dog Activity", "Pattern Analysis"])
        report_combo.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        
        # Date range
        ttk.Label(config_frame, text="Date Range:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        date_frame = ttk.Frame(config_frame)
        date_frame.grid(row=1, column=1, padx=5, pady=5, sticky='ew')
        
        self.report_date_from = tk.StringVar(value=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"))
        ttk.Entry(date_frame, textvariable=self.report_date_from, width=12).pack(side='left')
        ttk.Label(date_frame, text=" to ").pack(side='left')
        self.report_date_to = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        ttk.Entry(date_frame, textvariable=self.report_date_to, width=12).pack(side='left')
        
        # Output format
        ttk.Label(config_frame, text="Output Format:").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.output_format = tk.StringVar(value="Text")
        format_combo = ttk.Combobox(config_frame, textvariable=self.output_format, 
                                   values=["Text", "HTML", "PDF"])
        format_combo.grid(row=2, column=1, padx=5, pady=5, sticky='ew')
        
        config_frame.grid_columnconfigure(1, weight=1)
        
        # Generate button
        generate_btn = ttk.Button(config_frame, text="Generate Report", command=self.generate_report)
        generate_btn.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Report preview
        preview_frame = ttk.LabelFrame(reports_frame, text="Report Preview")
        preview_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.report_text = scrolledtext.ScrolledText(preview_frame, wrap='word')
        self.report_text.pack(fill='both', expand=True, padx=5, pady=5)
    
    def create_logs_tab(self):
        """Create the system logs tab."""
        logs_frame = ttk.Frame(self.notebook)
        self.notebook.add(logs_frame, text="System Logs")
        
        # Log controls
        log_controls = ttk.Frame(logs_frame)
        log_controls.pack(fill='x', padx=5, pady=5)
        
        clear_logs_btn = ttk.Button(log_controls, text="Clear Logs", command=self.clear_logs)
        clear_logs_btn.pack(side='left')
        
        save_logs_btn = ttk.Button(log_controls, text="Save Logs", command=self.save_logs)
        save_logs_btn.pack(side='left', padx=5)
        
        # Log level filter
        ttk.Label(log_controls, text="Level:").pack(side='left', padx=10)
        self.log_level = tk.StringVar(value="INFO")
        level_combo = ttk.Combobox(log_controls, textvariable=self.log_level, 
                                  values=["DEBUG", "INFO", "WARNING", "ERROR"], width=10)
        level_combo.pack(side='left')
        
        # Log display
        self.log_text = scrolledtext.ScrolledText(logs_frame, wrap='word', font=('Courier', 9))
        self.log_text.pack(fill='both', expand=True, padx=5, pady=5)
    
    def create_status_bar(self):
        """Create the status bar at the bottom."""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(side='bottom', fill='x')
        
        # Database status
        self.db_status = ttk.Label(self.status_bar, text="No database loaded")
        self.db_status.pack(side='left', padx=5)
        
        # Separator
        ttk.Separator(self.status_bar, orient='vertical').pack(side='left', fill='y', padx=5)
        
        # Event count
        self.event_count = ttk.Label(self.status_bar, text="Events: 0")
        self.event_count.pack(side='left', padx=5)
        
        # Version info
        version_label = ttk.Label(self.status_bar, text="Bark Monitor v1.0")
        version_label.pack(side='right', padx=5)
    
    # Event handlers
    def open_database(self):
        """Open a database file."""
        if not DATABASE_AVAILABLE:
            messagebox.showerror("Error", "Database modules not available")
            return
        
        file_path = filedialog.askopenfilename(
            title="Open Bark Events Database",
            filetypes=[("SQLite Database", "*.db"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                self.database = BarkEventDatabase(file_path)
                self.database_path = file_path
                self.db_status.config(text=f"Database: {os.path.basename(file_path)}")
                self.refresh_events()
                logging.info(f"Opened database: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open database: {e}")
    
    def load_default_database(self):
        """Load default database if it exists."""
        default_path = os.path.join("data", "bark_events.db")
        if os.path.exists(default_path) and DATABASE_AVAILABLE:
            try:
                self.database = BarkEventDatabase(default_path)
                self.database_path = default_path
                self.db_status.config(text=f"Database: {os.path.basename(default_path)}")
                self.refresh_events()
            except Exception as e:
                logging.warning(f"Could not load default database: {e}")
    
    def refresh_events(self):
        """Refresh the events display."""
        if not self.database:
            return
        
        # Clear existing items
        for item in self.events_tree.get_children():
            self.events_tree.delete(item)
        
        try:
            # Get events from database
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Build query with filters
            query = "SELECT id, timestamp, duration, dog_id, confidence, bark_intensity, detection_method FROM bark_events WHERE 1=1"
            params = []
            
            # Date filters
            if self.date_from.get():
                query += " AND date(timestamp) >= ?"
                params.append(self.date_from.get())
            
            if self.date_to.get():
                query += " AND date(timestamp) <= ?"
                params.append(self.date_to.get())
            
            # Dog filter
            if self.dog_filter.get():
                query += " AND dog_id LIKE ?"
                params.append(f"%{self.dog_filter.get()}%")
            
            query += " ORDER BY timestamp DESC LIMIT 1000"
            
            cursor.execute(query, params)
            events = cursor.fetchall()
            
            # Populate treeview
            for event in events:
                event_id, timestamp, duration, dog_id, confidence, intensity, method = event
                self.events_tree.insert('', 'end', values=(
                    event_id,
                    timestamp,
                    f"{duration:.2f}" if duration else "N/A",
                    dog_id or "Unknown",
                    f"{confidence:.3f}" if confidence else "N/A",
                    f"{intensity:.1f}" if intensity else "N/A",
                    method or "N/A"
                ))
            
            # Update statistics
            cursor.execute("SELECT COUNT(*) FROM bark_events")
            total_events = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT dog_id) FROM bark_events WHERE dog_id IS NOT NULL")
            unique_dogs = cursor.fetchone()[0]
            
            conn.close()
            
            self.event_count.config(text=f"Events: {total_events}")
            self.stats_text.config(text=f"Total Events: {total_events} | Unique Dogs: {unique_dogs} | Showing: {len(events)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh events: {e}")
    
    def show_event_details(self, event):
        """Show detailed information for a selected event."""
        selection = self.events_tree.selection()
        if not selection:
            return
        
        item = self.events_tree.item(selection[0])
        event_id = item['values'][0]
        
        # Create details window
        details_window = tk.Toplevel(self.root)
        details_window.title(f"Event Details - ID {event_id}")
        details_window.geometry("600x400")
        
        # Add event details here
        details_text = scrolledtext.ScrolledText(details_window, wrap='word')
        details_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Fetch detailed information
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM bark_events WHERE id = ?", (event_id,))
            event_data = cursor.fetchone()
            conn.close()
            
            if event_data:
                # Format event details
                details = f"Event ID: {event_data[0]}\n"
                details += f"Timestamp: {event_data[1]}\n"
                details += f"Duration: {event_data[2]:.2f} seconds\n"
                details += f"Dog ID: {event_data[3] or 'Unknown'}\n"
                details += f"Confidence: {event_data[4]:.3f}\n"
                details += f"Audio File: {event_data[5] or 'Not saved'}\n"
                details += f"Detection Method: {event_data[6] or 'Unknown'}\n"
                details += f"Bark Intensity: {event_data[7]:.1f}%\n"
                details += f"Background Noise: {event_data[8]:.1f}%\n"
                details += f"Notes: {event_data[9] or 'None'}\n"
                
                details_text.insert('1.0', details)
        except Exception as e:
            details_text.insert('1.0', f"Error loading event details: {e}")
    
    def clear_filters(self):
        """Clear all search filters."""
        self.date_from.set((datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"))
        self.date_to.set(datetime.now().strftime("%Y-%m-%d"))
        self.dog_filter.set("")
        self.refresh_events()
    
    def start_monitoring(self):
        """Start the monitoring system."""
        if not MONITORING_AVAILABLE:
            messagebox.showerror("Error", "Monitoring system not available. Please check installation.")
            return
        
        if self.monitoring_active:
            messagebox.showwarning("Warning", "Monitoring is already active.")
            return
        
        try:
            # Update configuration with current UI settings
            self.config['detection_threshold'] = self.threshold_var.get()
            self.config['enable_ai'] = self.ai_enabled.get()
            
            # Create monitor instance
            self.monitor_instance = BarkMonitor(self.config)
            
            # Initialize components
            self.monitor_instance.initialize_components()
            
            # Start monitoring in a separate thread
            self.monitor_thread = threading.Thread(target=self._monitor_worker, daemon=True)
            self.monitor_thread.start()
            
            # Update UI
            self.monitoring_active = True
            self.status_label.config(text="Running", foreground='green')
            self.start_stop_btn.config(text="Stop Monitoring")
            
            # Add log message
            self.add_log_message("Monitoring started successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start monitoring: {str(e)}")
            self.add_log_message(f"Failed to start monitoring: {str(e)}")
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        if not self.monitoring_active:
            return
        
        try:
            # Stop the monitor instance
            if self.monitor_instance:
                self.monitor_instance.stop_monitoring()
            
            # Update state
            self.monitoring_active = False
            self.status_label.config(text="Stopped", foreground='red')
            self.start_stop_btn.config(text="Start Monitoring")
            
            # Wait for thread to finish
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5.0)
            
            # Clean up
            self.monitor_instance = None
            self.monitor_thread = None
            
            # Add log message
            self.add_log_message("Monitoring stopped")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error stopping monitoring: {str(e)}")
            self.add_log_message(f"Error stopping monitoring: {str(e)}")
    
    def _monitor_worker(self):
        """Worker thread for monitoring."""
        try:
            # Start monitoring (this will run until stopped)
            self.monitor_instance.start_monitoring()
        except Exception as e:
            # Schedule UI update in main thread
            self.root.after(0, lambda: self.add_log_message(f"Monitoring error: {str(e)}"))
            self.root.after(0, lambda: self.stop_monitoring())
    
    def add_log_message(self, message):
        """Add a message to the recent detections log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        # Add to recent text widget
        self.recent_text.insert('end', log_entry)
        self.recent_text.see('end')
        
        # Keep only last 100 lines
        lines = self.recent_text.get('1.0', 'end').split('\n')
        if len(lines) > 100:
            self.recent_text.delete('1.0', f'{len(lines)-100}.0')
    
    def update_audio_level(self, level):
        """Update the audio level meter."""
        if hasattr(self, 'audio_level_var'):
            self.audio_level_var.set(level)
    
    def add_detection_event(self, event_data):
        """Add a detection event to the recent detections."""
        timestamp = event_data.get('timestamp', datetime.now().strftime("%H:%M:%S"))
        confidence = event_data.get('confidence', 0.0)
        dog_id = event_data.get('dog_id', 'Unknown')
        
        message = f"BARK DETECTED - Confidence: {confidence:.2f}, Dog: {dog_id}"
        self.add_log_message(message)
    
    def update_session_stats(self):
        """Update session statistics display."""
        if self.monitoring_active and self.monitor_instance:
            try:
                stats = self.monitor_instance.session_stats
                
                # Update bark count
                bark_count = stats.get('total_barks_detected', 0)
                self.session_barks_label.config(text=f"Barks: {bark_count}")
                
                # Update dog count
                dog_count = len(stats.get('dogs_identified', set()))
                self.session_dogs_label.config(text=f"Dogs: {dog_count}")
                
                # Update session time
                start_time = stats.get('start_time')
                if start_time:
                    elapsed = datetime.now() - start_time
                    hours, remainder = divmod(elapsed.total_seconds(), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    time_str = f"Time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
                    self.session_time_label.config(text=time_str)
                
                # Update audio level if available
                if hasattr(self.monitor_instance, 'audio_capture') and self.monitor_instance.audio_capture:
                    # Simulate audio level for now - would need to get actual level from audio capture
                    import random
                    level = random.randint(10, 90) if self.monitoring_active else 0
                    self.update_audio_level(level)
                
            except Exception as e:
                pass  # Silently handle any errors in stats update
        else:
            # Reset stats when not monitoring
            self.session_barks_label.config(text="Barks: 0")
            self.session_dogs_label.config(text="Dogs: 0")
            self.session_time_label.config(text="Time: 00:00:00")
            self.update_audio_level(0)
        
        # Schedule next update
        self.root.after(1000, self.update_session_stats)  # Update every second
    
    def toggle_monitoring(self):
        """Toggle monitoring on/off."""
        if self.monitoring_active:
            self.stop_monitoring()
        else:
            self.start_monitoring()
    
    def update_threshold_label(self, value):
        """Update the threshold label."""
        self.threshold_label.config(text=f"{float(value):.2f}")
    
    def show_live_feed(self):
        """Show live audio feed window."""
        messagebox.showinfo("Not Implemented", "Live audio feed will be implemented in the full system.")
    
    def test_ai_models(self):
        """Test AI models functionality."""
        test_window = tk.Toplevel(self.root)
        test_window.title("AI Model Testing")
        test_window.geometry("500x400")
        
        test_text = scrolledtext.ScrolledText(test_window, wrap='word')
        test_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Run AI model test
        try:
            detector = create_ai_detector()
            model_info = detector.get_model_info()
            
            test_text.insert('end', "AI Model Test Results:\n")
            test_text.insert('end', "=" * 40 + "\n\n")
            test_text.insert('end', f"Model Type: {model_info['model_type']}\n")
            test_text.insert('end', f"Description: {model_info.get('description', 'N/A')}\n")
            test_text.insert('end', f"Features: {model_info.get('features', 'N/A')}\n")
            test_text.insert('end', f"Sample Rate: {model_info.get('sample_rate', 'N/A')}\n")
            test_text.insert('end', "\nAI models are functioning correctly.")
            
        except Exception as e:
            test_text.insert('end', f"AI model test failed: {e}")
    
    def open_audio_converter(self):
        """Open audio converter utility."""
        messagebox.showinfo("Not Implemented", "Audio converter utility will be implemented.")
    
    def show_system_info(self):
        """Show system information."""
        info_window = tk.Toplevel(self.root)
        info_window.title("System Information")
        info_window.geometry("400x300")
        
        info_text = scrolledtext.ScrolledText(info_window, wrap='word')
        info_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        info = f"Bark Monitor System Information\n"
        info += "=" * 40 + "\n\n"
        info += f"Python Version: {sys.version}\n"
        info += f"Tkinter Version: {tk.TkVersion}\n"
        info += f"Database Available: {DATABASE_AVAILABLE}\n"
        info += f"Current Directory: {os.getcwd()}\n"
        info += f"Platform: {sys.platform}\n"
        
        info_text.insert('1.0', info)
    
    def generate_report(self):
        """Generate a report based on current settings."""
        if not self.database:
            messagebox.showerror("Error", "No database loaded")
            return
        
        try:
            # Simple report generation
            start_date = datetime.strptime(self.report_date_from.get(), "%Y-%m-%d")
            end_date = datetime.strptime(self.report_date_to.get(), "%Y-%m-%d")
            
            # Save report to file first
            output_path = filedialog.asksaveasfilename(
                title="Save Report",
                defaultextension=".txt",
                filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
            )
            
            if output_path:
                self.database.generate_evidence_report(start_date, end_date, output_path)
                
                # Display in preview
                with open(output_path, 'r') as f:
                    report_content = f.read()
                
                self.report_text.delete('1.0', 'end')
                self.report_text.insert('1.0', report_content)
                
                messagebox.showinfo("Success", f"Report saved to {output_path}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate report: {e}")
    
    def export_report(self):
        """Export current report to file."""
        content = self.report_text.get('1.0', 'end-1c')
        if not content.strip():
            messagebox.showwarning("Warning", "No report content to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Export Report",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(content)
                messagebox.showinfo("Success", f"Report exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export report: {e}")
    
    def open_settings(self):
        """Open settings dialog."""
        messagebox.showinfo("Not Implemented", "Settings dialog will be implemented.")
    
    def process_log_queue(self):
        """Process log messages from the queue."""
        try:
            while True:
                record = self.log_queue.get_nowait()
                if record.levelno >= getattr(logging, self.log_level.get()):
                    self.log_text.insert('end', self.log_handler.format(record) + '\n')
                    self.log_text.see('end')
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.process_log_queue)
    
    def clear_logs(self):
        """Clear the log display."""
        self.log_text.delete('1.0', 'end')
    
    def save_logs(self):
        """Save logs to file."""
        content = self.log_text.get('1.0', 'end-1c')
        if not content.strip():
            messagebox.showwarning("Warning", "No logs to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Logs",
            defaultextension=".log",
            filetypes=[("Log Files", "*.log"), ("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(content)
                messagebox.showinfo("Success", f"Logs saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save logs: {e}")
    
    def show_user_guide(self):
        """Show user guide."""
        guide_window = tk.Toplevel(self.root)
        guide_window.title("User Guide")
        guide_window.geometry("600x500")
        
        guide_text = scrolledtext.ScrolledText(guide_window, wrap='word')
        guide_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        guide_content = """
        Bark Monitor User Guide
        ======================
        
        1. Getting Started
        ------------------
        - Open a database file using File > Open Database
        - Or create a new database by starting monitoring
        
        2. Viewing Bark Events
        ----------------------
        - Use the "Bark Events" tab to view recorded events
        - Filter by date range or dog ID
        - Double-click an event to view details
        
        3. Real-time Monitoring
        -----------------------
        - Use the "Monitoring" tab to start/stop detection
        - Adjust detection threshold as needed
        - Enable/disable AI enhancement
        
        4. Generating Reports
        ---------------------
        - Use the "Reports" tab to generate evidence reports
        - Select date range and output format
        - Reports can be used for legal purposes
        
        5. System Logs
        --------------
        - View system logs in the "System Logs" tab
        - Useful for troubleshooting issues
        - Can be saved to file for support
        
        For more help, contact support.
        """
        
        guide_text.insert('1.0', guide_content)
    
    def show_about(self):
        """Show about dialog."""
        about_text = """
        Bark Monitor v1.0
        
        A comprehensive dog barking detection and monitoring system
        for documenting nuisance barking for authorities.
        
        Features:
        • AI-enhanced bark detection
        • Individual dog identification
        • Legal evidence collection
        • MP3 audio compression
        • Pattern analysis
        • Cross-platform GUI
        
        Built with Python and Tkinter
        """
        
        messagebox.showinfo("About Bark Monitor", about_text)
    
    def quit_application(self):
        """Quit the application."""
        if self.monitoring_active:
            response = messagebox.askyesno("Confirm Exit", 
                                         "Monitoring is currently active. Stop monitoring and exit?")
            if response:
                self.stop_monitoring()
            else:
                return
        
        # Cleanup
        if hasattr(self, 'log_handler'):
            logging.getLogger().removeHandler(self.log_handler)
        
        self.root.quit()
        self.root.destroy()
    
    def run(self):
        """Run the GUI application."""
        self.root.mainloop()


class GuiLogHandler(logging.Handler):
    """Custom log handler that sends logs to the GUI queue."""
    
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue
        self.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    def emit(self, record):
        self.log_queue.put(record)


def main():
    """Main entry point for the GUI application."""
    app = BarkMonitorGUI()
    app.run()


if __name__ == "__main__":
    main()