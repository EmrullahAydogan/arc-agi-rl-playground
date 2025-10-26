"""
Tkinter-based Info Panel
Status ve Agent Parameters için ayrı pencere
"""
import tkinter as tk
from tkinter import ttk
from typing import Dict, Optional
import threading


class InfoPanel:
    """
    Ayrı bir Tkinter penceresi olarak Status ve Agent Parameters gösterir
    İkinci monitörde açılabilir, bağımsız çalışır
    """

    def __init__(self, title: str = "ARC-AGI Info Panel", width: int = 450, height: int = 1000):
        """
        Args:
            title: Pencere başlığı
            width: Pencere genişliği
            height: Pencere yüksekliği
        """
        self.width = width
        self.height = height
        self.running = False

        # Tkinter root window
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry(f"{width}x{height}")

        # Pencere kapatıldığında
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Style
        self.bg_color = "#2b2b2b"
        self.fg_color = "#ffffff"
        self.panel_color = "#3a3a3a"
        self.highlight_color = "#4a9eff"

        self.root.configure(bg=self.bg_color)

        # Ana container
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Header
        header_label = tk.Label(
            main_frame,
            text="ARC-AGI INFO PANEL",
            font=("Arial", 16, "bold"),
            bg=self.bg_color,
            fg=self.highlight_color
        )
        header_label.pack(pady=(0, 20))

        # Status Section
        self._create_status_section(main_frame)

        # Separator
        sep1 = ttk.Separator(main_frame, orient='horizontal')
        sep1.pack(fill='x', pady=15)

        # Agent Parameters Section
        self._create_agent_params_section(main_frame)

        # Separator
        sep2 = ttk.Separator(main_frame, orient='horizontal')
        sep2.pack(fill='x', pady=15)

        # Last Action Section
        self._create_last_action_section(main_frame)

        # Separator
        sep3 = ttk.Separator(main_frame, orient='horizontal')
        sep3.pack(fill='x', pady=15)

        # Performance Metrics Section
        self._create_metrics_section(main_frame)

        # Separator
        sep4 = ttk.Separator(main_frame, orient='horizontal')
        sep4.pack(fill='x', pady=15)

        # Sample Status Section
        self._create_sample_status_section(main_frame)

        # Footer
        footer_label = tk.Label(
            main_frame,
            text="This panel can be moved to a second monitor",
            font=("Arial", 9, "italic"),
            bg=self.bg_color,
            fg="#888888"
        )
        footer_label.pack(side=tk.BOTTOM, pady=(10, 0))

        self.running = True

    def _create_status_section(self, parent):
        """Status bölümünü oluştur"""
        frame = tk.LabelFrame(
            parent,
            text="STATUS",
            font=("Arial", 12, "bold"),
            bg=self.panel_color,
            fg=self.fg_color,
            relief=tk.GROOVE,
            borderwidth=2
        )
        frame.pack(fill=tk.BOTH, expand=False, pady=(0, 10))

        # Status labels
        self.status_labels = {}

        status_items = [
            ("puzzle_id", "Puzzle ID:", "Unknown"),
            ("dataset", "Dataset:", "training"),
            ("mode", "Mode:", "TRAIN"),
            ("sample", "Sample:", "1 / 1"),
            ("steps", "Steps:", "0 / 100"),
            ("total_reward", "Total Reward:", "0.00"),
            ("last_reward", "Last Reward:", "0.00"),
            ("is_solved", "Solved:", "NO"),
            ("done", "Done:", "NO"),
        ]

        for key, label_text, default_value in status_items:
            row_frame = tk.Frame(frame, bg=self.panel_color)
            row_frame.pack(fill=tk.X, padx=10, pady=5)

            label = tk.Label(
                row_frame,
                text=label_text,
                font=("Arial", 10, "bold"),
                bg=self.panel_color,
                fg="#aaaaaa",
                width=15,
                anchor='w'
            )
            label.pack(side=tk.LEFT)

            value_label = tk.Label(
                row_frame,
                text=default_value,
                font=("Arial", 10),
                bg=self.panel_color,
                fg=self.fg_color,
                anchor='w'
            )
            value_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

            self.status_labels[key] = value_label

    def _create_agent_params_section(self, parent):
        """Agent Parameters bölümünü oluştur"""
        frame = tk.LabelFrame(
            parent,
            text="AGENT PARAMETERS",
            font=("Arial", 12, "bold"),
            bg=self.panel_color,
            fg=self.fg_color,
            relief=tk.GROOVE,
            borderwidth=2
        )
        frame.pack(fill=tk.BOTH, expand=False, pady=(0, 10))

        # Agent params labels
        self.agent_params_labels = {}

        params_items = [
            ("agent_type", "Agent Type:", "Unknown"),
            ("agent_actions", "Actions Taken:", "0"),
            ("learning_rate", "Learning Rate:", "N/A"),
            ("epsilon", "Epsilon:", "N/A"),
            ("gamma", "Discount (γ):", "N/A"),
            ("batch_size", "Batch Size:", "N/A"),
            ("memory_size", "Memory Size:", "N/A"),
            ("update_frequency", "Update Freq:", "N/A"),
        ]

        for key, label_text, default_value in params_items:
            row_frame = tk.Frame(frame, bg=self.panel_color)
            row_frame.pack(fill=tk.X, padx=10, pady=5)

            label = tk.Label(
                row_frame,
                text=label_text,
                font=("Arial", 10, "bold"),
                bg=self.panel_color,
                fg="#aaaaaa",
                width=15,
                anchor='w'
            )
            label.pack(side=tk.LEFT)

            value_label = tk.Label(
                row_frame,
                text=default_value,
                font=("Arial", 10),
                bg=self.panel_color,
                fg="#64ff64",
                anchor='w'
            )
            value_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

            self.agent_params_labels[key] = value_label

    def _create_last_action_section(self, parent):
        """Last Action bölümünü oluştur"""
        frame = tk.LabelFrame(
            parent,
            text="LAST ACTION",
            font=("Arial", 12, "bold"),
            bg=self.panel_color,
            fg=self.fg_color,
            relief=tk.GROOVE,
            borderwidth=2
        )
        frame.pack(fill=tk.BOTH, expand=False, pady=(0, 10))

        self.last_action_label = tk.Label(
            frame,
            text="No action yet",
            font=("Arial", 10),
            bg=self.panel_color,
            fg=self.fg_color,
            justify=tk.LEFT,
            wraplength=350
        )
        self.last_action_label.pack(padx=10, pady=10)

    def _create_metrics_section(self, parent):
        """Performance Metrics bölümünü oluştur"""
        frame = tk.LabelFrame(
            parent,
            text="PERFORMANCE METRICS",
            font=("Arial", 12, "bold"),
            bg=self.panel_color,
            fg=self.fg_color,
            relief=tk.GROOVE,
            borderwidth=2
        )
        frame.pack(fill=tk.BOTH, expand=False, pady=(0, 10))

        # Metrics labels
        self.metrics_labels = {}

        # Train metrics
        train_label = tk.Label(
            frame,
            text="TRAIN MODE",
            font=("Arial", 10, "bold"),
            bg=self.panel_color,
            fg="#64ff64"
        )
        train_label.pack(pady=(10, 5))

        train_metrics_items = [
            ("train_episodes", "Episodes:", "0"),
            ("train_solved", "Solved:", "0"),
            ("train_success_rate", "Success Rate:", "0.0%"),
            ("train_avg_accuracy", "Avg Accuracy:", "0.0%"),
        ]

        for key, label_text, default_value in train_metrics_items:
            row_frame = tk.Frame(frame, bg=self.panel_color)
            row_frame.pack(fill=tk.X, padx=10, pady=2)

            label = tk.Label(
                row_frame,
                text=label_text,
                font=("Arial", 9),
                bg=self.panel_color,
                fg="#aaaaaa",
                width=15,
                anchor='w'
            )
            label.pack(side=tk.LEFT)

            value_label = tk.Label(
                row_frame,
                text=default_value,
                font=("Arial", 9),
                bg=self.panel_color,
                fg=self.fg_color,
                anchor='w'
            )
            value_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

            self.metrics_labels[key] = value_label

        # Test metrics
        test_label = tk.Label(
            frame,
            text="TEST MODE",
            font=("Arial", 10, "bold"),
            bg=self.panel_color,
            fg="#ffb464"
        )
        test_label.pack(pady=(15, 5))

        test_metrics_items = [
            ("test_episodes", "Episodes:", "0"),
            ("test_solved", "Solved:", "0"),
            ("test_success_rate", "Success Rate:", "0.0%"),
            ("test_avg_accuracy", "Avg Accuracy:", "0.0%"),
        ]

        for key, label_text, default_value in test_metrics_items:
            row_frame = tk.Frame(frame, bg=self.panel_color)
            row_frame.pack(fill=tk.X, padx=10, pady=2)

            label = tk.Label(
                row_frame,
                text=label_text,
                font=("Arial", 9),
                bg=self.panel_color,
                fg="#aaaaaa",
                width=15,
                anchor='w'
            )
            label.pack(side=tk.LEFT)

            value_label = tk.Label(
                row_frame,
                text=default_value,
                font=("Arial", 9),
                bg=self.panel_color,
                fg=self.fg_color,
                anchor='w'
            )
            value_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

            self.metrics_labels[key] = value_label

    def _create_sample_status_section(self, parent):
        """Sample Status bölümünü oluştur - tüm sample'ların durumunu göster"""
        frame = tk.LabelFrame(
            parent,
            text="SAMPLE STATUS",
            font=("Arial", 12, "bold"),
            bg=self.panel_color,
            fg=self.fg_color,
            relief=tk.GROOVE,
            borderwidth=2
        )
        frame.pack(fill=tk.BOTH, expand=False, pady=(0, 10))

        # Açıklama
        desc_label = tk.Label(
            frame,
            text="Status of all training and test samples",
            font=("Arial", 8, "italic"),
            bg=self.panel_color,
            fg="#888888"
        )
        desc_label.pack(pady=(5, 10))

        # Sample status'lar için bir container
        # Scrollable Frame kullanabiliriz ama şimdilik basit tutalım
        self.sample_status_container = tk.Frame(frame, bg=self.panel_color)
        self.sample_status_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Sample status labels - dinamik olarak oluşturulacak
        self.sample_status_labels = []

    def _update_sample_status(self, num_train_samples: int, num_test_samples: int, metrics: dict):
        """
        Sample status'ları güncelle

        Args:
            num_train_samples: Train sample sayısı
            num_test_samples: Test sample sayısı
            metrics: Metrics dict
        """
        # Önce mevcut widget'ları temizle
        for widget in self.sample_status_container.winfo_children():
            widget.destroy()
        self.sample_status_labels.clear()

        # Train Samples
        train_header = tk.Label(
            self.sample_status_container,
            text="TRAIN SAMPLES",
            font=("Arial", 9, "bold"),
            bg=self.panel_color,
            fg="#64ff64"
        )
        train_header.pack(anchor='w', pady=(5, 5))

        # Train sample'lar için satırlar
        for i in range(num_train_samples):
            sample_metrics = metrics.get('train_sample_stats', {}).get(i, {
                'episodes': 0,
                'solved': 0,
                'avg_accuracy': 0.0
            })

            episodes = sample_metrics.get('episodes', 0)
            solved = sample_metrics.get('solved', 0)
            accuracy = sample_metrics.get('avg_accuracy', 0.0)

            # Status belirleme
            if solved > 0:
                status = "[OK] SOLVED"
                status_color = "#64ff64"
            elif episodes > 0:
                if accuracy >= 50:
                    status = "[~] PARTIAL"
                    status_color = "#ffdd64"
                else:
                    status = "[X] FAILED"
                    status_color = "#ff6464"
            else:
                status = "[-] NOT TRIED"
                status_color = "#888888"

            # Sample row
            row_frame = tk.Frame(self.sample_status_container, bg=self.panel_color)
            row_frame.pack(fill=tk.X, pady=2)

            # Sample numarası
            sample_label = tk.Label(
                row_frame,
                text=f"Sample {i+1}:",
                font=("Arial", 8, "bold"),
                bg=self.panel_color,
                fg="#aaaaaa",
                width=10,
                anchor='w'
            )
            sample_label.pack(side=tk.LEFT)

            # Episodes
            episodes_label = tk.Label(
                row_frame,
                text=f"Ep:{episodes}",
                font=("Arial", 8),
                bg=self.panel_color,
                fg="#cccccc",
                width=6,
                anchor='w'
            )
            episodes_label.pack(side=tk.LEFT)

            # Accuracy
            acc_label = tk.Label(
                row_frame,
                text=f"{accuracy:.0f}%",
                font=("Arial", 8),
                bg=self.panel_color,
                fg="#cccccc",
                width=5,
                anchor='w'
            )
            acc_label.pack(side=tk.LEFT)

            # Status
            status_label = tk.Label(
                row_frame,
                text=status,
                font=("Arial", 8, "bold"),
                bg=self.panel_color,
                fg=status_color,
                anchor='w'
            )
            status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Test Samples
        test_header = tk.Label(
            self.sample_status_container,
            text="TEST SAMPLES",
            font=("Arial", 9, "bold"),
            bg=self.panel_color,
            fg="#ffb464"
        )
        test_header.pack(anchor='w', pady=(15, 5))

        # Test metrics (genelde tek bir test sample var)
        test_metrics = metrics.get('test', {})
        test_episodes = test_metrics.get('episodes', 0)
        test_solved = test_metrics.get('solved', 0)
        test_accuracy = test_metrics.get('avg_accuracy', 0.0)

        # Test status
        if test_solved > 0:
            test_status = "[OK] SOLVED"
            test_status_color = "#64ff64"
        elif test_episodes > 0:
            if test_accuracy >= 50:
                test_status = "[~] PARTIAL"
                test_status_color = "#ffdd64"
            else:
                test_status = "[X] FAILED"
                test_status_color = "#ff6464"
        else:
            test_status = "[-] NOT TRIED"
            test_status_color = "#888888"

        # Test row
        test_row_frame = tk.Frame(self.sample_status_container, bg=self.panel_color)
        test_row_frame.pack(fill=tk.X, pady=2)

        test_sample_label = tk.Label(
            test_row_frame,
            text=f"Test 1:",
            font=("Arial", 8, "bold"),
            bg=self.panel_color,
            fg="#aaaaaa",
            width=10,
            anchor='w'
        )
        test_sample_label.pack(side=tk.LEFT)

        test_episodes_label = tk.Label(
            test_row_frame,
            text=f"Ep:{test_episodes}",
            font=("Arial", 8),
            bg=self.panel_color,
            fg="#cccccc",
            width=6,
            anchor='w'
        )
        test_episodes_label.pack(side=tk.LEFT)

        test_acc_label = tk.Label(
            test_row_frame,
            text=f"{test_accuracy:.0f}%",
            font=("Arial", 8),
            bg=self.panel_color,
            fg="#cccccc",
            width=5,
            anchor='w'
        )
        test_acc_label.pack(side=tk.LEFT)

        test_status_label = tk.Label(
            test_row_frame,
            text=test_status,
            font=("Arial", 8, "bold"),
            bg=self.panel_color,
            fg=test_status_color,
            anchor='w'
        )
        test_status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def update(self, info: Dict):
        """
        Panel bilgilerini güncelle

        Args:
            info: Güncellenecek bilgiler dict'i
        """
        if not self.running:
            return

        try:
            # Status güncelle
            if 'puzzle_id' in info:
                self.status_labels['puzzle_id'].config(text=info['puzzle_id'])

            if 'dataset' in info:
                self.status_labels['dataset'].config(text=info['dataset'])

            # Mode güncelle
            if 'mode' in info:
                mode_text = info['mode'].upper()
                mode_color = "#64ff64" if info['mode'] == 'train' else "#ffb464"
                self.status_labels['mode'].config(text=mode_text, fg=mode_color)

            # Sample güncelle (sadece train mode'da göster)
            if 'mode' in info:
                if info['mode'] == 'train' and 'train_sample_index' in info and 'num_train_samples' in info:
                    sample_text = f"{info['train_sample_index'] + 1} / {info['num_train_samples']}"
                    self.status_labels['sample'].config(text=sample_text)
                elif info['mode'] == 'test':
                    self.status_labels['sample'].config(text="N/A (Test)")

            if 'steps' in info and 'max_steps' in info:
                self.status_labels['steps'].config(
                    text=f"{info['steps']} / {info['max_steps']}"
                )

            if 'total_reward' in info:
                self.status_labels['total_reward'].config(
                    text=f"{info['total_reward']:.2f}"
                )

            if 'last_reward' in info:
                self.status_labels['last_reward'].config(
                    text=f"{info['last_reward']:.2f}"
                )

            if 'is_solved' in info:
                solved_text = "YES [OK]" if info['is_solved'] else "NO"
                solved_color = "#64ff64" if info['is_solved'] else "#ff6464"
                self.status_labels['is_solved'].config(
                    text=solved_text,
                    fg=solved_color
                )

            if 'done' in info:
                done_text = "YES" if info['done'] else "NO"
                self.status_labels['done'].config(text=done_text)

            # Agent params güncelle
            if 'agent_type' in info:
                self.agent_params_labels['agent_type'].config(text=info['agent_type'])

            if 'agent_actions' in info:
                self.agent_params_labels['agent_actions'].config(
                    text=str(info['agent_actions'])
                )

            if 'agent_params' in info:
                params = info['agent_params']
                for key in ['learning_rate', 'epsilon', 'gamma', 'batch_size', 'memory_size', 'update_frequency']:
                    if key in params:
                        self.agent_params_labels[key].config(text=str(params[key]))

            # Last action güncelle
            if 'last_action_decoded' in info:
                x, y, color = info['last_action_decoded']
                action_text = f"Position: ({x}, {y})\nColor: {color}"
                self.last_action_label.config(text=action_text)

            # Metrics güncelle
            if 'metrics' in info:
                metrics = info['metrics']

                # Train metrics
                if 'train' in metrics:
                    train = metrics['train']
                    self.metrics_labels['train_episodes'].config(text=str(train['episodes']))
                    self.metrics_labels['train_solved'].config(text=str(train['solved']))
                    self.metrics_labels['train_success_rate'].config(text=f"{train['success_rate']:.1f}%")
                    self.metrics_labels['train_avg_accuracy'].config(text=f"{train['avg_accuracy']:.1f}%")

                # Test metrics
                if 'test' in metrics:
                    test = metrics['test']
                    self.metrics_labels['test_episodes'].config(text=str(test['episodes']))
                    self.metrics_labels['test_solved'].config(text=str(test['solved']))
                    self.metrics_labels['test_success_rate'].config(text=f"{test['success_rate']:.1f}%")
                    self.metrics_labels['test_avg_accuracy'].config(text=f"{test['avg_accuracy']:.1f}%")

                # Sample status güncelle
                if 'num_train_samples' in info and 'num_test_samples' in info:
                    self._update_sample_status(
                        num_train_samples=info['num_train_samples'],
                        num_test_samples=info['num_test_samples'],
                        metrics=metrics
                    )

            # Tkinter güncelleme
            self.root.update()

        except tk.TclError:
            # Pencere kapatılmışsa
            self.running = False

    def _on_close(self):
        """Pencere kapatıldığında"""
        self.running = False
        self.root.destroy()

    def is_running(self) -> bool:
        """Panel hala açık mı?"""
        return self.running

    def close(self):
        """Paneli kapat"""
        if self.running:
            self.running = False
            try:
                self.root.destroy()
            except:
                pass

    def mainloop_iteration(self):
        """Bir Tkinter mainloop iterasyonu çalıştır (non-blocking)"""
        if self.running:
            try:
                self.root.update_idletasks()
                self.root.update()
            except tk.TclError:
                self.running = False
