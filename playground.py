#!/usr/bin/env python3
"""
ARC-AGI Reinforcement Learning Playground
Agent'ları izleyip müdahale edebileceğiniz interaktif oyun alanı
"""
import sys
import argparse
from pathlib import Path

# Projeyi path'e ekle
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.data_loader import ARCDataLoader
from src.utils.metrics_tracker import MetricsTracker
from src.environment.arc_env import ARCEnvironment
from src.agents.random_agent import RandomAgent
from src.visualization.pygame_viewer import PygameViewer
from src.visualization.info_panel import InfoPanel
from src.visualization.puzzle_browser import PuzzleBrowser


class ARCPlayground:
    """ARC RL oyun alanı ana sınıfı"""

    def __init__(
        self,
        data_dir: str = "arc-prize-2025",
        puzzle_id: str = None,
        dataset: str = "training",
        max_steps: int = 100,
        fps: int = 5,
        seed: int = None
    ):
        """
        Args:
            data_dir: ARC veri dizini
            puzzle_id: Hangi puzzle'ı yükleyeceğiz (None ise rastgele)
            dataset: "training" veya "evaluation"
            max_steps: Maksimum adım sayısı
            fps: Görselleştirme FPS'i
            seed: Random seed
        """
        self.data_dir = data_dir
        self.dataset = dataset
        self.max_steps = max_steps
        self.fps = fps
        self.seed = seed

        # Data loader
        print("[LOADING] ARC verileri yükleniyor...")
        self.loader = ARCDataLoader(data_dir)

        # Puzzle yükle
        if puzzle_id:
            print(f"[LOADING] Puzzle yükleniyor: {puzzle_id}")
            puzzle_data = self.loader.get_puzzle(puzzle_id, dataset)
            self.puzzle_id = puzzle_id
        else:
            print("[LOADING] Rastgele puzzle seçiliyor...")
            self.puzzle_id, puzzle_data = self.loader.get_random_puzzle(dataset)
            print(f"[INFO] Seçilen puzzle: {self.puzzle_id}")

        if puzzle_data is None:
            raise ValueError(f"Puzzle bulunamadı: {puzzle_id}")

        # Environment oluştur
        print("[SETUP] Environment oluşturuluyor...")
        self.env = ARCEnvironment(
            puzzle_data=puzzle_data,
            task_index=0,
            max_steps=max_steps
        )

        # Agent oluştur
        print("[SETUP] Agent oluşturuluyor...")
        self.agent = RandomAgent(
            action_space=self.env.action_space,
            observation_space=self.env.observation_space,
            seed=seed
        )

        # Visualizer oluştur (Grid'ler için)
        print("[SETUP] Görselleştirme arayüzü oluşturuluyor...")
        self.viewer = PygameViewer(fps=fps)

        # Info Panel oluştur (Status + Agent Params için)
        print("[SETUP] Info panel oluşturuluyor...")
        self.info_panel = InfoPanel(title=f"ARC-AGI Info Panel - {self.puzzle_id}")

        # Metrics Tracker oluştur
        print("[SETUP] Metrics tracker oluşturuluyor...")
        self.metrics = MetricsTracker()

        # Puzzle Browser (lazy - sadece açıldığında oluşturulacak)
        self.puzzle_browser = None

        print("[OK] Oyun alanı hazır!")
        print("[INFO] İki pencere açıldı: Grid'ler için ana pencere, bilgiler için info panel")
        print("[INFO] Info panelini ikinci monitörünüze sürükleyebilirsiniz\n")

    def run(self):
        """Ana oyun döngüsü"""
        print("=" * 60)
        print(">>> ARC-AGI RL PLAYGROUND BAŞLATILDI <<<")
        print("=" * 60)
        print(f"[INFO] Puzzle ID: {self.puzzle_id}")
        print(f"[INFO] Agent: {self.agent.__class__.__name__}")
        print(f"[INFO] Max Steps: {self.max_steps}")
        print(f"[INFO] FPS: {self.fps}")
        print("\n>>> KONTROLLER:")
        print("   BUTONLAR (Alt panelde):")
        print("      > START / || PAUSE : Eğitimi başlat/duraklat")
        print("      RESET              : Episode'u sıfırla")
        print("      <> MODE            : Mode değiştir (Train <-> Test)")
        print("      < PREV / NEXT >    : Sample'lar arası geçiş")
        print("      SPEED -/+          : Hızı ayarla")
        print("      LAYOUT             : Layout değiştir")
        print("\n   KLAVYE KISAYOLLARI:")
        print("      SPACE: Pause/Resume  |  R: Reset  |  Q: Çıkış")
        print("      Z/X: Zoom  |  A: Auto-fit  |  ->: Step")
        print("      E: Edit Mode (Click CURRENT grid to paint)")
        print("      H: Heatmap (Show agent activity overlay)")

        print("\n   *** Sistem PAUSE modunda başlar - START butonuna tıklayın! ***")
        print("=" * 60)
        print()

        # İlk reset
        observation, info = self.env.reset()
        self.agent.reset()

        # İlk episode'u başlat (metrics tracking için)
        state_info = self.env.get_state_info()
        self.metrics.start_episode(
            mode=state_info['mode'],
            sample_index=state_info['train_sample_index']
        )

        running = True
        episode_count = 0

        while running:
            # Event handling (pass current grid reference for editing)
            controls = self.viewer.handle_events(current_grid=self.env.current_grid)

            if controls['quit']:
                print("\n[EXIT] Çıkılıyor...")
                running = False
                break

            # Puzzle browser açma
            if controls.get('open_browser', False):
                if self.puzzle_browser is None or not self.puzzle_browser.is_running():
                    print("\n[BROWSER] Puzzle browser açılıyor...")
                    self.puzzle_browser = PuzzleBrowser(
                        self.loader,
                        on_puzzle_select=self._on_puzzle_selected,
                        title=f"ARC Puzzle Browser - {self.dataset}"
                    )
                self.viewer.reset_control_flags()

            if controls['reset']:
                print(f"\n[RESET] Episode {episode_count} sıfırlanıyor...")
                observation, info = self.env.reset()
                self.agent.reset()
                self.viewer.reset_heatmap()  # Reset heatmap
                # Yeni episode başlat
                state_info = self.env.get_state_info()
                self.metrics.start_episode(
                    mode=state_info['mode'],
                    sample_index=state_info['train_sample_index']
                )
                self.viewer.reset_control_flags()
                continue

            # Mode toggle (T tuşu)
            if controls.get('toggle_mode', False):
                new_mode = self.env.switch_mode()
                mode_str = "TRAIN" if new_mode == 'train' else "TEST"
                state_info = self.env.get_state_info()
                sample_info = ""
                if new_mode == 'train':
                    sample_info = f" (Sample {state_info['train_sample_index'] + 1}/{state_info['num_train_samples']})"
                print(f"\n[MODE] Mode değiştirildi -> {mode_str}{sample_info}")
                observation, info = self.env.reset()
                self.agent.reset()
                # Yeni episode başlat
                self.metrics.start_episode(
                    mode=state_info['mode'],
                    sample_index=state_info['train_sample_index']
                )
                self.viewer.reset_control_flags()
                continue

            # Next sample (N tuşu - sadece train mode'da)
            if controls.get('next_sample', False):
                if self.env.mode == 'train':
                    new_index = self.env.next_sample()
                    state_info = self.env.get_state_info()
                    print(f"\n[NEXT] Train Sample: {new_index + 1}/{state_info['num_train_samples']}")
                    observation, info = self.env.reset()
                    self.agent.reset()
                    # Yeni episode başlat
                    self.metrics.start_episode(
                        mode=state_info['mode'],
                        sample_index=state_info['train_sample_index']
                    )
                    self.viewer.reset_control_flags()
                    continue

            # Previous sample (P tuşu - sadece train mode'da)
            if controls.get('prev_sample', False):
                if self.env.mode == 'train':
                    new_index = self.env.previous_sample()
                    state_info = self.env.get_state_info()
                    print(f"\n[PREV] Train Sample: {new_index + 1}/{state_info['num_train_samples']}")
                    observation, info = self.env.reset()
                    self.agent.reset()
                    # Yeni episode başlat
                    self.metrics.start_episode(
                        mode=state_info['mode'],
                        sample_index=state_info['train_sample_index']
                    )
                    self.viewer.reset_control_flags()
                    continue

            # Pause kontrolü
            if controls['paused'] and not controls['step']:
                # Sadece görselleştir, adım atma
                state_info = self.env.get_state_info()
                self._render(state_info)

                # Info panel güncelle (Tkinter)
                if self.info_panel.is_running():
                    self.info_panel.mainloop_iteration()

                # Puzzle browser güncelle (Tkinter)
                if self.puzzle_browser and self.puzzle_browser.is_running():
                    self.puzzle_browser.mainloop_iteration()

                continue

            # Agent'tan action al
            action = self.agent.select_action(observation, info)

            # Environment'ta adım at
            next_observation, reward, terminated, truncated, step_info = self.env.step(action)

            # Record action for heatmap
            if 'action_decoded' in step_info:
                self.viewer.record_action(step_info['action_decoded'])

            # Metrics: Step kaydet
            state_info = self.env.get_state_info()
            # Accuracy hesapla
            import numpy as np
            target_height, target_width = state_info['target_grid'].shape
            current_slice = state_info['current_grid'][:target_height, :target_width]
            correct_pixels = np.sum(current_slice == state_info['target_grid'])
            total_pixels = target_height * target_width
            accuracy = correct_pixels / total_pixels
            self.metrics.record_step(reward, accuracy)

            # Agent'ı güncelle
            self.agent.update(observation, action, reward, next_observation, terminated or truncated, step_info)

            # Görselleştir
            self._render(state_info)

            # Info panel güncelle (Tkinter)
            if self.info_panel.is_running():
                self.info_panel.mainloop_iteration()

            # Puzzle browser güncelle (Tkinter)
            if self.puzzle_browser and self.puzzle_browser.is_running():
                self.puzzle_browser.mainloop_iteration()

            # Episode bitti mi?
            if terminated or truncated:
                # Metrics: Episode'u bitir
                self.metrics.end_episode(is_solved=terminated, final_accuracy=accuracy)

                episode_count += 1

                if terminated:
                    print(f"\n[SUCCESS] Episode {episode_count} ÇÖZÜLDÜ!")
                else:
                    print(f"\n[TIMEOUT] Episode {episode_count} zaman aşımı")

                print(f"   Total Steps: {state_info['steps']}")
                print(f"   Total Reward: {state_info['total_reward']:.2f}")
                print(f"   Final Accuracy: {accuracy * 100:.1f}%")
                print(f"   Agent Actions: {self.agent.get_info()['actions_taken']}")

                # Metrics özeti göster
                if state_info['mode'] == 'train':
                    train_metrics = self.metrics.get_train_metrics()
                    print(f"   [METRICS] Train Success Rate: {train_metrics['success_rate']:.1f}% ({train_metrics['solved']}/{train_metrics['episodes']})")
                else:
                    test_metrics = self.metrics.get_test_metrics()
                    print(f"   [METRICS] Test Success Rate: {test_metrics['success_rate']:.1f}% ({test_metrics['solved']}/{test_metrics['episodes']})")

                # Biraz bekle
                import time
                time.sleep(2)

                # Yeni episode
                print(f"\n[RESTART] Yeni episode başlatılıyor...")
                observation, info = self.env.reset()
                self.agent.reset()

                # Yeni episode için metrics başlat
                state_info = self.env.get_state_info()
                self.metrics.start_episode(
                    mode=state_info['mode'],
                    sample_index=state_info['train_sample_index']
                )
            else:
                observation = next_observation

            # Step mode'u sıfırla
            self.viewer.reset_control_flags()

        # Temizlik
        self.viewer.close()
        if self.info_panel.is_running():
            self.info_panel.close()
        if self.puzzle_browser and self.puzzle_browser.is_running():
            self.puzzle_browser.close()
        print("\n[OK] Oyun alanı kapatıldı. Görüşmek üzere!")

    def _on_puzzle_selected(self, puzzle_id: str, dataset: str):
        """Puzzle browser'dan puzzle seçildiğinde çağrılır"""
        print(f"\n[PUZZLE CHANGE] Yeni puzzle yükleniyor: {puzzle_id} ({dataset})")

        # Puzzle data'yı yükle
        puzzle_data = self.loader.get_puzzle(puzzle_id, dataset)
        if puzzle_data is None:
            print(f"[ERROR] Puzzle yüklenemedi: {puzzle_id}")
            return

        # Mevcut environment'ı kapat ve yenisini oluştur
        self.puzzle_id = puzzle_id
        self.dataset = dataset

        self.env = ARCEnvironment(
            puzzle_data=puzzle_data,
            task_index=0,
            max_steps=self.max_steps
        )

        # Agent'ı sıfırla
        self.agent.reset()

        # Viewer'ı sıfırla
        self.viewer.reset_heatmap()

        # Metrics'i sıfırla
        self.metrics = MetricsTracker()

        # Environment'ı reset et
        observation, info = self.env.reset()

        # Episode başlat
        state_info = self.env.get_state_info()
        self.metrics.start_episode(
            mode=state_info['mode'],
            sample_index=state_info['train_sample_index']
        )

        print(f"[PUZZLE CHANGE] Puzzle değiştirildi: {puzzle_id}")
        print(f"   Dataset: {dataset}")
        print(f"   Train Samples: {state_info['num_train_samples']}")
        print(f"   Test Samples: {state_info['num_test_samples']}")

    def _render(self, state_info: dict):
        """Mevcut durumu görselleştir"""
        # Metrics bilgilerini al
        all_metrics = self.metrics.get_all_metrics()

        # Viewer için info hazırla
        viewer_info = {
            'puzzle_id': self.puzzle_id,
            'dataset': self.dataset,
            'mode': state_info['mode'],
            'train_sample_index': state_info['train_sample_index'],
            'num_train_samples': state_info['num_train_samples'],
            'num_test_samples': state_info['num_test_samples'],
            'steps': state_info['steps'],
            'max_steps': state_info['max_steps'],
            'total_reward': state_info['total_reward'],
            'last_reward': state_info['last_reward'],
            'is_solved': state_info['is_solved'],
            'done': state_info['done'],
            'agent_type': self.agent.__class__.__name__,
            'agent_actions': self.agent.get_info()['actions_taken'],
            'agent_params': self.agent.get_params(),
            'metrics': all_metrics,  # Metrics bilgilerini ekle
        }

        if state_info['last_action'] is not None:
            x, y, color = self.env._action_to_grid_operation(state_info['last_action'])
            viewer_info['last_action_decoded'] = (x, y, color)

        # Pygame penceresini render et (Grid'ler)
        self.viewer.render(
            input_grid=state_info['input_grid'],
            current_grid=state_info['current_grid'],
            target_grid=state_info['target_grid'],
            info=viewer_info
        )

        # Tkinter penceresini güncelle (Info Panel)
        if self.info_panel.is_running():
            self.info_panel.update(viewer_info)


def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(description="ARC-AGI RL Playground")
    parser.add_argument(
        '--puzzle-id',
        type=str,
        default=None,
        help='Puzzle ID (boş bırakılırsa rastgele seçilir)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='training',
        choices=['training', 'evaluation'],
        help='Dataset seçimi'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=100,
        help='Maksimum adım sayısı'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=5,
        help='Görselleştirme FPS'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed'
    )

    args = parser.parse_args()

    # Playground oluştur ve çalıştır
    try:
        playground = ARCPlayground(
            puzzle_id=args.puzzle_id,
            dataset=args.dataset,
            max_steps=args.max_steps,
            fps=args.fps,
            seed=args.seed
        )
        playground.run()
    except KeyboardInterrupt:
        print("\n\n[WARNING] KeyboardInterrupt - Çıkılıyor...")
    except Exception as e:
        print(f"\n[ERROR] HATA: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
