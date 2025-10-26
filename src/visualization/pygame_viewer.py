"""
Pygame-based ARC Environment Visualizer
Agent'ın davranışlarını gerçek zamanlı izleme arayüzü
"""
import pygame
import numpy as np
from typing import Dict, Optional, Tuple, Callable
import time


class Button:
    """Pygame button widget"""

    def __init__(self, x: int, y: int, width: int, height: int, text: str,
                 action: Callable, bg_color=(70, 70, 70), hover_color=(100, 100, 100),
                 text_color=(255, 255, 255), font_size=20):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.action = action
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.text_color = text_color
        self.font = pygame.font.Font(None, font_size)
        self.is_hovered = False

    def draw(self, screen):
        """Butonu çiz"""
        color = self.hover_color if self.is_hovered else self.bg_color
        pygame.draw.rect(screen, color, self.rect, border_radius=5)
        pygame.draw.rect(screen, (150, 150, 150), self.rect, 2, border_radius=5)

        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def handle_event(self, event):
        """Mouse event'lerini handle et"""
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.rect.collidepoint(event.pos):
                self.action()
                return True
        return False


# ARC renk paleti (0-9 arası renkler)
ARC_COLORS = {
    0: (0, 0, 0),           # Siyah
    1: (0, 116, 217),       # Mavi
    2: (255, 65, 54),       # Kırmızı
    3: (46, 204, 64),       # Yeşil
    4: (255, 220, 0),       # Sarı
    5: (170, 170, 170),     # Gri
    6: (240, 18, 190),      # Magenta
    7: (255, 133, 27),      # Turuncu
    8: (127, 219, 255),     # Açık mavi
    9: (135, 12, 37),       # Bordo
}


class ColorPalette:
    """Color palette widget for selecting ARC colors"""

    def __init__(self, x: int, y: int, cell_size: int = 40):
        self.x = x
        self.y = y
        self.cell_size = cell_size
        self.selected_color = 1  # Default: Blue
        self.hover_color = None

    def draw(self, screen, font):
        """Draw the color palette"""
        # Title
        title_surface = font.render("COLOR PALETTE", True, (255, 255, 255))
        screen.blit(title_surface, (self.x, self.y - 30))

        # Draw color cells (2 columns, 5 rows)
        for color_id in range(10):
            row = color_id % 5
            col = color_id // 5

            cell_x = self.x + col * (self.cell_size + 5)
            cell_y = self.y + row * (self.cell_size + 5)

            # Cell background
            cell_rect = pygame.Rect(cell_x, cell_y, self.cell_size, self.cell_size)
            pygame.draw.rect(screen, ARC_COLORS[color_id], cell_rect)

            # Border
            border_color = (255, 255, 0) if color_id == self.selected_color else (150, 150, 150)
            border_width = 3 if color_id == self.selected_color else 1
            pygame.draw.rect(screen, border_color, cell_rect, border_width)

            # Hover effect
            if color_id == self.hover_color:
                hover_rect = pygame.Rect(cell_x - 2, cell_y - 2, self.cell_size + 4, self.cell_size + 4)
                pygame.draw.rect(screen, (255, 255, 255), hover_rect, 1)

            # Color number
            num_surface = font.render(str(color_id), True, (255, 255, 255))
            num_rect = num_surface.get_rect(center=(cell_x + self.cell_size // 2, cell_y + self.cell_size // 2))

            # Dark outline for visibility
            outline_surface = font.render(str(color_id), True, (0, 0, 0))
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                screen.blit(outline_surface, (num_rect.x + dx, num_rect.y + dy))
            screen.blit(num_surface, num_rect)

        # Selected color info
        info_text = f"Selected: Color {self.selected_color}"
        info_surface = font.render(info_text, True, (200, 200, 200))
        screen.blit(info_surface, (self.x, self.y + 5 * (self.cell_size + 5) + 10))

    def handle_event(self, event):
        """Handle mouse events"""
        if event.type == pygame.MOUSEMOTION:
            mouse_x, mouse_y = event.pos
            self.hover_color = None

            for color_id in range(10):
                row = color_id % 5
                col = color_id // 5
                cell_x = self.x + col * (self.cell_size + 5)
                cell_y = self.y + row * (self.cell_size + 5)
                cell_rect = pygame.Rect(cell_x, cell_y, self.cell_size, self.cell_size)

                if cell_rect.collidepoint(mouse_x, mouse_y):
                    self.hover_color = color_id
                    break

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                mouse_x, mouse_y = event.pos

                for color_id in range(10):
                    row = color_id % 5
                    col = color_id // 5
                    cell_x = self.x + col * (self.cell_size + 5)
                    cell_y = self.y + row * (self.cell_size + 5)
                    cell_rect = pygame.Rect(cell_x, cell_y, self.cell_size, self.cell_size)

                    if cell_rect.collidepoint(mouse_x, mouse_y):
                        self.selected_color = color_id
                        return True
        return False


class HeatmapOverlay:
    """Track and visualize agent's grid modifications"""

    def __init__(self, max_size: int = 30):
        self.max_size = max_size
        self.modification_count = np.zeros((max_size, max_size), dtype=np.int32)
        self.enabled = False  # H tuşu ile toggle

    def record_modification(self, x: int, y: int):
        """Record a modification at (x, y)"""
        if 0 <= x < self.max_size and 0 <= y < self.max_size:
            self.modification_count[x, y] += 1

    def reset(self):
        """Clear all modification counts"""
        self.modification_count = np.zeros((self.max_size, self.max_size), dtype=np.int32)

    def draw_heatmap(self, screen, grid_data):
        """Draw heatmap overlay on grid"""
        if not self.enabled:
            return

        grid_x = grid_data['x']
        grid_y = grid_data['y']
        cell_size = grid_data['cell_size']
        height, width = grid_data['shape']

        # Get max count for normalization
        max_count = np.max(self.modification_count[:height, :width])
        if max_count == 0:
            return

        # Draw semi-transparent overlay
        for i in range(height):
            for j in range(width):
                count = self.modification_count[i, j]
                if count > 0:
                    # Normalize to 0-255
                    intensity = int((count / max_count) * 255)

                    # Create color gradient: Blue (cold) -> Red (hot)
                    if intensity < 128:
                        # Blue to Yellow
                        r = int((intensity / 128) * 255)
                        g = int((intensity / 128) * 255)
                        b = 255 - int((intensity / 128) * 255)
                    else:
                        # Yellow to Red
                        r = 255
                        g = 255 - int(((intensity - 128) / 127) * 255)
                        b = 0

                    # Draw semi-transparent overlay
                    overlay = pygame.Surface((cell_size, cell_size))
                    overlay.set_alpha(120)  # Semi-transparent
                    overlay.fill((r, g, b))

                    cell_x = grid_x + j * cell_size
                    cell_y = grid_y + i * cell_size
                    screen.blit(overlay, (cell_x, cell_y))

                    # Draw count number (if count > 1)
                    if count > 1:
                        font = pygame.font.Font(None, max(12, cell_size // 2))
                        count_text = str(count)
                        text_surface = font.render(count_text, True, (255, 255, 255))
                        text_rect = text_surface.get_rect(center=(cell_x + cell_size // 2, cell_y + cell_size // 2))

                        # Dark outline for visibility
                        outline_surface = font.render(count_text, True, (0, 0, 0))
                        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                            screen.blit(outline_surface, (text_rect.x + dx, text_rect.y + dy))
                        screen.blit(text_surface, text_rect)


class GridEditor:
    """Interactive grid editing with mouse"""

    def __init__(self):
        self.edit_mode = False
        self.grid_rects = {}  # {grid_type: {'rect': ..., 'cell_rects': ...}}
        self.hover_cell = None  # (grid_type, x, y)

    def handle_click(self, event, selected_color, current_grid, grid_rects):
        """Handle mouse click on grid"""
        if not self.edit_mode:
            return False

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_x, mouse_y = event.pos

            # Check CURRENT grid only (editable)
            if 'current' in grid_rects:
                grid_data = grid_rects['current']
                grid_x = grid_data['x']
                grid_y = grid_data['y']
                cell_size = grid_data['cell_size']
                height, width = grid_data['shape']

                # Check if click is within grid bounds
                grid_rect = pygame.Rect(grid_x, grid_y, width * cell_size, height * cell_size)
                if grid_rect.collidepoint(mouse_x, mouse_y):
                    # Calculate grid cell
                    rel_x = mouse_x - grid_x
                    rel_y = mouse_y - grid_y
                    cell_x = rel_y // cell_size
                    cell_y = rel_x // cell_size

                    if 0 <= cell_x < height and 0 <= cell_y < width:
                        # Modify the grid
                        current_grid[cell_x, cell_y] = selected_color
                        return True

        elif event.type == pygame.MOUSEMOTION:
            # Track hover for visual feedback
            mouse_x, mouse_y = event.pos
            self.hover_cell = None

            if 'current' in grid_rects:
                grid_data = grid_rects['current']
                grid_x = grid_data['x']
                grid_y = grid_data['y']
                cell_size = grid_data['cell_size']
                height, width = grid_data['shape']

                grid_rect = pygame.Rect(grid_x, grid_y, width * cell_size, height * cell_size)
                if grid_rect.collidepoint(mouse_x, mouse_y):
                    rel_x = mouse_x - grid_x
                    rel_y = mouse_y - grid_y
                    cell_x = rel_y // cell_size
                    cell_y = rel_x // cell_size

                    if 0 <= cell_x < height and 0 <= cell_y < width:
                        self.hover_cell = ('current', cell_x, cell_y)

        return False

    def draw_hover_highlight(self, screen, grid_rects):
        """Draw highlight on hovered cell"""
        if self.hover_cell and self.edit_mode:
            grid_type, cell_x, cell_y = self.hover_cell

            if grid_type in grid_rects:
                grid_data = grid_rects[grid_type]
                grid_x = grid_data['x']
                grid_y = grid_data['y']
                cell_size = grid_data['cell_size']

                # Draw highlight
                highlight_x = grid_x + cell_y * cell_size
                highlight_y = grid_y + cell_x * cell_size
                highlight_rect = pygame.Rect(highlight_x, highlight_y, cell_size, cell_size)

                # Semi-transparent white overlay
                overlay = pygame.Surface((cell_size, cell_size))
                overlay.set_alpha(100)
                overlay.fill((255, 255, 255))
                screen.blit(overlay, (highlight_x, highlight_y))

                # Bright border
                pygame.draw.rect(screen, (255, 255, 0), highlight_rect, 2)


class PygameViewer:
    """
    Pygame tabanlı görselleştirme arayüzü

    Kontroller:
    - SPACE: Pause/Resume
    - RIGHT ARROW: Bir adım ileri (pause modunda)
    - R: Reset
    - Q/ESC: Çıkış
    - +/-: Hızı ayarla
    """

    def __init__(
        self,
        window_width: int = 1600,
        window_height: int = 900, 
        cell_size: int = 25,
        fps: int = 10
    ):
        """
        Args:
            window_width: Pencere genişliği
            window_height: Pencere yüksekliği
            cell_size: Her grid hücresinin piksel boyutu
            fps: Saniyedeki frame sayısı
        """
        pygame.init()

        self.window_width = window_width
        self.window_height = window_height
        self.base_cell_size = cell_size  # Orijinal cell size
        self.current_cell_size = cell_size  # Dinamik cell size
        self.fps = fps

        # Zoom sistemi
        self.zoom_level = 1.0  # 1.0 = normal
        self.min_cell_size = 8  # Minimum hücre boyutu
        self.max_cell_size = 40  # Maximum hücre boyutu
        self.zoom_step = 0.1  # Her zoom adımında %10 değişim
        self.auto_fit_enabled = True  # Otomatik ekrana sığdırma

        # Layout sistemi
        self.layout_mode = "horizontal"  # "horizontal" veya "vertical"

        # Minimum pencere boyutları
        self.min_width = 900
        self.min_height = 700

        # Pygame window - RESIZABLE flag ile
        self.screen = pygame.display.set_mode(
            (window_width, window_height),
            pygame.RESIZABLE
        )
        pygame.display.set_caption("ARC-AGI RL Playground")

        # Clock for FPS
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 22)
        self.font_tiny = pygame.font.Font(None, 18)

        # Control state
        self.paused = True  # Başlangıçta PAUSED modda başla
        self.step_mode = False  # Bir adım ilerleme
        self.should_quit = False
        self.should_reset = False
        self.toggle_mode = False  # Mode değiştir (Train ↔ Test)
        self.next_sample = False  # Sonraki sample
        self.prev_sample = False  # Önceki sample
        self.open_browser_flag = False  # Puzzle browser aç
        self.open_history_flag = False  # Episode history aç

        # Button system
        self.buttons = []
        self.button_hover = None  # Hangi butonun üzerinde mouse var
        self._create_buttons()  # Butonları oluştur

        # Edit system
        self.color_palette = ColorPalette(x=20, y=200, cell_size=35)
        self.grid_editor = GridEditor()
        self.heatmap_overlay = HeatmapOverlay(max_size=30)  # ARC max grid size
        self.grid_rects = {}  # Grid position tracking for editor
        self.current_grid_reference = None  # Reference to current grid for editing

        # Colors
        self.bg_color = (30, 30, 30)
        self.text_color = (255, 255, 255)
        self.panel_color = (50, 50, 50)
        self.border_color = (100, 100, 100)

    def _create_buttons(self):
        """Kontrol butonlarını oluştur"""
        button_height = 35
        button_spacing = 10

        # İlk satır (üst) - Control butonları
        button_y1 = self.window_height - 85
        start_x = 20

        # START/PAUSE butonu
        self.start_pause_btn = Button(
            start_x, button_y1, 120, button_height,
            "> START" if self.paused else "|| PAUSE",
            self._toggle_pause,
            bg_color=(46, 204, 64) if self.paused else (255, 200, 0),
            hover_color=(60, 220, 80) if self.paused else (255, 220, 50)
        )
        self.buttons.append(self.start_pause_btn)
        start_x += 120 + button_spacing

        # RESET butonu
        self.buttons.append(Button(
            start_x, button_y1, 100, button_height,
            "RESET",
            self._reset,
            bg_color=(255, 100, 100),
            hover_color=(255, 130, 130)
        ))
        start_x += 100 + button_spacing

        # MODE toggle butonu
        self.buttons.append(Button(
            start_x, button_y1, 140, button_height,
            "<> MODE",
            self._toggle_mode_btn,
            bg_color=(100, 150, 255),
            hover_color=(120, 170, 255)
        ))
        start_x += 140 + button_spacing

        # PREV SAMPLE butonu
        self.buttons.append(Button(
            start_x, button_y1, 100, button_height,
            "< PREV",
            self._prev_sample_btn,
            bg_color=(150, 100, 255),
            hover_color=(170, 120, 255)
        ))
        start_x += 100 + button_spacing

        # NEXT SAMPLE butonu
        self.buttons.append(Button(
            start_x, button_y1, 100, button_height,
            "NEXT >",
            self._next_sample_btn,
            bg_color=(150, 100, 255),
            hover_color=(170, 120, 255)
        ))
        start_x += 100 + button_spacing

        # SPEED - butonu
        self.buttons.append(Button(
            start_x, button_y1, 80, button_height,
            "SPEED -",
            self._decrease_speed,
            bg_color=(255, 150, 50),
            hover_color=(255, 170, 70)
        ))
        start_x += 80 + button_spacing

        # SPEED + butonu
        self.buttons.append(Button(
            start_x, button_y1, 80, button_height,
            "SPEED +",
            self._increase_speed,
            bg_color=(255, 150, 50),
            hover_color=(255, 170, 70)
        ))
        start_x += 80 + button_spacing

        # LAYOUT toggle butonu
        self.buttons.append(Button(
            start_x, button_y1, 100, button_height,
            "LAYOUT",
            self._toggle_layout_btn,
            bg_color=(200, 100, 200),
            hover_color=(220, 120, 220)
        ))

        # İkinci satır (alt) - Edit/Visual butonları
        button_y2 = self.window_height - 45
        start_x = 20

        # EDIT MODE butonu
        self.edit_mode_btn = Button(
            start_x, button_y2, 150, button_height,
            "EDIT: OFF",
            self._toggle_edit_mode,
            bg_color=(70, 70, 70),
            hover_color=(100, 100, 100)
        )
        self.buttons.append(self.edit_mode_btn)
        start_x += 150 + button_spacing

        # HEATMAP butonu
        self.heatmap_btn = Button(
            start_x, button_y2, 150, button_height,
            "HEATMAP: OFF",
            self._toggle_heatmap,
            bg_color=(70, 70, 70),
            hover_color=(100, 100, 100)
        )
        self.buttons.append(self.heatmap_btn)
        start_x += 150 + button_spacing

        # Top right buttons - History and Browse
        button_y_top = 20
        button_spacing_top = 10

        # BROWSE PUZZLES butonu - Sağ üst köşe (en sağda)
        browse_btn_width = 180
        browse_btn_x = self.window_width - browse_btn_width - 20
        self.buttons.append(Button(
            browse_btn_x, button_y_top, browse_btn_width, button_height,
            "BROWSE PUZZLES",
            self._open_browser,
            bg_color=(100, 100, 200),
            hover_color=(120, 120, 220)
        ))

        # HISTORY butonu - BROWSE butonunun solunda
        history_btn_width = 150
        history_btn_x = browse_btn_x - history_btn_width - button_spacing_top
        self.buttons.append(Button(
            history_btn_x, button_y_top, history_btn_width, button_height,
            "HISTORY",
            self._open_history,
            bg_color=(200, 100, 100),
            hover_color=(220, 120, 120)
        ))

    def _toggle_pause(self):
        """START/PAUSE toggle"""
        self.paused = not self.paused
        # Buton tekstini güncelle
        self.start_pause_btn.text = "> START" if self.paused else "|| PAUSE"
        self.start_pause_btn.bg_color = (46, 204, 64) if self.paused else (255, 200, 0)
        self.start_pause_btn.hover_color = (60, 220, 80) if self.paused else (255, 220, 50)

    def _reset(self):
        """RESET"""
        self.should_reset = True

    def _toggle_mode_btn(self):
        """MODE toggle"""
        self.toggle_mode = True

    def _prev_sample_btn(self):
        """PREV SAMPLE"""
        self.prev_sample = True

    def _next_sample_btn(self):
        """NEXT SAMPLE"""
        self.next_sample = True

    def _decrease_speed(self):
        """SPEED azalt"""
        self.fps = max(1, self.fps - 5)

    def _increase_speed(self):
        """SPEED artır"""
        self.fps = min(60, self.fps + 5)

    def _toggle_layout_btn(self):
        """LAYOUT toggle"""
        if self.layout_mode == "horizontal":
            self.layout_mode = "vertical"
        else:
            self.layout_mode = "horizontal"

    def _toggle_edit_mode(self):
        """EDIT MODE toggle"""
        self.grid_editor.edit_mode = not self.grid_editor.edit_mode

        # Buton tekstini ve rengini güncelle
        if self.grid_editor.edit_mode:
            self.edit_mode_btn.text = "EDIT: ON"
            self.edit_mode_btn.bg_color = (50, 150, 50)
            self.edit_mode_btn.hover_color = (60, 180, 60)
        else:
            self.edit_mode_btn.text = "EDIT: OFF"
            self.edit_mode_btn.bg_color = (70, 70, 70)
            self.edit_mode_btn.hover_color = (100, 100, 100)

        print(f"\n[EDIT MODE] {'ENABLED' if self.grid_editor.edit_mode else 'DISABLED'}")

    def _toggle_heatmap(self):
        """HEATMAP toggle"""
        self.heatmap_overlay.enabled = not self.heatmap_overlay.enabled

        # Buton tekstini ve rengini güncelle
        if self.heatmap_overlay.enabled:
            self.heatmap_btn.text = "HEATMAP: ON"
            self.heatmap_btn.bg_color = (150, 50, 50)
            self.heatmap_btn.hover_color = (180, 60, 60)
        else:
            self.heatmap_btn.text = "HEATMAP: OFF"
            self.heatmap_btn.bg_color = (70, 70, 70)
            self.heatmap_btn.hover_color = (100, 100, 100)

        print(f"\n[HEATMAP] {'ENABLED' if self.heatmap_overlay.enabled else 'DISABLED'}")

    def _open_browser(self):
        """Open puzzle browser"""
        self.open_browser_flag = True
        print("\n[BROWSER] Opening puzzle browser...")

    def _open_history(self):
        """Open episode history"""
        self.open_history_flag = True
        print("\n[HISTORY] Opening episode history...")

    def handle_events(self, current_grid=None) -> Dict[str, bool]:
        """
        Pygame event'lerini işle

        Args:
            current_grid: Reference to the current grid for editing

        Returns:
            Control flags dict
        """
        # Update current grid reference
        if current_grid is not None:
            self.current_grid_reference = current_grid

        for event in pygame.event.get():
            # Color palette event'lerini handle et (önce)
            if self.color_palette.handle_event(event):
                continue

            # Grid editor event'lerini handle et (color palette'ten sonra)
            if self.current_grid_reference is not None and self.grid_editor.handle_click(
                event,
                self.color_palette.selected_color,
                self.current_grid_reference,
                self.grid_rects
            ):
                continue

            # Önce buton event'lerini handle et
            for button in self.buttons:
                if button.handle_event(event):
                    break  # Bir buton tıklandıysa diğerlerine bakma

            if event.type == pygame.QUIT:
                self.should_quit = True

            elif event.type == pygame.VIDEORESIZE:
                # Pencere boyutu değiştirildiğinde
                new_width = max(event.w, self.min_width)
                new_height = max(event.h, self.min_height)

                self.window_width = new_width
                self.window_height = new_height

                # Ekranı yeni boyutla yeniden oluştur
                self.screen = pygame.display.set_mode(
                    (new_width, new_height),
                    pygame.RESIZABLE
                )

                # Butonları yeniden konumlandır
                self.buttons.clear()
                self._create_buttons()

            elif event.type == pygame.MOUSEWHEEL:
                # Mouse wheel ile zoom
                if event.y > 0:  # Scroll up = zoom in
                    self._zoom_in()
                elif event.y < 0:  # Scroll down = zoom out
                    self._zoom_out()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self._toggle_pause()  # Buton ile aynı fonksiyonu kullan

                elif event.key == pygame.K_RIGHT:
                    if self.paused:
                        self.step_mode = True

                elif event.key == pygame.K_r:
                    self.should_reset = True

                elif event.key in [pygame.K_q, pygame.K_ESCAPE]:
                    self.should_quit = True

                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.fps = min(60, self.fps + 5)

                elif event.key == pygame.K_MINUS:
                    self.fps = max(1, self.fps - 5)

                # Zoom kontrolleri
                elif event.key == pygame.K_z:
                    self._zoom_in()

                elif event.key == pygame.K_x:
                    self._zoom_out()

                elif event.key == pygame.K_a:
                    # Auto-fit toggle
                    self.auto_fit_enabled = not self.auto_fit_enabled

                elif event.key == pygame.K_l:
                    # Layout toggle
                    if self.layout_mode == "horizontal":
                        self.layout_mode = "vertical"
                    else:
                        self.layout_mode = "horizontal"

                elif event.key == pygame.K_t:
                    # Mode toggle (Train ↔ Test)
                    self.toggle_mode = True

                elif event.key == pygame.K_n:
                    # Next sample
                    self.next_sample = True

                elif event.key == pygame.K_LEFT:
                    # Previous sample
                    self.prev_sample = True

                elif event.key == pygame.K_e:
                    # Toggle edit mode
                    self.grid_editor.edit_mode = not self.grid_editor.edit_mode
                    print(f"\n[EDIT MODE] {'ENABLED' if self.grid_editor.edit_mode else 'DISABLED'}")

                elif event.key == pygame.K_h:
                    # Toggle heatmap overlay
                    self.heatmap_overlay.enabled = not self.heatmap_overlay.enabled
                    print(f"\n[HEATMAP] {'ENABLED' if self.heatmap_overlay.enabled else 'DISABLED'}")

        return {
            'quit': self.should_quit,
            'reset': self.should_reset,
            'paused': self.paused,
            'step': self.step_mode,
            'toggle_mode': self.toggle_mode,
            'next_sample': self.next_sample,
            'prev_sample': self.prev_sample,
            'open_browser': self.open_browser_flag,
            'open_history': self.open_history_flag
        }

    def _zoom_in(self):
        """Zoom in (hücreleri büyüt)"""
        self.auto_fit_enabled = False  # Manuel zoom yapıldığında auto-fit'i kapat
        new_cell_size = self.current_cell_size * (1 + self.zoom_step)
        self.current_cell_size = min(new_cell_size, self.max_cell_size)
        self.zoom_level = self.current_cell_size / self.base_cell_size

    def _zoom_out(self):
        """Zoom out (hücreleri küçült)"""
        self.auto_fit_enabled = False  # Manuel zoom yapıldığında auto-fit'i kapat
        new_cell_size = self.current_cell_size * (1 - self.zoom_step)
        self.current_cell_size = max(new_cell_size, self.min_cell_size)
        self.zoom_level = self.current_cell_size / self.base_cell_size

    def _calculate_auto_fit_cell_size(self, grid_height: int, grid_width: int) -> int:
        """
        Grid boyutuna göre otomatik cell_size hesapla

        Args:
            grid_height: Grid yüksekliği (hücre sayısı)
            grid_width: Grid genişliği (hücre sayısı)

        Returns:
            Hesaplanan cell_size
        """
        if not self.auto_fit_enabled:
            return int(self.current_cell_size)

        # Grid'lerin çizileceği alan
        header_height = 150
        controls_height = 100

        available_height = self.window_height - header_height - controls_height - 80
        available_width = self.window_width - 100

        if self.layout_mode == "horizontal":
            # Yan yana: 3 grid yan yana
            vertical_padding = 40
            available_height_per_grid = available_height - vertical_padding
            max_width_per_grid = available_width / 3

            cell_size_by_height = available_height_per_grid / grid_height
            cell_size_by_width = max_width_per_grid / grid_width

        else:  # vertical
            # Alt alta: 3 grid alt alta
            horizontal_padding = 100
            available_width_per_grid = available_width - horizontal_padding
            max_height_per_grid = available_height / 3

            cell_size_by_height = max_height_per_grid / grid_height
            cell_size_by_width = available_width_per_grid / grid_width

        # İkisinden küçük olanı al (her ikisine de sığmalı)
        optimal_cell_size = min(cell_size_by_height, cell_size_by_width)

        # Min/max sınırları içinde tut
        optimal_cell_size = max(self.min_cell_size, min(optimal_cell_size, self.max_cell_size))

        return int(optimal_cell_size)

    def render(
        self,
        input_grid: np.ndarray,
        current_grid: np.ndarray,
        target_grid: np.ndarray,
        info: Dict
    ):
        """
        Tüm sahneyi render et

        Args:
            input_grid: Başlangıç grid'i
            current_grid: Mevcut grid
            target_grid: Hedef grid
            info: Environment ve agent bilgileri
        """
        # Ekranı temizle
        self.screen.fill(self.bg_color)

        # Üst bilgi çubuğu (header)
        self._draw_header(info)

        # Auto-fit cell size hesapla (eğer aktifse)
        if self.auto_fit_enabled:
            # En büyük grid boyutunu bul
            max_height = max(input_grid.shape[0], current_grid.shape[0], target_grid.shape[0])
            max_width = max(input_grid.shape[1], current_grid.shape[1], target_grid.shape[1])
            self.current_cell_size = self._calculate_auto_fit_cell_size(max_height, max_width)

        # Set current grid reference for editor
        self.current_grid_reference = current_grid

        # Layout mode'a göre grid'leri çiz
        if self.layout_mode == "horizontal":
            self._draw_grids_horizontal(input_grid, current_grid, target_grid)
        else:
            self._draw_grids_vertical(input_grid, current_grid, target_grid)

        # Heatmap overlay çiz (CURRENT grid üzerine)
        if 'current' in self.grid_rects:
            self.heatmap_overlay.draw_heatmap(self.screen, self.grid_rects['current'])

        # Grid editor hover highlight çiz (grid'lerin üstüne)
        if self.grid_editor.edit_mode:
            self.grid_editor.draw_hover_highlight(self.screen, self.grid_rects)

        # Color palette çiz (sol tarafta)
        if self.grid_editor.edit_mode:
            self.color_palette.draw(self.screen, self.font_small)

        # Edit mode indicator çiz
        self._draw_edit_mode_indicator()

        # Heatmap indicator çiz
        self._draw_heatmap_indicator()

        # Alt kontrol çubuğu
        self._draw_controls()

        # Ekranı güncelle
        pygame.display.flip()

        # FPS kontrolü
        if not self.paused or self.step_mode:
            self.clock.tick(self.fps)
            self.step_mode = False
        else:
            self.clock.tick(30)  # Pause modunda daha düşük FPS

    def _draw_grids_horizontal(self, input_grid: np.ndarray, current_grid: np.ndarray, target_grid: np.ndarray):
        """Grid'leri yan yana (horizontal) çiz"""
        grid_y = 180

        # Grid spacing dinamik
        available_width = self.window_width - 100
        grid_width = current_grid.shape[1] * self.current_cell_size

        # 3 grid için gereken toplam genişlik
        total_grids_width = grid_width * 3
        # Kalan boşluğu grid'ler arasına eşit dağıt
        grid_spacing = max(50, (available_width - total_grids_width) // 4)

        # Grid'leri ortala
        total_width_with_spacing = total_grids_width + 2 * grid_spacing
        start_x = (self.window_width - total_width_with_spacing) // 2

        # Input grid
        input_x = start_x
        self._draw_grid(input_grid, input_x, grid_y, "INPUT")

        # Current grid (agent'ın üzerinde çalıştığı)
        current_x = input_x + grid_width + grid_spacing
        self._draw_grid(current_grid, current_x, grid_y, "CURRENT", highlight=True)

        # Target grid
        target_x = current_x + grid_width + grid_spacing
        self._draw_grid(target_grid, target_x, grid_y, "TARGET")

    def _draw_grids_vertical(self, input_grid: np.ndarray, current_grid: np.ndarray, target_grid: np.ndarray):
        """Grid'leri alt alta (vertical) çiz"""
        start_y = 180

        # Grid spacing dinamik
        available_height = self.window_height - 180 - 100  # Header ve controls için alan
        grid_height = current_grid.shape[0] * self.current_cell_size

        # 3 grid için gereken toplam yükseklik
        total_grids_height = grid_height * 3
        # Kalan boşluğu grid'ler arasına eşit dağıt
        grid_spacing = max(30, (available_height - total_grids_height) // 4)

        # Grid'leri yatayda ortala
        grid_width = current_grid.shape[1] * self.current_cell_size
        grid_x = (self.window_width - grid_width) // 2

        # Input grid
        input_y = start_y
        self._draw_grid(input_grid, grid_x, input_y, "INPUT")

        # Current grid (agent'ın üzerinde çalıştığı)
        current_y = input_y + grid_height + grid_spacing
        self._draw_grid(current_grid, grid_x, current_y, "CURRENT", highlight=True)

        # Target grid
        target_y = current_y + grid_height + grid_spacing
        self._draw_grid(target_grid, grid_x, target_y, "TARGET")

    def _draw_grid(
        self,
        grid: np.ndarray,
        x: int,
        y: int,
        label: str,
        highlight: bool = False
    ):
        """Bir grid'i çiz"""
        # Label
        label_surface = self.font_medium.render(label, True, self.text_color)
        self.screen.blit(label_surface, (x, y - 35))

        # Grid border
        grid_height, grid_width = grid.shape
        cell_size = int(self.current_cell_size)
        border_rect = pygame.Rect(
            x - 2,
            y - 2,
            grid_width * cell_size + 4,
            grid_height * cell_size + 4
        )

        border_color = (255, 200, 0) if highlight else self.border_color
        pygame.draw.rect(self.screen, border_color, border_rect, 2)

        # Grid cells
        for i in range(grid_height):
            for j in range(grid_width):
                cell_value = int(grid[i, j])
                color = ARC_COLORS.get(cell_value, (128, 128, 128))

                cell_rect = pygame.Rect(
                    x + j * cell_size,
                    y + i * cell_size,
                    cell_size,
                    cell_size
                )

                pygame.draw.rect(self.screen, color, cell_rect)
                pygame.draw.rect(self.screen, self.border_color, cell_rect, 1)

        # Save grid rect for editor (if this is the CURRENT grid)
        if label == "CURRENT":
            self.grid_rects['current'] = {
                'x': x,
                'y': y,
                'cell_size': cell_size,
                'shape': (grid_height, grid_width)
            }

    def _draw_status_panel(self, x: int, y: int, info: Dict):
        """Sağ taraftaki status panelini çiz"""
        panel_width = 300
        panel_height = 500

        # Panel arka planı
        panel_rect = pygame.Rect(x, y, panel_width, panel_height)
        pygame.draw.rect(self.screen, self.panel_color, panel_rect)
        pygame.draw.rect(self.screen, self.border_color, panel_rect, 2)

        # Başlık
        title = self.font_medium.render("STATUS", True, self.text_color)
        self.screen.blit(title, (x + 10, y + 10))

        # Bilgiler
        line_y = y + 50
        line_height = 30

        status_lines = [
            f"Steps: {info.get('steps', 0)} / {info.get('max_steps', 0)}",
            f"Total Reward: {info.get('total_reward', 0):.2f}",
            f"Last Reward: {info.get('last_reward', 0):.2f}",
            "",
            f"Solved: {'YES' if info.get('is_solved', False) else 'NO'}",
            f"Done: {'YES' if info.get('done', False) else 'NO'}",
            "",
            f"Agent: {info.get('agent_type', 'Unknown')}",
            f"Actions Taken: {info.get('agent_actions', 0)}",
        ]

        # Son action
        if info.get('last_action_decoded'):
            x_pos, y_pos, color = info['last_action_decoded']
            status_lines.append("")
            status_lines.append("Last Action:")
            status_lines.append(f"  Position: ({x_pos}, {y_pos})")
            status_lines.append(f"  Color: {color}")

        for line in status_lines:
            text_surface = self.font_small.render(line, True, self.text_color)
            self.screen.blit(text_surface, (x + 15, line_y))
            line_y += line_height

    def _draw_header(self, info: Dict):
        """Üst bilgi çubuğunu çiz"""
        # Header paneli
        header_height = 130
        header_rect = pygame.Rect(0, 0, self.window_width, header_height)
        pygame.draw.rect(self.screen, (40, 40, 40), header_rect)
        pygame.draw.line(self.screen, self.border_color, (0, header_height), (self.window_width, header_height), 2)

        # Başlık
        title = self.font_large.render("ARC-AGI Reinforcement Learning Playground", True, self.text_color)
        self.screen.blit(title, (20, 20))

        # Alt başlık
        subtitle = self.font_small.render("Train and visualize RL agents on ARC puzzles", True, (150, 150, 150))
        self.screen.blit(subtitle, (20, 60))

        # Puzzle bilgileri - daha aşağıda
        puzzle_id = info.get('puzzle_id', 'Unknown')
        puzzle_text = self.font_medium.render(f"Puzzle ID: {puzzle_id}", True, (100, 200, 255))
        self.screen.blit(puzzle_text, (20, 100))

        # Dataset bilgisi
        dataset = info.get('dataset', 'training')
        dataset_text = self.font_small.render(f"Dataset: {dataset}", True, (180, 180, 180))
        self.screen.blit(dataset_text, (250, 105))

        # Mode bilgisi (TRAIN veya TEST)
        mode = info.get('mode', 'train').upper()
        mode_color = (100, 255, 100) if mode == 'TRAIN' else (255, 180, 100)
        mode_text = self.font_medium.render(f"Mode: {mode}", True, mode_color)
        self.screen.blit(mode_text, (420, 100))

        # Sample bilgisi (sadece train mode'da)
        if info.get('mode') == 'train':
            sample_idx = info.get('train_sample_index', 0)
            num_samples = info.get('num_train_samples', 1)
            sample_text = self.font_small.render(f"Sample: {sample_idx + 1}/{num_samples}", True, (180, 180, 180))
            self.screen.blit(sample_text, (620, 105))

            # Mevcut sample'ın durumu (metrics'ten)
            if 'metrics' in info and 'current_sample' in info['metrics']:
                sample_metrics = info['metrics']['current_sample']

                # Accuracy
                accuracy = sample_metrics.get('avg_accuracy', 0)
                best_acc_text = f"Accuracy: {accuracy:.1f}%"

                # Color coding: >95% yeşil, 50-95% sarı, <50% kırmızı
                if accuracy >= 95:
                    acc_color = (100, 255, 100)  # Yeşil
                elif accuracy >= 50:
                    acc_color = (255, 220, 100)  # Sarı
                else:
                    acc_color = (255, 100, 100)  # Kırmızı

                acc_surface = self.font_small.render(best_acc_text, True, acc_color)
                self.screen.blit(acc_surface, (750, 105))

                # Solved status
                solved = sample_metrics.get('solved', 0) > 0
                status_text = "[OK] SOLVED" if solved else "[X] Not Solved"
                status_color = (100, 255, 100) if solved else (255, 100, 100)
                status_surface = self.font_small.render(status_text, True, status_color)
                self.screen.blit(status_surface, (880, 105))

    def _draw_agent_params_panel(self, x: int, y: int, info: Dict):
        """Agent parametreleri panelini çiz"""
        # Panel genişliği dinamik - pencere genişliğinin %65'i veya minimum 800
        panel_width = max(800, int(self.window_width * 0.65))

        # Panel yüksekliği - collapsed ise küçük
        panel_height = 40 if self.agent_params_collapsed else 150

        # Panel arka planı
        panel_rect = pygame.Rect(x, y, panel_width, panel_height)
        pygame.draw.rect(self.screen, self.panel_color, panel_rect)
        pygame.draw.rect(self.screen, self.border_color, panel_rect, 2)

        # Başlık
        collapse_indicator = "▼" if self.agent_params_collapsed else "▲"
        title_text = f"AGENT PARAMETERS {collapse_indicator} (Press P to toggle)"
        title = self.font_medium.render(title_text, True, self.text_color)
        self.screen.blit(title, (x + 10, y + 10))

        # Eğer collapsed ise sadece başlığı göster
        if self.agent_params_collapsed:
            return

        # Agent parametreleri - dinamik olarak info'dan alınacak
        agent_params = info.get('agent_params', {})

        # Parametreleri 3 sütunda göster - dinamik sütun pozisyonları
        col1_x = x + 20
        col2_x = x + int(panel_width * 0.35)  # Panel genişliğinin %35'i
        col3_x = x + int(panel_width * 0.68)  # Panel genişliğinin %68'i
        param_y = y + 50
        line_height = 30

        # Sütun 1
        self._draw_param_line(col1_x, param_y, "Agent Type:", info.get('agent_type', 'Unknown'))
        self._draw_param_line(col1_x, param_y + line_height, "Actions Taken:", str(info.get('agent_actions', 0)))

        # Sütun 2
        self._draw_param_line(col2_x, param_y, "Learning Rate:", str(agent_params.get('learning_rate', 'N/A')))
        self._draw_param_line(col2_x, param_y + line_height, "Epsilon:", str(agent_params.get('epsilon', 'N/A')))
        self._draw_param_line(col2_x, param_y + 2*line_height, "Discount (γ):", str(agent_params.get('gamma', 'N/A')))

        # Sütun 3
        self._draw_param_line(col3_x, param_y, "Batch Size:", str(agent_params.get('batch_size', 'N/A')))
        self._draw_param_line(col3_x, param_y + line_height, "Memory Size:", str(agent_params.get('memory_size', 'N/A')))
        self._draw_param_line(col3_x, param_y + 2*line_height, "Update Freq:", str(agent_params.get('update_frequency', 'N/A')))

        # Not: Gelecekte buraya düzenleme kontrolleri eklenebilir
        hint_text = "Note: Parameter editing will be available in future updates"
        hint_surface = self.font_small.render(hint_text, True, (120, 120, 120))
        self.screen.blit(hint_surface, (x + 20, y + panel_height - 30))

    def _draw_param_line(self, x: int, y: int, label: str, value: str):
        """Bir parametre satırı çiz"""
        # Label (sol tarafta)
        label_surface = self.font_small.render(label, True, (200, 200, 200))
        self.screen.blit(label_surface, (x, y))

        # Value (sağ tarafta)
        value_surface = self.font_small.render(value, True, (100, 255, 100))
        self.screen.blit(value_surface, (x + 150, y))

    def _draw_edit_mode_indicator(self):
        """Draw edit mode indicator in top-right corner"""
        if self.grid_editor.edit_mode:
            # Background panel
            panel_width = 200
            panel_height = 80
            panel_x = self.window_width - panel_width - 20
            panel_y = 150

            panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
            pygame.draw.rect(self.screen, (50, 150, 50), panel_rect)
            pygame.draw.rect(self.screen, (100, 255, 100), panel_rect, 2)

            # Title
            title_text = "EDIT MODE: ON"
            title_surface = self.font_medium.render(title_text, True, (255, 255, 255))
            title_rect = title_surface.get_rect(center=(panel_x + panel_width // 2, panel_y + 20))
            self.screen.blit(title_surface, title_rect)

            # Instructions
            inst1 = "Click CURRENT grid"
            inst2 = "Press E to exit"
            inst1_surface = self.font_tiny.render(inst1, True, (200, 200, 200))
            inst2_surface = self.font_tiny.render(inst2, True, (200, 200, 200))

            self.screen.blit(inst1_surface, (panel_x + 10, panel_y + 45))
            self.screen.blit(inst2_surface, (panel_x + 10, panel_y + 60))

    def _draw_heatmap_indicator(self):
        """Draw heatmap indicator"""
        if self.heatmap_overlay.enabled:
            # Small indicator in top-right
            panel_width = 200
            panel_height = 60
            panel_x = self.window_width - panel_width - 20
            panel_y = 240  # Below edit mode indicator

            panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
            pygame.draw.rect(self.screen, (150, 50, 50), panel_rect)
            pygame.draw.rect(self.screen, (255, 100, 100), panel_rect, 2)

            # Title
            title_text = "HEATMAP: ON"
            title_surface = self.font_medium.render(title_text, True, (255, 255, 255))
            title_rect = title_surface.get_rect(center=(panel_x + panel_width // 2, panel_y + 20))
            self.screen.blit(title_surface, title_rect)

            # Instructions
            inst = "Press H to hide"
            inst_surface = self.font_tiny.render(inst, True, (200, 200, 200))
            self.screen.blit(inst_surface, (panel_x + 10, panel_y + 40))

    def _draw_controls(self):
        """Alt kontrol çubuğunu çiz - butonlar ile"""
        y_pos = self.window_height - 100

        # Panel
        panel_rect = pygame.Rect(0, y_pos, self.window_width, 100)
        pygame.draw.rect(self.screen, self.panel_color, panel_rect)
        pygame.draw.line(self.screen, self.border_color, (0, y_pos), (self.window_width, y_pos), 2)

        # Butonları çiz
        for button in self.buttons:
            button.draw(self.screen)

        # Sağ tarafta bilgi göstergeleri
        info_x = self.window_width - 250

        # Status indicator
        status_text = "|| PAUSED" if self.paused else "> RUNNING"
        status_color = (255, 200, 0) if self.paused else (46, 204, 64)
        status_surface = self.font_medium.render(status_text, True, status_color)
        self.screen.blit(status_surface, (info_x, y_pos + 10))

        # FPS
        fps_text = f"FPS: {self.fps}"
        fps_surface = self.font_small.render(fps_text, True, (180, 180, 180))
        self.screen.blit(fps_surface, (info_x, y_pos + 40))

        # Layout
        layout_text = f"Layout: {self.layout_mode.upper()}"
        layout_surface = self.font_small.render(layout_text, True, (180, 180, 180))
        self.screen.blit(layout_surface, (info_x, y_pos + 65))

        # Klavye kısayolları (küçük ipucu)
        hint_text = "Keyboard: SPACE=Pause, R=Reset, Q=Quit, Z/X=Zoom, A=Auto-fit"
        hint_surface = self.font_tiny.render(hint_text, True, (100, 100, 100))
        self.screen.blit(hint_surface, (20, y_pos + 80))

    def _draw_info_box(self, x: int, y: int, text: str, color: Tuple[int, int, int]):
        """Bilgi kutusu çiz"""
        text_surface = self.font_tiny.render(text, True, color)
        self.screen.blit(text_surface, (x, y))

    def close(self):
        """Pygame'i kapat"""
        pygame.quit()

    def reset_control_flags(self):
        """Control flag'lerini sıfırla"""
        self.should_reset = False
        self.step_mode = False
        self.toggle_mode = False
        self.next_sample = False
        self.prev_sample = False
        self.open_browser_flag = False
        self.open_history_flag = False

    def record_action(self, action_decoded: tuple):
        """Record an agent action for heatmap tracking"""
        x, y, color = action_decoded
        self.heatmap_overlay.record_modification(x, y)

    def reset_heatmap(self):
        """Reset heatmap when episode resets"""
        self.heatmap_overlay.reset()
