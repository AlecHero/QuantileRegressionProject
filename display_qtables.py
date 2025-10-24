import pygame
import numpy as np

## Created with ChatGPT

# def value_to_color(v, vmin=-1, vmax=1):
#     v = max(min(v, vmax), vmin)  # clamp
#     if v == 0:
#         return (0, 0, 0)
#     elif v > 0:
#         return (0, int(255 * v / vmax), 0)  # green scale
#     else:
#         return (int(255 * -v / -vmin), 0, 0)  # red scale

import matplotlib.cm as cm
import matplotlib.colors as mcolors

# create reusable colormap
blues = cm.get_cmap("Blues")

def value_to_color(val, vmin, vmax):
    # normalize between 0–1
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    r, g, b, _ = blues(norm(val))
    return (int(r*255), int(g*255), int(b*255))

def blend_color(base, target=(255, 0, 0), alpha=0.6):
    return tuple(
        int(base[i] * (1 - alpha) + target[i] * alpha) for i in range(3)
    )

def draw_state(screen, x, y, qvals, vmin, vmax, GRID_SIZE, decimals=0, font_size=16, optimal_action=None):
    cx, cy = x * GRID_SIZE, y * GRID_SIZE
    half = GRID_SIZE // 2

    triangles = {
        0: [(cx, cy), (cx+GRID_SIZE, cy), (cx+half, cy+half)],           # up
        1: [(cx+GRID_SIZE, cy), (cx+GRID_SIZE, cy+GRID_SIZE), (cx+half, cy+half)],  # right
        2: [(cx+GRID_SIZE, cy+GRID_SIZE), (cx, cy+GRID_SIZE), (cx+half, cy+half)],  # down
        3: [(cx, cy+GRID_SIZE), (cx, cy), (cx+half, cy+half)],           # left
    }

    font = pygame.font.SysFont("Courier", font_size, bold=True)
    offsets = {0: (0, -half//2), 2: (0, half//2), 1: (half//2, 0), 3: (-half//2, 0)}

    for a, pts in triangles.items():
        val = qvals[a]
        base_color = value_to_color(val, vmin, vmax)

        if a == optimal_action:
            fill_color = blend_color(base_color, target=(255,0,0), alpha=0.2)
        else:
            fill_color = base_color

        brightness = 0.2126 * fill_color[0] + 0.7152 * fill_color[1] + 0.0722 * fill_color[2]
        text_color = (255, 255, 255) if brightness < 128 else (0, 0, 0)
        if optimal_action == a: text_color = (255, 0, 0)

        pygame.draw.polygon(screen, fill_color, pts)
        # pygame.draw.polygon(screen, (0, 0, 0), pts[1:], 1)

        ox, oy = offsets[a]
        text = font.render(f"{val:.{decimals}f}", True, text_color)
        rect = text.get_rect(center=(cx + half + ox, cy + half + oy))
        screen.blit(text, rect)
    
    pygame.draw.polygon(screen, (0, 0, 0), [(cx, cy), (cx+GRID_SIZE, cy), (cx+GRID_SIZE, cy+GRID_SIZE), (cx, cy+GRID_SIZE)], 2)


import pygame
import numpy as np

def display_qtables(qtables, params):
    pygame.init()
    screen_info = pygame.display.Info()
    screen_w, screen_h = screen_info.current_w, screen_info.current_h

    map_w, map_h = params.map_size
    margin_ratio = 0.9
    grid_size_w = (screen_w * margin_ratio) / map_h
    grid_size_h = (screen_h * margin_ratio) / map_w
    grid_size = int(min(grid_size_w, grid_size_h))

    if grid_size < 20: grid_size = 20
    font_size = max(10, min(20, int(grid_size * 0.35)))

    actions = [0, 1, 2, 3]
    fps = 30

    screen = pygame.display.set_mode((map_h * grid_size, map_w * grid_size))
    clock = pygame.time.Clock()

    qt_index = 0
    max_index = qtables.shape[0] - 1
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    qt_index = min(qt_index + 1, max_index)
                elif event.y < 0:
                    qt_index = max(qt_index - 1, 0)
                pygame.display.set_caption(f"Q-Table Viewer [{qt_index + 1}/{max_index + 1}]")

        screen.fill((50, 50, 50))
        qtable = qtables[qt_index]
        vmin, vmax = qtable.min(), qtable.max()
        # decimals = 2 if abs(vmax - vmin) < 2 else 1
        decimals = 1

        for x in range(map_w):
            for y in range(map_h):
                state = np.ravel_multi_index((x, y), params.map_size)
                qvals = {a: qtable[state, a] for a in actions}
                optimal_action = max(qvals, key=qvals.get)
                draw_state(
                    screen, y, x, qvals, vmin, vmax,
                    GRID_SIZE=grid_size, decimals=decimals, font_size=font_size, optimal_action=optimal_action
                )

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()
