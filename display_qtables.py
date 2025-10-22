import pygame
import numpy as np

## Created with ChatGPT

def value_to_color(v, vmin=-1, vmax=1):
    v = max(min(v, vmax), vmin)  # clamp
    if v == 0:
        return (0, 0, 0)
    elif v > 0:
        return (0, int(255 * v / vmax), 0)  # green scale
    else:
        return (int(255 * -v / -vmin), 0, 0)  # red scale


def draw_state(screen, x, y, qvals, vmin, vmax, GRID_SIZE, decimals=0):
    cx, cy = x * GRID_SIZE, y * GRID_SIZE
    half = GRID_SIZE // 2

    triangles = {
        0: [(cx, cy), (cx+GRID_SIZE, cy), (cx+half, cy+half)],
        2: [(cx, cy+GRID_SIZE), (cx+GRID_SIZE, cy+GRID_SIZE), (cx+half, cy+half)],
        1: [(cx+GRID_SIZE, cy), (cx+GRID_SIZE, cy+GRID_SIZE), (cx+half, cy+half)],
        3: [(cx, cy), (cx, cy+GRID_SIZE), (cx+half, cy+half)],
    }

    font = pygame.font.SysFont("Courier", 16, bold=True)
    offsets = {0: (0, -half//2), 2: (0, half//2), 1: (half//2, 0), 3: (-half//2, 0)}

    for a, pts in triangles.items():
        val = qvals[a]
        pygame.draw.polygon(screen, value_to_color(val, vmin, vmax), pts)
        pygame.draw.polygon(screen, (0,0,0), pts, 1)

        ox, oy = offsets[a]
        text = font.render(f"{val:.{decimals}f}", True, (255,255,255))
        rect = text.get_rect(center=(cx+half+ox, cy+half+oy))
        screen.blit(text, rect)


def display_qtables(qtables, map_size, decimals=0, grid_size=80):
    GRID_SIZE = grid_size
    GRID_W, GRID_H = map_size
    ACTIONS = [0, 1, 2, 3]
    FPS = 30
    
    pygame.init()
    screen = pygame.display.set_mode((GRID_H*GRID_SIZE, GRID_W*GRID_SIZE))
    clock = pygame.time.Clock()

    # starting index for qtables
    qt_index = 0
    max_index = qtables.shape[0] - 1

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEWHEEL:
                if event.y > 0:  # wheel up
                    qt_index = min(qt_index + 1, max_index)
                elif event.y < 0:  # wheel down
                    qt_index = max(qt_index - 1, 0)

        screen.fill((50,50,50))

        qtable = qtables[qt_index]
        vmin, vmax = qtable.min(), qtable.max()
        
        for x in range(GRID_W):
            for y in range(GRID_H):
                state = np.ravel_multi_index((x,y), map_size)
                qvals = {a: qtable[state,a] for a in ACTIONS}
                draw_state(screen, y, x, qvals, vmin, vmax, GRID_SIZE=GRID_SIZE, decimals=decimals)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()