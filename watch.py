import gymnasium as gym
from stable_baselines3 import PPO
import sys
import pygame
import math

mode = sys.argv[1] if len(sys.argv) > 1 else "normal"

# Model load - tera trained model
try:
    model = PPO.load(f"models/{mode}_final")
except:
    model = PPO.load(f"models/{mode}/best_model")

    model = PPO.load("models/normal_final_v2")

env = gym.make("LunarLander-v3", render_mode=None)
obs, _ = env.reset()

pygame.init()
W, H = 900, 650
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("LUNARIS - AI Rocket Landing")
clock = pygame.time.Clock()
font_big = pygame.font.SysFont("Arial", 36, bold=True)
font_small = pygame.font.SysFont("Arial", 22, bold=True)

def draw_background(score):
    # Sunset sky
    for y in range(H):
        r = int(20 + y * 0.12)
        g = int(20 + y * 0.08)
        b = int(60 + y * 0.15)
        pygame.draw.line(screen, (r, g, b), (0, y), (W, y))
    # Stars
    for x, y in [(100, 80), (300, 40), (700, 70), (800, 150), (50, 200)]:
        pygame.draw.circle(screen, (255, 255, 255), (x, y), 2)
    # Ground
    pygame.draw.rect(screen, (60, 60, 65), (0, H-90, W, 90))
    pygame.draw.rect(screen, (100, 100, 105), (0, H-90, W, 5))
    # Landing Pad - video jaisa
    pad_x = W//2
    pygame.draw.ellipse(screen, (255, 200, 0), (pad_x-110, H-110, 220, 35), 4)
    pygame.draw.rect(screen, (255, 255, 255), (pad_x-55, H-105, 110, 6))
    pygame.draw.rect(screen, (255, 255, 255), (pad_x-4, H-125, 8, 40))
    # Score Board - upar left me
    color = (0, 255, 120) if score >= 0 else (255, 60, 60)
    pygame.draw.rect(screen, (0, 0, 0), (20, 20, 200, 70))
    pygame.draw.rect(screen, color, (20, 20, 200, 70), 3)
    txt = font_small.render("SCORE", True, (255, 255, 255))
    sc = font_big.render(f"{score:+.0f}", True, color)
    screen.blit(txt, (35, 28))
    screen.blit(sc, (35, 50))
    # RESCUE PROJECT likh de
    t2 = font_small.render("RESUCE PROJECT", True, (255, 180, 0))
    screen.blit(t2, (W-220, 30))

running = True
for episode in range(5):
    obs, _ = env.reset()
    total = 0
    done = False
    while not done and running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
                done = True
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        total += reward
        if truncated:
            done = True

        # Physics se position nikalo
        x, y, vx, vy, angle, ang_v, l_leg, r_leg = obs
        # Gym coords (-1 to 1) ko screen coords me badlo
        sx = W//2 + int(x * 280)
        sy = H - 110 - int((y + 0.2) * 320)
        sy = max(60, min(H-110, sy))

        draw_background(total)

        # Rocket draw
        # Body
        ang_deg = math.degrees(angle)
        # Simple rocket - white body
        body = pygame.Surface((36, 70), pygame.SRCALPHA)
        pygame.draw.rect(body, (240, 240, 240), (8, 5, 20, 55), border_radius=10)
        pygame.draw.polygon(body, (255, 80, 80), [(8, 12), (18, 0), (28, 12)])
        pygame.draw.circle(body, (30, 30, 30), (18, 30), 6)
        pygame.draw.polygon(body, (40, 40, 40), [(8, 55), (0, 70), (8, 65)])
        pygame.draw.polygon(body, (40, 40, 40), [(28, 55), (36, 70), (28, 65)])

        # Flame - agar main engine chal raha hai
        if action == 2:
            pygame.draw.polygon(body, (255, 160, 0), [(12, 65), (18, 95), (24, 65)])
            pygame.draw.polygon(body, (255, 230, 0), [(14, 65), (18, 82), (22, 65)])

        rotated = pygame.transform.rotate(body, ang_deg)
        rect = rotated.get_rect(center=(sx, sy))
        screen.blit(rotated, rect)

        # Info neeche
        info = font_small.render(f"Landing {episode+1} | Reward: {total:.1f} | Speed: {abs(vy):.1f}", True, (255, 255, 255))
        screen.blit(info, (20, H-35))

        pygame.display.flip()
        clock.tick(50)
    print(f"Landing {episode+1}: {total:.2f}")

env.close()
pygame.quit() 