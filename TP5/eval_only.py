import gymnasium as gym
from stable_baselines3 import PPO
from PIL import Image

eval_env = gym.make("LunarLander-v3", render_mode="rgb_array")
model = PPO.load("ppo_lunar_lander", device="cpu")

obs, info = eval_env.reset()
done = False
frames = []
total_reward = 0.0
main_engine_uses = 0
side_engine_uses = 0

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    total_reward += reward
    if action == 2:
        main_engine_uses += 1
    elif action in [1, 3]:
        side_engine_uses += 1
    frame = eval_env.render()
    frames.append(Image.fromarray(frame))
    done = terminated or truncated

eval_env.close()

if reward == -100:
    issue = "CRASH DETECTE"
elif reward == 100:
    issue = "ATTERRISSAGE REUSSI"
else:
    issue = "TEMPS ECOULE OU SORTIE DE ZONE"

print("--- RAPPORT DE VOL PPO ---")
print(f"Issue du vol : {issue}")
print(f"Recompense totale cumulee : {total_reward:.2f} points")
print(f"Allumages moteur principal : {main_engine_uses}")
print(f"Allumages moteurs lateraux : {side_engine_uses}")
print(f"Duree du vol : {len(frames)} frames")
