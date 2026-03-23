import gymnasium as gym
from PIL import Image

# Initialisation de l'environnement avec le mode de rendu pour extraire les images
env = gym.make("LunarLander-v3", render_mode="rgb_array")

print("Espace d'observation (Capteurs) :", env.observation_space)
print("Espace d'action (Moteurs) :", env.action_space)

obs, info = env.reset()
done = False
frames = []

# Initialisation de nos métriques de télémétrie
total_reward = 0.0
main_engine_uses = 0
side_engine_uses = 0

while not done:
    # L'agent choisit une action aléatoire (0: Rien, 1: Gauche, 2: Principal, 3: Droite)
    action = env.action_space.sample()
    
    # L'environnement applique l'action et retourne les nouvelles valeurs
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Mise à jour des métriques
    total_reward += reward
    if action == 2:
        main_engine_uses += 1
    elif action in [1, 3]:
        side_engine_uses += 1
        
    # Capture de l'image courante de la simulation
    frame = env.render()
    frames.append(Image.fromarray(frame))
    
    done = terminated or truncated

env.close()

# Analyse de l'issue du vol (un crash donne -100 à la dernière étape, un succès +100)
if reward == -100:
    issue = "CRASH DETECTE"
elif reward == 100:
    issue = "ATTERRISSAGE REUSSI"
else:
    issue = "TEMPS ECOULE OU SORTIE DE ZONE"

print("\n--- RAPPORT DE VOL ---")
print(f"Issue du vol : {issue}")
print(f"Récompense totale cumulée : {total_reward:.2f} points")
print(f"Allumages moteur principal : {main_engine_uses}")
print(f"Allumages moteurs latéraux : {side_engine_uses}")
print(f"Durée du vol : {len(frames)} frames")

# Sauvegarde de l'animation
if frames:
    frames[0].save('random_agent.gif', save_all=True, append_images=frames[1:], duration=30, loop=0)
    print("Vidéo de la télémétrie sauvegardée sous 'random_agent.gif'")
