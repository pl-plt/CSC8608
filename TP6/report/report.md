## CI : IA Explicable et Interprétable

Dans les domaines critiques comme le diagnostic médical, la performance brute d'un modèle d'apprentissage profond (Deep Learning) ne suffit plus. Le déploiement de modèles en production requiert des garanties de transparence pour répondre aux exigences réglementaires et instaurer la confiance avec les experts métiers (médecins, radiologues). Ce TP illustre comment ouvrir la "boîte noire" d'un modèle de Vision par Ordinateur grâce aux méthodes d'explicabilité (XAI) dites spécifiques au modèle (White-box) et post-hoc.

Vous allez auditer un réseau de neurones convolutif (ResNet50) pré-entraîné sur des radiographies pulmonaires. En utilisant la bibliothèque **Captum** (développée par Meta pour l'écosystème PyTorch), vous implémenterez des méthodes d'attribution basées sur les gradients (Grad-CAM et Integrated Gradients) pour comprendre quelles zones spatiales d'une image déclenchent une prédiction. En tant que futurs ingénieurs, vous aborderez également ces méthodes sous un angle MLOps : vous mesurerez leur surcoût de calcul en temps réel et vous traquerez les biais potentiels (l'effet Clever Hans) silencieusement appris par le modèle.

*   Déployer un pipeline d'inférence utilisant un modèle PyTorch/HuggingFace pré-entraîné sur des données médicales.
*   Implémenter **Grad-CAM** pour extraire une explication sémantique de la dernière couche convolutive.
*   Implémenter **Integrated Gradients** pour obtenir une attribution exacte au niveau du pixel, et la lisser avec **SmoothGrad**.
*   Profiler les temps de calcul (Inférence vs Explicabilité) pour évaluer la faisabilité d'un déploiement temps réel.
*   Auditer visuellement le modèle pour détecter un biais de confusion sur une image aberrante.

### Mise en place, Inférence et Grad-CAM

Commencez par créer un dossier **TP6** dans votre dépôt Git de rendu. Placez-vous dans ce dossier et créez un fichier **rapport.md** qui contiendra vos réponses et vos captures d'écran.

```bash
pip install captum transformers matplotlib Pillow
```

Il est fortement conseillé de réaliser ce TP sur le cluster GPU en utilisant des sessions interactives. Notez que l'avertissement concernant use\_fast lors du chargement du processeur HuggingFace est normal et n'impactera pas vos résultats.

Téléchargez plusieurs images radiographiques de test (une saine et deux avec pneumonie) pour éprouver le modèle. Vous pouvez tout à fait ajouter d'autres images de votre choix issues d'internet pour prolonger vos tests !

```bash
wget -O normal_1.jpeg "https://storage.googleapis.com/kagglesdsdata/datasets/17810/23812/chest_xray/train/NORMAL/IM-0115-0001.jpeg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20260226%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260226T194829Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=6044d0e37f8bd56e5f8b5049bb998f745b6c9e37d78bf284089e0592622938bad0df8a5f2c70ee65e0254314ab7e7820ca18adc8cf20f6d7c4b043aa8229a9f6256ff738d235b8e092c363bde388a99687ed77fbf6e32604c6c40a3794bd2cfd14facab2bbc4b8275656d0c65be6462b469b9dac6f9d9d393cdab0b06a27c3d32829d03686ee67667a4372f96132e3687cf19d4f26973b0baaf61267057964b246d0e8c199949b0ad9bdad1e37c29847ff1dc6f57ad1da2a2ded9c58b494d3bb8303229f8169eda7fd56ea30438c59d4c6d10e2d2d04a6be025d525001ab29643166347bc8b0ff9cc30668b9cdda4a0d440a2e79826af91fc81babe26744e488"
wget -O normal_2.jpeg "https://storage.googleapis.com/kagglesdsdata/datasets/17810/23812/chest_xray/train/NORMAL/IM-0117-0001.jpeg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20260226%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260226T194930Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=63ef1b6a053a50225bea449ad7f2f373fff460164a41df1d6d02181489494123a2f3accf2ff69422dca45a739a3b313b220a803e0a26c7158db24d117c2182261eb375552d83f54362f44b58a9cb569b3a72df16cb073c161c4cff6f3d06b811d62dc1fe2f01824eaa39c505bd85f0f3d097e099c9cd8982ea3af4e14abf060be8524d799aaeb27555ac2abe5c0a71448f01d15cf4b8f7c339f1c6ebd48918514df2be649b18f2fa2369461cbe26db4b5f64733474337c735ddac44c440d100365756f145641329b3a1335469fc6d8d776051bb8dcd35b32d96508abf5cc6a9158979020f7b5f210f9604b92f65667c434d64503d0fd7a5faa0422bb93ee714d"
wget -O pneumo_1.jpeg "https://storage.googleapis.com/kagglesdsdata/datasets/17810/23812/chest_xray/train/PNEUMONIA/person1000_bacteria_2931.jpeg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20260224%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260224T035038Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=1e54ee2c1f30cc2ab44209e0b60298dadb8bf75926df454a2eb36163bdf9800eff841d48bca9a7190e6c4be83548d488334d08e036029e8040c8bf5d92be2696df8dd050c9f1337e406c2abf7cee8ee30b89099779063ad9ded3da79008dbfcb3ab5aa1d1edf239a3543edd03bdbc715785ce638a8ef8e3be915b805fafb3d075400318a4f58cd856a342abd80eb29e33128bad9461cda893b5f23085d22a74cb95516667a8fcb1ff447a125af1945a4d68b2cc510b3ec90350e78d764175a4d635c08eff7df0691132fb600e14794a9bfdd63953051fd26e7f82d1ac8e6f034a42a59b97bd7c1ffbc84c6bf504ed46f6c4187c2b658f42ab3cf50ceb1c58de8"
wget -O pneumo_2.jpeg "https://storage.googleapis.com/kagglesdsdata/datasets/17810/23812/chest_xray/train/PNEUMONIA/person1000_virus_1681.jpeg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20260224%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260224T035038Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=a419182488d15620b5f485d9561b2aeeff414e0013d93df69fca0064593139cf296d2a91b6e63ce32282a625b19f3c14ac4a373e89136d5f1167161d7a4b1715b8f33d0b49914512b415ebe0a331e67bca3d361e6f6eda0c62e330ca2487ef748b9ac04c7ee1270199d6b35f690155a86668bd6b8f14acc5c3a1007b1b14f287614a9f07a650c16a733e9d5baca2fb8a6e63a321c850e037bbc8a5652ab3db68880c381953b085ec6011246e6f2e678cab68fe6ddfbc768efd7772493976ce91fa4b79472b29c112336c0def89855c05eedb794c15abf3db184d79b7966a2679643933976dba922280548ca7c1f417522e5a9de80cfe65e3cff411af1c1142b0"
```

Créez le script **01\_gradcam.py**. Complétez le code ci-dessous. Le script utilise sys.argv pour que vous puissiez passer l'image de votre choix en argument lors de l'exécution (ex: python 01\_gradcam.py normal\_1.jpeg).

Dans l'architecture ResNet implémentée par HuggingFace, la dernière couche convolutive s'appelle wrapped\_model.model.resnet.encoder.stages\[-1\].layers\[-1\].

```python
import time
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from captum.attr import LayerGradCam, LayerAttribution

# Wrapper MLOps pour extraire les logits purs (requis par Captum)
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
    def forward(self, x):
        return self.model(x).logits

# 1. Chargement de l'image via argument terminal
image_path = sys.argv[1] if len(sys.argv) > 1 else "normal_1.jpeg"
print(f"Analyse de l'image : {image_path}")
image = Image.open(image_path).convert("RGB")

# 2. Chargement du processeur et du modèle
model_name = "Aunsiels/resnet-pneumonia-detection"
processor = AutoImageProcessor.from_pretrained(model_name)
hf_model = ________  # TODO: Charger le modèle pour la classification d'images

wrapped_model = ModelWrapper(hf_model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wrapped_model.to(device)
wrapped_model.eval()

inputs = processor(images=image, return_tensors="pt")
input_tensor = inputs["pixel_values"].to(device)
input_tensor.requires_grad = True

# Pour éviter le cold start, on fait un faux appel
logits = wrapped_model(input_tensor)

# 3. Inférence et chronométrage
start_infer = ________  # TODO: Enregistrer le temps de départ
logits = wrapped_model(input_tensor)
predicted_class_idx = logits.argmax(-1).item()
end_infer = ________    # TODO: Enregistrer le temps de fin

print(f"Temps d'inférence : {end_infer - start_infer:.4f} secondes")
print(f"Classe prédite : {hf_model.config.id2label[predicted_class_idx]}")

# 4. Explicabilité : Grad-CAM
target_layer = wrapped_model.model.resnet.encoder.stages[-1].layers[-1]

start_xai = time.time()
# TODO: Instancier LayerGradCam avec le modèle (wrapped) et la couche cible
layer_gradcam = ________ 

# TODO: Calculer les attributions pour la classe prédite
attributions_gradcam = layer_gradcam.________(input_tensor, target=predicted_class_idx)
end_xai = time.time()

print(f"Temps d'explicabilité (Grad-CAM) : {end_xai - start_xai:.4f} secondes")

# 5. Visualisation avec Captum Viz
# Formatage de l'image originale en tableau numpy (H, W, C)
upsampled_attr = LayerAttribution.interpolate(attributions_gradcam, input_tensor.shape[2:])
original_img_np = np.array(image.resize(input_tensor.shape[2:][::-1]))

# Formatage des attributions en tableau numpy (H, W, C)
attr_gradcam_np = upsampled_attr.squeeze().cpu().detach().numpy()
# Grad-CAM génère une carte 2D, l'API viz attend une dimension canal (H, W, 1)
attr_gradcam_np = np.expand_dims(attr_gradcam_np, axis=2)

fig, axis = viz.visualize_image_attr(
    attr_gradcam_np,
    original_img_np,
    method="blended_heat_map",
    sign="positive", # Grad-CAM ne conserve que les influences positives (équivalent au ReLU)
    show_colorbar=True,
    title=f"Grad-CAM - Pred: {hf_model.config.id2label[predicted_class_idx]}"
)

output_filename = f"gradcam_{image_path.split('.')[0]}.png"
fig.savefig(output_filename, bbox_inches='tight')
print(f"Visualisation sauvegardée dans {output_filename}")
```

Exécutez votre script sur les trois images téléchargées (normal\_1.jpeg, normal\_2.jpeg, pneumo\_1.jpeg, pneumo\_2.jpeg). Dans votre fichier **rapport.md**, incluez :

*   Les trois images Grad-CAM générées.
*   **Analyse des Faux Positifs :** Le modèle devrait prédire "PNEUMONIA" même pour l'image saine. En observant la zone mise en évidence par Grad-CAM sur cette erreur, le modèle regarde-t-il une anomalie pulmonaire, ou se base-t-il sur un artefact de la radiographie (effet Clever Hans) ?
*   **Granularité :** Que remarquez-vous concernant la résolution de l'explication (les zones colorées ressemblent à de gros blocs flous) ? Sachant que l'on extrait l'information de la dernière couche d'un ResNet, expliquez d'où vient cette perte de résolution spatiale.

> **Résultats d'exécution (device: cpu) :**
>
> ```
> normal_1.jpeg → Classe prédite: PNEUMONIA | inférence: 0.0785s | Grad-CAM: 0.0835s
> normal_2.jpeg → Classe prédite: PNEUMONIA | inférence: 0.0731s | Grad-CAM: 0.0681s
> pneumo_1.jpeg → Classe prédite: PNEUMONIA | inférence: 0.0734s | Grad-CAM: 0.0657s
> pneumo_2.jpeg → Classe prédite: PNEUMONIA | inférence: 0.0695s | Grad-CAM: 0.0551s
> ```
>
> **Images Grad-CAM générées :**
>
> | normal_1 | normal_2 | pneumo_1 | pneumo_2 |
> |----------|----------|----------|----------|
> | ![](img/gradcam_normal_1.png) | ![](img/gradcam_normal_2.png) | ![](img/gradcam_pneumo_1.png) | ![](img/gradcam_pneumo_2.png) |
>
> **Analyse des Faux Positifs :** Les deux images saines (`normal_1` et `normal_2`) sont prédites PNEUMONIA. La carte Grad-CAM sur `normal_1` montre une activation concentrée dans la région centrale du thorax (médiastin et bases pulmonaires) plutôt que sur une anomalie parenchymateuse clairement localisée. Cette activation diffuse et symétrique ressemble davantage à une réponse à des artefacts d'exposition (densité globale des tissus, bords de la cage thoracique, voire la présence d'équipements médicaux ou de marqueurs de positionnement visibles dans certaines radiographies de cette validation set) qu'à une véritable lésion pulmonaire. C'est un signal d'effet Clever Hans : le modèle a peut-être appris à associer certains artefacts systématiques du dataset d'entraînement (angles de prise de vue, matériel radiologique, textures globales) à la classe PNEUMONIA, plutôt que d'apprendre la sémiologie pulmonaire réelle.
>
> **Granularité :** Les zones d'activation de Grad-CAM apparaissent comme de larges blobs flous, à une résolution bien inférieure à l'image originale. Cela vient directement de l'architecture ResNet : après plusieurs blocs de convolution avec strides et max-pooling, la dernière feature map a une résolution spatiale très réduite (typiquement 7×7 pour une entrée 224×224 dans ResNet50, soit un facteur de réduction de 32×). Grad-CAM extrait les gradients et les activations à cette résolution de sortie, puis les suréchantillonne (bilinear interpolation) vers la taille originale. L'information spatiale fine a été irrémédiablement perdue lors des opérations de sous-échantillonnage des couches profondes.

### Integrated Gradients et SmoothGrad

Les méthodes de type CAM ont une faible résolution spatiale (elles s'arrêtent à la taille de la dernière _feature map_). Pour obtenir une explication précise au pixel près, nous allons utiliser **Integrated Gradients (IG)**. Cependant, IG produit souvent une carte très bruitée. Nous lui appliquerons donc un lissage stochastique appelé **SmoothGrad**.

Créez le script **02\_ig.py** et complétez-le. Attention, pour éviter une erreur CUDA Out of Memory due au nombre massif d'images générées en interne par ces algorithmes, vous devrez utiliser l'argument internal\_batch\_size. Il faudra également contrôler l'amplitude du bruit avec stdevs.

```python
import time
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from captum.attr import IntegratedGradients, NoiseTunnel

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
    def forward(self, x):
        return self.model(x).logits

# 1. Chargement de l'image et du modèle
image_path = sys.argv[1] if len(sys.argv) > 1 else "normal_1.jpeg"
print(f"Analyse fine au pixel sur : {image_path}")
image = Image.open(image_path).convert("RGB")

model_name = "Aunsiels/resnet-pneumonia-detection"
processor = AutoImageProcessor.from_pretrained(model_name)
hf_model = AutoModelForImageClassification.from_pretrained(model_name)
wrapped_model = ModelWrapper(hf_model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wrapped_model.to(device)
wrapped_model.eval()

inputs = processor(images=image, return_tensors="pt")
input_tensor = inputs["pixel_values"].to(device)
input_tensor.requires_grad = True

# Warm-up (Cold Start fix)
_ = wrapped_model(input_tensor)

# Inférence propre
start_infer = time.time()
logits = wrapped_model(input_tensor)
predicted_class_idx = logits.argmax(-1).item()
end_infer = time.time()

# 2. Integrated Gradients
# TODO: Instancier IntegratedGradients sur le modèle (wrapped)
ig = ________ 

# TODO: IG nécessite une image de référence neutre (baseline).
# Créez un tenseur rempli de zéros ayant exactement la même forme que l'input_tensor
baseline = torch.zeros_like(________)

start_ig = time.time()
# TODO: Lancer l'attribution IG avec 50 étapes (n_steps) et limiter la mémoire (internal_batch_size=2)
attributions_ig = ig.________(
    input_tensor, 
    baselines=baseline, 
    target=predicted_class_idx, 
    n_steps=50, 
    internal_batch_size=________
)
end_ig = time.time()

# 3. SmoothGrad (Via NoiseTunnel)
# TODO: Envelopper votre instance "ig" dans un NoiseTunnel
noise_tunnel = ________ 

start_sg = time.time()
# Génération de 100 échantillons bruités (nt_samples) autour de l'image, à réduire pour aller plus vite
# TODO: Ajouter l'argument internal_batch_size=2 pour protéger la VRAM
# TODO: Ajouter l'argument stdevs=0.1 pour adapter le bruit à la normalisation de l'image
attributions_sg = noise_tunnel.attribute(
    input_tensor, 
    nt_samples=100, 
    nt_type='smoothgrad', 
    target=predicted_class_idx,
    stdevs=________,
    internal_batch_size=________
)
end_sg = time.time()

print(f"Temps IG pur : {end_ig - start_ig:.4f}s")
print(f"Temps SmoothGrad (IG x 100) : {end_sg - start_sg:.4f}s")

# 4. Visualisation (Valeur absolue et filtrage du bruit)
# On prend la valeur absolue pour mesurer l'importance globale du pixel (qu'elle soit positive ou négative)
attr_ig_vis = np.sum(np.abs(attributions_ig.squeeze().cpu().detach().numpy()), axis=0)
attr_sg_vis = np.sum(np.abs(attributions_sg.squeeze().cpu().detach().numpy()), axis=0)

# Seuillage stochastique : on met à zéro tous les pixels dont l'importance est inférieure au 70e centile
threshold_ig = np.percentile(attr_ig_vis, 70)
attr_ig_vis[attr_ig_vis < threshold_ig] = 0

threshold_sg = np.percentile(attr_sg_vis, 70)
attr_sg_vis[attr_sg_vis < threshold_sg] = 0

# Normalisation pour l'affichage
vmax_ig = np.max(attr_ig_vis)
vmax_sg = np.max(attr_sg_vis)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Affichage de l'image de base sur le premier axe
axes[0].imshow(image.resize(input_tensor.shape[2:][::-1]))
axes[0].set_title("Image Originale")
axes[0].axis('off')

# Superposition IG sur l'image originale
axes[1].imshow(image.resize(input_tensor.shape[2:][::-1]), alpha=0.6)
# Utilisation de la colormap 'hot' (noir vers rouge/jaune/blanc)
axes[1].imshow(attr_ig_vis, cmap='hot', alpha=0.6, vmin=0, vmax=vmax_ig)
axes[1].set_title("Integrated Gradients (Seuillé)")
axes[1].axis('off')

# Superposition SmoothGrad sur l'image originale
axes[2].imshow(image.resize(input_tensor.shape[2:][::-1]), alpha=0.6)
axes[2].imshow(attr_sg_vis, cmap='hot', alpha=0.6, vmin=0, vmax=vmax_sg)
axes[2].set_title("SmoothGrad (Seuillé)")
axes[2].axis('off')

output_filename = f"ig_smooth_{image_path.split('.')[0]}.png"
plt.savefig(output_filename, bbox_inches='tight')
print(f"Visualisation sauvegardée dans {output_filename}")
```

Soumettez ce script sur votre cluster. Dans **rapport.md** :

*   Ajoutez l'image comparative générée.
*   Relevez les temps d'exécution. Au vu du temps de calcul de SmoothGrad par rapport à une inférence classique, pensez-vous qu'il soit technologiquement possible de générer cette explication de manière synchrone (en temps réel) pour un médecin lors du premier clic d'analyse ? Proposez une architecture logicielle en une phrase pour résoudre ce problème (ex: utilisation de files d'attente).
*   En observant les couleurs de la visualisation finale (bleu et rouge), expliquez brièvement l'avantage mathématique d'avoir une carte qui descend "en dessous de zéro" par rapport au filtre ReLU classique utilisé dans Grad-CAM.

> **Résultats d'exécution sur `pneumo_1.jpeg` (device: cpu) :**
>
> ```
> Temps d'inférence pur  : 0.0938s
> Temps IG pur           : 5.8282s   (× 62 vs inférence)
> Temps SmoothGrad (×100): 541.72s   (× 5778 vs inférence)
> ```
>
> **Image comparative IG vs SmoothGrad :**
>
> ![IG et SmoothGrad sur pneumo_1](img/ig_smooth_pneumo_1.png)
>
> **Faisabilité temps réel :** Non — 541 secondes pour générer un SmoothGrad est totalement incompatible avec une réponse synchrone. Une architecture adaptée consisterait à **découpler l'affichage et l'explication** : au premier clic, renvoyer la prédiction immédiatement (0.09s), puis soumettre la génération d'explication à une **file de tâches asynchrone** (ex: Celery + Redis) qui notifie l'interface une fois le calcul terminé en arrière-plan.
>
> **Avantage mathématique des valeurs négatives :** Contrairement à Grad-CAM qui applique un ReLU final (ne conservant que les gradients positifs), Integrated Gradients calcule des attributions **signées** : une valeur négative signifie que le pixel a activement fait baisser le score de la classe prédite (il plaidait contre cette prédiction). Cette information directionnelle permet d'identifier non seulement "ce qui pousse vers la classe" (zones chaudes) mais aussi "ce qui résiste à cette prédiction" (zones froides), donnant ainsi une explication **plus complète et plus fidèle** à la décision réelle du modèle. Le ReLU de Grad-CAM supprime ce signal, au risque de masquer des zones ambiguës ou contradictoires cliniquement importantes.

**Interprétation IG absolue :** Les couleurs chaudes (rouge, jaune, blanc) localisent les régions de l'image qui ont le plus fortement activé les gradients du modèle. L'absence de couleur indique les régions ignorées. Comparé à Grad-CAM, vous observez ici les contours exacts (bords d'organes, textures fines) qui ont été déterminants, plutôt qu'une zone sémantique floue.

### Modélisation Intrinsèquement Interprétable (Glass-box) sur Données Tabulaires

Pour cette seconde partie du TP, nous quittons les images pour analyser des données cliniques tabulaires. Avant de manipuler des modèles complexes dits "boîtes noires", l'approche d'ingénierie standard consiste à commencer par un modèle intrinsèquement interprétable (Glass-box).

Installez les bibliothèques d'analyse de données nécessaires pour la suite de ce TP :

```bash
pip install scikit-learn pandas
```

Créez le script **03\_glassbox.py**. Nous allons utiliser le jeu de données _Breast Cancer Wisconsin_ (intégré dans scikit-learn) pour prédire si une tumeur est bénigne ou maligne à partir de 30 caractéristiques cellulaires numérisées.

Complétez le code ci-dessous pour entraîner une Régression Logistique. Afin que les poids d'une régression logistique soient interprétables et comparables entre eux, il est impératif de normaliser les données en amont.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Chargement des données
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target # 0: Maligne, 1: Bénigne

# 2. Séparation des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Normalisation (Crucial pour l'interprétabilité des coefficients)
scaler = StandardScaler()
# TODO: Adapter le scaler sur les données d'entraînement et les transformer
X_train_scaled = scaler.________(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Entraînement du modèle "Glass-box"
# On utilise une pénalité L2 standard
model = LogisticRegression(max_iter=1000, random_state=42)
# TODO: Entraîner le modèle
model.________(X_train_scaled, y_train)

# Évaluation rapide
y_pred = model.predict(X_test_scaled)
print(f"Accuracy de la Régression Logistique : {accuracy_score(y_test, y_pred):.4f}")

# 5. Extraction de l'explication (Intrinsèque)
# TODO: Extraire les coefficients appris par le modèle (un tableau numpy)
coefficients = model.________[0]

# Création d'un DataFrame pour faciliter la manipulation
feature_importance = pd.DataFrame({
    'Feature': data.feature_names,
    'Coefficient': coefficients
})

# Tri par importance absolue
feature_importance['Abs_Coefficient'] = feature_importance['Coefficient'].abs()
feature_importance = feature_importance.sort_values(by='Abs_Coefficient', ascending=True)

# 6. Visualisation
plt.figure(figsize=(10, 8))
# On colorie en rouge les coefficients négatifs (poussent vers la classe 0: Maligne)
# et en bleu les positifs (poussent vers la classe 1: Bénigne)
colors = ['red' if c < 0 else 'blue' for c in feature_importance['Coefficient']]

plt.barh(feature_importance['Feature'][-15:], feature_importance['Coefficient'][-15:], color=colors[-15:])
plt.xlabel('Valeur du Coefficient (\u03b2)')
plt.title('Top 15 - Importance des variables (Régression Logistique)')
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
plt.tight_layout()

output_filename = "glassbox_coefficients.png"
plt.savefig(output_filename)
print(f"Graphique sauvegardé dans {output_filename}")
```

Exécutez ce script sur votre machine ou votre cluster. Dans votre **rapport.md** :

*   Ajoutez l'image glassbox\_coefficients.png générée.
*   En observant le graphique, identifiez la caractéristique (_feature_) qui a le plus d'impact pour pousser la prédiction vers la classe "Maligne" (classe 0, coefficients négatifs).
*   Expliquez en une phrase l'avantage d'avoir un modèle directement interprétable par rapport aux méthodes post-hoc vues précédemment.

> **Résultats d'exécution :**
>
> ```
> Accuracy de la Régression Logistique : 0.9737
>
> Top 5 features (coefficient absolu, négatif = pousse vers Maligne) :
>   worst texture        : -1.3506
>   radius error         : -1.2682
>   worst symmetry       : -1.2082
>   mean concave points  : -1.1198
>   worst concavity      : -0.9431
> ```
>
> **Graphique des coefficients :**
>
> ![Coefficients Régression Logistique](img/glassbox_coefficients.png)
>
> **Feature avec le plus d'impact vers "Maligne" :** `worst texture` est la caractéristique avec le coefficient le plus négatif (β = −1.35). Une valeur élevée de la texture "pire" (variance des niveaux de gris dans la zone la plus sombre de la tumeur) pousse le modèle fortement vers la prédiction Maligne — ce qui est cohérent avec la médecine : les tumeurs malignes ont des textures cellulaires plus irrégulières.
>
> **Avantage du modèle directement interprétable :** Contrairement aux méthodes post-hoc qui approximent l'explication après coup, la régression logistique expose ses coefficients directement comme explication exacte et sans hypothèse supplémentaire.

Notez bien les 2 ou 3 variables les plus importantes. Nous allons vérifier si une méthode "boîte noire" complexe utilise la même logique médicale dans l'exercice suivant.

### Explicabilité Post-Hoc avec SHAP sur un Modèle Complexe

Les modèles linéaires sont faciles à interpréter, mais ils ne capturent pas les interactions complexes non-linéaires entre les caractéristiques. Pour obtenir de meilleures performances, les ingénieurs utilisent souvent des modèles d'ensemble (comme les Random Forests) ou du Deep Learning, créant ainsi des "boîtes noires". Pour auditer ces modèles, nous utilisons des méthodes post-hoc comme **SHAP** (SHapley Additive exPlanations).

Commencez par installer la bibliothèque SHAP dans votre environnement :

```bash
pip install shap
```

Créez le script **04\_shap.py**. Complétez le code ci-dessous. Vous remarquerez que contrairement à la Régression Logistique, un Random Forest n'a pas besoin que les données soient normalisées. Nous analyserons les valeurs brutes.

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap

# 1. Chargement et Séparation des données
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target # 0: Maligne, 1: Bénigne

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Entraînement d'une "Boîte Noire" (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
# TODO: Entraîner le modèle sur les données d'entraînement
model.________(X_train, y_train)

print(f"Accuracy du Random Forest : {model.score(X_test, y_test):.4f}")

# 3. Explicabilité Post-Hoc avec SHAP
# TODO: Instancier le TreeExplainer de SHAP en lui passant le modèle entraîné
explainer = shap.________(model)

# TODO: Calculer les valeurs SHAP (Explanation object) pour l'ensemble du jeu de test.
# Indice : l'objet explainer peut être appelé directement comme une fonction.
shap_values = explainer(________)

# L'API de SHAP prédit par défaut la probabilité de la classe 1 (Bénigne) pour un RandomForest binaire.
# Nous allons extraire les valeurs SHAP spécifiques à cette classe (index 1) pour les visualisations.
shap_values_class1 = shap_values[:, :, 1]

# 4. Explicabilité Locale : Waterfall Plot (Un seul patient)
patient_idx = 0
plt.figure(figsize=(10, 6))
# show=False permet à matplotlib de sauvegarder l'image au lieu d'ouvrir une fenêtre graphique
shap.plots.waterfall(shap_values_class1[patient_idx], show=False)
plt.title(f"Explication Locale SHAP - Patient {patient_idx}")
plt.tight_layout()
output_local = "shap_waterfall.png"
plt.savefig(output_local)
plt.close()
print(f"Waterfall plot sauvegardé dans {output_local}")

# 5. Explicabilité Globale : Summary Plot
plt.figure(figsize=(10, 8))
# Le summary plot analyse toutes les prédictions du jeu de test en même temps
shap.summary_plot(shap_values_class1, X_test, show=False)
plt.title("Importance globale et directionnelle des variables (SHAP)")
plt.tight_layout()
output_global = "shap_summary.png"
plt.savefig(output_global)
plt.close()
print(f"Summary plot sauvegardé dans {output_global}")
```

Exécutez le script et récupérez les deux images générées. Dans votre **rapport.md** :

*   Ajoutez les images shap\_waterfall.png et shap\_summary.png.
*   **Explicabilité Globale :** En regardant le _Summary Plot_, vérifiez si les 2-3 variables les plus importantes identifiées par le Random Forest (SHAP) sont les mêmes que celles trouvées par la Régression Logistique (Exercice 3). Que déduisez-vous sur la robustesse de ces caractéristiques cliniques (biomarqueurs) ?
*   **Explicabilité Locale :** En analysant le _Waterfall Plot_ du patient 0, citez la caractéristique (feature) ayant le plus contribué à tirer la prédiction vers sa valeur finale, ainsi que la valeur numérique exacte de cette caractéristique pour ce patient précis.

> **Résultats d'exécution :**
>
> ```
> Accuracy du Random Forest : 0.9649
>
> Top 5 features par SHAP (mean |SHAP|) :
>   worst area           : 0.0769
>   worst concave points : 0.0694
>   mean concave points  : 0.0491
>   worst radius         : 0.0405
>   worst perimeter      : 0.0396
>
> Patient 0 : prédit BENIGN (classe 1)
>   Feature la plus contributive : worst area  (SHAP=+0.0672, valeur=677.90)
> ```
>
> **Waterfall Plot (Patient 0) :**
>
> ![SHAP Waterfall Patient 0](img/shap_waterfall.png)
>
> **Summary Plot (Global) :**
>
> ![SHAP Summary Plot](img/shap_summary.png)
>
> **Explicabilité Globale :** Les top features SHAP pour le Random Forest (`worst area`, `worst concave points`, `mean concave points`) recoupent partiellement celles de la Régression Logistique (`worst texture`, `radius error`, `worst symmetry`, `mean concave points`). La feature `mean concave points` apparaît dans les deux classements. Cette convergence entre deux familles de modèles très différentes (linéaire vs. ensemble d'arbres) suggère que ces biomarqueurs — notamment les descripteurs de concavité et de texture associés au "pire" noyau cellulaire — sont des signal cliniques robustes et fiables pour discriminer tumeurs bénignes et malignes, indépendamment de l'algorithme utilisé.
>
> **Explicabilité Locale :** Pour le patient 0, la caractéristique ayant le plus contribué à pousser la prédiction vers la classe Bénigne est `worst area` avec une valeur SHAP de +0.0672, pour une valeur brute de 677.90 mm². La valeur positive indique que cette zone "pire" relativement modérée (par rapport aux tumeurs malignes qui ont typiquement une aire plus élevée) a été le facteur le plus déterminant pour le Random Forest dans ce diagnostic.

**Interprétation du Summary Plot SHAP :** Chaque point représente un patient. La couleur représente la valeur brute de la caractéristique (rouge = valeur élevée, bleu = valeur faible). La position sur l'axe horizontal indique l'impact sur la prédiction (à droite de 0 = pousse vers la classe Bénigne, à gauche de 0 = pousse vers la classe Maligne).