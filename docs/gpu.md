# Accélération GPU : NVIDIA, AMD ou CPU

Le projet fonctionne **sur CPU par défaut** (aucune configuration). Le GPU n'accélère vraiment que l'étape CNN (notebook 04) ; sur ~1 500 images, l'extraction d'embeddings reste raisonnable sur CPU.

| Votre machine | GPU dans Docker ? | Comment |
|---|---|---|
| Windows + **NVIDIA** | Oui (Docker Desktop + WSL2 + pilote NVIDIA récent) | override `docker-compose.nvidia.yml` |
| Linux + **NVIDIA** | Oui (NVIDIA Container Toolkit) | override `docker-compose.nvidia.yml` |
| Linux natif + **AMD** | Oui (ROCm : `/dev/kfd`, `/dev/dri`) | override `docker-compose.amd.yml` |
| Windows + **AMD** | **Pas en configuration standard** (voir plus bas) | **CPU** (par défaut) |
| macOS | Non (pas de GPU dans les conteneurs) | CPU |

## Vérifier ce que voit PyTorch

```powershell
docker compose run --rm gpu-check                                                    # CPU (défaut)
docker compose -f docker-compose.yml -f docker-compose.nvidia.yml run --rm gpu-check # NVIDIA
docker compose -f docker-compose.yml -f docker-compose.amd.yml run --rm gpu-check    # AMD (Linux)
```
La sortie indique le build (CPU / NVIDIA CUDA / AMD ROCm), si un GPU est utilisable et son nom. Sous ROCm, l'API PyTorch s'appelle aussi `torch.cuda` : le code du projet (`device: auto`) fonctionne donc sans modification.

## Lancer le pipeline avec un GPU

```powershell
# NVIDIA
docker compose -f docker-compose.yml -f docker-compose.nvidia.yml run --rm pipeline
# AMD, Linux natif
export RENDER_GID=$(getent group render | cut -d: -f3)
docker compose -f docker-compose.yml -f docker-compose.amd.yml run --rm pipeline
```
Les services concernés par les overrides sont `gpu-check`, `features`, `semi` et `pipeline` ; les autres étapes n'utilisent pas le GPU. Variantes de wheels PyTorch : `TORCH_VARIANT` (NVIDIA : `cu126` par défaut, ou `cu128`, `cu130`... ; AMD : `rocm7.2` par défaut). Choisir une variante compatible avec votre pilote ; les index disponibles évoluent (https://pytorch.org/get-started/locally/).

> Les images GPU ne sont pas les mêmes que l'image CPU (`brainscanai:nvidia`, `brainscanai:amd`) : le premier build télécharge de gros paquets (plusieurs Go).

## AMD sous Windows : ce qui est réaliste

- **Docker Desktop ne donne pas accès aux GPU AMD** dans les conteneurs (le passage de GPU documenté est celui de NVIDIA). Les fichiers `/dev/kfd` et `/dev/dri` requis par ROCm n'existent pas dans ce contexte. Pour cette raison, `docker-compose.amd.yml` est **réservé à Linux natif**.
- **Recommandation : exécuter sur CPU** avec la commande standard. C'est correct pour ce jeu de données ; pour aller plus vite, voir « Accélérer sur CPU ».
- **Piste expérimentale, non fournie ni testée ici** : AMD documente ROCm sous **WSL2** pour certaines cartes Radeon récentes (par exemple RX 9070, RX 9060, RX 7800 XT, RX 7700 XT, certaines PRO W7000), avec le pilote Adrenalin « for WSL2 » et PyTorch + ROCm installés **dans une distribution Ubuntu WSL2**, hors Docker Desktop. Faire tourner Docker *dans* WSL2 avec la passerelle `/dev/dxg` a été rapporté par la communauté mais reste complexe. La liste des cartes compatibles change : consulter la matrice de compatibilité WSL de la documentation ROCm avant d'investir du temps. Si votre carte y figure, le plus simple est d'utiliser WSL2 comme un hôte Linux avec ROCm natif, puis l'override `docker-compose.amd.yml`.

## Accélérer sur CPU

Dans `configs/config.yaml` :
- `semi_supervised.n_repeats: 2` (au lieu de 5) pour un essai ; 5 pour le rendu final ;
- réduire les `epochs` des trois étapes du CNN ;
- `data.image_size: 160` (au lieu de 224) accélère nettement, au prix de détails fins ;
- `docker compose run --rm semi python scripts/run_semi_supervised.py --skip-cnn` ne lance que les méthodes sur embeddings (rapides).
