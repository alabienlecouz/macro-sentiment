# Déploiement sur une VM Linux (Ubuntu 22.04 / 24.04)

Deux services tournent en permanence :

| Service | Rôle | Fréquence |
|---|---|---|
| `fxbot-tableau` | interface Streamlit (port 8501, en local sur la VM) | 24h/24, redémarre seule |
| `fxbot-maj.timer` | lance `scripts/mettre_a_jour.py` (fair values, sentiment, signaux) | 7h et 19h, heure de Paris |

L'interface rafraîchit les cours toutes les 5 minutes ; les modèles sont recalculés par le timer.

**Sur la VM, pas de torch** : le sentiment tourne avec ONNX Runtime (`config.local.yaml`),
avec le modèle DeBERTa exporté sur le Mac par `scripts/exporter_onnx.py`. FinBERT (comparaison)
reste sur le Mac. Il faut environ 2 Go d'espace disque et 1 Go de RAM + 1 Go de swap.

## 1. Copier le projet sur la VM

Depuis le Mac, dans le dossier du projet (sans l'environnement virtuel, le cache ni les secrets) :

```bash
rsync -az --delete --no-owner --no-group -e "ssh -i ~/.ssh/ma-cle.pem" --rsync-path="sudo rsync" \
  --exclude-from=deploiement/exclusions_rsync.txt ./ UTILISATEUR@IP_VM:/opt/fxbot/
```

Envoi direct dans `/opt/fxbot` (le `/tmp` d'une petite VM est souvent trop petit pour le modèle
de 750 Mo). Les fichiers propres à la VM (`.venv`, `sorties`, `.env`, `config.local.yaml`…) sont
listés dans `deploiement/exclusions_rsync.txt` : ils ne sont ni écrasés ni supprimés.

## 2. Installer sur la VM

```bash
# 1 Go de swap (filet de sécurité pendant la mise à jour)
sudo fallocate -l 1G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
# uv (gestionnaire Python rapide) installé pour tout le système
curl -LsSf https://astral.sh/uv/install.sh | sudo env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
sudo useradd --system --create-home --home-dir /opt/fxbot --shell /usr/sbin/nologin fxbot
sudo cp /opt/fxbot/deploiement/config.local.yaml /opt/fxbot/
sudo chown -R fxbot:fxbot /opt/fxbot
cd /opt/fxbot
# Python 3.13 géré par uv (la VM a 3.14, encore récent pour certaines bibliothèques)
sudo -u fxbot env UV_PYTHON_INSTALL_DIR=/opt/fxbot/.python uv venv --python 3.13 .venv
sudo -u fxbot env UV_CACHE_DIR=/opt/fxbot/.cache/uv uv pip install --python .venv/bin/python -r requirements.txt
```

## 3. Clé FRED (sans l'afficher ni l'écrire dans l'historique)

```bash
read -s -p "Clé FRED : " cle && echo "FRED_API_KEY=$cle" | sudo -u fxbot tee /opt/fxbot/.env > /dev/null && unset cle && sudo chmod 600 /opt/fxbot/.env
```

## 4. Premier calcul puis activation des services

```bash
sudo -u fxbot /opt/fxbot/.venv/bin/python scripts/mettre_a_jour.py
sudo cp deploiement/fxbot-*.service deploiement/fxbot-*.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now fxbot-tableau.service fxbot-maj.timer
systemctl list-timers fxbot-maj.timer
```

Le modèle de sentiment est déjà dans `modeles/` (copié par rsync) : rien à télécharger.

## 5. Accéder à l'interface

L'interface n'écoute que sur la VM (`127.0.0.1`, voir `.streamlit/config.toml`) : elle n'est
pas exposée sur Internet, car Streamlit n'a pas de mot de passe. On y accède par un tunnel SSH,
depuis le Mac :

```bash
ssh -N -L 8501:localhost:8501 UTILISATEUR@IP_VM
```

puis ouvrir http://localhost:8501 dans le navigateur.

Pour un accès sans tunnel (téléphone, etc.), il faut un reverse proxy avec HTTPS et
authentification (par ex. Caddy avec `basic_auth`) : ne jamais simplement passer
`address = "0.0.0.0"`.

## Entretien

| Action | Commande |
|---|---|
| Journal de l'interface | `journalctl -u fxbot-tableau -f` |
| Journal des mises à jour | `journalctl -u fxbot-maj -n 100` |
| Forcer une mise à jour | `sudo systemctl start fxbot-maj.service` |
| Après une modification du code | rsync (étape 1), puis `sudo chown -R fxbot:fxbot /opt/fxbot && sudo systemctl restart fxbot-tableau` |

**Communiqués de banques centrales** : après chaque réunion, ajouter la nouvelle adresse dans
`data/communiques/sources.yaml`. Le site de la RBA bloque les scripts : coller son texte dans
`data/communiques/RBA/<date>.txt`.
