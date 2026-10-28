# Chess_Scan_Video

Extrait le PGN d'une partie d'échecs (interface chess.com) visible dans une vidéo.

## Installation
```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Pour les vidéos YouTube, installez également [FFmpeg](https://ffmpeg.org/download.html)
et assurez-vous que la commande `ffmpeg` est disponible dans le `PATH`. Il est utilisé
par `yt-dlp` lorsque YouTube fournit séparément les flux vidéo et audio.

## Utilisation

### Interface web
```
.venv\Scripts\python app.py
```
Ouvrir http://127.0.0.1:5000, glisser la vidéo sur la zone prévue, cliquer sur « Extraire le PGN »,
puis « Copier » pour récupérer le PGN. La vidéo envoyée est supprimée après analyse.

Il est également possible de coller une URL YouTube publique dans le champ prévu à cet effet,
puis de cliquer sur « Extraire le PGN ». La vidéo est téléchargée dans un dossier temporaire,
analysée de la même façon qu'un fichier local, puis supprimée automatiquement.

Les playlists ne sont pas traitées : une seule vidéo est analysée. Les vidéos privées,
bloquées par une restriction d'âge, nécessitant une authentification ou indisponibles
ne peuvent pas être téléchargées. L'utilisation des vidéos doit respecter les conditions
d'utilisation de YouTube et les droits applicables.

### Ligne de commande
```
.venv\Scripts\python chess_scan.py [video.mp4] [-o sortie.pgn] [--white NOM] [--black NOM]
```
Sans argument : premier `.mp4` de `video\`, sortie dans `output\`.

Options utiles : `--box X0 Y0 X1 Y1` (impose la position de l'échiquier ; sinon détection automatique),
`--step` (1 image analysée sur N), `--tol`, `--stable`, `--plies 2`, `--lost`.

## Principe
1. Détection automatique de l'échiquier (position et taille quelconques, 2D à cases unies).
2. Recherche de la position initiale ; l'orientation (noirs en haut ou en bas) en est déduite.
3. Les 12 modèles de pièces sont appris sur cette position, dans les couleurs de la vidéo.
4. À chaque image, on retient le coup légal (python-chess) dont la position résultante
   correspond à l'image, ce qui élimine les erreurs de reconnaissance.
5. Arrêt à la fin de la partie : mat/pat, échiquier remis à zéro (début de l'analyse),
   échiquier absent ou position incohérente pendant `--lost` secondes.

Limites : la position initiale doit être visible, une seule partie est extraite, thèmes
texturés (bois, marbre) non garantis, résultat `*` sauf mat/pat.
