# CipherLink

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

CipherLink a été créé dans le cadre d'un projet de la matière "Principle of Digital Communications" donnée en 3ème année aux étudiants de la section Systèmes de Communications à l'EPFL.
L'objectif était d'encoder un message de 40 caractères provenant d'un alphabet de 64 caractères possibles, de le passer à un canal bruité, puis de le décoder sans erreur, le tout en utilisant le moins d'énergie possible.
Le code est basé sur les librairies sionna et reedsolo.

## Installation

1. Assurez-vous d'avoir Python 3.8 installé sur votre système.
2. Clonez ce dépôt sur votre machine locale :
   ```sh
   git clone https://github.com/matthias-wyss/CipherLink.git
   ```
3. Accédez au répertoire du projet :
   ```sh
   cd CipherLink
   ```
4. Créez un environnement virtuel :
   ```sh
   python3.8 -m venv cipherlink
   ```
5. Activez l'environnement virtuel :
   - Sous Windows :
     ```sh
     cipherlink\Scripts\activate
     ```
   - Sous macOS/Linux :
     ```sh
     source cipherlink/bin/activate
     ```
6. Installez les dépendances à partir du fichier `requirements.txt` :
   ```sh
   pip install -r requirements.txt
   ```

## Utilisation

Les codes de correction d'erreurs supportés sont :
- LDPC
- code de convolution
- code turbo
- code polar avec annulation succesive
- code polar avec liste d'annulation succesive

## Licence

Ce projet est sous licence [MIT](LICENSE).

---

Crée avec ❤️ par Matthias Wyss
