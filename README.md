# MLX OpenAI Compatible API Server

Un serveur API compatible avec le format OpenAI qui exécute des modèles de langage localement sur les Mac avec Apple Silicon en utilisant MLX.

## 🌟 Fonctionnalités

- 🚀 **Compatible OpenAI** : Implémente l'API `/v1/chat/completions` standard d'OpenAI
- 🍎 **Optimisé pour Apple Silicon** : Utilise MLX pour une inférence rapide sur les puces Apple
- 🔥 **Local** : Exécution 100% locale pour la confidentialité et le fonctionnement hors ligne
- 🧠 **Flexible** : Supporte différents modèles compatibles MLX (Llama, Mistral, etc.)
- 🌡️ **Paramétrable** : Contrôle de la température, du nombre de tokens, etc.

## 📋 Prérequis

- Mac avec puce Apple Silicon (M1, M2, M3, etc.)
- Python 3.10+
- pip3

## ⚙️ Installation

1. Clonez le dépôt :
```bash
git clone https://github.com/VOTRE_NOM_UTILISATEUR/mlx-openai-server.git
cd mlx-openai-server
```

2. Installez les dépendances :
```bash
pip3 install flask flask_session mlx-lm huggingface_hub --break-system-packages
```

## 🚀 Démarrage rapide

Lancez le serveur avec le modèle par défaut :

```bash
python3 server.py
```

Ou spécifiez un modèle personnalisé et d'autres options :

```bash
python3 server.py --model mlx-community/Llama-3.2-8B-Instruct-4bit --temperature 0.7 --port 1982
```

## 🔧 Options de configuration

Le serveur accepte les options suivantes :

| Option | Description | Par défaut |
|--------|-------------|------------|
| `--model` | Nom du modèle MLX à utiliser | `mlx-community/Llama-3.2-3B-Instruct-4bit` |
| `--temperature` | Température pour la génération (0.0 = déterministe) | `0.0` |
| `--host` | Adresse IP du serveur | `127.0.0.1` |
| `--port` | Port d'écoute | `1982` |

## 📝 Exemples d'utilisation

### cURL

```bash
curl http://localhost:1982/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Explique-moi la relativité restreinte en termes simples"}],
    "temperature": 0.7,
    "max_tokens": 500
  }'
```

### Python

```python
import requests

response = requests.post(
    "http://localhost:1982/v1/chat/completions",
    json={
        "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "messages": [
            {"role": "system", "content": "Tu es un assistant utile et précis."},
            {"role": "user", "content": "Quels sont les meilleurs langages de programmation pour l'IA en 2025?"}
        ],
        "temperature": 0.3
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

### Intégration avec des clients compatibles OpenAI

Le serveur peut être utilisé avec n'importe quelle bibliothèque cliente compatible OpenAI en remplaçant simplement l'URL de base.

```python
from openai import OpenAI

client = OpenAI(
    api_key="FAUX-CLÉ-API",  # N'importe quelle chaîne fonctionne
    base_url="http://localhost:1982/v1/"
)

response = client.chat.completions.create(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    messages=[
        {"role": "user", "content": "Écris un court poème sur la programmation"}
    ]
)

print(response.choices[0].message.content)
```

## 📊 Modèles testés

Le serveur a été testé avec les modèles suivants :

- `mlx-community/Llama-3.2-3B-Instruct-4bit`
- `mlx-community/Mistral-Small-24B-Instruct-2501-bf16`
- `mlx-community/Mistral-Nemo-Instruct-2407-bf16`

## ⚠️ Limitations

- Le support des fonctions/outils n'est pas encore implémenté
- Certains modèles MLX peuvent avoir des formats de messages différents
- Les performances dépendent de la puce Apple Silicon et de la taille du modèle

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou une pull request.

## 📄 Licence

Ce projet est sous licence MIT - voir le fichier LICENSE pour plus de détails.
