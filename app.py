import argparse
from flask import Flask, jsonify, request
from flask_session import Session
import time
import os
from mlx_lm import generate, load
from mlx_lm.models.cache import make_prompt_cache
from huggingface_hub.utils import disable_progress_bars

# Désactiver les barres de progression pour un déploiement propre
disable_progress_bars()

app = Flask(__name__)

# Configuration des sessions
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Variables globales
tokenizer = None
model = None
prompt_cache = None
MODEL_NAME = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DEFAULT_TEMPERATURE = 0.0

# Chargement du modèle et tokenizer avec température
def load_model_and_tokenizer(model_name, temperature=DEFAULT_TEMPERATURE):
    print(f"🔄 Chargement du modèle MLX: {model_name} (température: {temperature})")
    model_config = {'temperature': temperature}
    model, tokenizer = load(model_name, model_config=model_config)
    prompt_cache = make_prompt_cache(model)
    return tokenizer, model, prompt_cache

# Route pour lister les modèles disponibles
@app.route("/v1/models", methods=["GET"])
def models():
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": 1677610602,
                "owned_by": "local-mlx",
                "permission": [],
                "root": MODEL_NAME,
                "parent": None
            }
        ]
    }), 200

# Route pour les chat completions
@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    global tokenizer, model, prompt_cache
    
    # Traitement de la requête
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": {"message": "JSON invalide", "type": "invalid_request_error", "code": "400"}}), 400
        
        # Extraction des messages
        if isinstance(data, list):
            messages = data
        elif isinstance(data, dict) and "messages" in data:
            messages = data["messages"]
        else:
            return jsonify({"error": {"message": "Le champ 'messages' est requis.", "type": "invalid_request_error", "code": "400"}}), 400
        
        # Récupération des paramètres optionnels
        max_tokens = data.get("max_tokens", 1024)
        temperature = data.get("temperature", DEFAULT_TEMPERATURE)
        
        # Si la température demandée est différente, recharger le modèle
        if model is None or temperature != DEFAULT_TEMPERATURE:
            tokenizer, model, prompt_cache = load_model_and_tokenizer(MODEL_NAME, temperature)
        
        # Formatage des messages pour le tokenizer
        chat_messages = []
        system_content = "Vous êtes un assistant intelligent et serviable."
        
        # Extraire le message système s'il existe
        for msg in messages:
            if msg.get('role') == 'system':
                system_content = msg.get('content', system_content)
                break
        
        # Préparer les messages pour le template
        for msg in messages:
            role = msg.get('role')
            if role == 'system':
                continue  # Le système est traité séparément
            
            content = msg.get('content', '')
            if isinstance(content, list):
                # Traitement des contenus structurés
                text_parts = [part.get('text', '') for part in content if part.get('type') == 'text']
                content = "\n".join(text_parts).strip()
            
            chat_messages.append({"role": role, "content": content})
        
        # Génération du prompt avec le template du modèle
        try:
            prompt_parts = tokenizer.apply_chat_template(chat_messages, add_generation_prompt=True)
            prompt_str = tokenizer.decode(prompt_parts)
            full_prompt = system_content + "\n" + prompt_str
        except Exception as e:
            # Fallback en cas d'erreur avec apply_chat_template
            full_prompt = format_messages_fallback(messages)
        
        # Génération avec MLX
        start_time = time.time()
        response_text = generate(
            model, 
            tokenizer, 
            prompt=full_prompt, 
            max_tokens=max_tokens,
            prompt_cache=prompt_cache,
            verbose=False
        )
        end_time = time.time()
        
        # Calcul des statistiques
        duration = end_time - start_time
        response_tokens = len(tokenizer.encode(response_text))
        
        # Construction de la réponse au format OpenAI
        return jsonify({
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_NAME,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text.strip()
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(tokenizer.encode(full_prompt)),
                "completion_tokens": response_tokens,
                "total_tokens": len(tokenizer.encode(full_prompt)) + response_tokens
            }
        }), 200
        
    except Exception as e:
        return jsonify({"error": {"message": f"Erreur: {str(e)}", "type": "server_error", "code": "500"}}), 500

# Fonction de secours pour formater les messages si le template échoue
def format_messages_fallback(messages):
    system_content = "Vous êtes un assistant intelligent et serviable."
    for msg in messages:
        if msg.get('role') == 'system':
            system_content = msg.get('content', system_content)
            break
    
    prompt = f"<|system|>\n{system_content}\n"
    
    for msg in messages:
        role = msg.get('role')
        if role == 'system':
            continue
        
        content = msg.get('content', '')
        if isinstance(content, list):
            text_parts = [part.get('text', '') for part in content if part.get('type') == 'text']
            content = "\n".join(text_parts).strip()
        
        if role == 'user':
            prompt += f"<|user|>\n{content}\n"
        elif role == 'assistant':
            prompt += f"<|assistant|>\n{content}\n"
    
    prompt += "<|assistant|>\n"
    return prompt

if __name__ == "__main__":
    # Chargement initial du modèle
    parser = argparse.ArgumentParser(description="Serveur API OpenAI compatible avec MLX")
    parser.add_argument("--model", type=str, default="mlx-community/Llama-3.2-3B-Instruct-4bit", 
                        help="Nom du modèle MLX à utiliser")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, 
                        help=f"Température par défaut (défaut: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Adresse d'hôte")
    parser.add_argument("--port", type=int, default=1982, help="Port d'écoute")
    
    args = parser.parse_args()
    MODEL_NAME = args.model
    DEFAULT_TEMPERATURE = args.temperature
    
    print(f"🚀 Démarrage du serveur avec le modèle: {MODEL_NAME} (température: {DEFAULT_TEMPERATURE})")
    tokenizer, model, prompt_cache = load_model_and_tokenizer(MODEL_NAME, DEFAULT_TEMPERATURE)
    
    app.run(host=args.host, port=args.port, debug=False)