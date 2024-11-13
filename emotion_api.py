from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Carrega o pipeline de análise de emoções
emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True, truncation=True, max_length=512)

# Criação da aplicação FastAPI
app = FastAPI()

# Definição do modelo de entrada para a requisição
class TextRequest(BaseModel):
    texto: str

# Função para dividir o texto em partes menores (máximo 512 tokens)
def dividir_texto(texto, max_tokens=512):
    tokens = texto.split()
    partes = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [' '.join(parte) for parte in partes]

# Mapeamento das emoções de inglês para português
mapa_emocoes = {
    "anger": "raiva",
    "disgust": "desgosto",
    "fear": "medo",
    "joy": "alegria",
    "neutral": "neutro",
    "sadness": "tristeza",
    "surprise": "surpresa"
}

# Rota para a análise de emoções
@app.post("/analisar-emocao/")
async def analisar_emocao(request: TextRequest):
    texto = request.texto
    
    # Divide o texto em partes menores, se necessário
    partes_texto = dividir_texto(texto)
    
    # Inicializa um dicionário para armazenar as pontuações totais das emoções
    emocao_totals = {emocoes: 0 for emocoes in mapa_emocoes.values()}
    total_score = 0
    
    # Processa cada parte do texto
    for parte in partes_texto:
        # Obtém a classificação das emoções para a parte do texto
        resultado = emotion_analyzer(parte)
        
        # Acumula as pontuações e calcula o total
        for item in resultado[0]:
            emocao = mapa_emocoes.get(item['label'], item['label'])
            score = item['score']
            emocao_totals[emocao] += score
            total_score += score
    
    # Calcula a porcentagem de cada emoção e adiciona o símbolo de %
    porcentagens = {emocao: f"{(score / total_score) * 100:.2f}%" if total_score > 0 else "0%" for emocao, score in emocao_totals.items()}
    
    return {"resultado": porcentagens}
