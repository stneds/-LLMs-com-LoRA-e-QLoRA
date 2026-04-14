import json
import os
import random
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Pasta de saída conforme sua estrutura de diretórios
FOLDER_DATA = Path("data")
FOLDER_DATA.mkdir(parents=True, exist_ok=True)

# Domínio focado em suporte de academia
CONTEXTO = "suporte técnico e atendimento ao cliente para academias de musculação"
TOTAL_ITENS = 60
PROPORCAO_TREINO = 0.9

def salvar_arquivo_jsonl(caminho, lista_dados):
    with open(caminho, "w", encoding="utf-8") as arquivo:
        for entrada in lista_dados:
            arquivo.write(json.dumps(entrada, ensure_ascii=False) + "\n")

def fabricar_dados():
    cliente_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    instrucao_sistema = (
        f"Você é um especialista em '{CONTEXTO}'. "
        f"Gere uma lista JSON com exatamente {TOTAL_ITENS} objetos. "
        "Cada objeto deve ter as chaves 'prompt' e 'response'."
    )
    
    solicitacao = cliente_ai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": instrucao_sistema}],
        temperature=0.7
    )
    
    conteudo_bruto = solicitacao.choices[0].message.content
    if "```json" in conteudo_bruto:
        conteudo_bruto = conteudo_bruto.split("```json")[1].split("```")[0].strip()
        
    base_conhecimento = json.loads(conteudo_bruto)
    random.shuffle(base_conhecimento)
    
    # Divisão 90/10 [cite: 10]
    ponto_corte = int(len(base_conhecimento) * PROPORCAO_TREINO)
    lote_treino = base_conhecimento[:ponto_corte]
    lote_validacao = base_conhecimento[ponto_corte:]
    
    salvar_arquivo_jsonl(FOLDER_DATA / "dataset.jsonl", base_conhecimento)
    salvar_arquivo_jsonl(FOLDER_DATA / "train.jsonl", lote_treino)
    salvar_arquivo_jsonl(FOLDER_DATA / "test.jsonl", lote_validacao)
    
    print(f"Processo concluído: {len(lote_treino)} treinos e {len(lote_validacao)} testes criados.")

if __name__ == "__main__":
    fabricar_dados()