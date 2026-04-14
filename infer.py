import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PATH_ADAPTADOR = "./artifacts/lora"

def executar_pergunta(texto_usuario):
    tokenizador = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    # Mescla o conhecimento treinado com o modelo base
    modelo_final = PeftModel.from_pretrained(base, PATH_ADAPTADOR)
    
    entrada = tokenizador(
        f"### Pergunta: {texto_usuario}\n### Resposta:", 
        return_tensors="pt"
    ).to("cuda")
    
    with torch.no_grad():
        geracao = modelo_final.generate(**entrada, max_new_tokens=80)
    
    print(tokenizador.decode(geracao[0], skip_special_tokens=True))

if __name__ == "__main__":
    pergunta = "Quais são os benefícios do treino de hipertrofia?"
    executar_pergunta(pergunta)