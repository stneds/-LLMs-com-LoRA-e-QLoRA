import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

MODELO_BASE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DIRETORIO_SAIDA = "./artifacts/lora"

def iniciar_treinamento():
    # Carregamento do dataset gerado
    colecao = load_dataset("json", data_files={
        "train": "data/train.jsonl", 
        "test": "data/test.jsonl"
    })

    # Passo 2: Configuração de Quantização (QLoRA) [cite: 11, 14]
    config_bits = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Carregar modelo com quantização [cite: 14]
    modelo_llm = AutoModelForCausalLM.from_pretrained(
        MODELO_BASE,
        quantization_config=config_bits,
        device_map="auto"
    )
    
    processador_texto = AutoTokenizer.from_pretrained(MODELO_BASE)
    processador_texto.pad_token = processador_texto.eos_token

    # Passo 3: Arquitetura LoRA [cite: 15, 17, 18]
    config_peft = LoraConfig(
        r=64,                # Rank [cite: 19]
        lora_alpha=16,       # Alpha [cite: 20]
        lora_dropout=0.1,    # Dropout [cite: 21]
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )

    # Passo 4: Pipeline e Otimizador [cite: 22, 31]
    parametros_treino = TrainingArguments(
        output_dir=DIRETORIO_SAIDA,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",   # Otimizador exigido [cite: 32, 43]
        learning_rate=2e-4,
        lr_scheduler_type="cosine",  # Scheduler exigido [cite: 33]
        warmup_ratio=0.03,           # Warmup exigido [cite: 34]
        num_train_epochs=1,
        fp16=True,
        logging_steps=5,
        evaluation_strategy="steps",
        eval_steps=20,
        report_to="none"
    )

    orquestrador = SFTTrainer(
        model=modelo_llm,
        train_dataset=colecao["train"],
        eval_dataset=colecao["test"],
        peft_config=config_peft,
        dataset_text_field="prompt",
        max_seq_length=512,
        tokenizer=processador_texto,
        args=parametros_treino,
    )

    orquestrador.train()
    
    # Salvar o modelo adaptador final [cite: 35]
    orquestrador.model.save_pretrained(DIRETORIO_SAIDA)
    print("Treinamento finalizado com sucesso.")

if __name__ == "__main__":
    iniciar_treinamento()