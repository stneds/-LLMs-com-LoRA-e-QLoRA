# Laboratório 07: Especialização de LLMs com LORA e QLORA

[cite_start]Este repositório contém a implementação de um pipeline completo de fine-tuning para o modelo de linguagem **TinyLlama-1.1B**, utilizando as técnicas de **LoRA (Low-Rank Adaptation)** e **QLoRA (Quantized LoRA)** para otimização de memória em hardware limitado.

## 📋 Objetivo
[cite_start]O objetivo deste projeto é realizar a especialização de um modelo fundacional em um domínio específico (atendimento ao cliente para academias) através de quantização de 4-bits e ajuste de hiperparâmetros de eficiência de parâmetros.

## 🏗️ Estrutura do Projeto
Seguindo a organização do repositório proposta:

- [cite_start]`data/`: Contém os datasets sintéticos gerados em formato `.jsonl`[cite: 10].
- `src/`: Scripts principais de execução.
  - [cite_start]`generate_dataset.py`: Script que consome a API da OpenAI para criar 60 pares de instrução/resposta.
  - [cite_start]`train.py`: Script de treinamento com configurações de QLORA e SFTTrainer[cite: 15, 23].
- `infer.py`: Script para carregar o adaptador treinado e realizar inferências.
- `requirements.txt`: Lista de dependências para o ambiente Python.

## ⚙️ Configurações de Fine-Tuning
[cite_start]Para atender aos critérios técnicos do laboratório, foram utilizados os seguintes hiperparâmetros[cite: 18, 19, 20, 21, 32, 33, 34]:

| Parâmetro | Valor Configurado |
| :--- | :--- |
| **Rank (r)** | 64 |
| **Alpha (α)** | 16 |
| **Dropout** | 0.1 |
| **Otimizador** | `paged_adamw_32bit` |
| **LR Scheduler** | `cosine` |
| **Warmup Ratio** | 0.03 |
| **Quantização** | 4-bit (nf4) |

## 🚀 Como Executar
1. Instale as dependências: `pip install -r requirements.txt`.
2. Gere o dataset sintético: `python src/generate_dataset.py`.
3. Inicie o treinamento: `python src/train.py`.
4. Teste o modelo especializado: `python infer.py`.

## 📌 Versão
[cite_start]Este projeto segue o contrato pedagógico e está marcado com a tag **v1.0** para avaliação final.
