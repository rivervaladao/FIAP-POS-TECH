# Processo de Seleção e Preparação do Dataset e Fine-Tuning do Modelo

## 1. Processo de Seleção e Preparação do Dataset

### 1.1 Seleção do Dataset

O dataset utilizado para este projeto é o "The AmazonTitles-1.3MM", que consiste em consultas textuais reais de usuários e títulos associados de produtos relevantes encontrados na Amazon, juntamente com suas descrições. Este dataset foi escolhido por sua relevância para a tarefa de geração de respostas baseadas em títulos e descrições de produtos.

### 1.2 Carregamento do Dataset

O dataset é carregado a partir de um arquivo JSON local usando a biblioteca `datasets` do Hugging Face:

```python
dataset = load_dataset('json', data_files={'train': f'{WORKSPACE_PATH}trn.json'})
```

### 1.3 Amostragem e Embaralhamento

Para fins de treinamento e eficiência computacional, uma amostra aleatória de 100.000 exemplos é selecionada do dataset original. O dataset é embaralhado para garantir uma distribuição aleatória dos exemplos:

```python
dataset = dataset['train'].shuffle(seed=42).select(range(100000))
```

### 1.4 Preparação dos Dados

#### 1.4.1 Formatação com to_sharegpt

O dataset é formatado usando a função `to_sharegpt` da biblioteca Unsloth. Esta função prepara os dados para o formato esperado pelo modelo de linguagem:

```python
dataset = to_sharegpt(
    dataset,
    merged_prompt="Título do produto: {title}",
    output_column_name="content",
    conversation_extension=3,
)
```

- O título do produto é usado como entrada (prompt).
- A descrição do produto (coluna "content") é usada como saída esperada.
- `conversation_extension=3` é usado para criar exemplos de conversação mais longos.

#### 1.4.2 Padronização com standardize_sharegpt

Os dados são então padronizados usando a função `standardize_sharegpt`:

```python
dataset = standardize_sharegpt(dataset)
```

Esta etapa garante que todos os exemplos sigam um formato consistente.

#### 1.4.3 Aplicação do Template de Chat

Um template de chat personalizado é aplicado aos dados usando a função `apply_chat_template`:

```python
chat_template = """Responda à pergunta sobre o produto com base no título e na descrição fornecidos.

### Título do produto:
{INPUT}

### Pergunta:
{question}

### Resposta (baseada na descrição):
{OUTPUT}"""

dataset = apply_chat_template(
    dataset,
    tokenizer=tokenizer,
    chat_template=chat_template,
)
```

Este template estrutura os dados em um formato de pergunta-resposta, preparando-os para o treinamento do modelo.

### 1.5 Verificação Final

Após a preparação, uma amostra do dataset processado é impressa para verificação:

```python
print(dataset[:2])
```

Esta etapa final permite uma inspeção visual dos dados preparados, garantindo que estejam no formato correto antes do treinamento.

## 2. Processo de Fine-Tuning do Modelo

### 2.1 Seleção do Modelo Base

O modelo base escolhido para o fine-tuning é o "unsloth/llama-3-8b-bnb-4bit", um modelo Llama-3 de 8 bilhões de parâmetros otimizado para eficiência computacional.

### 2.2 Carregamento do Modelo e Tokenizador

O modelo e o tokenizador são carregados usando a biblioteca FastLanguageModel da Unsloth:

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
```

Parâmetros:
- `max_seq_length=2048`: Define o comprimento máximo da sequência.
- `dtype=None`: Permite a detecção automática do tipo de dados mais adequado.
- `load_in_4bit=True`: Carrega o modelo em formato de 4 bits para economia de memória.

### 2.3 Adição de LoRA (Low-Rank Adaptation)

LoRA é aplicado ao modelo para permitir um fine-tuning eficiente:

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
)
```

Parâmetros LoRA:
- `r=16`: Rank da matriz de adaptação.
- `target_modules`: Lista de módulos do modelo a serem adaptados.
- `lora_alpha=16`: Escala da adaptação LoRA.
- `lora_dropout=0`: Sem dropout na adaptação LoRA.
- `bias="none"`: Não adapta os bias.
- `use_gradient_checkpointing="unsloth"`: Usa checkpointing de gradiente otimizado.

### 2.4 Configuração do Treinamento

O treinamento é configurado usando `TrainingArguments`:

```python
training_args = TrainingArguments(
    output_dir=f'{WORKSPACE_PATH}results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    max_steps=1000,
    learning_rate=2e-4,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=10,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    report_to="none",
)
```

Principais parâmetros:
- `num_train_epochs=3`: Número de épocas de treinamento.
- `per_device_train_batch_size=2`: Tamanho do batch por dispositivo.
- `gradient_accumulation_steps=4`: Acumulação de gradientes para simular batches maiores.
- `learning_rate=2e-4`: Taxa de aprendizado.
- `optim="adamw_8bit"`: Otimizador AdamW em 8 bits para economia de memória.

### 2.5 Configuração do SFTTrainer

O `SFTTrainer` é configurado para realizar o fine-tuning:

```python
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=training_args,
)
```

Parâmetros importantes:
- `dataset_text_field="text"`: Campo do dataset contendo o texto para treinamento.
- `max_seq_length=max_seq_length`: Comprimento máximo da sequência.
- `packing=False`: Desativa o empacotamento de sequências.

### 2.6 Execução do Fine-Tuning

O fine-tuning é executado chamando o método `train()`:

```python
trainer.train()
```

### 2.7 Salvamento do Modelo

Após o treinamento, o modelo e o tokenizador são salvos:

```python
trainer.save_model(f"{WORKSPACE_PATH}unsloth_model_optimized")
tokenizer.save_pretrained(f"{WORKSPACE_PATH}unsloth_model_optimized")
```

Este processo de fine-tuning utiliza técnicas avançadas como LoRA e otimizações de memória para adaptar eficientemente o modelo Llama-3 à tarefa específica de geração de respostas baseadas em títulos e descrições de produtos.
```