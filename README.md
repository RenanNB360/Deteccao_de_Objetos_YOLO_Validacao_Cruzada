# Projeto de Detecção de Veículos com YOLOv8

## 🎯 Objetivo
Treinar e otimizar um modelo YOLO (You Only Look Once) no conjunto de dados **vehicles-q0x2v** do Roboflow, com foco especial na maximização dos indicadores de **Precision** e **Recall** para detecção precisa de veículos em cenários diversos.

**Dataset**: [vehicles-q0x2v no Roboflow](https://universe.roboflow.com/roboflow-100/vehicles-q0x2v)

## 📊 Métricas-Chave
| Métrica       | Objetivo   | Estratégia de Otimização       |
|---------------|------------|--------------------------------|
| **Precision** | Maximizar  | Aumento de dados difíceis      |
| **Recall**    | Maximizar  | Balanceamento de classes       |


## 📁 Arquivo 1: `code/utils.py`

Este script reúne as funções que desenvolvi para organizar, preparar e validar os dados utilizados no treinamento do modelo YOLO. Desde o início, decidi automatizar todas as etapas importantes para garantir reprodutibilidade, clareza nos dados e uma estrutura sólida para validação cruzada.

---

### 🔧 Minhas Funções e o Porquê de Cada Uma

---

#### `extract_zip_files(path_zip, file_dir)`

**Objetivo:** Extrair os arquivos compactados em `.zip` contendo o dataset para uma pasta local.

**Motivação:**  
Optei por criar essa função para facilitar a etapa inicial de manipulação dos dados, garantindo que a pasta de destino seja criada automaticamente e que o conteúdo seja extraído corretamente, sem depender de ações manuais.

---

#### `initial_imbalance_analysis(train_labels_path, yaml_path)`

**Objetivo:** Analisar a distribuição das classes a partir dos arquivos de rótulo e do arquivo `data.yaml`.

**Motivação:**  
Como o YOLO pode ser sensível ao desbalanceamento entre classes, criei essa função para me ajudar a visualizar a quantidade de instâncias por classe e entender melhor o comportamento do dataset logo no início.

---

#### `vision_balance(class_idx, class_labels, output_path)`

**Objetivo:** Gerar um gráfico de barras com a distribuição de classes e salvar como imagem.

**Motivação:**  
Quis complementar a análise numérica com uma visualização clara e objetiva que pudesse ser usada tanto para entendimento quanto para documentação do processo.

---

#### `prepare_cv_folders(base_output_dir, k=5, class_names=None)`

**Objetivo:** Criar automaticamente a estrutura de diretórios para validação cruzada com `k` folds.

**Motivação:**  
Optei por implementar essa etapa para automatizar a organização dos dados em cada fold, incluindo as subpastas de imagens e rótulos, além da criação do `data.yaml` adaptado para cada um deles.

---

#### `copy_files_to_folds(original_data_dir, base_output_dir, k=5, seed=42)`

**Objetivo:** Distribuir imagens e rótulos entre os folds, separando adequadamente os dados de treino e validação.

**Motivação:**  
Como parte da estratégia de validação cruzada, precisei criar uma forma controlada e aleatória de particionar os dados. Assim, assegurei que os folds não compartilhassem imagens e que o processo fosse reproduzível usando uma `seed`.

---

#### `remove_empty_labels(root_dir)`

**Objetivo:** Identificar e remover arquivos de rótulo vazios e suas respectivas imagens nas pastas de validação.

**Motivação:**  
Decidi implementar essa verificação porque arquivos `.txt` vazios (sem objetos anotados) poderiam comprometer o treinamento e os resultados do YOLO. Para evitar ruído nos dados, optei por removê-los de forma segura.

---

#### `results_fold(fold: int, summary_file: str)`

**Objetivo:** Ler os resultados (Precision e Recall) de cada fold, extraídos do CSV gerado ao final do treino, e armazená-los em um resumo.

**Motivação:**  
Essa função surgiu da necessidade de consolidar os resultados em um único arquivo. Eu queria facilitar a comparação entre folds e centralizar as métricas importantes sem precisar abrir múltiplos arquivos.

---

#### `calculate_summary_stats(summary_file: str)`

**Objetivo:** Calcular e exibir as médias de precisão, recall e a soma de ambos (score combinado).

**Motivação:**  
Ao final do experimento, quis ter uma visão geral e quantitativa do desempenho do modelo, agregando os resultados de todos os folds em uma média final clara.

---

### ✅ Considerações Finais

Esse primeiro script é uma peça-chave do meu pipeline. Ao criá-lo, minha intenção foi garantir:

- Uma **preparação de dados automatizada e confiável**;
- **Validação cruzada estruturada** para análises mais robustas;
- **Visualizações e métricas** que ajudam a entender e melhorar o desempenho do modelo.

Todas as decisões aqui partiram da necessidade de ganhar eficiência, controle e reprodutibilidade no processo de desenvolvimento.


## 📁 Arquivo 2: `code/yolo_weighted_dataset.py`

Neste segundo script, desenvolvi a classe `YOLOWeightedDataset`, uma extensão personalizada da classe `YOLODataset` da biblioteca Ultralytics. A motivação por trás dessa construção surgiu da percepção de que meu conjunto de dados apresentava um forte **desbalanceamento entre classes**, o que poderia comprometer a qualidade do aprendizado do modelo. Com isso, decidi aplicar o conceito de **amostragem ponderada** durante o treinamento, dando mais chance de seleção para imagens que contêm classes menos representadas.

---

### 🧠 Minha Lógica e o Fundamento Matemático por Trás

---

#### `count_instances(self)`

**Objetivo:** Contar quantas vezes cada classe aparece no conjunto de dados.

**Motivação:**  
Implementei essa função para obter uma visão quantitativa da distribuição das classes. A contagem é essencial para gerar os pesos inversamente proporcionais à frequência de cada classe. Para evitar divisão por zero, as contagens zeradas são substituídas por 1.

---

#### `calculate_weights(self)`

**Objetivo:** Calcular um peso para cada imagem com base nas classes que ela contém.

**Motivação:**  
A ideia central aqui foi aplicar o conceito de **peso inversamente proporcional à frequência**. Cada classe recebe um peso com base em sua raridade, e para cada imagem, agrego os pesos das classes que ela possui usando a função `np.mean`. Isso garante que imagens contendo classes raras recebam maior prioridade de amostragem.

---

#### `calculate_probabilities(self)`

**Objetivo:** Normalizar os pesos das imagens em probabilidades de amostragem.

**Motivação:**  
Essa função transforma os pesos calculados anteriormente em uma distribuição de probabilidade válida. O objetivo foi garantir que o método `np.random.choice()` pudesse realizar uma **amostragem ponderada**, respeitando a importância relativa de cada imagem.

---

#### `__getitem__(self, index)`

**Objetivo:** Definir o comportamento de acesso aos itens do dataset.

**Motivação:**  
Aqui defini dois comportamentos distintos:  
- Se estivermos em **modo de validação**, sigo com o carregamento padrão da imagem e seus rótulos.  
- Se estivermos em **modo de treinamento**, utilizo `np.random.choice()` com as probabilidades calculadas para selecionar a imagem a ser usada naquela iteração.

Essa lógica garante que, durante o aprendizado, o modelo tenha maior exposição a exemplos de classes menos frequentes.

---

### 🧩 Conceito Estatístico Aplicado

A base teórica por trás deste código está enraizada no conceito de **amostragem com probabilidade proporcional ao inverso da frequência**, uma técnica comum em cenários com dados desbalanceados. A fórmula utilizada:

```python
class_weights = np.sum(self.counts) / self.counts
```

Reflete exatamente isso: quanto menor a presença de uma classe, maior o seu peso, e portanto, maior sua chance de ser amostrada. Esse tipo de balanceamento ajuda a evitar que o modelo ignore classes minoritárias, o que é crucial em problemas de classificação multiclasse.

Decidi desenvolver esta versão personalizada do YOLODataset porque percebi que o simples balanceamento estrutural dos dados não era suficiente. Ao aplicar pesos estatísticos diretamente sobre a etapa de carregamento de dados, consegui influenciar o aprendizado do modelo de forma mais eficaz e controlada.

Essa abordagem aumentou a equidade entre as classes durante o treino e serviu como uma solução prática para o problema de desbalanceamento, sem alterar diretamente os dados ou forçar a duplicação de exemplos.

Foi uma decisão consciente minha aplicar esse tipo de estratégia, pois me deu mais domínio sobre o processo e reforçou o compromisso com a qualidade e a robustez do modelo final.


## 📁 Arquivo 3: `code/main.py`

Este script representa o momento central do meu pipeline de detecção com YOLO: a execução do treinamento com validação cruzada. A estrutura está pensada para garantir reprodutibilidade, análise detalhada do desbalanceamento, e maior controle estatístico por meio da ponderação dos dados. Todas as escolhas e etapas descritas aqui foram decisões minhas, pensadas para dar mais robustez ao experimento.

---

### 📌 Objetivo Geral

Treinar o modelo YOLOv8 em um dataset multiclasse com desbalanceamento, utilizando uma abordagem com k-fold cross-validation e ponderação amostral com base na frequência das classes. Cada passo foi automatizado para garantir controle, eficiência e reprodutibilidade.

---

### 🔢 Etapas do Script com Minhas Justificativas

---

#### 1. Extração dos Dados Originais

```python
extract_zip_files(raw_data, origin_raw_data)
```

Automatizei a extração dos dados compactados diretamente para a pasta `data` para garantir que o início do pipeline fosse simples, rápido e livre de intervenção manual. Essa etapa evita retrabalho e mantém consistência sempre que o processo for repetido.

---

#### 2. Análise Inicial do Desbalanceamento

```python
class_labels, class_counts = initial_imbalance_analysis(raw_train_labels, origin_yaml)
first_distribution_analysis = os.path.join(result, 'raw_distribution.png')
vision_balance(class_counts, class_labels, first_distribution_analysis)
```

Logo após a extração, faço uma análise quantitativa e visual da distribuição das classes. Isso me permite ter um diagnóstico inicial claro e justificável para a necessidade de balanceamento. A visualização em gráfico torna fácil perceber quais classes estão sub-representadas.

---

#### 3. Substituição do Dataset Interno do YOLO

```python
build.YOLODataset = YOLOWeightedDataset
```

Essa substituição foi uma das decisões mais importantes do processo. Ao trocar o `YOLODataset` padrão pelo meu `YOLOWeightedDataset`, consigo alterar o comportamento do carregamento dos dados para aplicar pesos conforme a frequência das classes. Quanto menor a frequência de uma classe, maior seu peso, e maior a probabilidade da imagem ser amostrada. Essa lógica evita que o modelo negligencie as classes minoritárias sem forçar duplicação de dados ou intervenções nos rótulos.

---

#### 4. Preparação das Estruturas de Validação Cruzada

```python
prepare_cv_folders(kfold_dir, class_names=class_labels)
copy_files_to_folds(origin_raw_data, kfold_dir)
```

Implementei a criação automatizada de pastas com base na estratégia de validação cruzada `k=5`. Essa abordagem garante uma avaliação estatisticamente mais confiável do modelo. Cada fold contém seus próprios arquivos e `data.yaml`, e as amostras são distribuídas de forma controlada com seed, para reprodutibilidade.

---

#### 5. Remoção de Labels Vazios

```python
remove_empty_labels(kfold_dir)
```

Decidi remover imagens que não possuíam objetos anotados, pois o YOLO pode interpretar essas imagens de forma ambígua, resultando em falsos negativos. Essa verificação automática evita que rótulos vazios comprometam o treinamento ou distorçam as métricas.

---

#### 6. Loop de Treinamento com os 5 Folds

```python
for fold in range(1, 6):
    ...
    results = model.train(
        data=str(fold_yaml),
        epochs=20,
        imgsz=640,
        batch=16,
        name=f"fold_{fold}_results",
        project="runs/train",
        verbose=True,
        workers=0,
        device=0,
        save_period=1,
        patience=3,
    )
```

Aqui executo o treino de cada fold individualmente. Algumas das principais decisões nos parâmetros:

- `epochs=20`: Avaliei que esse número era suficiente para o modelo aprender, sem overfitting excessivo, dado o tamanho do dataset.
- `imgsz=640`: Resolvi manter o padrão do YOLOv8, que oferece bom equilíbrio entre performance e custo computacional.
- `batch=16`: Escolhi esse valor com base nos recursos da GPU e estabilidade do treinamento.
- `device=0`: Uso explícito da GPU, garantindo que o processamento seja feito com maior velocidade.
- `patience=3`: Adicionei paciência no early stopping para interromper o treino cedo caso não houvesse melhora consistente.
- `save_period=1`: Decidi salvar os pesos a cada época para possibilitar inspeções posteriores e comparações entre checkpoints.

Após cada treino, chamo `results_fold()` para consolidar as métricas em um arquivo CSV.

---

#### 7. Consolidação dos Resultados

```python
calculate_summary_stats(summary_file)
```

No fim do processo, aplico a função que calcula a média das métricas de *Precision* e *Recall* para todos os folds. Essa visão global me ajuda a entender a performance real do modelo, reduzindo o viés que pode existir em uma única divisão treino/validação.

---

# Resultados Apresentados

Ao final do processo de validação cruzada utilizando 5 folds, os resultados médios do modelo YOLO aplicado ao dataset `vehicles-q0x2v` mostraram desempenho consistente, conforme evidenciado pelos indicadores de *Precision* e *Recall*. Abaixo, apresento a tabela com os resultados por fold:

| Fold   | Precision | Recall | Combined Score |
|--------|-----------|--------|----------------|
| Fold_1 | 0.8826    | 0.8700 | 1.7527         |
| Fold_2 | 0.8658    | 0.8546 | 1.7203         |
| Fold_3 | 0.8816    | 0.9032 | 1.7849         |
| Fold_4 | 0.8901    | 0.8843 | 1.7744         |
| Fold_5 | 0.8426    | 0.8630 | 1.7056         |

Esses resultados estão disponíveis no diretório `results`, organizados por fold, contendo logs, métricas e checkpoints do treinamento.

---

## Melhorias Futuras

A performance do modelo YOLOv8 foi positiva de forma geral, mas há pontos a serem aperfeiçoados com base em observações feitas durante o desenvolvimento:

1. **Amostragem Ponderada com Pesos por Classe**  
   A estratégia de balanceamento com a classe `YOLOWeightedDataset` já trouxe ganhos ao priorizar imagens com classes menos frequentes. Contudo, há espaço para testar funções alternativas de agregação, como `np.max` ou `np.sum`, além da `np.mean` já utilizada.

2. **Utilização de `counters_per_class`**  
   Um próximo passo natural seria integrar a contagem de objetos por classe diretamente na estratégia de ponderação, usando algo como `counters_per_class=` para ajustar a exposição do modelo às instâncias menos representadas de forma ainda mais direta.

3. **Loss Functions Alternativas — Uso do Focal Loss**  
   A substituição da função de perda padrão pelo *Focal Loss* pode ser bastante eficaz para lidar com o desbalanceamento. Ela força o modelo a prestar mais atenção em exemplos difíceis e classes minoritárias, o que complementa bem a estratégia atual de amostragem.

4. **Otimização de Hiperparâmetros**  
   Os parâmetros usados — como `epochs=20`, `batch=16` e `imgsz=640` — foram adequados para o cenário inicial, mas é recomendável utilizar frameworks como *Optuna* para automatizar a busca por combinações mais eficazes.

5. **Aprimoramento na Validação Cruzada**  
   A estrutura dos folds está sólida, porém futuramente posso incluir uma separação estratificada por classe para garantir melhor uniformidade entre os folds, especialmente útil se novas classes forem adicionadas.

6. **Exploração de Técnicas de Data Augmentation Avançadas**  
   Atualmente o pipeline YOLO já faz augmentations básicas, mas inserir técnicas como *Mosaic* ou *MixUp* de forma controlada e específica por classe pode ajudar o modelo a aprender com menos viés.

7. **Melhor Tratamento de Labels Vazios**  
   A função de remoção de labels vazios é uma etapa essencial, mas futuramente posso marcar essas imagens em vez de removê-las, permitindo seu uso para avaliação de falsos positivos.

---



