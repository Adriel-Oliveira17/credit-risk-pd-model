# Modelagem de PD (*Probability of Default*) com Regressão Logística e Validação Out-of-Time

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-LogisticRegression-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![Risk](https://img.shields.io/badge/Credit%20Risk-Basel%20%2F%20IFRS%209-B10909)

Projeto de Ciência de Dados aplicado a **Risco de Crédito** para estimar a probabilidade de um cliente de crédito rotativo (cartão de crédito) entrar em *default* (> 90 dias de atraso) nos próximos 12 meses (**PD — *Probability of Default***). A abordagem utiliza **Regressão Logística**, escolhida pela transparência e facilidade de auditoria, com seleção de variáveis por ***Information Value* (IV)**, validação temporal (*Out-of-Time*) e construção de uma **matriz de ratings (A–H)**.

---

## Visão Geral e Contexto de Negócio

A PD é um dos três componentes da perda esperada ($\text{EL} = \text{PD} \times \text{LGD} \times \text{EAD}$) e sustenta decisões de concessão de crédito, precificação, provisão e capital regulatório.

Para cada cliente adimplente na safra de observação, o modelo estima:

$$
\text{PD}_{12m} = P(\text{default}_{12m} = 1 \mid X)
$$

Em seguida, a PD é convertida em **ratings (A–H)**, agrupando clientes de risco homogêneo e permitindo acompanhar a carteira por faixa de risco.

---

## Estrutura do Projeto

```text
├── gerador.ipynb      # Simulação da base sintética de clientes (gera o Dados.csv)
├── modelo PD.ipynb    # Target, seleção por IV, treino OOT, avaliação, ratings e scoring
└── README.md          # Documentação do projeto
```

> O `Dados.csv` **não é versionado** (está no `.gitignore`). Ele é gerado ao executar o `gerador.ipynb`.

---

## Dicionário de Dados

A base simulada contém **30.000 registros** distribuídos em quatro safras anuais de observação (`2022` a `2025`), com variáveis observadas na safra e o indicador de *default* nos 12 meses seguintes:

| Variável | Tipo | Descrição |
| :--- | :--- | :--- |
| `safra_obs` | `int` | Ano da safra de observação (`2022`, `2023`, `2024` ou `2025`). |
| `renda_mensal` | `float` | Renda mensal declarada/estimada do cliente (R\$). |
| `limite` | `float` | Limite total de crédito disponível na safra de observação (R\$). |
| `saldo` | `float` | Saldo devedor utilizado na safra de observação (R\$). |
| `utilizacao_inicial` | `float` | Taxa de utilização do limite (`saldo / limite`), entre `0` e `1`. |
| `variacao_saldo_3m` | `float` | Variação percentual do saldo nos 3 meses anteriores. |
| `qtd_atrasos_curtos_6m` | `int` | Quantidade de atrasos de curta duração nos 6 meses anteriores. |
| `pagou_minimo_ult_fatura` | `int` | Indicador binário (`0` ou `1`) se o cliente pagou apenas o mínimo na última fatura. |
| `default_12m` | `int` | **Target.** `1` se o cliente atrasou > 90 dias nos 12 meses seguintes à observação. |
| `saldo_default` | `float` | Saldo no dia do *default*. Preenchido apenas quando `default_12m = 1`; usado no modelo de EAD, não no PD. |

---

## Metodologia e Pipeline de Modelagem

1. **Geração de Dados (`gerador.ipynb`):**
   - Simulação de 30.000 clientes com `seed=42`. O `default_12m` é gerado por uma função logística de atrasos, utilização, variação de saldo e pagamento mínimo, com ruído, de forma que `renda_mensal` e `limite` não tenham efeito direto no risco.
2. **Estruturação da Base e Target:**
   - População adimplente na `safra_obs`, janela de performance de 12 meses (padrão Basileia/IFRS 9) e `default_12m = 1` para atraso > 90 dias.
3. **Engenharia de Atributos:**
   - Razões financeiras: `percentual_uso`, `comprometimento_renda`, `alavancagem_limite` e `limite_restante`.
4. **Seleção por *Information Value* (IV):**
   - Binning por faixas (`pd.cut`) e cálculo do IV de cada variável:

   | IV | Decisão |
   | :---: | :--- |
   | `< 0,02` | Descartar |
   | `0,02 – 0,10` | Fraco |
   | `0,10 – 0,30` | Médio |
   | `0,30 – 0,50` | Forte |
   | `> 0,50` | Alerta de *leakage* |

5. **Validação Temporal (*Out-of-Time* — OOT):**
   - **In-Time (treino):** o modelo é ajustado nas safras de `2022` a `2024`.
   - **Out-of-Time (validação):** o desempenho é medido na safra de `2025`, que não participa do ajuste, simulando o uso real do modelo em dados futuros.
6. **Treinamento do Modelo:**
   - `StandardScaler` ajustado **apenas no treino** + `LogisticRegression`.
7. **Avaliação:**
   - AUC, Gini, KS, *Brier Score*, curva de calibração, PSI e matriz de ratings.
8. **Aplicação:**
   - *Scoring* de uma carteira sintética de 20 clientes, classificados em risco Baixo, Médio, Alto e Crítico.

---

## Resultados e Desempenho (Safra OOT 2025)

| Métrica | Treino (2022–2024) | OOT (2025) |
| :--- | :---: | :---: |
| **AUC** | `0,6756` | `0,6869` |
| **Gini** (`2·AUC − 1`) | `0,3512` | `0,3738` |
| **KS** | `0,2547` | `0,2973` |
| **Brier Score** | — | `0,0674` |
| **PSI (treino vs. 2025)** | — | `0,0012` (estável, < 0,10) |

O desempenho no OOT é equivalente ao do treino, sem sinal de *overfitting*. A taxa média de *default* é estável entre as safras (de `6,92%` a `7,56%`).

### Matriz de Ratings (OOT 2025)

Pontos de corte definidos nos quantis da PD prevista no treino:

| Rating | Clientes | % Carteira | PD prevista média | *Bad rate* real |
| :---: | :---: | :---: | :---: | :---: |
| A | 903 | 12,2% | 2,41% | 2,66% |
| B | 893 | 12,0% | 3,47% | 3,47% |
| C | 909 | 12,2% | 4,30% | 3,63% |
| D | 945 | 12,7% | 5,20% | 5,19% |
| E | 922 | 12,4% | 6,28% | 6,62% |
| F | 946 | 12,7% | 7,65% | 9,41% |
| G | 945 | 12,7% | 9,80% | 10,69% |
| H | 969 | 13,0% | 16,37% | 17,96% |

A taxa de *default* real cresce de forma monotônica de A a H.

### *Information Value*

| Variável | IV | Decisão |
| :--- | :---: | :---: |
| `qtd_atrasos_curtos_6m` | `0,1795` | Médio |
| `percentual_uso` | `0,1111` | Médio |
| `utilizacao_inicial` | `0,1108` | Médio |
| `comprometimento_renda` | `0,1024` | Médio |
| `limite_restante` | `0,0999` | Fraco |
| `variacao_saldo_3m` | `0,0712` | Fraco |
| `saldo` | `0,0316` | Fraco |
| `renda_mensal` | `0,0148` | Descartar |
| `limite` | `0,0111` | Descartar |
| `alavancagem_limite` | `0,0002` | Descartar |

---

## Como Executar

### 1. Pré-requisitos
Python 3.9+ e as seguintes bibliotecas (também listadas no `requirements.txt`):

| Biblioteca | Uso |
| :--- | :--- |
| `pandas` | Manipulação dos dados, binning e agregações |
| `numpy` | Operações numéricas e simulação da base |
| `scikit-learn` | `StandardScaler`, `LogisticRegression` e métricas |
| `matplotlib` | Gráficos |
| `seaborn` | Gráficos estatísticos |
| `jupyter` | Execução dos notebooks |

Instale tudo com:

```bash
pip install -r requirements.txt
```

Ou diretamente:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 2. Ordem de Execução
1. Execute todas as células de **`gerador.ipynb`** para gerar o arquivo `Dados.csv`.
2. Execute todas as células de **`modelo PD.ipynb`** para reproduzir a seleção por IV, treinar o modelo, avaliar as métricas na safra OOT (`2025`), montar a matriz de ratings e escorar a carteira sintética.

---

## Limitações

- Gini OOT de ~0,37, um pouco abaixo da referência de ~0,40 para modelos comportamentais.
- A base é **sintética**: o desempenho reflete as relações definidas no gerador, e não o comportamento de uma carteira real.
- `pagou_minimo_ult_fatura` ainda não foi avaliada no IV nem testada no modelo.
- A carteira de teste da última seção é fictícia e serve apenas para demonstração.

---

## Stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn
