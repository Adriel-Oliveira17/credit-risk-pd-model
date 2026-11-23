# Modelagem de PD (*Probability of Default*)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-LogisticRegression-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![Risk](https://img.shields.io/badge/Credit%20Risk-Basel%20%2F%20IFRS%209-B10909)

Projeto de Ciência de Dados aplicado a **Risco de Crédito** para estimar a probabilidade de inadimplência (**PD — *Probability of Default***) em um horizonte de 12 meses para uma carteira de crédito rotativo. A abordagem utiliza **Regressão Logística** (`LogisticRegression` com `StandardScaler`), seleção de variáveis via **Information Value (IV)** e validação temporal (**Out-of-Time — OOT**) com monitoramento de estabilidade populacional (**PSI**).

---

## Visão Geral e Contexto de Negócio

No arcabouço regulatório de **Basileia** e **IFRS 9**, o cálculo da Perda Esperada ($\text{EL} = \text{PD} \times \text{LGD} \times \text{EAD}$) exige estimar a probabilidade de um cliente adimplente entrar em *default* (atraso superior a 90 dias) dentro de uma janela fixa de **12 meses** após a data de observação (`safra_obs`).

Para garantir aderência conceitual e evitar ruído estatístico, a estruturação da base segue critérios rigorosos de elegibilidade:

- **População elegível ($t = 0$):** Apenas contratos ativos e adimplentes na data de observação (`safra_obs`), excluindo clientes já em *default* ou cancelados antes da foto inicial.
- **Horizonte de performance (12 meses):** Marcação binária `default_12m = 1` (Mau) para clientes com atraso $> 90$ dias em qualquer momento dos 12 meses subsequentes, e `default_12m = 0` (Bom) caso contrário.
- **Tratamento de contratos indeterminados (produção):** Contratos encerrados antecipadamente sem atraso são mantidos como `0` (Bom) caso tenham completado ao menos 6 meses de observação, ou removidos da base de treino quando o histórico é insuficiente.

---

## Estrutura do Projeto

```text
├── gerador.ipynb      # Simulação da base sintética de clientes e eventos de default (Dados.csv)
├── modelo PD.ipynb    # Estruturação do target, engenharia de atributos, IV, treino OOT e validação
├── Dados.csv          # Dataset gerado com 60.000 clientes e safras de observação de 2022 a 2025
└── README.md          # Documentação do projeto
```

---

## Dicionário de Dados

A base utilizada contém **60.000 registros** distribuídos nas safras de observação de `2022` a `2025`:

| Variável | Tipo | Descrição |
| :--- | :--- | :--- |
| `safra_obs` | `int` | Ano da safra de observação do cliente (`2022`, `2023`, `2024` ou `2025`). |
| `renda_mensal` | `float` | Renda mensal declarada/estimada do cliente (R\$). |
| `limite` | `float` | Limite total de crédito disponível na data de observação (R\$). |
| `saldo` | `float` | Saldo devedor utilizado na data de observação (R\$). |
| `variacao_saldo_3m` | `float` | Variação percentual do saldo nos 3 meses anteriores à data de observação. |
| `qtd_atrasos_curtos_6m` | `int` | Quantidade de atrasos de curta duração nos 6 meses anteriores. |
| `pagou_minimo_ult_fatura` | `int` | Indicador binário (`0` ou `1`) se o cliente pagou apenas o mínimo na última fatura. |
| `saldo_default` | `float` | Saldo devedor real registrado no momento do *default* (`0.0` para adimplentes) (R\$). |
| `default_12m` | `int` | **Variável-alvo:** indicador binário (`0` = Bom, `1` = Mau) de *default* ($> 90$ dias) em 12 meses. |

---

## Metodologia e Pipeline de Modelagem

1. **Engenharia de Atributos (*Feature Engineering*):**
   - `percentual_uso` = `saldo / limite` — captura o nível de alavancagem e consumo do limite aprovado.
   - `variacao_saldo_clip` = `np.clip(variacao_saldo_3m, -0.5, 1.0)` — aplicação de *winsorization* para mitigar o impacto de *outliers* nos coeficientes lineares.
   - **Pré-seleção de 6 candidatas:** `qtd_atrasos_curtos_6m`, `percentual_uso`, `variacao_saldo_clip`, `pagou_minimo_ult_fatura`, `renda_mensal` e `limite` (removendo `saldo` isolado para evitar colinearidade com `percentual_uso`).
2. **Validação Temporal (*Out-of-Time* — OOT):**
   - **Treino:** Safras de observação anteriores a 2025 (`safra_obs < 2025`, contemplando `2022`, `2023` e `2024`).
   - **Teste OOT:** Safra futura `safra_obs == 2025`, simulando a aplicação real em produção.
3. **Seleção de Variáveis (*Information Value* — IV):**
   - Cálculo do IV realizado **exclusivamente na base de treino** (com discretização em 5 faixas quantílicas) para evitar *data leakage*.
   - Corte pelo critério padrão de mercado ($\text{IV} \ge 0.02$), mantendo **4 variáveis finais**: `qtd_atrasos_curtos_6m`, `percentual_uso`, `variacao_saldo_clip` e `pagou_minimo_ult_fatura` (descartando `renda_mensal` e `limite` por baixo poder discriminatório).
4. **Treinamento do Modelo (`StandardScaler` + `LogisticRegression`):**
   - Pipeline construído com `make_pipeline(StandardScaler(), LogisticRegression())` para padronização das escalas e alta interpretabilidade dos coeficientes, gerando o escore de probabilidade `pd_estimada`.

---

## Resultados e Desempenho (Safra OOT 2025)

O modelo apresentou poder de discriminação consistente e excelente estabilidade populacional na safra futura de teste (`safra_obs == 2025`):

| Métrica | Valor Obtido | Referência de Mercado | Interpretação |
| :--- | :---: | :---: | :--- |
| **AUC** | `0.7073` | Saudável: $\ge 0.70$ \| Ideal: $> 0.75$ | Boa capacidade global de ordenação entre Bons e Maus pagadores |
| **Gini** | `0.4146` | Saudável: $\ge 0.40$ \| Ideal: $> 0.50$ | Coeficiente de discriminação dentro do patamar saudável para crédito rotativo |
| **KS** | `0.3045` | Saudável: $\ge 0.30$ \| Ideal: $> 0.40$ | Separação máxima satisfatória entre as distribuições acumuladas de Bons e Maus |
| **PSI** | `0.0009` | Saudável: $< 0.10$ \| Crítico: $> 0.25$ | Estabilidade populacional excelente, sem sinais de *drift* na safra 2025 |

---

## Como Executar

### 1. Pré-requisitos
Instale as bibliotecas necessárias:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

### 2. Ordem de Execução
1. Certifique-se de que o arquivo **`Dados.csv`** (gerado pelo pipeline de simulação) esteja no diretório raiz do projeto.
2. Execute todas as células de **`modelo PD.ipynb`** para processar a engenharia de atributos, calcular o IV na base de treino, ajustar a Regressão Logística e validar as métricas de discriminação e estabilidade na safra OOT (`2025`).
