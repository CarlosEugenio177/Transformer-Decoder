# Implementação Didática de Transformer com Encoder e Decoder

## Descrição Geral

Este repositório apresenta uma implementação simplificada e educacional da arquitetura Transformer, tal como descrita no artigo seminal "Attention Is All You Need" (Vaswani et al., 2017). O projeto foi desenvolvido com foco explícito na compreensão dos fundamentos matemáticos subjacentes à arquitetura, utilizando Python, NumPy e Pandas para implementar os blocos essenciais de um modelo Transformer funcional.

A implementação não visa produzir um modelo treinável ou otimizado para aplicações em produção, mas sim servir como ferramenta pedagógica para clarificar os mecanismos internos de processamento sequencial e contextualização de dados através de mecanismos de atenção. Cada componente foi desenvolvido de forma modular e annotada para facilitar o acompanhamento lógico da execução.

## Objetivos de Aprendizagem

Ao estudar este projeto, o aprendizado esperado inclui:

- **Compreender o mecanismo de atenção**: Entender como a atenção escalada por produto escalar (scaled dot-product attention) permite que o modelo pondere diferentes partes da entrada.
- **Visualizar o fluxo do Encoder**: Observar como múltiplas camadas de auto-atenção e redes feed-forward contextualizam progressivamente as representações dos dados.
- **Analisar o funcionamento do Decoder**: Entender como máscaras causais previnem o acesso a tokens futuros durante a geração auto-regressiva.
- **Explorar cross-attention**: Compreender como o Decoder acessa informações codificadas pelo Encoder através de mecanismos de atenção cruzada.
- **Reconhecer normalização e regularização**: Observar o papel da normalização de camada e da normalização residual (Add & Norm).
- **Conectar teoria e prática**: Relacionar equações matemáticas com implementações em código Python.

## Estrutura do Repositório

```
Transformer Decoder/
├── README.md                 # Este arquivo
├── data.py                   # Preparação de dados e embeddings
├── attention.py              # Componentes de atenção e feed-forward
├── encoder.py                # Stack de camadas do Encoder
├── decoder.py                # Stack de camadas do Decoder com auto-regressão
└── __pycache__/              # Cache de bytecode Python (ignorar)
```

## Tecnologias Utilizadas

- **Python 3.7+**: Linguagem de programação base.
- **NumPy**: Biblioteca de computação numérica para manipulação de arrays multidimensionais e operações matriciais.
- **Pandas**: Estruturas de dados e ferramentas para análise (utilizada principalmente para visualização e validação).

## Como Executar

### Configuração do Ambiente Virtual

Para isolar as dependências do projeto, recomenda-se criar um ambiente virtual Python:

**No Windows (PowerShell ou CMD):**
```bash
python -m venv venv
venv\Scripts\activate
```

**No Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

Após ativar o ambiente virtual, instale as dependências necessárias:
```bash
pip install numpy pandas
```

### Executando os Módulos

O projeto deve ser executado respeitando a seguinte ordem, pois cada módulo depende dos anteriores. Certifique-se de que o ambiente virtual está ativado antes de prosseguir:

### 1. Preparação de Dados
```bash
python data.py
```
Este script inicializa o vocabulário, cria embeddings iniciais e gera o tensor de entrada no formato esperado.

### 2. Componentes de Atenção
```bash
python attention.py
```
Este script valida os componentes de atenção isolados, incluindo softmax, atenção escalada e redes feed-forward.

### 3. Encoder
```bash
python encoder.py
```
Este script executa o stack de 6 camadas do Encoder, processando a entrada e gerando representações contextualizadas.

### 4. Decoder
```bash
python decoder.py
```
Este script executa o Decoder com máscara causal e cross-attention, demonstrando a geração auto-regressiva de sequências.

**Observação**: Se você não instalou as dependências via pip, certifique-se de que estão disponíveis no seu ambiente:
```bash
pip install numpy pandas
```

## Explicação do Fluxo do Encoder

### Visão Geral

O Encoder é responsável por processar a sequência de entrada e gerar representações contextualizadas. Implementa 6 camadas idênticas, cada uma composta por dois sub-blocos: auto-atenção e feed-forward.

### Fluxo Detalhado

1. **Entrada**: tensor de shape `(Batch, Seq_Len, d_model)`, onde `Batch` é o tamanho do lote, `Seq_Len` é o comprimento da sequência e `d_model` é a dimensionalidade do modelo (geralmente 512).

2. **Camada de Auto-atenção (Self-Attention)**:
   - Computa simultaneamente múltiplas "cabeças" de atenção em paralelo (multi-head attention).
   - Para cada cabeça:
     - Projeto linear: entrada → Query (Q), Key (K), Value (V)
     - Atenção escalada: `Attention(Q, K, V) = softmax(Q·K^T / √d_k)·V`
     - Cada cabeça mantém sua projeção dimensional reduzida.
   - Concatenação: todas as cabeças são concatenadas e projetadas novamente.

3. **Add & Norm (Conexão Residual + Normalização de Camada)**:
   - `Output = LayerNorm(Input + MultiHeadAttention(Input))`
   - A conexão residual facilita o fluxo de gradientes.
   - A normalização estabiliza o treinamento.

4. **Feed-Forward Network**:
   - Rede totalmente conectada com ativação (geralmente ReLU ou GELU):
   - `FFN(x) = ReLU(x·W_1 + b_1)·W_2 + b_2`
   - Tipicamente, a discussão sobre expansão de dimensionalidade ocorre aqui.

5. **Add & Norm (novamente)**:
   - `Output = LayerNorm(Input + FFN(Input))`

6. **Repetição**: Este processo é repetido 6 vezes, permitindo que cada camada refine as representações.

### Forma do Tensor

A forma do tensor **é preservada** ao longo do Encoder:
- **Entrada**: `(Batch, Seq_Len, d_model)`
- **Saída**: `(Batch, Seq_Len, d_model)`

O que muda é o **conteúdo** do tensor: valores são progressivamente atualizados para incorporar contexto.

### Exemplo de Saída Esperada

Para uma entrada com informações iniciais:
```
Entrada X:
Shape: (2, 4, 8)  # Batch=2, Seq_Len=4, d_model=8

Saída Encoder:
Shape: (2, 4, 8)  # Forma preservada
Valores: Atualizados com contexto de auto-atenção
Exemplo (simplificado):
  [[0.1234, -0.5678, ...],  # Token 1 contextualizado
   [0.9876, 0.4321, ...],   # Token 2 contextualizado
   ...]
```

## Explicação do Fluxo do Decoder

### Visão Geral

O Decoder é responsável por gerar sequências de forma auto-regressiva, utilizando informações do Encoder. Implementa 6 camadas idênticas, cada uma composta por três sub-blocos: auto-atenção com máscara causal, cross-attention, e feed-forward.

### Fluxo Detalhado

1. **Entrada**: tensor de shape `(Batch, Seq_Gen, d_model)`, onde `Seq_Gen` é a sequência gerada até o momento.

2. **Auto-atenção Causal (Masked Self-Attention)**:
   - Similar à auto-atenção do Encoder, mas com máscara causal aplicada.
   - **Máscara Causal**: Previne que uma posição $t$ tenha acesso a posições futuras $t'>t$.
   - Implementação:
     ```
     scores = Q·K^T / √d_k
     scores[t, t':t'>t] = -∞  # Máscara
     Attention = softmax(scores)·V
     ```
   - Resultado: Cada token só "vê" a si mesmo e aos tokens anteriores.

3. **Add & Norm**:
   - `Output = LayerNorm(Input + MaskedMultiHeadAttention(Input))`

4. **Cross-Attention**:
   - Conecta o Decoder com as representações do Encoder.
   - Query (Q): provém do Decoder
   - Key (K) e Value (V): provêm do Encoder
   - Permite que o Decoder acesse seletivamente informações codificadas.

5. **Add & Norm**:
   - `Output = LayerNorm(Input + CrossAttention(Decoder, Encoder))`

6. **Feed-Forward Network**:
   - Idêntico ao do Encoder.

7. **Add & Norm**:
   - `Output = LayerNorm(Input + FFN(Input))`

8. **Repetição**: Este processo é repetido 6 vezes.

### Loop de Geração Auto-regressiva

```
Inicializar: decoder_input = [<START>]
Repetir enquanto output ≠ <EOS>:
    encoder_output = Encoder(entrada_original)
    decoder_output = Decoder(decoder_input, encoder_output)
    proximo_token = argmax(linear_layer(decoder_output[-1]))
    decoder_input.append(proximo_token)
```

A geração termina quando o token `<EOS>` (End of Sequence) é produzido.

### Exemplo de Saída Esperada

```
Entrada (Encoder): "Hello world"
Saída Processada Encoder: Representações contextualizadas

Geração Decoder (auto-regressiva):
Iteração 1: Entrada=[<START>]     → Saída=Token_A
Iteração 2: Entrada=[<START>, A]  → Saída=Token_B
Iteração 3: Entrada=[<START>, A, B] → Saída=<EOS> (parar)

Sequência gerada: [<START>, Token_A, Token_B, <EOS>]
```

## Validação de Sanidade

Este projeto inclui verificações de sanidade para garantir o correto funcionamento dos componentes:

### Encoder

1. **Preservação de Forma**: O tensor mantém a forma `(Batch, Seq_Len, d_model)` após o processamento.
   - ✓ Entrada e saída têm as mesmas dimensões.
   - ✓ Nenhuma dimensão é reduzida ou expandida incorretamente.

2. **Contextualização de Valores**: Os valores numéricos são alterados progressivamente a cada camada.
   - ✓ A saída da camada 1 difere significativamente da entrada.
   - ✓ As diferenças diminuem progressivamente (devido à convergência).

### Decoder

1. **Máscara Causal**: Verifica se a máscara está funcionando corretamente.
   - ✓ A posição 0 só pode atender a si mesma.
   - ✓ A posição 2 pode atender às posições [0, 1, 2], mas não a [3, 4, ...].
   - ✓ A atenção a posições futuras é zero (ou muito proximamente de zero após softmax).

2. **Cross-Attention**: Confirma que o Decoder consegue acessar o Encoder.
   - ✓ Mudanças na entrada do Encoder produzem alterações nas saídas do Decoder.
   - ✓ A dimensionalidade da cross-attention é consistente.

3. **Geração Auto-regressiva**:
   - ✓ O loop de geração termina quando `<EOS>` é encontrado.
   - ✓ A sequência gerada tem comprimento crescente em cada iteração.
   - ✓ Nenhum token duplicado desnecessariamente (exceto por coincidência aleatória).

## Limitações do Projeto

Este projeto educacional possui várias limitações inerentes, importantes de reconhecer:

1. **Pesos Aleatórios**: Todos os pesos das redes neurais são inicializados aleatoriamente.
   - Consequência: O modelo não aprendeu padrões significativos.
   - Nenhuma estrutura útil foi capturada dos dados.

2. **Ausência de Treinamento**: O modelo não passa por um processo de otimização.
   - Os pesos permanecem aleatórios do início ao fim.
   - Nenhuma função de perda é minimizada.

3. **Sem Backpropagation**: Não há computação de gradientes ou atualização de parâmetros.
   - O modelo não melhora a partir dos dados.
   - É puramente demonstrativo.

4. **Vocabulário Simplificado e Fictício**:
   - Utiliza um vocabulário pré-definido pequeno.
   - As palavras são fictícias ou genéricas (ex: "token_0", "token_1", ...).
   - Não representa dados reais de linguagem natural.

5. **Geração Apenas Demonstrativa**:
   - A geração de sequências é pseudo-aleatória (cada token é escolhido com probabilidades aleatórias).
   - Não produz saídas semanticamente coerentes.
   - Serve apenas para ilustrar o mecanismo, não para uso prático.

6. **Ausência de Otimizações Computacionais**:
   - Implementação direta em NumPy, sem otimizações de GPU ou batching avançado.
   - Adequada para fins pedagógicos, inadequada para escalabilidade.

7. **Foco Restrito em Funcionalidade**:
   - Omite alguns componentes avançados (ex: positional encoding complexos, dropout, label smoothing).
   - Visa clareza conceitual, não completude prática.

## Uso de Inteligência Artificial

Durante o desenvolvimento deste projeto, foram utilizadas ferramentas de inteligência artificial generativa como apoio técnico nos seguintes aspectos:

- **Curadoria do Código**: Assistência na estruturação e organização dos módulos Python, visando clareza e modularidade.
- **Revisão de Sintaxe**: Verificação de compatibilidade Python, imports coretos e convenções de codificação.
- **Correções e Adequações**: Refinamento do código para atender aos requisitos acadêmicos e aos objetivos educacionais do projeto.
- **Esclarecimento Conceitual**: Consulta para validação de conceitos teóricos, fórmulas matemáticas e descrições técnicas.

---

**Última atualização**: 08/03/2026  
**Versão**: 1.0  
**Licença**: Educacional (Creative Commons Attribution-NonCommercial-ShareAlike 4.0)
