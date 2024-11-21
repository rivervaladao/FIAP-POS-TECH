## O Problema

O cenário apresentado é uma demanda real do software de Planejamento de Manutenção, da empresa para qual trabalho. O sofware precisa implementar uma solução inteligente que realize uma sugestão de planejamento e programação de manutenção em forma de uma agenda de trabalho. Os dados são importados do Módulo de Planejamento do SAP.

---

### Justificativa

O problema de **planejamento de manutenção** envolve a alocação de tarefas complexas e interdependentes a recursos limitados (como colaboradores e equipamentos) com o objetivo de otimizar a execução das operações dentro de uma ordem de serviço. Essas operações precisam ser atribuídas considerando diversas restrições, como a disponibilidade dos trabalhadores, suas qualificações, tempo de execução, prioridades das ordens e possíveis sobreposições de horários. 

Este problema se enquadra na classe dos **problemas NP-difíceis (NP-hard)**, pois a quantidade de combinações possíveis para alocar trabalhadores e tarefas cresce exponencialmente à medida que aumentam o número de operações, colaboradores e restrições. Verificar todas as combinações possíveis de forma exaustiva para encontrar a solução ideal dentro de um tempo polinomial se torna impraticável, já que o tempo necessário para resolver o problema aumenta exponencialmente conforme o tamanho do problema cresce. Essa característica define os problemas NP-difíceis, nos quais não há algoritmos conhecidos que possam resolver todos os casos em tempo polinomial.

Diante da natureza do problema, vamos explorar a utilização de **algoritmos genéticos** como uma abordagem adequada, que oferece uma solução robusta para o problema de planejamento de manutenção, permitindo otimizar a alocação de tarefas complexas e restritivas de forma eficiente.

---

### Objetivo

Implementar uma solução usando algoritmos genéticos que realiza uma sugestão de **planejamento e programação de manutenção** em forma de uma agenda de trabalho.

A solução deverá alocar o melhor colaborador para a atividade, considerando: 

- Qual habilidade é obrigatória para a realização da operação da ordem com base em histórico.
- Quem do centro de trabalho (equipe) listado na operação da ordem possui matriz de habilidade compatível com a tarefa caso seja exigido.
- Na data de execução prevista na ordem, qual colaborador apto, pertencente ao centro de trabalho, tem disponibilidade.
- Dos que têm disponibilidade dentro do centro de trabalho, quem tem mais atuações no ativo a ser inspecionado.
- Quais materiais geralmente são utilizados em tarefas semelhantes para que o algoritmo sugira ao planejador.
- Quais permissões de trabalho geralmente são atribuídas em manutenções no ativo em questão, ou no desempenho da função que a operação exige.

A solução deve ajustar dinamicamente o planejamento conforme as tarefas do dia a dia surgem, sugerindo reprogramações das atividades caso seja necessário.

---

### Regras Coletadas

A solução deverá alocar o melhor colaborador para aquela atividade, considerando: 
1. O agendamento das ordens deve ser priorizado pelo `indice_irpe`.
2. Cada agendamento deve alocar a quantidade de colaboradores definida na `quantidade_executantes`.
3. O colaborador deve ser selecionado pela qualificação compatível especificada na ordem de manutenção.
4. Se não houver qualificação definida no campo `qualificacao` da ordem, deve ser alocado o colaborador com maior quantidade de execuções sobre o equipamento, com base na contagem da tabela de `historico_manutencao.csv`.
5. O colaborador pode estar em mais de uma ordem para uma ou mais operações, desde que `hora_total - soma (tempo já alocado em outras ordens) - soma (esforco_individual da operacao)` seja menor ou igual a zero.
6. Cada colaborador deve ser alocado respeitando a ordem crescente definida no campo `operacao`.
7. Os mesmos colaboradores podem ser alocados em mais de uma operação dentro da mesma ordem.

Como resultado, devem ser apresentados:
- `ordem`, `operacao`, [matriculas dos operadores alocados], `data de inicio`, `horario de inicio`, `horario de término`.

### Descrição dos dados de entrada (arquivos CSV)

A solução deve receber como entrada os arquivos `ordens_manutencao.csv`, `disponibilidade.csv`, `historico_manutencao.csv`, estruturados conforme descrição:

1. Arquivo **ordens_manutencao.csv** contém os dados para execução de uma ordem de manutenção, conforme descrição:
   - `ordem`: ID único da ordem.
   - `operacao`: Código da operação que será executada pela equipe definida na coluna `centro_trabalho`; cada operação é executada por um ou mais colaboradores que pertencem à equipe de trabalho, o número de colaboradores por operação é definido na coluna `quantidade_executantes`.
   - `centro_trabalho_id`: ID único da equipe responsável pela execução da operação de manutenção definida na coluna `operacao`.
   - `centro_trabalho`: Título do `centro_trabalho_id`.
   - `quantidade_executantes`: Número de colaboradores necessários para realizar a operação.
   - `esforco_individual`: Tempo estimado para execução da operação dentro da ordem de serviço.
   - `unidade`: Unidade de tempo utilizada na coluna `esforco_individual`, ex. H = horas.
   - `equipamento_ordem`: ID único do equipamento que irá receber a manutenção.
   - `qualificacao`: Certificações ou habilidades que o colaborador deve possuir para ser alocado na execução da operação da ordem de serviço. Este campo se relaciona ao campo `qualificacao` em `disponibilidade.csv`; a listagem está separada por "/".
   - `indice_irpe`: Índice de prioridade de execução da ordem de serviço; quanto maior o valor, maior a prioridade de execução da ordem de serviço. Isso envolve alocar o máximo de horas dos colaboradores disponíveis para executar a ordem no dia e horário planejado. Não havendo disponibilidade de horas de colaboradores habilitados, a ordem deve ser postergada para outra data.
   - `data_inicio_base`: Data de início prevista para a execução da ordem.
   - `hora_inicio_base`: Horário previsto para execução da ordem.
   - `data_fim_base`: Data prevista para finalização da execução da ordem.
   - `hora_fim_base`: Horário previsto para finalização da execução da ordem.

2. Arquivo **disponibilidade.csv**, contém os dados para seleção do colaborador habilitado e com maior proficiência na manutenção de um equipamento, conforme descrição:
   - `centro_trabalho`: Representa a equipe de trabalho à qual o colaborador pertence.
   - `matricula`: ID único do colaborador.
   - `nome`: Nome do colaborador.
   - `grau_utilizacao_ct`: Percentual de horas úteis do total de horas disponível para trabalho.
   - `inicio_vigencia`: Horário de início das atividades diárias no turno.
   - `hora_inicio_trabalho`: Horário de início do turno semanal, se preenchido, indica um turno diferente do normal.
   - `horas_trabalho`: Horas de trabalho no turno.
   - `fim_vigencia`: Horário de fim das atividades diárias no turno.
   - `horas_folga`: Quantidade de horas de folga que devem ser deduzidas do tempo total disponível para alocação.
   - `dia`: Dia da semana em que o colaborador trabalha (1-segunda-feira, 2-terça-feira, etc).
   - `hora_inicio`: Horário diário normal de início de trabalho do colaborador no turno normal.
   - `hora_inicio_intervalo`: Horário de início do intervalo.
   - `hora_fim_intervalo`: Horário de fim do intervalo.
   - `hora_fim`: Horário de fim das atividades diárias no turno normal.
   - `hora_total`: Quantidade de horas efetivamente disponível.
   - `qualificacao`: Certificações que o colaborador possui e que o habilitam a realizar a ordem de serviço segundo a restrição de habilidades necessárias. Os valores estão separados por '/', ignore tudo a partir do hífen.

3. Arquivo **historico_manutencao.csv**, contém os dados históricos de manutenção de um equipamento executado pelo colaborador numa data, conforme descrição:
   - `matricula`: ID único do colaborador.
   - `data_inicio_execucao`: Data de início da execução da manutenção realizada pelo colaborador sobre o equipamento.
   - `equipamento`: ID único do equipamento que recebeu manutenção do colaborador.
   - `trabalho_real`: Tempo realmente gasto na execução da ordem de manutenção.

---   

## Solução para o Problema de Planejamento de Manutenção

A solução proposta utiliza algoritmos genéticos para otimizar o problema de planejamento de manutenção, onde as operações de manutenção precisam ser atribuídas a colaboradores com base em suas qualificações, disponibilidade, e tempo de execução. O objetivo é otimizar essa alocação de forma eficiente, considerando as múltiplas restrições do problema. Aqui está um resumo dos principais aspectos da solução:

---

### Seleção do Cromossomo

O **cromossomo** representa uma solução individual no algoritmo genético. Cada cromossomo é um **indivíduo** que corresponde a uma lista de ordens de serviço, onde cada ordem contém operações e seus respectivos colaboradores alocados. A classe que implementa a estrutura do cromossomo e a modelagem das tarefas e operações é a **classe `MaintenanceTask`**, com suas operações modeladas pela **classe `OperationTask`**.

A estrutura do cromossomo foi escolhida para permitir que cada tarefa seja representada por uma sequência de operações, facilitando a aplicação dos operadores genéticos, como **crossover** e **mutação**, e permitindo uma fácil avaliação pela função de fitness.

- **Classe `MaintenanceTask`**: Representa a ordem de serviço como um conjunto de operações. Cada operação é alocada a trabalhadores, respeitando suas qualificações e disponibilidade.
- **Classe `OperationTask`**: Modela as operações individuais, incluindo a hora de início, a duração (esforço), e os trabalhadores alocados.

---

### Escolha da Função de Fitness

A **função de fitness** foi projetada para refletir a qualidade da alocação de tarefas em termos de:
1. **Qualificação dos trabalhadores**: Soluções em que os trabalhadores alocados possuem as qualificações adequadas recebem uma pontuação mais alta.
2. **Priorização de ordens de serviço**: Ordens de maior prioridade recebem uma pontuação maior.
3. **Minimização do tempo de execução**: O tempo total gasto para concluir as operações é penalizado, incentivando soluções mais rápidas.
4. **Uso eficiente da disponibilidade**: Soluções que alocam colaboradores dentro de suas janelas de disponibilidade são recompensadas.

Essa função foi escolhida para garantir que a solução final não apenas atenda às restrições, mas também otimize o uso dos recursos de maneira eficiente.

---

### Escolha do Crossover

O **crossover** selecionado utiliza uma abordagem de ponto duplo, onde duas partes dos genes dos pais são combinadas para formar um novo filho. A implementação da função de crossover está na **classe `GeneticAlgorithm`**, e usa dois pontos de corte aleatórios para dividir os cromossomos e gerar novas combinações de soluções:

- **Classe `GeneticAlgorithm`**: A função `crossover` é responsável por pegar pedaços de soluções dos pais e combiná-las para formar novos indivíduos, promovendo diversidade genética.
- **Crossover por ponto duplo**: Essa escolha foi feita para garantir que os filhos herdem partes diversificadas de ambos os pais, aumentando a chance de encontrar soluções novas e promissoras.

---

### Melhorias Sugeridas

Embora o algoritmo genético utilizado consiga explorar o espaço de soluções, algumas melhorias podem ser feitas para melhorar a performance e a qualidade das soluções encontradas:

1. **Introdução de mutação mais inteligente**: Em vez de realizar mutações aleatórias, a mutação poderia ser mais informada, focando em áreas problemáticas do cromossomo, como a realocação de trabalhadores subutilizados ou ajuste de horários sobrepostos.
   
2. **Aprimoramento da função de fitness**: A função de fitness pode ser ajustada para considerar ainda mais fatores, como a **experiência do trabalhador com o equipamento** ou **custos associados** à alocação de recursos, aumentando a precisão da avaliação da solução.
   
3. **Hiperparâmetros do algoritmo**: A taxa de mutação e o tamanho da população poderiam ser ajustados dinamicamente ao longo da execução, permitindo uma melhor adaptação ao longo do tempo.
   
4. **Implementação de seleção elitista**: Para garantir que as melhores soluções sempre avancem para a próxima geração, a introdução de uma estratégia elitista que mantém os melhores indivíduos de cada geração pode ser benéfica.

---

### Justificativa: Por que o modelo atual não apresenta alteração significativa na função de fitness?

A falta de melhorias significativas na função de fitness pode estar relacionada a alguns fatores:

1. **Exploração limitada do espaço de soluções**: O crossover utilizado pode não estar gerando filhos suficientemente diferentes dos pais, o que limita a exploração de novas soluções.
   
2. **Mutação limitada**: A taxa de mutação pode estar muito baixa, ou a mutação pode não estar introduzindo variações significativas, resultando em soluções que rapidamente convergem para ótimos locais.
   
3. **Diversidade genética insuficiente**: Se a população inicial não for suficientemente diversa, o algoritmo pode convergir rapidamente para soluções subótimas, sem explorar regiões mais promissoras do espaço de busca.
   
4. **Função de fitness com critérios limitados**: A função de fitness pode não estar refletindo adequadamente todos os fatores que influenciam a qualidade da solução. Adicionar mais critérios ou ajustar os pesos dos critérios existentes pode tornar a função mais sensível a pequenas mudanças nas soluções, resultando em uma evolução mais eficaz.

---

### Conclusão

A implementação atual do algoritmo genético apresenta uma abordagem funcional para resolver o problema de planejamento de manutenção, mas melhorias podem ser feitas na função de fitness, nos operadores genéticos, e na diversificação da população para obter resultados mais eficientes e diferenciados.