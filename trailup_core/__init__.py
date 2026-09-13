"""trailup_core - medidas locais do estado do aluno.

Cada modulo responde a UMA pergunta, traz no cabecalho os numeros que o
sustentam, e nao depende de nada alem da biblioteca padrao.

CONVENCAO DE SINAL, E ELA JA CAUSOU DOIS DEFEITOS
=================================================
Em todo o pacote, a dificuldade de uma questao e representada pela sua TAXA
DE ACERTO - ou seja, por FACILIDADE. Valor alto = questao facil.

    dificuldade.estimar(...).taxa      0,9 = 90% acertam = FACIL
    dominio(acerto_na_questao=0.9)     questao facil
    gate.risco(acerto_medio_topico=)   topico onde se acerta muito

Passar "dificuldade" no sentido comum (alto = dificil) inverte o resultado em
silencio. Dois parametros ja se chamaram `dificuldade_*` recebendo facilidade,
e os dois foram renomeados em 2026-09-13 depois de a auditoria medir que
passar 0,9 como "muito dificil" produzia o resultado de "muito facil".

Se voce tem dificuldade no sentido comum, passe `1 - dificuldade`.
"""
__version__ = '0.1.0'
