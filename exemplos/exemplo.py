"""Os modulos trabalhando juntos sobre uma questao e um aluno."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from trailup_core.dificuldade import estimar, Prior, afirmar
from trailup_core import ritmo, tempo, chute, dominio, revisao, engajamento

print('=' * 78)
print('QUESTAO 42 — 30 alunos da turma responderam, 11 acertaram, mediana de 52s')
print('=' * 78)
f = estimar(acertos=11, respostas=30, prior=Prior())
r = ritmo.ritmo(latencia_mediana=52.0, respostas=30)
t = tempo.esperado(latencia_mediana=52.0, respostas=30)
print(f'  dificuldade : {f}')
print(f'  ritmo       : {r}')
print(f'  tempo       : {t}')
print(f'  afirmacao   : {"dificil" if afirmar(f, .50, "abaixo") else ("facil" if afirmar(f, .80, "acima") else "so o intervalo, sem cravar")}')

print()
print('=' * 78)
print('ALUNO — 8 respostas neste topico, 3 certas; responde a questao 42 em 14s e erra')
print('=' * 78)
d = dominio.dominio(dificuldade_questao=f.taxa, acertos_no_topico=3, respostas_no_topico=8)
print(f'  dominio      : {d}')
print(f'  reforco?     : {dominio.precisa_reforco(d)}  (exige confianca >= 70%)')
foi = chute.foi_chute(segundos=14, acertou=False, latencia_mediana_questao=52.0, respostas_questao=30)
print(f'  foi chute?   : {foi}  (14s contra limiar de {0.30*52:.0f}s)')
print(f'  demorando?   : {tempo.demorando(14, t)}')
print(f'  -> leitura   : {"nao tentou - cobrar tentativa, nao explicar de novo" if foi else "tentou e nao conseguiu - reforcar conteudo"}')

print()
print('=' * 78)
print('REVISAO — quando trazer esta questao de volta')
print('=' * 78)
for errou in (True, False):
    p = revisao.dias_ate_revisar(acertou_antes=not errou)
    lab = 'errou' if errou else 'acertou'
    d7 = revisao.retencao(7, acertou_antes=not errou)
    d60 = revisao.retencao(60, acertou_antes=not errou)
    prazo = f'{p:.1f} dias' if p != float('inf') else 'nao cai a 75% no horizonte medido'
    print(f'  quem {lab:>7}: retencao em 7d {d7:.0%}, em 60d {d60:.0%} | revisar em {prazo}')
