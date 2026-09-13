"""Testes de invariante. Nao testam 'o numero bate' - testam que o modulo
nao afirma mais do que mediu, que nao aceita entrada impossivel, e que as
propriedades que sustentam o desenho continuam valendo."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pytest
from trailup_core import dificuldade, ritmo, tempo, dominio, chute, revisao
from trailup_core import discriminacao, engajamento, pre_avaliacao


# ---------------- dificuldade ----------------
def test_dificuldade_sem_dado_devolve_o_prior():
    f = dificuldade.estimar(0, 0)
    p = dificuldade.Prior()
    assert abs(f.taxa - p.media) < 1e-9
    assert f.largura > 0.4, 'questao sem resposta tem de vir com incerteza grande'

def test_dificuldade_intervalo_encolhe_com_n():
    larguras = [dificuldade.estimar(int(0.7 * n), n).largura for n in (5, 30, 100, 400)]
    assert larguras == sorted(larguras, reverse=True)

def test_dificuldade_rejeita_entrada_impossivel():
    with pytest.raises(ValueError):
        dificuldade.estimar(10, 5)
    with pytest.raises(ValueError):
        dificuldade.estimar(3, 10, nivel=0.77)

def test_afirmar_so_quando_o_intervalo_inteiro_sustenta():
    f = dificuldade.estimar(3, 10)           # 30% em 10 respostas: incerto
    assert not dificuldade.afirmar(f, 0.50, 'abaixo')
    g = dificuldade.estimar(10, 100)         # 10% em 100: claro
    assert dificuldade.afirmar(g, 0.50, 'abaixo')

def test_prior_derivado_do_corpus_bate_o_esperado():
    taxas = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.55, 0.65, 0.75, 0.85]
    p = dificuldade.derivar_prior(taxas, [200] * len(taxas))
    assert 2 < p.forca < 40, 'prior fora de faixa plausivel'
    assert abs(p.media - sum(taxas) / len(taxas)) < 0.02

def test_correcao_por_alunos_alarga_o_intervalo():
    sem = dificuldade.estimar(70, 100)
    com = dificuldade.estimar(70, 100, alunos=60)
    assert com.largura > sem.largura, 'reencontro reduz evidencia efetiva'


# ---------------- engajamento ----------------
def test_engajamento_curva_de_referencia_e_monotonica():
    for base, tabela in engajamento.REFERENCIA.items():
        vals = [tabela[k] for k in sorted(tabela)]
        assert vals == sorted(vals), f'{base} nao e monotonica'

def test_engajamento_sem_calibracao_so_entrega_recencia():
    e = engajamento.medir(dias_recentes=5, dias_ativos=12, eventos=200)
    assert e.frequencia is None and e.profundidade is None
    assert e.retencao_esperada is None, 'sem calibracao nao se afirma probabilidade'

def test_engajamento_poucos_eventos_nao_gera_alerta():
    e = engajamento.medir(dias_recentes=0, dias_ativos=1, eventos=5)
    assert not e.confiavel and e.alertas() == []

def test_calibrar_exige_amostra():
    coorte = [{'dias_recentes': i % 11, 'dias_ativos': 10, 'voltou': i % 2}
              for i in range(50)]
    c = engajamento.calibrar(coorte)
    assert not c.confiavel, 'coorte pequena nao pode ser marcada confiavel'

def test_calibrar_produz_faixas_com_amostra_suficiente():
    coorte = [{'dias_recentes': i % 11, 'dias_ativos': (i % 30),
               'profundidade_seg': float(i % 60), 'voltou': 1 if i % 11 > 4 else 0}
              for i in range(900)]
    c = engajamento.calibrar(coorte)
    assert c.confiavel and 'frequencia' in c.cortes and 'profundidade' in c.cortes
    lo, hi = c.cortes['frequencia']
    assert lo <= hi

def test_sessoes_conta_pelo_gap():
    assert engajamento.sessoes([]) == 0
    assert engajamento.sessoes([0, 60, 120]) == 1
    assert engajamento.sessoes([0, 60, 5000]) == 2


# ---------------- ritmo / tempo ----------------
def test_ritmo_separa_pelos_40_segundos():
    assert ritmo.ritmo(15.0, respostas=5).valor == 'rapida'
    assert ritmo.ritmo(75.0, respostas=5).valor == 'lenta'
    assert ritmo.FRONTEIRA_SEG == 40.0

def test_tempo_nao_ajusta_por_aluno():
    """O achado que define o modulo: ajustar pelo ritmo do aluno PIORA."""
    import inspect
    fonte = inspect.getsource(tempo)
    assert 'ritmo_do_aluno' not in fonte.replace('#', '').split('"""')[-1], \
        'o corpo do modulo nao pode ajustar por aluno'


# ---------------- dominio ----------------
def test_dominio_pesa_mais_a_questao():
    assert dominio.PESO_QUESTAO >= 0.5, 'a questao explica mais que o aluno'

def test_confianca_cresce_e_satura():
    c = [dominio.confianca(n, int(n * 0.7)) for n in (1, 5, 20, 100, 1000)]
    assert c == sorted(c)
    assert c[-1] <= 0.95, 'o teto honesto é 0,95 — a incerteza nunca zera'
    assert c[-1] < 1.0

def test_incerteza_encolhe_com_n():
    v = [dominio.incerteza(n, int(n * 0.7)) for n in (2, 10, 50, 500)]
    assert v == sorted(v, reverse=True)

def test_incerteza_nunca_zera_no_extremo():
    """Com n=2 e taxa 0 ou 1, o desvio binomial daria zero. O posterior não."""
    assert dominio.incerteza(2, 0) > 0.05
    assert dominio.incerteza(2, 2) > 0.05


# ---------------- chute ----------------
def test_chute_exige_corpus_na_questao():
    assert chute.MIN_RESPOSTAS_QUESTAO >= 30


# ---------------- revisao ----------------
def test_revisao_quem_errou_volta_antes():
    assert revisao.dias_ate_revisar(acertou_antes=False) < revisao.dias_ate_revisar(acertou_antes=True)


# ---------------- pre_avaliacao ----------------
def test_stopwords_exige_corpus():
    """O bug que quase passou: top-N de 2 textos curtos zera o grafo."""
    assert pre_avaliacao.stopwords(['um texto curto', 'outro texto']) == set()

def test_pre_avaliacao_ordena_o_obvio():
    gab = ('Inflacao e o aumento generalizado e continuo dos precos numa economia. '
           'O banco central controla a inflacao elevando os juros.')
    ref = pre_avaliacao.preparar(gab)
    completa = pre_avaliacao.avaliar(
        'Inflacao e o aumento generalizado dos precos numa economia e o banco '
        'central controla elevando os juros', ref)
    vazia = pre_avaliacao.avaliar('nao sei responder isso agora', ref)
    assert completa.nota > vazia.nota
    assert completa.cobertura > vazia.cobertura
    assert vazia.divagacao > completa.divagacao


# ---------------- engajamento: ordenador comum ----------------
def _coorte(n=900):
    import random
    r = random.Random(11)
    out = []
    for _ in range(n):
        dr = r.choice([0] * 5 + list(range(1, 11)))
        da = min(30, dr * 2 + r.randint(0, 8))
        out.append({'dias_recentes': dr, 'dias_ativos': da,
                    'profundidade_seg': r.random() * 40,
                    'voltou': 1 if r.random() < 0.1 + 0.08 * dr else 0})
    return out

def test_ordenar_exige_calibracao():
    """Sem media/desvio da coorte o peso comum nao significa nada."""
    vazia = engajamento.Calibracao()
    with pytest.raises(ValueError):
        engajamento.ordenar(3, 10, vazia)

def test_ordenar_e_monotonico_na_recencia():
    cal = engajamento.calibrar(_coorte())
    scores = [engajamento.ordenar(k, 15, cal) for k in range(11)]
    assert scores == sorted(scores)

def test_risco_dispara_perto_da_taxa_pedida():
    cal = engajamento.calibrar(_coorte())
    coorte = _coorte()
    for taxa in (0.10, 0.20, 0.30):
        disp = sum(engajamento.risco(c['dias_recentes'], c['dias_ativos'], cal, taxa)
                   for c in coorte) / len(coorte)
        assert abs(disp - taxa) < 0.08, f'{taxa:.0%} disparou {disp:.0%}'

def test_risco_rejeita_taxa_nao_calibrada():
    cal = engajamento.calibrar(_coorte())
    with pytest.raises(ValueError):
        engajamento.risco(0, 0, cal, taxa_alerta=0.42)

def test_pesos_comuns_priorizam_recencia():
    """O achado das duas bases: recencia pesa ~6x mais que frequencia."""
    assert engajamento.PESOS['recencia'] > 5 * engajamento.PESOS['frequencia']

def test_z_neutraliza_escala_entre_coortes():
    """O passo que impede o modelo de aprender 'de qual plataforma veio'."""
    baixa = [{'dias_recentes': k % 6, 'dias_ativos': k % 12, 'voltou': k % 2}
             for k in range(600)]
    alta = [{'dias_recentes': 5 + k % 6, 'dias_ativos': 18 + k % 12, 'voltou': k % 2}
            for k in range(600)]
    ca, cb = engajamento.calibrar(baixa), engajamento.calibrar(alta)
    # o aluno mediano de cada coorte tem z ~ 0 nas duas, apesar das escalas
    assert abs(ca.z('recencia', ca.media['recencia'])) < 1e-9
    assert abs(cb.z('recencia', cb.media['recencia'])) < 1e-9
    assert cb.media['recencia'] > ca.media['recencia'] + 4


# ---------------- engajamento: trajetoria ----------------
def test_trajetoria_rotula_a_forma():
    assert engajamento.trajetoria([8, 4, 1]) == 'caiu'
    assert engajamento.trajetoria([2, 2, 2]) == 'estavel'
    assert engajamento.trajetoria([1, 3, 7]) == 'subiu'

def test_trajetoria_limiar_afrouxavel():
    """[5,5,4] e 'caiu' no limiar medido; com limiar folgado vira 'estavel'."""
    assert engajamento.trajetoria([5, 5, 4]) == 'caiu'
    assert engajamento.trajetoria([5, 5, 4], limiar=1.0) == 'estavel'

def test_trajetoria_exige_duas_janelas():
    with pytest.raises(ValueError):
        engajamento.trajetoria([3])

def test_trajetoria_nao_entra_no_score():
    """A trajetoria e descricao. ordenar() nao pode aceita-la como feature."""
    import inspect
    assert 'trajetoria' not in inspect.getsource(engajamento.ordenar)


# ---------------- melhorias medidas ----------------
def test_prever_turma_e_mais_largo_que_estimar():
    """A pergunta da turma é mais incerta que a da questão. Tem de aparecer."""
    f = dificuldade.estimar(20, 30)
    t = dificuldade.prever_turma(20, 30, 30)
    assert t.largura > f.largura * 1.3
    assert abs(t.taxa - f.taxa) < 1e-9, 'o centro é o mesmo; só a incerteza muda'

def test_prever_turma_estreita_com_turma_maior():
    l = [dificuldade.prever_turma(20, 30, m).largura for m in (15, 30, 60, 120)]
    assert l == sorted(l, reverse=True)

def test_prever_turma_nunca_fica_abaixo_do_posterior():
    for n in (10, 50, 200, 1000):
        f = dificuldade.estimar(int(0.7 * n), n)
        t = dificuldade.prever_turma(int(0.7 * n), n, 30)
        assert t.largura >= f.largura

def test_prever_turma_rejeita_turma_vazia():
    with pytest.raises(ValueError):
        dificuldade.prever_turma(20, 30, 0)

def test_esperado_de_amostra_usa_media_do_log():
    """Média do log fica entre a mediana e a média aritmética."""
    v = [10, 12, 15, 60, 200, 11, 13]
    e = tempo.esperado_de_amostra(v)
    assert sorted(v)[len(v) // 2] < e.segundos < sum(v) / len(v)

def test_esperado_de_amostra_rejeita_amostra_vazia():
    with pytest.raises(ValueError):
        tempo.esperado_de_amostra([0, -1, None])


# ---------------- dominio: terceiro termo ----------------
def test_dominio_dois_pesos_distintos():
    """Sem histórico global o ótimo é 0,70; com ele é 0,60. Não reutilizar."""
    assert dominio.PESO_QUESTAO_2 == 0.70
    assert dominio.PESO_QUESTAO + dominio.PESO_TOPICO + dominio.PESO_GLOBAL == pytest.approx(1.0)

def test_dominio_sem_global_usa_a_formula_de_dois_termos():
    d = dominio.dominio(0.45, 3, 8, recalibrar=False)
    esperado = dominio.PESO_QUESTAO_2 * 0.45 + (1 - dominio.PESO_QUESTAO_2) * (
        (3 + 0.67 * dominio.PRIOR_ALUNO) / (8 + dominio.PRIOR_ALUNO))
    assert d.p == pytest.approx(round(esperado, 3))

def test_recalibracao_expande_em_torno_da_taxa_base():
    """A média linear comprime; a recalibração desfaz — mas o ponto fixo é a
    taxa base (~0,65), não 0,5. O corpus acerta 67%, não metade."""
    fixo = 0.646
    for p in (0.25, 0.40, 0.80, 0.90):
        rec = dominio._recalibrar(p)
        assert abs(rec - fixo) > abs(p - fixo), f'p={p} não foi afastado da base'

def test_recalibracao_tem_ponto_fixo_na_taxa_base():
    assert dominio._recalibrar(0.646) == pytest.approx(0.646, abs=0.01)

def test_recalibracao_preserva_a_ordem():
    crus = [dominio.dominio(q, 5, 10, recalibrar=False).p for q in (0.2, 0.4, 0.6, 0.8)]
    recs = [dominio.dominio(q, 5, 10, recalibrar=True).p for q in (0.2, 0.4, 0.6, 0.8)]
    assert crus == sorted(crus) and recs == sorted(recs)

def test_dominio_global_puxa_na_direcao_certa():
    fraco = dominio.dominio(0.45, 3, 8, acertos_totais=40, respostas_totais=150)
    forte = dominio.dominio(0.45, 3, 8, acertos_totais=130, respostas_totais=150)
    assert forte.p > fraco.p

def test_dominio_rejeita_global_impossivel():
    with pytest.raises(ValueError):
        dominio.dominio(0.45, 3, 8, acertos_totais=200, respostas_totais=150)


# ---------------- discriminacao: confirmacao por metades ----------------
def test_confirmar_exige_as_duas_metades():
    assert discriminacao.confirmar(-0.10, -0.05) is True
    assert discriminacao.confirmar(-0.30, +0.40) is False, 'uma metade só não basta'
    assert discriminacao.confirmar(+0.20, +0.10) is False

def test_confirmar_limiar_mais_exigente():
    assert discriminacao.confirmar(-0.04, -0.04) is True
    assert discriminacao.confirmar(-0.04, -0.04, limiar=-0.05) is False


# ---------------- pre_avaliacao: conjunto de 10 features ----------------
def test_pre_avaliacao_usa_dez_features():
    assert len(pre_avaliacao._PESOS) == 10
    assert 'n_nos' in pre_avaliacao._PESOS and 'n_arestas' in pre_avaliacao._PESOS

def test_triar_poe_a_mais_fraca_primeiro():
    gab = ('Inflacao e o aumento generalizado e continuo dos precos numa economia. '
           'O banco central controla a inflacao elevando os juros.')
    ref = pre_avaliacao.preparar(gab)
    respostas = ['nao sei',
                 'Inflacao e o aumento generalizado dos precos numa economia e o '
                 'banco central controla elevando os juros']
    ordem = [i for i, _ in pre_avaliacao.triar(respostas, ref)]
    assert ordem[0] == 0, 'a resposta vazia tem de vir primeiro'


# ---------------- revisao: interpolacao pela proporcao ----------------
def test_retencao_proporcao_interpola_entre_as_duas_curvas():
    for dias in (1, 7, 60):
        baixo = revisao.retencao(dias, acertou_antes=False)
        alto = revisao.retencao(dias, acertou_antes=True)
        meio = revisao.retencao(dias, acertou_antes=True, proporcao_acertos=0.5)
        assert baixo < meio < alto

def test_retencao_proporcao_nos_extremos_bate_as_tabelas():
    for dias in (0.5, 3, 21, 200):
        assert revisao.retencao(dias, True, 1.0) == pytest.approx(revisao.retencao(dias, True))
        assert revisao.retencao(dias, False, 0.0) == pytest.approx(revisao.retencao(dias, False))

def test_retencao_rejeita_proporcao_invalida():
    with pytest.raises(ValueError):
        revisao.retencao(7, True, proporcao_acertos=1.5)

def test_retencao_cai_com_o_tempo_para_quem_errou():
    v = [revisao.retencao(d, acertou_antes=False) for d in (1, 3, 7, 21, 60)]
    assert v == sorted(v, reverse=True)


# ---------------- gate ----------------
def test_gate_saturacao_preservada():
    """A saturação é defeito na estimativa e informação no extremo. Manter."""
    from trailup_core import gate
    assert gate.sequencial([False] * 50) == gate.PISO
    assert gate.sequencial([True] * 50) == gate.TETO

def test_gate_risco_sobe_com_erro():
    from trailup_core import gate
    bom = gate.risco([True] * 10, acerto_global=0.8, dificuldade_media_topico=0.7)
    ruim = gate.risco([False] * 10, acerto_global=0.3, dificuldade_media_topico=0.5)
    assert ruim.score > bom.score

def test_gate_nao_opina_com_pouca_evidencia():
    from trailup_core import gate
    r = gate.risco([False, False], acerto_global=0.3, dificuldade_media_topico=0.5)
    assert not r.confiavel
    assert gate.deve_disparar(r, limiar_da_coorte=0.0) is False

def test_gate_pesos_somam_um():
    from trailup_core import gate
    assert gate.PESO_ACUMULADO + gate.PESO_SEQUENCIAL == pytest.approx(1.0)
    assert gate.PESO_TOPICO + gate.PESO_GLOBAL + gate.PESO_DIF == pytest.approx(1.0)


# ---------------- ritmo: a confianca nao pode afirmar certeza ----------------
def test_ritmo_confianca_nunca_e_certeza():
    for n in (5, 10, 21, 50, 500, 100000):
        assert ritmo.ritmo(15.0, n).confianca < 1.0, f'certeza afirmada em n={n}'

def test_ritmo_confianca_cresce_com_n():
    v = [ritmo.ritmo(15.0, n).confianca for n in (5, 10, 21, 50)]
    assert v == sorted(v)
    assert v[0] == pytest.approx(0.973)

def test_ritmo_recusa_amostra_pequena():
    with pytest.raises(ValueError):
        ritmo.ritmo(15.0, 4)


def test_ritmo_nunca_exibe_100_por_cento():
    """0,997 arredondado com .0% vira "100%" - a falsa certeza pela porta dos fundos."""
    for n in (5, 50, 100000):
        assert '100%' not in str(ritmo.ritmo(15.0, n))


def test_calibrar_reporta_a_taxa_efetiva():
    """A taxa pedida não é a que dispara: dias_recentes é discreto e empata."""
    cal = engajamento.calibrar(_coorte())
    assert cal.taxa_efetiva, 'calibrar tem de reportar a taxa efetiva'
    for taxa, ef in cal.taxa_efetiva.items():
        assert ef >= taxa - 1e-9, 'empate só pode fazer disparar MAIS, nunca menos'

def test_taxa_efetiva_bate_com_risco():
    cal = engajamento.calibrar(_coorte())
    coorte = _coorte()
    for taxa in (0.10, 0.20):
        disp = sum(engajamento.risco(c['dias_recentes'], c['dias_ativos'], cal, taxa)
                   for c in coorte) / len(coorte)
        assert abs(disp - cal.taxa_efetiva[taxa]) < 0.05


# ---------------- evasao.sql: a aritmética conferida em Python ----------------
import math, re as _re

def _coef_do_sql():
    """Lê os coeficientes do SQL publicado — pega drift entre doc e arquivo."""
    caminho = os.path.join(os.path.dirname(__file__), '..', 'sql', 'evasao.sql')
    corpo = open(caminho).read()
    corpo = corpo[corpo.index('SELECT 1.0'):corpo.index('$$;')]
    return [float(x) for x in _re.findall(r'\(?(-?\d+\.\d{6})\)?', corpo)]

def _risco_sql(cl1, cl4, sem_sem_acesso, semana, nota, feitas, perdidas, taxa, atraso):
    c = _coef_do_sql()
    z = (c[0] + c[1]*math.log1p(max(cl1, 0)) + c[2]*math.log1p(max(cl4, 0))
         + c[3]*(cl1/(cl4/4.0 + 1)) + c[4]*sem_sem_acesso + c[5]*semana
         + c[6]*76.68 + c[7]*nota + c[8]*feitas + c[9]*perdidas
         + c[10]*taxa + c[11]*atraso)
    return 1/(1 + math.exp(-z))

def test_sql_nao_superestima_o_risco():
    """O defeito encontrado na auditoria: devolvia 11x a taxa real."""
    tipico = _risco_sql(cl1=20, cl4=80, sem_sem_acesso=0, semana=10, nota=60,
                        feitas=3, perdidas=1, taxa=0.75, atraso=0)
    assert tipico < 0.15, f'aluno típico com risco {tipico:.2f} - alto demais'

def test_sql_entrega_perdida_aumenta_o_risco():
    poucos = _risco_sql(20, 80, 0, 10, 60, 3, 0, 0.9, 0)
    muitos = _risco_sql(20, 80, 0, 10, 60, 3, 4, 0.4, 0)
    assert muitos > poucos

def test_sql_mais_cliques_reduz_o_risco():
    ativo = _risco_sql(200, 800, 0, 10, 60, 3, 1, 0.75, 0)
    sumido = _risco_sql(1, 10, 0, 10, 60, 3, 1, 0.75, 0)
    assert sumido > ativo

def test_sql_semanas_sem_acesso_nao_reduz_o_risco():
    """Era o sinal invertido: -0,081 fazia sumir baixar o risco."""
    v = [_risco_sql(5, 40, k, 10, 60, 3, 1, 0.75, 0) for k in (0, 2, 4)]
    assert v[-1] >= v[0] - 1e-3, f'sumir reduz o risco: {v}'


# ---------------- limiares a jusante da recalibração ----------------
def test_precisa_reforco_exige_confianca():
    from trailup_core.dominio import Dominio
    baixo = Dominio(p=0.20, confianca=0.50, tendencia='estavel', respostas_topico=2)
    assert not dominio.precisa_reforco(baixo), 'p baixo sem evidência não dispara'

def test_precisa_reforco_limiar_mais_baixo_dispara_menos():
    from trailup_core.dominio import Dominio
    d = Dominio(p=0.35, confianca=0.80, tendencia='estavel', respostas_topico=20)
    assert dominio.precisa_reforco(d, limiar=0.45)
    assert not dominio.precisa_reforco(d, limiar=0.28), 'limiar de equivalência antiga'

def test_demorando_usa_a_distribuicao_da_questao():
    """Não é limiar absoluto de segundos: é razão contra a mediana da questão."""
    rapida = tempo.esperado(latencia_mediana=10.0, respostas=50)
    lenta = tempo.esperado(latencia_mediana=120.0, respostas=50)
    assert tempo.demorando(40.0, rapida), '40s numa questão de 10s é 4x'
    assert not tempo.demorando(40.0, lenta), '40s numa questão de 120s é rápido'

def test_demorando_nao_opina_sem_corpus():
    frouxo = tempo.esperado(latencia_mediana=10.0, respostas=2)
    assert not tempo.demorando(1000.0, frouxo), 'sem corpus não afirma'
