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
    c = [dominio.confianca(n) for n in (0, 5, 20, 100, 1000)]
    assert c == sorted(c) and c[-1] <= 0.92


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
