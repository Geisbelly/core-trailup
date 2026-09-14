"""ETAPA 1 - constroi o painel a partir das duas bases cruas.

Uma linha por aluno, com os eixos medidos nos dias 0-29 e o desfecho nos
dias 30-59. As duas bases viram o MESMO formato para poderem ser comparadas.

Saida: saida/painel.parquet

-- PROVENIENCIA ------------------------------------------------------------
Este script e a etapa 1 do pipeline do seminario, em `seminario/scripts/`,
que tem README e diario de decisoes. A copia aqui existe para o modelo
`engajamento` guardar, junto dos demais scripts de auditoria, o codigo que
produziu os numeros do relatorio.

Para RODAR o pipeline na ordem e com as instrucoes, use `seminario/scripts/`.
Aqui os caminhos foram normalizados como nos outros scripts de `docs/` -
ver docs/LEIA-ME_scripts.md.
"""
import sys, glob, os
import numpy as np
import pandas as pd
from comum import (EDNET, OULAD, SAIDA, JANELA, HORIZONTE, GAP_SESSAO_SEG,
                   MIN_EVENTOS, JANELA_RECENTE, exige)

AJUDA_EDNET = ('EdNet KT3: https://github.com/riiid/ednet\n'
               'Baixe KT3.zip, descompacte em dados/EdNet-KT3/KT3/ (um CSV por aluno).\n'
               'Licenca CC BY-NC 4.0 - uso academico.')
AJUDA_OULAD = ('OULAD: https://analyse.kmi.open.ac.uk/open_dataset\n'
               'Baixe e descompacte os CSV em dados/oulad/.\n'
               'Licenca CC BY 4.0.')

LIMITE_ALUNOS = int(os.environ.get('LIMITE_ALUNOS', '60000'))


def eixos(dias_ativos_set, n_eventos, n_sessoes, janela=JANELA):
    """Os quatro eixos candidatos. So dois sobrevivem - ver etapa 2."""
    dias = sorted(dias_ativos_set)
    recentes = [d for d in dias if d >= janela - JANELA_RECENTE]
    return {
        'frequencia':   len(dias) / janela,
        'recencia':     len(recentes) / JANELA_RECENTE,
        'volume':       n_eventos / max(n_sessoes, 1),
        'profundidade': n_eventos / max(len(dias), 1),
        'dias_ativos':  len(dias),
        'dias_recentes': len(recentes),
        'eventos':      n_eventos,
    }


def ler_ednet():
    exige(EDNET, AJUDA_EDNET)
    arquivos = sorted(glob.glob(str(EDNET / 'u*.csv')))
    print(f'EdNet: {len(arquivos):,} arquivos de aluno', flush=True)
    rng = np.random.default_rng(0)
    if len(arquivos) > LIMITE_ALUNOS:
        arquivos = [arquivos[i] for i in
                    sorted(rng.choice(len(arquivos), LIMITE_ALUNOS, replace=False))]
        print(f'  amostrando {len(arquivos):,} (LIMITE_ALUNOS)', flush=True)
    linhas = []
    for k, caminho in enumerate(arquivos):
        if k % 10000 == 0:
            print(f'  {k:,}...', flush=True)
        try:
            d = pd.read_csv(caminho, usecols=['timestamp', 'action_type'])
        except Exception:
            continue
        if len(d) < MIN_EVENTOS:
            continue
        t = d.timestamp.to_numpy(dtype='int64')
        t.sort()
        # O "dia 0" e o primeiro evento DAQUELE aluno, nao uma data de calendario.
        # Cada aluno entra quando entra; o que se compara e o mesmo trecho de vida.
        dia = (t - t[0]) // 86_400_000
        na_janela = dia < JANELA
        no_horizonte = (dia >= JANELA) & (dia < HORIZONTE)
        if na_janela.sum() < MIN_EVENTOS:
            continue
        tj = t[na_janela]
        sessoes = 1 + int((np.diff(tj) > GAP_SESSAO_SEG * 1000).sum())
        e = eixos(set(dia[na_janela].tolist()), int(na_janela.sum()), sessoes)
        e['voltou'] = int(no_horizonte.sum() > 0)
        e['coorte'] = 'EdNet'
        e['aluno'] = os.path.basename(caminho)[:-4]
        linhas.append(e)
    return pd.DataFrame(linhas)


def ler_oulad():
    exige(OULAD / 'studentVle.csv', AJUDA_OULAD)
    print('OULAD: lendo studentVle.csv', flush=True)
    v = pd.read_csv(OULAD / 'studentVle.csv',
                    usecols=['code_module', 'code_presentation', 'id_student',
                             'date', 'sum_click'])
    v = v[v.date >= 0]
    chave = ['code_module', 'code_presentation', 'id_student']
    # O OULAD registra o DIA, nao a hora. Sem timestamp nao da para cortar
    # sessao, entao o eixo `volume` (eventos por sessao) NAO E MEDIVEL aqui.
    # Deixamos NaN de proposito. A primeira versao deste script preenchia com
    # `profundidade`, o que fazia o teste de replicacao do volume comparar a
    # coluna com ela mesma - passava sem medir nada. Ver DIARIO, 13/09.
    dia0 = v.groupby(chave).date.transform('min')
    v['d'] = v.date - dia0
    jan = v[v.d < JANELA]
    hor = v[(v.d >= JANELA) & (v.d < HORIZONTE)]
    g = jan.groupby(chave)
    base = pd.DataFrame({
        'dias_ativos': g.d.nunique(),
        'eventos': g.sum_click.sum(),
    })
    rec = jan[jan.d >= JANELA - JANELA_RECENTE].groupby(chave).d.nunique()
    base['dias_recentes'] = rec.reindex(base.index).fillna(0)
    base = base[base.eventos >= MIN_EVENTOS]
    base['frequencia'] = base.dias_ativos / JANELA
    base['recencia'] = base.dias_recentes / JANELA_RECENTE
    base['profundidade'] = base.eventos / base.dias_ativos.clip(lower=1)
    base['volume'] = np.nan
    voltou = hor.groupby(chave).size()
    base['voltou'] = (voltou.reindex(base.index).fillna(0) > 0).astype(int)
    base['coorte'] = 'OULAD'
    base = base.reset_index()
    base['aluno'] = (base.code_module + '_' + base.code_presentation + '_' +
                     base.id_student.astype(str))
    return base[['frequencia', 'recencia', 'volume', 'profundidade', 'dias_ativos',
                 'dias_recentes', 'eventos', 'voltou', 'coorte', 'aluno']]


if __name__ == '__main__':
    partes = []
    quais = sys.argv[1:] or ['ednet', 'oulad']
    if 'ednet' in quais:
        partes.append(ler_ednet())
    if 'oulad' in quais:
        partes.append(ler_oulad())
    p = pd.concat(partes, ignore_index=True)
    p.to_parquet(SAIDA / 'painel.parquet')
    print(f'\npainel: {len(p):,} alunos -> {SAIDA / "painel.parquet"}')
    print(p.groupby('coorte').agg(alunos=('voltou', 'size'),
                                  retencao=('voltou', 'mean')).to_string())
