-- Risco de evasao: funcao SQL, para rodar no banco via pg_cron.
--
-- Vive no Postgres e nao na API por duas razoes registradas no CLAUDE.md do
-- TrailUp: precisa de relogio (rodada diaria), e a API hiberna no free tier do
-- Render. Regressao logistica vira aritmetica; nao precisa de runtime de ML.
--
-- Coeficientes REFEITOS em 2026-09-13. Os anteriores tinham dois defeitos que
-- a auditoria encontrou (ver docs/auditoria/AUDITORIA.md):
--
--   1. A funcao devolvia probabilidade 11,1x maior que a real - media 0,302
--      contra taxa de evasao de 2,7%, ECE 0,275. Vinham de um ajuste com
--      class_weight='balanced', que otimiza ordenacao e destroi calibracao.
--      Quem lesse o numero cru concluiria que um terco da turma esta saindo.
--
--   2. p_semanas_sem_acesso tinha coeficiente NEGATIVO (-0,081): mais semanas
--      sumido REDUZIA o risco. No dado bruto a evasao sobe monotonicamente com
--      as semanas sem acesso (2,11% -> 3,28% -> 3,76% -> 4,42%).
--
-- O defeito era invisivel para quem usava so a FAIXA, porque a ordenacao
-- continuava razoavel (AUC 0,764). So aparecia para quem lesse o score.
--
-- Painel aluno x semana do OULAD (CC BY 4.0), 564.006 linhas, split por coorte
-- (treina 2013B/2013J/2014B, testa 2014J em 195.797 linhas):
--   AUC 0,783 | ECE 0,006 | media prevista 0,021 contra 0,026 real
--
-- ENTRE COORTES (treina em tres, testa na quarta):
--   2013B 0,770 | 2013J 0,759 | 2014B 0,757 | 2014J 0,783
--   media 0,767 +- 0,012. O 0,783 acima e a coorte 2014J, que e a MELHOR das
--   quatro. Espere 0,767, nao 0,783. O lift no top 10% vai de 3,4x a 4,1x.
--   alertando o top 10%: precisao 10,9%, lift 4,1x
--   alertando o top  5%: precisao 14,7%, lift 5,6x
--   equidade: cobertura 40% (disability=N) x 45% (Y) - sem gap contra o grupo
--   de maior risco (4,1% x 2,5% de evasao).
--
-- Reproduzivel por docs/auditoria/scripts/64_sql.py.
--
-- ATENCAO: os coeficientes sao de universitarios a distancia do Reino Unido.
-- Transferem a FORMA (quais sinais importam), nao os valores. Recalibrar com
-- dado proprio assim que houver uma coorte que termine.

CREATE OR REPLACE FUNCTION trailup_risco_evasao(
  p_cliques_semana        numeric,   -- cliques/eventos na ultima semana
  p_cliques_4semanas      numeric,   -- soma das ultimas 4
  p_semanas_sem_acesso    numeric,   -- semanas inteiras sem nenhum acesso
  p_semana_do_curso       numeric,
  p_nota_media            numeric,   -- 0-100; -1 quando ainda nao ha nota
  p_entregas_feitas       numeric,
  p_entregas_perdidas     numeric,
  p_taxa_entrega          numeric,   -- feitas / esperadas; 1 quando nada venceu
  p_atraso_medio          numeric    -- dias; negativo = entrega adiantada
) RETURNS numeric
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT 1.0 / (1.0 + exp(-(
     -0.998883
    + (-0.239863) * ln(1 + greatest(p_cliques_semana,    0))
    + ( 0.013061) * ln(1 + greatest(p_cliques_4semanas,  0))
    + ( 0.002436) * (p_cliques_semana / (p_cliques_4semanas / 4.0 + 1))
    + (-0.000836) * p_semanas_sem_acesso     -- ~zero: os cliques da semana ja
                                             -- codificam "sumiu". A relacao
                                             -- MARGINAL e positiva e forte, mas
                                             -- condicionada aos cliques some.
    + (-0.040591) * p_semana_do_curso
    + ( 0.005945) * 76.68                    -- carga: media do corpus (sem analogo)
    + (-0.008495) * p_nota_media
    + (-0.027413) * p_entregas_feitas
    + ( 0.251735) * p_entregas_perdidas
    + (-1.688502) * p_taxa_entrega           -- o preditor mais forte depois dos cliques
    + ( 0.019125) * p_atraso_medio
  )));
$$;

COMMENT ON FUNCTION trailup_risco_evasao IS
  'Risco de abandono nas proximas 4 semanas. Coeficientes do OULAD; recalibrar com dado proprio.';

-- LIMIARES REFEITOS EM 2026-09-13, junto com os coeficientes. Os anteriores
-- (0,28 e 0,15) vinham da escala antiga, que superestimava o risco em 11x.
-- Sobre a escala corrigida eles ficaram MORTOS:
--
--   faixa 'alto' com limiar 0,28  ->  pegava 0,27% dos alunos-semana
--   faixa 'atencao' com 0,15      ->  pegava 0,63%
--
-- A precisao continuava alta (25% de evasao entre os marcados), mas a
-- cobertura era nula: a funcao praticamente nao classificava ninguem. Foi a
-- terceira vez neste modulo que consertar uma escala deixou para tras o
-- limiar que cortava nela - as outras duas foram `precisa_reforco` e o
-- limiar de `tendencia` do dominio.
--
-- Distribuicao do risco na escala corrigida (teste 2014J, 195.797 linhas):
--   media 0,0212 | mediana 0,0141 | p70 0,0213 | p90 0,0428 | p95 0,0557
--
-- Os limiares sao os percentis EXATOS, nao arredondados: arredondar 0,0428
-- para 0,043 empurra o corte para cima do p90 e a faixa pega menos de 10%.
--
-- Faixa em vez de score bruto: o professor ve a faixa, nao o numero.
-- Ver a issue de LGPD (#195): score bruto e serie historica de risco de menor
-- sao perfil; a faixa e acionavel sem expor o perfil.
CREATE OR REPLACE FUNCTION trailup_faixa_risco(p_risco numeric)
RETURNS text LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT CASE
    WHEN p_risco IS NULL THEN 'sem dados'
    WHEN p_risco >= 0.0428 THEN 'alto'      -- p90 medido: evasao 9,96%, lift 3,8x
    WHEN p_risco >= 0.0213 THEN 'atencao'   -- p70 medido: evasao 4,24%, lift 1,6x
    ELSE 'normal'                           -- 70%: evasao 1,11%, lift 0,4x
  END;
$$;
