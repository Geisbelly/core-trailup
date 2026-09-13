-- Risco de evasao: funcao SQL, para rodar no banco via pg_cron.
--
-- Vive no Postgres e nao na API por duas razoes registradas no CLAUDE.md do
-- TrailUp: precisa de relogio (rodada diaria), e a API hiberna no free tier do
-- Render. Regressao logistica vira aritmetica; nao precisa de runtime de ML.
--
-- Coeficientes medidos no OULAD (CC BY 4.0), painel aluno x semana, 523 mil
-- linhas, split por coorte (treina 2013B/2013J/2014B, testa 2014J):
--   AUC 0,737 (logistica) / 0,746 (boosting) | alertando o top 10%:
--   precisao 12,2% contra base de 3,6% -> lift 3,3x, cobrindo 1/3 de quem evade.
--   Sem gap de equidade: AUC 0,747 (disability=N) x 0,732 (Y).
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
      1.560766
    + (-0.332613) * ln(1 + greatest(p_cliques_semana,    0))
    + (-0.040751) * ln(1 + greatest(p_cliques_4semanas,  0))
    + ( 0.076028) * (p_cliques_semana / (p_cliques_4semanas / 4.0 + 1))
    + (-0.081478) * p_semanas_sem_acesso
    + (-0.032992) * p_semana_do_curso
    + ( 0.007351) * 76.68                    -- carga: media do corpus (sem analogo)
    + (-0.007206) * p_nota_media
    + ( 0.011575) * p_entregas_feitas
    + ( 0.012417) * p_entregas_perdidas
    + (-1.023119) * p_taxa_entrega           -- o preditor mais forte depois dos cliques
    + ( 0.007044) * p_atraso_medio
  )));
$$;

COMMENT ON FUNCTION trailup_risco_evasao IS
  'Risco de abandono nas proximas 4 semanas. Coeficientes do OULAD; recalibrar com dado proprio.';

-- Faixa em vez de score bruto: o professor ve a faixa, nao o numero.
-- Ver a issue de LGPD (#195): score bruto e serie historica de risco de menor
-- sao perfil; a faixa e acionavel sem expor o perfil.
CREATE OR REPLACE FUNCTION trailup_faixa_risco(p_risco numeric)
RETURNS text LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT CASE
    WHEN p_risco IS NULL THEN 'sem dados'
    WHEN p_risco >= 0.28 THEN 'alto'        -- ~top 10%: precisao 12,2%, lift 3,3x
    WHEN p_risco >= 0.15 THEN 'atencao'
    ELSE 'normal'
  END;
$$;
