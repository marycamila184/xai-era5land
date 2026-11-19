setwd("C:/Users/Vitor/Desktop/Faculdade/06 - Sexto Período/Explainable AI/Avaliação 1/R/R atualizado")

library(ranger)

modelo_final <- read_rds("modelo_rf_grade_reduzida.rds")

# Predições ---------------------------------------------------------------

predito <- predict(
  modelo_final,
  splits$teste %>%
    select(-anchor_id, -date, -tp_mm, -spatial_group)
)$predictions

predito <- data.frame(
  anchor_id = splits$teste$anchor_id,
  date = splits$teste$date,
  real = splits$teste$tp_mm,
  predito = predito
)

predito <- predito %>%
  mutate(
    erro = abs(real - predito),
    erro_s_abs = real - predito
  )


# Seleção das observações -------------------------------------------------

# Maior precipitação
maxprec <- predito %>%
  slice_max(real, n = 1, with_ties = FALSE) %>%
  mutate(tipo = "Maior precipitação")

# Menor precipitação
minprec <- predito %>%
  slice_min(real, n = 1, with_ties = FALSE) %>%
  mutate(tipo = "Menor precipitação")

# Maior erro de superestimação
maxerro <- predito %>%
  slice_min(erro_s_abs, n = 1, with_ties = FALSE) %>%
  mutate(tipo = "Maior superestimação")

# Menor erro
minerro <- predito %>%
  slice_min(erro, n = 1, with_ties = FALSE) %>%
  mutate(tipo = "Menor erro")


# Identificadores das quatro observações ----------------------------------

obs_ids <- bind_rows(
  maxprec,
  minprec,
  maxerro,
  minerro
) %>%
  select(anchor_id, date, tipo)


# Recuperar as observações originais --------------------------------------

obs_interesse <- splits$teste %>%
  inner_join(
    obs_ids,
    by = c("anchor_id", "date")
  )


# Conferência -------------------------------------------------------------

obs_interesse %>%
  select(
    anchor_id,
    date,
    tp_mm,
    tipo
  )


