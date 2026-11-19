library(tidyverse)
library(ranger)
library(DALEX)
library(ingredients)
library(patchwork)

# Preditores usados pelo modelo
X_treino <- splits$treino %>%
  select(-tp_mm, -date, -anchor_id, -spatial_group)

X_interesse <- obs_interesse %>%
  select(-tp_mm, -date, -anchor_id, -spatial_group)

# Função de predição
pred_fun <- function(modelo, newdata) {
  predict(modelo, data = newdata)$predictions
}

# Explainer
explainer_rf <- DALEX::explain(
  model = modelo_final,
  data = X_treino,
  y = splits$treino$tp_mm,
  predict_function = pred_fun,
  label = "Random Forest",
  verbose = FALSE
)

# Índices das quatro observações
indices <- seq_len(nrow(X_interesse))

nos <- c('n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw')

# Grade de valores para a temperatura nos oito nós
variavel <- "t2m"
colunas <- paste0(variavel, "_no_", nos)

valores <- quantile(
  unlist(X_treino[, colunas]),
  probs = seq(0.05, 0.95, length.out = 25),
  na.rm = TRUE
) %>%
  as.numeric() %>%
  unique() %>%
  sort()

# Função para variar a temperatura conjuntamente nos oito nós
curvas_cp <- map_dfr(indices, function(i) {
  map_dfr(valores, function(v) {
    novo <- X_interesse[i, , drop = FALSE]
    novo[, colunas] <- v
    
    tibble(
      observacao = obs_interesse$tipo[i],
      valor = v,
      predicao = pred_fun(modelo_final, novo)
    )
  })
})

ggplot(
  curvas_cp,
  aes(x = valor, y = predicao, color = observacao)
) +
  geom_line(linewidth = 1) +
  labs(
    title = "Ceteris Paribus — temperatura",
    x = "Temperatura nos oito nós",
    y = "Precipitação prevista (mm)",
    color = "Observação"
  ) +
  theme_minimal()

grade_pdp <- tibble(valor = valores) %>%
  mutate(
    predicao_media = map_dbl(valor, function(v) {
      dados <- X_treino
      
      dados[, colunas] <- v
      
      mean(pred_fun(modelo_final, dados))
    })
  )

ggplot(
  grade_pdp,
  aes(x = valor, y = predicao_media)
) +
  geom_line(linewidth = 1, color = "#2C7FB8") +
  labs(
    title = "Partial Dependence Plot — temperatura",
    x = "Temperatura nos oito nós",
    y = "Precipitação prevista média (mm)"
  ) +
  theme_minimal()



# Para todas as variáveis -------------------------------------------------

library(tidyverse)
library(ggplot2)

variaveis <- c(
  "tp_mm", "ssrd_wm2", "pev_mm",
  "t2m", "d2m", "u10", "v10", "wind_res"
)

nos <- c('n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw')

# Resultados CP
curvas_cp_todas <- map_dfr(variaveis, function(var) {
  
  colunas <- paste0(var, "_no_", nos)
  
  valores <- quantile(
    unlist(X_treino[, colunas]),
    probs = seq(0.05, 0.95, length.out = 25),
    na.rm = TRUE
  ) %>%
    as.numeric() %>%
    unique() %>%
    sort()
  
  map_dfr(seq_len(nrow(X_interesse)), function(i) {
    
    map_dfr(valores, function(v) {
      
      novo <- X_interesse[i, , drop = FALSE]
      novo[, colunas] <- v
      
      tibble(
        variavel = var,
        observacao = obs_interesse$tipo[i],
        valor = v,
        predicao = pred_fun(modelo_final, novo)
      )
    })
  })
})

# Resultados PDP
grade_pdp_todas <- map_dfr(variaveis, function(var) {
  
  colunas <- paste0(var, "_no_", nos)
  
  valores <- quantile(
    unlist(X_treino[, colunas]),
    probs = seq(0.05, 0.95, length.out = 25),
    na.rm = TRUE
  ) %>%
    as.numeric() %>%
    unique() %>%
    sort()
  
  map_dfr(valores, function(v) {
    
    dados <- X_treino
    dados[, colunas] <- v
    
    tibble(
      variavel = var,
      valor = v,
      predicao = mean(pred_fun(modelo_final, dados))
    )
  })
})

ggplot() +
  geom_line(
    data = curvas_cp_todas,
    aes(
      x = valor,
      y = predicao,
      color = observacao,
      group = observacao
    ),
    linewidth = 0.7,
    alpha = 0.8
  ) +
  geom_line(
    data = grade_pdp_todas,
    aes(
      x = valor,
      y = predicao
    ),
    color = "black",
    linewidth = 1.1,
    linetype = "dashed"
  ) +
  facet_wrap(
    ~ variavel,
    scales = "free",
    ncol = 3
  ) +
  labs(
    title = "Ceteris Paribus e Partial Dependence",
    subtitle = "CP: linhas coloridas por observação | PDP: linha preta tracejada",
    x = "Valor da variável nos oito nós",
    y = "Precipitação prevista (mm)",
    color = "Observação"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    strip.text = element_text(face = "bold"),
    plot.title = element_text(face = "bold")
  )
