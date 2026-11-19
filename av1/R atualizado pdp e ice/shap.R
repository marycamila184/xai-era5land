library(tidyverse)
library(kernelshap)
library(shapviz)

X_treino <- splits$treino %>%
  select(-tp_mm, -date, -anchor_id, -spatial_group)

X_interesse <- obs_interesse %>%
  select(-tp_mm, -date, -anchor_id, -spatial_group, -tipo)

pred_fun <- function(modelo, newdata) {
  
  predict(
    modelo,
    data = newdata
  )$predictions
}

set.seed(42)

bg_X <- X_treino %>%
  slice_sample(n = min(200, nrow(X_treino)))

shap_local <- kernelshap(
  modelo_final,
  X_interesse,
  bg_X = bg_X,
  pred_fun = pred_fun
)

variaveis <- c(
  # "dist_km",
  # "altitude",
  "tp_mm",
  "ssrd_wm2",
  "pev_mm",
  "t2m",
  "d2m",
  "u10",
  "v10",
  "wind_res"
)

nos <- c('n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw')

shap <- as.data.frame(shap_local$S)

shap_variaveis <- map_dfc(
  variaveis,
  function(var) {
    
    colunas <- paste0(var, "_no_", nos)
    
    tibble(
      !!var := rowMeans(
        shap %>%
          select(all_of(colunas))
      )
    )
  }
)

shap_variaveis_long <- shap_variaveis %>%
  mutate(
    anchor_id = obs_interesse$anchor_id,
    date = obs_interesse$date,
    tipo = obs_interesse$tipo
  ) %>%
  pivot_longer(
    cols = all_of(variaveis),
    names_to = "variavel",
    values_to = "shap"
  )

shap_variaveis_long %>%
  filter(tipo == unique(tipo)[1]) %>%
  arrange(desc(abs(shap)))

ggplot(
  shap_variaveis_long %>%
    filter(tipo == unique(tipo)[4]) %>%
    arrange(shap) %>%
    mutate(
      variavel = factor(variavel, levels = variavel)
    ),
  aes(x = shap, y = variavel)
) +
  geom_col() +
  labs(
    x = "SHAP médio",
    y = NULL,
    title = "SHAP local — variáveis agregadas"
  ) +
  theme_minimal()





ggplot(
  shap_variaveis_long %>%
    group_by(tipo) %>%
    arrange(shap, .by_group = TRUE) %>%
    mutate(
      variavel = factor(variavel, levels = unique(variavel))
    ) %>%
    ungroup(),
  aes(x = shap, y = variavel)
) +
  geom_col() +
  facet_wrap(~ tipo, scales = "free_y") +
  labs(
    x = "SHAP médio",
    y = NULL,
    title = "SHAP local — variáveis agregadas",
    subtitle = "Comparação das contribuições nas quatro observações"
  ) +
  theme_minimal() +
  theme(
    strip.text = element_text(face = "bold")
  )



# Contribuição dos nós ----------------------------------------------------

importancia_nos <- map_dfc(
  nos,
  function(no) {
    
    colunas <- paste0(variaveis, "_no_", no)
    
    tibble(
      !!paste0("no_", no) :=
        rowSums(
          abs(
            shap %>%
              select(all_of(colunas))
          )
        )
    )
  }
)

importancia_nos_long <- importancia_nos %>%
  mutate(
    anchor_id = obs_interesse$anchor_id,
    date = obs_interesse$date,
    tipo = obs_interesse$tipo
  ) %>%
  pivot_longer(
    cols = starts_with("no_"),
    names_to = "no",
    values_to = "importancia"
  )

importancia_nos_long <- importancia_nos_long %>%
  group_by(anchor_id, date, tipo) %>%
  mutate(
    importancia_relativa =
      importancia / sum(importancia)
  ) %>%
  ungroup()







# importancia_nos_long %>%
#   filter(tipo == unique(tipo)[4]) %>%
#   ggplot(
#     aes(
#       x = reorder(no, importancia_relativa),
#       y = importancia_relativa
#     )
#   ) +
#   geom_col() +
#   scale_y_continuous(labels = scales::percent) +
#   labs(
#     x = "Nó",
#     y = "Importância relativa",
#     title = "Importância espacial dos nós"
#   ) +
#   theme_minimal()

importancia_nos_long %>%
  ggplot(
    aes(
      x = reorder(no, importancia_relativa),
      y = importancia_relativa
    )
  ) +
  geom_col() +
  facet_wrap(~ tipo) +
  scale_y_continuous(
    labels = scales::percent
  ) +
  labs(
    x = "Nó",
    y = "Importância relativa",
    title = "Importância espacial dos nós",
    subtitle = "Comparação entre as quatro observações"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(face = "bold")
  )
