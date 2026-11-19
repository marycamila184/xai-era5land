library(ranger)

# Treino dos modelos ------------------------------------------------------

## Random Forests ----------------------------------------------------------

# Grade de parâmetros -----------------------------------------------------
set.seed(42)

grid_rf <- expand_grid(
  numtrees = c(100, 300, 500),
  mtry = seq(2, 62, by = 5),   # ALTERADO (Claude): ia até 77; agora são 64 preditores (8 variáveis x 8 vizinhas)
  min.node.size = c(10, 20, 50, 100),
  splitrule = c("variance", "extratrees")
)

# Treino e validação ------------------------------------------------------

resultados_rf <- data.frame()

for (g in seq_len(nrow(grid_rf))) {
  
  numtrees <- grid_rf$numtrees[g]
  mtry <- grid_rf$mtry[g]
  min.node.size <- grid_rf$min.node.size[g]
  splitrule <- grid_rf$splitrule[g]
  
  metricas_folds <- data.frame(
    RMSE = numeric(length(blocos)),
    MAE = numeric(length(blocos)),
    R2 = numeric(length(blocos))
  )
  
  message('Sequencia de ', g, ' de ',
          nrow(grid_rf), ' combinações de parâmetros sendo avaliada')
  
  for (i in seq_along(blocos)) {
    
    treino_fold <- blocos[[i]]$treino
    validacao_fold <- blocos[[i]]$validacao
    
    modelo <- ranger(
      tp_mm ~ .,
      data = treino_fold %>%
        select(-date, -anchor_id, -spatial_group),
      num.trees = numtrees,
      mtry = mtry,
      min.node.size = min.node.size,
      splitrule = splitrule
    )
    
    pred <- predict(
      modelo,
      data = validacao_fold %>%
        select(-tp_mm, -date, -anchor_id, -spatial_group)
    )$predictions
    
    observado <- validacao_fold$tp_mm
    
    metricas_folds$RMSE[i] <- sqrt(
      mean((observado - pred)^2, na.rm = TRUE)
    )
    
    metricas_folds$MAE[i] <- mean(
      abs(observado - pred),
      na.rm = TRUE
    )
    
    metricas_folds$R2[i] <- 1 -
      sum((observado - pred)^2, na.rm = TRUE) /
      sum((observado - mean(observado, na.rm = TRUE))^2,
          na.rm = TRUE)
  }
  
  resultados_rf <- rbind(
    resultados_rf,
    data.frame(
      numtrees = numtrees,
      mtry = mtry,
      min.node.size = min.node.size,
      splitrule = splitrule,
      RMSE = mean(metricas_folds$RMSE, na.rm = TRUE),
      MAE = mean(metricas_folds$MAE, na.rm = TRUE),
      R2 = mean(metricas_folds$R2, na.rm = TRUE)
    )
  )
}

# Ordenar pelas menores RMSE
resultados_rf %>%
  arrange(RMSE) %>% head()

# Melhor modelo RF --------------------------------------------------------

(melhores_params_rf <- resultados_rf %>% arrange(RMSE) %>% head(1) %>% select(1:4))

set.seed(42)

# ALTERADO (Claude): faltava a vírgula depois do num.trees; anchor_id e spatial_group
# saem do treino e do teste, como na validação
melhor_modelo_rf <- ranger(
  tp_mm ~ .,
  data = splits$treino %>%
    select(-date, -anchor_id, -spatial_group),
  num.trees = melhores_params_rf$numtrees[1],
  mtry = melhores_params_rf$mtry[1],
  min.node.size = melhores_params_rf$min.node.size[1],
  splitrule = melhores_params_rf$splitrule[1]
)

pred <- predict(
  melhor_modelo_rf,
  data = splits$teste %>%
    select(-tp_mm, -date, -anchor_id, -spatial_group)
)$predictions

observado <- splits$teste$tp_mm

RMSE_final <- yardstick::rmse_vec(truth = observado, estimate = pred)

MAE_final <- yardstick::mae_vec(truth = observado, estimate = pred)

R2_final <- yardstick::rsq_trad_vec(truth = observado, estimate = pred)

resultados_rf_final <- data.frame(
  num.trees = melhores_params_rf$numtrees[1],
  mtry = melhores_params_rf$mtry[1],
  min.node.size = melhores_params_rf$min.node.size[1],
  splitrule = melhores_params_rf$splitrule[1],
  RMSE = RMSE_final,
  MAE = MAE_final,
  R2 = R2_final
)

# Exportar o modelo -------------------------------------------------------

# ALTERADO (Claude): salva o modelo final e os dados de treino e teste para usar em
# explicabilidade (por exemplo DALEX), sem precisar treinar de novo
saveRDS(melhor_modelo_rf, 'modelo_rf_final.rds')
saveRDS(splits, 'splits_treino_teste.rds')
