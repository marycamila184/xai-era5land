# Diretório ---------------------------------------------------------------

setwd("C:/Users/Vitor/Desktop/Faculdade/06 - Sexto Período/Explainable AI/Avaliação 1/R")

# Bibliotecas -------------------------------------------------------------

library(tidyverse)
library(arrow)
library(lubridate)
library(ggplot2)

# Dados -------------------------------------------------------------------

grupos_espaciais <- read.csv('../Dados/grupos_espaciais.csv')
grupos_espaciais <- grupos_espaciais %>% filter(region == 'Sul') %>% 
  select(
    anchor_id, spatial_group
  )

dados <- read_parquet("../Dados/era5land_amostra_sul.parquet")
# ALTERADO (Claude): as datas vêm em UTC; sem converter, o R mostra no fuso local e o
# dia 1º de janeiro cai no ano anterior em year()
# ALTERADO (Claude): início em 2012, não em 2014, para a validação ir de 2014 a 2023 e
# 2024-2025 ficarem só para o teste
dados <- dados %>% 
  mutate(date = as.Date(date, tz = 'UTC')) %>% 
  filter(    
    date >= as.Date('2012-01-01'),
    date < as.Date('2026-01-01')
  )
dados <- dados %>% 
  select(
    date,
    anchor_id,
    position,
    # ALTERADO (Claude): dist_km e altitude saíram, são constantes por âncora e
    # deixavam o modelo reconhecer a âncora
    tp_mm,
    ssrd_wm2,
    pev_mm,
    t2m,
    d2m,
    u10,
    v10,
    wind_res
  )

ancoras <- dados %>% filter(position == 'center') %>% select(date, anchor_id, tp_mm)

vizinhos <- dados %>%
  filter(position != 'center') %>%
  pivot_wider(
    names_from = position,
    values_from = c(
      # ALTERADO (Claude): sem dist_km e altitude, como no select acima
      tp_mm,
      ssrd_wm2,
      pev_mm,
      t2m,
      d2m,
      u10,
      v10,
      wind_res
    ),
    names_glue = "{.value}_no_{position}"
  )

base_modelo <- ancoras %>% left_join(vizinhos, by = c('anchor_id', 'date')) %>% 
  left_join(grupos_espaciais, by = c('anchor_id'))
# head(base_modelo)
# base_modelo$spatial_group
# ALTERADO (Claude): o view() saiu, ele só funciona no RStudio


blocos <- vector('list', 5)

# ALTERADO (Claude): validação de 2014 a 2023, antes ia de 2016 a 2025 e usava os anos
# do teste; 2024 e 2025 ficam só para o teste
visu <- data.frame(
  spatial_group = 0:4,
  ini_train = rep(2012, 5),
  fin_train = seq(2013, 2021, by = 2),
  ini_validation = seq(2014, 2022, by = 2),
  fin_validation = seq(2015, 2023, by = 2)
)

for (i in 1:5) {
  
  blocos[[i]] <- list(
    treino = base_modelo %>%
      filter(
        year(date) <= visu$fin_train[i],
        spatial_group != visu$spatial_group[i]
      ),
    validacao = base_modelo %>% 
      filter(
        year(date) >= visu$ini_validation[i],
        year(date) <= visu$fin_validation[i],
        spatial_group == visu$spatial_group[i]
      )
  )
  
}

# Treino e teste do modelo final ------------------------------------------

# ALTERADO (Claude): o splits não existia e o treino_ancoras_150.R usa ele no modelo final

splits <- list(
  treino = base_modelo %>% filter(year(date) <= 2023),
  teste = base_modelo %>% filter(year(date) >= 2024)
)
