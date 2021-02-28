library(shiny)

library(rio)
url <- 'https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx'
ENB2012_data <- rio::import(file = url)


library(shinythemes)

ui <- fluidPage(theme = shinytheme("flatly"),
                headerPanel("Energy-Efficiency"),
                fluidRow(
                  column(4,
                         wellPanel(
                           numericInput("x1", "Relative Compactness:", 0.98),
                           numericInput("x2", "Surface Area:", 514.5),
                           numericInput("x3", "Wall Area:", 294.0),
                           numericInput("x4", "Roof Area:", 110.25),
                           numericInput("x5", "Overall Height:", 7.0),
                           numericInput("x6", "Orientation:", 2),
                           numericInput("x7", "Glazing Area:", 0),
                           numericInput("x8", "Glazing Area Distribution:", 0),
                           actionButton("hl_but", "Heating Load",
                                        class = "btn-success"),
                           actionButton("cl_but", "Cooling Load",
                                        class = "btn-success")
                         )
                  ),
                  column(8,
                         tabsetPanel(
                           tabPanel("Heating Load", textOutput("hl")),
                           tabPanel("Cooling Load", textOutput("cl"))
                         )
                  )
                )
)


server <- function(input, output){
  predictinput_hl <- eventReactive(input$hl_but, {
    
    library(tidymodels)
    
    ##rsample
    set.seed(123)
    
    enb_split <- initial_split(ENB2012_data, prop = 0.75)
    
    enb_train <- training(enb_split)
    enb_test  <- testing(enb_split)
    
    ##recipes
    
    ## heating load
    enb_recipe_hl <- 
      recipe(Y1 ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8, data = enb_train)
    
    enb_recipe_steps_hl <- enb_recipe_hl %>% 
      step_sqrt(all_predictors())
    
    prepped_recipe_hl <- prep(enb_recipe_steps_hl, training = enb_train)
    
    enb_train_preprocessed_hl <- bake(prepped_recipe_hl, enb_train) 
    
    enb_test_preprocessed_hl <- bake(prepped_recipe_hl, enb_test)
    
    boost_model <- 
      boost_tree(mtry = 6, trees = 116, min_n = 19, tree_depth = 8, sample_size = 0.986) %>% 
      set_engine("xgboost") %>%
      set_mode("regression") %>%
      translate()
    
    
    boost_form_fit <- 
      boost_model %>% 
      fit(Y1 ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8, data = enb_train_preprocessed_hl)
    
    
    ##workflow
    boost_wflow <- 
      workflow() %>% 
      add_model(boost_model)
    
    boost_wflow <- 
      boost_wflow %>% 
      add_formula(Y1 ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8)
    
    boost_fit <- fit(boost_wflow, enb_train_preprocessed_hl)
    
    new_predict_hl <- predict(boost_fit, new_data = data.frame(X1 = sqrt(input$x1), X2 = sqrt(input$x2),
                                             X3 = sqrt(input$x3), X4 = sqrt(input$x4),
                                             X5 = sqrt(input$x5), X6 = sqrt(input$x6),
                                             X7 = sqrt(input$x7), X8 = sqrt(input$x8)))
    new_predict_hl$.pred
    
  })
  
  predictinput_cl <- eventReactive(input$cl_but, {
    
    library(tidymodels)
    
    ##rsample
    set.seed(123)
    
    enb_split <- initial_split(ENB2012_data, prop = 0.75)
    
    enb_train <- training(enb_split)
    enb_test  <- testing(enb_split)
    
    ##recipes
    
    ## heating load
    enb_recipe_cl <- 
      recipe(Y2 ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8, data = enb_train)
    
    enb_recipe_steps_cl <- enb_recipe_cl %>% 
      step_sqrt(all_predictors())
    
    prepped_recipe_cl <- prep(enb_recipe_steps_cl, training = enb_train)
    
    enb_train_preprocessed_cl <- bake(prepped_recipe_cl, enb_train) 
    
    enb_test_preprocessed_cl <- bake(prepped_recipe_cl, enb_test)
    
    enb_cv_preprocessed_cl <- vfold_cv(enb_train_preprocessed_cl, v = 10)
    
    boost_model <- 
      boost_tree(mtry = 1, trees = 752, min_n = 6, tree_depth = 11, sample_size = 0.562) %>% 
      set_engine("xgboost") %>%
      set_mode("regression") %>%
      translate()
    
    
    boost_form_fit <- 
      boost_model %>% 
      fit(Y2 ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8, data = enb_train_preprocessed_cl)
    
    
    ##workflow
    boost_wflow <- 
      workflow() %>% 
      add_model(boost_model)
    
    boost_wflow <- 
      boost_wflow %>% 
      add_formula(Y2 ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8)
    
    boost_fit <- fit(boost_wflow, enb_train_preprocessed_cl)
    
    new_predict_cl <- predict(boost_fit, new_data = data.frame(X1 = sqrt(input$x1), X2 = sqrt(input$x2),
                                                               X3 = sqrt(input$x3), X4 = sqrt(input$x4),
                                                               X5 = sqrt(input$x5), X6 = sqrt(input$x6),
                                                               X7 = sqrt(input$x7), X8 = sqrt(input$x8)))
    new_predict_cl$.pred
    
  })
  
  output$hl <- renderText(
    predictinput_hl()
  )
  
  output$cl <- renderText(
    predictinput_cl()
  )
}

shinyApp(ui, server)
