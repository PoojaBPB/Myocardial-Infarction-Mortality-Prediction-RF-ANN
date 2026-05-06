library(shiny)
library(caret)
library(nnet)

# -------------------------------------------------------------
# LOAD SAVED MODEL AND PREPROCESSING OBJECTS
# -------------------------------------------------------------

deployment_objects <- readRDS("mi_ann_model.rds")

model <- deployment_objects$model
keep_cols <- deployment_objects$keep_cols
categorical_cols <- deployment_objects$categorical_cols
continuous_cols <- deployment_objects$continuous_cols
mode_values <- deployment_objects$mode_values
median_values <- deployment_objects$median_values
scale_obj <- deployment_objects$scale_obj
nzv_cols <- deployment_objects$nzv_cols
final_feature_names <- deployment_objects$final_feature_names
positive_class <- deployment_objects$positive_class
negative_class <- deployment_objects$negative_class
classification_threshold <- deployment_objects$classification_threshold


# -------------------------------------------------------------
# USER INTERFACE
# -------------------------------------------------------------

ui <- fluidPage(
  
  tags$head(
    tags$style(HTML("
      body {
        background-color: whitesmoke;
        font-family: Arial, sans-serif;
        font-size: 17px;
      }

      .header {
        background-color: darkred;
        color: white;
        padding: 25px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
      }

      .card {
        background-color: white;
        padding: 22px;
        border-radius: 10px;
        box-shadow: 0 2px 8px lightgray;
        margin-bottom: 20px;
      }

      h2 {
        font-size: 34px;
        font-weight: bold;
      }

      h3 {
        color: darkred;
        font-size: 24px;
        font-weight: bold;
      }

      p {
        font-size: 17px;
        line-height: 1.6;
      }

      label {
        font-size: 16px;
        font-weight: bold;
      }

      .btn {
        width: 100%;
        background-color: darkred;
        color: white;
        font-size: 18px;
        padding: 10px;
        margin-top: 10px;
        border: none;
      }

      .btn:hover {
        background-color: firebrick;
        color: white;
      }

      .app-img {
        width: 100%;
        max-height: 260px;
        object-fit: cover;
        border-radius: 8px;
        margin-bottom: 15px;
      }

      .result-waiting {
        background-color: lightyellow;
        border-left: 7px solid darkred;
        padding: 18px;
        border-radius: 8px;
      }

      .result-survived {
        background-color: honeydew;
        border-left: 7px solid green;
        padding: 18px;
        border-radius: 8px;
      }

      .result-death {
        background-color: mistyrose;
        border-left: 7px solid red;
        padding: 18px;
        border-radius: 8px;
      }
    "))
  ),
  
  div(
    class = "header",
    h2("Myocardial Infarction Mortality Prediction App")
  ),
  
  sidebarLayout(
    
    sidebarPanel(
      width = 4,
      
      div(
        class = "card",
        
        img(src = "heart.png", class = "app-img"),
        
        h3("📝 Enter Patient Details"),
        
        textInput(
          inputId = "Age",
          label = "Age",
          value = ""
        ),
        
        selectInput(
          inputId = "Feature_2",
          label = "Gender",
          choices = c(
            "Please select" = "",
            "Female" = 0,
            "Male" = 1
          )
        ),
        
        selectInput(
          inputId = "Feature_3",
          label = "Quantity of myocardial infarctions in the anamnesis",
          choices = c(
            "Please select" = "",
            "Zero" = 0,
            "One" = 1,
            "Two" = 2,
            "Three and more" = 3
          )
        ),
        
        selectInput(
          inputId = "Feature_4",
          label = "Exertional angina pectoris in the anamnesis",
          choices = c(
            "Please select" = "",
            "Never" = 0,
            "During the last year" = 1,
            "One year ago" = 2,
            "Two years ago" = 3,
            "Three years ago" = 4,
            "4-5 years ago" = 5,
            "More than 5 years ago" = 6
          )
        ),
        
        selectInput(
          inputId = "Feature_8",
          label = "Presence of essential hypertension",
          choices = c(
            "Please select" = "",
            "No essential hypertension" = 0,
            "Stage 1" = 1,
            "Stage 2" = 2,
            "Stage 3" = 3
          )
        ),
        
        selectInput(
          inputId = "Feature_29",
          label = "Chronic bronchitis in the anamnesis",
          choices = c(
            "Please select" = "",
            "No" = 0,
            "Yes" = 1
          )
        ),
        
        actionButton("predict_btn", "Generate Prediction")
      )
    ),
    
    mainPanel(
      width = 8,
      
      div(
        class = "card",
        h3("🫀 About Myocardial Infarction"),
        p("Myocardial infarction, commonly known as a heart attack, occurs when blood flow to part of the heart muscle is reduced or blocked. This can damage heart tissue and may lead to serious complications if not treated quickly. After myocardial infarction, some patients recover, while others may be at higher risk of death depending on age, previous cardiac history, hypertension, and other clinical factors."),
        p("The purpose of this app is to provide a simple demonstration of mortality risk prediction after myocardial infarction using key patient information entered by the user")
      ),
      
      div(
        class = "card",
        h3("📊 Prediction Result"),
        uiOutput("prediction_result")
      ),
      
      div(
        class = "card",
        h3("🤖 Model Information"),
        p("This app uses an Artificial Neural Network model developed using the UCI Myocardial Infarction Complications dataset. It estimates the probability of Death after myocardial infarction and classifies the outcome as Death or Survived. Selected patient details are entered by the user, while the remaining required variables are filled using saved training defaults so that the model receives the correct input structure.")
      ),
      
      div(
        class = "card",
        h3("⚠️ Disclaimer"),
        p("This application is a prototype developed for demonstration purposes only. It should not be used for real clinical decision-making.")
      )
    )
  )
)


# -------------------------------------------------------------
# SERVER
# -------------------------------------------------------------

server <- function(input, output) {
  
  output$prediction_result <- renderUI({
    div(
      class = "result-waiting",
      "Enter patient details and click Generate Prediction to generate a prediction."
    )
  })
  
  observeEvent(input$predict_btn, {
    
    # Check that all user inputs are completed
    if (
      input$Age == "" ||
      input$Feature_2 == "" ||
      input$Feature_3 == "" ||
      input$Feature_4 == "" ||
      input$Feature_8 == "" ||
      input$Feature_29 == ""
    ) {
      output$prediction_result <- renderUI({
        div(
          class = "result-waiting",
          "Please complete all patient details before prediction."
        )
      })
      return()
    }
    
    # Validate age
    age_value <- as.numeric(input$Age)
    
    if (is.na(age_value) || age_value < 0 || age_value > 120) {
      output$prediction_result <- renderUI({
        div(
          class = "result-waiting",
          "Please enter a valid age between 0 and 120."
        )
      })
      return()
    }
    
    # -------------------------------------------------------------
    # CREATE ONE PATIENT ROW WITH ALL REQUIRED MODEL COLUMNS
    # -------------------------------------------------------------
    
    new_patient <- as.data.frame(matrix(nrow = 1, ncol = length(keep_cols)))
    colnames(new_patient) <- keep_cols
    
    
    # -------------------------------------------------------------
    # FILL ALL FEATURES WITH SAVED TRAINING DEFAULTS
    # This keeps the same input structure expected by the trained model.
    # -------------------------------------------------------------
    
    for (col in keep_cols) {
      
      if (col %in% continuous_cols) {
        new_patient[[col]] <- median_values[[col]]
      } else if (col %in% categorical_cols) {
        new_patient[[col]] <- mode_values[[col]]
      } else {
        new_patient[[col]] <- 0
      }
    }
    
    
    # -------------------------------------------------------------
    # REPLACE SELECTED FEATURES WITH USER INPUTS
    # -------------------------------------------------------------
    
    new_patient[["Age"]] <- age_value
    new_patient[["Feature_2"]] <- as.numeric(input$Feature_2)
    new_patient[["Feature_3"]] <- as.numeric(input$Feature_3)
    new_patient[["Feature_4"]] <- as.numeric(input$Feature_4)
    new_patient[["Feature_8"]] <- as.numeric(input$Feature_8)
    new_patient[["Feature_29"]] <- as.numeric(input$Feature_29)
    
    
    # -------------------------------------------------------------
    # APPLY THE SAME SAVED PREPROCESSING USED DURING TRAINING
    # -------------------------------------------------------------
    
    new_patient_scaled <- predict(scale_obj, new_patient)
    
    if (length(nzv_cols) > 0) {
      new_patient_final <- new_patient_scaled[, -nzv_cols, drop = FALSE]
    } else {
      new_patient_final <- new_patient_scaled
    }
    
    new_patient_final <- new_patient_final[, final_feature_names, drop = FALSE]
    
    
    # -------------------------------------------------------------
    # MAKE PREDICTION
    # -------------------------------------------------------------
    
    prob_death <- predict(
      model,
      new_patient_final,
      type = "prob"
    )[, positive_class]
    
    predicted_class <- ifelse(
      prob_death >= classification_threshold,
      positive_class,
      negative_class
    )
    
    probability_percent <- round(prob_death * 100, 2)
    
    
    # -------------------------------------------------------------
    # DISPLAY RESULT
    # -------------------------------------------------------------
    
    output$prediction_result <- renderUI({
      
      if (predicted_class == "Death") {
        
        div(
          class = "result-death",
          h3("Predicted Outcome: Death"),
          p(strong("Predicted probability of Death: "), paste0(probability_percent, "%"))
        )
        
      } else {
        
        div(
          class = "result-survived",
          h3("Predicted Outcome: Survived"),
          p(strong("Predicted probability of Death: "), paste0(probability_percent, "%"))
        )
      }
    })
  })
}


# -------------------------------------------------------------
# RUN APP
# -------------------------------------------------------------

shinyApp(ui = ui, server = server)