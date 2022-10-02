library(shiny)
library(shinydashboard)

#devtools::install_github("rstudio/rsconnect", ref='737cd48')

if (!Sys.info()[['user']] == 'ajay_'){
  # When running on shinyapps.io, create a virtualenv 
  envs<-reticulate::virtualenv_list()
  if(!'venv_shiny_app' %in% envs)
  {
    reticulate::virtualenv_create(envname = 'venv_shiny_app', 
                                  python = '/usr/bin/python3')
    reticulate::virtualenv_install('venv_shiny_app', 
                                   packages = c('numpy', 'keras==2.3.1', 
                                                'opencv-python', 'tensorflow'))
    
    reticulate::use_virtualenv("venv_shiny_app", required = TRUE)
  }
}

library(reticulate)

source_python("helpers/dog_breed_classifier.py")


# Define UI for data upload app ----
ui <- dashboardPage(
  
  # App title ----
  dashboardHeader(title = "Dog Breed Classifier"),
  
  # Sidebar layout with input and output definitions ----
  dashboardSidebar(
    sidebarMenu(
      menuItem("Upload Image File",
               tabName = "upload_csv",
               icon = icon("image"))
    )
  ),
  
  # Main panel for displaying outputs ----
  dashboardBody(
    
    tabItems(
      tabItem(tabName = "upload_csv",
              fluidRow(box(fileInput("file1", "Upload Image to be Classified",
                                     multiple = FALSE,
                                     accept = "image/*"),
                           height = 100,
                           width = 200)
              ),
              
              # Output: Image file ----
              fluidRow(column(width = 8, box(imageOutput("img1"),
                           width = 500,
                           height = 600)),
                       column(width = 4, box(textOutput("txt1"),
                                             width = 400,
                                             height = 600)))
      )
              
      )
      
    )
)


# Define server logic to read selected file ----
server <- function(input, output) {
  
  observe({
    if (is.null(input$file1)) return()
    file.copy(input$file1$datapath, "temp/",recursive = TRUE)
  })
  
  output$txt1 <- renderText({
    
    req(input$file1)
    
#    if (length(list.files("temp/")) > 0)
#    {
      return(what_are_you("temp/0.jpg"))
      
#    } else return("Upload an Image")
    
  })
  
  output$img1 <- renderImage(
    {
      req(input$file1)
      
      filename <- normalizePath(file.path('temp/0.jpg'))
      
      # Return a list containing the filename
      list(src = filename, width = 400, height = 500,alt = "This is alternate text")
    }
    , deleteFile = TRUE)
  
  
  
}

# Create Shiny app ----
shinyApp(ui, server)

