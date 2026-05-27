library(ggplot2)

# Put the real data in:
#data = read.csv("path/to/data")

# This is just a manually defined placeholder
data = data.frame(IR = c("Semantic", "Semantic", "Semantic"),
           Scenario = c("No Rag", "Rag", "g-Rag"),
           author_qa = c(0.25, 0.6, 0.8),
           chronology_qa = c(0.25, 0.6, 0.8)
           )

# Make sure this is correct
number_of_questions = 1000

# Set up the name path for the output
output = "/home/gabriel/Documents/RAG/example.png"

####################################
# Leave as is from now on

author_lower_intervals = vector()
author_upper_intervals = vector()
chronology_lower_intervals = vector()
chronology_upper_intervals = vector()
situations = vector()

for (i in 1:nrow(data)) {
  
  situation = paste0(data[i, 1], " / ", data[i, 2])
  
  author_interval <- binom.test(data[i,3] * number_of_questions,
                                number_of_questions,
                                conf.level = 0.99)$conf.int
  
  chronology_interval <- binom.test(data[i,4] * number_of_questions,
                                    number_of_questions,
                                    conf.level = 0.99)$conf.int
  

  author_lower_intervals = c(author_lower_intervals, author_interval[1])
  author_upper_intervals = c(author_upper_intervals, author_interval[2])
  chronology_lower_intervals = c(chronology_lower_intervals, chronology_interval[1])
  chronology_upper_intervals = c(chronology_upper_intervals, chronology_interval[2])
  situations = c(situations, situation)
  
}

data = cbind(data,
               author_lower_intervals,
               author_upper_intervals,
               chronology_lower_intervals,
               chronology_upper_intervals)



png(filename = output)

ggplot(data, aes(y = situations, x = author_qa)) +
  geom_point(size = 3) +
  geom_errorbarh(
    aes(xmin = author_lower_intervals, xmax = author_upper_intervals),
    height = 0.2
  ) +
  labs(
    x = "Probability of correct answer",
    y = "Scenarios",
    title = "Estimated values for p with 99% confidence intervals"
  ) +
  theme_minimal()

dev.off()

