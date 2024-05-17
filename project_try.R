## Set WD
setwd("C:/Users/Monica/OneDrive - Brown University/Desktop/PHP2530 Bayesian/Project/")
options("install.lock"=FALSE)

## Load libraries
library(naniar)
library(dplyr)
library(ggplot2)
library(corrplot)
library(caret)
library(tidymodels)
library(naivebayes)
library(nnet)
theme_set(theme_minimal())

## Load data
dat <- read.csv("ObesityDataSet_raw_and_data_sinthetic.csv")
dat <- dat[1:498,] #take original data before SMOTE



## Data Preprocessing (covariates)
dat$CALC[dat$CALC == "Always"] <- "Frequently"
dat$CAEC[dat$CAEC == "Always"] <- "Frequently"
dat$MTRANS[dat$MTRANS == "Motorbike"] <- "Motor_Vehicle"
dat$MTRANS[dat$MTRANS == "Automobile"] <- "Motor_Vehicle"
dat$MTRANS[dat$MTRANS == "Bike"] <- "Walking/Bike"
dat$MTRANS[dat$MTRANS == "Walking"] <- "Walking/Bike"

dat$NObeyesdad <- ifelse(dat$NObeyesdad %in% c("Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"), "Obese", dat$NObeyesdad)
dat$NObeyesdad <- ifelse(dat$NObeyesdad %in% c("Overweight_Level_I", "Overweight_Level_II"), "Overweight", dat$NObeyesdad)
dat$NObeyesdad[dat$NObeyesdad == "Insufficient_Weight"] <- "Underweight"

## Factor variables
dat$NObeyesdad <- factor(dat$NObeyesdad,levels = c("Underweight","Normal_Weight","Obese","Overweight"))
dat$Gender <- factor(dat$Gender, levels=c("Female","Male"))
dat$family_history_with_overweight <- factor(dat$family_history_with_overweight,levels = c("no","yes"))
dat$FAVC <- factor(dat$FAVC,levels = c("no","yes"))
dat$FCVC <- factor(dat$FCVC)
dat$NCP <- factor(dat$NCP)
dat$CAEC <- factor(dat$CAEC, levels = c("no","Sometimes","Frequently"))
dat$SMOKE <- factor(dat$SMOKE,levels = c("no","yes"))
dat$SCC <- factor(dat$SCC,levels = c("no","yes"))
dat$CALC <- factor(dat$CALC, levels = c("no","Sometimes","Frequently"))
dat$MTRANS <- factor(dat$MTRANS, levels = c("Walking/Bike", "Public_Transportation","Motor_Vehicle"))
 
## Remove weight variable and rename fam history
dat <- dat %>% dplyr::select(-Weight)
dat <- dat %>% rename(fam_hist = family_history_with_overweight)

frequentist.multinomial <- multinom(NObeyesdad ~., data = dat)
summary(frequentist.multinomial)


dat2 <- dat

h <- interaction(dat$FAF, dat$SCC)
table(h, dat$NObeyesdad)


# NObeyesdad as the response variable, convert to integer to use in Stan
dat$NObeyesdad <- as.integer(factor(dat$NObeyesdad)) 

# Create the model matrix for predictors, dropping the first level of each factor
#add gender*age and ncp*scc*faf*smoke

#predictors_matrix <- model.matrix(~ . -1 - NObeyesdad, data = dat)
predictors_matrix <- model.matrix(~ Gender+Age+Height+fam_hist+FAVC+FCVC+
                                    NCP+CAEC+SMOKE+CH2O+SCC+FAF+TUE+CALC+MTRANS+
                                    FAVC*NCP*FAF*CH2O-1, data = dat) #also try FAF*SMOKE in a different model
predictors_matrix <- predictors_matrix[,-1]
# Here, '-1' removes the intercept which effectively drops one level from each factor

# Check the structure to understand what was created
str(predictors_matrix)


# Preparing data for Stan
stan_data <- list(
  N = nrow(dat),         # number of observations
  K = length(unique(dat$NObeyesdad)), # number of levels in the outcome variable
  y = dat$NObeyesdad,    # outcome variable
  P = ncol(predictors_matrix), # number of predictors
  x = predictors_matrix  # predictor matrix
)


library(rstan)
library(coda)
library(bayesplot)

################################################################################
################################################################################
##################################### Stan #####################################
################################################################################
################################################################################
stan_model <- stan_model("C:/Users/Monica/OneDrive - Brown University/Desktop/PHP2530 Bayesian/Project/mutli.stan")

# Fit the model
fit <- sampling(stan_model, data = stan_data, 
                iter = 4000, warmup = 3000, 
                chains = 2,
                control = list(max_treedepth = 15))
print(fit)

################################################################################
################################################################################
######################## Plot Credible Intervals ###############################
################################################################################
################################################################################
library(reshape2)
beta_samples <- rstan::extract(fit,pars = "beta")
# Convert the 3D array to a 2D data frame (iterations * chains, predictors * categories)
n_iterations <- dim(beta_samples)[1]
n_predictors <- dim(beta_samples)[2]
n_categories <- dim(beta_samples)[3]


beta_df <- melt(beta_samples, varnames = c("Iteration", "Predictor", "Category"), value.name = "Value")

beta_df <- beta_df %>%
  mutate(parameter = paste0("Predictor", Predictor, "_Category", Category)) %>%
  select(-Predictor, -Category)

lower_quantiles <- beta_df %>%
  group_by(parameter) %>%
  summarise(lower = quantile(Value, probs = 0.025), .groups = 'drop')

upper_quantiles <- beta_df %>%
  group_by(parameter) %>%
  summarise(upper = quantile(Value, probs = 0.975), .groups = 'drop')

# Join lower and upper quantiles
significance_df <- lower_quantiles %>%
  left_join(upper_quantiles, by = "parameter") %>%
  mutate(significant = ifelse(lower > 0 | upper < 0, "significant", "not significant"))

plot_data <- beta_df %>%
  group_by(parameter) %>%
  summarise(estimate = mean(Value), .groups = 'drop') %>%
  left_join(significance_df, by = "parameter")

plot_data$parameter <- factor(plot_data$parameter, levels = unique(plot_data$parameter))

# Create the plot
ggplot(plot_data, aes(y = parameter, x = estimate)) +
  geom_errorbarh(aes(xmin = lower, xmax = upper, color = significant), height = 0.2) +
  geom_point(aes(color = significant)) +
  scale_color_manual(values = c("significant" = "red", "not significant" = "black")) +
  labs(title = " Parameter Estimates with 95% \n Credible Intervals (FAVC*NCP*FAF*CH2O)",
       x = "Estimate", y = "Parameter") +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 7))
#6x10
################################################################################
################################################################################
############################## Diagnostics Plots ###############################
################################################################################
################################################################################
#10x15
As.mcmc.list(fit, pars = c("beta")) %>% plot()
par(mfrow = c(5, 4))  # Adjust the layout to have multiple plots per page (adjust rows and columns as needed)
#10x10
acfplot(As.mcmc.list(fit, pars = c("beta")), lag.max = 30) 
As.mcmc.list(fit, pars = c("beta")) %>% acfplot()

As.mcmc.list(fit, pars = "beta") %>% gelman.diag()


################################################################################
################################################################################
############################### Plot Fitted VAls ###############################
################################################################################
################################################################################
# Extract generated quantities
generated_quantities <- extract(fit, pars = "predicted_probabilities")$predicted_probabilities
df <- as.data.frame(generated_quantities[1,,])
colnames(df) <- c("C1","C2","C3","C4")
N <- dim(df)[1]
df$Observation <- 1:N

# Convert to long format
long_df <- df %>%
  pivot_longer(cols = starts_with("C"),
               names_to = "Category",
               values_to = "Probability")

# Plot the predicted probabilities
ggplot(long_df, aes(x = Observation, y = Probability, color = Category)) +
  geom_line(alpha = 0.5) +
  labs(title = "Predicted Probabilities",
       x = "Observation",
       y = "Probability") +
  theme_minimal()


df$outcome <- dat$NObeyesdad
df <- df %>%
  mutate(predicted_class = apply(df[, c("C1", "C2", "C3", "C4")], 1, function(x) which.max(x)))
df$correct_prediction <- ifelse(df$predicted_class == df$outcome, TRUE, FALSE)
sum(df$correct_prediction)/N*100

conf_matrix <- confusionMatrix(data = as.factor(df$predicted_class),
                reference = as.factor(df$outcome))

res <- as.data.frame(conf_matrix$byClass)
res$Sensitivity


conf_matrix
cm_table <- as.data.frame(conf_matrix$table)
cm_long <- melt(cm_table)



results <- matrix(NA,nrow = 200,ncol = 4)
sample <- sample(1:2000,200)
acc <- rep(NA,200)
j <- 1
for (i in sample) {
  generated_quantities <- extract(fit, pars = "predicted_probabilities")$predicted_probabilities
  df <- as.data.frame(generated_quantities[i,,])
  colnames(df) <- c("C1", "C2", "C3", "C4")
  df$outcome <- dat$NObeyesdad
  
  df <- df %>%
    mutate(predicted_class = apply(df[, c("C1", "C2", "C3", "C4")], 1, function(x) which.max(x)))
  conf_matrix <- confusionMatrix(data = as.factor(df$predicted_class),
                                 reference = as.factor(df$outcome))
  
  res <- as.data.frame(conf_matrix$byClass)
  res$Sensitivity
  acc[j] <- conf_matrix$overall[1] ## accuracy
  results[j,] <- res$Specificity #Change as nessesary to calculate specificity etc
  print(j)
  j <- j+1
}
colMeans(results)
mean(acc)










# Create the heatmap
ggplot(cm_long, aes(x = Reference, y = Prediction, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "red") +
  geom_text(aes(label = value), color = "black") +
  theme_minimal() +
  labs(title = "Confusion Matrix Heatmap",
       x = "Actual Class",
       y = "Predicted Class",
       fill = "Count")




################################################################################
################################################################################
################################## NaiveBayes #################################
################################################################################
################################################################################
## Set WD
setwd("C:/Users/Monica/OneDrive - Brown University/Desktop/PHP2530 Bayesian/Project/")

## Load data
dat <- read.csv("ObesityDataSet_raw_and_data_sinthetic.csv")
dat <- dat[1:498,] #take original data before SMOTE


## Data Preprocessing (covariates)
dat$CALC[dat$CALC == "Always"] <- "Frequently"
dat$CAEC[dat$CAEC == "Always"] <- "Frequently"
dat$MTRANS[dat$MTRANS == "Motorbike"] <- "Motor_Vehicle"
dat$MTRANS[dat$MTRANS == "Automobile"] <- "Motor_Vehicle"
dat$MTRANS[dat$MTRANS == "Bike"] <- "Walking/Bike"
dat$MTRANS[dat$MTRANS == "Walking"] <- "Walking/Bike"

dat$NObeyesdad <- ifelse(dat$NObeyesdad %in% c("Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"), "Obese", dat$NObeyesdad)
dat$NObeyesdad <- ifelse(dat$NObeyesdad %in% c("Overweight_Level_I", "Overweight_Level_II"), "Overweight", dat$NObeyesdad)
dat$NObeyesdad[dat$NObeyesdad == "Insufficient_Weight"] <- "Underweight"

## Factor variables
dat$NObeyesdad <- factor(dat$NObeyesdad,levels = c("Underweight","Normal_Weight","Obese","Overweight"))
dat$Gender <- factor(dat$Gender, levels=c("Female","Male"))
dat$family_history_with_overweight <- factor(dat$family_history_with_overweight,levels = c("no","yes"))
dat$FAVC <- factor(dat$FAVC,levels = c("no","yes"))
dat$FCVC <- factor(dat$FCVC)
dat$NCP <- factor(dat$NCP)
dat$CAEC <- factor(dat$CAEC, levels = c("no","Sometimes","Frequently"))
dat$SMOKE <- factor(dat$SMOKE,levels = c("no","yes"))
dat$SCC <- factor(dat$SCC,levels = c("no","yes"))
dat$CALC <- factor(dat$CALC, levels = c("no","Sometimes","Frequently"))
dat$MTRANS <- factor(dat$MTRANS, levels = c("Walking/Bike", "Public_Transportation","Motor_Vehicle"))
dat$CH2O <- factor(dat$CH2O)
dat$FAF <- factor(dat$FAF)
dat$TUE <- factor(dat$TUE)



## Remove weight variable and rename fam history
dat <- dat %>% dplyr::select(-Weight)
dat <- dat %>% rename(fam_hist = family_history_with_overweight)



mod <- naive_bayes(NObeyesdad ~ ., data = dat,
                   usekernel = T,
                   laplace = T) 
summary(mod)

if (is.factor(dat$NCP)) {
  plot(mod, 'predictor_name')
} else {
  # For numeric predictors, you might need a different plotting approach
  hist(mod$tables$NCP, main = "Conditional Distributions of predictor_name",
       xlab = "predictor_name", col = rainbow(length(mod$tables$NCP)))
}

plot(mod,which = 3)
plot_grobs <- list()

# Generate each plot and store it
for (i in 1:15) {  # Assuming there are 2 plots you want to capture
  plot(mod, which = i)
  grid.echo()  # Echoes the current base plot to a grid plot
  plot_grobs[[i]] <- grid.grab() 
}

# Now use grid.arrange to combine these plots
combined_plot <- gridExtra::grid.arrange(grobs = plot_list, ncol = 4)

grid.arrange(plot(mod, which = 1),plot(mod, which = 2))

par(mfrow = c(3, 5))
plot(mod, which = 1)
plot(mod, which = 2)
plot(mod, which = 3)
plot(mod, which = 4)
plot(mod, which = 5)
plot(mod, which = 6)
plot(mod, which = 7)
plot(mod, which = 8)
plot(mod, which = 9)
plot(mod, which = 10)
plot(mod, which = 11)
plot(mod, which = 12)
plot(mod, which = 13)
plot(mod, which = 14)
plot(mod, which = 15)


par(mfrow = c(2, 2))
plot(mod, which = 1)
#plot(mod, which = 15)
plot(mod, which = 10) #se mira horizontal
plot(mod, which = 4)
plot(mod, which = 11)
