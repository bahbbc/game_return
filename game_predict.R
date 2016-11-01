library(rpart)
library(randomForest)

base <- read.csv('~/workspace/games/training_set.csv', na.strings = 'NULL')
summary(base)
last_moment_normalized <- as.POSIXct(base$last_moment, origin="1970-01-01")

# normalize float vars

base$revenue <- as.double(as.character(base$revenue))
base$unit_count <-  as.double(as.character(base$unit_count))
base$try_1 <- as.integer(as.character(base$try_1))
base$win_1 <- as.integer(as.character(base$win_1))
base$try_2 <- as.integer(as.character(base$try_2))
base$win_2 <- as.integer(as.character(base$win_2))
base$try_3 <- as.integer(as.character(base$try_3))
base$win_3 <- as.integer(as.character(base$win_3))
base$ttp <- as.integer(as.character(base$ttp))
base$session_count <- as.integer(as.character(base$session_count))

# change some NAs to 0

base$try_1[is.na(base$try_1)] <- 0
base$win_1[is.na(base$win_1)] <- 0

base$try_2[is.na(base$try_2)] <- 0
base$win_2[is.na(base$win_2)] <- 0

base$try_3[is.na(base$try_3)] <- 0
base$win_3[is.na(base$win_3)] <- 0

base$session_count[is.na(base$session_count)] <- 0
base$ttp[is.na(base$ttp)] <- 0
base$unit_count[is.na(base$unit_count)] <- 0
base$revenue[is.na(base$revenue)] <- 0

base$country <- as.character(base$country)
base$country[is.na(base$revenue)] <- 'no_data'
base$country <- as.factor(base$country)

base$last_hour_played <- as.integer(format(last_moment_normalized, '%H'))
base$last_hour_played[is.na(base$last_hour_played)] <- 999

#exploratory analysis

#Country

barplot(table(base$country), main = 'Country')

# 

barplot(table(base$device), main = 'Device')

barplot(table(base$session_count), main = 'Session count')

plot(table(base$ttp), main = 'Total time played')

barplot(table(base$unit_count), main = '#Purchases')

# extrair hora do last_moment_normalized

barplot(table(as.double(base$revenue)), main = '#Revenue')

plot(base$try_1, base$win_1, main='1')
cor(base$try_1, base$win_1)

plot(base$try_2, base$win_2, main='2')
cor(base$try_2, base$win_2)

plot(base$try_3, base$win_3, main='3')
cor(base$try_3, base$win_3)

# data partition
# generate validation set

base$win_rate_1 <- (base$win_1/base$try_1)
base$win_rate_1[is.nan(base$win_rate_1)] <- 0
base$win_rate_1[is.infinite(base$win_rate_1)] <- 0
base$win_rate_1 <- base$win_rate_1 *100

base$high_win_rate_1 <- base$win_rate_1 >= 0.5

base$win_rate_2 <- (base$win_2/base$try_2)
base$win_rate_2[is.nan(base$win_rate_2)] <- 0
base$win_rate_2[is.infinite(base$win_rate_2)] <- 0
base$win_rate_2 <- base$win_rate_2 *100

base$high_win_rate_2 <- base$win_rate_2 >= 0.5

base$win_rate_3 <- (base$win_3/base$try_3)
base$win_rate_3[is.nan(base$win_rate_3)] <- 0
base$win_rate_3[is.infinite(base$win_rate_3)] <- 0
base$win_rate_3 <- base$win_rate_3 *100

base$high_win_rate_3 <- base$win_rate_3 >= 0.5

base$less_than_three_sessions <- base$session_count < 3
base$more_than_half_wins <- ifelse( base$high_win_rate_1 > 0, TRUE, ifelse(base$high_win_rate_2 > 0, TRUE, ifelse(base$high_win_rate_3, TRUE, FALSE)))

base$play_more_one_hour <- base$ttp > 1800

set.seed(78)
sample_size <- floor(0.7 * length(base[,1]))
training_data_rows <- sample(seq_len(nrow(base)), size = sample_size)
train_data <- base[training_data_rows,]
test_data <- base[-training_data_rows,]
base_list <- list(train_data, test_data)
train_set <- base_list[[1]]
test_set <- base_list[[2]]

MSE = function(m, o){
  mean((m - o)^2)
}

fit <- lm(days_active ~ play_more_one_hour + more_than_half_wins + country + device + session_count * ttp + less_than_three_sessions + unit_count + revenue + high_win_rate_2 + win_rate_1 + win_rate_2 + win_rate_3, data = train_set)
summary(fit)

test_set$days_predict <- round(predict(fit, test_set))

MSE(test_set$days_predict, test_set$days_active)

all_fit <- lm(days_active ~ more_than_half_wins + less_than_five_sessions + last_hour_played + country + device + session_count * ttp + unit_count + revenue + try_1 + try_2 + try_3 + win_1 + win_2 + win_3 + win_rate_1 + win_rate_2 + win_rate_3 + high_win_rate_1 + high_win_rate_2 + high_win_rate_3, data = train_set)
summary(best_fit)

step <- stepAIC(all_fit, direction="both")
step$anova # display results 

best_fit <- lm(days_active ~ less_than_five_sessions + country + device + session_count + ttp + unit_count + try_1 + win_2 + win_3 + win_rate_2 + win_rate_3 + high_win_rate_1 + session_count*ttp, data = train_set)
summary(best_fit)

test_set$days_predict <- round(predict(best_fit, test_set))

MSE(test_set$days_predict, test_set$days_active)

plot(test_set$days_predict, test_set$days_active)

plot(table(test_set$days_predict))

plot(table(test_set$days_active))

tail(test_set[test_set$days_active > 7,])

###############

tree_fit <- randomForest(days_active ~ country + device + session_count + ttp + unit_count + revenue + try_1 + try_2 + try_3 + win_1 + win_2 + win_3, data = train_set, importance=TRUE, na.action = na.omit)
print(tree_fit)
round(importance(tree_fit), 2)
summary(tree_fit)

test_set$tree_days_predict <- round(predict(tree_fit, test_set))

MSE(test_set$tree_days_predict, test_set$days_active)

plot(table(test_set$tree_days_predict))

plot(test_set$tree_days_predict, test_set$days_active)

# verificar a media do conj. de teste original