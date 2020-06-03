##############################################################

# Mercari Price Suggestion Challenge Data Science Project (Kaggle)
# Build a machine learning algorithm that automatically suggests the right product prices.

# Problem Statement: Mercari, Japan's biggest community-powered shopping app, knows this problem deeply. They'd like to offer pricing suggestions to sellers, but this is tough because their sellers are enabled to put just about anything, or any bundle of things, on Mercari's marketplace.
# Objective: In this machine learning project, we will build an algorithm that automatically suggests the right product prices. You'll be provided user-inputted text descriptions of their products, including details like product category name, brand name, and item condition.

##############################################################


###################################
# Setting the workspace envoirnment
#######################################

# clearing the work space envoirnment
rm(list = ls())

# Setting the working directory
setwd('C://Users//sukes//Desktop//Projects//product_price_prediction_NLP')

# Importing the libraries
library(data.table)
library(stringr)
library(ggplot2)
library(dplyr)
#install.packages('tm')
#install.packages('quanteda')
library(tm)
library(quanteda)


#######################################
# DATA SOURCING - Loading the data
########################################

train <- fread('train.tsv')

test <- fread('test.tsv')

paste('Train File Size in MB:',format(object.size(train),units='auto'))

paste('Test File Size in MB:',format(object.size(test),units='auto'))

############## Understanding the data ########################

paste('Looking into train and test sample data')

View(head(train))

View(head(test))

paste('Data strucutre')

dim(train)

dim(test)

paste('Data statistics and summarization')

summary(train)

summary(test)

##################################################
# EDA - EXPLORATORY DATA ANALYSIS

#1. what is the variation in prices?
#2. How good the products are?
#3. Comment of the shipping condition, how good the shipping condition is?
#4. What are the most expensive brands present?
#5. does expensive brands posses higher prices?
#6. How many product categories are there?
#7. Does the prices vary by product categories?

# Use hypothesis testing process and answer the questions.
##################################################


####################  what is the variation in prices? ##############################

range(train$price)

ggplot(data=train,aes(x=price)) + 
  geom_histogram(fill='red') + 
  labs(title='Histogram of Prices')

ggplot(data=train,aes(x=log(price))) + 
  geom_histogram(fill='red') + 
  labs(title='Histogram of Prices')

############################ #2. How good the products are (condition of the products)? ############

table(train$item_condition_id)

train[,.(count=.N),by=item_condition_id] %>%
  ggplot(mapping = aes(x=as.factor(item_condition_id),y=count)) +
  geom_bar(stat='identity',fill='blue')

# which condition id is superior in terms of price
train[,.(count=.N,price_mean=mean(price),price_median= median(price)),by=item_condition_id]

ggplot(data=train,mapping = aes(x=as.factor(item_condition_id),y=price)) +
  geom_boxplot(fill='orange',color='blue')

ggplot(data=train,mapping = aes(x=as.factor(item_condition_id),y=log(price))) +
  geom_boxplot(fill='orange',color='blue')

####################### how good the shipping condition is ####################

table(train$shipping)/nrow(train)

train %>%
  ggplot(aes(x=log(price+1),fill=factor(shipping))) +
  geom_density(alpha=0.5)

t.test(train$price~train$shipping)

##################### #4. What are the top 10 most expensive brands present? ################

View(head(train))

train[,.(med_price = median(price)), by= brand_name] %>%
  head(10) %>%
  ggplot(aes(x= reorder(brand_name,med_price),y=med_price)) +
  geom_point(color='red') +
  coord_flip() +
  scale_y_continuous(labels = scales::dollar)


train[,.(mean_price = mean(price)), by= brand_name] %>%
  head(10) %>%
  ggplot(aes(x= reorder(brand_name,mean_price),y=mean_price)) +
  geom_point(color='red') +
  coord_flip() +
  scale_y_continuous(labels = scales::dollar)


########################### #6. How many product categories are there? #####################

#6. How many product categories are there?

table(train$category_name)

unique(train$category_name)

length(unique(train$category_name))

sort(table(train$category_name),decreasing = T)[1:10]


################### Feature ENgineerng #################################

# splitting the columns based on regex '/'
train[,c('level1_cat','level2_cat') := tstrsplit(train$category_name,split='/',keep = c(1,2))]

head(train[,c('level1_cat','level2_cat')])

table(train$level1_cat)

table(train$level2_cat)


###############################
# Modelling - 1
###############################


############################# We cannot use regression/ feature engineering must ####################

fit <- lm(price~factor(item_condition_id)+factor(shipping),data=train)
summary(fit)

fit <- lm(price~factor(item_condition_id)+
            factor(shipping)+
            factor(level1_cat)+
            factor(level2_cat),data=train)
summary(fit)


# we cannot train a regression based model since the R-square value is less than 1%
# we need to create better features
# calculate the item description length

##########################################
# Feature Engineering
# NLP

# Item description Length - 1
# Text Corpus  from Text Mining - 2 (NLP)
########################################

#################3 1- Item description ###########################

train[,desc_length:=nchar(item_description)]

summary(train$desc_length)

cor(train$desc_length,train$price)

cor.test(train$desc_length,train$price)

#################### 2 - Text Corpus from text mining ##########################

train[item_description=='No Description Found', item_description:= NA]

dcorpus <- corpus(train$item_description)

summary(dcorpus)[1:5,]

########### Unigram Approach #############

# term-document frequency

dfm1 <- tokens(dcorpus)%>%
        tokens_ngrams(n=1)%>%
        dfm(remove = c('rm',stopwords('english')),
            remove_punct = T,
            remove_numbers=T,
            stem=T)
# N-gram approach to extract the features (ONly use top features)

tf <- topfeatures(dfm1,n=30)


# Let's visulize the top features
data.frame(term=names(tf),freq = unname(tf)) %>%
  ggplot(aes(x=reorder(term,freq),y=freq/1000))+
  geom_bar(stat='identity',fill='red') +
  coord_flip()

# let's visuliaze the word cloud represemtation
textplot_wordcloud(dfm1,min.freq=2,rot.per=0.2)


########### Bi-gram Approach ###############

# term-document frequency
dfm2 <- tokens(dcorpus,remove_punct = T,remove_numbers = T )%>%
        tokens_remove(pattern=stopwords('en')) %>%
        tokens_ngrams(n=2)%>%
        dfm(stem=T)

tf <- topfeatures(dfm2,n=30)

data.frame(term=names(tf),freq = unname(tf)) %>%
  ggplot(aes(x=reorder(term,freq),y=freq/1000))+
  geom_bar(stat='identity',fill='red') +
  coord_flip()




  
