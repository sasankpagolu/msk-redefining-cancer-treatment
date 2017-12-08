install.packages('ggplot2') # visualization
install.packages('ggthemes') # visualization
install.packages('scales') # visualization
install.packages('grid') # visualisation
install.packages('gridExtra') # visualisation
install.packages('corrplot') # visualisation
install.packages('ggfortify') # visualisation
install.packages('ggraph') # visualisation
install.packages('igraph') # visualisation
install.packages('dplyr') # data manipulation
install.packages('readr') # data input
install.packages('tibble') # data wrangling
install.packages('tidyr') # data wrangling
install.packages('stringr') # string manipulation
install.packages('forcats') # factor manipulation
install.packages('tidytext') # text mining
install.packages('SnowballC') # text analysis
install.packages('wordcloud') # test visualisation


library('ggplot2') # visualization
library('ggthemes') # visualization
library('scales') # visualization
library('grid') # visualisation
library('gridExtra') # visualisation
library('corrplot') # visualisation
library('ggfortify') # visualisation
library('ggraph') # visualisation
library('igraph') # visualisation
library('dplyr') # data manipulation
library('readr') # data input
library('tibble') # data wrangling
library('tidyr') # data wrangling
library('stringr') # string manipulation
library('forcats') # factor manipulation
library('tidytext') # text mining
library('SnowballC') # text analysis
library('wordcloud') # test visualisation

#read in variants files
train <- read_csv('training_variants')
test  <- read_csv('test_variants')

#read in text files
train_txt_dump <- tibble(text = read_lines('training_text', skip = 1))
train_txt <- train_txt_dump %>%
  separate(text, into = c("ID", "txt"), sep = "\\|\\|")
train_txt <- train_txt %>%
  mutate(ID = as.integer(ID))

test_txt_dump <- tibble(text = read_lines('test_text', skip = 1))
test_txt <- test_txt_dump %>%
  separate(text, into = c("ID", "txt"), sep = "\\|\\|")
test_txt <- test_txt %>%
  mutate(ID = as.integer(ID))

train <- train %>%
  mutate(Gene = factor(Gene),
         Variation = factor(Variation),
         Class = factor(Class))

test <- test %>%
  mutate(Gene = factor(Gene),
         Variation = factor(Variation))

head(train_txt,n=1)

#glimpse of the data
glimpse(train)
glimpse(test)


# Most frequent Gene and variations
train %>%
  group_by(Gene) %>%
  summarise(ct = n()) %>%
  arrange(desc(ct))
test %>%
  group_by(Gene) %>%
  summarise(ct = n()) %>%
  arrange(desc(ct))

train %>%
  group_by(Variation) %>%
  summarise(ct = n()) %>%
  arrange(desc(ct))

test %>%
  group_by(Variation) %>%
  summarise(ct = n()) %>%
  arrange(desc(ct))

#frequency distribution of most frequent Gene values
most_freq_gene <- train %>%
  group_by(Gene) %>%
  summarise(ct = n()) %>%
  filter(ct > 40)

most_freq_gene %>%
  ggplot(aes(reorder(Gene, -ct, FUN = min), ct)) +
  geom_point(size = 4) +
  labs(x = "Gene", y = "Frequency") +
  coord_flip()

most_freq_gene_test <- test %>%
  group_by(Gene) %>%
  summarise(ct = n()) %>%
  filter(ct > 40)

most_freq_gene_test %>%
  ggplot(aes(reorder(Gene, -ct, FUN = min), ct)) +
  geom_point(size = 4) +
  labs(x = "Gene", y = "Frequency") +
  coord_flip()

#frequency distribution of variations
train_1 <- train %>% mutate(set = factor("train")) %>% select(-Class, -ID)
test_1 <- test %>% mutate(set = factor("test")) %>% select(-ID)

train_1 <- full_join(train_1,test_1)

train_1 %>%
  group_by(Variation, set) %>%
  summarise(ct = n()) %>%
  filter(ct > 3) %>%
  ggplot(aes(reorder(Variation, -ct, FUN = median), ct, colour = set)) +
  geom_point(size = 4) +
  coord_cartesian(ylim = c(0, 100)) +
  labs(x = "Variation", y = "Frequency")

#Class distribution
train %>%
  ggplot(aes(Class)) +
  geom_bar(fill = "darkgreen")


#Text files
#length of text files
train_txt <- train_txt %>%
  mutate(txt_len = str_length(txt),
         set = "train")
mean(train_txt$txt_len)
test_txt <- test_txt %>%
  mutate(txt_len = str_length(txt),
         set = "test")

total_txt <- full_join(train_txt,test_txt)

total_txt %>%
  ggplot(aes(txt_len, fill = set)) +
  geom_histogram(bins = 50) +
  labs(x = "Length of text entry")

#length by class
train_2 <- train_txt %>%
  select(ID, txt_len)
test_2 <- train %>%
  select(ID, Class)

full_join(train_2, test_2, by = "ID") %>%
  ggplot(aes(txt_len)) +
  geom_density(fill = "purple", bw = 5e3) +
  labs(x = "Length of text entry") +
  facet_wrap(~ Class)


#missing values in text
total_txt %>%
  filter(txt_len < 100)

train_3 <- train_txt %>% select(ID, txt) %>% unnest_tokens(word, txt)


#stopword removal  
data("stop_words")
my_stopwords <- data_frame(word = c(as.character(1:100),
                                    "fig", "figure", "et", "al", "table",
                                    "data", "analysis", "analyze", "study",
                                    "method", "result", "conclusion", "author",
                                    "find", "found", "show", "perform",
                                    "demonstrate", "evaluate", "discuss"))
train_3 <- train_3 %>%
  anti_join(stop_words, by = "word") %>%
  anti_join(my_stopwords, by = "word") %>%
  filter(str_detect(word, "[a-z]"))

#frequencies of words
train_3 %>%
  count(word) %>%
  filter(n > 5e4) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n)) +
  geom_col() +
  xlab(NULL) +
  coord_flip()

#considering word stems
train_3 <- train_3 %>%
  mutate(word = wordStem(word))

train_3 %>%
  count(word) %>%
  filter(n > 5e4) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n)) +
  geom_col() +
  xlab(NULL) +
  coord_flip()

#word cloud
train_3 %>% 
  count(word) %>%
  with(wordcloud(word, n, max.words = 100))

#tf-idf
train_4 <- train %>%
  select(ID, Class)

train_3_class <- full_join(train_3, train_4, by = "ID")
frequency <-train_3_class %>%
  count(Class, word)
tf_idf <- frequency %>%
  bind_tf_idf(word, Class, n)

#frequency of most characteristic words
tf_idf %>%
  arrange(desc(tf_idf)) %>%
  mutate(word = factor(word, levels = rev(unique(word)))) %>%
  top_n(20, tf_idf) %>%
  ggplot(aes(word, tf_idf, fill = Class)) +
  geom_col() +
  labs(x = NULL, y = "tf-idf") +
  coord_flip()

#most characteristic words by class
tf_idf %>%
  arrange(desc(tf_idf)) %>%
  mutate(word = factor(word, levels = rev(unique(word)))) %>%
  group_by(Class) %>%
  top_n(10, tf_idf) %>%
  ungroup() %>%  
  ggplot(aes(word, tf_idf, fill = Class)) +
  geom_col() +
  labs(x = NULL, y = "tf-idf") +
  theme(legend.position = "none") +
  facet_wrap(~ Class, ncol = 3, scales = "free") +
  coord_flip()


