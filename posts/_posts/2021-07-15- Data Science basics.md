* Tree how works
  * Random forest
  * XG_boost
* Feature Selection
  * Filter Methods
  1. Linear Discrimination analysis
  2. ANOVA
  3. Chi-Square

  * Wrapper methods
  1. Forward method: test one and add them until good one come
  2. Backward selection: remove them see how it works
  3. Recursive Feature Elimination: Recursively looks through different features and see how they pair together.

* Missing value dealing
  * if big data remove
  * if small data replace with context

* How dimensionality reduction helps
  * convert data into smaller dimension by conveying almost same information.
  * helps visualisation, storage, computational time, redundant features removal.

* Eigenvalues and eigenvectors

* How should you maintain and deployed model?

 1. Evaluate
 2. Compare
 3. Rebuild

* Recommendation system  

 1. Collaborative Filtering
 2. Content based Filtering

* Kmeans clustering select k

  * Elbow method

* P-value
  * p> 0.05 -accept null hypothesis
  * p< 0.05 -reject null hypothesis
  * p = 0.05 - can be either way

* How can time series can be stationary?
  * don't change over time

  * Accuracy

    * $$\frac{(True positive + True Negative)}{ (total observation)}$$
  * Precision
    * $$\frac{(True positive)}{(True positive + False positive)}$$
  * Recall
    * $$\frac{(True positive)}{(True positive + False negative)}$$

* Your organization has a website where visitors randomly receive one of two coupons. It is also possible that visitors to the website will not receive a coupon. You have been asked to determine if offering a coupon to website visitors has any impact on their purchase decisions. Which analysis method should you use
  * One way ANOVA

* What is the purpose of/B testing ?
  * This is statistical hypothesis testing for randomized experiments with two variables, A and B. The objective of/B testing is to detect any changes to a web page to maximize or increase the outcome of a strategy.

* What is the law of large numbers?
  * It is a theorem that describes the result of performing the same experiment very frequently. This theorem forms the basis of frequency-style thinking. It states that the sample mean, sample variance, and sample standard deviation converge to what they are trying to estimate.

* What are the confounding variables?
  * These are extraneous variables in a statistical model that correlates directly or inversely with both the dependent and the independent variable. The estimate fails to account for the confounding factor.

* What are eigenvalue and eigenvector?
  * Eigenvalues are the directions along which a particular linear transformation acts by flipping, compressing, or stretching.

 Eigenvectors are for understanding linear transformations. In data analysis, we usually calculate the eigenvectors for a correlation or covariance matrix.

* What is star schema?
  * It is a traditional database schema with a central table. Satellite tables map IDs to physical names or descriptions and can be connected to the central fact table using the ID fields; these tables are known as lookup tables and are principally useful in real-time applications, as they save a lot of memory. Sometimes, star schemas involve several layers of summarization to recover information faster.

* Why is resampling done?
  * Resampling is done in any of these cases:

 Estimating the accuracy of sample statistics by using subsets of accessible data, or drawing randomly with replacement from a set of data points
 Substituting labels on data points when performing significance tests
 Validating models by using random subsets (bootstrapping, cross-validation

* What are the types of biases that can occur during sampling?
  * Selection bias: Selection bias, in general, is a problematic situation in which error is introduced due to a non-random population sample.

  * Undercoverage bias
  * Survivorship bias: Survivorship bias is the logical error of focusing on aspects that support surviving a process and casually overlooking those that did not because of their lack of prominence. This can lead to wrong conclusions in numerous ways

* Regularization
  * L1 (lasso) - L1 more robust but unstable
  * L2 (Ridge) - less robust but stable

* Cross validation
  * Secure generalization. simple was split into test, trian data

* Explain what a false positive and a false negative are. Why is it important these from each other? Provide examples when false positives are more important than false negatives, false negatives are more important than false positives and when these two types of errors are equally important?
  * False positive: models says positive but actually not positive
 False negative : model says negative but actually positive

 canceer detection : False negative important
 porn classifier: Model says not porn but porn dangerous for children. (False negative) dangerous

* Assume you need to generate a predictive model using multiple regression. Explain how you intend to validate this model ?
  * Adjusted R-squared.

 R Squared is a measurement that tells you to what extent the proportion of variance in the dependent variable is explained by the variance in the independent variables. In simpler terms, while the coefficients estimate trends, R-squared represents the scatter around the line of best fit.
However, every additional independent variable added to a model always increases the R-squared value — therefore, a model with several independent variables may seem to be a better fit even if it isn’t. This is where adjusted R² comes in. The adjusted R² compensates for each additional independent variable and only increases if each given variable improves the model above what is possible by probability. This is important since we are creating a multiple regression model.

* Cross validation

* curse of dimensionality

  * The common theme of these problems is that when the dimensionality increases, the volume of the space increases so fast that the available data become sparse. This sparsity is problematic for any method that requires statistical significance. In order to obtain a statistically sound and reliable result, the amount of data needed to support the result often grows exponentially with the dimensionality. Also, organizing and searching data often relies on detecting areas where objects form groups with similar properties; in high dimensional data, however, all objects appear to be sparse and dissimilar in many ways, which prevents common data organization strategies from being efficient.

* Principle Component analysis
  * In its simplest sense, PCA involves project higher dimensional data (eg. 3 dimensions) to a smaller space (eg. 2 dimensions). This results in a lower dimension of data, (2 dimensions instead of 3 dimensions) while keeping all original variables in the model.
PCA is commonly used for compression purposes, to reduce required memory and to speed up the algorithm, as well as for visualization purposes, making it easier to summarize data.

* Why is Naive Bayes so bad? How would you improve a spam detection algorithm that uses naive Bayes
  * One major drawback of Naive Bayes is that it holds a strong assumption in that the features are assumed to be uncorrelated with one another, which typically is never the case.
One way to improve such an algorithm that uses Naive Bayes is by decorrelating the features so that the assumption holds true.

* Difference between random forest and XGboost ?
  * Random forest take all trees average at the end of training
  * XG boost take sample tree and prediction error is used for next trianing.

  * Unlike Randome forest, XG boost is more sensitive to selection of hyper-parameter

* Why is mean square error a bad measure of model performance? What would you suggest instead?
  * Mean Squared Error (MSE) gives a relatively high weight to large errors — therefore, MSE tends to put too much emphasis on large deviations. A more robust alternative is MAE (mean absolute deviation).

* What are the assumptions required for linear regression? What if some of these assumptions are violated?
  * The sample data used to fit the model is representative of the population
 The relationship between X and the mean of Y is linear
 The variance of the residual is the same for any value of X (homoscedasticity)
 Observations are independent of each other
 For any value of X, Y is normally distributed.
  * Extreme violations of these assumptions will make the results redundant. Small violations of these assumptions will result in a greater bias or variance of the estimate.

* What is collinearity and what to do with it? How to remove multicollinearity
  * Multicollinearity exists when an independent variable is highly correlated with another independent variable in a multiple regression equation. This can be problematic because it undermines the statistical significance of an independent variable.
You could use the Variance Inflation Factors (VIF) to determine if there is any multicollinearity between independent variables — a standard benchmark is that if the VIF is greater than 5 then multicollinearity exists.

* How to check if the regression model fits the data well ?
  * R-squar/Adjusted R-squared: Relative measure of fit. This was explained in a previous answer
 F1 Score: Evaluates the null hypothesis that all regression coefficients are equal to zero vs the alternative hypothesis that at least one doesn’t equal zero
 RMSE: Absolute measure of fit.

* What is a kernel? Explain the kernel trick ?
  * A kernel is a way of computing the dot product of two vectors 𝐱x and 𝐲y in some (possibly very high dimensional) feature space, which is why kernel functions are sometimes called “generalized dot product
  * The kernel trick is a method of using a linear classifier to solve a non-linear problem by transforming linearly inseparable data to linearly separable ones in a higher dimension.

  <img src='/images/kernel.png'>

* Is it beneficial to perform dimensionality reduction before fitting an SVM? Why or why not?
  * When the number of features is greater than the number of observations, then performing dimensionality reduction will generally improve the SVM.

* What is overfitting
  * Overfitting is an error where the model ‘fits’ the data too well, resulting in a model with high variance and low bias. As a consequence, an overfit model will inaccurately predict new data points even though it has a high accuracy on the training data.

# Statistics, Probability, and Mathematics

* The probability that item an item at location A is 0.6, and 0.8 at location B. What is the probability that item would be found on Amazon website
  * We need to make some assumptions about this question before we can answer it. Let’s assume that there are two possible places to purchase a particular item on Amazon and the probability of finding it at location A is 0.6 and B is 0.8. The probability of finding the item on Amazon can be explained as so:

  We can reward the above as P(A) = 0.6 and P(B) = 0.8. Furthermore, let’s assume that these are independent events, meaning that the probability of one event is not impacted by the other. We can then use the formula…
  $$ p(A or B) = P(A) + P(B)- P(A and B)$$
  $$ p(A or B) = 0.6 + 0.8 - (0.6 * 0.8)$$

* You randomly draw a coin from 100 coins — 1 unfair coin (head-head), 99 fair coins (head-tail) and roll it 10 times. If the result is 10 heads, what is the probability that the coin is unfair?
  * Probability of unfair coin is $P(A)$.
    Probability of 10 time head is $P(B)$.

    So we need to find out $$P(A|B)$$

    from Bayes theorem
    $$ P(A|B) = \frac{P(A)* P(B|A)}{P(B|A)* p(A)+ P(B| notA)P(not A)}$$
    
    $P(B|A)$ is the probability of 10 head when the coin is unfair. it is 1. because if the coin is unfair, then everytime 10 head will come. so it is 1.
    $P(not A)$ is 0.99
    $P(B|not A)$ is $0^{510}$.
    Answer $P(A|B) is 0.918$
  
* Difference between convex and non-convex cost function; what does it mean when a cost function is non-convex
  * Convex function - only one global minima
  * non convex function - possibility more than one minima, possibility to land on local minima

* Eight rules of Probability

  1. Maximum probability is 1 of an event.
  2. Sum of all events probability is 1.
  3. $P(not A) = 1 - P(A)$
  4. $P(A or B) = P(A) + P(B)$ when disjoint or mutually exclusive
  5. $P(A or B) = P(A) + P(B) - P(A and B)$ 
  6. $P(A and B) = P(A) * P(B)$ when $A$ and $B$ independent
  7. $P(A and B) = P(A) * P(B\|A)$ 
  8. $P(A\|B) = P(A and B)| P(B)$

* Describe Markov chain
  * A Markov chain is a stochastic model describing a sequence of possible events in which the probability of each event depends only on the state attained in the previous event

*  A box has 12 red cards and 12 black cards. Another box has 24 red cards and 24 black cards. You want to draw two cards at random from one of the two boxes, one card at a time. Which box has a higher probability of getting cards of the same color and why
  * The box with 24 red cards and 24 black cards has a higher probability of getting two cards of the same color. Let’s walk through each step.
  * For example if we pick first card red from both the box. Then for first box to pick another red is $\frac{11}{(11 + 12)}$, in case of second box the probability is $\frac{23}{23 + 24}$
  * Since $\frac{23}{23 + 24} > \frac{11}{11 + 12}$ therefore second box has better probability to pick same card.

* You are at a Casino and have two dices to play with. You win $10 every time you roll a 5. If you play till you win and then stop, what is the expected payout?
  * if we think for each game to play we need to spend $5
  * We have 2 dices, which have 36 combination.
  * possibility of adding 5 is 4.
  * So we have a probability =$\frac{4}{36}$ = $\frac{1}{9}$
  * Theoratically 9 times needs to be play to win a game. 9 times cost $$-9*5\\$  =-35\\$ $$

* How can you tell if a given coin is biased
  * Hypothesis testing
  * Null hypothesis, the coin is not biased.
  * The probability of getting head is 0.5, alternative hyposthesis would be p!=5
  * Flip coin 500 times.
  * Calcualte Z statistics, if sample number is less than 30, then perform t statistics
  * Compare against $\alpha$ two tailed $\frac{05}{2}$ =$0.025$
  * If p value is less than $\alpha$ then reject null hypothesis.

* Make an unfair coin fair
  Since a coin flip is a binary outcome, you can make an unfair coin fair by flipping it twice. If you flip it twice, there are two outcomes that you can bet on: heads followed by tails or tails followed by heads.
  $ P(heads) * P(Tails) = P(tails) * P(heads)$
  * This makes sense since each coin toss is an independent event. This means that if you get heads → heads or tails → tails, you would need to reflip the coin.

* You are about to get on a plane to London, you want to know whether you have to bring an umbrella or not. You call three of your random friends and ask each one of them if it’s raining. The probability that your friend is telling the truth is/3 and the probability that they are playing a prank on you by lying is/3. If all 3 of them tell that it is raining, then what is the probability that it is actually raining in London
  *  Assume the Probability or raining is 0.25
  * Assume The probability of raining is P(A) = 0.25
  * The question is Probability of raining when three friends are telling it is raining P(A|B)
  * $$ P(A|B) = \frac{P(B|A)* P(A)}{P(B)}$$
  or 
  * $$ P(A|B) = \frac{P(B|A)* P(A)}{P(B|A)* P(A) +P(B|not A)* P(not A) }$$

* You are given 40 cards with four different colors- 10 Green cards, 10 Red Cards, 10 Blue cards, and 10 Yellow cards. The cards of each color are numbered from one to ten. Two cards are picked at random. Find out the probability that the cards picked are not of the same number and same color.


  * Since these events are not independent, we can use the rule:
  $$ P(A and B) = P(A) * P(B|A) $$
  which is also equal to
  $$P(not A and not B) = P(not A) * P(not B | not A)$$
  For example:
  $$P(not 4 and not yellow) = P(not 4) * P(not yellow | not 4)$$
  $$P(not 4 and not yellow) = (/39) * (/36)$$
  $$P(not 4 and not yellow) = 0.692$$
  Therefore, the probability that the cards picked are not the same number and the same color is 69.2%.

*  How do you assess the statistical significance of an insight
  * You would perform hypothesis testing to determine statistical significance. First, you would state the null hypothesis and alternative hypothesis. Second, you would calculate the p-value, the probability of obtaining the observed results of a test assuming that the null hypothesis is true. Last, you would set the level of the significance (alpha) and if the p-value is less than the alpha, you would reject the null — in other words, the result is statistically significant.

* Explain what a long-tailed distribution is and provide three examples of relevant phenomena that have long tails. Why are they important in classification and regression problems?

   <img src='/images/long_tailed.png'>
  * pareto principle (/20 rule)
  * Best selling vs other product sales
  * Power law

  ** Majority of the population lies in least frequently occuring values
  ** Needs to be change how to deal with outliers
  ** Some machine learning assumption that population is normally distributed is also changed.

* What is the Central Limit Theorem? Explain it. Why is it important?
  * The central limit theorem states that the sampling distribution of the sample mean approaches a normal distribution as the sample size gets larger no matter what the shape of the population distribution.” [1]
  The central limit theorem is important because it is used in hypothesis testing and also to calculate confidence intervals.

* What is the statistical power?
  * ‘Statistical power’ refers to the power of a binary hypothesis, which is the probability that the test rejects the null hypothesis given that the alternative hypothesis is true. 
  $$ Power = P(reject Null hypothesis| alternative hypothesis is true)$$


* Explain selection bias (with regard to a dataset, not variable selection). Why is it important? How can data management procedures such as missing data handling make it worse?
  * 
  * Selection bias is the phenomenon of selecting individuals, groups or data for analysis in such a way that proper randomization is not achieved, ultimately resulting in a sample that is not representative of the population.
  Understanding and identifying selection bias is important because it can significantly skew results and provide false insights about a particular population group.
  Types of selection bias include:
  sampling bias: a biased sample caused by non-random sampling
  time interval: selecting a specific time frame that supports the desired conclusion. e.g. conducting a sales analysis near Christmas.
  exposure: includes clinical susceptibility bias, protopathic bias, indication bias. Read more here.
  data: includes cherry-picking, suppressing evidence, and the fallacy of incomplete evidence.
  attrition: attrition bias is similar to survivorship bias, where only those that ‘survived’ a long process are included in an analysis, or failure bias, where those that ‘failed’ are only included
  observer selection: related to the Anthropic principle, which is a philosophical consideration that any data we collect about the universe is filtered by the fact that, in order for it to be observable, it must be compatible with the conscious and sapient life that observes it. 
  Handling missing data can make selection bias worse because different methods impact the data in different ways. For example, if you replace null values with the mean of the data, you adding bias in the sense that you’re assuming that the data is not as spread out as it might actually be.

* Provide a simple example of how an experimental design can help answer a question about behavior. How does experimental data contrast with observational data?
  * Observational data comes from observational studies which are when you observe certain variables and try to determine if there is any correlation.
  Experimental data comes from experimental studies which are when you control certain variables and hold them constant to determine if there is any causality.
  An example of experimental design is the following: split a group up into two. The control group lives their lives normally. The test group is told to drink a glass of wine every night for 30 days. Then research can be conducted to see how wine affects sleep.

* Is mean imputation of missing data acceptable practice? Why or why not?
  *Mean imputation is the practice of replacing null values in a data set with the mean of the data.
  Mean imputation is generally bad practice because it doesn’t take into account feature correlation. For example, imagine we have a table showing age and fitness score and imagine that an eighty-year-old has a missing fitness score. If we took the average fitness score from an age range of 15 to 80, then the eighty-year-old will appear to have a much higher fitness score that he actually should.
  Second, mean imputation reduces the variance of the data and increases bias in our data. This leads to a less accurate model and a narrower confidence interval due to a smaller variance. 

* What is an outlier? Explain how you might screen for outliers and what would you do if you found them in your dataset. Also, explain what an inlier is and how you might screen for them and what would you do if you found them in your dataset.
  *

* How do you handle missing data? What imputation techniques do you recommend?
  There are several ways to handle missing data:
  Delete rows with missing data
  Me/Medi/Mode imputation
  Assigning a unique value
  Predicting the missing values
  Using an algorithm which supports missing values, like random forests
  The best method is to delete rows with missing data as it ensures that no bias or variance is added or removed, and ultimately results in a robust and accurate model. However, this is only recommended if there’s a lot of data to start with and the percentage of missing values is low.

* Explain likely differences between administrative datasets and datasets gathered from experimental studies. What are likely problems encountered with administrative data? How do experimental methods help alleviate these problems? What problem do they bring?
  Administrative datasets are typically datasets used by governments or other organizations for non-statistical reasons.
  Administrative datasets are usually larger and more cost-efficient than experimental studies. They are also regularly updated assuming that the organization associated with the administrative dataset is active and functioning. At the same time, administrative datasets may not capture all of the data that one may want and may not be in the desired format either. It is also prone to quality issues and missing entries.

*  You are compiling a report for user content uploaded every month and notice a spike in uploads in October. In particular, a spike in picture uploads. What might you think is the cause of this, and how would you test it?
  * There are a number of potential reasons for a spike in photo uploads:
  A new feature may have been implemented in October which involves uploading photos and gained a lot of traction by users. For example, a feature that gives the ability to create photo albums.
  Similarly, it’s possible that the process of uploading photos before was not intuitive and was improved in the month of October.
  There may have been a viral social media movement that involved uploading photos that lasted for all of October. Eg. Movember but something more scalable.
  It’s possible that the spike is due to people posting pictures of themselves in costumes for Halloween.
  The method of testing depends on the cause of the spike, but you would conduct hypothesis testing to determine if the inferred cause is the actual cause.

* Give examples of data that does not have a Gaussian distribution, nor log-normal.
  * Any type of categorical data won’t have a gaussian distribution or lognormal distribution.
Exponential distributions — eg. the amount of time that a car battery lasts or the amount of time until an earthquake occurs.

* What is root cause analysis? How to identify a cause vs. a correlation? Give examples

* What is the Law of Large Numbers?
  * The Law of Large Numbers is a theory that states that as the number of trials increases, the average of the result will become closer to the expected value.
  Eg. flipping heads from fair coin 100,000 times should be closer to 0.5 than 100 times.

* How do you calculate the needed sample size?
* <img src='/imagese/sample_size.png'>

* When you sample, what bias are you inflicting?
  * Potential biases include the following:

  Sampling bias: a biased sample caused by non-random sampling
  Under coverage bias: sampling too few observations
  Survivorship bias: error of overlooking observations that did not make it past a form of selection process.

* How do you control for biases?
There are many things that you can do to control and minimize bias. Two common things include __randomization__, where participants are __assigned by chance__, and __random sampling__, sampling in which each member has an equal probability of being chosen.

* What are confounding variables?
  * A confounding variable, or a confounder, is a variable that influences both the dependent variable and the independent variable, causing a spurious association, a mathematical relationship in which two or more variables are associated but not causally related.

* What is/B testing?
  */B testing is a form of hypothesis testing and two-sample hypothesis testing to compare two versions, the control and variant, of a single variable. It is commonly used to improve and optimize user experience and marketing.

* How do you prove that males are on average taller than females by knowing just gender height?
  * You can use hypothesis testing to prove that males are taller on average than females.
  The null hypothesis would state that males and females are the same height on average, while the alternative hypothesis would state that the average height of males is greater than the average height of females.
Then you would collect a random sample of heights of males and females and use a t-test to determine if you reject the null or not.

<img src='/images/prb.png'/>
<img src='/images/prb2.png'>
<img src='/images/prb3.png'>

*  An HIV test has a sensitivity of 99.7% and a specificity of 98.5%. A subject from a population of prevalence 0.1% receives a positive test result. What is the precision of the test (i.e the probability he is HIV positive)?
  * $$ Precision(PPV = Positive predictive value) = \frac{prevalence * sensitivity}{(prevalence * sensitivity) + (1 - prevalence)* (1 - specifity)}$$

  $$ Sensitivity = \frac{TP}{TP + FN}$$
  $$ Specifity = \frac{TN}{TN + FP}$$
  $$ Precision = \frac{TP}{TP + FP}$$
  $$ Recall = \frac{TP}{TP + FN}$$

* You are running for office and your pollster polled hundred people. Sixty of them claimed they will vote for you. Can you relax?
  * $$\hat{p}\pm z*\sqrt{\frac{\hat{p}(1 - \hat{p})}{n}}$$
  * Assume there is only one another candidate
  * And we want a 95% confidence interval, so we have $Z*$ = 1.96
  * n= 100,$\hat{p}=\frac{60}{100}$ = 0.6
  * This gives [50.4, 69.6]
  * Therefore, given a confidence interval of 95%, if you are okay with the worst scenario of tying then you can relax. Otherwise, you cannot relax until you got 61 out of 100 to claim yes.

*  Geiger counter records 100 radioactive decays in 5 minutes. Find an approximate 95% interval for the number of decays per hour.
  * Since this is a Poisson distribution question, mean = lambda = variance, which also means that standard deviation = square root of the mean
  a 95% confidence interval implies a z score of 1.96
  one standard deviation = 10
  Therefore the confidence interval = 100/- 19.6 = [964.8, 1435.2]

* The homicide rate in Scotland fell last year to 99 from 115 the year before. Is this reported change really noteworthy?
   * Since this is a Poisson distribution question, mean = lambda = variance, which also means that standard deviation = square root of the mean
a 95% confidence interval implies a z score of 1.96
one standard deviation = sqrt(115) = 10.724
Therefore the confidence interval = 11/- 21.45 = [93.55, 136.45]. Since 99 is within this confidence interval, we can assume that this change is not very noteworthy.

* In a population of interest, a sample of 9 men yielded a sample average brain volume of 1,100cc and a standard deviation of 30cc. What is a 95% Student’s T confidence interval for the mean brain volume in this new population?
  * $$ \bar{X}\pm t\frac{s}{\sqrt{n}}$$
  * degree of freedom = n -1 = 9 -1 = 8
  * mean = 1000
  * std = 30
  * confidence = 95%, tscore = 2.306
  * confidence interval = [1000$\pm$ 2.306 * $\frac{30}{3}$]

* A diet pill is given to 9 subjects over six weeks. The average difference in weight (follow up — baseline) is -2 pounds. What would the standard deviation of the difference in weight have to be for the upper endpoint of the 95% T confidence interval to touch 0?
  * Upper bound = mean + t-score*(standard deviati/sqrt(sample size))
0 = -2 + 2.306*/3)
2 = 2.306 * / 3
s = 2.601903
Therefore the standard deviation would have to be at least approximately 2.60 for the upper bound of the 95% T confidence interval to touch 0.


* In a study of emergency room waiting times, investigators consider a new and the standard triage systems. To test the systems, administrators selected 20 nights and randomly assigned the new triage system to be used on 10 nights and the standard system on the remaining 10 nights. They calculated the nightly median waiting time (MWT) to see a physician. The average MWT for the new system was 3 hours with a variance of 0.60 while the average MWT for the old system was 5 hours with a variance of 0.68. Consider the 95% confidence interval estimate for the differences of the mean MWT associated with the new system. Assume a constant variance. What is the interval? Subtract in this order (New System — Old System).

  * Z or t distribution depends on whether sample size larger than 30 or not. Here the sample size less than 30. so we will use t distribution
  * Standard error is the estimation of standard deviation assuming the variance is constant
  $$SE(\bar{X_{1}}- \bar{X_{2}}) = S_{p}\sqrt{\frac{1}{n_{1}} +\frac{1}{n_{2}}}$$
  $$Confidence_{Interval} = Mean \pm t_{statistics} * Standard_{error}$$
  * tscore is based on Confidence interval and degree of freedom. degree of freedom = n1 + n2 - 2 = 10 + 10 -2 = 18
  * tscore is 2.101
  * standard error = 0.352
  * confidence interval = [-2.75, -1.25]

* To further test the hospital triage system, administrators selected 200 nights and randomly assigned a new triage system to be used on 100 nights and a standard system on the remaining 100 nights. They calculated the nightly median waiting time (MWT) to see a physician. The average MWT for the new system was 4 hours with a standard deviation of 0.5 hours while the average MWT for the old system was 6 hours with a standard deviation of 2 hours. Consider the hypothesis of a decrease in the mean MWT associated with the new treatment. What does the 95% independent group confidence interval with unequal variances suggest vis a vis this hypothesis? (Because there’s so many observations per group, just use the Z quantile instead of the T.)

* Z statics will be used.



* Consider influenza epidemics for two-parent heterosexual families. Suppose that the probability is 17% that at least one of the parents has contracted the disease. The probability that the father has contracted influenza is 12% while the probability that both the mother and father have contracted the disease is 6%. What is the probability that the mother has contracted influenza?

 * Using the General Addition Rule in probability:
  P(mother or father) = P(mother) + P(father) — P(mother and father)
  P(mother) = P(mother or father) + P(mother and father) — P(father)
  P(mother) = 0.17 + 0.06–0.12
  P(mother) = 0.11 

*  Suppose that diastolic blood pressures (DBPs) for men aged 35–44 are normally distributed with a mean of 80 (mm Hg) and a standard deviation of 10. About what is the probability that a random 35–44 year old has a DBP less than 70?
  * Since 70 is one standard deviation below the mean, take the area of the Gaussian distribution to the left of one standard deviation.
= 2.3 + 13.6 = 15.9%


## SQl

* Write a SQL query to get the second highest salary from the Employee table. For example, given the Employee table below, the query should return 200 as the second highest salary. If there is no second highest salary, then the query should return null.
  * ```SQL
    SELECT
    IFNULL(
        (SELECT DISTINCT Salary
        FROM Employee
        ORDER BY Salary DESC
        LIMIT 1 OFFSET 1
        ), null) as SecondHighestSalary
    FROM Employee
    LIMIT 1
  ```
  Write a SQL query to find all duplicate emails in a table named Person.
  ```SQL
  SELECT Email
FROM (
    SELECT Email, count(Email) AS count
    FROM Person
    GROUP BY Email
) as email_count
WHERE count > 1
  
  ```
  or 
 ```SQL
SELECT Email
FROM Person
GROUP BY Email
HAVING count(Email) > 1 
 ``` 

* Given a Weather table, write a SQL query to find all dates' Ids with higher temperature compared to its previous (yesterday's) dates.
```SQL
SELECT DISTINCT a.Id
FROM Weather a, Weather b
WHERE a.Temperature > b.Temperature
AND DATEDIFF(a.Recorddate, b.Recorddate) = 1 
 ```

* The Employee table holds all employees. Every employee has an Id, a salary, and there is also a column for the department Id.
```SQL
SELECT
    Department.name AS 'Department',
    Employee.name AS 'Employee',
    Salary
FROM Employee
INNER JOIN Department ON Employee.DepartmentId = Department.Id
WHERE (DepartmentId , Salary) 
    IN
    (   SELECT
            DepartmentId, MAX(Salary)
        FROM
            Employee
        GROUP BY DepartmentId
 ) 
 ```

 * Mary is a teacher in a middle school and she has a table seat storing students' names and their corresponding seat ids. The column id is a continuous increment. Mary wants to change seats for the adjacent students.
   * If the number of students is odd, there is no need to change the last one’s seat.
   * Think of a CASE WHEN THEN statement like an IF statement in coding.
The first WHEN statement checks to see if there’s an odd number of rows, and if there is, ensure that the id number does not change.
The second WHEN statement adds 1 to each id (eg. 1,3,5 becomes 2,4,6)
Similarly, the third WHEN statement subtracts 1 to each id (2,4,6 becomes 1,3,5)
```SQL
SELECT 
    CASE 
        WHEN((SELECT MAX(id) FROM seat)%2 = 1) AND id = (SELECT MAX(id) FROM seat) THEN id
        WHEN id%2 = 1 THEN id + 1
        ELSE id - 1
    END AS id, student
FROM seat
ORDER BY id 
 ```

 *  Explain what Glove embeddings are.
  Rather than use contextual words, we calculate a co-occurrence matrix of all words. Glove will also take local contexts into account, per a fixed window size, then calculate the covariance matrix. Then, we predict the co-occurence ratio between the words in the neural network.

  GloVe will learn this matrix and train word vectors that predict co-occurrence ratios. Loss is weighted by word frequency.

  It’s clear that these questions are meant to test if candidates understand the situations in which they would apply different types of models. They’re also mostly definition based questions, so if you memorize a bunch of different machine learning definitions and applications, you will usually do okay in this part.

* Given three random variables independent and identically distributed from a uniform distribution of 0 to 4, what is the probability that the median is greater than 3?
  * Solution:

If we break down this question, we'll find that another way to phrase it is to ask what the probability is that at least two of the variables are larger than 3.

For example, if look at the combination of events that satisfy the condition, the events can actually be divided into two exclusive events.

Event A: All three random variables are larger than 3.
Event B: One random variable is smaller than 3 and two are larger than 3.

Given these two events satisfy the condition of the median > 3, we can now calculate the probability of both of the events occuring. The question can now be rephrased as P(Median > 3) = P(A) + P(B).

Let's calculate the probability of the event A. The probability that a random variable > 3 but less than 4 is equal to/4. So the probability of event A is:

P(A) = /4) * /4) * /4) =/64

The probability of event B is that two values must be greater than 3, but one random variable is smaller than 3. We can calculate this the same way as the calculating the probability of A. The probability of a value being greater than 3 is/4 and the probability of a value being less than 3 is/4. Given this has to occur three times we multiply the condition three times.

P(B) = 3 * (/4) * /4) * /4)) =/64

Therefore the total probability is P(A)+P(B) =/64 +/64 = /64


* What is the difference between a ROC curve and a precision-recall curve? When should I use each?
  * There is a very important difference between what a ROC curve represents vs that of a PRECISION vs RECALL curve.

Remember, a ROC curve represents a relation between sensitivity (RECALL) and False Positive Rate (NOT PRECISION). Sensitivity is the other name for recall but the False Positive Rate is not PRECISION.

Reca/Sensitivity is the measure of the probability that your estimate is 1 given all the samples whose true class label is 1. It is a measure of how many of the positive samples have been identified as being positive.

Specificity is the measure of the probability that your estimate is 0 given all the samples whose true class label is 0. It is a measure of how many of the negative samples have been identified as being negative.

PRECISION on the other hand is different. It is a measure of the probability that a sample is a true positive class given that your classifier said it is positive. It is a measure of how many of the samples predicted by the classifier as positive is indeed positive. Note here that this changes when the base probability or prior probability of the positive class changes. Which means PRECISION depends on how rare is the positive class. In other words, it is used when positive class is more interesting than the negative class.

So, if your problem involves kind of searching a needle in the haystack when for ex: the positive class samples are very rare compared to the negative classes, use a precision recall curve. Othwerwise use a ROC curve because a ROC curve remains the same regardless of the baseline prior probability of your positive class (the important rare class).

* Type 1 error - False Positive
* Type 2 error - False negative

* What are the primacy and novelty effects?
  * The primacy effect involves users being resistant to change, while the novelty effect involves users becoming temporarily excited by new things.
* What is a null hypothesis?
  * There is no significant difference between populations that can’t be explained by chance or sampling error.

* What is a holdback experiment?
  * A holdback experiment rolls out a feature to a high proportion of users, but holds back the remaining percent of users to monitor their behavior over a longer period of time. This allows analysts to quantify the ‘lift’ of a change over a longer amount of time.


  

##/B testing
  * Setting a metric
  * Setting a threshhold
  * Sample size and Experiment length( rule of thumb two weeks to run the experiment)
  * Randomization and Assignment 




  #### References
  * [over 100 data science interview question](http//towardsdatascience.c/over-100-data-scientist-interview-questions-and-answers-c5a66186769a)
  * [google Interview question](http//www.interviewquery.c/blog-google-data-science-interview-questions-and-solutio/)
  * [ROC curve and precision recall curve difference](http//www.quora.c/What-is-the-difference-between-a-ROC-curve-and-a-precision-recall-curve-When-should-I-use-each)