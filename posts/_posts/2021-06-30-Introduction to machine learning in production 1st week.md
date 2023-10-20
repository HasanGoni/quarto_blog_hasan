# Introduction to machine learning in production week 1
## Common Deployment cases
1. New product/ capability
2. Automate / assist with manual task
3. Replace previous ML systems


# Key ideas 
* Gradual ramp up with monitoring
* Rollback

# Deployment Mode

## 1. Shadow mode
    * ML system and human will do parallely the deployment
    * But ML system recommendation will not be used in production.
    * Compare the both human and machine prediction and can evaluate the ML systems performance

## 2. Canary mode
    * Roll out small amount of (say 5 %)traffic to production 
    * if there is a error small amount of will be effected

## 3. Blue green mode
    * New ML model will be green and old will be blue
    * just change the traffic to green model
    * if there is a problem we can just simply change to blue model

# Degrees of automation

1. Human only 
2. Shadow mode
3. AI Assistance
    * Human will do prediction but AI system will help him. say in image detection algorithm will create bounding box in a specific area of the model
4. Partial automation
    * if the prediction model is not sure then only sent to human for prediction.
    * Very effective
5. Full automation

# Monitoring Dashboard

1. Server load
2. Fraction of null output
3. Fraction of missing input values

# Best practise
* Sit with everyone and brainstorm what could possibly go wrong.
* Brainstorm a statistic what will detect this problem
* when starting model, it is ok to start with lots of monitoring metric. You can always delete metric if you think it will not change so you want to delete

# Example of metric to track
* Software metrics: memory computer, latency, throughput, server load

* Input mode: 
    * Avg. input length,
    * Avg. input volume
    * Avg. missing values
    * Avg. image brightness	

* Output mode: Highly related to use case, some example
    * times return null
    * times users redoes search
    * times switch to typing

# Iterative process
* Deployment --> see traffic --> performance anaylsis

* Set threshold to trigger a alarm.

# Require retrain

* manual 
* automatic

# Example Speech recoginition example
* Normally in speech recognition we have a audion as an input and get a transcript script
* however it is not very straightforward. First audio goes to a (VAD- voice activity detection).VAD search whether anyone is speaking. If only anyone is speaking then vAD send it to speech recognition system.
* So here we have two models 
1. VAD
2. Speech recognition system
* both model is correlated, if one performence degrade, other will also be affected.

VAD--> speech recognition system
User Data --> user profile --> Recommender system --> product recommendation.

* So it is a very pipeline where both machine learning and non machine learning components are available. 
* Metrics to monitor
    1. Software metrics
    2. Input metrics
    3. Output metrics

* How quickly do they change( there is a exception)
    1. User data generally has slower drift
    2. Enterprise data (B2B applications) can shift fast

[Second week class notes](https://hasangoni.github.io/2021/06/30/Introduction_ml_in_production_week2.html)