# DIN
## Introduction
#### OVERVIEW
The overall scenario of the display advertising system is illustrated below.When a user visits the e-commerce advertising system, it

- Checks user historical behavior data.

- Generates candidate ads by matching module.

- Predicts the click probability of each ad and selects appropriate ads which can attract attention (click) by ranking module.

- Logs the user reactions given the displayed ads.

This turns to be a closed-loop consumption and generation of user behavior data.
#### Target
To fetch user's interest by utilising and excavating the rich historical behavior data is very crucial for building the click-through rate (CTR) prediction model in the online advertising system in e-commerce industry.

There are two key observations on user behavior data:

- Diversity: Users are interested in different kinds of goods when visiting e-commerce site. For example, a young mother may be interested in T-shits, leather handbag, shoes, earrings, children’s coat, etc at the same time.

- Local Activation: Whether users click or not click an item depends only on part of their related historical behavior.For example, a swimmer will click a recommended goggle mostly due to the fact her recent purchase of bathing suit while not the books in her last week’s shopping list.

## Background 

CPC(Cost-Per-Click): In CPC advertising systems like the one in Alibaba, advertisements are ranked based on eCPM(effective Cost Per Mille) which is a product of bid price and CTR( Click-Through-Rate).

Overall if we look at a performance of CTR prediction model it has a direct impact on the overall revenue and plays a crucial role in advertising systems. Most traditional CTR models lack capturing the structures of behavioral data.

Deep learning methods because of its success rate are extensively used in CTR prediction models.They usually first employ embedding layer on the input, mapping original large scale sparse id features to the distributed representations, then add fully connected layers i.e. MLP (Multi Layer Perceptrons) to automatically learn the nonlinear relations among features.MLP reduce a lot of feature engineering jobs, which are time and effort consuming in industry applications and have become a popular model structure on CTR prediction problem. However, in the fields with rich internet-scale user behavior data, such as online advertising and recommendation system in e-commence industry, these MLPs models often lack a deeper understanding and exploiting the specific structures of behavior data, thus leaving a space for further improvement.

DIN represents users’ diverse interests with an interest distribution and designs an attention-like network structure to locally activate the related interests according to the candidate ad, which is proven to be effective and significantly outperforms traditional model. Overfitting problem is easy to encounter on training such industrial deep network with large scale sparse inputs and will be handled with a new proposed adaptive regularization technique.

Inspired by the attention mechanism used in machine translation model, DIN represents users’ diverse interests with an interest distribution and designs an attention-like network structure to locally activate the related interests according to the candidate ad. Behaviors with higher relevance to the candidate ad get higher attention scores and dominate the prediction. Experiments on Alibaba’s productive CTR prediction datasets prove that the proposed DIN model significantly outperforms MLPs under the GAUC (Group weighted AUC) metric measurement.Let us understand the GAUC metric in detail.

Area under receiver operator curve (AUC)is a commonly used metric in CTR prediction area. In practice,a new metric named GAUC, which is the generalization of AUC is designed which is a weighted average of AUC calculated in the subset of samples group by each user. The weight can be impressions or clicks. An impression based GAUC is calculated as follows:
~~~
GAUC = Sigma(wi* AUCi)/Sigma(wi) where i = 1 to n
~~~
GAUC is practically proven to be more indicative in display advertisement settings, where CTR model is applied to rank candidate ads for each user and model performance is mainly measured by how good the ranking list is, that is, a user specific AUC. Hence, this method can remove the impact of user bias and measure more accurately the performance of the model over all users. With years of application implementation effectiveness in production systems, GAUC metric is verified to be more stable and reliable than AUC.

#### challenging

Overfitting problem is easy to encounter on training such industrial deep network with large scale sparse inputs. The deep network models easily fall into the overfitting trap and cause the model performance to drop rapidly which is overcome by proposing an efficient adaptive regularization technique.

![](https://raw.githubusercontent.com/guanqin-123/kkb_notes/main/currency/currency_data/DIN.png)

## BASE MODEL
The base model is composed with two steps:

- Transfer each sparse id feature into a embedded vector space.

-  Apply MLPs to fit the output.

Note that the input contains user behavior sequence ids, of which the length can be varied. Thus we add a pooling layer (e.g. sum operation) to summarize the sequence and get a fixed size vector. As illustrated in the left part of the model architecture, the base model works well practically, which now serves the main traffic of our online display advertising system.

However, going deep into the pooling operation, we will find that much information is lost, that is, it destroys the inner structure of user behavior data. This observation inspires us to design a better model.


## DEEP INTEREST NETWORK (DIN) DESIGN
In our display advertising scenario, we wish our model to truly reveal the relationship between the candidate ad and users’ interest based on their historical behaviors.

As discussed above, behavior data contains two structures: diversity and local activation.

The diversity of behavior data reflects users’ various interests. User click of ad often originates from just part of user’s interests. In NMT task it is assumed that the importance of each word in each decode process is different in a sentence. Attention network can be viewed as a specially designed pooling layer which learns to assign attention scores to each word in the sentence, which in other words follows the diversity structure of data.

Note :It is unsuitable and highly not recommended to directly apply the attention layer in our applications, where embedding vector of user interest varies with different candidate ads but rather should follow the local activation structure.

The distributed representation of users(Vu) and ads(Va). For the same user,Vu is a fixed point in embedding space. It is the same to ad embedding Va. It is assumed that we use inner product to calculate the relevance between user and ad,
~~~
F(U,A) =Vu∙Va
~~~
If both F(U,A) and F(U,B) are high, which means user U is relevant to both ads "A" and "B". Under this way of calculation, any point on the line between the vector of Va and Vb will get high relevance score.


DIN is implemented at a multi-GPU distributed training platform named X-Deep Learning (XDL), which supports model-parallelism and data-parallelism. Due to the high performance and flexibility of XDL platform, we accelerate training process about 10 times and optimize hyparameters automatically with high tuning efficiency.XDL is designed to solve the challenges of training industrial scale deep learning networks with large scale sparse inputs and tens of billions of parameters. 

## My evaluations:
Employ the embedding technique to cast the original sparse input into low dimensional and dense vectors ii) Bridge with networks like MLPs, RNN, CNN etc. Most of the parameters are focused in the first embedding step which needs to be distributed over multi machines. The second network step can be handled within single machine. Under such circumstance, we architecture the XDL platform is architected in a bridge manner, as shown above, which is composed of three main kinds of components:

- Distributed Embedding Layer: It is a model-parallelism module, parameters of embedding layer are distributed over multi-GPUs. Embedding Layer works as a predefined network unit, which provides with forward and backward modes.

- Local Backend: It is a standalone module, which aims to handle the local network training. Here we reuse the open-sourced deep learning frameworks, like tensorflow. With the unified data exchange interface and abstraction, it is easy for us to integrate and switch in different kinds of frameworks.

- Communication Component: It is the base module, which helps to parallel both the embedding layer and backend.


