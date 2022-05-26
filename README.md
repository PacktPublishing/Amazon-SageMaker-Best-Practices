## [Get this title for $10 on Packt's Spring Sale](https://www.packt.com/B17249?utm_source=github&utm_medium=packt-github-repo&utm_campaign=spring_10_dollar_2022)
-----
For a limited period, all eBooks and Videos are only $10. All the practical content you need \- by developers, for developers

# Amazon SageMaker Best Practices

<a href="https://www.packtpub.com/in/data/amazon-sagemaker-best-practices"><img src="https://www.packtpub.com/media/catalog/product/cache/c2dd93b9130e9fabaf187d1326a880fc/9/7/9781801070522-original_104.jpeg" alt="Book Name" height="256px" align="right"></a>

This is the code repository for [Amazon SageMaker Best Practices](https://www.packtpub.com/in/data/amazon-sagemaker-best-practices), published by Packt.

**Proven tips and tricks to build successful machine learning solutions on Amazon SageMaker**

## What is this book about?
Amazon SageMaker is a fully managed AWS service that provides the ability to build, train, deploy, and monitor machine learning models. The book begins with a high-level overview of Amazon SageMaker capabilities that map to the various phases of the machine learning process to help set the right foundation. You'll learn efficient tactics to address data science challenges such as processing data at scale, data preparation, connecting to big data pipelines, identifying data bias, running A/B tests, and model explainability using Amazon SageMaker.

This book covers the following exciting features: 
* Perform data bias detection with AWS Data Wrangler and SageMaker Clarify
* Speed up data processing with SageMaker Feature Store
* Overcome labeling bias with SageMaker Ground Truth
* Improve training time with the monitoring and profiling capabilities of SageMaker Debugger
* Address the challenge of model deployment automation with CI/CD using the SageMaker model registry
* Explore SageMaker Neo for model optimization

If you feel this book is for you, get your [copy](https://www.amazon.com/dp/1801070520) today!

<a href="https://www.packtpub.com/?utm_source=github&utm_medium=banner&utm_campaign=GitHubBanner"><img src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png" alt="https://www.packtpub.com/" border="5" /></a>

## Instructions and Navigations
All of the code is organized into folders. For example, Chapter04.

The code will look like the following:

```
for batch in chunks(partitions_to_add, 100):
    response = glue.batch_create_partition(
        DatabaseName=glue_db_name,
        TableName=glue_tbl_name,
        PartitionInputList=[get_part_def(p) for p in batch]
    )

```

**Following is what you need for this book:**
This book is for expert data scientists responsible for building machine learning applications using Amazon SageMaker. Working knowledge of Amazon SageMaker, machine learning, deep learning, and experience using Jupyter Notebooks and Python is expected. Basic knowledge of AWS related to data, security, and monitoring will help you make the most of the book.

With the following software and hardware list you can run all code files present in the book (Chapter 1-14).

### Software and Hardware List

| Chapter  | Software required                                                                                  | OS required                        |
| -------- | ---------------------------------------------------------------------------------------------------| -----------------------------------|
| 1-14     | AWS Account, Amazon SageMaker, Amazon SageMaker Studio, Amazon Athena                              | Windows, Mac OS X, and Linux (Any) |

We also provide a PDF file that has color images of the screenshots/diagrams used in this book. [Click here to download it](https://static.packt-cdn.com/downloads/9781801070522_ColorImages.pdf).

### Related products <Other books you may enjoy>
* Learn Amazon SageMaker[[Packt]](https://www.packtpub.com/product/learn-amazon-sagemaker/9781800208919) [[Amazon]](https://www.amazon.com/Learn-Amazon-SageMaker-developers-scientists/dp/180020891X)

## Get to Know the Authors
**Sireesha Muppala**
She is a Principal Enterprise Solutions Architect, AI/ML at Amazon Web Services (AWS). Sireesha holds a PhD in computer science and post-doctorate from the University of Colorado. She is a prolific content creator in the ML space with multiple journal articles, blogs, and public speaking engagements. Sireesha is a co-creator and instructor of the Practical Data Science specialization on Coursera.
  
**Randy DeFauw**
He is a Principal Solution Architect at AWS. He holds an MSEE from the University of Michigan, where his graduate thesis focused on computer vision for autonomous vehicles. He also holds an MBA from Colorado State University. Randy has held a variety of positions in the technology space, ranging from software engineering to product management.

**Shelbee Eigenbrode**
She is a Principal AI and ML Specialist Solutions Architect at AWS. She holds six AWS certifications and has been in technology for 23 years, spanning multiple industries, technologies, and roles. She is currently focusing on combining her DevOps and ML background to deliver and manage ML workloads at scale.
