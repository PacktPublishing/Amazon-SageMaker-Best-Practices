# Introduction

This repository contains the example CloudFormation templates that can be used to setup a data science environment using either Amazon SageMaker Studio or Amazon SageMaker Notebook Instances. The pre-requisities are noted where applicable for either option.  There is also a section included for those unfamiliar with using AWS CloudFormation on how to launch the CloudFormation templates provided.  

# Prerequisites

## Clone Repository 

To use the CloudFormation templates provided in this repository, you must first clone this repository creating a local copy. These local files will be used to upload the appropriate template to AWS CloudFormation to create your data science environment. 

## Create or Identify VPC 

The templates in this repository are setup using a VPC so you must either (1) have an existing VPC that can be used ~OR~ (2) create a new VPC that will be referenced as parameters when launching the CloudFormation stacks.  

If you do not have a VPC, a CloudFormation template is provided in this directory, [datascience-vpc.yaml](datascience-vpc.yaml), to get you up and running quickly. 

## Create a KMS Key for Encryption

The templates provided accept a KMS key on input to be used for encrypting storage directly attached to your environments.  

You can use an existing key or alternatively create your own symmetric encryption key using the following instructions:  https://docs.aws.amazon.com/kms/latest/developerguide/create-keys.html

Make sure you note the KMS Key ID & ARN to provide that as an input parameter to your CloudFormation templates. 


## Using AWS CloudFormation
 
To launch the CloudFormation templates provided you can utilize the CLI, SDK, or the AWS Console.  Instructions provided include using the AWS console to launch the provided templates. 

1. Sign in to the AWS Management Console and open the AWS CloudFormation console at https://console.aws.amazon.com/cloudformation

2. In the **Stacks** section, select **Create stack** and select **With new resources (standard)** from the dropdown.  

3. Under **Prerequisite - Prepare template** choose **Template is ready** 

4. Under **Specify template** choose **Upload a template file**

5. This will allow you to upload the appropriate CloudFormation template that will be used to create your data science environment.  Because the input parameters, additional pre-requisites, and post launch tasks vary between the two environment types, they are covered specifically in the sections below. 

# Studio Environment

## Studio Environment Pre-Requisites
1. **Existing Studio Domain**  Because it's common to add new users to an existing Studio Domain, the CloudFormation templates to create a domain and add a new user are typically separated.  Creating a Studio Domain is a one-time setup activity per AWS Account / AWS Region.   The CloudFormation template to create a new user assumes an existing Studio Domain.  If you do not have a Studio Domain, a CloudFormation template is provided to create a new domain as a pre-requisite: [studio-domain.yaml](datascience-vpc.yaml).

## Creating the Stack & Template Usage
 
Go through the provided instructions to launch the [studio-environment.yaml](studio-environment.yaml) CloudFormation template

## Post Launch Tasks

We need to copy the dataset we'll be using throughout the chapters from a public S3 bucket to the bucket created by the CloudFormation template.  There are multiple ways this could be done but for simplicity in this case, we'll simply execute a command using the AWS Command Line Interface (CLI) using the terminal available in your Studio environment. 

To do this: 
1. Sign in to the AWS Management Console and open the AWS CloudFormation console at https://console.aws.amazon.com/cloudformation
2. Go to the stack provisioned above -> Go to the **Outputs** tab and capture the name of the S3Bucket that was created
3. Open the Amazon SageMaker console at https://console.aws.amazon.com/sagemaker, select **Amazon SageMaker Studio** then select **Open Studio** for the user name you provided.
4. Once you're in your Studio environment, go to **File** -> **New** -> **Terminal** 
5. Copy & paste the command below, replacing s3 bucket with the value from #2 above: 

       nohup aws s3 cp s3://openaq-fetches/ s3://<enter S3Bucket>/data/  --recursive &

# Notebook Instance Environment

## Notebook Instance Environment Pre-Requisites

None other than what is already noted above under Prerequisites

## Creating the Stack & Template Usage

Go through the provided instructions to launch the [notebook-instance-environment.yaml](notebook-instance-environment.yaml) CloudFormation template

## Post Launch Tasks

There are no post launch tasks following the successful completion of stack creation above and your data science environment is available for use in the AWS console under **Amazon SageMaker** -> **Notebook** -> **Notebook instances**
