import boto3
from zipfile import ZipFile
import logging
import sys
import json

class SagemakerHelper:
    def __init__(self, region, iam_role_prefix):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('SagemakerHelper')
        self.sagemaker = boto3.client('sagemaker')
        self.iam = boto3.client('iam')
        self.s3 = boto3.client('s3')
        self.lambdac = boto3.client('lambda')
        self.region = region
        self.iam_role_prefix = iam_role_prefix

    def create_workteam(self, WorkteamName, user_pool_id, group_name, user_pool_client_id):
        response = self.sagemaker.create_workteam(
            WorkteamName=WorkteamName,
            MemberDefinitions=[
                {
                    'CognitoMemberDefinition': {
                        'UserPool': user_pool_id,
                        'UserGroup': group_name,
                        'ClientId': user_pool_client_id
                    }
                }
            ],
            Description = WorkteamName
        )
        self.workteam_arn = response['WorkteamArn']
        self.workteam_name = WorkteamName
        self.logger.info(f"Created workteam {self.workteam_arn}")

    def get_workforce_domain(self):
        response = self.sagemaker.describe_workteam(
            WorkteamName=self.workteam_name
        )
        self.workforce_domain = response['Workteam']['SubDomain']
        return self.workforce_domain

    def create_manifest(self, s3_bucket, s3_prefix, s3_prefix_manifest, max_entries = 20):
        self.s3_prefix_manifest = s3_prefix_manifest
        manifest_file_local = 'manifest.txt'
        manifests = []
        response = self.s3.list_objects_v2(
            Bucket=s3_bucket,
            Prefix=s3_prefix
        )
        r = response['Contents'][0]
        self.s3.download_file(s3_bucket, r['Key'], 'temp.json')
        self.logger.debug("Processing " + r['Key'])
        with open('temp.json', 'r') as F:
            for l in F.readlines():
                if len(manifests) > max_entries:
                    break
                j = json.loads(l)
                manifests.append(f"{j['parameter']},{j['value']},{j['unit']},{j['coordinates']['latitude']},{j['coordinates']['longitude']}")
        self.logger.debug(f"Got {len(manifests)} manifest entries")
        with open(manifest_file_local, 'wt') as F:
            for m in manifests:
                F.write('{"source": "' + m + '"}' + "\n")
        self.s3.upload_file(manifest_file_local, s3_bucket, f"{self.s3_prefix_manifest}/openaq.manifest")

        label_file_local = 'label.txt'
        with open(label_file_local, 'wt') as F:
            F.write('{' + "\n")
            F.write('"document-version": "2018-11-28",' + "\n")
            F.write('"labels": [{"label": "good"},{"label": "bad"}]' + "\n")
            F.write('}' + "\n")
        self.s3.upload_file(label_file_local, s3_bucket, f"{self.s3_prefix_manifest}/openaq.labels")
    
    def create_role(self, service_for, name, policies = []):
        role_doc = """{
        "Version": "2012-10-17",
        "Statement": [
            {
            "Effect": "Allow",
            "Principal": {
                "Service": f"{service_for}.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
            }
        ]
        }
        """
        role =f"{self.iam_role_prefix}-{name}-role",
        response = self.iam.create_role(
            RoleName=role,
            AssumeRolePolicyDocument=role_doc
        )
        role_arn, role_name = response['Role']['Arn'],response['Role']['RoleName']
        for p in policies:
            self.iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn=p
            )

        return role_arn, role_name

    def create_fn(self, fn_prefix, fname, l_prefix, role_arn, handler):
        fzip = f"{fname}.zip"
        with ZipFile(fzip,'w') as zip:
            zip.write(f"workflow/{fname}")
        with open(fzip, 'rb') as file_data:
            f_bytes = file_data.read()
        
        response = self.lambdac.create_function(
            FunctionName=f"{fn_prefix}-{l_prefix}-LabelingFunction",
            Runtime='python3.7',
            Role=role_arn,
            Handler=handler,
            Code={
                'ZipFile': f_bytes
            },
            Description=f"{fn_prefix}-{l_prefix}-LabelingFunction",
            Timeout=300,
            MemorySize=1024,
            Publish=True,
            PackageType='Zip'
        )
        return response['FunctionArn']


    def create_workflow(self, s3_bucket, s3_prefix_workflow, fn_prefix, label_prefix, s3_prefix_labels):
        self.workflow_role_arn, self.workflow_role_name = self.create_role("sagemaker", "workflow",
            policies = ['arn:aws:iam::aws:policy/AmazonS3FullAccess',
                'arn:aws:iam::aws:policy/AmazonSageMakerFullAccess']
        )
        self.lambda_role_arn, self.lambda_role_name = self.create_role("lambda", "lambda",
            policies = ['arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
                'arn:aws:iam::aws:policy/AmazonS3FullAccess']
        )

        self.s3.upload_file('workflow/ui.liquid.html', s3_bucket, f"{s3_prefix_workflow}/openaq.liquid.html")
        self.pre_arn = self.create_fn(fn_prefix, 'pre.py', "pre", self.lambda_role_arn, "pre.handler")
        self.post_arn = self.create_fn(fn_prefix, 'post.py', "post", self.lambda_role_arn, "post.handler")

        self.sagemaker.create_labeling_job(
            LabelingJobName=fn_prefix,
            LabelAttributeName='badair',
            InputConfig={
                'DataSource': {
                    'S3DataSource': {
                        'ManifestS3Uri': f"s3://{s3_bucket}/{self.s3_prefix_manifest}/openaq.manifest"
                    }
                }
            },
            OutputConfig={
                'S3OutputPath': f"s3://{s3_bucket}/{s3_prefix_labels}/openaq"
            },
            RoleArn=self.workflow_role_arn,
            LabelCategoryConfigS3Uri=f"s3://{s3_bucket}/{self.s3_prefix_manifest}/openaq.labels",
            StoppingConditions={
                'MaxHumanLabeledObjectCount': 10,
                'MaxPercentageOfInputDatasetLabeled': 5
            },
            HumanTaskConfig={
                'WorkteamArn': self.workteam_arn,
                'UiConfig': {
                    'UiTemplateS3Uri': f"s3://{s3_bucket}/{s3_prefix_workflow}/openaq.liquid.html"
                },
                'PreHumanTaskLambdaArn': self.pre_arn,
                'TaskTitle': 'Label Air Quality',
                'TaskDescription': 'Was it a good air day?',
                'NumberOfHumanWorkersPerDataObject': 1,
                'TaskTimeLimitInSeconds': 3600,
                'AnnotationConsolidationConfig': {
                    'AnnotationConsolidationLambdaArn': self.post_arn
                }
            }
        )

    def create_workflow_multiple_workers(self, s3_bucket, s3_prefix_workflow, fn_prefix, label_prefix, s3_prefix_labels):
        self.mpost_arn = self.create_fn(fn_prefix, 'post-multiple.py', "mpost", self.lambda_role_arn, "post.handler")
        self.sagemaker.create_labeling_job(
            LabelingJobName=fn_prefix,
            LabelAttributeName='badair',
            InputConfig={
                'DataSource': {
                    'S3DataSource': {
                        'ManifestS3Uri': f"s3://{s3_bucket}/{self.s3_prefix_manifest}/openaq.manifest"
                    }
                }
            },
            OutputConfig={
                'S3OutputPath': f"s3://{s3_bucket}/{s3_prefix_labels}/openaq"
            },
            RoleArn=self.workflow_role_arn,
            LabelCategoryConfigS3Uri=f"s3://{s3_bucket}/{self.s3_prefix_manifest}/openaq.labels",
            StoppingConditions={
                'MaxHumanLabeledObjectCount': 10,
                'MaxPercentageOfInputDatasetLabeled': 5
            },
            HumanTaskConfig={
                'WorkteamArn': self.workteam_arn,
                'UiConfig': {
                    'UiTemplateS3Uri': f"s3://{s3_bucket}/{s3_prefix_workflow}/openaq.liquid.html"
                },
                'PreHumanTaskLambdaArn': self.pre_arn,
                'TaskTitle': 'Label Air Quality',
                'TaskDescription': 'Was it a good air day?',
                'NumberOfHumanWorkersPerDataObject': 2,
                'TaskTimeLimitInSeconds': 3600,
                'AnnotationConsolidationConfig': {
                    'AnnotationConsolidationLambdaArn': self.mpost_arn
                }
            }
        )

if __name__ == "__main__":
    logging.warn("No main function defined")
    sys.exit(0)