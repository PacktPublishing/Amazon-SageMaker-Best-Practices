import boto3
import logging
import sys
import hmac
import hashlib
import base64

class CognitoHelper:
    def __init__(self, region, iam_role_prefix):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('CognitoHelper')
        self.cognito = boto3.client('cognito-idp')
        self.cognitoid = boto3.client('cognito-identity')
        self.iam = boto3.client('iam')
        self.region = region
        self.iam_role_prefix = iam_role_prefix

    def create_user_pool(self, PoolName):
        response = self.cognito.create_user_pool(PoolName=PoolName)
        self.user_pool_id = response['UserPool']['Id']
        self.user_pool_arn = response['UserPool']['Arn']
        self.logger.info(f"Created user pool with ID: {self.user_pool_id}; ARN: {self.user_pool_arn}")

    def create_user_pool_client(self, ClientName):
        response = self.cognito.create_user_pool_client(
            UserPoolId=self.user_pool_id,
            ClientName=ClientName,
            GenerateSecret=True,
            SupportedIdentityProviders = ['COGNITO'],
            ExplicitAuthFlows=[
                'ADMIN_NO_SRP_AUTH'
            ]
        )
        self.user_pool_client_id = response['UserPoolClient']['ClientId']
        self.logger.info(f"Created user pool client with ID: {self.user_pool_client_id}")

    def create_identity_pool(self, IdentityPoolName):
        response = self.cognitoid.create_identity_pool(
            IdentityPoolName=IdentityPoolName,
            AllowUnauthenticatedIdentities=False,
            CognitoIdentityProviders=[
                {
                    'ProviderName': f"cognito-idp.{self.region}.amazonaws.com/{self.user_pool_id}",
                    'ClientId': self.user_pool_client_id
                },
            ]
        )
        self.id_pool_id = response['IdentityPoolId']
        self.logger.info(f"Created identity pool {self.id_pool_id}")

    def create_group(self, GroupName):
        assume_role_doc = """{
            "Version": "2012-10-17",
            "Statement": [
                {
                "Effect": "Allow",
                "Principal": {
                    "Federated": "cognito-identity.amazonaws.com"
                },
                "Action": "sts:AssumeRoleWithWebIdentity",
                "Condition": {
                    "StringEquals": {
                    "cognito-identity.amazonaws.com:aud": """ + '"' + self.id_pool_id + '"' + """
                    }
                }
                },
                {
                "Effect": "Allow",
                "Principal": {
                    "Federated": "cognito-identity.amazonaws.com"
                },
                "Action": "sts:AssumeRoleWithWebIdentity",
                "Condition": {
                    "ForAnyValue:StringLike": {
                    "cognito-identity.amazonaws.com:amr": "authenticated"
                    }
                }
                }
            ]
            }
        """
        response = self.iam.create_role(
            RoleName=f"{self.iam_role_prefix}-worker-group-role",
            AssumeRolePolicyDocument=assume_role_doc
        )
        self.group_role_id = response['Role']['RoleId']
        self.group_role_arn = response['Role']['Arn']

        response = self.cognito.create_group(
            GroupName=GroupName,
            UserPoolId=self.user_pool_id,    
            RoleArn=self.group_role_arn,
            Precedence=1
        )
        self.group_name = response['Group']['GroupName']
        print(f"Created worker group {self.group_name}")

    def create_user_pool_domain(self, DomainName):
        self.cognito.create_user_pool_domain(
            Domain=DomainName,
            UserPoolId=self.user_pool_id
        )

    def get_client_secret(self):
        response = self.cognito.describe_user_pool_client(
            UserPoolId=self.user_pool_id,
            ClientId=self.user_pool_client_id
        )
        self.client_secret = response['UserPoolClient']['ClientSecret']

    def update_client(self, labeling_domain):
        self.cognito.update_user_pool_client(
            UserPoolId=self.user_pool_id,
            ClientId=self.user_pool_client_id,
            CallbackURLs=['https://{labeling_domain}/oauth2/idpresponse'],
            LogoutURLs=['https://{labeling_domain}/logout'],
            AllowedOAuthFlows=['code','implicit'],
            AllowedOAuthScopes=['email','profile','openid']
        )

    def add_user(self, UserEmail, Password):
        self.get_client_secret()
        dig = hmac.new(bytearray(self.client_secret, encoding='utf-8'), 
            msg=f"{UserEmail}{self.user_pool_client_id}".encode('utf-8'), 
            digestmod=hashlib.sha256).digest()
        secret_hash = base64.b64encode(dig).decode() 
        self.cognito.sign_up(
            ClientId=self.user_pool_client_id,
            Username=UserEmail,
            Password=Password,
            SecretHash=secret_hash,
            UserAttributes=[
                {
                    'Name': 'email',
                    'Value': UserEmail
                },
                {
                    'Name': 'phone_number',
                    'Value': '+12485551212'
                }
            ]
        )
        self.cognito.admin_confirm_sign_up(
            UserPoolId=self.user_pool_id,
            Username=UserEmail
        )
        self.cognito.admin_add_user_to_group(
            UserPoolId=self.user_pool_id,
            Username=UserEmail,
            GroupName=self.group_name
        )

if __name__ == "__main__":
    logging.warn("No main function defined")
    sys.exit(0)