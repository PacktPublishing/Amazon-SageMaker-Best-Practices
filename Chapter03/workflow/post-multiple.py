import json
import boto3

"""
Input:

{
    "version": "2018-10-16",
    "labelingJobArn": <labelingJobArn>,
    "labelCategories": [<string>],
    "labelAttributeName": <string>,
    "roleArn" : "string",
    "payload": {
        "s3Uri": <string>
    }
 }

Contents of payload:

[
    {
        "datasetObjectId": <string>,
        "dataObject": {
            "s3Uri": <string>,
            "content": <string>
        },
        "annotations": [{
            "workerId": <string>,
            "annotationData": {
                "content": <string>,
                "s3Uri": <string>
            }
       }]
    }
]

Output:

[
   {        
        "datasetObjectId": <string>,
        "consolidatedAnnotation": {
            "content": {
                "<labelattributename>": {
                    # ... label content
                }
            }
        }
    },
   {        
        "datasetObjectId": <string>,
        "consolidatedAnnotation": {
            "content": {
                "<labelattributename>": {
                    # ... label content
                }
            }
        }
    }
]

"""
def handler(event, context):
    input_uri = event["payload"]['s3Uri']
    parts = input_uri.split('/')
    s3 = boto3.client('s3')
    s3.download_file(parts[2], "/".join(parts[3:]), '/tmp/input.json')
    
    with open('/tmp/input.json', 'r') as F:
        input_data = json.load(F)
        
    output_data = []
    for p in range(len(input_data)):
        d_id = input_data[p]['datasetObjectId']
                
        annotations = input_data[p]['annotations']
        annotation = annotations[len(annotations)-1]['annotationData']['content']

        response = {
            "datasetObjectId": d_id,
                "consolidatedAnnotation": {
                    "content": annotation
                }
            }

        output_data.append(response)

    # Perform consolidation
    return output_data