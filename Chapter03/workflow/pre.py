import json

"""
The input event looks like this:

        {
           "version":"2018-10-16",
           "labelingJobArn":"<your labeling job ARN>",
           "dataObject":{
              "source":"metric type,metric value,metric unit,lat,lon"
           }
        }

The output should look like this:

        {
           "taskInput":{
              "metric": "PM2.5 = 30.0",
              "lat": 0.0,
              "lon": 0.0
           },
           "isHumanAnnotationRequired":"true"
        }
"""
def handler(event, context):
    sourceText = event['dataObject']['source']
    parts = sourceText.split(',')
    
    output = {
        "taskInput": {
            "metric": f"{parts[0]}={parts[1]} {parts[2]}",
            "lat": parts[3],
            "lon": parts[4] 
        },
        "isHumanAnnotationRequired": "true"
    }
    
    return output