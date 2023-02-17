####### FUNCTION 1 ########
import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3

    Args:
        event (json object): event data
        context (_type_): _description_
    """
    
    key = event["s3_key"]
    bucket = event["s3_bucket"]
    
    # Download the data from S3 to /tmp/image.png
    s3 = boto3.client('s3')
    s3.download_file(bucket, key, '/tmp/image.png')
    
    # Read the image data from /tmp/image.png
    with open("/tmp/image.png", "rb") as image_file:
        image_data = base64.b64encode(image_file.read())
    
    # Pass the data back to the Step Function
    print("Event: ", event.keys())
    
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }
    

####### FUNCTION 2 ########
import json
import base64
import boto3

ENDPOINT = "image-classification-2023-02-17-11-00-27-570"

def lambda_handler(event, context):
    
    # Get the S3 bucket and object key from the event
    s3_bucket = event["s3_bucket"]
    s3_key = event["s3_key"]
    
    # Download the image file from S3 to a local file in /tmp/
    s3 = boto3.client("s3")
    local_file_path = "/tmp/image.png"
    s3.download_file(s3_bucket, s3_key, local_file_path)
    
    # Load the image into memory
    image = base64.b64decode(event['image_data'])
    
    # Instantiate a Lambda client
    sagemaker_runtime = boto3.client("sagemaker-runtime")
    
    # Make a prediction request to the deployed model endpoint
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=ENDPOINT,
        Body=image,
        ContentType="image/png"
    )
    
    # Get the response body
    inferences = response["Body"].read().decode("utf-8")
    
    # We return the data back to the Step Function
    event["inferences"] = inferences
    
    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }


###### FUNCTION 3 #######
import json

THRESHOLD = 0.93

def lambda_handler(event, context):
    
    body = json.loads(event['body'])
    
    # Grab the inference from the event
    inferences = body['inferences']
    
    # Check if any values in our inferences are above THRESHOLD
    inferences = inferences.replace("[", "").replace("]", "").split(", ")
    inferences = [float(x) for x in inferences]
    
    meets_threshold = any([x > THRESHOLD for x in inferences])
    
    # If our threshold is met, pass our data back out of the Step Function
    # else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise("THRESHOLD_CONFIDENCE_NOT_MET")
    
    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }
