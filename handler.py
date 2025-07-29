import runpod

def handler(job):
    print("Received job input:", job["input"])
    return job["input"]  # Echo the input back

runpod.serverless.start({"handler": handler})
