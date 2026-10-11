# HeatGuard AI on AWS - Deployment Guide

Target architecture:

```
User -> CloudFront (static) ----> S3 (static, optional)
     -> Application Load Balancer (HTTPS)
            -> ECS Fargate tasks (Docker: Flask + gunicorn)   [auto scaling 1-4 tasks]
                  |-- S3: models (.pkl) + dataset (CSV)
                  |-- Secrets Manager: GEMINI_API_KEY
                  |-- DynamoDB: prediction/advisory logs
                  |-- CloudWatch Logs + Alarms
EventBridge (daily) -> Lambda -> calls /api/predict-all -> SNS (email/SMS alerts)
GitHub Actions -> ECR -> ECS (CI/CD)
```

Use ONE region for everything (e.g. ap-south-1, Mumbai).

---
## Step 0 - Prepare the project (local)
1. **Revoke your current Gemini key** (it was inside `.env` in your zip) and create a new one.
2. Copy `Dockerfile` and `.dockerignore` from this folder into the project root.
3. In your venv run `pip freeze | findstr -i "scikit numpy pandas joblib"` and pin those exact
   versions in `requirements.txt` (so the .pkl models unpickle correctly).
4. Add a lightweight health endpoint to `app.py` (the existing `/api/health` loads the model,
   which is too heavy for load-balancer checks):
   ```python
   @app.route("/healthz")
   def healthz():
       return {"status": "ok"}, 200
   ```
5. (Security) In `app.py`, the 500 handler returns the full traceback to the client. For the cloud
   version, return only a generic message and keep the traceback in CloudWatch logs.
6. Test locally:
   ```
   docker build -t heatguard-ai .
   docker run -p 8080:8080 -e GEMINI_API_KEY=your_new_key heatguard-ai
   ```
   Open http://localhost:8080 and http://localhost:8080/healthz

## Step 1 - Push the image to Amazon ECR
```
aws ecr create-repository --repository-name heatguard-ai --region ap-south-1
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com
docker tag heatguard-ai:latest <ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com/heatguard-ai:latest
docker push <ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com/heatguard-ai:latest
```

## Step 2 - Store the API key in Secrets Manager
```
aws secretsmanager create-secret --name heatguard/gemini-api-key --secret-string "YOUR_NEW_KEY" --region ap-south-1
```
Copy the secret ARN. You will reference it in the ECS task definition (never put it in the image).

## Step 3 - Run on ECS Fargate behind a load balancer (Console)
1. **ECS -> Clusters -> Create cluster** (`heatguard-cluster`, Fargate).
2. **Task definitions -> Create**: Fargate, 1 vCPU, 2 GB memory, container port 8080,
   image = your ECR URI. Under *Environment -> ValueFrom (Secrets)* add
   `GEMINI_API_KEY` -> the secret ARN. Enable the awslogs log driver (CloudWatch).
   - Task *execution* role: attach `AmazonECSTaskExecutionRolePolicy` plus permission
     `secretsmanager:GetSecretValue` on that secret.
3. **Cluster -> Create service**: launch type Fargate, desired tasks = 2, new Application Load
   Balancer, listener 80 (add 443 with an ACM certificate if you have a domain),
   target group health check path = `/healthz`.
4. Security groups: ALB allows 80/443 from anywhere; the ECS tasks allow 8080 **only from the ALB's
   security group**.
5. Open the ALB DNS name in a browser. That's your public URL.

## Step 4 - Auto scaling (Console or CLI)
Service -> *Service auto scaling*: min 1, max 4, policy *target tracking* on ECS
Average CPU = 50%. Or via CLI:
```
aws application-autoscaling register-scalable-target --service-namespace ecs \
  --resource-id service/heatguard-cluster/heatguard-service \
  --scalable-dimension ecs:service:DesiredCount --min-capacity 1 --max-capacity 4
aws application-autoscaling put-scaling-policy --service-namespace ecs \
  --resource-id service/heatguard-cluster/heatguard-service \
  --scalable-dimension ecs:service:DesiredCount --policy-name cpu50 \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration '{"TargetValue":50.0,"PredefinedMetricSpecification":{"PredefinedMetricType":"ECSServiceAverageCPUUtilization"}}'
```

## Step 5 - Load test (your best demo)
Install Locust (`pip install locust`) and save as `locustfile.py`:
```python
from locust import HttpUser, task, between
class Visitor(HttpUser):
    wait_time = between(0.5, 2)
    @task(3)
    def predict(self):
        self.client.post("/api/predict", json={"city": "Delhi", "date": "2024-05-20"})
    @task(1)
    def analytics(self):
        self.client.get("/api/analytics/summary")
```
Run `locust -f locustfile.py --host http://<ALB-DNS>` and ramp to 200+ users. Screenshot the
CloudWatch ECS CPU graph and the running-task count going 1 -> 4. (Check the `/api/predict`
request body against `app.py`; adjust field names if needed.)

## Step 6 - Move models and data to S3
```
aws s3 mb s3://heatguard-<yourname>-assets
aws s3 cp models/saved/hybrid_predictor.pkl s3://heatguard-<yourname>-assets/models/
aws s3 cp data/heatguard_clean.csv s3://heatguard-<yourname>-assets/data/
```
Give the ECS **task role** (not the execution role) `s3:GetObject` on that bucket, then download at
container start. Add `cloud_assets.py` and call it at the top of `app.py`:
```python
import os, boto3
def fetch_assets():
    bucket = os.getenv("ASSET_BUCKET")
    if not bucket:
        return  # local dev: use files already on disk
    s3 = boto3.client("s3")
    for key, dest in [("models/hybrid_predictor.pkl", "models/saved/hybrid_predictor.pkl"),
                      ("data/heatguard_clean.csv", "data/heatguard_clean.csv")]:
        if not os.path.exists(dest):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            s3.download_file(bucket, key, dest)
```
Set `ASSET_BUCKET` as an environment variable in the task definition. Enable S3 versioning so
model versions are tracked. You can then remove the CSV and .pkl from the Docker image.

## Step 7 - Logs in DynamoDB + real alerts with SNS/Lambda
- **DynamoDB** table `heatguard_predictions` (partition key `city`, sort key `timestamp`).
  After each `/api/predict`, write the result with `boto3.resource("dynamodb")`
  (task role needs `dynamodb:PutItem`).
- **SNS** topic `heatguard-alerts`; subscribe your email. Replace your simulated alert-dispatch
  endpoint with `sns.publish(...)`.
- **Lambda + EventBridge**: a daily schedule triggers a Lambda that calls
  `POST /api/predict-all`, and publishes to SNS any city whose risk is High/Severe.

## Step 8 - CI/CD with GitHub Actions
`.github/workflows/deploy.yml`:
```yaml
name: deploy
on: { push: { branches: [main] } }
jobs:
  deploy:
    runs-on: ubuntu-latest
    permissions: { id-token: write, contents: read }
    steps:
      - uses: actions/checkout@v4
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::<ACCOUNT_ID>:role/github-deploy-role
          aws-region: ap-south-1
      - id: ecr
        uses: aws-actions/amazon-ecr-login@v2
      - run: |
          docker build -t ${{ steps.ecr.outputs.registry }}/heatguard-ai:latest .
          docker push ${{ steps.ecr.outputs.registry }}/heatguard-ai:latest
          aws ecs update-service --cluster heatguard-cluster --service heatguard-service --force-new-deployment
```

## Step 9 - Monitoring
CloudWatch dashboard with: ALB request count, target response time, 5xx count, ECS CPU/memory,
running task count. Alarms: 5xx > 5 per 5 min and CPU > 80%, both notifying the SNS topic.

## Step 10 - Cost estimate (put this in your report)
Use the AWS Pricing Calculator for: Fargate (1 vCPU / 2 GB x 2 tasks), ALB, S3, DynamoDB
(on-demand), Secrets Manager, CloudWatch. Fargate + ALB cost roughly $50-70/month if left running
continuously, so **delete the stack after your demo/presentation** (set service desired count to 0,
delete the ALB, NAT gateway if any, and the ECR images). Set an AWS Budget alert at $10.

## Cleanup checklist
ECS service -> 0 tasks, delete service and cluster, delete ALB + target group, delete ECR images,
delete S3 bucket contents, delete DynamoDB table, delete secret.

## Report checklist
Architecture diagram - justification for each service - screenshots (ALB URL, ECS tasks,
CloudWatch, autoscaling during load test) - security (IAM roles, Secrets Manager, security groups,
HTTPS) - cost analysis - challenges / limitations - future work.
