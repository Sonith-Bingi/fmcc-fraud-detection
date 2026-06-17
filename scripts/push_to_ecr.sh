#!/bin/bash
# Phase 2: Build Docker image and push to AWS ECR.
# Usage: AWS_ACCOUNT_ID=123456789012 bash scripts/push_to_ecr.sh
set -e

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT="${AWS_ACCOUNT_ID:?Set AWS_ACCOUNT_ID}"
REPO="fmcc-prod-api"
ECR_URI="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO"
TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD)}"

echo "=== Logging into ECR ==="
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$ECR_URI"

echo "=== Building image ==="
docker build -t "$ECR_URI:$TAG" -t "$ECR_URI:latest" .

echo "=== Pushing to ECR ==="
docker push "$ECR_URI:$TAG"
docker push "$ECR_URI:latest"

echo ""
echo "Image pushed: $ECR_URI:$TAG"
echo "Set this in GitHub Secrets: ECR_REPOSITORY=$ECR_URI"
