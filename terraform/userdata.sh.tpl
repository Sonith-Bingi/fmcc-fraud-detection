#!/bin/bash
set -e

# Install Docker
yum update -y
yum install -y docker
systemctl enable docker
systemctl start docker
usermod -aG docker ec2-user

# Install AWS CLI v2
curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp
/tmp/aws/install

# Log into ECR and pull the latest image
aws ecr get-login-password --region ${aws_region} \
  | docker login --username AWS --password-stdin ${ecr_repo}

docker pull ${ecr_repo}:latest

# Write systemd service
cat > /etc/systemd/system/fmcc-api.service << 'EOF'
[Unit]
Description=FMCC Fraud Detection API
After=docker.service
Requires=docker.service

[Service]
Restart=always
ExecStartPre=-/usr/bin/docker stop fmcc-api
ExecStartPre=-/usr/bin/docker rm fmcc-api
ExecStart=/usr/bin/docker run --rm --name fmcc-api \
  -p 8000:8000 \
  -e FRAUD_THRESHOLD=0.5 \
  -e POSTGRES_URL=${postgres_url} \
  ${ecr_repo}:latest
ExecStop=/usr/bin/docker stop fmcc-api

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable fmcc-api
systemctl start fmcc-api
