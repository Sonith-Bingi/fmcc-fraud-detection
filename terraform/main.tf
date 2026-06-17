terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  backend "s3" {
    bucket = "fmcc-terraform-state"
    key    = "prod/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region
}

# ── Variables ─────────────────────────────────────────────────────────────────
variable "aws_region"    { default = "us-east-1" }
variable "project"       { default = "fmcc" }
variable "env"           { default = "prod" }
variable "ec2_ami"       { default = "ami-0c02fb55956c7d316" }  # Amazon Linux 2023
variable "ec2_type"      { default = "t3.medium" }
variable "alert_email"   { description = "Email for drift/drift SNS alerts" }
variable "db_password"   { sensitive = true }

locals {
  name_prefix = "${var.project}-${var.env}"
  tags = {
    Project     = var.project
    Environment = var.env
    ManagedBy   = "terraform"
  }
}

# ── S3 Buckets ─────────────────────────────────────────────────────────────────
resource "aws_s3_bucket" "raw_cdrs" {
  bucket = "${local.name_prefix}-raw-cdrs"
  tags   = local.tags
}

resource "aws_s3_bucket" "features" {
  bucket = "${local.name_prefix}-features"
  tags   = local.tags
}

resource "aws_s3_bucket" "artifacts" {
  bucket = "${local.name_prefix}-artifacts"
  tags   = local.tags
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration { status = "Enabled" }
}

# ── ECR Repository ─────────────────────────────────────────────────────────────
resource "aws_ecr_repository" "api" {
  name                 = "${local.name_prefix}-api"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = true }
  tags = local.tags
}

resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 10 images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 10 }
      action       = { type = "expire" }
    }]
  })
}

# ── VPC + Security Groups ──────────────────────────────────────────────────────
resource "aws_security_group" "api" {
  name        = "${local.name_prefix}-api-sg"
  description = "FMCC API server"
  tags        = local.tags

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "FastAPI"
  }
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "SSH"
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ── IAM Role for EC2 ───────────────────────────────────────────────────────────
resource "aws_iam_role" "ec2" {
  name = "${local.name_prefix}-ec2-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{ Effect = "Allow", Principal = { Service = "ec2.amazonaws.com" }, Action = "sts:AssumeRole" }]
  })
  tags = local.tags
}

resource "aws_iam_role_policy" "ec2" {
  name = "${local.name_prefix}-ec2-policy"
  role = aws_iam_role.ec2.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Action = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
        Resource = ["${aws_s3_bucket.artifacts.arn}/*", "${aws_s3_bucket.artifacts.arn}",
                    "${aws_s3_bucket.features.arn}/*"] },
      { Effect = "Allow", Action = ["ecr:GetAuthorizationToken", "ecr:BatchGetImage",
                                     "ecr:GetDownloadUrlForLayer"],
        Resource = "*" },
      { Effect = "Allow", Action = ["cloudwatch:PutMetricData", "logs:*"], Resource = "*" },
      { Effect = "Allow", Action = ["sns:Publish"], Resource = aws_sns_topic.alerts.arn },
    ]
  })
}

resource "aws_iam_instance_profile" "ec2" {
  name = "${local.name_prefix}-ec2-profile"
  role = aws_iam_role.ec2.name
}

# ── EC2 Instance ───────────────────────────────────────────────────────────────
resource "aws_instance" "api" {
  ami                    = var.ec2_ami
  instance_type          = var.ec2_type
  iam_instance_profile   = aws_iam_instance_profile.ec2.name
  vpc_security_group_ids = [aws_security_group.api.id]
  tags                   = merge(local.tags, { Name = "${local.name_prefix}-api" })

  user_data = base64encode(templatefile("${path.module}/userdata.sh.tpl", {
    ecr_repo   = aws_ecr_repository.api.repository_url
    aws_region = var.aws_region
    project    = var.project
  }))

  root_block_device {
    volume_size = 30
    volume_type = "gp3"
  }
}

resource "aws_eip" "api" {
  instance = aws_instance.api.id
  domain   = "vpc"
  tags     = local.tags
}

# ── API Gateway (HTTP API → EC2) ───────────────────────────────────────────────
resource "aws_apigatewayv2_api" "fmcc" {
  name          = "${local.name_prefix}-api-gw"
  protocol_type = "HTTP"
  tags          = local.tags
}

resource "aws_apigatewayv2_integration" "ec2" {
  api_id             = aws_apigatewayv2_api.fmcc.id
  integration_type   = "HTTP_PROXY"
  integration_uri    = "http://${aws_eip.api.public_ip}:8000/{proxy}"
  integration_method = "ANY"
}

resource "aws_apigatewayv2_route" "proxy" {
  api_id    = aws_apigatewayv2_api.fmcc.id
  route_key = "ANY /{proxy+}"
  target    = "integrations/${aws_apigatewayv2_integration.ec2.id}"
}

resource "aws_apigatewayv2_stage" "prod" {
  api_id      = aws_apigatewayv2_api.fmcc.id
  name        = "prod"
  auto_deploy = true
  tags        = local.tags
}

# ── RDS PostgreSQL (prediction log + MLflow backend) ──────────────────────────
resource "aws_db_instance" "postgres" {
  identifier        = "${local.name_prefix}-db"
  engine            = "postgres"
  engine_version    = "16"
  instance_class    = "db.t3.micro"
  allocated_storage = 20
  db_name           = "fmcc"
  username          = "fmcc"
  password          = var.db_password
  skip_final_snapshot     = false
  final_snapshot_identifier = "${local.name_prefix}-final"
  vpc_security_group_ids = [aws_security_group.api.id]
  tags = local.tags
}

# ── SNS Alerts ─────────────────────────────────────────────────────────────────
resource "aws_sns_topic" "alerts" {
  name = "${local.name_prefix}-alerts"
  tags = local.tags
}

resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# ── CloudWatch Alarm — drift detected ─────────────────────────────────────────
resource "aws_cloudwatch_metric_alarm" "drift" {
  alarm_name          = "${local.name_prefix}-drift-detected"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 1
  metric_name         = "DriftDetected"
  namespace           = "FMCC/ModelMonitoring"
  period              = 86400
  statistic           = "Sum"
  threshold           = 1
  alarm_description   = "Feature drift detected in FMCC model inputs"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  treat_missing_data  = "notBreaching"
  tags                = local.tags
}

# ── CloudWatch Alarm — high fraud rate ────────────────────────────────────────
resource "aws_cloudwatch_metric_alarm" "high_fraud" {
  alarm_name          = "${local.name_prefix}-high-fraud-rate"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "FraudRate"
  namespace           = "FMCC/Predictions"
  period              = 86400
  statistic           = "Average"
  threshold           = 0.15
  alarm_description   = "Daily fraud rate exceeds 15%"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  treat_missing_data  = "notBreaching"
  tags                = local.tags
}

# ── Outputs ────────────────────────────────────────────────────────────────────
output "api_gateway_url" {
  value       = "${aws_apigatewayv2_stage.prod.invoke_url}"
  description = "Public HTTPS endpoint for fraud detection API"
}

output "ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "ec2_public_ip" {
  value = aws_eip.api.public_ip
}

output "rds_endpoint" {
  value     = aws_db_instance.postgres.endpoint
  sensitive = true
}

output "sns_topic_arn" {
  value = aws_sns_topic.alerts.arn
}
