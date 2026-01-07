terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
  
  backend "s3" {
    # Backend configuration will be provided by CI/CD pipeline
    # Example configuration:
    # bucket         = "my-terraform-state-bucket"
    # key            = "aws/prod/terraform.tfstate"
    # region         = "us-west-2"
    # dynamodb_table = "terraform-state-locks"
    # encrypt        = true
  }
}

provider "aws" {
  region = "us-west-2"  # Change this to your preferred region
}

module "eks_cluster" {
  source = "../../modules/kubernetes"

  # Basic Configuration
  cluster_name       = var.cluster_name
  kubernetes_version = var.kubernetes_version

  # VPC Configuration
  vpc_id             = var.vpc_id
  subnet_ids         = var.subnet_ids
  private_subnet_ids = var.private_subnet_ids

  # Node Groups
  node_groups = var.node_groups

  # Cluster Configuration
  endpoint_private_access = var.endpoint_private_access
  endpoint_public_access  = var.endpoint_public_access
  public_access_cidrs     = var.public_access_cidrs
  
  # Add-ons
  cluster_addons = var.cluster_addons

  # Logging
  cluster_log_types           = var.cluster_log_types
  cluster_log_retention_days  = var.cluster_log_retention_days

  # Common tags
  tags = var.tags
}