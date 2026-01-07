# Basic Configuration
cluster_name       = "my-eks-dev-cluster"
kubernetes_version = "1.28"

# VPC Configuration - Provide existing VPC and subnet IDs
vpc_id             = "vpc-xxxxxxxxx"  # Replace with your VPC ID
subnet_ids         = ["subnet-xxxxxxxxx", "subnet-yyyyyyyyy"]  # Replace with your subnet IDs
private_subnet_ids = ["subnet-aaaaaaa", "subnet-bbbbbbb"]      # Replace with your private subnet IDs

# Cluster Configuration
endpoint_private_access = true
endpoint_public_access  = true
public_access_cidrs     = ["0.0.0.0/0"]

# Node Groups - Development configuration with smaller instances
node_groups = {
  general = {
    capacity_type               = "ON_DEMAND"
    instance_types              = ["t3.small"]
    ami_type                    = "AL2_x86_64"
    disk_size                   = 20
    desired_size                = 2
    max_size                    = 4
    min_size                    = 1
    max_unavailable_percentage  = 25
    labels = {
      Environment = "development"
      NodeGroup   = "general"
    }
    taints = []
  }
}

# Add-ons
cluster_addons = {
  coredns = {
    version                  = "v1.10.1-eksbuild.5"
    resolve_conflicts        = "OVERWRITE"
    service_account_role_arn = null
  }
  kube-proxy = {
    version                  = "v1.28.2-eksbuild.2"
    resolve_conflicts        = "OVERWRITE"
    service_account_role_arn = null
  }
  vpc-cni = {
    version                  = "v1.15.4-eksbuild.1"
    resolve_conflicts        = "OVERWRITE"
    service_account_role_arn = null
  }
}

# Logging - Minimal for development
cluster_log_types          = ["api", "audit"]
cluster_log_retention_days = 7

# Common tags
tags = {
  Environment = "development"
  Project     = "my-project"
  Team        = "devops"
  ManagedBy   = "terraform"
  CostCenter  = "engineering"
}