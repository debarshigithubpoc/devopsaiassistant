# Basic Configuration
cluster_name       = "my-eks-staging-cluster"
kubernetes_version = "1.28"

# VPC Configuration - Provide existing VPC and subnet IDs
vpc_id             = "vpc-xxxxxxxxx"  # Replace with your VPC ID
subnet_ids         = ["subnet-xxxxxxxxx", "subnet-yyyyyyyyy"]  # Replace with your subnet IDs
private_subnet_ids = ["subnet-aaaaaaa", "subnet-bbbbbbb"]      # Replace with your private subnet IDs

# Cluster Configuration
endpoint_private_access = true
endpoint_public_access  = true
public_access_cidrs     = ["10.0.0.0/8", "172.16.0.0/12"] # More restricted access

# Node Groups - Staging configuration with medium instances
node_groups = {
  general = {
    capacity_type               = "ON_DEMAND"
    instance_types              = ["t3.medium"]
    ami_type                    = "AL2_x86_64"
    disk_size                   = 30
    desired_size                = 3
    max_size                    = 6
    min_size                    = 2
    max_unavailable_percentage  = 25
    labels = {
      Environment = "staging"
      NodeGroup   = "general"
    }
    taints = []
  }
  
  compute = {
    capacity_type               = "SPOT" # Use spot instances for cost savings
    instance_types              = ["c5.large"]
    ami_type                    = "AL2_x86_64"
    disk_size                   = 50
    desired_size                = 1
    max_size                    = 3
    min_size                    = 0
    max_unavailable_percentage  = 50
    labels = {
      Environment = "staging"
      NodeGroup   = "compute"
      WorkloadType = "compute-intensive"
    }
    taints = [
      {
        key    = "compute"
        value  = "true"
        effect = "NO_SCHEDULE"
      }
    ]
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
  aws-ebs-csi-driver = {
    version                  = "v1.24.1-eksbuild.1"
    resolve_conflicts        = "OVERWRITE"
    service_account_role_arn = null
  }
}

# Logging - More comprehensive for staging
cluster_log_types          = ["api", "audit", "authenticator", "controllerManager"]
cluster_log_retention_days = 14

# Common tags
tags = {
  Environment = "staging"
  Project     = "my-project"
  Team        = "devops"
  ManagedBy   = "terraform"
  CostCenter  = "engineering"
  Backup      = "required"
}