# Basic Configuration
cluster_name       = "my-eks-prod-cluster"
kubernetes_version = "1.28"

# VPC Configuration - Provide existing VPC and subnet IDs
vpc_id             = "vpc-xxxxxxxxx"  # Replace with your VPC ID
subnet_ids         = ["subnet-xxxxxxxxx", "subnet-yyyyyyyyy"]  # Replace with your subnet IDs
private_subnet_ids = ["subnet-aaaaaaa", "subnet-bbbbbbb"]      # Replace with your private subnet IDs

# Cluster Configuration
endpoint_private_access = true
endpoint_public_access  = false # Private cluster for production
public_access_cidrs     = [] # No public access

# Node Groups - Production configuration with larger, more resilient instances
node_groups = {
  system = {
    capacity_type               = "ON_DEMAND"
    instance_types              = ["m5.large"]
    ami_type                    = "AL2_x86_64"
    disk_size                   = 50
    desired_size                = 3
    max_size                    = 6
    min_size                    = 3
    max_unavailable_percentage  = 25
    labels = {
      Environment = "production"
      NodeGroup   = "system"
      Purpose     = "system-workloads"
    }
    taints = [
      {
        key    = "CriticalAddonsOnly"
        value  = "true"
        effect = "NO_SCHEDULE"
      }
    ]
  }
  
  general = {
    capacity_type               = "ON_DEMAND"
    instance_types              = ["m5.xlarge"]
    ami_type                    = "AL2_x86_64"
    disk_size                   = 100
    desired_size                = 6
    max_size                    = 20
    min_size                    = 3
    max_unavailable_percentage  = 25
    labels = {
      Environment = "production"
      NodeGroup   = "general"
    }
    taints = []
  }
  
  compute = {
    capacity_type               = "ON_DEMAND"
    instance_types              = ["c5.2xlarge"]
    ami_type                    = "AL2_x86_64"
    disk_size                   = 100
    desired_size                = 2
    max_size                    = 10
    min_size                    = 0
    max_unavailable_percentage  = 25
    labels = {
      Environment = "production"
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
  
  memory = {
    capacity_type               = "ON_DEMAND"
    instance_types              = ["r5.2xlarge"]
    ami_type                    = "AL2_x86_64"
    disk_size                   = 100
    desired_size                = 1
    max_size                    = 5
    min_size                    = 0
    max_unavailable_percentage  = 25
    labels = {
      Environment = "production"
      NodeGroup   = "memory"
      WorkloadType = "memory-intensive"
    }
    taints = [
      {
        key    = "memory"
        value  = "true"
        effect = "NO_SCHEDULE"
      }
    ]
  }
}

# Add-ons - Full production suite
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

# Logging - Full logging for production
cluster_log_types          = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
cluster_log_retention_days = 30

# Common tags
tags = {
  Environment = "production"
  Project     = "my-project"
  Team        = "devops"
  ManagedBy   = "terraform"
  CostCenter  = "engineering"
  Backup      = "required"
  Monitoring  = "required"
  Compliance  = "required"
  Critical    = "true"
}