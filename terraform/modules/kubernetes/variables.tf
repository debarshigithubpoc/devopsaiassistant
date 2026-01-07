variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
}

variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

variable "vpc_id" {
  description = "VPC ID where the cluster will be created"
  type        = string
}

variable "subnet_ids" {
  description = "List of subnet IDs for the EKS cluster"
  type        = list(string)
}

variable "private_subnet_ids" {
  description = "List of private subnet IDs for node groups"
  type        = list(string)
}

variable "endpoint_private_access" {
  description = "Whether the Amazon EKS private API server endpoint is enabled"
  type        = bool
  default     = true
}

variable "endpoint_public_access" {
  description = "Whether the Amazon EKS public API server endpoint is enabled"
  type        = bool
  default     = true
}

variable "public_access_cidrs" {
  description = "List of CIDR blocks that can access the Amazon EKS public API server endpoint"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "cluster_log_types" {
  description = "List of control plane logging to enable"
  type        = list(string)
  default     = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
}

variable "cluster_log_retention_days" {
  description = "Retention period for cluster logs"
  type        = number
  default     = 7
}

variable "node_groups" {
  description = "Map of EKS node group definitions"
  type = map(object({
    capacity_type               = string
    instance_types              = list(string)
    ami_type                    = string
    disk_size                   = number
    desired_size                = number
    max_size                    = number
    min_size                    = number
    max_unavailable_percentage  = number
    labels                      = map(string)
    taints = list(object({
      key    = string
      value  = string
      effect = string
    }))
  }))
  default = {
    general = {
      capacity_type              = "ON_DEMAND"
      instance_types             = ["t3.medium"]
      ami_type                   = "AL2_x86_64"
      disk_size                  = 50
      desired_size               = 2
      max_size                   = 4
      min_size                   = 1
      max_unavailable_percentage = 25
      labels = {
        Environment = "development"
        NodeGroup   = "general"
      }
      taints = []
    }
  }
}

variable "cluster_addons" {
  description = "Map of cluster addon configurations"
  type = map(object({
    version                  = string
    resolve_conflicts        = string
    service_account_role_arn = string
  }))
  default = {
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
}

variable "tags" {
  description = "A map of tags to add to all resources"
  type        = map(string)
  default     = {}
}