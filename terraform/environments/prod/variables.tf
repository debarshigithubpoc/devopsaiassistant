# Basic Configuration
variable "cluster_name" {
  description = "The name of the EKS cluster"
  type        = string
}

variable "kubernetes_version" {
  description = "Kubernetes version for the EKS cluster"
  type        = string
  default     = "1.28"
}

# VPC Configuration
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

# Cluster Configuration
variable "endpoint_private_access" {
  description = "Enable private API server endpoint"
  type        = bool
  default     = true
}

variable "endpoint_public_access" {
  description = "Enable public API server endpoint"
  type        = bool
  default     = true
}

variable "public_access_cidrs" {
  description = "List of CIDR blocks that can access the Amazon EKS public API server endpoint"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

# Node Groups Configuration
variable "node_groups" {
  description = "Map of EKS managed node group definitions to create"
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
}

# Add-ons Configuration
variable "cluster_addons" {
  description = "Map of cluster addon configurations"
  type = map(object({
    version                  = string
    resolve_conflicts        = string
    service_account_role_arn = string
  }))
  default = {}
}

# Logging Configuration
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

# Common Tags
variable "tags" {
  description = "A map of tags to add to all resources"
  type        = map(string)
  default     = {}
}