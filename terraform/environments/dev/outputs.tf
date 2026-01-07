output "cluster_id" {
  description = "The name/id of the EKS cluster"
  value       = module.eks_cluster.cluster_id
}

output "cluster_arn" {
  description = "The Amazon Resource Name (ARN) of the cluster"
  value       = module.eks_cluster.cluster_arn
}

output "cluster_endpoint" {
  description = "Endpoint for your Kubernetes API server"
  value       = module.eks_cluster.cluster_endpoint
  sensitive   = true
}

output "cluster_version" {
  description = "The Kubernetes server version for the EKS cluster"
  value       = module.eks_cluster.cluster_version
}

output "cluster_ca_certificate" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks_cluster.cluster_ca_certificate
  sensitive   = true
}

output "cluster_oidc_issuer_url" {
  description = "The URL on the EKS cluster OIDC Issuer"
  value       = module.eks_cluster.cluster_oidc_issuer_url
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks_cluster.cluster_security_group_id
}

output "node_groups" {
  description = "EKS node group information"
  value       = module.eks_cluster.node_groups
}

output "cluster_iam_role_arn" {
  description = "IAM role ARN of the EKS cluster"
  value       = module.eks_cluster.cluster_iam_role_arn
}